"""Optional LLM coach: rewrite explanations and draft a weekly plan from a verified facts packet (C4). Owner: llm.

The Claude API gets the coaching packet (``packet.build_packet``: the report's claims, study plan and up to 20
explained positions) and returns structured JSON: ``explanations`` [{epd, text, claim_ids}] and ``weekly_plan``
[{item, minutes_per_week, recheck_date}]. Every text goes through ``verify`` before it replaces a template, so the
LLM can reword the report's facts but cannot add to them:

* an accepted text replaces ``Explanation.text`` (``text_source = "llm"``); a rejected one keeps its template;
* weekly-plan entries that name no study-plan item are dropped, and the plan is trimmed to
  ``cfg.practice_minutes`` a day;
* ``coaching.llm`` records the model and the counts, and the notes say how many texts failed the check.

The Anthropic SDK is an optional extra (``pip install "chess-insights[llm]"``), imported only when it is needed.
With the LLM off, or no API key, nothing in the report changes; an API error leaves the templates and adds a note.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Optional

from ..models import Report
from .config import CoachConfig
from .packet import build_packet, position_index
from .verify import verify_explanation, verify_plan

log = logging.getLogger(__name__)

MAX_TOKENS = 16000  # thinking + the JSON reply; a non-streaming request of this size stays within the SDK's limits
EFFORT = "high"  # claude-opus-5-5 defaults to "medium"; a text the verifier rejects wastes the call
TIMEOUT_S = 300.0  # per attempt
MAX_RETRIES = 2  # the SDK retries connection errors, 408, 409, 429 and 5xx with backoff
MAX_REJECTIONS_KEPT = 20  # rejected texts and their problems kept in coaching.llm (to check the verifier's work)
INSTALL_HINT = 'pip install "chess-insights[llm]"'

SYSTEM_PROMPT = """\
You write the coaching notes in a chess report for one club player, from a JSON packet of facts the report has \
already established. The player reads the report on a phone. A program checks every note against the packet \
before it is shown and throws away any note that breaks one of these rules, so follow them exactly:

1. Use only facts in the packet. Every number you write must appear in the packet; you may round it. Give \
evaluations in pawns ("1.5 pawns"), never in centipawns.
2. Name only moves that appear in the position's lines: best_line, refutation, opening.cloud_lines, \
opening.masters, opening.peers and tablebase.best. Write each move in standard algebraic notation with its move \
number, exactly as the packet numbers it ("5...e5", "6.Ndb5", "7.Nd6+"). When you mean a square rather than a \
move, write it on its own ("the d6 square").
3. Call something a strength or a weakness of the player only when it is one of the packet's claims (claims or \
format_claims), and put that claim's id in claim_ids. Never write an id in the text. A position whose is_claim is \
false is not a claim: say what happened in it without calling it a habit or a weakness.
4. At most three sentences per position, in plain text without markdown, speaking to the player as "you": what \
the refutation does (its motif), what it wins (material, or the concept differences, which are in pawns from the \
player's side), and what to check next time. Mention the format of the game (time_class: bullet, blitz or rapid) \
when it helps. maia, when present, gives how often players at the player's rating find the best move (p_best) \
and play the player's move (p_played).

The weekly plan takes its items from study_plan only, each named by its exact title. Share the player's practice \
time among them; the minutes of all entries together must fit the budget given with the packet. recheck_date \
(YYYY-MM-DD) is the day on which the player should look at the item's target again."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "epd": {"type": "string"},
                    "text": {"type": "string"},
                    "claim_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["epd", "text", "claim_ids"],
                "additionalProperties": False,
            },
        },
        "weekly_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "minutes_per_week": {"type": "integer"},
                    "recheck_date": {"type": "string"},
                },
                "required": ["item", "minutes_per_week", "recheck_date"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["explanations", "weekly_plan"],
    "additionalProperties": False,
}


class LLMError(Exception):
    """The request failed or the reply was unusable; the message says why in plain words."""


# --------------------------------------------------------------------------- the API client
def api_key(cfg: CoachConfig) -> Optional[str]:
    return cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or None


def has_credentials(cfg: CoachConfig) -> bool:
    """An API key (setting or ANTHROPIC_API_KEY), or another credential the SDK resolves by itself."""
    return bool(api_key(cfg) or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"))


def load_sdk() -> Any:
    """The ``anthropic`` package, or None when the optional extra is not installed."""
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic


def make_client(cfg: CoachConfig, sdk: Any) -> Any:
    kwargs: dict[str, Any] = {"timeout": TIMEOUT_S, "max_retries": MAX_RETRIES}
    key = api_key(cfg)
    if key:
        kwargs["api_key"] = key
    return sdk.Anthropic(**kwargs)


def describe_error(exc: BaseException, sdk: Any, model: str) -> str:
    """A plain-language reason for a failed request, the most specific SDK error class first."""
    checks = (
        ("AuthenticationError", "the API key was rejected"),
        ("PermissionDeniedError", f"this API key may not use {model}"),
        ("NotFoundError", f"the model {model} is not available to this API key"),
        ("RateLimitError", "the API is still rate-limiting this key after retrying"),
        ("BadRequestError", "the API refused the request"),
        ("APIStatusError", "the API returned an error"),
        ("APITimeoutError", f"the request timed out after {TIMEOUT_S:.0f} s"),
        ("APIConnectionError", "the API could not be reached"),
    )
    status = getattr(exc, "status_code", None)
    for name, text in checks:
        cls = getattr(sdk, name, None) if sdk is not None else None
        if isinstance(cls, type) and isinstance(exc, cls):
            return f"{text} ({status})" if status else text
    return f"{type(exc).__name__}{f' {status}' if status else ''}: {exc}"


def call_json(
    client: Any, cfg: CoachConfig, system: str, user: str, schema: dict[str, Any], sdk: Any = None
) -> dict[str, Any]:
    """One Messages API request whose reply is JSON matching ``schema`` (structured output). Raises LLMError."""
    request: dict[str, Any] = {
        "model": cfg.llm_model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
    }
    try:
        try:
            response = client.messages.create(**request)
        except TypeError as exc:  # an SDK release that predates output_config: send it as a raw body field
            if "output_config" not in str(exc):
                raise
            request["extra_body"] = {"output_config": request.pop("output_config")}
            response = client.messages.create(**request)
    except Exception as exc:  # noqa: BLE001 — every API failure leaves the templates in place
        raise LLMError(describe_error(exc, sdk, cfg.llm_model)) from exc
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        raise LLMError("the model declined the request")
    if stop == "max_tokens":
        raise LLMError(f"the reply was cut off at {MAX_TOKENS} tokens")
    # read blocks by type: thinking blocks come first
    text = next((b.text for b in getattr(response, "content", None) or [] if getattr(b, "type", "") == "text"), "")
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise LLMError("the reply was not valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("the reply was not a JSON object")
    return data


# --------------------------------------------------------------------------- the coach
def user_message(packet: dict[str, Any], cfg: CoachConfig, today: date) -> str:
    """The packet first, then the task (long material before the request)."""
    minutes = max(0, int(cfg.practice_minutes))
    facts = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    return (
        f"<packet>\n{facts}\n</packet>\n\n"
        f"Today is {today.isoformat()}. The player has {minutes} minutes a day for practice ({minutes * 7} minutes a "
        f"week for the whole plan). Recheck dates fall between {(today + timedelta(days=7)).isoformat()} and "
        f"{(today + timedelta(days=42)).isoformat()}.\n\n"
        "Write one explanation for each position in the packet (identify it by its epd) and a weekly plan."
    )


def report_day(report: Any, cfg: CoachConfig) -> date:
    """``cfg.today``, else the report's date."""
    if cfg.today:
        return cfg.today
    generated = getattr(report, "generated_at", None)
    return generated.date() if hasattr(generated, "date") else date.today()


def annotate(report: Report, cfg: CoachConfig, client: Any = None) -> None:
    """Replace template texts that pass the verifier; fill ``coaching.weekly_plan`` and ``coaching.llm``.

    ``client`` is for tests (anything with ``messages.create``); normally the SDK client is built from
    ``cfg.anthropic_api_key`` or ANTHROPIC_API_KEY. Without credentials this changes nothing at all.
    """
    coaching = report.coaching
    if coaching is None:
        return
    sdk = None
    if client is None:
        if not has_credentials(cfg):
            log.warning("LLM coach skipped: no ANTHROPIC_API_KEY. The explanations keep their template text.")
            return
        sdk = load_sdk()
        if sdk is None:
            coaching.notes.append(f"LLM coach skipped: the Anthropic SDK is not installed ({INSTALL_HINT}).")
            return
        client = make_client(cfg, sdk)
    packet = build_packet(report)
    if not packet["positions"] and not packet["study_plan"]:
        return
    today = report_day(report, cfg)
    try:
        data = call_json(client, cfg, SYSTEM_PROMPT, user_message(packet, cfg, today), OUTPUT_SCHEMA, sdk)
    except LLMError as exc:
        log.warning("LLM coach failed: %s", exc)
        coaching.notes.append(f"LLM coach skipped: {exc}. The explanations keep their template text.")
        return

    positions = position_index(packet)
    accepted, rejected, seen = 0, 0, set()
    rejections: list[dict[str, Any]] = []
    for item in data.get("explanations") or []:
        epd = str(item.get("epd") or "") if isinstance(item, dict) else ""
        text = str(item.get("text") or "").strip() if isinstance(item, dict) else ""
        position = positions.get(epd)
        if position is None or epd in seen:
            rejected += 1
            problems = ["not a packet position, or a second text for it"]
            rejections.append({"epd": epd, "text": text, "problems": problems})
            continue
        seen.add(epd)
        cited = [str(c) for c in item.get("claim_ids") or []]
        check = verify_explanation(text, position, packet, cited)
        # the packet keeps the first explanation of each EPD: the text belongs to that one (same move played)
        played = position.get("played", "")
        target = next((e for e in coaching.explanations if e.epd == epd and e.played == played), None)
        if not check.ok or target is None:
            rejected += 1
            rejections.append({"epd": epd, "text": text, "problems": check.problems or ["no matching explanation"]})
            log.info("LLM text for %s rejected: %s", epd, "; ".join(check.problems))
            continue
        target.text, target.text_source = text, "llm"
        accepted += 1

    plan, dropped = verify_plan(data.get("weekly_plan") or [], packet, cfg.practice_minutes, today)
    coaching.weekly_plan = plan
    coaching.llm = {
        "model": cfg.llm_model,
        "accepted": accepted,
        "rejected": rejected,
        "plan_dropped": len(dropped),
        "rejections": rejections[:MAX_REJECTIONS_KEPT],
    }
    note = f"LLM coach ({cfg.llm_model}): {accepted} of {len(packet['positions'])} explanations rewritten"
    if rejected:
        note += f"; {rejected} failed the fact check and keep their template text"
    if dropped:
        note += f"; {len(dropped)} weekly-plan line{'s' if len(dropped) != 1 else ''} dropped ({dropped[0]})"
    coaching.notes.append(note + ".")
