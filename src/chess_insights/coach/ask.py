"""`chess-insights ask USERNAME "question"`: answers from the last report's JSON, through the verifier (C4).
Owner: llm.

The question goes to the Claude API with the coaching packet built from the report JSON. The answer is shown only
if it passes the same checks as the report's explanations (every move in a packet line, every number in the
packet, strengths and weaknesses only where the report claims them). Otherwise, and whenever no answer can be
had (no API key, the llm extra not installed, an API error), the reply says so and quotes what the report itself
says about the question, which is always safe.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..visuals import format_text
from .config import CoachConfig
from . import llm
from .packet import build_packet, line_text
from .verify import claim_words, content_words, verify_answer

MAX_FACTS = 4  # report facts quoted when there is no checked answer
MAX_SENTENCES = 8

SYSTEM_PROMPT = """\
You answer a chess player's question about their own games, from a JSON packet of the facts their report \
established. A program checks the answer before the player sees it and discards it if it breaks one of these rules:

1. Use only facts in the packet. Every number you write must appear in the packet; you may round it. Give \
evaluations in pawns, never in centipawns.
2. Name only moves from the lines of the packet's positions (best_line, refutation, opening moves, \
tablebase.best), numbered as the packet numbers them ("5...e5", "6.Ndb5"), and list the epd of every position \
whose moves you name in epds. When you mean a square rather than a move, write it on its own ("the d6 square").
3. Call something a strength or a weakness only when it is one of the packet's claims (claims or format_claims), \
and list its id in claim_ids. Never write an id in the answer.
4. When the packet does not answer the question, say so in one sentence and point to the closest facts it has.

Answer in at most eight sentences of plain text without markdown, speaking to the player as "you". Say which \
format (bullet, blitz or rapid) a fact is about whenever the packet says so."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claim_ids": {"type": "array", "items": {"type": "string"}},
        "epds": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "claim_ids", "epds"],
    "additionalProperties": False,
}


def _formats(counts: Any) -> str:
    text = format_text(dict(counts or {})) if isinstance(counts, dict) else ""
    return f" ({text})" if text else ""


def relevant_facts(question: str, packet: dict[str, Any], limit: int = MAX_FACTS) -> list[str]:
    """What the report says that shares words with ``question``: claims, study-plan items, explained positions.

    Falls back to the report's summary lines when nothing matches.
    """
    asked = content_words(question)
    scored: list[tuple[int, int, str]] = []
    claims = list(packet.get("claims") or [])
    main_ids = {c.get("id") for c in claims}
    for view in (packet.get("format_claims") or {}).values():  # per-format claims the main list doesn't have
        claims += [c for c in view or [] if c.get("id") not in main_ids]
    for i, c in enumerate(claims):
        hits = len(asked & claim_words(c))
        if hits:
            kind = "Strength" if c.get("kind") == "strength" else "Weakness"
            view = f", {c['view']} games only" if c.get("view") else ""
            scored.append((hits, i, f"{kind}{_formats(c.get('formats'))}{view}: {c.get('title')}. {c.get('detail')}"))
    for i, item in enumerate(packet.get("study_plan") or []):
        hits = len(asked & content_words(" ".join(str(item.get(k) or "") for k in ("title", "why", "target"))))
        if hits:
            target = f" Target: {item['target']}" if item.get("target") else ""
            scored.append((hits, 100 + i, f"Study plan: {item.get('title')}.{target}"))
    for i, p in enumerate(packet.get("positions") or []):
        opening = (p.get("opening") or {}).get("name") or ""
        themes = " ".join(str(m.get("theme") or "") for m in p.get("motifs") or [])
        words = content_words(" ".join([opening, str(p.get("time_class") or ""), themes]))
        hits = len(asked & words)
        if hits:
            where = ", ".join(x for x in (opening, str(p.get("time_class") or "")) if x)
            refutation = line_text((p.get("refutation") or {}).get("moves") or [])
            text = f"Position{f' ({where})' if where else ''}: you played {p.get('played')}"
            text += f", and the engine's line runs {refutation}" if refutation else ""
            text += f"; better was {p['best']}." if p.get("best") else "."
            scored.append((hits, 200 + i, text))
    if not scored:
        return [str(s) for s in (packet.get("report") or {}).get("summary") or []][:limit]
    return [text for _, _, text in sorted(scored, key=lambda s: (-s[0], s[1]))[:limit]]


def _fallback(reason: str, question: str, packet: dict[str, Any]) -> str:
    facts = relevant_facts(question, packet)
    if not facts:
        return f"{reason} Your report has nothing more to go on yet."
    return f"{reason} Here is what your report says that may help:\n" + "\n".join(f"- {f}" for f in facts)


def answer(question: str, report_json: dict[str, Any], cfg: CoachConfig, client: Optional[Any] = None) -> str:
    """A grounded answer to ``question`` (moves and numbers checked against the report), or an explanation of
    why none can be given (no API key, the llm extra not installed, nothing relevant in the report).

    ``client`` is for tests (anything with ``messages.create``).
    """
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    packet = build_packet(report_json or {})
    if not question:
        return 'Ask a question about your report, e.g. chess-insights ask USERNAME "why do I lose with the Alapin?"'
    sdk = None
    if client is None:
        sdk = llm.load_sdk()
        missing = []
        if sdk is None:
            missing.append(f"the Anthropic SDK ({llm.INSTALL_HINT})")
        if not llm.has_credentials(cfg):
            missing.append("an API key (set ANTHROPIC_API_KEY)")
        if missing:
            return _fallback(f"Answering questions needs {' and '.join(missing)}.", question, packet)
        client = llm.make_client(cfg, sdk)
    facts = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    user = f"<packet>\n{facts}\n</packet>\n\nQuestion: {question}"  # the long packet first, the question last
    try:
        data = llm.call_json(client, cfg, SYSTEM_PROMPT, user, OUTPUT_SCHEMA, sdk)
    except llm.LLMError as exc:
        return _fallback(f"No answer this time: {exc}.", question, packet)
    text = str(data.get("answer") or "").strip()
    check = verify_answer(
        text,
        packet,
        epds=[str(e) for e in data.get("epds") or []],
        claim_ids=[str(c) for c in data.get("claim_ids") or []],
        max_sentences=MAX_SENTENCES,
    )
    if not check.ok:
        return _fallback("I could not check an answer to that against your report.", question, packet)
    return text
