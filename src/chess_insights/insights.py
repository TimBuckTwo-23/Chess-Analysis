"""Turn every module's insights into one ranked summary and a study plan.

Whether something is a strength or weakness at all is decided by each module under the
project-wide significance rule in ``stats`` (``significance``, ``ALPHA``, ``STRICT_ALPHA``);
this module only ranks, dedupes and diversifies what passed.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, Optional, Sequence

from .models import Insight, ModuleResult, StudyItem
from .stats import MINUS, pct, per100_games

# Generic, well-known study material per weakness category. Module insights carry
# the *specific* actions ("review these 7 Caro-Kann losses"); this library adds the
# "how to train it" layer underneath.
STUDY_LIBRARY: dict[str, dict[str, object]] = {
    "results": {
        "label": "Overall results",
        "actions": [
            "Play fewer games in your weakest time control and more slower games you can learn from.",
        ],
    },
    "color": {
        "label": "Colour balance",
        "actions": [
            "Build a narrower, deeper repertoire for your weaker colour: one answer to 1.e4, one to 1.d4.",
        ],
    },
    "openings": {
        "label": "Opening repertoire",
        "actions": [
            "Study 3-5 annotated master games in the line so you learn the plans, not just moves.",
        ],
    },
    "opponents": {
        "label": "Playing to your rating",
        "actions": [
            "Against lower-rated opponents, slow down once you are ahead: check every capture and check before moving.",
        ],
    },
    "time": {
        "label": "Clock management",
        "actions": [
            "Set a budget: aim to reach move 15 with at least 70% of your starting clock.",
            "Play some games with increment (e.g. 3+2, 10+5) to practise finishing games without flagging.",
        ],
    },
    "endings": {
        "label": "How games end",
        "actions": [
            "Before resigning, check whether the opponent still has to prove a technical win.",
        ],
    },
    "habits": {
        "label": "Playing habits",
        "actions": [
            "Stop rule: after any loss, take at least a 15-minute break before the next rated game.",
        ],
    },
    "accuracy": {
        "label": "Overall accuracy",
        "actions": [
            "Use a blunder check before every move: what does my opponent's last move threaten, and what does my move leave undefended?",
        ],
    },
    "blunders": {
        "label": "Blunders",
        "actions": [
            "Do 20 minutes of puzzles a day at a rating slightly above yours, taking time to calculate fully.",
            "After each game, find your worst move and write down why you missed it.",
        ],
    },
    "phases": {
        "label": "Game phase",
        "actions": [
            "Endgame: work through the essential theoretical positions (Lucena, Philidor, king-and-pawn opposition).",
            "Middlegame: study annotated games in your openings' typical structures.",
        ],
    },
    "conversion": {
        "label": "Converting advantages",
        "actions": [
            "Practise winning won positions against an engine or a friend: start from +3 and play it out.",
            "When winning, trade pieces (not pawns), remove counterplay first, and don't rush.",
        ],
    },
    "positions": {
        "label": "Your own positions",
        "actions": [
            "Keep a file of the positions you got wrong and replay them once a week until the right move is automatic.",
        ],
    },
    "tactics": {
        "label": "Tactics",
        "actions": [
            "Solve themed puzzle sets (forks, pins, mating patterns: lichess.org/training/themes) until the patterns "
            "become automatic.",
            "In every position, scan checks, captures and threats for both sides before choosing a move.",
        ],
    },
}


def all_insights(modules: Iterable[ModuleResult]) -> list[Insight]:
    return [ins for m in modules for ins in m.insights]


# Findings from different modules about the same thing, e.g. openings.py ("you score less
# with the Caro-Kann") and engine_stats.py ("you leave the Caro-Kann worse off"). They share a
# topic and are merged in the headline lists; each module section still shows its own.
_TOPIC_PATTERNS = (
    re.compile(r"^openings\.(strength|weakness)\.(white|black)\.(.+)$"),
    re.compile(r"^engine\.(strength|weakness)\.openings-(white|black)-(.+)$"),
)


def topic_key(ins: Insight) -> str:
    for rx in _TOPIC_PATTERNS:
        m = rx.match(ins.id)
        if m:
            return f"opening:{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return ins.id


def _merge(keep: Insight, other: Insight) -> Insight:
    """``keep`` with the other finding's actions and games added (neither input is modified)."""
    study = list(keep.study) + [a for a in other.study if a not in keep.study]
    games = list(dict.fromkeys(list(keep.example_games) + list(other.example_games)))
    evidence = dict(keep.evidence)
    evidence["also_found_by"] = list(evidence.get("also_found_by") or []) + [other.id]
    return replace(keep, study=study[:5], example_games=games[:5], evidence=evidence)


def _dedupe(insights: Iterable[Insight]) -> list[Insight]:
    best: dict[str, Insight] = {}
    for ins in insights:
        key = topic_key(ins)
        cur = best.get(key)
        if cur is None:
            best[key] = ins
        elif cur.id == ins.id:
            best[key] = ins if ins.priority > cur.priority else cur
        else:
            best[key] = _merge(ins, cur) if ins.priority > cur.priority else _merge(cur, ins)
    return list(best.values())


def _diverse_top(insights: list[Insight], top_n: Optional[int], per_category: int) -> list[Insight]:
    """Highest priority first, but at most ``per_category`` per category unless we run short (all when ``top_n`` is None)."""
    ranked = sorted(insights, key=lambda i: (-i.priority, i.id))
    if top_n is None:
        return ranked
    picked: list[Insight] = []
    counts: dict[str, int] = {}
    for ins in ranked:
        if len(picked) >= top_n:
            break
        if counts.get(ins.category, 0) < per_category:
            picked.append(ins)
            counts[ins.category] = counts.get(ins.category, 0) + 1
    for ins in ranked:  # fill the remaining slots if categories were too concentrated
        if len(picked) >= top_n:
            break
        if ins not in picked:
            picked.append(ins)
    return sorted(picked, key=lambda i: (-i.priority, i.id))


# Every strength/weakness a module emits has passed the project-wide rule (stats.significance:
# enough games and an adjusted p-value at or below stats.ALPHA / STRICT_ALPHA), which always gives
# a confidence of at least 0.50 (stats.claim_confidence). The floor below is a safety net: anything
# under it cannot have passed the rule (e.g. a module still using a looser threshold) and stays out
# of the headline lists.
MIN_CONFIDENCE = 0.5


def rank_insights(
    modules: Iterable[ModuleResult],
    *,
    top_n: Optional[int] = 5,
    min_confidence: float = MIN_CONFIDENCE,
    min_priority: float = 0.05,
    per_category: int = 2,
) -> tuple[list[Insight], list[Insight]]:
    """(top strengths, top weaknesses) across all modules; ``top_n=None`` keeps every one, most important first.

    Observations never rank. The report's false-claim rate on data without real effects is
    checked by tests/test_null_calibration.py, and its power by tests/test_power.py.
    """
    pool = [
        i
        for i in _dedupe(all_insights(modules))
        if i.kind in ("strength", "weakness") and i.confidence >= min_confidence and i.priority >= min_priority
    ]
    strengths = _diverse_top([i for i in pool if i.kind == "strength"], top_n, per_category)
    weaknesses = _diverse_top([i for i in pool if i.kind == "weakness"], top_n, per_category)
    return strengths, weaknesses


# Study-plan groups: findings with the same cause become one plan item (a slow opening and losing
# on time are one clock problem; two weak openings are one repertoire job). Items are ordered by
# effort, cheapest first: when you play costs nothing, a clock habit takes a few games, your own
# positions ten minutes a day, a repertoire change a few weeks, technique longer.
# category -> (group key, label, effort note, effort rank)
PLAN_GROUPS: dict[str, tuple[str, str, str, int]] = {
    "habits": ("habits", "When you play", "costs nothing: start today", 0),
    "time": ("time", "Your clock", "practise it in your next games", 1),
    "positions": ("positions", "Your own positions", "10 minutes a day", 2),
    "openings": ("repertoire", "Your repertoire", "a few weeks of work", 3),
    "color": ("repertoire", "Your repertoire", "a few weeks of work", 3),
    "conversion": ("conversion", "Converting winning positions", "longer-term training", 4),
    "phases": ("phases", "Game phases", "longer-term training", 4),
    "blunders": ("blunders", "Blunders", "longer-term training", 4),
    "tactics": ("tactics", "Tactics", "longer-term training", 4),
    "accuracy": ("accuracy", "Accuracy", "longer-term training", 4),
}
_OTHER_EFFORT = 5
MAX_PLAN_GAMES = 3

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def similar_actions(a: str, b: str, threshold: float = 0.6) -> bool:
    """Two study actions saying nearly the same thing (most of the shorter one's words are in the other)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return a.strip().lower() == b.strip().lower()
    return len(wa & wb) / min(len(wa), len(wb)) >= threshold


def _add_action(actions: list[str], action: str, fuzzy: bool = True) -> bool:
    """Append ``action`` unless it repeats one already there (``fuzzy``: nearly the same words; else exactly)."""
    if not action or any(similar_actions(action, a) if fuzzy else action.strip() == a.strip() for a in actions):
        return False
    actions.append(action)
    return True


def plan_group(category: str) -> tuple[str, str, str, int]:
    """(group key, label, effort note, effort rank) of a weakness category."""
    if category in PLAN_GROUPS:
        return PLAN_GROUPS[category]
    label = str(STUDY_LIBRARY.get(category, {}).get("label", category or "Other"))
    return category or "other", label, "", _OTHER_EFFORT


def _move_label(fen: Any, san: Any) -> str:
    parts = str(fen or "").split()
    if not san or len(parts) < 2:
        return str(san or "")
    number = parts[5] if len(parts) > 5 else "1"
    return f"{number}.{san}" if parts[1] == "w" else f"{number}...{san}"


def _pawns(cp: float) -> str:
    return f"{cp / 100:+.1f}".replace("-", MINUS)


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value else None


def target_for(ins: Insight) -> tuple[str, dict[str, Any]]:
    """What to aim for by the next report, with today's number: (sentence, {"metric", "value"}); ("", {}) if none."""
    ev = dict(ins.evidence or {})
    classes = ev.get("time_classes")
    if classes and isinstance(ev.get(classes[0]), dict):  # merged across time controls: the first one
        ev = {**ev[classes[0]], "time_class": classes[0]}
    tc = ev.get("time_class") or ""
    i = ins.id
    if ".slow-opening" in i and _num(ev.get("your_share")) is not None and _num(ev.get("opponent_share")) is not None:
        mine, theirs = ev["your_share"], ev["opponent_share"]
        goal = max(0.05, round((theirs + 0.05) / 0.05) * 0.05)
        return (
            f"Use at most {pct(goal)} of your clock on your first 15 moves in {tc} (now {pct(mine)}).",
            {"metric": f"share of the clock used on the first 15 moves ({tc})", "value": mine},
        )
    if ".lost-on-time" in i and ev.get("n"):
        lost, won, n = ev.get("lost_on_time", 0), ev.get("won_on_time", 0), ev["n"]
        return (
            f"Lose no more {tc} games on time than you win on time (now {lost} lost and {won} won in {n} games).",
            {"metric": f"losses on time minus wins on time per game ({tc})", "value": (lost - won) / n},
        )
    if ".time-trouble" in i and _num(ev.get("trouble_rate")) is not None:
        return (
            f"Get into time trouble no more often than your opponents in {tc}: "
            f"{pct(ev.get('opponent_trouble_rate'))} of games (now {pct(ev['trouble_rate'])}).",
            {"metric": f"share of games in time trouble ({tc})", "value": ev["trouble_rate"]},
        )
    if i.endswith(".after-a-loss") and _num(ev.get("delta")) is not None:
        return (
            f"Score in line with your rating in games started right after a loss (now {per100_games(ev['delta'])}).",
            {"metric": "score vs rating right after a loss, per game", "value": ev["delta"]},
        )
    if (".late-night" in i or ".time-of-day-" in i) and ev.get("n"):
        window = ev.get("window") or ev.get("block") or ""
        return (
            f"No rated games started {window} (now {ev['n']} of your rated games).",
            {"metric": f"rated games started {window}", "value": ev["n"]},
        )
    if ".long-sessions" in i and ev.get("from_game"):
        return (
            f"Stop each session after {int(ev['from_game']) - 1} games.",
            {"metric": "score vs rating late in a session, per game", "value": ev.get("delta")},
        )
    if i.startswith("openings.weakness.early-losses") and ev.get("losses"):
        return (
            f"Fewer quick losses in the {ev.get('family')} (now {ev.get('short_losses')} of {ev['losses']} losses "
            "were over within 25 moves).",
            {"metric": f"quick losses in the {ev.get('family')}", "value": ev.get("short_losses")},
        )
    if i.startswith("openings.weakness.") and _num(ev.get("delta")) is not None:
        return (
            f"Score in line with your rating in the {ev.get('family')} (now {per100_games(ev['delta'])}).",
            {"metric": f"score vs rating in the {ev.get('family')} ({ev.get('color')}), per game", "value": ev["delta"]},
        )
    if ".openings-" in i and _num(ev.get("avg_cp")) is not None:
        return (
            f"Come out of the {ev.get('family')} level by move 10 (now {_pawns(ev['avg_cp'])} on average).",
            {"metric": f"engine eval after move 10 in the {ev.get('family')} (centipawns)", "value": ev["avg_cp"]},
        )
    if i.endswith(".conversion") and _num(ev.get("rate")) is not None and _num(ev.get("opp_rate")) is not None:
        return (
            f"Win {pct(ev['opp_rate'])} of the games where you reach a winning position, like your opponents "
            f"(now {pct(ev['rate'])}).",
            {"metric": "share of winning positions converted", "value": ev["rate"]},
        )
    if (".phase-" in i or ".blunder-rate" in i) and _num(ev.get("per100")) is not None:
        where = f" in the {ev['phase']}" if ev.get("phase") else ""
        what = "mistakes and blunders" if ev.get("phase") else "blunders"
        return (
            f"Cut your {what}{where} to your opponents' {ev.get('opp_per100', 0):.1f} per 100 moves "
            f"(now {ev['per100']:.1f}).",
            {"metric": f"{what}{where} per 100 moves", "value": ev["per100"]},
        )
    if i.startswith("mistakes.weakness") and ev.get("best"):
        best, played = _move_label(ev.get("fen"), ev["best"]), _move_label(ev.get("fen"), ev.get("move"))
        return (
            f"Play {best} the next time this position comes up (you played {played} in {ev.get('errors')} of "
            f"{ev.get('reached')} games).",
            {"metric": f"games with {played} in this position", "value": ev.get("errors")},
        )
    if _num(ev.get("delta")) is not None:
        return (
            f"Score in line with your rating here (now {per100_games(ev['delta'])}).",
            {"metric": "score vs rating, per game", "value": ev["delta"]},
        )
    return "", {}


def _round_robin(
    lists: Sequence[Sequence[str]], limit: int, taken: Optional[list[str]] = None, fuzzy_within: bool = True
) -> list[str]:
    """The first entry of each list, then the second of each ..., up to ``limit``, skipping repeats.

    Two findings' actions often share a template ("Replay your 5 most recent losses in the X ...") while being
    about different things, so across lists only exact repeats are dropped; within one list, near-duplicates too.
    """
    out = list(taken or [])
    for k in range(max((len(x) for x in lists), default=0)):
        for items in lists:
            if len(out) >= limit:
                return out
            if k < len(items):
                action = str(items[k])
                if not fuzzy_within or not any(similar_actions(action, str(prev)) for prev in items[:k]):
                    _add_action(out, action, fuzzy=False)
    return out


class PracticeActions(list):
    """What :func:`practice_actions` returns: the actions for the "your own positions" item (a plain list of
    strings, as before), plus ``drills``: one action per drill pack (coach/drills.py), keyed by Lichess theme in
    pack order, for the tactics and blunders items of the study plan (:func:`drill_actions_for`)."""

    def __init__(self, actions: Iterable[str] = (), drills: Optional[dict[str, str]] = None):
        super().__init__(actions)
        self.drills: dict[str, str] = dict(drills or {})


# Plan items that get a drill pack, and the packs that fit a finding (by its id) better than any other.
DRILL_CATEGORIES = ("tactics", "blunders")
_DRILL_THEMES_BY_ID = (
    ("hung-material", ("hangingPiece",)),
    ("missed-mates", ("mate", "mateIn1", "mateIn2", "mateIn3", "backRankMate")),
)


def drill_pack_action(drill: Any) -> str:
    """'30 fork puzzles in me-drill-fork.pgn, or lichess.org/training/fork (you missed 23 forks in 300 games ...).'"""
    where = f" in {drill.file}, or" if getattr(drill, "file", "") else " at"
    link = str(getattr(drill, "link", "") or "").replace("https://", "")
    reason = f" ({drill.reason})" if getattr(drill, "reason", "") else ""
    return f"{drill.title}{where} {link}{reason}."


def is_generic_puzzle_action(action: str) -> bool:
    """A generic 'solve puzzles' tip (STUDY_LIBRARY, engine TACTIC_TEXT), which a drill pack replaces."""
    text = str(action).lower()
    return "puzzle" in text and "your own" not in text


def _own_themes(lead: Insight) -> list[str]:
    """The drill themes a finding is about: its pattern (motif findings), or the patterns behind its engine tag."""
    themes = [str(lead.evidence["theme"])] if (lead.evidence or {}).get("theme") else []
    for key, extra in _DRILL_THEMES_BY_ID:
        if key in lead.id:
            themes += list(extra)
    return list(dict.fromkeys(themes))


def drill_actions_for(leads: Sequence[Insight], practice: Sequence[str]) -> dict[str, str]:
    """Lead finding id -> the drill action for its plan item, for the tactics and blunders items ({} without packs).

    Items whose finding is about a pattern with a pack get that pack first; the others take the remaining packs in
    pack order (your most-missed pattern first). A pack serves two items only when there are more items than packs.
    """
    drills: dict[str, str] = getattr(practice, "drills", None) or {}
    eligible = [lead for lead in leads if lead.category in DRILL_CATEGORIES]
    if not drills or not eligible:
        return {}
    chosen: dict[str, str] = {}
    for lead in eligible:
        theme = next((t for t in _own_themes(lead) if t in drills and t not in chosen.values()), None)
        if theme:
            chosen[lead.id] = theme
    for lead in eligible:
        if lead.id not in chosen:
            free = [t for t in drills if t not in chosen.values()]
            own = [t for t in _own_themes(lead) if t in drills]
            chosen[lead.id] = (free or own or list(drills))[0]
    return {lead_id: drills[theme] for lead_id, theme in chosen.items()}


def practice_actions(modules: Iterable[ModuleResult], coaching: Any = None) -> PracticeActions:
    """Study actions built from facts, not claims: your own mistakes as puzzles (needs the engine), and one
    puzzle-pack action per drill pack of the coaching layer (``coaching.drills``) for the tactics and blunders items.

    The result is a list (the positions item's actions, as before) with the pack actions in ``.drills``.
    """
    m = next((m for m in modules if m.key == "mistakes"), None)
    stats = (m.stats if m else None) or {}
    n = stats.get("puzzles_exported") or 0
    drills = {
        d.theme: drill_pack_action(d)
        for d in (getattr(coaching, "drills", None) or [])
        if getattr(d, "puzzles", None) and getattr(d, "theme", "openings") != "openings"
    }
    if not n:
        return PracticeActions([], drills)
    where = f"in {stats['puzzle_file']}" if stats.get("puzzle_file") else "(export them with --puzzles)"
    return PracticeActions(
        [
            f"Solve 10 of your own puzzles a day: {n} positions from your games where your move cost a lot, "
            f"{where}. Import the file into a Lichess study or any chess program."
        ],
        drills,
    )


def build_study_plan(
    weaknesses: list[Insight],
    *,
    max_items: int = 6,
    max_actions: int = 3,
    practice: Sequence[str] = (),
) -> list[StudyItem]:
    """One plan item per cause, cheapest to fix first (see PLAN_GROUPS).

    Each item takes its title, reasons and target from its most important finding, lists the others it
    covers, and gets at most ``max_actions`` actions: the findings' own, taken in turn, near-duplicates
    left out; generic training advice only fills empty places. ``practice`` actions (your own puzzles)
    go to the positions item, which is created for them when no repeated-position finding exists; with drill
    packs (``practice.drills``, from :func:`practice_actions`) each tactics and blunders item gets one pack
    action in place of the generic "solve puzzles" tips.
    """
    groups: dict[str, list[Insight]] = {}
    for ins in sorted(weaknesses, key=lambda i: (-i.priority, i.id)):
        groups.setdefault(plan_group(ins.category)[0], []).append(ins)
    if practice and "positions" not in groups:
        groups["positions"] = []
    info = {k: plan_group(members[0].category if members else "positions") for k, members in groups.items()}
    order = sorted(groups, key=lambda k: (info[k][3], -(groups[k][0].priority if groups[k] else 0.0), k))
    plan: list[StudyItem] = []
    drill_actions = drill_actions_for([groups[k][0] for k in order[:max_items] if groups[k]], practice)
    for key in order[:max_items]:
        members = groups[key]
        _, label, effort, _ = info[key]
        kicker = f"{label} · {effort}" if effort else label
        if not members:  # practice only: no finding behind it
            actions = _round_robin([list(practice)], max_actions)
            plan.append(
                StudyItem(
                    title="Practise your own mistakes as puzzles",
                    why="Stockfish marked the moves in your games that cost the most. Solving them is the most "
                    "direct training there is: the positions come from your openings and your kind of game.",
                    actions=actions,
                    category=kicker,
                    target="Solve every puzzle in the file once before your next report.",
                )
            )
            continue
        lead = members[0]
        drill = drill_actions.get(lead.id, "")
        keep = (lambda a: not is_generic_puzzle_action(a)) if drill else (lambda a: True)
        reserved = 1 if (key == "positions" and practice) or drill else 0
        actions = _round_robin([[a for a in m.study if keep(a)] for m in members], max_actions - reserved)
        for tip in STUDY_LIBRARY.get(lead.category, {}).get("actions", []):  # type: ignore[union-attr]
            if len(actions) >= max_actions - reserved:
                break
            if keep(str(tip)):
                _add_action(actions, str(tip))
        if drill:
            actions = _round_robin([[drill]], max_actions, taken=actions)
        elif reserved:
            actions = _round_robin([list(practice)], max_actions, taken=actions)
        games = _round_robin([m.example_games for m in members], MAX_PLAN_GAMES, fuzzy_within=False)
        ids: list[str] = []
        for m in members:
            ids += [m.id, *[str(x) for x in (m.evidence or {}).get("also_found_by") or []]]
        target, baseline = target_for(lead)
        if baseline:
            baseline = {**baseline, "insight": lead.id}
        plan.append(
            StudyItem(
                title=lead.title,
                why=lead.detail,
                actions=actions,
                games=games,
                category=kicker,
                priority=max(m.priority for m in members),
                insight_ids=list(dict.fromkeys(ids)),
                findings=[m.title for m in members],
                target=target,
                baseline=baseline,
            )
        )
    return plan


def standing_line(modules: Iterable[ModuleResult], min_games: int = 20, max_pools: int = 3) -> str:
    """'Your ratings: rapid 1756 (+225 over these games), blitz 1379 (−79).' from the results section."""
    m = next((m for m in modules if m.key == "results"), None)
    pools = ((m.stats if m else None) or {}).get("by_time_control") or {}
    rows = []
    for pool, s in pools.items():
        rating = (s or {}).get("rating") or {}
        if not isinstance(rating, dict) or rating.get("current") is None or (rating.get("n") or 0) < min_games:
            continue
        rows.append((rating.get("n") or 0, str(pool), rating))
    if not rows:
        return ""
    parts = []
    for k, (_, pool, rating) in enumerate(sorted(rows, key=lambda r: (-r[0], r[1]))[:max_pools]):
        text = f"{pool.lower()} {rating['current']}"
        change = rating.get("change")
        if change is None and isinstance(rating.get("start"), (int, float)):
            change = rating["current"] - rating["start"]
        if isinstance(change, (int, float)):
            sign = f"{change:+d}".replace("-", MINUS) if change else "±0"
            text += f" ({sign} over these games)" if k == 0 else f" ({sign})"
        parts.append(text)
    return "Your ratings: " + ", ".join(parts) + "."


def summary_lines(
    modules: Iterable[ModuleResult], plan: Sequence[StudyItem], strengths: Sequence[Insight], weaknesses: Sequence[Insight]
) -> list[str]:
    """The top of the report in three short lines: where you stand, where to start, a strength to build on."""
    modules = list(modules)
    lines = []
    standing = standing_line(modules)
    if standing:
        lines.append(standing)
    if plan:
        firsts = [f"{k}. {item.title.rstrip('.')}" for k, item in enumerate(plan[:3], 1)]
        lines.append("Start with: " + " · ".join(firsts) + ".")
    if strengths:
        lines.append(f"Strength to build on: {strengths[0].title.rstrip('.')}.")
    if not strengths and not weaknesses:
        lines.append("No clear strengths or weaknesses yet: play more games (or widen the filters) for reliable insights.")
    return lines


def headline(
    strengths: Sequence[Insight], weaknesses: Sequence[Insight], plan: Optional[Sequence[StudyItem]] = None
) -> str:
    """One or two sentences for the top of the report / terminal: where to start, and a strength to build on."""
    if not strengths and not weaknesses:
        return "No clear patterns yet — play more games (or widen the filters) for reliable insights."
    parts = []
    if plan:
        parts.append(f"Start with: {plan[0].title.rstrip('.')}.")
    elif weaknesses:
        parts.append(f"Biggest opportunity: {weaknesses[0].title.rstrip('.')}.")
    if strengths:
        parts.append(f"Strength to build on: {strengths[0].title.rstrip('.')}.")
    return " ".join(parts)
