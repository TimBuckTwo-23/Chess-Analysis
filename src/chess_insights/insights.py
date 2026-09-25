"""Turn every module's insights into one ranked summary and a study plan.

Whether something is a strength or weakness at all is decided by each module under the
project-wide significance rule in ``stats`` (``significance``, ``ALPHA``, ``STRICT_ALPHA``);
this module only ranks, dedupes and diversifies what passed.
"""

from __future__ import annotations

from typing import Iterable

from .models import Insight, ModuleResult, StudyItem

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
            "Replay your losses in this opening and mark the exact move where you left known theory.",
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
            "Adopt a stop rule: after two losses in a row, take at least a 15-minute break.",
            "Avoid rated games when tired; play puzzles instead.",
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
    "tactics": {
        "label": "Tactics",
        "actions": [
            "Solve themed puzzle sets (forks, pins, mating patterns) until the patterns become automatic.",
            "In every position, scan checks, captures and threats for both sides before choosing a move.",
        ],
    },
}


def all_insights(modules: Iterable[ModuleResult]) -> list[Insight]:
    return [ins for m in modules for ins in m.insights]


def _dedupe(insights: Iterable[Insight]) -> list[Insight]:
    best: dict[str, Insight] = {}
    for ins in insights:
        cur = best.get(ins.id)
        if cur is None or ins.priority > cur.priority:
            best[ins.id] = ins
    return list(best.values())


def _diverse_top(insights: list[Insight], top_n: int, per_category: int) -> list[Insight]:
    """Highest priority first, but at most ``per_category`` per category unless we run short."""
    ranked = sorted(insights, key=lambda i: (-i.priority, i.id))
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
    top_n: int = 5,
    min_confidence: float = MIN_CONFIDENCE,
    min_priority: float = 0.05,
    per_category: int = 2,
) -> tuple[list[Insight], list[Insight]]:
    """(top strengths, top weaknesses) across all modules.

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


def build_study_plan(weaknesses: list[Insight], *, max_items: int = 6, max_actions: int = 5) -> list[StudyItem]:
    """One study item per top weakness, most important first.

    Specific actions from the insight come first, then generic training advice
    for its category (each generic tip used once across the whole plan).
    """
    used_generic: set[str] = set()
    plan: list[StudyItem] = []
    for ins in sorted(weaknesses, key=lambda i: -i.priority)[:max_items]:
        lib = STUDY_LIBRARY.get(ins.category, {})
        actions = [a for a in ins.study if a]
        for tip in lib.get("actions", []):  # type: ignore[union-attr]
            if len(actions) >= max_actions:
                break
            if tip not in used_generic and tip not in actions:
                actions.append(tip)
                used_generic.add(tip)
        plan.append(
            StudyItem(
                title=ins.title,
                why=ins.detail,
                actions=actions[:max_actions],
                games=list(ins.example_games[:5]),
                category=str(lib.get("label", ins.category)),
                priority=ins.priority,
            )
        )
    return plan


def headline(strengths: list[Insight], weaknesses: list[Insight]) -> str:
    """One or two sentences for the top of the report / terminal."""
    if not strengths and not weaknesses:
        return "No clear patterns yet — play more games (or widen the filters) for reliable insights."
    parts = []
    if weaknesses:
        parts.append(f"Biggest opportunity: {weaknesses[0].title.rstrip('.')}.")
    if strengths:
        parts.append(f"Biggest strength: {strengths[0].title.rstrip('.')}.")
    return " ".join(parts)
