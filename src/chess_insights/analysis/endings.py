"""How your games end: result mix, loss profile, abandoned games, game length and draws.

* Result mix per time class: wins, draws and losses split by how they ended
  (every game counts here, including aborted starts).
* Loss profile: how your losses end compared with how your wins end.
* Game length: score vs the Elo expectation by number of moves; each length band is
  compared with your games of other lengths (Benjamini-Hochberg adjusted over the five
  bands).
* Draw rate per time class.

Skill metrics (loss profile, game length) leave out games with fewer than 4
plies; game length also leaves out abandoned games, which say nothing about how
long you can hold a position. Timeouts are only described here: the clock
module owns the "you lose on time" finding.

Nothing here becomes a strength or weakness, only an observation, and only when its
test passes the project-wide rule (``stats.significance`` at ``stats.STRICT_ALPHA``).
How long a game lasts and how a loss ends are consequences of the result and of both
players' resignation habits as much as of skill: a player who never resigns is mated
more often and loses more long games, one who resigns early loses more short ones,
with no difference in strength. From results alone these readings can't be told apart
(the engine analysis can: conversion, endgame accuracy and missed mates).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..context import AnalysisContext
from ..models import TIME_CLASSES, Chart, Game, Insight, Kpi, ModuleResult, Series, Table
from ..stats import (
    STRICT_ALPHA,
    MeanTest,
    ScoreSummary,
    bh_adjust,
    clamp,
    difference_test,
    pct,
    sample_confidence,
    score_to_elo_diff,
    severity_from_points,
    shrink_effect,
    significance,
    summarize,
    two_proportion_test,
)
from .results import MAX_EXAMPLES, fmt_points, recent_urls

KEY = "endings"
TITLE = "How your games end"

MIN_PLIES = 4

# Result mix: (key, label). Decisive games split by how they ended, all draws together.
MIX: tuple[tuple[str, str], ...] = (
    ("win_checkmate", "Won by checkmate"),
    ("win_resignation", "Won by resignation"),
    ("win_timeout", "Won on time"),
    ("win_other", "Won: abandoned / other"),
    ("draw", "Drawn"),
    ("loss_other", "Lost: abandoned / other"),
    ("loss_timeout", "Lost on time"),
    ("loss_resignation", "Lost by resignation"),
    ("loss_checkmate", "Lost by checkmate"),
)
METHODS: tuple[tuple[str, str], ...] = (
    ("checkmate", "Checkmate"),
    ("resignation", "Resignation"),
    ("timeout", "On time"),
    ("abandoned", "Abandoned"),
    ("other", "Other"),
)
DRAW_TYPES: tuple[tuple[str, str], ...] = (
    ("agreement", "Agreement"),
    ("repetition", "Repetition"),
    ("stalemate", "Stalemate"),
    ("insufficient", "Insufficient material"),
    ("fifty_move", "50-move rule"),
    ("timeout_vs_insufficient", "Flag vs insufficient material"),
    ("other", "Other"),
)
# (key, label, first move, last move) by Game.full_moves
LENGTH_BUCKETS: tuple[tuple[str, str, int, Optional[int]], ...] = (
    ("short", "≤ 20", 0, 20),
    ("early", "21–30", 21, 30),
    ("middle", "31–45", 31, 45),
    ("long", "46–60", 46, 60),
    ("very_long", "61+", 61, None),
)


@dataclass(frozen=True)
class Thresholds:
    """Minimum sample sizes and effect sizes; override with ``ctx.options["endings.<name>"]``."""

    min_games: int = 10  # below this the summary says there is not enough data
    strict_alpha: float = STRICT_ALPHA  # observations here must still pass the strict test (see the module docstring)
    min_losses: int = 20  # losses before comparing how your losses and wins end
    min_wins: int = 10
    min_mate_gap: float = 0.10  # share of losses by mate minus share of wins by mate
    min_length_games: int = 20  # rated games in a length bucket
    min_length_effect: float = 0.08  # points per game
    min_abandoned: int = 2  # abandoned games before mentioning them
    min_tc_games: int = 15  # games in a time class for the draw-rate observation

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- classification
def method(game: Game) -> str:
    """How a decisive game ended: one of METHODS' keys."""
    return game.termination if game.termination in ("checkmate", "resignation", "timeout", "abandoned") else "other"


def mix_key(game: Game) -> str:
    """Result-mix category (MIX key) of a game."""
    if game.outcome == "draw":
        return "draw"
    how = method(game)
    if how == "abandoned":
        how = "other"
    return f"{game.outcome}_{how}"


def draw_type(game: Game) -> str:
    keys = {k for k, _ in DRAW_TYPES}
    return game.termination if game.termination in keys else "other"


def length_bucket(full_moves: int) -> str:
    """LENGTH_BUCKETS key for a game of ``full_moves`` moves."""
    for key, _, lo, hi in LENGTH_BUCKETS:
        if full_moves >= lo and (hi is None or full_moves <= hi):
            return key
    return LENGTH_BUCKETS[-1][0]


def _group_by_class(games: Sequence[Game]) -> dict[str, list[Game]]:
    groups: dict[str, list[Game]] = {}
    for g in games:
        groups.setdefault(g.time_class, []).append(g)
    rank = {tc: i for i, tc in enumerate(TIME_CLASSES)}
    return {tc: groups[tc] for tc in sorted(groups, key=lambda tc: (rank.get(tc, len(rank)), tc))}


# --------------------------------------------------------------------------- tables & charts
def mix_shares(games: Sequence[Game]) -> list[Optional[float]]:
    """Share of ``games`` in each MIX category (None for an empty group)."""
    counts = Counter(mix_key(g) for g in games)
    n = len(games)
    return [counts.get(key, 0) / n if n else None for key, _ in MIX]


def mix_table(by_class: dict[str, list[Game]], games: Sequence[Game]) -> Table:
    rows = [[tc.capitalize(), len(gs), *mix_shares(gs)] for tc, gs in by_class.items()]
    if len(by_class) > 1:
        rows.append(["All", len(games), *mix_shares(games)])
    return Table(
        title="Result mix by time control",
        columns=["Time control", "Games", *(label for _, label in MIX)],
        rows=rows,
        formats=["text", "int", *(["pct"] * len(MIX))],
        note="Share of all games in each time control, including aborted and abandoned ones.",
    )


def mix_chart(by_class: dict[str, list[Game]]) -> Chart:
    shares = {tc: mix_shares(gs) for tc, gs in by_class.items()}
    return Chart(
        kind="stacked_bar",
        title="How your games end",
        labels=[tc.capitalize() for tc in by_class],
        series=[Series(label, [shares[tc][i] for tc in by_class]) for i, (_, label) in enumerate(MIX)],
        value_format="pct",
        note="Each bar adds up to all games in that time control.",
    )


@dataclass
class DecisiveProfile:
    """How the decisive games among ``games`` ended."""

    wins: int
    losses: int
    by_method: dict[str, tuple[int, int]]  # method -> (losses, wins)

    def loss_share(self, how: str) -> Optional[float]:
        return self.by_method[how][0] / self.losses if self.losses else None

    def win_share(self, how: str) -> Optional[float]:
        return self.by_method[how][1] / self.wins if self.wins else None


def decisive_profile(games: Sequence[Game]) -> DecisiveProfile:
    counts = Counter((method(g), g.outcome) for g in games if g.outcome != "draw")
    return DecisiveProfile(
        wins=sum(1 for g in games if g.outcome == "win"),
        losses=sum(1 for g in games if g.outcome == "loss"),
        by_method={k: (counts.get((k, "loss"), 0), counts.get((k, "win"), 0)) for k, _ in METHODS},
    )


def profile_table(p: DecisiveProfile, n_short: int) -> Table:
    note = "Share of your losses and of your wins that ended each way."
    if n_short:
        note += f" {n_short} game(s) with fewer than {MIN_PLIES} plies are left out."
    return Table(
        title="How your losses and wins end",
        columns=["Ending", "Losses", "Share of losses", "Wins", "Share of wins"],
        rows=[[label, p.by_method[k][0], p.loss_share(k), p.by_method[k][1], p.win_share(k)] for k, label in METHODS],
        formats=["text", "int", "pct", "int", "pct"],
        note=note,
    )


def draw_table(by_class: dict[str, list[Game]]) -> Table:
    rows = []
    for tc, gs in by_class.items():
        draws = [g for g in gs if g.outcome == "draw"]
        types = Counter(draw_type(g) for g in draws)
        counts = [types.get(k, 0) for k, _ in DRAW_TYPES]
        rows.append([tc.capitalize(), len(gs), len(draws), len(draws) / len(gs), *counts])
    return Table(
        title="Draws by time control",
        columns=["Time control", "Games", "Draws", "Draw rate", *(label for _, label in DRAW_TYPES)],
        rows=rows,
        formats=["text", "int", "int", "pct", *(["int"] * len(DRAW_TYPES))],
    )


def length_table(summaries: dict[str, ScoreSummary], n_left_out: int) -> Table:
    rows = []
    for key, label, _, _ in LENGTH_BUCKETS:
        s = summaries[key]
        rows.append([label, s.n, s.wdl, s.rated_score, s.expected, s.delta, s.half_width])
    note = (
        "Moves = full moves (a 41-ply game has 21). Difference = your score minus the Elo expected score, per game; "
        "± is the 95% margin of error."
    )
    if n_left_out:
        note += f" {n_left_out} abandoned or very short game(s) are left out."
    return Table(
        title="Score by game length",
        columns=["Moves", "Games", "W/D/L", "Score", "Expected", "Difference", "± (95%)"],
        rows=rows,
        formats=["text", "int", "text", "pct", "pct", "signed_pct", "pct"],
        note=note,
    )


def length_chart(summaries: dict[str, ScoreSummary]) -> Chart:
    return Chart(
        kind="bar",
        title="Score vs expected by game length (moves)",
        labels=[label for _, label, _, _ in LENGTH_BUCKETS],
        series=[Series("Score − expected", [summaries[key].delta for key, _, _, _ in LENGTH_BUCKETS])],
        value_format="signed_pct",
        note="Per game, against the Elo expectation. Empty bars are lengths without rated games.",
        reference=0.0,
    )


# --------------------------------------------------------------------------- insights
def mate_insight(
    p: DecisiveProfile, played: Sequence[Game], th: Thresholds
) -> tuple[Optional[Insight], dict[str, Any]]:
    """Checkmated in a larger share of losses than you checkmate in wins (self-relative): an observation.

    A pooled two-proportion test at the strict level. Never a weakness: a player who plays on in
    lost positions instead of resigning is mated more often with no difference in king safety.
    """
    mated, mating = p.by_method["checkmate"]
    test = two_proportion_test(mated, p.losses, mating, p.wins)
    stats = {
        "mated": mated,
        "losses": p.losses,
        "mating": mating,
        "wins": p.wins,
        "gap": test.mean,
        "p_value": test.p_value,
    }
    if p.losses < th.min_losses or p.wins < th.min_wins:
        return None, stats
    significant, confidence = significance(test, th.min_wins, alpha=th.strict_alpha, n=min(p.losses, p.wins))
    if test.mean < th.min_mate_gap or not significant:
        return None, stats
    mated_games = [g for g in played if g.outcome == "loss" and g.termination == "checkmate"]
    resigned, opp_resigned = p.by_method["resignation"]
    k = min(MAX_EXAMPLES, len(mated_games))
    return (
        Insight(
            id=f"{KEY}.observation.checkmated-often",
            kind="observation",
            category="endings",
            title=(
                f"{pct(p.loss_share('checkmate'))} of your losses end in checkmate, against "
                f"{pct(p.win_share('checkmate'))} of your wins"
            ),
            detail=(
                f"{mated} of your {p.losses} losses ended in checkmate, and {mating} of your {p.wins} wins ended "
                f"with you mating your opponent. You resigned {pct(p.loss_share('resignation'))} of your losses; "
                f"your opponents resigned {pct(p.win_share('resignation'))} of your wins. Being mated in a larger "
                "share of games can point at king-safety problems, but it is also exactly what playing on in lost "
                "positions looks like (a resignation becomes a mate), so on its own it is not evidence of a "
                "weakness. The engine analysis (--engine) can tell the two apart."
            ),
            severity=clamp(test.mean / 0.30),
            confidence=confidence,
            evidence={**stats, "resigned": resigned, "opponent_resigned": opp_resigned},
            study=[
                f"Replay your last {k} losses by checkmate: was the position already lost when the attack started? "
                "If not, find the move where your king's cover was first weakened.",
                "Solve 15 mating-pattern puzzles a day (back rank, smothered, Anastasia's, Arabian mate) until you "
                "spot them instantly for both sides.",
            ],
            example_games=recent_urls(mated_games),
        ),
        stats,
    )


def abandoned_insight(games: Sequence[Game], th: Thresholds) -> tuple[Optional[Insight], dict[str, Any]]:
    mine = [g for g in games if g.outcome == "loss" and g.my_result_code == "abandoned"]
    theirs = [g for g in games if g.outcome == "win" and g.opp_result_code == "abandoned"]
    expected = [g.expected_score for g in mine if g.expected_score is not None]
    stats = {"abandoned": len(mine), "opponent_abandoned": len(theirs), "expected_points": sum(expected)}
    if len(mine) < th.min_abandoned:
        return None, stats
    detail = (
        f"You abandoned {len(mine)} of your {len(games)} games (your opponents abandoned {len(theirs)}). "
        "Each one counts as a full loss and costs rating points in rated games."
    )
    if expected:
        detail += (
            f" Going by the ratings you could have expected about {sum(expected):.1f} points from "
            f"{'those games' if len(expected) > 1 else 'that game'}."
        )
    return (
        Insight(
            id=f"{KEY}.observation.abandoned-games",
            kind="observation",
            category="endings",
            title=f"You abandoned {len(mine)} games, and each one costs rating points",
            detail=detail,
            severity=clamp(len(mine) / len(games) * 10.0),
            confidence=sample_confidence(len(games)),
            evidence=stats,
            study=[
                "Only start a rated game when you are sure you have time to finish it; play unrated or daily games "
                "when you might be interrupted.",
                "Check your connection and battery before a rated session, and play on even in bad positions: "
                "opponents at club level often let you back in.",
            ],
            example_games=recent_urls(mine),
        ),
        stats,
    )


# Study suggestions per (length band, direction); the observation's title is neutral (see length_insights).
_LENGTH_TEXT: dict[tuple[str, str], tuple[str, list[str]]] = {
    ("short", "below"): (
        "short games",
        [
            "Replay your {k} shortest losses and find the move that lost material or let an attack through: short "
            "losses usually come from an opening trap or a one-move tactic.",
            "Learn the standard traps in the openings you play most, for both sides, so you neither fall into nor "
            "miss them.",
            "In the first 15 moves, check every capture and every check your opponent has before you move.",
        ],
    ),
    ("early", "below"): (
        "games decided between moves 21 and 30",
        [
            "Replay your {k} most recent losses of 21-30 moves and mark the first move after which you were "
            "clearly worse: most are early-middlegame tactics.",
            "Do 20 minutes of tactics puzzles a day, focusing on forks, pins and discovered attacks.",
            "When the opening ends, make a plan based on the pawn structure before starting operations.",
        ],
    ),
    ("middle", "below"): (
        "long middlegame battles",
        [
            "Replay your {k} most recent losses of 31-45 moves and note where the evaluation turned: a tactic, "
            "a bad plan, or the clock.",
            "Study annotated master games in the structures your openings lead to, so you know the plans.",
            "Use a blunder check on every move in complex middlegames: what does my move leave undefended?",
        ],
    ),
    ("long", "below"): (
        "long games",
        [
            "Work through the essential endgames: king-and-pawn opposition, the Lucena and Philidor positions, "
            "and basic rook endings.",
            "Replay your {k} most recent long losses from move 40 onwards and find where the endgame turned.",
            "Keep a clock reserve for the ending: aim to have at least a quarter of your time left at move 40.",
        ],
    ),
    ("very_long", "below"): (
        "very long games",
        [
            "Work through the essential endgames: king-and-pawn opposition, the Lucena and Philidor positions, "
            "and basic rook endings.",
            "Replay your {k} most recent very long losses and look for the moment concentration slipped: a "
            "hurried move, a missed defence, or the clock.",
            "Practise converting and holding endgames against an engine from positions taken from your own games.",
        ],
    ),
    ("short", "above"): (
        "short games",
        [
            "Keep studying opening traps and early tactics: they are winning you games quickly.",
            "Review your {k} most recent quick wins to see which patterns recur, and look for them in new openings.",
        ],
    ),
    ("early", "above"): (
        "games decided between moves 21 and 30",
        [
            "Your early-middlegame play is paying off: review your {k} most recent wins of this length to see which "
            "ideas keep working.",
            "Steer towards the openings that lead to these sharp middlegames.",
        ],
    ),
    ("middle", "above"): (
        "long middlegame battles",
        [
            "Review your {k} most recent wins of 31-45 moves and note the plans that worked.",
            "Choose openings that lead to rich middlegames where this strength shows.",
        ],
    ),
    ("long", "above"): (
        "long games",
        [
            "Your technique in long games is an asset: when in doubt, keep pieces on and play for a long game.",
            "Review your {k} most recent long wins to reinforce the endgame patterns that work for you.",
        ],
    ),
    ("very_long", "above"): (
        "very long games",
        [
            "Your stamina and endgame technique are assets: do not agree to early draws in playable positions.",
            "Review your {k} most recent very long wins to reinforce the endgame patterns that work for you.",
        ],
    ),
}


def length_insights(
    groups: dict[str, list[Game]], summaries: dict[str, ScoreSummary], th: Thresholds
) -> tuple[list[Insight], dict[str, Optional[float]]]:
    """Length bands that score differently from your games of other lengths: observations only.

    Each band is compared with all your other games (a difference test, so a rating that lags
    your strength doesn't make every band look strong or weak), BH-adjusted over the bands, at
    the strict level. Never a strength or weakness: game length follows from the result and
    from both players' resignation habits (see the module docstring).
    """
    labels = {key: label for key, label, _, _ in LENGTH_BUCKETS}
    tests: dict[str, tuple[MeanTest, ScoreSummary]] = {}
    for key in labels:
        s = summaries[key]
        rest = summarize([g for k2 in labels if k2 != key for g in groups[k2]])
        if s.n_rated >= th.min_length_games and rest.n_rated >= th.min_length_games:
            tests[key] = (difference_test(s.test, rest.test), rest)
    tested = list(tests)
    adjusted = dict(zip(tested, bh_adjust([tests[k][0].p_value for k in tested]), strict=True))
    out: list[Insight] = []
    for key in tested:
        s, (test, rest) = summaries[key], tests[key]
        n = min(s.n_rated, rest.n_rated)
        significant, confidence = significance(test, th.min_length_games, adjusted[key], th.strict_alpha, n)
        if abs(test.mean) < th.min_length_effect or not significant or test.mean * s.test.mean <= 0:
            continue
        direction = "below" if test.mean < 0 else "above"
        what, study = _LENGTH_TEXT[(key, direction)]
        group = groups[key]
        if direction == "below":
            losses = [g for g in group if g.outcome == "loss"]
            if key == "short":  # the quickest collapses are the clearest lessons
                losses.sort(key=lambda g: (g.plies, -g.end_time.timestamp()))
                examples = [g.url for g in losses if g.url][:MAX_EXAMPLES]
            else:
                examples = recent_urls(losses)
            k = min(MAX_EXAMPLES, len(losses)) or "few"
        else:
            examples = recent_urls(group, "win")
            k = min(MAX_EXAMPLES, s.wins) or "few"
        out.append(
            Insight(
                id=f"{KEY}.observation.length-{key.replace('_', '-')}",
                kind="observation",
                category="endings",
                title=f"Your {what} ({labels[key]} moves) score {direction} your games of other lengths",
                detail=(
                    f"In games lasting {labels[key]} moves you scored {pct(s.rated_score)} in {s.n_rated} games where "
                    f"{pct(s.expected)} was expected ({fmt_points(s.test.mean)} points per game); in your games of "
                    f"other lengths {fmt_points(rest.test.mean)}. The gap is {fmt_points(test.mean)} points per game "
                    f"(about {abs(score_to_elo_diff(test.mean, s.expected or 0.5)):.0f} Elo). How long a game lasts "
                    "depends on the result and on when you and your opponents resign, so this describes your games "
                    "rather than proving a strength or weakness."
                ),
                severity=severity_from_points(shrink_effect(test.mean, test.se)),
                confidence=confidence,
                evidence={
                    "bucket": labels[key],
                    "n": s.n,
                    "n_rated": s.n_rated,
                    "score": s.rated_score,
                    "expected": s.expected,
                    "delta": s.delta,
                    "rest_delta": rest.delta,
                    "gap": test.mean,
                    "p_value": test.p_value,
                    "p_adjusted": adjusted[key],
                },
                study=[a.format(k=k) for a in study],
                example_games=examples,
            )
        )
    return out, {labels[k]: p for k, p in adjusted.items()}


def draw_insight(by_class: dict[str, list[Game]], games: Sequence[Game], th: Thresholds) -> Optional[Insight]:
    rates = {
        tc: sum(1 for g in gs if g.outcome == "draw") / len(gs)
        for tc, gs in by_class.items()
        if len(gs) >= th.min_tc_games
    }
    if not rates:
        return None
    draws = [g for g in games if g.outcome == "draw"]
    overall = len(draws) / len(games)
    parts = ", ".join(f"{pct(r)} in {tc}" for tc, r in rates.items())
    detail = f"You drew {len(draws)} of {len(games)} games ({pct(overall)}): {parts}."
    if draws:
        common, count = Counter(draw_type(g) for g in draws).most_common(1)[0]
        label = dict(DRAW_TYPES)[common].lower()
        detail += f" The most common kind of draw is by {label} ({count} of {len(draws)})."
    return Insight(
        id=f"{KEY}.observation.draw-rate",
        kind="observation",
        category="endings",
        title=f"You draw {pct(overall)} of your games",
        detail=detail,
        severity=clamp(overall),
        confidence=sample_confidence(len(games)),
        evidence={"n": len(games), "draws": len(draws), "rate": overall, "by_time_class": rates},
        study=[
            "Replay your draws by repetition and agreement: were any of them in positions where you were better "
            "and could have played on?",
            "In equal endgames against lower-rated players, keep playing: most club players go wrong under "
            "sustained pressure.",
        ],
        example_games=recent_urls(draws),
    )


# --------------------------------------------------------------------------- module entry point
def _kpis(games: Sequence[Game], p: DecisiveProfile, abandoned: int) -> list[Kpi]:
    n = len(games)
    decisive = sum(1 for g in games if g.outcome != "draw")
    mated, mating = p.by_method["checkmate"]
    return [
        Kpi("Decisive games", decisive / n, "pct", hint=f"{decisive} of {n}"),
        Kpi("Losses by checkmate", p.loss_share("checkmate"), "pct", hint=f"{mated} of {p.losses} losses"),
        Kpi("Wins by checkmate", p.win_share("checkmate"), "pct", hint=f"{mating} of {p.wins} wins"),
        Kpi("Losses on time", p.loss_share("timeout"), "pct", hint=f"{p.by_method['timeout'][0]} games"),
        Kpi("Games you abandoned", abandoned, "int"),
        Kpi("Draw rate", (n - decisive) / n, "pct"),
    ]


def _summary(games: Sequence[Game], p: DecisiveProfile, insights: list[Insight], th: Thresholds) -> str:
    n = len(games)
    wins = sum(1 for g in games if g.outcome == "win")
    losses = sum(1 for g in games if g.outcome == "loss")
    text = (
        f"Of your {n} games, {wins} were wins, {n - wins - losses} draws and {losses} losses."
        if n > 1
        else f"Your only game was a {games[0].outcome}."
    )
    parts = []
    if p.losses:
        k, label = max(METHODS, key=lambda m: p.by_method[m[0]][0])
        parts.append(f"most losses end by {label.lower()} ({pct(p.loss_share(k))})")
    if p.wins:
        k, label = max(METHODS, key=lambda m: p.by_method[m[0]][1])
        parts.append(f"most wins by {label.lower()} ({pct(p.win_share(k))})")
    if parts:
        joined = " and ".join(parts)
        text += f" {joined[0].upper()}{joined[1:]}."
    if n < th.min_games:
        games_text = f"{n} game{'s' if n != 1 else ''}"
        return f"Not enough data yet: only {games_text}, so treat these numbers as a snapshot. {text}"
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        text += f" Key finding: {ranked[0].title}."
    return text


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Result mix, loss profile, abandoned games, score by game length and draw rates."""
    th = Thresholds.from_ctx(ctx)
    games = sorted(ctx.games, key=lambda g: g.end_time)
    if not games:
        return ModuleResult(
            key=KEY, title=TITLE, summary="Not enough data: there are no games to analyse yet.", stats={"n": 0}
        )

    by_class = _group_by_class(games)
    played = [g for g in games if g.plies >= MIN_PLIES]
    profile = decisive_profile(played)
    length_games = [g for g in played if g.termination != "abandoned"]
    groups: dict[str, list[Game]] = {key: [] for key, _, _, _ in LENGTH_BUCKETS}
    for g in length_games:
        groups[length_bucket(g.full_moves)].append(g)
    summaries = {key: summarize(gs) for key, gs in groups.items()}

    insights: list[Insight] = []
    mate, mate_stats = mate_insight(profile, played, th)
    abandoned, abandoned_stats = abandoned_insight(games, th)
    lengths, length_p = length_insights(groups, summaries, th)
    draws = draw_insight(by_class, games, th)
    insights += [i for i in (mate, abandoned) if i] + lengths + ([draws] if draws else [])
    if len(games) < th.min_games:
        insights = []

    tables = [
        mix_table(by_class, games),
        profile_table(profile, len(games) - len(played)),
        length_table(summaries, len(games) - len(length_games)),
        draw_table(by_class),
    ]
    charts = [mix_chart(by_class)]
    if any(s.delta is not None for s in summaries.values()):
        charts.append(length_chart(summaries))

    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(games, profile, insights, th),
        kpis=_kpis(games, profile, abandoned_stats["abandoned"]),
        tables=tables,
        charts=charts,
        insights=insights,
        stats={
            "n": len(games),
            "n_played": len(played),
            "result_mix": {
                tc: dict(zip((k for k, _ in MIX), mix_shares(gs), strict=True)) for tc, gs in by_class.items()
            },
            "losses_by": {k: profile.by_method[k][0] for k, _ in METHODS},
            "wins_by": {k: profile.by_method[k][1] for k, _ in METHODS},
            "mate_comparison": mate_stats,
            "abandoned": abandoned_stats,
            "by_length": {
                label: {
                    "n": summaries[key].n,
                    "n_rated": summaries[key].n_rated,
                    "score": summaries[key].rated_score,
                    "expected": summaries[key].expected,
                    "delta": summaries[key].delta,
                    "p_adjusted": length_p.get(label),
                }
                for key, label, _, _ in LENGTH_BUCKETS
            },
            "draw_rate": {tc: sum(1 for g in gs if g.outcome == "draw") / len(gs) for tc, gs in by_class.items()},
        },
    )
