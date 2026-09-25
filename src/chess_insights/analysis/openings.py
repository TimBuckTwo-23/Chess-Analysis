"""Openings: how your repertoire scores compared with what your rating predicts.

Only games from the standard start position with at least one full move (2 plies)
count; Chess960 / set-up positions and variants have no comparable opening.
Groups:

* as White: by your first move (1.e4, 1.d4, 1.c4, 1.Nf3, other) and by opening family;
* as Black: by your opponent's first move (vs 1.e4, vs 1.d4, vs other) and by family;
* individual lines (``Game.opening``) played at least a handful of times.

Each group's score-vs-expected is shrunk toward 0 (``stats.shrink``) before it is
ranked, charted or called a strength/weakness, and the per-family significance
tests are Benjamini-Hochberg adjusted because every family is tested at once.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..context import AnalysisContext
from ..models import Chart, Color, Game, Insight, InsightKind, Kpi, ModuleResult, Series, Table
from ..stats import (
    MeanTest,
    bh_adjust,
    clamp,
    combined_confidence,
    pct,
    sample_confidence,
    score_to_elo_diff,
    severity_from_points,
    two_proportion_test,
)
from .results import MAX_EXAMPLES, ScoreSummary, fmt_points, recent_urls, slugify, summarize, with_p_value

KEY = "openings"
TITLE = "Openings"

UNNAMED = "Unnamed opening"
WHITE_FIRST_MOVES = ("e4", "d4", "c4", "Nf3")
BLACK_FIRST_MOVES = ("e4", "d4")
# A short loss is a real opening/early-middlegame collapse, not a flag or an abandoned game.
SHORT_LOSS_TERMINATIONS = frozenset({"checkmate", "resignation"})
# Families named after Black's choice although the name doesn't end in "Defense"/"Countergambit".
_BLACK_NAMED = frozenset({"Englund Gambit", "Latvian Gambit", "Elephant Gambit", "Benko Gambit", "Budapest Gambit"})

GROUP_COLUMNS = [
    "Games",
    "Share",
    "W/D/L",
    "Score",
    "Expected",
    "Difference",
    "Adjusted",
    "± (95%)",
    "Avg moves",
    "Short losses",
]
GROUP_FORMATS = ["int", "pct", "text", "pct", "pct", "signed_pct", "signed_pct", "pct", "float1", "int"]


@dataclass(frozen=True)
class Thresholds:
    """Sample sizes and effect sizes; override with ``ctx.options["openings.<name>"]``."""

    min_games: int = 10  # below this the summary says there is not enough data
    min_family_games: int = 8  # rated games in a family (one colour) before it can be a strength/weakness
    min_line_games: int = 5  # games for the "Most played lines" table
    min_confidence: float = 0.4
    min_effect: float = 0.06  # shrunk points per game
    prior_n: float = 10.0  # shrinkage strength (games at exactly the expected score)
    short_loss_moves: int = 25
    min_short_losses: int = 5
    short_loss_share: float = 0.30  # share of a family's losses that were short
    min_short_share_gap: float = 0.10  # ... and how much higher than your other losses, for a weakness
    min_breadth_games: int = 20  # games of one colour with a named opening
    breadth_coverage: float = 0.80
    top_families: int = 12
    max_chart_bars: int = 20
    max_lines: int = 20

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


@dataclass
class OpeningGroup:
    """One set of games (a family, a first move or a line) for one colour."""

    label: str
    color: Color
    games: list[Game]
    colour_total: int  # eligible games with this colour (denominator of ``share``)
    summary: ScoreSummary
    shrunk: Optional[float]
    short_losses: list[Game]

    @property
    def n(self) -> int:
        return self.summary.n

    @property
    def share(self) -> Optional[float]:
        return self.n / self.colour_total if self.colour_total else None

    @property
    def avg_moves(self) -> Optional[float]:
        return sum(g.full_moves for g in self.games) / self.n if self.n else None

    @property
    def losses(self) -> list[Game]:
        return [g for g in self.games if g.outcome == "loss"]


def is_short_loss(g: Game, max_moves: int) -> bool:
    """Lost by mate or resignation within ``max_moves`` moves (ignoring 0-1 move abandons)."""
    return (
        g.outcome == "loss" and g.termination in SHORT_LOSS_TERMINATIONS and 4 <= g.plies and g.full_moves <= max_moves
    )


def build_group(label: str, color: Color, games: Sequence[Game], colour_total: int, th: Thresholds) -> OpeningGroup:
    s = summarize(games)
    return OpeningGroup(
        label=label,
        color=color,
        games=list(games),
        colour_total=colour_total,
        summary=s,
        shrunk=s.shrunk(th.prior_n),
        short_losses=[g for g in games if is_short_loss(g, th.short_loss_moves)],
    )


def is_eligible(g: Game) -> bool:
    """Standard chess from the normal start position with at least one full move."""
    return g.rules == "chess" and not g.initial_fen and g.plies >= 2


def first_move_label(g: Game) -> str:
    move = g.moves_san[0]
    if g.color == "white":
        return f"1.{move}" if move in WHITE_FIRST_MOVES else "Other first moves"
    return f"vs 1.{move}" if move in BLACK_FIRST_MOVES else "vs other"


FIRST_MOVE_ORDER = {
    "white": [f"1.{m}" for m in WHITE_FIRST_MOVES] + ["Other first moves"],
    "black": [f"vs 1.{m}" for m in BLACK_FIRST_MOVES] + ["vs other"],
}


def chosen_by(family: str) -> Color:
    """Which side's choice names the opening: a Defense is Black's, a Game/Opening/Attack/System White's."""
    last = family.split()[-1] if family.split() else ""
    return "black" if last in ("Defense", "Countergambit") or family in _BLACK_NAMED else "white"


def _group_by(games: Sequence[Game], key: Any) -> dict[str, list[Game]]:
    out: dict[str, list[Game]] = {}
    for g in games:
        out.setdefault(key(g), []).append(g)
    return out


def _sub_lines(games: Sequence[Game], family: str) -> list[str]:
    """Most common variations inside a family ('Advance Variation'), most played first."""
    counts = Counter(
        g.opening.split(": ", 1)[1]
        for g in games
        if g.opening and g.opening.startswith(f"{family}: ") and ": " in g.opening
    )
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _colour(c: Color) -> str:
    return c.capitalize()


# --------------------------------------------------------------------------- tables & chart
def _group_cells(grp: OpeningGroup) -> list[Any]:
    s = grp.summary
    return [
        s.n,
        grp.share,
        s.wdl,
        s.score,
        s.expected,
        s.delta,
        grp.shrunk,
        s.half_width,
        grp.avg_moves,
        len(grp.short_losses),
    ]


def family_table(
    color: Color, families: dict[str, OpeningGroup], colour_total: int, th: Thresholds, note: str
) -> Table:
    named = sorted((f for f in families.values() if f.label != UNNAMED), key=lambda f: (-f.n, f.label))
    top, more = named[: th.top_families], named[th.top_families :]
    rows = [[f.label, *_group_cells(f)] for f in top]
    rest = more + ([families[UNNAMED]] if UNNAMED in families else [])
    if rest:
        other = build_group("other", color, [g for f in rest for g in f.games], colour_total, th)
        if more:
            plural = "s" if len(more) != 1 else ""
            label = f"Other ({len(more)} more opening{plural}{' + unnamed' if UNNAMED in families else ''})"
        else:
            label = UNNAMED
        rows.append([label, *_group_cells(other)])
    return Table(
        title=f"As {_colour(color)}",
        columns=["Opening", *GROUP_COLUMNS],
        rows=rows,
        formats=["text", *GROUP_FORMATS],
        note=note,
    )


def first_move_table(groups: dict[Color, dict[str, OpeningGroup]]) -> Table:
    rows = []
    for color in ("white", "black"):
        for label in FIRST_MOVE_ORDER[color]:
            grp = groups[color].get(label)
            if grp:
                rows.append([_colour(color), label, *_group_cells(grp)])
    return Table(
        title="By first move",
        columns=["Colour", "First move", *GROUP_COLUMNS],
        rows=rows,
        formats=["text", "text", *GROUP_FORMATS],
        note="As White: your first move. As Black: your opponent's first move.",
    )


def lines_table(games: Sequence[Game], totals: dict[Color, int], th: Thresholds) -> Optional[Table]:
    lines = [
        build_group(opening, color, gs, totals[color], th)
        for (color, opening), gs in _group_by([g for g in games if g.opening], lambda g: (g.color, g.opening)).items()
        if len(gs) >= th.min_line_games
    ]
    if not lines:
        return None
    lines.sort(key=lambda grp: (-grp.n, grp.label, grp.color))
    return Table(
        title="Most played lines",
        columns=["Line", "Colour", "Games", "W/D/L", "Score", "Expected", "Difference", "Adjusted"],
        rows=[
            [
                grp.label,
                _colour(grp.color),
                grp.n,
                grp.summary.wdl,
                grp.summary.score,
                grp.summary.expected,
                grp.summary.delta,
                grp.shrunk,
            ]
            for grp in lines[: th.max_lines]
        ],
        formats=["text", "text", "int", "text", "pct", "pct", "signed_pct", "signed_pct"],
        note=f"Lines played at least {th.min_line_games} times.",
    )


def family_chart(families: list[OpeningGroup], th: Thresholds) -> Optional[Chart]:
    bars = [
        f for f in families if f.label != UNNAMED and f.summary.n_rated >= th.min_family_games and f.shrunk is not None
    ]
    if not bars:
        return None
    bars = sorted(bars, key=lambda f: (-f.n, f.label))[: th.max_chart_bars]
    bars.sort(key=lambda f: (f.shrunk, f.label))  # most costly first
    return Chart(
        kind="hbar",
        title="Score vs expected by opening",
        labels=[f"{f.label} ({_colour(f.color)})" for f in bars],
        series=[Series("Score − expected (adjusted)", [f.shrunk for f in bars])],
        value_format="signed_pct",
        note=(
            f"Openings with {th.min_family_games}+ rated games. Adjusted: each result is pulled toward 0 as if you had "
            f"played {th.prior_n:.0f} extra games scoring exactly as expected, so small samples don't dominate."
        ),
        reference=0.0,
    )


# --------------------------------------------------------------------------- insights
def _score_study(grp: OpeningGroup, kind: InsightKind, own: bool, th: Thresholds) -> list[str]:
    fam, colour = grp.label, _colour(grp.color)
    subs = _sub_lines(grp.games, fam)
    if kind == "weakness":
        k = min(MAX_EXAMPLES, len(grp.losses))
        study = [
            f"Replay your {k} most recent losses in the {fam} as {colour} and mark the move where you left known "
            "theory or first got a worse position."
            if k
            else f"Replay your recent games in the {fam} as {colour} and mark where your position started to slip."
        ]
        if not own:
            study.append(
                f"Pick one solid, well-trodden answer to the {fam} and learn its first 8-10 moves and the main plans."
            )
        if subs:
            lines = f"the {subs[0]}" + (f" and the {subs[1]}" if len(subs) > 1 else "")
            study.append(f"Most of these games went into {lines}: learn the main ideas and typical plans there first.")
        else:
            study.append(
                f"Study 3-5 annotated master games in the {fam} to learn its plans and pawn structures, "
                "not just the moves."
            )
        if len(grp.short_losses) >= 3:
            study.append(
                f"{len(grp.short_losses)} of these losses were over within {th.short_loss_moves} moves: learn the "
                f"typical traps for your side of the {fam}."
            )
        elif own and grp.color == "black":
            first = Counter(g.moves_san[0] for g in grp.games).most_common(1)[0][0]
            study.append(f"If it still isn't working after another 20 games, try a different reply to 1.{first}.")
        elif own:
            study.append(
                "If it still isn't working after another 20 games, switch to a system whose plans you "
                "understand better."
            )
        return study[:4]
    k = min(MAX_EXAMPLES, grp.summary.wins)
    if own:
        study = [
            f"Keep the {fam} in your repertoire: replay your {k} most recent wins in it to reinforce the plans "
            "that work."
        ]
        if len(subs) > 1:
            study.append(
                f"Deepen it: you meet the {subs[-1]} less often, so learn it well enough that opponents can't "
                "steer you out of your comfort zone."
            )
        study.append(f"Build the rest of your {colour} repertoire around similar pawn structures and plans.")
    else:
        study = [
            f"You handle the {fam} well as {colour}: replay your {k} most recent wins against it to note what works.",
            "Keep your current answer to it, and aim for the same kind of positions against other openings.",
        ]
    return study


def score_insights(families: list[OpeningGroup], th: Thresholds) -> tuple[list[Insight], dict[str, Any]]:
    """Strength/weakness per (colour, family) on the shrunk delta, BH-adjusted across all families tested."""
    tested = [f for f in families if f.label != UNNAMED and f.summary.n_rated >= th.min_family_games]
    adjusted = bh_adjust([f.summary.test.p_value for f in tested])
    out: list[Insight] = []
    stats: dict[str, Any] = {}
    for grp, p_adj in zip(tested, adjusted, strict=True):
        s = grp.summary
        confidence = combined_confidence(with_p_value(s.test, p_adj), th.min_family_games)
        stats[f"{grp.color}:{grp.label}"] = {"p_value": s.test.p_value, "p_adjusted": p_adj, "confidence": confidence}
        shrunk = grp.shrunk or 0.0
        if abs(shrunk) < th.min_effect or confidence < th.min_confidence:
            continue
        kind: InsightKind = "weakness" if shrunk < 0 else "strength"
        own = chosen_by(grp.label) == grp.color
        fam, colour = grp.label, _colour(grp.color)
        if own:
            title = f"The {fam} is {'costing you points' if kind == 'weakness' else 'working for you'} as {colour}"
        else:
            title = f"You {'struggle' if kind == 'weakness' else 'do well'} against the {fam} as {colour}"
        elo = abs(score_to_elo_diff(s.test.mean, s.expected if s.expected is not None else 0.5))
        detail = (
            f"You score {pct(s.rated_score)} in {s.n_rated} games where {pct(s.expected)} was expected "
            f"({fmt_points(s.test.mean)} points per game, like playing about {elo:.0f} Elo "
            f"{'below' if s.test.mean < 0 else 'above'} your rating). "
            f"It makes up {pct(grp.share)} of your games as {colour}."
        )
        if kind == "weakness" and grp.short_losses:
            detail += (
                f" {len(grp.short_losses)} of your {len(grp.losses)} losses in it were over within "
                f"{th.short_loss_moves} moves."
            )
        out.append(
            Insight(
                id=f"{KEY}.{kind}.{grp.color}.{slugify(fam)}",
                kind=kind,
                category="openings",
                title=title,
                detail=detail,
                severity=severity_from_points(shrunk),
                confidence=confidence,
                evidence={
                    "color": grp.color,
                    "family": fam,
                    "n": s.n,
                    "n_rated": s.n_rated,
                    "score": s.rated_score,
                    "expected": s.expected,
                    "delta": s.test.mean,
                    "shrunk_delta": shrunk,
                    "half_width": s.half_width,
                    "p_value": s.test.p_value,
                    "p_adjusted": p_adj,
                    "share": grp.share,
                    "short_losses": len(grp.short_losses),
                },
                study=_score_study(grp, kind, own, th),
                example_games=recent_urls(grp.games, "loss" if kind == "weakness" else "win"),
            )
        )
    return out, stats


def early_loss_insights(
    families: list[OpeningGroup], all_games: Sequence[Game], weak_keys: set[tuple[str, str]], th: Thresholds
) -> list[Insight]:
    """Families where many losses were quick collapses, more often than in your other losses.

    Families that already carry a score weakness are skipped: that insight reports their short losses.
    """
    candidates = [
        f
        for f in families
        if f.label != UNNAMED
        and (f.color, f.label) not in weak_keys
        and len(f.short_losses) >= th.min_short_losses
        and f.losses
        and len(f.short_losses) / len(f.losses) >= th.short_loss_share
    ]
    if not candidates:
        return []
    all_losses = [g for g in all_games if g.outcome == "loss"]
    all_short = sum(1 for g in all_losses if is_short_loss(g, th.short_loss_moves))
    tests: list[MeanTest] = []
    for f in candidates:
        n_l, n_s = len(f.losses), len(f.short_losses)
        other_l, other_s = len(all_losses) - n_l, all_short - n_s
        t = two_proportion_test(n_s, n_l, other_s, other_l)
        tests.append(MeanTest(n_l, t.mean, t.se, t.z, t.p_value))  # n = this family's losses (sample-size factor)
    adjusted = bh_adjust([t.p_value for t in tests])
    out: list[Insight] = []
    for f, t, p_adj in zip(candidates, tests, adjusted, strict=True):
        n_l, n_s = len(f.losses), len(f.short_losses)
        other_l = len(all_losses) - n_l
        other_share = (all_short - n_s) / other_l if other_l else None
        if other_share is not None and n_s / n_l <= other_share:
            continue  # quick losses are no more common here than in the rest of your games
        confidence = combined_confidence(with_p_value(t, p_adj), th.min_short_losses)  # full weight at 3x
        weak = other_share is not None and t.mean >= th.min_short_share_gap and confidence >= th.min_confidence
        kind: InsightKind = "weakness" if weak else "observation"
        detail = (
            f"{n_s} of your {n_l} losses in the {f.label} as {_colour(f.color)} ({pct(n_s / n_l)}) ended by checkmate "
            f"or resignation within {th.short_loss_moves} moves"
            + (f", compared with {pct(other_share)} of your other losses." if other_share is not None else ".")
        )
        shortest = sorted(f.short_losses, key=lambda g: (g.plies, -g.end_time.timestamp()))
        out.append(
            Insight(
                id=f"{KEY}.{kind}.early-losses.{f.color}.{slugify(f.label)}",
                kind=kind,
                category="openings",
                title=f"You often lose quickly in the {f.label} as {_colour(f.color)}",
                detail=detail,
                severity=clamp(max(t.mean, 0.0) / 0.3),
                confidence=confidence,
                evidence={
                    "color": f.color,
                    "family": f.label,
                    "losses": n_l,
                    "short_losses": n_s,
                    "other_short_share": other_share,
                    "p_value": t.p_value,
                    "p_adjusted": p_adj,
                },
                study=[
                    f"Replay your {min(MAX_EXAMPLES, n_s)} shortest losses in the {f.label} and find the move where it "
                    "went wrong; it is usually a tactic in the first 15 moves.",
                    f"Learn the common traps and tactical tricks in the {f.label} for your side.",
                    "Before every opening move, ask what your opponent's last move attacks and what it threatens next.",
                ],
                example_games=[g.url for g in shortest if g.url][:MAX_EXAMPLES],
            )
        )
    return out


def breadth_insight(color: Color, games: Sequence[Game], th: Thresholds) -> tuple[Optional[Insight], Optional[int]]:
    """How many families make up ``breadth_coverage`` of one colour's games with a named opening."""
    counts = Counter(g.opening_family for g in games if g.opening_family)
    total = sum(counts.values())
    if total < th.min_breadth_games:
        return None, None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    covered, k = 0, 0
    for _, n in ranked:
        covered += n
        k += 1
        if covered >= th.breadth_coverage * total:
            break
    colour, cov = _colour(color), f"{th.breadth_coverage:.0%}"
    top = ", ".join(f"{name} ({pct(n / total)})" for name, n in ranked[:3])
    if k <= 3:
        title = (
            f"As {colour} you keep a compact repertoire: {k} opening{'s' if k != 1 else ''} cover {cov} of your games"
        )
        study = [
            "A compact repertoire is efficient: make sure you know your main lines 8-10 moves deep and their "
            "typical plans.",
            "Prepare one reliable answer to the rarer replies you meet so they don't catch you out.",
        ]
        severity = 0.1
    else:
        very = "very " if k >= 8 else ""
        title = f"Your {colour} repertoire is {very}broad: it takes {k} openings to cover {cov} of your games"
        if color == "black":
            narrow = (
                "Narrow it down: choose one reply to 1.e4 and one to 1.d4 and play only those for your next "
                "50 games as Black."
            )
        else:
            narrow = (
                "Narrow it down: choose one first move and one system against each main reply, and play only "
                "those for your next 50 games as White."
            )
        study = [narrow, "Keep a small repertoire file for those lines and review it once a week."]
        severity = 0.3 if k >= 8 else 0.15
    return (
        Insight(
            id=f"{KEY}.observation.breadth-{color}",
            kind="observation",
            category="openings",
            title=title,
            detail=f"{total} games as {colour} with a named opening; most played: {top}.",
            severity=severity,
            confidence=sample_confidence(total),
            evidence={"color": color, "games": total, "families": len(counts), "families_for_coverage": k},
            study=study,
        ),
        k,
    )


# --------------------------------------------------------------------------- module entry point
def _most_played(families: dict[str, OpeningGroup]) -> Optional[OpeningGroup]:
    named = [f for f in families.values() if f.label != UNNAMED]
    return max(named, key=lambda f: (f.n, f.label)) if named else None


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Repertoire performance by first move, opening family and line, for each colour."""
    th = Thresholds.from_ctx(ctx)
    games = sorted(ctx.games, key=lambda g: g.end_time)
    eligible = [g for g in games if is_eligible(g)]
    n_custom = sum(1 for g in games if g.rules != "chess" or g.initial_fen)
    n_short = sum(1 for g in games if g.rules == "chess" and not g.initial_fen and g.plies < 2)
    excluded_note = ""
    if n_custom or n_short:
        parts = []
        if n_custom:
            parts.append(f"{n_custom} started from a custom position or were a variant")
        if n_short:
            parts.append(f"{n_short} had fewer than 2 moves (plies)")
        excluded_note = f"{n_custom + n_short} game(s) left out: {' and '.join(parts)}."
    base_stats: dict[str, Any] = {"n": len(eligible), "excluded_custom": n_custom, "excluded_short": n_short}
    if not eligible:
        summary = (
            "Not enough data: there are no standard-chess games from the normal start position to analyse openings."
        )
        if excluded_note:
            summary += f" {excluded_note}"
        return ModuleResult(key=KEY, title=TITLE, summary=summary, stats=base_stats)

    by_colour: dict[Color, list[Game]] = {"white": [], "black": []}
    for g in eligible:
        by_colour[g.color].append(g)
    totals: dict[Color, int] = {c: len(gs) for c, gs in by_colour.items()}
    families: dict[Color, dict[str, OpeningGroup]] = {}
    first_moves: dict[Color, dict[str, OpeningGroup]] = {}
    for color, gs in by_colour.items():
        families[color] = {
            fam: build_group(fam, color, fgs, totals[color], th)
            for fam, fgs in _group_by(gs, lambda g: g.opening_family or UNNAMED).items()
        }
        first_moves[color] = {
            label: build_group(label, color, mgs, totals[color], th)
            for label, mgs in _group_by(gs, first_move_label).items()
        }
    all_families = [f for color in ("white", "black") for f in families[color].values()]

    table_note = (
        "Expected: the Elo expected score. Difference: score minus expected, per game. Adjusted: the difference pulled "
        f"toward 0 as if you had played {th.prior_n:.0f} extra games scoring exactly as expected. ±: 95% margin of "
        f"the difference. Short losses: lost by mate or resignation within {th.short_loss_moves} moves."
    )
    if excluded_note:
        table_note += f" {excluded_note}"
    tables = [
        family_table(color, families[color], totals[color], th, table_note)
        for color in ("white", "black")
        if totals[color]
    ]
    tables.append(first_move_table(first_moves))
    lines = lines_table(eligible, totals, th)
    if lines:
        tables.append(lines)
    chart = family_chart(all_families, th)

    insights, test_stats = score_insights(all_families, th)
    weak_keys = {(i.evidence["color"], i.evidence["family"]) for i in insights if i.kind == "weakness"}
    insights += early_loss_insights(all_families, eligible, weak_keys, th)
    breadth: dict[str, Optional[int]] = {}
    for color in ("white", "black"):
        ins, k = breadth_insight(color, by_colour[color], th)
        breadth[color] = k
        if ins:
            insights.append(ins)
    if len(eligible) < th.min_games:
        insights = []

    # KPIs
    kpis = [
        Kpi("Games analysed", len(eligible), "int", hint=excluded_note or "standard start position"),
        Kpi(
            "Openings played",
            len({g.opening_family for g in eligible if g.opening_family}),
            "int",
            hint="opening families",
        ),
    ]
    for color in ("white", "black"):
        top = _most_played(families[color])
        if top:
            kpis.append(
                Kpi(f"Most played as {_colour(color)}", top.label, "text", hint=f"{top.n} games, {pct(top.share)}")
            )
    rankable = [
        f
        for f in all_families
        if f.label != UNNAMED and f.summary.n_rated >= th.min_family_games and f.shrunk is not None
    ]
    if rankable:
        best = max(rankable, key=lambda f: (f.shrunk, f.n))
        worst = min(rankable, key=lambda f: (f.shrunk, -f.n))
        if (best.shrunk or 0) > 0:
            kpis.append(
                Kpi(
                    "Best opening",
                    f"{best.label} ({_colour(best.color)})",
                    "text",
                    hint=f"{fmt_points(best.shrunk or 0)} points per game (adjusted)",
                )
            )
        if (worst.shrunk or 0) < 0:
            kpis.append(
                Kpi(
                    "Most costly opening",
                    f"{worst.label} ({_colour(worst.color)})",
                    "text",
                    hint=f"{fmt_points(worst.shrunk or 0)} points per game (adjusted)",
                )
            )

    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(eligible, families, insights, th),
        kpis=kpis,
        tables=tables,
        charts=[chart] if chart else [],
        insights=insights,
        stats={
            **base_stats,
            "families": {
                f"{f.color}:{f.label}": _group_stats(f)
                for f in sorted(all_families, key=lambda f: (f.color, -f.n, f.label))
            },
            "first_moves": {
                f"{c}:{label}": _group_stats(grp) for c in ("white", "black") for label, grp in first_moves[c].items()
            },
            "family_tests": test_stats,
            "breadth": breadth,
        },
    )


def _group_stats(grp: OpeningGroup) -> dict[str, Any]:
    s = grp.summary
    return {
        "n": s.n,
        "wins": s.wins,
        "draws": s.draws,
        "losses": s.losses,
        "score": s.score,
        "n_rated": s.n_rated,
        "expected": s.expected,
        "delta": s.delta,
        "shrunk_delta": grp.shrunk,
        "half_width": s.half_width,
        "share": grp.share,
        "avg_moves": grp.avg_moves,
        "short_losses": len(grp.short_losses),
    }


def _summary(
    eligible: Sequence[Game], families: dict[Color, dict[str, OpeningGroup]], insights: list[Insight], th: Thresholds
) -> str:
    n = len(eligible)
    if n < th.min_games:
        games = "game" if n == 1 else "games"
        return f"Not enough data yet: only {n} {games} with a normal start, too few to judge any opening."
    parts = []
    for color in ("white", "black"):
        top = _most_played(families[color])
        if top:
            parts.append(f"as {_colour(color)} the {top.label} ({pct(top.share)} of those games)")
    text = f"{n} games analysed" + (f"; your most common openings are {' and '.join(parts)}." if parts else ".")
    weak = sorted((i for i in insights if i.kind == "weakness"), key=lambda i: -i.priority)
    strong = sorted((i for i in insights if i.kind == "strength"), key=lambda i: -i.priority)
    if weak:
        text += f" Biggest opening problem: {weak[0].title}."
    if strong:
        text += f" Best: {strong[0].title}."
    if not weak and not strong:
        text += " No opening stands out clearly from your overall level yet."
    return text
