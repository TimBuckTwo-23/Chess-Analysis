"""Openings: how your repertoire scores compared with what your rating predicts.

Only games from the standard start position with at least one full move (2 plies)
count; Chess960 / set-up positions and variants have no comparable opening.
Groups:

* as White: by your first move (1.e4, 1.d4, 1.c4, 1.Nf3, other) and by opening family;
* as Black: by your opponent's first move (vs 1.e4, vs 1.d4, vs other) and by family;
* individual lines (``Game.opening``) played at least a handful of times.

Each group's score-vs-expected is shrunk toward 0 (``stats.shrink``) before it is
ranked or charted. A family is only called a strength/weakness under the project-wide
rule (``stats.significance`` at ``stats.ALPHA``), and what is tested is how it compares
with *your other openings with the same colour* (a difference test of score minus
expected):

* not with the rating's expectation alone: a rating that lags your current strength
  (an improving or declining player) shifts every game, which would make every opening
  you play often look like a "strength" (or "weakness");
* not with your games of the other colour: how you do with Black as a whole is the
  results module's colour question, so a weak Black makes no Black opening a weakness,
  and White's first-move edge (whose exact size is uncertain) cancels out;
* the per-family tests are weighted-Benjamini-Hochberg adjusted
  (``stats.weighted_bh_adjust``, weights = games played) because every family is tested
  at once, and the openings you play most get most of the error budget. A family is
  only claimed when its own score against the rating points the same way (so of two
  openings that differ, only the one that is off your rating is named).

If one opening is almost all of your games with a colour, it cannot be told apart from
that colour, and it is not claimed here.

"You often lose quickly in X" compares, within the same family, how often your losses
are quick collapses with how often your *wins* are (your opponents' quick collapses in
the same kind of positions: a sharp opening produces short games for both sides), and
then with the same gap in your other openings (your resignation habits: a player who
resigns early has quick losses in every opening). Difference-in-differences of
proportions (``stats.proportion_gap_test``), weighted BH over families,
``stats.STRICT_ALPHA``.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..context import AnalysisContext
from ..models import Chart, Color, Game, Insight, InsightKind, Kpi, ModuleResult, Series, Table
from ..stats import (
    ALPHA,
    STRICT_ALPHA,
    MeanTest,
    ScoreSummary,
    clamp,
    difference_test,
    pct,
    proportion_gap_test,
    sample_confidence,
    score_to_elo_diff,
    severity_from_points,
    shrink,
    significance,
    summarize,
    weighted_bh_adjust,
)
from .results import MAX_EXAMPLES, fmt_points, recent_urls, slugify

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
    alpha: float = ALPHA  # opening families: a primary claim
    strict_alpha: float = STRICT_ALPHA  # quick losses in one opening
    min_effect: float = 0.06  # shrunk, colour-adjusted points per game
    prior_n: float = 10.0  # shrinkage strength (games at exactly the expected score)
    short_loss_moves: int = 25
    min_short_losses: int = 5
    min_decisive: int = 8  # losses and wins in a family before comparing how quickly each end
    short_loss_share: float = 0.30  # share of a family's losses that were short
    min_short_share_gap: float = 0.10  # ... minus the share of its wins that were short
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
    summary: ScoreSummary  # against the Elo expectation (tables, charts, KPIs)
    shrunk: Optional[float]
    short_losses: list[Game]
    short_wins: list[Game]

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

    @property
    def wins(self) -> list[Game]:
        return [g for g in self.games if g.outcome == "win"]


def is_short_decisive(g: Game, max_moves: int, outcome: str) -> bool:
    """``outcome`` ("win"/"loss") by mate or resignation within ``max_moves`` moves (ignoring 0-1 move abandons)."""
    return (
        g.outcome == outcome
        and g.termination in SHORT_LOSS_TERMINATIONS
        and 4 <= g.plies
        and g.full_moves <= max_moves
    )


def is_short_loss(g: Game, max_moves: int) -> bool:
    """Lost by mate or resignation within ``max_moves`` moves (ignoring 0-1 move abandons)."""
    return is_short_decisive(g, max_moves, "loss")


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
        short_wins=[g for g in games if is_short_decisive(g, th.short_loss_moves, "win")],
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


@dataclass
class FamilyTest:
    """One family compared with the player's other games of the same colour."""

    group: OpeningGroup
    rest: ScoreSummary  # the other games with that colour
    test: MeanTest  # difference_test(family, rest): points per game, both against the Elo expectation
    prior_n: float = 10.0  # Thresholds.prior_n
    p_adjusted: float = 1.0  # weighted BH over every family tested

    @property
    def effect(self) -> float:
        """The gap shrunk toward 0 by the family's sample size (what must clear ``min_effect``)."""
        return shrink(self.test.mean, self.group.summary.n_rated, 0.0, self.prior_n)


def family_tests(families: list[OpeningGroup], by_colour: dict[Color, list[Game]], th: Thresholds) -> list[FamilyTest]:
    """Each named family with enough rated games vs the rest of that colour's games, weighted-BH adjusted.

    When a colour's games are exactly two groups (two families, or one family and some unnamed
    games), both comparisons are the same test; it is kept once, for the group whose own score
    is further from its rating.
    """
    out: list[FamilyTest] = []
    for grp in families:
        if grp.label == UNNAMED or grp.summary.n_rated < th.min_family_games:
            continue
        ids = {id(g) for g in grp.games}
        rest = summarize([g for g in by_colour[grp.color] if id(g) not in ids])
        if rest.n_rated < th.min_family_games:
            continue
        out.append(FamilyTest(grp, rest, difference_test(grp.summary.test, rest.test), th.prior_n))
    for color in ("white", "black"):
        mine = [t for t in out if t.group.color == color]
        others = [f for f in families if f.color == color and f.summary.n_rated]
        if len(mine) == 2 and len(others) == 2:  # a mirrored pair: one test
            drop = min(mine, key=lambda t: abs(t.group.summary.test.mean))
            out.remove(drop)
    adjusted = weighted_bh_adjust([t.test.p_value for t in out], [t.group.summary.n_rated for t in out])
    for t, p in zip(out, adjusted, strict=True):
        t.p_adjusted = p
    return out


def claimed_families(games: Sequence[Game], th: Thresholds) -> list[tuple[Color, str, InsightKind]]:
    """(colour, family, kind) of every family this module calls a strength or weakness in ``games``.

    The results module uses it to tell a colour gap that comes from one opening apart from a
    colour-wide one (the same finding should not be reported twice).
    """
    by_colour, families = _families(games, th)
    tests = family_tests([f for fs in families.values() for f in fs.values()], by_colour, th)
    return [(t.group.color, t.group.label, kind) for t in tests if (kind := _claim_kind(t, th)) is not None]


def _claim_kind(t: FamilyTest, th: Thresholds) -> Optional[InsightKind]:
    """strength / weakness if ``t`` passes the claim rule, else None."""
    own = t.group.summary.test.mean  # the family against its own rating expectation
    significant, _ = significance(
        t.test, th.min_family_games, t.p_adjusted, alpha=th.alpha, n=min(t.group.summary.n_rated, t.rest.n_rated)
    )
    if not significant or abs(t.effect) < th.min_effect:
        return None
    # the family must itself be off its rating in the same direction, not merely differ from the rest
    if own * t.test.mean <= 0 or abs(own) < th.min_effect / 2:
        return None
    return "weakness" if t.test.mean < 0 else "strength"


def score_insights(
    families: list[OpeningGroup], by_colour: dict[Color, list[Game]], th: Thresholds
) -> tuple[list[Insight], dict[str, Any]]:
    """Strength/weakness per (colour, family): the family vs your other games with that colour.

    Weighted BH over every family tested (weights = rated games), at ``th.alpha``; the gap that
    must clear ``min_effect`` is shrunk toward 0 by the family's sample size.
    """
    out: list[Insight] = []
    stats: dict[str, Any] = {}
    for t in family_tests(families, by_colour, th):
        grp, s, rest, test = t.group, t.group.summary, t.rest, t.test
        n_eff = min(s.n_rated, rest.n_rated)
        significant, confidence = significance(test, th.min_family_games, t.p_adjusted, alpha=th.alpha, n=n_eff)
        kind = _claim_kind(t, th)
        stats[f"{grp.color}:{grp.label}"] = {
            "p_value": test.p_value,
            "p_adjusted": t.p_adjusted,
            "confidence": confidence,
            "significant": significant,
            "gap": test.mean,
            "rest_delta": rest.test.mean,
            "rest_n": rest.n_rated,
        }
        if kind is None:
            continue
        own = chosen_by(grp.label) == grp.color
        fam, colour = grp.label, _colour(grp.color)
        more = "more" if kind == "strength" else "less"
        if own:
            title = f"You score {more} with the {fam} than with your other {colour} openings"
        else:
            title = f"You score {more} against the {fam} than against other openings as {colour}"
        elo = abs(score_to_elo_diff(test.mean, s.expected if s.expected is not None else 0.5))
        detail = (
            f"You score {pct(s.rated_score)} in {s.n_rated} games where {pct(s.expected)} was expected "
            f"({fmt_points(s.test.mean)} points per game against your rating); in your other {rest.n_rated} games as "
            f"{colour}, {fmt_points(rest.test.mean)}. The gap is {fmt_points(test.mean)} points per game (about "
            f"{elo:.0f} Elo). It makes up {pct(grp.share)} of your games as {colour}."
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
                severity=severity_from_points(t.effect),
                confidence=confidence,
                evidence={
                    "color": grp.color,
                    "family": fam,
                    "n": s.n,
                    "n_rated": s.n_rated,
                    "score": s.rated_score,
                    "expected": s.expected,
                    "delta": s.test.mean,
                    "shrunk_delta": grp.shrunk,
                    "rest_delta": rest.test.mean,
                    "rest_n": rest.n_rated,
                    "gap": test.mean,
                    "effect": t.effect,
                    "half_width": s.half_width,
                    "p_value": test.p_value,
                    "p_adjusted": t.p_adjusted,
                    "share": grp.share,
                    "short_losses": len(grp.short_losses),
                },
                study=_score_study(grp, kind, own, th),
                example_games=recent_urls(grp.games, "loss" if kind == "weakness" else "win"),
            )
        )
    return out, stats


def early_loss_insights(
    families: list[OpeningGroup], eligible: Sequence[Game], weak_keys: set[tuple[str, str]], th: Thresholds
) -> tuple[list[Insight], dict[str, Any]]:
    """Families where your losses are quick collapses much more often than your wins are, beyond your usual gap.

    In the same family, the share of your losses that ended by mate or resignation within
    ``short_loss_moves`` moves minus the share of your wins that did (your opponents' quick
    collapses in the same openings: a sharp opening is short for both sides); minus the same
    gap over all your other games (your own resignation habits). The tested set depends only
    on how many losses and wins a family has; tests are weighted-BH adjusted (weights =
    decisive games) and a weakness needs ``th.strict_alpha``. Families that already carry a
    score weakness are skipped: that insight reports their short losses.
    """
    tested = [
        f
        for f in families
        if f.label != UNNAMED and len(f.losses) >= th.min_decisive and len(f.wins) >= th.min_decisive
    ]
    all_losses = sum(1 for g in eligible if g.outcome == "loss")
    all_wins = sum(1 for g in eligible if g.outcome == "win")
    all_short_losses = sum(1 for g in eligible if is_short_loss(g, th.short_loss_moves))
    all_short_wins = sum(1 for g in eligible if is_short_decisive(g, th.short_loss_moves, "win"))
    tests: list[MeanTest] = []
    for f in tested:
        n_l, n_s, n_w, n_ws = len(f.losses), len(f.short_losses), len(f.wins), len(f.short_wins)
        tests.append(
            proportion_gap_test(
                n_s, n_l, n_ws, n_w, all_short_losses - n_s, all_losses - n_l, all_short_wins - n_ws, all_wins - n_w
            )
        )
    adjusted = weighted_bh_adjust([t.p_value for t in tests], [len(f.losses) + len(f.wins) for f in tested])
    out: list[Insight] = []
    stats: dict[str, Any] = {}
    for f, t, p_adj in zip(tested, tests, adjusted, strict=True):
        n_l, n_s, n_w, n_ws = len(f.losses), len(f.short_losses), len(f.wins), len(f.short_wins)
        loss_share, win_share = n_s / n_l, n_ws / n_w
        rest_l, rest_w = all_losses - n_l, all_wins - n_w
        rest_gap = (
            (all_short_losses - n_s) / rest_l - (all_short_wins - n_ws) / rest_w if rest_l and rest_w else None
        )
        significant, confidence = significance(t, th.min_decisive, p_adj, alpha=th.strict_alpha, n=min(n_l, n_w))
        stats[f"{f.color}:{f.label}"] = {
            "short_loss_share": loss_share,
            "short_win_share": win_share,
            "rest_gap": rest_gap,
            "gap": t.mean,
            "p_value": t.p_value,
            "p_adjusted": p_adj,
            "significant": significant,
        }
        if (f.color, f.label) in weak_keys or not significant:
            continue
        if n_s < th.min_short_losses or loss_share < th.short_loss_share or t.mean < th.min_short_share_gap:
            continue  # quick losses are not a pattern worth a finding here
        rest_text = (
            f" In your other openings the two shares differ by {abs(rest_gap) * 100:.0f} points"
            f"{' (losses quicker)' if rest_gap > 0 else ' (wins quicker)' if rest_gap < 0 else ''}."
            if rest_gap is not None
            else ""
        )
        detail = (
            f"{n_s} of your {n_l} losses in the {f.label} as {_colour(f.color)} ({pct(loss_share)}) ended by checkmate "
            f"or resignation within {th.short_loss_moves} moves, while {n_ws} of your {n_w} wins in it "
            f"({pct(win_share)}) were over that quickly.{rest_text}"
        )
        shortest = sorted(f.short_losses, key=lambda g: (g.plies, -g.end_time.timestamp()))
        out.append(
            Insight(
                id=f"{KEY}.weakness.early-losses.{f.color}.{slugify(f.label)}",
                kind="weakness",
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
                    "wins": n_w,
                    "short_wins": n_ws,
                    "rest_gap": rest_gap,
                    "gap": t.mean,
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
    return out, stats


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
def _families(
    games: Sequence[Game], th: Thresholds
) -> tuple[dict[Color, list[Game]], dict[Color, dict[str, OpeningGroup]]]:
    """Eligible games by colour, and their opening families (unnamed games in one ``UNNAMED`` group)."""
    by_colour: dict[Color, list[Game]] = {"white": [], "black": []}
    for g in games:
        if is_eligible(g):
            by_colour[g.color].append(g)
    families = {
        color: {
            fam: build_group(fam, color, fgs, len(gs), th)
            for fam, fgs in _group_by(gs, lambda g: g.opening_family or UNNAMED).items()
        }
        for color, gs in by_colour.items()
    }
    return by_colour, families


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

    by_colour, families = _families(eligible, th)
    totals: dict[Color, int] = {c: len(gs) for c, gs in by_colour.items()}
    first_moves: dict[Color, dict[str, OpeningGroup]] = {
        color: {
            label: build_group(label, color, mgs, totals[color], th)
            for label, mgs in _group_by(gs, first_move_label).items()
        }
        for color, gs in by_colour.items()
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

    insights, test_stats = score_insights(all_families, by_colour, th)
    weak_keys = {(i.evidence["color"], i.evidence["family"]) for i in insights if i.kind == "weakness"}
    early, early_stats = early_loss_insights(all_families, eligible, weak_keys, th)
    insights += early
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
            "early_loss_tests": early_stats,
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
