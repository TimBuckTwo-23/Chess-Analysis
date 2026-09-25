"""Results & rating: how you score compared with what your rating predicts.

Every comparison is against the Elo expected score of each game, never a raw 50%:

* overall and per time control (each time control has its own rating pool, so each
  row is judged against its own expectation; a time control is only called your
  best or weakest when it differs significantly from your other time controls);
* White vs Black, allowing for the edge White normally has;
* opponent-strength buckets, judged against the rating-noise-attenuated
  expectation (``stats.attenuated_expected``); a claim must hold for every
  attenuation in ``stats.ATTENUATION_RANGE``, i.e. against both the attenuated and
  the plain Elo expectation;
* the rating history, the recent rating trend (an observation: a rating random walk
  is not evidence of a strength) and chess.com Game Review accuracies.

Claims follow the project-wide rule ``stats.significance``: colour at ``stats.ALPHA``,
opponent strength and time controls (exploratory scans) at ``stats.STRICT_ALPHA``.

``summarize`` / ``ScoreSummary`` / ``difference_test`` / ``with_p_value`` now live in
``stats`` and are re-exported here, with the small formatting helpers reused by the
other analysis modules.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Sequence

from ..context import AnalysisContext
from ..models import TIME_CLASSES, Chart, Game, Insight, InsightKind, Kpi, ModuleResult, Series, Table
from ..stats import (  # noqa: F401  (ScoreSummary, summarize, difference_test, with_p_value: re-exported)
    ALPHA,
    ATTENUATION_RANGE,
    STRICT_ALPHA,
    WHITE_EDGE,
    MeanTest,
    ScoreSummary,
    bh_adjust,
    clamp,
    difference_test,
    pct,
    sample_confidence,
    score_to_elo_diff,
    severity_from_points,
    significance,
    summarize,
    with_p_value,
)

KEY = "results"
TITLE = "Results & rating"

MINUS = "−"
MAX_EXAMPLES = 5
TREND_DAYS = 90
TREND_FLAT = 15.0  # rating points over TREND_DAYS below which the rating counts as stable
MAX_RATING_KPIS = 3

# (key, label) of the opponent-strength buckets, by opponent rating minus yours.
OPPONENT_BUCKETS: tuple[tuple[str, str], ...] = (
    ("much_lower", f"much lower (< {MINUS}150)"),
    ("lower", f"lower ({MINUS}150 to {MINUS}50)"),
    ("similar", "similar (±50)"),
    ("higher", "higher (+50 to +150)"),
    ("much_higher", "much higher (> +150)"),
)


@dataclass(frozen=True)
class Thresholds:
    """Minimum sample sizes, effect sizes and significance levels; override with ``ctx.options["results.<name>"]``."""

    min_games: int = 10  # below this the summary says there is not enough data
    alpha: float = ALPHA  # colour: a primary claim
    strict_alpha: float = STRICT_ALPHA  # opponent strength, time controls: exploratory scans
    min_effect: float = 0.06  # points per game (about 40 Elo near 50%)
    min_color_games: int = 20  # rated games per colour
    min_bucket_games: int = 20  # rated games in an opponent-strength group
    min_tc_games: int = 25  # rated games per time control for the time-control comparison
    min_chart_games: int = 10  # games for a time control to get a rating-history line / delta bar
    min_trend_games: int = 15  # rated games in the last TREND_DAYS for a trend
    min_accuracy_games: int = 20  # games with a chess.com accuracy

    @classmethod
    def from_ctx(cls, ctx: AnalysisContext) -> "Thresholds":
        return cls(**{f.name: ctx.opt(f"{KEY}.{f.name}", f.default) for f in dataclasses.fields(cls)})


# --------------------------------------------------------------------------- shared helpers
def fmt_points(x: float) -> str:
    """'+0.07' / '−0.12' points per game (real minus sign)."""
    if round(x, 2) == 0:
        return "0.00"
    return f"{x:+.2f}".replace("-", MINUS)


def fmt_signed(x: float) -> str:
    """'+45' / '−12' (whole number, real minus sign)."""
    return f"{round(x):+d}".replace("-", MINUS) if round(x) else "0"


def slugify(text: str) -> str:
    """'King's Indian Defense' -> 'kings-indian-defense'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-") or "unknown"


def recent_urls(games: Iterable[Game], outcome: Optional[str] = None, limit: int = MAX_EXAMPLES) -> list[str]:
    """URLs of the most recent games (optionally of one outcome), newest first."""
    picked = sorted(
        (g for g in games if outcome is None or g.outcome == outcome), key=lambda g: g.end_time, reverse=True
    )
    return [g.url for g in picked if g.url][:limit]


# --------------------------------------------------------------------------- rating pools
def pool_of(g: Game) -> str:
    """Rating pool label: chess.com keeps one rating per time class (and per variant)."""
    label = (g.time_class or "unknown").capitalize()
    return label if g.rules == "chess" else f"{label} ({g.rules})"


def _pool_sort_key(g: Game) -> tuple[int, int, str]:
    rank = TIME_CLASSES.index(g.time_class) if g.time_class in TIME_CLASSES else len(TIME_CLASSES)
    return (rank, 0 if g.rules == "chess" else 1, pool_of(g))


def group_by_pool(games: Sequence[Game]) -> dict[str, list[Game]]:
    """Games per rating pool, pools in bullet/blitz/rapid/daily order, games in end-time order."""
    keys: dict[str, tuple[int, int, str]] = {}
    groups: dict[str, list[Game]] = {}
    for g in sorted(games, key=lambda g: g.end_time):
        pool = pool_of(g)
        keys.setdefault(pool, _pool_sort_key(g))
        groups.setdefault(pool, []).append(g)
    return {pool: groups[pool] for pool in sorted(groups, key=lambda p: keys[p])}


def rating_games(games: Sequence[Game]) -> list[Game]:
    """Games with a ``my_rating`` to chart: the rated ones, or every game with a rating if none is rated."""
    rated = [g for g in games if g.rated and g.my_rating is not None]
    return rated or [g for g in games if g.my_rating is not None]


@dataclass
class RatingHistory:
    n: int
    start: Optional[int]  # rating before the first game (post-game rating if unknown)
    current: Optional[int]  # rating after the last game
    peak: Optional[int]
    peak_time: Optional[datetime]

    @property
    def change(self) -> Optional[int]:
        return self.current - self.start if self.current is not None and self.start is not None else None


def rating_history(games: Sequence[Game]) -> RatingHistory:
    rg = rating_games(games)
    if not rg:
        return RatingHistory(0, None, None, None, None)
    first = rg[0]
    start = first.my_rating_before if first.rated and first.my_rating_before is not None else first.my_rating
    peak_game = max(rg, key=lambda g: (g.my_rating, g.end_time))
    return RatingHistory(len(rg), start, rg[-1].my_rating, peak_game.my_rating, peak_game.end_time)


def opponent_bucket(rating_diff: Optional[float]) -> Optional[str]:
    """Bucket key for opponent-minus-my rating; -150 and +150 fall in the inner buckets, +-50 in 'similar'."""
    if rating_diff is None or (isinstance(rating_diff, float) and math.isnan(rating_diff)):
        return None
    if rating_diff < -150:
        return "much_lower"
    if rating_diff < -50:
        return "lower"
    if rating_diff <= 50:
        return "similar"
    if rating_diff <= 150:
        return "higher"
    return "much_higher"


@dataclass
class Trend:
    pool: str
    n: int
    change: float  # least-squares slope x TREND_DAYS, rating points
    first: int
    last: int
    test: MeanTest  # score - expected over the window
    games: list[Game]


def rating_trend(
    pool: str, games: Sequence[Game], end: datetime, min_games: int, days: int = TREND_DAYS
) -> Optional[Trend]:
    """Least-squares rating slope over the ``days`` before ``end``; None with too few rated games."""
    since = end - timedelta(days=days)
    window = [g for g in rating_games(games) if g.end_time > since]
    if len(window) < min_games:
        return None
    xs = [(g.end_time - since).total_seconds() / 86400.0 for g in window]
    ys = [float(g.my_rating) for g in window]  # type: ignore[arg-type]  # rating_games guarantees ratings
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx if sxx > 0 else 0.0
    first = window[0].my_rating_before if window[0].my_rating_before is not None else window[0].my_rating
    return Trend(
        pool=pool,
        n=len(window),
        change=slope * days,
        first=int(first),  # type: ignore[arg-type]
        last=int(window[-1].my_rating),  # type: ignore[arg-type]
        test=summarize(window).test,
        games=window,
    )


# --------------------------------------------------------------------------- tables & charts
_SCORE_FORMATS = ["pct", "pct", "signed_pct"]


def _score_cells(s: ScoreSummary) -> list[Any]:
    return [s.score, s.expected, s.delta]


def time_control_table(by_pool: dict[str, list[Game]], summaries: dict[str, ScoreSummary]) -> Table:
    rows = []
    for pool, games in by_pool.items():
        s, h = summaries[pool], rating_history(games)
        rows.append([pool, s.n, s.wdl, *_score_cells(s), h.current, h.change, h.peak])
    unrated = sum(s.n - s.n_rated for s in summaries.values())
    note = (
        "Difference = your score minus the Elo expected score, per game (+5% = 5 extra points per 100 games). "
        "Each time control has its own rating, so judge each row against its own expectation."
    )
    if unrated:
        note += f" Expected and Difference leave out {unrated} game(s) without ratings."
    return Table(
        title="By time control",
        columns=[
            "Time control",
            "Games",
            "W/D/L",
            "Score",
            "Expected",
            "Difference",
            "Current rating",
            "Change",
            "Peak",
        ],
        rows=rows,
        formats=["text", "int", "text", *_SCORE_FORMATS, "rating", "signed_int", "rating"],
        note=note,
    )


def rating_chart(by_pool: dict[str, list[Game]], min_games: int) -> Optional[Chart]:
    """Monthly closing rating per time control with at least ``min_games`` games."""
    per_pool: dict[str, dict[str, int]] = {}
    for pool, games in by_pool.items():
        rg = rating_games(games)
        if len(rg) < min_games:
            continue
        months: dict[str, int] = {}
        for g in rg:  # end-time order: the month's last game wins
            months[f"{g.end_time:%Y-%m}"] = int(g.my_rating)  # type: ignore[arg-type]
        per_pool[pool] = months
    if not per_pool:
        return None
    labels = sorted({m for months in per_pool.values() for m in months})
    return Chart(
        kind="line",
        title="Rating by month",
        labels=labels,
        series=[
            Series(pool, [float(m[label]) if label in m else None for label in labels]) for pool, m in per_pool.items()
        ],
        value_format="rating",
        note="Your rating after the last game of each month. Gaps are months without games in that time control.",
    )


def delta_chart(summaries: dict[str, ScoreSummary], min_games: int) -> Optional[Chart]:
    pools = [p for p, s in summaries.items() if s.n_rated >= min_games]
    if not pools:
        return None
    return Chart(
        kind="bar",
        title="Score vs expected by time control",
        labels=pools,
        series=[Series("Score − expected", [summaries[p].delta for p in pools])],
        value_format="signed_pct",
        note=(
            "Per game, against each time control's own rating expectation "
            f"(time controls with {min_games}+ rated games)."
        ),
        reference=0.0,
    )


# --------------------------------------------------------------------------- insights
def _colour_study(colour: str, games: Sequence[Game]) -> list[str]:
    losses = [g for g in games if g.outcome == "loss"]
    families: dict[str, int] = {}
    for g in games:
        if g.opening_family and not g.initial_fen:
            families[g.opening_family] = families.get(g.opening_family, 0) + 1
    k = min(MAX_EXAMPLES, len(losses)) or "few"
    study = [
        f"Replay your last {k} losses with {colour.capitalize()} and note where you first got a worse position: "
        "the opening, a tactic, or later in the game."
    ]
    if families:
        fam, n = max(families.items(), key=lambda kv: (kv[1], kv[0]))
        study.append(
            f"Your most common opening with {colour.capitalize()} is the {fam} ({n} games): study its main plans and "
            "typical middlegames so you reach positions you understand."
        )
    if colour == "black":
        study.append("Choose one reply to 1.e4 and one to 1.d4 and stick with them for your next 50 games as Black.")
    else:
        study.append(
            "Pick White openings that lead to familiar, comfortable middlegames and learn them 8-10 moves deep "
            "so you keep the first-move edge."
        )
    return study


def colour_insight(
    games: Sequence[Game], white: ScoreSummary, black: ScoreSummary, th: Thresholds
) -> tuple[Optional[Insight], MeanTest]:
    """White-vs-Black gap beyond the normal White edge (difference of the two mean tests), at ``th.alpha``."""
    test = difference_test(white.test, black.test, offset=WHITE_EDGE)
    if white.n_rated < th.min_color_games or black.n_rated < th.min_color_games:
        return None, test
    excess = test.mean  # > 0: White does better than usual relative to Black
    significant, confidence = significance(test, 2 * th.min_color_games, alpha=th.alpha)
    if abs(excess) < th.min_effect or not significant:
        return None, test
    weaker, stronger = ("black", "white") if excess > 0 else ("white", "black")
    ws, ss = (black, white) if weaker == "black" else (white, black)
    # colour-adjusted performance: White is expected to beat the Elo expectation by WHITE_EDGE/2, Black to trail by it
    weaker_adj = ws.test.mean + (WHITE_EDGE / 2 if weaker == "black" else -WHITE_EDGE / 2)
    kind: InsightKind = "weakness" if weaker_adj < 0 else "observation"
    title = (
        f"You score noticeably worse with the {weaker} pieces"
        if kind == "weakness"
        else f"You get much more out of the {stronger} pieces than the {weaker} ones"
    )
    colour_games = [g for g in games if g.color == weaker]
    detail = (
        f"With {weaker.capitalize()} you score {pct(ws.rated_score)} in {ws.n_rated} games where "
        f"{pct(ws.expected)} was expected ({fmt_points(ws.test.mean)} points per game); "
        f"with {stronger.capitalize()} {pct(ss.rated_score)} where "
        f"{pct(ss.expected)} was expected ({fmt_points(ss.test.mean)}). White normally scores about "
        f"{WHITE_EDGE * 100:.0f} points per 100 games more than Black; your White-minus-Black gap is "
        f"{fmt_signed((excess + WHITE_EDGE) * 100)}, {abs(excess) * 100:.0f} points per 100 games "
        f"{'wider' if excess > 0 else 'narrower'} than usual (about {abs(score_to_elo_diff(abs(excess))):.0f} Elo)."
    )
    return (
        Insight(
            id=f"{KEY}.{kind}.colour-{weaker}",
            kind=kind,
            category="color",
            title=title,
            detail=detail,
            severity=severity_from_points(excess),
            confidence=confidence,
            evidence={
                "weaker": weaker,
                "white": _evidence(white),
                "black": _evidence(black),
                "excess_gap": excess,
                "white_edge": WHITE_EDGE,
                "p_value": test.p_value,
            },
            study=_colour_study(weaker, colour_games),
            example_games=recent_urls(colour_games, "loss"),
        ),
        test,
    )


def _evidence(s: ScoreSummary, **extra: Any) -> dict[str, Any]:
    return {
        "n": s.n,
        "n_rated": s.n_rated,
        "wins": s.wins,
        "draws": s.draws,
        "losses": s.losses,
        "score": s.score,
        "rated_score": s.rated_score,
        "expected": s.expected,
        "delta": s.delta,
        "se": s.test.se if math.isfinite(s.test.se) else None,
        "p_value": s.test.p_value,
        **extra,
    }


_OPPONENT_TEXT = {
    ("lower", "weakness"): (
        "You drop points against lower-rated players",
        [
            "Replay your last {k} losses to lower-rated players and find the move where the game turned: a blunder, "
            "overpressing, or the clock.",
            "When you are ahead against a weaker player, trade pieces and keep it simple instead of looking for a "
            "brilliant finish.",
            "Blunder-check every move against lower-rated opponents: what does their last move attack?",
        ],
    ),
    ("lower", "strength"): (
        "You reliably beat lower-rated players",
        [
            "Keep your routine against weaker players: review your recent wins to see how you convert.",
            "Use that technique against equal opponents too: simplify when ahead and limit counterplay.",
        ],
    ),
    ("higher", "strength"): (
        "You punch above your weight against stronger players",
        [
            "Seek out stronger opponents: review your best wins against them to see what worked.",
            "Bring the same fighting spirit and care to games against equal and weaker opponents.",
        ],
    ),
    ("higher", "weakness"): (
        "You struggle to score against stronger players",
        [
            "Replay your last {k} losses to stronger players and note where you fell behind: the opening, a tactic, "
            "or a slow squeeze.",
            "Against stronger opponents aim for solid, familiar positions rather than sharp lines you know less well.",
            "Play some longer games against stronger opponents and review them with an engine afterwards.",
        ],
    ),
}


def opponent_insights(games: Sequence[Game], th: Thresholds) -> tuple[list[Insight], dict[str, Any]]:
    """Weaker (rating_diff < -50) vs stronger (> +50) opponents, robust to the unknown rating-noise attenuation.

    Each group is tested against the expectation at both ends of ``ATTENUATION_RANGE`` (the
    attenuated one and the plain Elo formula), BH-adjusted over the groups, at the strict
    level. A claim needs both tests significant in the same direction: then it holds for any
    attenuation in between. Otherwise the rating noise alone could explain it (with realistic
    noise the plain Elo formula makes everyone look weak against lower-rated players, and a
    too-strong correction makes everyone look strong against them).
    """
    groups = {
        "lower": [g for g in games if g.rating_diff is not None and g.rating_diff < -50],
        "higher": [g for g in games if g.rating_diff is not None and g.rating_diff > 50],
    }
    lo, hi = ATTENUATION_RANGE
    summaries = {k: summarize(v, attenuate=lo) for k, v in groups.items()}
    plain = {k: summarize(v, attenuate=hi) for k, v in groups.items()}
    tested = [k for k, s in summaries.items() if s.n_rated >= th.min_bucket_games]
    adjusted = dict(zip(tested, bh_adjust([summaries[k].test.p_value for k in tested]), strict=True))
    adjusted_plain = dict(zip(tested, bh_adjust([plain[k].test.p_value for k in tested]), strict=True))
    out: list[Insight] = []
    for key in tested:
        s, r = summaries[key], plain[key]
        sig_att, conf_att = significance(s.test, th.min_bucket_games, adjusted[key], alpha=th.strict_alpha)
        sig_raw, conf_raw = significance(r.test, th.min_bucket_games, adjusted_plain[key], alpha=th.strict_alpha)
        same_direction = s.test.mean * r.test.mean > 0
        effect = min(abs(s.test.mean), abs(r.test.mean))  # the weaker of the two readings
        if not (sig_att and sig_raw and same_direction) or effect < th.min_effect:
            continue
        kind: InsightKind = "strength" if s.test.mean > 0 else "weakness"
        title, study = _OPPONENT_TEXT[(key, kind)]
        group = groups[key]
        losses = [g for g in group if g.outcome == "loss"]
        k = min(MAX_EXAMPLES, len(losses)) or "few"
        side = "below" if key == "lower" else "above"
        if kind == "weakness":
            examples = recent_urls(group, "loss")
        elif key == "lower":
            examples = recent_urls(group, "win")
        else:  # best wins against stronger players: biggest rating gap first
            wins = sorted(
                (g for g in group if g.outcome == "win"), key=lambda g: (g.rating_diff or 0, g.end_time), reverse=True
            )
            examples = [g.url for g in wins if g.url][:MAX_EXAMPLES]
        out.append(
            Insight(
                id=f"{KEY}.{kind}.{key}-rated-opponents",
                kind=kind,
                category="opponents",
                title=title,
                detail=(
                    f"Against opponents rated more than 50 points {side} you, you score {pct(s.rated_score)} in "
                    f"{s.n_rated} games where about {pct(s.expected)} would be normal for the rating gap "
                    f"({fmt_points(s.test.mean)} points per game; {fmt_points(r.test.mean)} against the plain "
                    f"Elo formula's {pct(r.expected)})."
                ),
                severity=severity_from_points(effect),
                confidence=min(conf_att, conf_raw),
                evidence=_evidence(
                    s,
                    p_adjusted=adjusted[key],
                    expected_kind="attenuated",
                    elo_expected=r.expected,
                    elo_delta=r.test.mean,
                    elo_p_adjusted=adjusted_plain[key],
                ),
                study=[a.format(k=k) for a in study],
                example_games=examples,
            )
        )
    stats = {
        k: _evidence(s, p_adjusted=adjusted.get(k), elo_delta=plain[k].delta, elo_p_adjusted=adjusted_plain.get(k))
        for k, s in summaries.items()
    }
    return out, stats


def time_control_insights(
    by_pool: dict[str, list[Game]], summaries: dict[str, ScoreSummary], th: Thresholds
) -> tuple[list[Insight], dict[str, Any]]:
    """Each time control vs your other time controls, relative to each one's own rating expectation.

    Every pool is judged against its own rating, so each pool's score-vs-expected is near 0
    by construction and picking the best and the worst of several noisy numbers would find a
    "gap" in pure noise. Instead each pool is compared with all the other tested pools
    together (a difference test), the p-values are BH-adjusted over the pools, and a claim
    needs the strict level.
    """
    pools = [p for p, s in summaries.items() if s.n_rated >= th.min_tc_games]
    if len(pools) < 2:
        return [], {}
    ranked = sorted(pools, key=lambda p: summaries[p].shrunk() or 0.0)
    rests = {p: summarize([g for q in pools if q != p for g in by_pool[q]]) for p in pools}
    tests = {p: difference_test(summaries[p].test, rests[p].test) for p in pools}
    adjusted = dict(zip(pools, bh_adjust([tests[p].p_value for p in pools]), strict=True))
    stats: dict[str, Any] = {
        "best": ranked[-1],
        "worst": ranked[0],
        "tests": {
            p: {"gap": tests[p].mean, "p_value": tests[p].p_value, "p_adjusted": adjusted[p]} for p in pools
        },
    }
    out: list[Insight] = []
    for pool in pools:
        s, rest, test = summaries[pool], rests[pool], tests[pool]
        significant, confidence = significance(test, 2 * th.min_tc_games, adjusted[pool], alpha=th.strict_alpha)
        # the pool must itself be clearly off its own expectation, not merely better/worse than the rest
        if not significant or abs(test.mean) < th.min_effect or abs(s.test.mean) < th.min_effect / 2:
            continue
        if test.mean * s.test.mean <= 0:
            continue
        name = pool.lower()
        gap_elo = abs(score_to_elo_diff(test.mean))
        detail = (
            f"Measured against your own rating in each: in {name} you score {pct(s.rated_score)} where "
            f"{pct(s.expected)} was expected ({fmt_points(s.test.mean)} points per game, {s.n_rated} games); in your "
            f"other time controls {pct(rest.rated_score)} where {pct(rest.expected)} was expected "
            f"({fmt_points(rest.test.mean)}, {rest.n_rated} games). The gap is worth about {gap_elo:.0f} Elo."
        )
        evidence = {
            "pool": _evidence(s, pool=pool),
            "rest": _evidence(rest),
            "gap": test.mean,
            "p_value": test.p_value,
            "p_adjusted": adjusted[pool],
        }
        if test.mean > 0:
            out.append(
                Insight(
                    id=f"{KEY}.strength.time-control-{slugify(pool)}",
                    kind="strength",
                    category="results",
                    title=f"{pool} is your best time control",
                    detail=detail,
                    severity=severity_from_points(test.mean),
                    confidence=confidence,
                    evidence=evidence,
                    study=[
                        f"Play more of your rated games at {name}: that is where your chess currently shows best.",
                        f"Compare two {name} wins with two losses from your other time controls: how did you use "
                        "your clock, and where did the losses turn?",
                    ],
                    example_games=recent_urls(by_pool[pool], "win"),
                )
            )
        else:
            fast = by_pool[pool][0].time_class in ("bullet", "blitz")
            study = [
                f"Replay your last {min(MAX_EXAMPLES, s.losses) or 'few'} {name} losses and note whether they "
                "were lost on the clock, to a blunder, or in the opening.",
                f"Play fewer {name} games for a while, or treat them as practice and review every loss.",
            ]
            if fast:
                study.append("Add increment (for example 3+2 instead of 3+0) so you can finish games without scrambling.")
            out.append(
                Insight(
                    id=f"{KEY}.weakness.time-control-{slugify(pool)}",
                    kind="weakness",
                    category="results",
                    title=f"{pool} is your weakest time control",
                    detail=detail,
                    severity=severity_from_points(test.mean),
                    confidence=confidence,
                    evidence=evidence,
                    study=study,
                    example_games=recent_urls(by_pool[pool], "loss"),
                )
            )
    return out, stats


def trend_insight(trend: Trend, th: Thresholds) -> Insight:
    """The recent rating trend: always an observation.

    A rating is a random walk around your strength (a 90-day swing of 100+ points happens
    by chance), so a rising or falling line is not by itself evidence of a strength or a
    weakness.
    """
    name = trend.pool.lower()
    change = trend.change
    detail = (
        f"Over the last {TREND_DAYS} days ({trend.n} rated games) your {name} rating went from {trend.first} to "
        f"{trend.last}; the trend line says {fmt_signed(change)} points. In those games you scored "
        f"{fmt_points(trend.test.mean)} points per game against your rating's expectation."
    )
    if change >= TREND_FLAT:
        title = f"Your {name} rating is up about {change:.0f} points in the last {TREND_DAYS} days"
        study = [
            "Keep doing what works: note what changed recently (more puzzles, slower games, a new opening) "
            "and stick with it.",
            f"Replay your three best {name} wins from this period to lock in the patterns.",
        ]
        wins = sorted(
            (g for g in trend.games if g.outcome == "win"), key=lambda g: (g.rating_diff or 0, g.end_time), reverse=True
        )
        examples = [g.url for g in wins if g.url][:MAX_EXAMPLES]
    elif change <= -TREND_FLAT:
        title = f"Your {name} rating has slipped about {abs(change):.0f} points in the last {TREND_DAYS} days"
        study = [
            f"Replay your last five {name} losses and look for a common thread: the opening, blunders, or the clock.",
            "Take a short break from rated games, or play a slower time control for a week, then come back fresh.",
        ]
        examples = recent_urls(trend.games, "loss")
    else:
        title = f"Your {name} rating has been stable over the last {TREND_DAYS} days"
        study = [
            "A plateau usually needs a change in training, not more games: pick one weakness from this report and "
            "work on it for the next 30 days.",
            "Spend five minutes on every loss right after the game before starting another.",
        ]
        examples = recent_urls(trend.games, "loss")
    return Insight(
        id=f"{KEY}.observation.trend-{slugify(trend.pool)}",
        kind="observation",
        category="results",
        title=title,
        detail=detail,
        severity=clamp(abs(change) / 100.0),
        confidence=sample_confidence(trend.n),
        evidence={
            "pool": trend.pool,
            "n": trend.n,
            "change": change,
            "first": trend.first,
            "last": trend.last,
            "delta": trend.test.mean,
            "p_value": trend.test.p_value,
        },
        study=study,
        example_games=examples,
    )


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def accuracy_section(
    by_pool: dict[str, list[Game]], th: Thresholds
) -> tuple[Optional[Table], Optional[Insight], dict[str, Any]]:
    """chess.com Game Review accuracy (only present for reviewed games)."""
    rows, stats = [], {}
    for pool, games in by_pool.items():
        rev = [g for g in games if g.my_accuracy is not None]
        if not rev:
            continue
        mine = _avg(g.my_accuracy for g in rev)
        wins = _avg(g.my_accuracy for g in rev if g.outcome == "win")
        losses = _avg(g.my_accuracy for g in rev if g.outcome == "loss")
        opp = _avg(g.opp_accuracy for g in rev)
        rows.append([pool, len(rev), mine, wins, losses, opp])
        stats[pool] = {"n": len(rev), "mine": mine, "wins": wins, "losses": losses, "opponents": opp}
    if not rows:
        return None, None, stats
    table = Table(
        title="chess.com accuracy (reviewed games)",
        columns=["Time control", "Reviewed games", "Your accuracy", "In wins", "In losses", "Opponents"],
        rows=rows,
        formats=["text", "int", "float1", "float1", "float1", "float1"],
        note="chess.com only reports accuracy for games you ran Game Review on, so these games may not be typical.",
    )
    reviewed = [g for games in by_pool.values() for g in games if g.my_accuracy is not None]
    if len(reviewed) < th.min_accuracy_games:
        return table, None, stats
    mine = _avg(g.my_accuracy for g in reviewed) or 0.0
    wins = _avg(g.my_accuracy for g in reviewed if g.outcome == "win")
    losses_g = [g for g in reviewed if g.outcome == "loss"]
    losses = _avg(g.my_accuracy for g in losses_g)
    opp = _avg(g.opp_accuracy for g in reviewed)
    parts = [f"{r[0].lower()} {r[2]:.1f} ({r[1]} games)" for r in rows]
    split = []
    if wins is not None:
        split.append(f"{wins:.1f} in wins")
    if losses is not None:
        split.append(f"{losses:.1f} in losses")
    detail = (
        f"Across {len(reviewed)} reviewed games: {', '.join(split) or 'no decisive games'}; "
        f"by time control: {', '.join(parts)}."
    )
    if opp is not None:
        detail += f" Your opponents averaged {opp:.1f} in the same games."
    title = f"Your chess.com accuracy averages {mine:.1f}"
    if wins is not None and losses is not None:
        title += f" ({wins:.1f} in wins, {losses:.1f} in losses)"
    worst = sorted(losses_g, key=lambda g: (g.my_accuracy or 0.0, -g.end_time.timestamp()))
    insight = Insight(
        id=f"{KEY}.observation.chesscom-accuracy",
        kind="observation",
        category="accuracy",
        title=title,
        detail=detail,
        severity=clamp(abs(wins - losses) / 40.0) if wins is not None and losses is not None else 0.1,
        confidence=sample_confidence(len(reviewed)),
        evidence={"n": len(reviewed), "mine": mine, "wins": wins, "losses": losses, "opponents": opp},
        study=[
            "Run Game Review on your losses as well as your wins: losses teach the most, and reviewing only wins "
            "makes these averages look better than they are.",
            "In each reviewed loss, find the first move marked as a mistake or blunder and write down what you missed.",
        ],
        example_games=[g.url for g in worst if g.url][:MAX_EXAMPLES],
    )
    return table, insight, stats


# --------------------------------------------------------------------------- module entry point
def _kpis(games: Sequence[Game], overall: ScoreSummary, by_pool: dict[str, list[Game]], min_games: int) -> list[Kpi]:
    kpis = [
        Kpi("Games", overall.n, "int", hint=f"{games[0].end_time:%Y-%m} to {games[-1].end_time:%Y-%m}"),
        Kpi("Win / draw / loss", f"{overall.wins} / {overall.draws} / {overall.losses}", "text"),
        Kpi("Score", overall.score, "pct", hint="wins + half the draws"),
        Kpi("Expected score", overall.expected, "pct", hint="what the Elo ratings predict"),
        Kpi("Score vs expected", overall.delta, "signed_pct", hint="per game: +2% = 2 extra points per 100 games"),
        Kpi(
            "Rating equivalent",
            round(score_to_elo_diff(overall.delta, overall.expected))
            if overall.delta is not None and overall.expected is not None
            else None,
            "signed_int",
            hint="roughly how many Elo above (+) or below (-) your rating you played",
        ),
    ]
    histories = {pool: rating_history(g) for pool, g in by_pool.items()}
    main = sorted((p for p, h in histories.items() if h.n >= min_games), key=lambda p: -histories[p].n)[
        :MAX_RATING_KPIS
    ]
    if not main:
        with_rating = [p for p, h in histories.items() if h.n]
        main = sorted(with_rating, key=lambda p: -histories[p].n)[:1]
    for pool in main:
        h = histories[pool]
        hint = f"{h.n} games" + (f", {fmt_signed(h.change)} over the period" if h.change is not None else "")
        kpis.append(Kpi(f"{pool} rating", h.current, "rating", hint=hint))
    if main:
        h = histories[main[0]]
        kpis.append(
            Kpi(
                "Peak rating",
                h.peak,
                "rating",
                hint=f"{main[0].lower()}, {h.peak_time:%Y-%m-%d}" if h.peak_time else main[0].lower(),
            )
        )
    return kpis


def _summary(overall: ScoreSummary, insights: list[Insight], th: Thresholds) -> str:
    n = overall.n
    if overall.n_rated == 0:
        text = f"You scored {pct(overall.score)} in {n} games (no ratings available to compare against)."
    else:
        where = f"in {n} games" if overall.n_rated == n else f"in the {overall.n_rated} of {n} games with ratings"
        text = (
            f"You scored {pct(overall.rated_score)} {where} where your ratings predicted {pct(overall.expected)} "
            f"({fmt_points(overall.test.mean)} points per game)."
        )
    if n < th.min_games:
        return (
            f"Not enough data yet: only {n} game{'s' if n != 1 else ''}, so treat these numbers as a snapshot. {text}"
        )
    ranked = sorted((i for i in insights if i.kind != "observation"), key=lambda i: -i.priority)
    if ranked:
        text += f" Key finding: {ranked[0].title}."
    return text


def analyze(ctx: AnalysisContext) -> ModuleResult:
    """Score vs expectation overall, by time control, colour and opponent strength; rating history and trend."""
    th = Thresholds.from_ctx(ctx)
    games = sorted(ctx.games, key=lambda g: g.end_time)
    if not games:
        return ModuleResult(
            key=KEY, title=TITLE, summary="Not enough data: there are no games to analyse yet.", stats={"n": 0}
        )

    overall = summarize(games)
    by_pool = group_by_pool(games)
    pool_summaries = {pool: summarize(g) for pool, g in by_pool.items()}
    insights: list[Insight] = []
    tables: list[Table] = [time_control_table(by_pool, pool_summaries)]
    charts: list[Chart] = [
        c for c in (rating_chart(by_pool, th.min_chart_games), delta_chart(pool_summaries, th.min_chart_games)) if c
    ]

    # colour
    white_games = [g for g in games if g.color == "white"]
    black_games = [g for g in games if g.color == "black"]
    white, black = summarize(white_games), summarize(black_games)
    tables.append(
        Table(
            title="By colour",
            columns=["Colour", "Games", "W/D/L", "Score", "Expected", "Difference"],
            rows=[
                ["White", white.n, white.wdl, *_score_cells(white)],
                ["Black", black.n, black.wdl, *_score_cells(black)],
            ],
            formats=["text", "int", "text", *_SCORE_FORMATS],
            note=(
                "The Elo expectation ignores colour: White typically scores about 2% above it and Black about 2% below."
            ),
        )
    )
    colour, colour_test = colour_insight(games, white, black, th)
    if colour:
        insights.append(colour)

    # opponent strength
    buckets: dict[str, list[Game]] = {key: [] for key, _ in OPPONENT_BUCKETS}
    for g in games:
        key = opponent_bucket(g.rating_diff)
        if key:
            buckets[key].append(g)
    rows, bucket_stats, deltas = [], {}, []
    for key, label in OPPONENT_BUCKETS:
        raw, att = summarize(buckets[key]), summarize(buckets[key], attenuate=True)
        rows.append([label, att.n, att.wdl, att.score, raw.expected, att.expected, att.delta])
        deltas.append(att.delta)
        bucket_stats[key] = _evidence(att, elo_expected=raw.expected)
    no_diff = sum(1 for g in games if g.rating_diff is None)
    tables.append(
        Table(
            title="By opponent strength",
            columns=["Opponent vs you", "Games", "W/D/L", "Score", "Elo expected", "Realistic expected", "Difference"],
            rows=rows,
            formats=["text", "int", "text", "pct", "pct", "pct", "signed_pct"],
            note=(
                "Opponent rating minus yours before the game. Ratings are imperfect, so everyone scores a little below "
                "the Elo formula against weaker players and above it against stronger ones; Difference is measured "
                "against that realistic expectation."
                + (f" {no_diff} game(s) without ratings are not included." if no_diff else "")
            ),
        )
    )
    if any(d is not None for d in deltas):
        charts.append(
            Chart(
                kind="bar",
                title="Score vs realistic expectation by opponent strength",
                labels=[label for _, label in OPPONENT_BUCKETS],
                series=[Series("Score − expected", deltas)],
                value_format="signed_pct",
                note="Opponent rating minus yours. Empty bars are buckets without games.",
                reference=0.0,
            )
        )
    opp_insights, opp_stats = opponent_insights(games, th)
    insights.extend(opp_insights)

    # time controls
    tc_insights, tc_stats = time_control_insights(by_pool, pool_summaries, th)
    insights.extend(tc_insights)

    # rating trend
    end = games[-1].end_time
    trends = [t for pool, g in by_pool.items() if (t := rating_trend(pool, g, end, th.min_trend_games))]
    insights.extend(trend_insight(t, th) for t in trends)

    # chess.com accuracy
    acc_table, acc_insight, acc_stats = accuracy_section(by_pool, th)
    if acc_table:
        tables.append(acc_table)
    if acc_insight:
        insights.append(acc_insight)

    if overall.n < th.min_games:  # too little data for any claim beyond the raw numbers
        insights = []

    return ModuleResult(
        key=KEY,
        title=TITLE,
        summary=_summary(overall, insights, th),
        kpis=_kpis(games, overall, by_pool, th.min_chart_games),
        tables=tables,
        charts=charts,
        insights=insights,
        stats={
            **_evidence(overall),
            "delta_elo": score_to_elo_diff(overall.delta, overall.expected)
            if overall.delta is not None and overall.expected is not None
            else None,
            "by_time_control": {
                pool: _evidence(s, rating=dataclasses.asdict(rating_history(by_pool[pool])))
                for pool, s in pool_summaries.items()
            },
            "by_colour": {"white": _evidence(white), "black": _evidence(black)},
            "colour_gap": {"excess": colour_test.mean, "p_value": colour_test.p_value, "white_edge": WHITE_EDGE},
            "opponent_buckets": bucket_stats,
            "opponent_groups": opp_stats,
            "time_control_comparison": tc_stats,
            "trends": {
                t.pool: {"n": t.n, "change": t.change, "first": t.first, "last": t.last, "delta": t.test.mean}
                for t in trends
            },
            "accuracy": acc_stats,
        },
    )
