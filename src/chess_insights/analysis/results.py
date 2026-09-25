"""Results & rating: how you score compared with what your rating predicts.

Every comparison is against the Elo expected score of each game, never a raw 50%:

* overall and per time control (each time control has its own rating pool, so each
  row is judged against its own expectation; one time control is only singled out
  when it differs significantly from your other time controls, and then only one);
* White vs Black, allowing for the edge White normally has (a claim must hold for
  every edge in ``stats.WHITE_EDGE_RANGE``); a colour gap that comes from one opening
  (the openings module names it) is not reported a second time as a colour claim;
* opponent-strength groups compared with your *other* games (so a rating that lags
  your current strength doesn't create a finding), against the
  rating-noise-attenuated expectation (``stats.attenuated_expected``); a claim must
  hold for every attenuation in ``stats.ATTENUATION_RANGE``, i.e. against both the
  attenuated and the plain Elo expectation;
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
    WHITE_EDGE_RANGE,
    MeanTest,
    ScoreSummary,
    bh_adjust,
    clamp,
    difference_test,
    pct,
    per100,
    per100_games,
    sample_confidence,
    score_to_elo_diff,
    severity_from_points,
    shrink_effect,
    significance,
    summarize,
    vs_rating,
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
    """'+0.07' / '−0.12' points per game (real minus sign). The report text uses ``stats.per100_games``."""
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
        "Rating predicts = the Elo expected score for the rating gap in each game; vs rating = your score minus that, "
        "per game (+5% = 5 extra points per 100 games). Each time control has its own rating, so judge each row "
        "against its own prediction."
    )
    if unrated:
        note += f" Rating predicts and vs rating leave out {unrated} game(s) without ratings."
    return Table(
        title="By time control",
        columns=[
            "Time control",
            "Games",
            "W/D/L",
            "Score",
            "Rating predicts",
            "vs rating",
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
        note="Your rating after the last game of each month. Longer breaks without games in a time control show as a dashed line.",
    )


def delta_chart(summaries: dict[str, ScoreSummary], min_games: int) -> Optional[Chart]:
    pools = [p for p, s in summaries.items() if s.n_rated >= min_games]
    if not pools:
        return None
    return Chart(
        kind="bar",
        title="Score vs rating by time control",
        labels=pools,
        series=[Series("Score vs rating", [summaries[p].delta for p in pools])],
        value_format="signed_pct",
        note=(
            "Per game, against each time control's own rating expectation "
            f"(time controls with {min_games}+ rated games)."
        ),
        reference=0.0,
    )


# --------------------------------------------------------------------------- insights
def _colour_study(colour: str, games: Sequence[Game], flagged: Iterable[str] = ()) -> list[str]:
    """``flagged``: openings of this colour with a finding of their own; the advice never pushes you deeper into one."""
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
    flagged = set(flagged)
    if families:
        fam, n = max(families.items(), key=lambda kv: (kv[1], kv[0]))
    if families and fam not in flagged:
        study.append(
            f"Your most common opening with {colour.capitalize()} is the {fam} ({n} games): study its main plans and "
            "typical middlegames so you reach positions you understand."
        )
    if flagged:
        names = " and the ".join(sorted(flagged))
        study.append(f"Start with the {names}: see its finding under Openings.")
    elif colour == "black":
        study.append("Choose one reply to 1.e4 and one to 1.d4 and stick with them for your next 50 games as Black.")
    else:
        study.append(
            "Pick White openings that lead to familiar, comfortable middlegames and learn them 8-10 moves deep "
            "so you keep the first-move edge."
        )
    return study


def colour_tests(white: ScoreSummary, black: ScoreSummary) -> dict[str, MeanTest]:
    """White-minus-Black gap beyond the normal White edge, at the central edge and both ends of its range."""
    lo, hi = WHITE_EDGE_RANGE
    return {
        "central": difference_test(white.test, black.test, offset=WHITE_EDGE),
        "low": difference_test(white.test, black.test, offset=lo),  # "White is weaker" must hold here
        "high": difference_test(white.test, black.test, offset=hi),  # "Black is weaker" must hold here
    }


def colour_claim(white: ScoreSummary, black: ScoreSummary, th: Thresholds) -> tuple[Optional[str], float, MeanTest]:
    """(weaker colour or None, confidence, the deciding test) under the claim rule.

    Significance is judged at the edge of ``WHITE_EDGE_RANGE`` least favourable to the claim
    (Black weaker: the largest normal edge; White weaker: the smallest), the size of the effect
    against the central ``WHITE_EDGE``.
    """
    tests = colour_tests(white, black)
    central = tests["central"]
    if white.n_rated < th.min_color_games or black.n_rated < th.min_color_games or abs(central.mean) < th.min_effect:
        return None, 0.0, central
    weaker = "black" if central.mean > 0 else "white"
    deciding = tests["high"] if weaker == "black" else tests["low"]
    if deciding.mean * central.mean <= 0:
        return None, 0.0, deciding
    n = min(white.n_rated, black.n_rated)
    significant, confidence = significance(deciding, th.min_color_games, alpha=th.alpha, n=n)
    return (weaker if significant else None), confidence, deciding


def colour_insight(
    games: Sequence[Game],
    white: ScoreSummary,
    black: ScoreSummary,
    th: Thresholds,
    opening_claims: Sequence[tuple[str, str, str]] = (),
) -> tuple[Optional[Insight], MeanTest]:
    """White-vs-Black gap beyond the normal White edge (difference of the two mean tests), at ``th.alpha``.

    ``opening_claims`` are the (colour, family, kind) the openings module claims. When the
    weaker colour's gap disappears once its claimed weak openings (and the stronger colour's
    claimed strong ones) are left out, it is the same finding: the colour claim becomes an
    observation that points to those openings.
    """
    central = colour_tests(white, black)["central"]
    weaker, confidence, deciding = colour_claim(white, black, th)
    if weaker is None:
        return None, central
    stronger = "white" if weaker == "black" else "black"
    excess = central.mean  # > 0: White does better than usual relative to Black
    ws, ss = (black, white) if weaker == "black" else (white, black)
    explained_by = [
        (c, fam) for c, fam, kind in opening_claims if (c, kind) in ((weaker, "weakness"), (stronger, "strength"))
    ]
    explained = False
    if explained_by:
        left_out = set(explained_by)
        kept = [g for g in games if (g.color, g.opening_family) not in left_out]
        w2 = summarize([g for g in kept if g.color == "white"])
        b2 = summarize([g for g in kept if g.color == "black"])
        explained = colour_claim(w2, b2, th)[0] != weaker
    # colour-adjusted performance: White is expected to beat the Elo expectation by WHITE_EDGE/2, Black to trail by it
    weaker_adj = ws.test.mean + (WHITE_EDGE / 2 if weaker == "black" else -WHITE_EDGE / 2)
    kind: InsightKind = "weakness" if weaker_adj < 0 else "observation"
    if explained:
        kind = "observation"
        names = " and the ".join(fam for _, fam in explained_by)
        verb = "accounts" if len(explained_by) == 1 else "account"
        title = f"The {names} {verb} for most of your gap between White and Black"
    elif kind == "weakness":
        title = f"You score noticeably worse with the {weaker} pieces"
    else:
        title = f"You get much more out of the {stronger} pieces than the {weaker} ones"
    colour_games = [g for g in games if g.color == weaker]
    lo, hi = WHITE_EDGE_RANGE
    detail = (
        f"With {weaker.capitalize()} you score {pct(ws.rated_score)} in {ws.n_rated} games where your rating "
        f"predicts {pct(ws.expected)} ({per100_games(ws.test.mean)} vs your rating); "
        f"with {stronger.capitalize()} {pct(ss.rated_score)} where it predicts "
        f"{pct(ss.expected)} ({per100_games(ss.test.mean)}). White normally scores about "
        f"{lo * 100:.0f}-{hi * 100:.0f} points per 100 games more than Black; your White-minus-Black gap is "
        f"{fmt_signed((excess + WHITE_EDGE) * 100)}, about {abs(excess) * 100:.0f} points per 100 games "
        f"{'wider' if excess > 0 else 'narrower'} than usual (about {abs(score_to_elo_diff(abs(excess))):.0f} Elo)."
    )
    if explained:
        detail += (
            f" Without your games in {'that opening' if len(explained_by) == 1 else 'those openings'}, the gap is "
            "within the normal range (see Openings)."
        )
    return (
        Insight(
            id=f"{KEY}.{kind}.colour-{weaker}",
            kind=kind,
            category="color",
            title=title,
            detail=detail,
            severity=severity_from_points(shrink_effect(excess, central.se)),
            confidence=confidence,
            evidence={
                "weaker": weaker,
                "white": _evidence(white),
                "black": _evidence(black),
                "excess_gap": excess,
                "white_edge": WHITE_EDGE,
                "white_edge_range": list(WHITE_EDGE_RANGE),
                "p_value": deciding.p_value,
                "explained_by": [f"{c}:{fam}" for c, fam in explained_by] if explained else [],
            },
            study=_colour_study(weaker, colour_games, {fam for c, fam, k in opening_claims if c == weaker}),
            example_games=recent_urls(colour_games, "loss"),
        ),
        central,
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
        "You score above par against lower-rated players",
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
    """Weaker (rating_diff < -50) and stronger (> +50) opponents vs your other games, robust to rating noise.

    Each group is compared with all your other rated games (a difference test), so a rating
    that lags your current strength, which shifts every game alike, doesn't create a finding.
    How much rating noise pulls results toward 50% is uncertain, so the comparison is made
    against the expectation at both ends of ``ATTENUATION_RANGE`` (the attenuated one and the
    plain Elo formula), BH-adjusted over the two groups, at the strict level. A claim needs
    both significant in the same direction: then it holds for any attenuation in between.
    Otherwise the rating noise alone could explain it (with realistic noise the plain Elo
    formula makes everyone look weak against lower-rated players, and a too-strong correction
    makes everyone look strong against them).
    """
    rated = [g for g in games if g.rating_diff is not None]
    groups = {
        "lower": [g for g in rated if g.rating_diff < -50],  # type: ignore[operator]
        "higher": [g for g in rated if g.rating_diff > 50],  # type: ignore[operator]
    }
    ends = dict(zip(("attenuated", "plain"), ATTENUATION_RANGE, strict=True))
    summaries: dict[str, dict[str, ScoreSummary]] = {}
    rests: dict[str, dict[str, ScoreSummary]] = {}
    tests: dict[str, dict[str, MeanTest]] = {}
    for key, group in groups.items():
        ids = {id(g) for g in group}
        rest = [g for g in rated if id(g) not in ids]
        summaries[key] = {end: summarize(group, attenuate=a) for end, a in ends.items()}
        rests[key] = {end: summarize(rest, attenuate=a) for end, a in ends.items()}
        tests[key] = {end: difference_test(summaries[key][end].test, rests[key][end].test) for end in ends}
    tested = [
        k
        for k in groups
        if min(summaries[k]["attenuated"].n_rated, rests[k]["attenuated"].n_rated) >= th.min_bucket_games
    ]
    adjusted = {
        end: dict(zip(tested, bh_adjust([tests[k][end].p_value for k in tested]), strict=True)) for end in ends
    }
    out: list[Insight] = []
    for key in tested:
        s, r = summaries[key]["attenuated"], rests[key]["attenuated"]
        t_att, t_raw = tests[key]["attenuated"], tests[key]["plain"]
        n = min(s.n_rated, r.n_rated)
        sig_att, conf_att = significance(t_att, th.min_bucket_games, adjusted["attenuated"][key], th.strict_alpha, n)
        sig_raw, conf_raw = significance(t_raw, th.min_bucket_games, adjusted["plain"][key], th.strict_alpha, n)
        same_direction = t_att.mean * t_raw.mean > 0
        effect = min(abs(t_att.mean), abs(t_raw.mean))  # the weaker of the two readings
        if not (sig_att and sig_raw and same_direction) or effect < th.min_effect:
            continue
        kind: InsightKind = "strength" if t_att.mean > 0 else "weakness"
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
        se = max(t_att.se, t_raw.se)
        out.append(
            Insight(
                id=f"{KEY}.{kind}.{key}-rated-opponents",
                kind=kind,
                category="opponents",
                title=title,
                detail=(
                    f"Against opponents rated more than 50 points {side} you, you score {pct(s.rated_score)} in "
                    f"{s.n_rated} games where about {pct(s.expected)} would be normal for the rating gap "
                    f"({per100_games(s.test.mean)}); in your other games {per100_games(r.test.mean)}. "
                    f"The gap is {per100(t_att.mean)} points per 100 games ({per100(t_raw.mean)} against the plain "
                    "Elo formula), so it holds whatever allowance is made for rating noise."
                ),
                severity=severity_from_points(shrink_effect(effect, se)),
                confidence=min(conf_att, conf_raw),
                evidence=_evidence(
                    s,
                    rest_delta=r.test.mean,
                    rest_n=r.n_rated,
                    gap=t_att.mean,
                    p_adjusted=adjusted["attenuated"][key],
                    expected_kind="attenuated",
                    elo_expected=summaries[key]["plain"].expected,
                    elo_gap=t_raw.mean,
                    elo_p_adjusted=adjusted["plain"][key],
                ),
                study=[a.format(k=k) for a in study],
                example_games=examples,
            )
        )
    stats = {
        k: _evidence(
            summaries[k]["attenuated"],
            gap=tests[k]["attenuated"].mean,
            p_adjusted=adjusted["attenuated"].get(k),
            elo_gap=tests[k]["plain"].mean,
            elo_p_adjusted=adjusted["plain"].get(k),
        )
        for k in groups
    }
    return out, stats


def time_control_insights(
    by_pool: dict[str, list[Game]], summaries: dict[str, ScoreSummary], th: Thresholds
) -> tuple[list[Insight], dict[str, Any]]:
    """At most one time control that stands out from your other time controls, relative to each one's rating.

    Every pool is judged against its own rating, so each pool's score-vs-expected is near 0
    by construction and picking the best and the worst of several noisy numbers would find a
    "gap" in pure noise. Instead each pool is compared with all the other tested pools
    together (a difference test), the p-values are BH-adjusted over the pools, and a claim
    needs the strict level and the pool's own score pointing the same way. The comparisons
    overlap (with two pools they are one test, seen from both ends), so only the strongest one
    becomes a finding. What it measures is how far each rating is from your current results:
    a pool whose rating has not caught up with recent progress (or decline) looks the same.
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
    candidates: list[tuple[float, float, float, str, float]] = []
    for pool in pools:
        s, rest, test = summaries[pool], rests[pool], tests[pool]
        n = min(s.n_rated, rest.n_rated)
        significant, confidence = significance(test, th.min_tc_games, adjusted[pool], alpha=th.strict_alpha, n=n)
        # the pool must itself be clearly off its own expectation, not merely better/worse than the rest
        if not significant or abs(test.mean) < th.min_effect or abs(s.test.mean) < th.min_effect / 2:
            continue
        if test.mean * s.test.mean <= 0:
            continue
        # ties (two pools: one test seen from both ends) go to the pool further from its own rating
        candidates.append((adjusted[pool], -abs(test.mean), -abs(s.test.mean), pool, confidence))
    if not candidates:
        return [], stats
    *_, pool, confidence = min(candidates)
    s, rest, test = summaries[pool], rests[pool], tests[pool]
    stats["claimed"] = pool
    name = pool.lower()
    gap_elo = abs(score_to_elo_diff(test.mean))
    detail = (
        f"Measured against your own rating in each: in {name} you score {pct(s.rated_score)} where your rating "
        f"predicts {pct(s.expected)} ({per100_games(s.test.mean)}, {s.n_rated} games); in your "
        f"other time controls {pct(rest.rated_score)} where it predicts {pct(rest.expected)} "
        f"({per100_games(rest.test.mean)}, {rest.n_rated} games). The gap is worth about {gap_elo:.0f} Elo. Each time "
        f"control has its own rating, so this is also what a {name} rating that lags behind your recent results "
        "there looks like."
    )
    evidence = {
        "pool": _evidence(s, pool=pool),
        "rest": _evidence(rest),
        "gap": test.mean,
        "p_value": test.p_value,
        "p_adjusted": adjusted[pool],
    }
    severity = severity_from_points(shrink_effect(test.mean, test.se))
    if test.mean > 0:
        return [
            Insight(
                id=f"{KEY}.strength.time-control-{slugify(pool)}",
                kind="strength",
                category="results",
                title=f"You outperform your {name} rating more than your other ratings",
                detail=detail,
                severity=severity,
                confidence=confidence,
                evidence=evidence,
                study=[
                    f"Your {name} results are running ahead of your {name} rating: keep playing {name} and "
                    "expect the rating to follow.",
                    f"Compare two {name} wins with two losses from your other time controls: how did you use "
                    "your clock, and where did the losses turn?",
                ],
                example_games=recent_urls(by_pool[pool], "win"),
            )
        ], stats
    fast = by_pool[pool][0].time_class in ("bullet", "blitz")
    study = [
        f"Replay your last {min(MAX_EXAMPLES, s.losses) or 'few'} {name} losses and note whether they "
        "were lost on the clock, to a blunder, or in the opening.",
        f"Play fewer {name} games for a while, or treat them as practice and review every loss.",
    ]
    if fast:
        study.append("Add increment (for example 3+2 instead of 3+0) so you can finish games without scrambling.")
    return [
        Insight(
            id=f"{KEY}.weakness.time-control-{slugify(pool)}",
            kind="weakness",
            category="results",
            title=f"You underperform your {name} rating more than your other ratings",
            detail=detail,
            severity=severity,
            confidence=confidence,
            evidence=evidence,
            study=study,
            example_games=recent_urls(by_pool[pool], "loss"),
        )
    ], stats


def trend_insight(trend: Trend, th: Thresholds) -> Insight:
    """The recent rating trend: always an observation.

    A rating is a random walk around your strength (a 90-day swing of 100+ points happens
    by chance), so a rising or falling line is not by itself evidence of a strength or a
    weakness.
    """
    name = trend.pool.lower()
    change = trend.change
    moved = trend.last - trend.first
    detail = (
        f"Over the last {TREND_DAYS} days ({trend.n} rated games) your {name} rating went from {trend.first} to "
        f"{trend.last} ({fmt_signed(moved)}); a straight line through all those games says {fmt_signed(change)}. "
        f"In those games you scored {per100_games(trend.test.mean)} vs your rating."
    )
    title = (
        f"Your {name} rating went from {trend.first} to {trend.last} in the last {TREND_DAYS} days "
        f"({fmt_signed(moved) if moved else 'no change'})"
    )
    if change >= TREND_FLAT:
        study = [
            f"Replay your three best {name} wins from this period to see what is working.",
            f"Compare your recent {name} games with earlier ones: which openings or habits changed?",
        ]
        wins = sorted(
            (g for g in trend.games if g.outcome == "win"), key=lambda g: (g.rating_diff or 0, g.end_time), reverse=True
        )
        examples = [g.url for g in wins if g.url][:MAX_EXAMPLES]
    elif change <= -TREND_FLAT:
        study = [
            f"Replay your last five {name} losses and look for a common thread: the opening, blunders, or the clock.",
            "Take a short break from rated games, or play a slower time control for a week, then come back fresh.",
        ]
        examples = recent_urls(trend.games, "loss")
    else:
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
            "moved": moved,
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
    by_pool: dict[str, list[Game]], th: Thresholds, engine_ran: bool = False
) -> tuple[Optional[Table], Optional[Insight], dict[str, Any]]:
    """chess.com Game Review accuracy (only present for reviewed games).

    With engine analysis (``engine_ran``) the Engine review section reports accuracy over games nobody
    picked for review, so the chess.com numbers stay a table (no second, differently scaled finding)."""
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
    note = "chess.com only reports accuracy for games you ran Game Review on, so these games may not be typical."
    if engine_ran:
        note += " Its scale differs from the engine accuracy in the Engine review section: compare each only with itself."
    table = Table(
        title="chess.com Game Review accuracy (games you reviewed)",
        columns=["Time control", "Reviewed games", "Your accuracy", "In wins", "In losses", "Opponents"],
        rows=rows,
        formats=["text", "int", "float1", "float1", "float1", "float1"],
        note=note,
    )
    reviewed = [g for games in by_pool.values() for g in games if g.my_accuracy is not None]
    if len(reviewed) < th.min_accuracy_games or engine_ran:
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
    title = f"Your chess.com Game Review accuracy averages {mine:.1f}"
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
def _opening_claims(ctx: AnalysisContext, games: Sequence[Game]) -> list[tuple[str, str, str]]:
    """(colour, family, kind) the openings module claims, with its own thresholds (see ``colour_insight``)."""
    from . import openings  # imported here: openings imports this module's helpers

    return openings.claimed_families(games, openings.Thresholds.from_ctx(ctx))


def _kpis(
    games: Sequence[Game],
    overall: ScoreSummary,
    by_pool: dict[str, list[Game]],
    min_games: int,
    trends: Optional[dict[str, Trend]] = None,
) -> list[Kpi]:
    kpis = [
        Kpi("Games", overall.n, "int", hint=f"{games[0].end_time:%Y-%m} to {games[-1].end_time:%Y-%m}"),
        Kpi("Win / draw / loss", f"{overall.wins} / {overall.draws} / {overall.losses}", "text"),
        Kpi("Score", overall.score, "pct", hint="wins + half the draws"),
        Kpi("Rating predicts", overall.expected, "pct", hint="the Elo expected score for your games"),
        Kpi(
            "Score vs rating",
            overall.delta,
            "signed_pct",
            hint="per game (+2% would be 2 extra points per 100 games)",
        ),
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
        hint = f"{h.n} rated games" + (f", {fmt_signed(h.change)} over the period" if h.change is not None else "")
        trend = (trends or {}).get(pool)
        if trend is not None:
            hint += f", {fmt_signed(trend.last - trend.first)} in the last {TREND_DAYS} days"
        kpis.append(Kpi(f"{pool} rating", h.current, "rating", hint=hint))
    if main:
        h = histories[main[0]]
        kpis.append(
            Kpi(
                f"Peak {main[0].lower()} rating",
                h.peak,
                "rating",
                hint=f"{h.peak_time:%Y-%m-%d}" if h.peak_time else "",
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
            f"({per100_games(overall.test.mean)} vs your rating)."
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
            columns=["Colour", "Games", "W/D/L", "Score", "Rating predicts", "vs rating"],
            rows=[
                ["White", white.n, white.wdl, *_score_cells(white)],
                ["Black", black.n, black.wdl, *_score_cells(black)],
            ],
            formats=["text", "int", "text", *_SCORE_FORMATS],
            note=(
                "The rating's prediction ignores colour: White typically scores about 2% above it and Black about 2% below."
            ),
        )
    )
    colour, colour_test = colour_insight(games, white, black, th, _opening_claims(ctx, games))
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
            columns=["Opponent vs you", "Games", "W/D/L", "Score", "Elo formula", "Predicted", "vs rating"],
            rows=rows,
            formats=["text", "int", "text", "pct", "pct", "pct", "signed_pct"],
            note=(
                "Opponent rating minus yours before the game. Ratings are imperfect, so everyone scores a little below "
                "the Elo formula against weaker players and above it against stronger ones; Predicted allows for "
                "that, and vs rating is measured against it."
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
                series=[Series("Score vs rating", deltas)],
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
    acc_table, acc_insight, acc_stats = accuracy_section(by_pool, th, engine_ran=bool(ctx.evals))
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
        kpis=_kpis(games, overall, by_pool, th.min_chart_games, {t.pool: t for t in trends}),
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
