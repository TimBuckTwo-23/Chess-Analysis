"""Small statistics toolkit shared by the analysis modules.

The whole point of this project is to separate real patterns from noise in a
few hundred games, so every "you are weak/strong at X" claim should go through
these helpers:

* compare against the Elo *expected* score, not 50% (opponent strength confounds
  raw win rates);
* attach a sample size + interval, and derive ``Insight.confidence`` from it;
* shrink small-sample rates toward the player's overall rate before ranking;
* decide "is this a claim?" with ONE project-wide rule, :func:`significance`
  (see "Significance policy" below and docs/METHODOLOGY.md section 2).

``ScoreSummary`` / ``summarize`` / ``difference_test`` (score vs expectation of a
group of games) live here because every result-based module uses them.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    from .models import Game


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def expected_score(my_rating: float, opp_rating: float) -> float:
    """Elo expected score for the player rated ``my_rating``."""
    return 1.0 / (1.0 + 10 ** ((opp_rating - my_rating) / 400.0))


def performance_rating(avg_opp_rating: float, score: float) -> float:
    """Classic performance rating, clamped to +-800 around the opponents' average."""
    s = clamp(score, 0.01, 0.99)
    return avg_opp_rating + clamp(400.0 * math.log10(s / (1.0 - s)), -800.0, 800.0)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. ``successes`` may be fractional (draws = 0.5)."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (clamp(centre - half), clamp(centre + half))


@dataclass
class MeanTest:
    n: int
    mean: float
    se: float
    z: float
    p_value: float  # two-sided, H0: mean == 0

    @property
    def confidence(self) -> float:
        """1 - p, i.e. how sure we are the effect is not zero (0..1)."""
        return clamp(1.0 - self.p_value)


# Per-game standard deviation of (score - expected) when a player performs exactly to
# their rating with ~10% draws: sqrt(E(1-E) - draws/4) ~ 0.47 near E = 0.5. Used as a
# floor so that streaky small samples (8 straight losses) don't produce se ~ 0, p ~ 0.
SCORE_RESIDUAL_SD = 0.45


def mean_test(values: Iterable[float], min_se: float = 1e-9, min_sd: float = 0.0) -> MeanTest:
    """z-test of mean(values) against 0 (normal approximation).

    Typical use: ``mean_test((g.score - g.expected_score for g in games), min_sd=SCORE_RESIDUAL_SD)``
    — is the player over/under-performing their rating in this subset? ``min_sd``
    floors the per-item standard deviation so small, streaky samples keep honest
    p-values.
    """
    xs = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return MeanTest(0, 0.0, float("inf"), 0.0, 1.0)
    m = sum(xs) / n
    if n == 1:
        return MeanTest(1, m, float("inf"), 0.0, 1.0)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = max(math.sqrt(max(var, min_sd**2) / n), min_se)
    z = m / se
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return MeanTest(n, m, se, z, p)


def two_proportion_test(x1: float, n1: int, x2: float, n2: int) -> MeanTest:
    """z-test for p1 - p2 (pooled). Returned ``mean`` is p1 - p2."""
    if n1 <= 0 or n2 <= 0:
        return MeanTest(0, 0.0, float("inf"), 0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    z = (p1 - p2) / se
    return MeanTest(n1 + n2, p1 - p2, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


def one_sided(test: MeanTest, direction: int) -> MeanTest:
    """``test`` with a one-sided p-value for H1: mean > 0 (``direction`` = +1) or mean < 0 (-1).

    Only for hypotheses whose direction is fixed in advance (the tool only ever claims
    "you play *worse* after a loss"); a result in the other direction gets p >= 0.5.
    """
    if not math.isfinite(test.se) or test.n == 0:
        return dataclasses.replace(test, p_value=1.0)
    p = 1.0 - normal_cdf(test.z) if direction > 0 else normal_cdf(test.z)
    return dataclasses.replace(test, p_value=clamp(p))


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (same order as the input).

    Use when testing many groups at once (every opening, every hour of the day):
    with 20 openings, one will look "significant" at p < 0.05 by luck alone.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        adjusted[i] = clamp(running)
    return adjusted


# Ratings are noisy estimates of strength, so real results regress toward 50%:
# everyone seems to "overperform" against stronger and "underperform" against weaker
# opponents. Judge opponent-strength buckets against this attenuated expectation.
RATING_NOISE_ATTENUATION = 0.75


# How much rating noise attenuates real results is uncertain: 0.75 is a pessimistic
# guess, and a realistic 75-Elo noise on the rating difference gives about 0.94. A claim
# about opponent strength must hold for every attenuation in this range (both ends).
ATTENUATION_RANGE = (RATING_NOISE_ATTENUATION, 1.0)

# At equal ratings White scores roughly 52% and Black 48% in club play, so a
# White-minus-Black gap of about 0.04 points per game is normal. Groups that are all one
# colour (an opening as Black) are judged against the colour-adjusted expectation.
WHITE_EDGE = 0.04


def attenuated_expected(expected: float, attenuation: float = RATING_NOISE_ATTENUATION) -> float:
    return 0.5 + attenuation * (expected - 0.5)


def colour_expected(expected: float, color: str, edge: float = WHITE_EDGE) -> float:
    """``expected`` shifted by half the normal White edge: up for White, down for Black."""
    shift = edge / 2.0 if color == "white" else -edge / 2.0 if color == "black" else 0.0
    return clamp(expected + shift, 0.01, 0.99)


def weighted_bh_adjust(p_values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Weighted Benjamini-Hochberg (Genovese, Roeder & Wasserman 2006), same order as the input.

    Weights are rescaled to average 1 and each p-value is divided by its weight before the
    usual BH step, so the error budget still adds up to ``alpha`` but tests with more weight
    get more of it (the result is never below the raw p-value, which only makes it safer). The analyses weight groups by games played: a leak in the opening you
    play in 40% of your games costs far more points than one in a sideline.
    """
    m = len(p_values)
    if m == 0:
        return []
    w = [max(float(x), 0.0) for x in weights]
    total = sum(w)
    if total <= 0:
        return bh_adjust(p_values)
    scaled = [p / (wi * m / total) if wi > 0 else float("inf") for p, wi in zip(p_values, w, strict=True)]
    # never below the raw p-value: a weight never makes one test more significant than it is alone
    return [max(a, p) for a, p in zip(bh_adjust(scaled), p_values, strict=True)]


def shrink(mean: float, n: int, prior_mean: float, prior_n: float = 10.0) -> float:
    """Empirical-Bayes style shrinkage of a small-sample mean toward a prior."""
    if n <= 0:
        return prior_mean
    return (mean * n + prior_mean * prior_n) / (n + prior_n)


def sample_confidence(n: int, half_at: float = 20.0) -> float:
    """Map a sample size to 0..1 (n = half_at -> 0.5). Use when no test statistic applies."""
    if n <= 0:
        return 0.0
    return n / (n + half_at)


def combined_confidence(test: MeanTest, min_n: int) -> float:
    """Confidence for an insight backed by ``test``; zero below ``min_n`` samples."""
    if test.n < min_n or test.n == 0:
        return 0.0
    size_factor = min(1.0, math.sqrt(test.n / (3.0 * max(min_n, 1))))  # full weight at 3x the minimum
    return clamp(test.confidence * size_factor)


# --------------------------------------------------------------------------- significance policy
# One rule decides whether a test result may become a strength or weakness, in every module:
#
#   enough data (n >= min_n)  AND  (multiple-testing adjusted) p <= alpha  AND  a big enough effect
#
# The effect-size bar stays with each module (it is domain knowledge); this is the statistical
# part. A report runs a few dozen tests. Each family of tests (every opening family, every
# time class...) is Benjamini-Hochberg adjusted, so under pure noise a family makes a false
# claim with probability about alpha when it can claim either direction, and alpha / 2 when
# it only ever claims one (a two-sided p-value). The expected number of false claims per
# report is roughly the sum over families, so the families are split into two tiers that keep
# it near 0.2 on data with no real effects (tests/test_null_calibration.py) while still
# finding realistic effects (tests/test_power.py):
#
# * ALPHA (0.05): the four claims that are both most useful and hardest to detect, because
#   the effects are modest (5-15 points per 100 games) and only part of the games carry them:
#   colour, opening families, playing on straight after a loss and late-night play. The last
#   two are fixed in advance with their direction, so they use a one-sided p-value
#   (``one_sided``).
# * STRICT_ALPHA (0.01): everything else. Exploratory scans (other times of day, game length,
#   time controls, opponent strength, session length), claims with a less direct reading (how
#   losses end), and clock habits and quick losses in one opening, whose effects are large when
#   they are real (paired or mirror tests with plenty of power).
ALPHA = 0.05
STRICT_ALPHA = 0.01


def with_p_value(test: MeanTest, p_value: float) -> MeanTest:
    """Copy of ``test`` with a (multiple-testing adjusted) p-value."""
    return dataclasses.replace(test, p_value=p_value)


def is_significant(test: MeanTest, min_n: int, p_adjusted: Optional[float] = None, alpha: float = ALPHA) -> bool:
    """The project-wide claim rule: at least ``min_n`` samples and adjusted p <= ``alpha``."""
    p = test.p_value if p_adjusted is None else p_adjusted
    return test.n >= max(min_n, 1) and math.isfinite(test.se) and p <= alpha


def significance(
    test: MeanTest, min_n: int, p_adjusted: Optional[float] = None, alpha: float = ALPHA
) -> tuple[bool, float]:
    """(significant, confidence) for a claim backed by ``test``.

    ``p_adjusted`` is the family-wise adjusted p-value (``bh_adjust`` / ``weighted_bh_adjust``)
    when the test is one of several; confidence is :func:`combined_confidence` with that
    p-value, so it reflects the multiple testing too. A significant claim has confidence of
    at least ``(1 - alpha) * 0.58`` (0.55 at the minimum sample size, rising to 1 - p at three
    times it).
    """
    p = test.p_value if p_adjusted is None else p_adjusted
    return is_significant(test, min_n, p, alpha), combined_confidence(with_p_value(test, p), min_n)


def severity_from_points(points_per_game: float, full_at: float = 0.20) -> float:
    """Severity for a score-vs-expected effect: 0.20 points/game (~140 Elo) = 1.0."""
    return clamp(abs(points_per_game) / full_at)


def score_to_elo_diff(delta_score: float, around: float = 0.5) -> float:
    """Rough Elo equivalent of a score change near ``around`` (for human-readable details)."""
    lo = clamp(around, 0.01, 0.99)
    hi = clamp(around + delta_score, 0.01, 0.99)
    to_elo = lambda s: -400.0 * math.log10(1.0 / s - 1.0)  # noqa: E731
    return to_elo(hi) - to_elo(lo)


# --------------------------------------------------------------------------- groups of games
@dataclass
class ScoreSummary:
    """Results of a group of games, compared with the Elo expectation."""

    n: int
    wins: int
    draws: int
    losses: int
    score: Optional[float]  # mean score over every game in the group
    n_rated: int  # games with both ratings: the ones compared with the expectation
    rated_score: Optional[float]  # mean score over the rated games
    expected: Optional[float]  # mean expected score over the rated games
    test: MeanTest  # mean_test(score - expected) over the rated games

    @property
    def delta(self) -> Optional[float]:
        """Points per game above (+) or below (-) the expectation; None without rated games."""
        return self.test.mean if self.n_rated else None

    @property
    def half_width(self) -> Optional[float]:
        """Half-width of the 95% interval around ``delta`` (None below two rated games)."""
        if self.n_rated < 2 or not math.isfinite(self.test.se):
            return None
        return 1.96 * self.test.se

    @property
    def wdl(self) -> str:
        return f"{self.wins}/{self.draws}/{self.losses}"

    def shrunk(self, prior_n: float = 10.0) -> Optional[float]:
        """``delta`` pulled toward 0 as if ``prior_n`` extra games had been played at exactly the expectation."""
        return shrink(self.test.mean, self.n_rated, 0.0, prior_n) if self.n_rated else None


def summarize(
    games: Sequence["Game"], *, attenuate: Union[bool, float] = False, colour_adjust: bool = False
) -> ScoreSummary:
    """W/D/L, score and score-vs-expected for ``games``.

    ``attenuate`` judges against ``attenuated_expected`` (``True``: RATING_NOISE_ATTENUATION,
    or a factor such as 1.0). ``colour_adjust`` adds half the normal White edge to White games'
    expectation and takes it from Black's (``colour_expected``).

    The standard error never drops below its value for a player scoring exactly as expected
    (per-game variance e(1-e) - draw_rate/4, the game-by-game version of SCORE_RESIDUAL_SD),
    so a streaky small sample such as 8 wins out of 9, or 8 straight losses (zero sample
    variance), doesn't look more certain than it is.
    """
    factor = None if attenuate is False else (RATING_NOISE_ATTENUATION if attenuate is True else float(attenuate))
    n = len(games)
    wins = sum(1 for g in games if g.outcome == "win")
    draws = sum(1 for g in games if g.outcome == "draw")
    pairs: list[tuple[float, float]] = []
    for g in games:
        e = g.expected_score
        if e is None:
            continue
        if factor is not None:
            e = attenuated_expected(e, factor)
        if colour_adjust:
            e = colour_expected(e, g.color)
        pairs.append((g.score, e))
    n_rated = len(pairs)
    draw_rate = sum(1 for s, _ in pairs if s == 0.5) / n_rated if n_rated else 0.0
    null_var = sum(max(e * (1.0 - e) - draw_rate / 4.0, 0.0) for _, e in pairs)
    min_se = max(math.sqrt(null_var) / n_rated, 1e-9) if n_rated else 1e-9
    return ScoreSummary(
        n=n,
        wins=wins,
        draws=draws,
        losses=n - wins - draws,
        score=sum(g.score for g in games) / n if n else None,
        n_rated=n_rated,
        rated_score=sum(s for s, _ in pairs) / n_rated if n_rated else None,
        expected=sum(e for _, e in pairs) / n_rated if n_rated else None,
        test=mean_test((s - e for s, e in pairs), min_se=min_se),
    )


def difference_test(a: MeanTest, b: MeanTest, offset: float = 0.0) -> MeanTest:
    """Welch-style z-test of ``a.mean - b.mean - offset`` for two independent samples."""
    diff = a.mean - b.mean - offset
    if a.n < 2 or b.n < 2 or not (math.isfinite(a.se) and math.isfinite(b.se)):
        return MeanTest(a.n + b.n, diff, float("inf"), 0.0, 1.0)
    se = math.sqrt(a.se**2 + b.se**2)
    z = diff / se
    return MeanTest(a.n + b.n, diff, se, z, 2.0 * (1.0 - normal_cdf(abs(z))))


# --------------------------------------------------------------------------- formatting
def pct(x: Optional[float], digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def signed_pct(x: Optional[float], digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:+.{digits}f}%"


def median(xs: Sequence[float]) -> Optional[float]:
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0
