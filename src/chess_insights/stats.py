"""Small statistics toolkit shared by the analysis modules.

The whole point of this project is to separate real patterns from noise in a
few hundred games, so every "you are weak/strong at X" claim should go through
these helpers:

* compare against the Elo *expected* score, not 50% (opponent strength confounds
  raw win rates);
* attach a sample size + interval, and derive ``Insight.confidence`` from it;
* shrink small-sample rates toward the player's overall rate before ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


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


def mean_test(values: Iterable[float], min_se: float = 1e-9) -> MeanTest:
    """z-test of mean(values) against 0 (normal approximation).

    Typical use: ``mean_test(g.score - g.expected_score for g in games)`` — is the
    player over/under-performing their rating in this subset?
    """
    xs = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return MeanTest(0, 0.0, float("inf"), 0.0, 1.0)
    m = sum(xs) / n
    if n == 1:
        return MeanTest(1, m, float("inf"), 0.0, 1.0)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = max(math.sqrt(var / n), min_se)
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


def attenuated_expected(expected: float, attenuation: float = RATING_NOISE_ATTENUATION) -> float:
    return 0.5 + attenuation * (expected - 0.5)


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


def severity_from_points(points_per_game: float, full_at: float = 0.20) -> float:
    """Severity for a score-vs-expected effect: 0.20 points/game (~140 Elo) = 1.0."""
    return clamp(abs(points_per_game) / full_at)


def score_to_elo_diff(delta_score: float, around: float = 0.5) -> float:
    """Rough Elo equivalent of a score change near ``around`` (for human-readable details)."""
    lo = clamp(around, 0.01, 0.99)
    hi = clamp(around + delta_score, 0.01, 0.99)
    to_elo = lambda s: -400.0 * math.log10(1.0 / s - 1.0)  # noqa: E731
    return to_elo(hi) - to_elo(lo)


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
