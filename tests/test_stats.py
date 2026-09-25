import math

import pytest

from chess_insights import stats


def test_expected_score_direction():
    assert stats.expected_score(1500, 1600) == pytest.approx(0.36, abs=0.001)
    assert stats.expected_score(1600, 1500) == pytest.approx(0.64, abs=0.001)
    assert stats.expected_score(1500, 1500) == 0.5


def test_performance_rating_is_clamped():
    assert stats.performance_rating(1500, 0.5) == pytest.approx(1500)
    assert stats.performance_rating(1500, 1.0) == pytest.approx(1500 + 400 * math.log10(0.99 / 0.01))
    assert stats.performance_rating(1500, 0.0) == pytest.approx(1500 - 400 * math.log10(0.99 / 0.01))


def test_wilson_interval_contains_p_and_handles_edges():
    lo, hi = stats.wilson_interval(30, 100)
    assert lo < 0.3 < hi
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


def test_mean_test():
    t = stats.mean_test([0.1] * 50 + [-0.1] * 50)
    assert t.n == 100 and abs(t.mean) < 1e-12 and t.p_value == pytest.approx(1.0)
    strong = stats.mean_test([-0.2, -0.3, -0.1, -0.25] * 20)
    assert strong.mean < 0 and strong.p_value < 0.001 and strong.confidence > 0.99
    assert stats.mean_test([]).p_value == 1.0
    assert stats.mean_test([0.5]).p_value == 1.0
    assert stats.mean_test([float("nan"), 0.2, 0.3]).n == 2


def test_two_proportion_test():
    t = stats.two_proportion_test(30, 100, 10, 100)
    assert t.mean == pytest.approx(0.2) and t.p_value < 0.01
    assert stats.two_proportion_test(1, 0, 1, 10).p_value == 1.0


def test_combined_confidence_requires_min_n_and_grows_with_n():
    small = stats.mean_test([-0.2, -0.3, -0.1] * 3)
    big = stats.mean_test([-0.2, -0.3, -0.1] * 30)
    assert stats.combined_confidence(small, min_n=20) == 0.0
    assert 0 < stats.combined_confidence(big, min_n=20) <= 1.0
    assert stats.combined_confidence(big, min_n=20) >= stats.combined_confidence(stats.mean_test([-0.2, -0.3, -0.1] * 7), min_n=20)


def test_bh_adjust():
    adj = stats.bh_adjust([0.01, 0.04, 0.03, 0.2])
    assert adj == pytest.approx([0.04, 0.0533333, 0.0533333, 0.2], rel=1e-4)
    assert stats.bh_adjust([]) == []
    assert all(a >= p for a, p in zip(stats.bh_adjust([0.001, 0.5, 0.9]), [0.001, 0.5, 0.9]))


def test_attenuated_expected_pulls_toward_half():
    assert stats.attenuated_expected(0.5) == 0.5
    assert stats.attenuated_expected(0.9) == pytest.approx(0.8)
    assert stats.attenuated_expected(0.1) == pytest.approx(0.2)


def test_shrink_and_sample_confidence():
    assert stats.shrink(-0.3, 10, 0.0, prior_n=10) == pytest.approx(-0.15)
    assert stats.shrink(0.2, 0, 0.05) == 0.05
    assert stats.sample_confidence(20, half_at=20) == 0.5
    assert stats.sample_confidence(0) == 0.0


def test_formatting_helpers():
    assert stats.pct(0.534) == "53%" and stats.signed_pct(-0.07) == "-7%" and stats.pct(None) == "n/a"
    assert stats.median([3, 1, 2]) == 2 and stats.median([1, 2, 3, 4]) == 2.5 and stats.median([]) is None
    assert stats.score_to_elo_diff(0.0) == 0.0
    assert stats.score_to_elo_diff(0.1) == pytest.approx(-400 * math.log10(1 / 0.6 - 1))  # 60% score = +70 Elo


def test_mean_test_min_sd_keeps_small_streaky_samples_honest():
    losses = [-0.5] * 8  # eight straight losses where 50% was expected
    naive = stats.mean_test(losses)
    floored = stats.mean_test(losses, min_sd=stats.SCORE_RESIDUAL_SD)
    assert naive.p_value < 1e-6  # zero variance -> fake certainty
    assert 0.001 < floored.p_value < 0.01
    assert stats.mean_test([0.3, -0.7] * 40, min_sd=0.01).se == stats.mean_test([0.3, -0.7] * 40).se


# --------------------------------------------------------------------------- significance policy & group maths
def test_significance_needs_min_n_finite_se_and_adjusted_p():
    strong = stats.mean_test([-0.2, -0.3, -0.1, -0.25] * 20)  # n = 80, p ~ 0
    ok, conf = stats.significance(strong, min_n=20)
    assert ok and conf > 0.99
    assert stats.significance(strong, min_n=100) == (False, 0.0)  # too few samples
    ok, conf = stats.significance(strong, min_n=20, p_adjusted=0.03)
    assert ok and conf == pytest.approx(stats.evidence_confidence(0.03)) and 0.75 < conf < 0.8  # not 1 - p = 0.97
    assert not stats.significance(strong, min_n=20, p_adjusted=0.03, alpha=stats.STRICT_ALPHA)[0]
    # n: the effective sample size of a comparison (its smaller group), for the minimum and the confidence
    assert stats.significance(strong, min_n=20, n=15) == (False, 0.0)
    ok, small = stats.significance(strong, min_n=20, p_adjusted=0.03, n=20)
    assert ok and small == pytest.approx(stats.evidence_confidence(0.03) * math.sqrt(0.5))
    assert not stats.significance(strong, min_n=20, p_adjusted=0.06)[0]
    assert not stats.is_significant(stats.MeanTest(30, -0.5, float("inf"), 0.0, 0.0), min_n=10)
    assert stats.STRICT_ALPHA < stats.ALPHA == 0.05


def test_significant_claims_always_clear_the_ranking_floor():
    from chess_insights.insights import MIN_CONFIDENCE

    borderline = stats.MeanTest(20, -0.2, 0.1, -1.97, 0.049)  # just significant at the minimum sample size
    ok, conf = stats.significance(borderline, min_n=20)
    assert ok and conf >= MIN_CONFIDENCE


def test_evidence_confidence_is_the_sellke_bayarri_berger_bound():
    # the most a p-value can support on even prior odds: 1 / (1 + (-e p ln p))
    assert stats.evidence_confidence(0.05) == pytest.approx(0.711, abs=1e-3)
    assert stats.evidence_confidence(0.01) == pytest.approx(0.889, abs=1e-3)
    assert stats.evidence_confidence(0.001) == pytest.approx(0.982, abs=1e-3)
    assert stats.evidence_confidence(0.5) == stats.evidence_confidence(1.0) == 0.5  # no evidence either way
    assert stats.evidence_confidence(0.0) == 1.0
    assert stats.claim_confidence(0.001, n=10, min_n=20) == 0.0


def test_shrink_effect_pulls_noisy_effects_hardest():
    assert stats.shrink_effect(-0.2, 0.09) == pytest.approx(-0.2 * 0.01 / (0.01 + 0.0081))
    assert abs(stats.shrink_effect(-0.2, 0.03)) > abs(stats.shrink_effect(-0.2, 0.09))
    assert stats.shrink_effect(0.2, float("inf")) == 0.0


def test_proportion_gap_test_is_a_difference_of_differences():
    # 6/8 quick losses vs 0/16 quick wins in one opening; 0/20 and 0/20 elsewhere
    t = stats.proportion_gap_test(6, 8, 0, 16, 0, 20, 0, 20)
    assert t.mean == pytest.approx(0.75) and t.p_value < 0.001
    same_gap = stats.proportion_gap_test(12, 12, 0, 12, 40, 40, 0, 40)  # a quick resigner everywhere
    assert same_gap.mean == 0.0 and same_gap.p_value == 1.0
    assert stats.proportion_gap_test(1, 0, 1, 1, 1, 1, 1, 1).p_value == 1.0


def test_weighted_bh_gives_big_groups_more_of_the_budget():
    p = [0.012, 0.012, 0.5, 0.5, 0.5]
    plain = stats.bh_adjust(p)
    assert plain[0] == pytest.approx(0.03)
    weighted = stats.weighted_bh_adjust(p, [3, 1, 1, 1, 1])  # the first group has 3x the games
    assert weighted[0] < plain[0] < weighted[1]
    assert stats.weighted_bh_adjust(p, [1] * 5) == pytest.approx(plain)
    assert all(a >= q for a, q in zip(stats.weighted_bh_adjust([0.9, 0.001], [10, 1]), [0.9, 0.001]))
    assert stats.weighted_bh_adjust([], []) == [] and stats.weighted_bh_adjust([0.2], [0]) == [0.2]


def test_weighted_bh_keeps_the_family_error_rate_under_the_null():
    import random

    rng = random.Random(4)
    hits = 0
    for _ in range(4000):
        p = [rng.random() for _ in range(8)]
        hits += min(stats.weighted_bh_adjust(p, [5, 1, 1, 1, 2, 3, 1, 1])) <= 0.05
    assert hits / 4000 < 0.06


def test_colour_expected_and_colour_adjusted_summaries():
    from factories import make_game

    assert stats.colour_expected(0.5, "white") == pytest.approx(0.52)
    assert stats.colour_expected(0.5, "black") == pytest.approx(0.48)
    assert stats.colour_expected(0.995, "white") == 0.99
    games = [make_game(color="black", outcome=o) for o in ["win", "loss"] * 10]
    plain, adjusted = stats.summarize(games), stats.summarize(games, colour_adjust=True)
    assert plain.delta == pytest.approx(0.0) and adjusted.delta == pytest.approx(0.02)
    raw = stats.summarize([make_game(opp_rating=1700)], attenuate=1.0)
    att = stats.summarize([make_game(opp_rating=1700)], attenuate=True)
    assert raw.expected == pytest.approx(stats.expected_score(1500, 1700))
    assert att.expected == pytest.approx(stats.attenuated_expected(raw.expected))
    assert stats.ATTENUATION_RANGE == (stats.RATING_NOISE_ATTENUATION, 1.0)
    lo, hi = stats.WHITE_EDGE_RANGE
    assert lo < stats.WHITE_EDGE < hi


def test_group_helpers_are_still_importable_from_results():
    from chess_insights.analysis import results

    assert results.summarize is stats.summarize and results.ScoreSummary is stats.ScoreSummary
    assert results.difference_test is stats.difference_test and results.with_p_value is stats.with_p_value
    assert results.WHITE_EDGE == stats.WHITE_EDGE
