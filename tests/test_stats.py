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
