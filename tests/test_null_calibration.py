"""The report must not invent strengths or weaknesses.

On null-world data (tests/null_world.py: realistic rating noise and White's
first-move edge, but no real effects), every strength/weakness that reaches
the report's top lists is a false claim. Each module runs dozens of tests, so a
small rate is unavoidable; the significance policy in ``stats`` (two alpha tiers,
Benjamini-Hochberg within each family of tests) is tuned to keep it near 0.2
claims per report (docs/METHODOLOGY.md, section 2).

The default run (10 seeds of 400 games) takes about 20 s. For a heavier check set
CALIBRATION_RUNS (e.g. 64) and optionally CALIBRATION_GAMES (e.g. 600); with 16 or
more runs the mean must also meet the 0.3 target, not just the 0.5 hard limit.
"""

import os
from datetime import timedelta

import pytest

from chess_insights.pipeline import run_analysis
from null_world import LATE_NIGHT_HOURS, null_games

RUNS = int(os.environ.get("CALIBRATION_RUNS", "10"))
GAMES = int(os.environ.get("CALIBRATION_GAMES", "400"))
MAX_MEAN_FALSE_CLAIMS = 0.5  # hard limit per report, averaged over runs
TARGET_MEAN_FALSE_CLAIMS = 0.3  # checked when there are enough runs to measure it
MAX_FALSE_CLAIMS_ONE_RUN = 2


@pytest.fixture(scope="module")
def null_reports():
    return [run_analysis(null_games(GAMES, seed=s), "nullplayer") for s in range(RUNS)]


def test_few_false_claims_on_null_data(null_reports):
    claims = [[(i.id, round(i.confidence, 2)) for i in r.strengths + r.weaknesses] for r in null_reports]
    mean_claims = sum(len(c) for c in claims) / len(claims)
    assert mean_claims <= MAX_MEAN_FALSE_CLAIMS, claims
    assert max(len(c) for c in claims) <= MAX_FALSE_CLAIMS_ONE_RUN, claims
    if RUNS >= 16:
        assert mean_claims <= TARGET_MEAN_FALSE_CLAIMS, claims


def test_modules_ran_cleanly_on_null_data(null_reports):
    for report in null_reports[:3]:
        for m in report.modules:
            if m.key not in ("engine_stats", "mistakes"):  # engine modules need evals, which the null world lacks
                assert "error" not in m.stats, (m.key, m.stats.get("error"))


def test_null_world_is_sane():
    games = null_games(300, seed=99)
    wins = sum(g.outcome == "win" for g in games)
    assert 0.3 < wins / len(games) < 0.6
    assert all(len(g.clocks) == g.plies for g in games)
    assert all(g.rating_diff is not None for g in games[1:])
    # a realistic schedule: mostly evening and late-night sessions, enough late-night games to test
    late = sum(g.start_time.hour in LATE_NIGHT_HOURS for g in games) / len(games)
    assert 0.2 < late < 0.45
    assert all(b.start_time >= a.end_time for a, b in zip(games, games[1:]))


def test_planted_effects_change_only_what_they_say():
    n, seed = 160, 5
    base = null_games(n, seed=seed)
    assert [(g.outcome, g.moves_san, g.clocks) for g in null_games(n, seed=seed, planted={})] == [
        (g.outcome, g.moves_san, g.clocks) for g in base
    ]
    flag = null_games(n, seed=seed, planted={"flagging": 0.6})
    lost_on_time = lambda gs: sum(g.outcome == "loss" and g.termination == "timeout" for g in gs)  # noqa: E731
    assert lost_on_time(flag) > lost_on_time(base) + 20
    assert [g.outcome for g in flag] == [g.outcome for g in base]  # flagging changes how losses end, not results
    main = null_games(n, seed=seed, planted={"opening": ("black", "Caro-Kann Defense", -0.15, 0.6)})
    black = [g for g in main if g.color == "black"]
    assert sum(g.opening_family == "Caro-Kann Defense" for g in black) / len(black) > 0.45
    tilt = null_games(n, seed=seed, planted={"tilt_after_loss": -0.4})
    after_loss = [
        b for a, b in zip(tilt, tilt[1:]) if a.outcome == "loss" and b.start_time - a.end_time <= timedelta(minutes=15)
    ]
    assert sum(g.score for g in after_loss) / len(after_loss) < 0.3
    with pytest.raises(ValueError):
        null_games(10, planted={"no_such_effect": 1})
