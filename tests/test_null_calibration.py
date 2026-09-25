"""The report must not invent strengths or weaknesses.

On null-world data (tests/null_world.py: realistic rating noise and White's
first-move edge, but no real effects), every strength/weakness that reaches
the report's top lists is a false claim. Each module runs dozens of tests, so a
small rate is unavoidable; the significance policy in ``stats`` (two alpha tiers,
Benjamini-Hochberg within each family of tests) is tuned to keep it near 0.1-0.2
claims per report (docs/METHODOLOGY.md, section 2). The null player lives in UTC and
says so (option tz="Etc/UTC"), so the late-night window is tested at the looser level:
the harder case.

The same must hold when the player's habits or rating move in ways that are not
skill effects: a rating that lags the player's strength (every game a little above or
below expectation), and a player who never resigns (losses are longer and end in mate).

The default run (10 seeds of 400 games, plus four shifted and two never-resign worlds)
takes about 35 s. For a heavier check set CALIBRATION_RUNS (e.g. 64) and optionally
CALIBRATION_GAMES (e.g. 600); with 16 or more runs the mean must also meet the 0.3
target, not just the 0.5 hard limit.
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


OPTIONS = {"tz": "Etc/UTC"}  # the null world's schedule is in UTC


def claims_of(report) -> list[tuple[str, float]]:
    return [(i.id, round(i.confidence, 2)) for i in report.strengths + report.weaknesses]


@pytest.fixture(scope="module")
def null_reports():
    return [run_analysis(null_games(GAMES, seed=s), "nullplayer", options=OPTIONS) for s in range(RUNS)]


def test_few_false_claims_on_null_data(null_reports):
    claims = [claims_of(r) for r in null_reports]
    mean_claims = sum(len(c) for c in claims) / len(claims)
    assert mean_claims <= MAX_MEAN_FALSE_CLAIMS, claims
    assert max(len(c) for c in claims) <= MAX_FALSE_CLAIMS_ONE_RUN, claims
    if RUNS >= 16:
        assert mean_claims <= TARGET_MEAN_FALSE_CLAIMS, claims


@pytest.mark.parametrize("shift", [0.06, -0.06])
def test_a_rating_that_lags_creates_no_findings(shift):
    # Every game 6 points per 100 above (below) the rating's expectation: an improving (declining) player
    # whose rating hasn't caught up. No subset of games is different, so nothing may be singled out; the
    # old tests against the expectation alone named openings, game lengths and opponent groups.
    reports = [run_analysis(null_games(600, seed=s, planted={"baseline": shift}), "p", options=OPTIONS) for s in (0, 1)]
    claims = [claims_of(r) for r in reports]
    assert sum(len(c) for c in claims) <= 1, claims


def test_a_player_who_never_resigns_gets_no_endings_claims():
    # 80% of the losses the player would have resigned are played on to mate: longer losses, more mates,
    # identical results. The old endings module called that a king-safety weakness and a weakness in long
    # games (plus "strengths" in short ones) in every report.
    reports = [run_analysis(null_games(400, seed=s, planted={"play_on": 0.8}), "p", options=OPTIONS) for s in (0, 1)]
    for r in reports:
        endings = next(m for m in r.modules if m.key == "endings")
        assert not [i.id for i in endings.insights if i.kind != "observation"]
        assert any(i.id == "endings.observation.checkmated-often" for i in endings.insights)  # described, not claimed
    claims = [claims_of(r) for r in reports]
    assert sum(len(c) for c in claims) <= 1, claims


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
    play_on = null_games(n, seed=seed, planted={"play_on": 1.0})
    assert [g.outcome for g in play_on] == [g.outcome for g in base]
    mated = lambda gs: sum(g.outcome == "loss" and g.termination == "checkmate" for g in gs)  # noqa: E731
    assert mated(play_on) > mated(base) + 20 and all(len(g.clocks) == g.plies for g in play_on)
    tilt = null_games(n, seed=seed, planted={"tilt_after_loss": -0.4})
    after_loss = [
        b for a, b in zip(tilt, tilt[1:]) if a.outcome == "loss" and b.start_time - a.end_time <= timedelta(minutes=15)
    ]
    assert sum(g.score for g in after_loss) / len(after_loss) < 0.3
    with pytest.raises(ValueError):
        null_games(10, planted={"no_such_effect": 1})
