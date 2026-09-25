"""The report must not invent strengths or weaknesses.

On null-world data (tests/null_world.py: realistic rating noise and White's
first-move edge, but no real effects), every strength/weakness that reaches
the report's top lists is a false claim. We allow a small rate because each
module runs dozens of tests, but the report has to stay trustworthy.
"""

import os

import pytest

from chess_insights.pipeline import run_analysis
from null_world import null_games

RUNS = int(os.environ.get("CALIBRATION_RUNS", "6"))
GAMES = 400
MAX_MEAN_FALSE_CLAIMS = 0.5  # per report, averaged over runs
MAX_FALSE_CLAIMS_ONE_RUN = 2


@pytest.fixture(scope="module")
def null_reports():
    return [run_analysis(null_games(GAMES, seed=s), "nullplayer") for s in range(RUNS)]


@pytest.mark.xfail(reason="modules not yet calibrated against the null world", strict=False)
def test_few_false_claims_on_null_data(null_reports):
    claims = [[(i.id, round(i.confidence, 2)) for i in r.strengths + r.weaknesses] for r in null_reports]
    mean_claims = sum(len(c) for c in claims) / len(claims)
    assert mean_claims <= MAX_MEAN_FALSE_CLAIMS, claims
    assert max(len(c) for c in claims) <= MAX_FALSE_CLAIMS_ONE_RUN, claims


def test_null_world_is_sane():
    games = null_games(300, seed=99)
    wins = sum(g.outcome == "win" for g in games)
    assert 0.3 < wins / len(games) < 0.6
    assert all(len(g.clocks) == g.plies for g in games)
    assert all(g.rating_diff is not None for g in games[1:])
