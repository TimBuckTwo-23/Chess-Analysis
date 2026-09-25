"""The report must still find real effects of realistic size.

Each test plants one effect in the null world (tests/null_world.py) and checks that
the report's top lists name it, as the right kind (weakness) and category, without
contradicting it. The detection rates behind these fixed seeds, measured over 32
seeds at 400 and 600 games, are in docs/METHODOLOGY.md (section 2): at 600 games
every effect below except the colour one is found in 78-100% of seeds, at 400
games tilt, late night and time trouble are (colour needs about 1,000 games).

POWER_RUNS=32 (optionally POWER_GAMES=400) re-measures those rates; it takes a few
minutes per effect, so it only runs when asked for.
"""

import os
from functools import lru_cache

import pytest

from chess_insights.pipeline import run_analysis
from null_world import null_games

GAMES = 600
SEED = 2

# name: (planted effects, category, keywords that must appear in the insight's id or title)
EFFECTS = {
    "opening": ({"opening": ("black", "Caro-Kann Defense", -0.15)}, "openings", ("caro-kann",)),
    "tilt": ({"tilt_after_loss": -0.12}, "habits", ("after a loss",)),
    "late_night": ({"late_night": -0.10}, "habits", ("night",)),
    "time_trouble": ({"time_trouble": (0.35, -0.10)}, "time", ("time-trouble",)),
    "flagging": ({"flagging": 0.25}, "time", ("lost-on-time",)),
    "short_losses": ({"short_losses": ("white", "Italian Game", 0.40)}, "openings", ("early-losses", "italian")),
    "colour": ({"colour": ("black", -0.08)}, "color", ("black",)),
}


@lru_cache(maxsize=None)
def report_for(name: str):
    planted, _, _ = EFFECTS[name]
    return run_analysis(null_games(GAMES, seed=SEED, planted=planted), "nullplayer")


def found(report, category: str, keywords: tuple[str, ...], kind: str = "weakness"):
    return [
        i
        for i in report.strengths + report.weaknesses
        if i.kind == kind and i.category == category and all(k in f"{i.id} {i.title}".lower() for k in keywords)
    ]


@pytest.mark.parametrize("name", list(EFFECTS))
def test_planted_weakness_reaches_the_top_list(name):
    _, category, keywords = EFFECTS[name]
    report = report_for(name)
    hits = found(report, category, keywords)
    assert hits, [(i.id, round(i.confidence, 2)) for i in report.strengths + report.weaknesses]
    assert all(i.confidence >= 0.5 for i in hits)
    # never the opposite claim about the same thing
    assert not found(report, category, keywords, kind="strength")


def test_planted_effects_are_described_with_their_numbers():
    opening = found(report_for("opening"), "openings", ("caro-kann",))[0]
    assert opening.title == "The Caro-Kann Defense is costing you points as Black"
    assert opening.evidence["colour_adjusted_delta"] < -0.08 and opening.evidence["p_adjusted"] <= 0.05
    tilt = found(report_for("tilt"), "habits", ("after a loss",))[0]
    assert tilt.evidence["gap"] < -0.06 and tilt.evidence["p_value_one_sided"] <= 0.05
    late = found(report_for("late_night"), "habits", ("night",))[0]
    assert "23:00" in late.title or "00:00" in late.title


POWER_RUNS = int(os.environ.get("POWER_RUNS", "0"))
POWER_GAMES = int(os.environ.get("POWER_GAMES", str(GAMES)))
MIN_DETECTION = 0.7  # at 600 games; colour is below it (information-limited, see the methodology)


@pytest.mark.skipif(POWER_RUNS == 0, reason="set POWER_RUNS to measure detection rates over many seeds")
@pytest.mark.parametrize("name", [n for n in EFFECTS if n != "colour"])
def test_detection_rate(name):
    planted, category, keywords = EFFECTS[name]
    hits = sum(
        bool(found(run_analysis(null_games(POWER_GAMES, seed=s, planted=planted), "nullplayer"), category, keywords))
        for s in range(POWER_RUNS)
    )
    print(f"{name}: detected in {hits}/{POWER_RUNS} worlds of {POWER_GAMES} games")
    if POWER_GAMES >= 600:
        assert hits / POWER_RUNS >= MIN_DETECTION
