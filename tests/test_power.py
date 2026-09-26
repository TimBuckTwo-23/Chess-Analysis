"""The report must still find real effects of realistic size.

Planted effects (tests/null_world.py) must reach the report's top lists as the right kind
(weakness) and category, without the opposite claim. To keep this quick, effects that
touch different games share a world: a 1,200-game player with four habits (tilt, late
night, time trouble, flagging), a 1,200-game player with two opening problems (a weak
Caro-Kann, quick losses in the Italian) and a 2,000-game player who is weaker with
Black. The seed is simply 0: at these sizes each effect is found in 90-100% of worlds
(measured over 32 seeds; the colour effect in about 80%), so passing does not hinge on a
lucky seed. Detection rates by size, for one effect at a time, are in
docs/METHODOLOGY.md (section 2): at 400-600 games the opening, quick-loss and colour
effects are found much less often.

The opening-habit claims (analysis/structure.py: castling, early queen moves, development,
you against your opponents in the same games) get one 600-game world per effect: the tests
are paired within each game, so realistic habit gaps are found at the size of an ordinary
report. These runs name the module explicitly, whether or not pipeline.MODULES has it yet.

POWER_RUNS=32 (optionally POWER_GAMES=600) re-measures those single-effect rates; it
takes a few minutes per effect, so it only runs when asked for.
"""

import os
from functools import lru_cache

import pytest

from chess_insights import pipeline
from chess_insights.pipeline import run_analysis
from null_world import null_games

SEED = 0
OPTIONS = {"tz": "Etc/UTC"}  # the null world's schedule is in UTC

PLANTED = {
    "opening": {"opening": ("black", "Caro-Kann Defense", -0.15)},
    "short_losses": {"short_losses": ("white", "Italian Game", 0.40)},
    "tilt": {"tilt_after_loss": -0.12},
    "late_night": {"late_night": -0.10},
    "time_trouble": {"time_trouble": (0.35, -0.10)},
    "flagging": {"flagging": 0.25},
    "colour": {"colour": ("black", -0.08)},
}
WORLDS = {  # name: (effects, games)
    "habits_and_clock": (("tilt", "late_night", "time_trouble", "flagging"), 1200),
    "openings": (("opening", "short_losses"), 1200),
    "colour": (("colour",), 2000),
}
# effect: (category, keywords that must all appear in the insight's id or title)
EXPECT = {
    "opening": ("openings", ("caro-kann",)),
    "short_losses": ("openings", ("early-losses", "italian")),
    "tilt": ("habits", ("after a loss",)),
    "late_night": ("habits", ("late-night",)),
    "time_trouble": ("time", ("time-trouble",)),
    "flagging": ("time", ("lost-on-time",)),
    "colour": ("color", ("black",)),
}
WORLD_OF = {effect: world for world, (effects, _) in WORLDS.items() for effect in effects}


@lru_cache(maxsize=None)
def report_for(world: str):
    effects, games = WORLDS[world]
    planted = {k: v for e in effects for k, v in PLANTED[e].items()}
    return run_analysis(null_games(games, seed=SEED, planted=planted), "nullplayer", options=OPTIONS)


def found(report, category: str, keywords: tuple[str, ...], kind: str = "weakness"):
    return [
        i
        for i in report.strengths + report.weaknesses
        if i.kind == kind and i.category == category and all(k in f"{i.id} {i.title}".lower() for k in keywords)
    ]


@pytest.mark.parametrize("effect", list(EXPECT))
def test_planted_weakness_reaches_the_top_list(effect):
    category, keywords = EXPECT[effect]
    report = report_for(WORLD_OF[effect])
    hits = found(report, category, keywords)
    assert hits, [(i.id, round(i.confidence, 2)) for i in report.strengths + report.weaknesses]
    assert all(i.confidence >= 0.5 for i in hits)
    # never the opposite claim about the same thing
    assert not found(report, category, keywords, kind="strength")


def test_planted_worlds_carry_few_other_claims():
    # each world's top lists name its planted effects and little else (knock-on findings are re-tested)
    for world, (effects, _) in WORLDS.items():
        report = report_for(world)
        planted = [i for e in effects for i in found(report, *EXPECT[e])]
        others = [i.id for i in report.strengths + report.weaknesses if i not in planted]
        assert len(others) <= 1, (world, others)


def test_planted_effects_are_described_with_their_numbers():
    opening = found(report_for("openings"), *EXPECT["opening"])[0]
    assert opening.title == "You score less with the Caro-Kann Defense than with your other Black openings"
    assert opening.evidence["gap"] < -0.08 and opening.evidence["p_adjusted"] <= 0.05
    tilt = found(report_for("habits_and_clock"), *EXPECT["tilt"])[0]
    assert tilt.evidence["gap"] < -0.06 and tilt.evidence["p_value_one_sided"] <= 0.05
    late = found(report_for("habits_and_clock"), *EXPECT["late_night"])[0]
    assert late.title == "You score below your rating in late-night games (23:00–03:00)"
    flags = found(report_for("habits_and_clock"), *EXPECT["flagging"])
    assert len(flags) == 1  # one finding, however many time controls it shows up in


POWER_RUNS = int(os.environ.get("POWER_RUNS", "0"))
POWER_GAMES = int(os.environ.get("POWER_GAMES", "1200"))
MIN_DETECTION = 0.85  # one effect at a time, at 1,200 games; colour is below it (information-limited)


@pytest.mark.skipif(POWER_RUNS == 0, reason="set POWER_RUNS to measure detection rates over many seeds")
@pytest.mark.parametrize("effect", list(EXPECT))
def test_detection_rate(effect):
    category, keywords = EXPECT[effect]
    hits = sum(
        bool(
            found(
                run_analysis(null_games(POWER_GAMES, seed=s, planted=PLANTED[effect]), "p", options=OPTIONS),
                category,
                keywords,
            )
        )
        for s in range(1000, 1000 + POWER_RUNS)
    )
    print(f"{effect}: detected in {hits}/{POWER_RUNS} worlds of {POWER_GAMES} games")
    if POWER_GAMES >= 1200 and effect != "colour":
        assert hits / POWER_RUNS >= MIN_DETECTION


# --------------------------------------------------------------------------- opening habits (analysis/structure.py)
STRUCTURE = ("structure", "Development & king safety")
STRUCTURE_MODULES = pipeline.MODULES if STRUCTURE in pipeline.MODULES else pipeline.MODULES + [STRUCTURE]
STRUCTURE_GAMES = 600
STRUCTURE_PLANTED = {  # the null world's players castle by move 10 in about 50% of games, 14% bring the queen
    # out early, and they have 3.0 of 4 knights and bishops out by move 10
    "late_castling": {"late_castling": 0.3},  # the player castles by move 10 in about 35% of games
    "early_queen": {"early_queen": 0.2},  # ... brings the queen out early in about 30%
    "slow_development": {"slow_development": 0.6},  # ... has about 0.45 fewer pieces out by move 10
}
STRUCTURE_EXPECT = {
    "late_castling": ("structure", ("castled",)),
    "early_queen": ("structure", ("early-queen",)),
    "slow_development": ("structure", ("minors",)),
}


@lru_cache(maxsize=None)
def structure_report_for(effect: str):
    games = null_games(STRUCTURE_GAMES, seed=SEED, planted=STRUCTURE_PLANTED[effect])
    return run_analysis(games, "nullplayer", options=OPTIONS, modules=STRUCTURE_MODULES)


@pytest.mark.parametrize("effect", list(STRUCTURE_EXPECT))
def test_planted_opening_habit_is_found_at_600_games(effect):
    category, keywords = STRUCTURE_EXPECT[effect]
    report = structure_report_for(effect)
    hits = found(report, category, keywords)
    assert hits, [(i.id, round(i.confidence, 2)) for i in report.strengths + report.weaknesses]
    assert all(i.confidence >= 0.5 and i.formats and i.chart is not None for i in hits)
    assert not found(report, category, keywords, kind="strength")
    # nothing else about the player's first moves
    others = [i.id for i in report.strengths + report.weaknesses if i.category == "structure" and i not in hits]
    assert not others, others


STRUCTURE_POWER_GAMES = int(os.environ.get("POWER_GAMES", str(STRUCTURE_GAMES)))


@pytest.mark.skipif(POWER_RUNS == 0, reason="set POWER_RUNS to measure detection rates over many seeds")
@pytest.mark.parametrize("effect", list(STRUCTURE_EXPECT))
def test_opening_habit_detection_rate(effect):
    category, keywords = STRUCTURE_EXPECT[effect]
    hits = sum(
        bool(
            found(
                run_analysis(
                    null_games(STRUCTURE_POWER_GAMES, seed=s, planted=STRUCTURE_PLANTED[effect]), "p",
                    options=OPTIONS, modules=[STRUCTURE],
                ),
                category,
                keywords,
            )
        )
        for s in range(1000, 1000 + POWER_RUNS)
    )
    print(f"{effect}: detected in {hits}/{POWER_RUNS} worlds of {STRUCTURE_POWER_GAMES} games")
    if STRUCTURE_POWER_GAMES >= STRUCTURE_GAMES:
        assert hits / POWER_RUNS >= MIN_DETECTION
