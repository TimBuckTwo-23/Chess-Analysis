"""fill.py: every finding gets its formats and a picture where its module gave none, and nothing else changes."""

import copy
import math

import pytest

from chess_insights import fill, parse, synth
from chess_insights.context import AnalysisContext
from chess_insights.models import Chart, Diagram, Insight, ModuleResult, Series
from chess_insights.pipeline import run_analysis, run_modules
from factories import make_game, make_game_eval

SICILIAN_FEN = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5"


def ins(id_="m.weakness.x", evidence=None, **kw):
    return Insight(id=id_, kind=kw.pop("kind", "weakness"), category=kw.pop("category", "results"), title="t",
                   detail="d", severity=0.6, confidence=0.8, evidence=evidence or {}, **kw)


def ctx_of(games, evals=None):
    return AnalysisContext(username="tester", games=games, evals=evals or {})


def one_module(*items, key="results"):
    return [ModuleResult(key=key, title=key, summary="", insights=list(items))]


# --------------------------------------------------------------------------- the demo report
@pytest.fixture(scope="module")
def demo_games():
    archives = synth.generate_archives(400, seed=7, engine_path=None, workers=2)
    return parse.parse_games([g for key in sorted(archives) for g in archives[key]], synth.DEFAULT_PERSONA.username)


@pytest.fixture(scope="module")
def demo_report(demo_games):
    return run_analysis(demo_games, synth.DEFAULT_PERSONA.username, options={"tz": "Etc/UTC", "demo": True})


def test_every_finding_of_the_demo_report_has_formats_and_a_picture(demo_report):
    reports = [demo_report, *demo_report.format_reports.values()]
    assert len(reports) >= 3  # the demo player has enough blitz and rapid games for their own views
    found = 0
    for report in reports:
        for m in report.modules:
            for i in m.insights:
                found += 1
                assert i.formats and all(n > 0 for n in i.formats.values()), i.id
                assert set(i.formats) <= set(report.formats), i.id
                assert i.chart is not None or i.diagram is not None, i.id
                if i.chart is not None:
                    assert i.chart.labels and all(len(s.values) == len(i.chart.labels) for s in i.chart.series)
                    assert all(v is None or math.isfinite(v) for s in i.chart.series for v in s.values)
        for i in report.strengths + report.weaknesses:  # the ranked copies carry them too
            assert i.formats and (i.chart is not None or i.diagram is not None)
    assert found >= 20


def test_the_fallbacks_change_no_finding(demo_games):
    ctx = ctx_of(sorted(demo_games, key=lambda g: (g.end_time, g.game_id)))
    ctx.options["tz"] = "Etc/UTC"
    modules = run_modules(ctx)
    for m in modules:
        for i in m.insights:
            i.formats, i.chart, i.diagram = {}, None, None
    before = [(i.id, i.kind, i.category, i.severity, i.confidence, i.title, i.detail, i.study, i.example_games,
               copy.deepcopy(i.evidence)) for m in modules for i in m.insights]
    fill.fill(modules, ctx)
    after = [(i.id, i.kind, i.category, i.severity, i.confidence, i.title, i.detail, i.study, i.example_games,
              i.evidence) for m in modules for i in m.insights]
    assert before == after


# --------------------------------------------------------------------------- formats
def test_formats_come_from_the_modules_games_or_the_format_the_evidence_names():
    games = ([make_game(time_class="blitz") for _ in range(5)] + [make_game(time_class="rapid") for _ in range(3)]
             + [make_game(time_class="bullet") for _ in range(2)])
    evals = {g.game_id: make_game_eval(g.game_id, []) for g in games if g.time_class != "blitz"}
    general = ins("results.x")
    clock = ins("time.x", {"time_class": "blitz", "n": 4})
    merged = ins("time.y", {"time_classes": ["rapid", "blitz"], "blitz": {"n": 5}, "rapid": {"n": 2}})
    pool = ins("results.trend", {"pool": "Rapid", "n": 3, "first": 1500, "last": 1510})
    nested_pool = ins("results.tc", {"pool": {"pool": "Bullet (chess960)", "n": 2}, "rest": {"n": 8}})
    engine = ins("engine.x", category="blunders")
    given = ins("results.given", formats={"daily": 7})
    modules = one_module(general, clock, merged, pool, nested_pool, given) + one_module(engine, key="engine_stats")
    fill.fill_formats(modules, ctx_of(games, evals))
    assert general.formats == {"bullet": 2, "blitz": 5, "rapid": 3}
    assert list(general.formats) == ["bullet", "blitz", "rapid"]
    assert clock.formats == {"blitz": 4}
    assert merged.formats == {"blitz": 5, "rapid": 2}
    assert pool.formats == {"rapid": 3}
    assert nested_pool.formats == {"bullet": 2}
    assert engine.formats == {"bullet": 2, "rapid": 3}  # the engine-analysed games only
    assert given.formats == {"daily": 7}  # a module's own formats are kept


# --------------------------------------------------------------------------- visuals
def chart_of(evidence, **kw):
    item = ins(evidence=evidence, **kw)
    fill.fill_visuals(one_module(item))
    return item


def values(chart):
    return {s.name: s.values for s in chart.series}


def test_score_against_expected_becomes_you_vs_expected_bars():
    item = chart_of({"n": 93, "score": 0.40, "expected": 0.50, "rest_delta": 0.03})
    c = item.chart
    assert c.labels == ["You", "Expected from ratings"] and values(c) == {"Score": [0.40, 0.50]}
    assert c.value_format == "pct" and "93 games" in c.note and "+3 per 100 games" in c.note
    grouped = chart_of({"white": {"rated_score": 0.57, "score": 0.5, "expected": 0.50},
                        "black": {"rated_score": 0.42, "expected": 0.50}}).chart
    assert grouped.labels == ["As White", "As Black"]
    assert values(grouped) == {"You": [0.57, 0.42], "Expected from ratings": [0.50, 0.50]}


def test_your_rate_against_your_opponents_in_the_same_games():
    c = chart_of({"time_class": "blitz", "n": 200, "trouble_rate": 0.3, "opponent_trouble_rate": 0.2}).chart
    assert values(c) == {"You": [0.3], "Your opponents": [0.2]} and c.value_format == "pct"
    phase = chart_of({"phase": "endgame", "per100": 9.0, "opp_per100": 6.0, "rest_per100": 4.0,
                      "opp_rest_per100": 4.2}).chart
    assert phase.labels == ["Endgame", "Other phases"] and values(phase)["You"] == [9.0, 4.0]
    split = chart_of({"time_classes": ["blitz", "rapid"], "blitz": {"n": 90, "trouble_rate": 0.4,
                      "opponent_trouble_rate": 0.2}, "rapid": {"n": 40, "trouble_rate": 0.3,
                      "opponent_trouble_rate": 0.1}}).chart
    assert split.labels == ["Blitz", "Rapid"] and values(split) == {"You": [0.4, 0.3], "Your opponents": [0.2, 0.1]}


def test_a_position_finding_gets_the_board_with_both_moves():
    item = chart_of({"fen": SICILIAN_FEN, "move": "e5", "best": "a6", "errors": 10}, category="positions",
                    example_games=["https://www.chess.com/game/live/1"])
    assert item.chart is None and isinstance(item.diagram, Diagram)
    d = item.diagram
    assert d.orientation == "black" and d.link == "https://www.chess.com/game/live/1"
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("e6", "e5", "played"), ("a7", "a6", "best")]


@pytest.mark.parametrize("evidence", [
    {},
    {"n": 40, "p_value": 0.01},  # nothing to compare
    {"score": 0.4, "expected": math.nan},  # not a number
    {"score": 1.7, "expected": 0.5},  # not a share
    {"trouble_rate": 0.3},  # no benchmark
    {"fen": "not a fen", "move": "e5"},
    {"fen": SICILIAN_FEN, "move": "Qxh7"},  # not legal there
])
def test_evidence_that_fits_no_pattern_gets_no_picture(evidence):
    item = chart_of(evidence)
    assert item.chart is None and item.diagram is None


def test_a_modules_own_picture_is_kept():
    own = Chart(kind="bar", title="own", labels=["a"], series=[Series("s", [1.0])])
    item = chart_of({"score": 0.4, "expected": 0.5}, chart=own)
    assert item.chart is own and item.diagram is None
    board = Diagram(title="own", fen=SICILIAN_FEN)
    item = chart_of({"score": 0.4, "expected": 0.5}, diagram=board)
    assert item.chart is None and item.diagram is board


def test_a_broken_evidence_value_never_sinks_the_report():
    class Weird(dict):
        def get(self, *a, **k):
            raise RuntimeError("odd evidence")

    item = ins()
    item.evidence = Weird()
    fill.fill(one_module(item), ctx_of([make_game()]))  # logged, not raised
    assert item.chart is None


# --------------------------------------------------------------------------- review fixes: formats of one position / opening
SICILIAN = ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6", "Nc3"]  # Black to move: SICILIAN_FEN


def sicilian(tc, color="black", **kw):
    return make_game(time_class=tc, color=color, moves_san=SICILIAN + ["e5", "Ndb5", "d6"], **kw)


def test_a_position_finding_gets_the_formats_of_the_games_that_reached_it():
    """Not every format of the module: the games in which the position came up with you to move."""
    reached = [sicilian("blitz") for _ in range(3)] + [sicilian("rapid")]
    as_white = sicilian("bullet", color="white")  # the same position, but not with you to move
    elsewhere = [make_game(time_class="bullet", color="black") for _ in range(5)]  # never reach it
    games = reached + [as_white] + elsewhere
    evals = {g.game_id: make_game_eval(g.game_id, []) for g in games}
    ctx = ctx_of(games, evals)
    found = ins("mistakes.weakness.p", {"fen": SICILIAN_FEN, "move": "e5", "reached": 4}, category="positions")
    wrong = ins("mistakes.weakness.q", {"fen": SICILIAN_FEN, "move": "e5", "reached": 7}, category="positions")
    fill.fill_formats(one_module(found, wrong, key="mistakes"), ctx)
    assert found.formats == {"blitz": 3, "rapid": 1}
    assert wrong.formats == {"bullet": 6, "blitz": 3, "rapid": 1}  # counts disagree: the module's games instead
    assert fill.games_reaching([], games) == {}


def test_an_opening_finding_gets_the_formats_of_that_openings_games():
    caro = [make_game(time_class="blitz", color="black", opening_family="Caro-Kann Defense", outcome=o)
            for o in ("loss", "loss", "win")]
    caro.append(make_game(time_class="rapid", color="black", opening_family="Caro-Kann Defense", outcome="loss"))
    caro_as_white = make_game(time_class="bullet", color="white", opening_family="Caro-Kann Defense")
    other = [make_game(time_class="bullet", color="black", opening_family="Sicilian Defense") for _ in range(4)]
    games = caro + [caro_as_white] + other
    scored = ins("openings.weakness.black.caro", {"color": "black", "family": "Caro-Kann Defense", "n": 4})
    quick = ins("openings.weakness.early", {"color": "black", "family": "Caro-Kann Defense", "losses": 3, "wins": 1})
    engine = ins("engine.x", {"colour": "black", "family": "Caro-Kann Defense", "games": 4})
    miscounted = ins("openings.weakness.y", {"color": "black", "family": "Caro-Kann Defense", "n": 9})
    by_colour = ins("openings.observation.breadth-black", {"color": "black", "games": 8})
    colour_only = ins("openings.observation.z", {"color": "black"})  # nothing to confirm which games
    fill.fill_formats(one_module(scored, quick, engine, miscounted, by_colour, colour_only, key="openings"),
                      ctx_of(games))
    assert scored.formats == quick.formats == engine.formats == {"blitz": 3, "rapid": 1}
    everything = {"bullet": 5, "blitz": 3, "rapid": 1}
    assert miscounted.formats == everything and colour_only.formats == everything
    assert by_colour.formats == {"bullet": 4, "blitz": 3, "rapid": 1}


def test_one_odd_finding_does_not_leave_the_others_without_formats():
    class Weird(dict):
        def get(self, *a, **k):
            raise RuntimeError("odd evidence")

    odd, fine = ins("results.odd"), ins("results.fine", {"n": 3})
    odd.evidence = Weird()
    fill.fill_formats(one_module(odd, fine), ctx_of([make_game(time_class="rapid")]))
    assert odd.formats == {} and fine.formats == {"rapid": 1}


# --------------------------------------------------------------------------- review fixes: pictures
def test_time_pressure_without_the_opponents_numbers_shows_your_own_comparison():
    c = chart_of({"time_class": "blitz", "per100_low": 9.0, "per100_ok": 3.0, "opp_per100_low": None,
                  "opp_per100_ok": 2.5, "opp_moves_low": 0}, category="time").chart
    assert c.labels == ["Short of time", "With time left"] and values(c) == {"You": [9.0, 3.0]}
    assert "you vs your opponents" not in c.title and "too few" in c.note
    # a chart about you against your opponents is not drawn with one side missing
    assert chart_of({"rate": 0.7, "opp_rate": None}).chart is None


def test_a_chart_is_only_drawn_for_the_finding_it_describes():
    """"per100" is blunders in one finding, one phase's mistakes and blunders in another: without "rest_per100"
    for the phase, the blunder chart must not be drawn for it."""
    phase = chart_of({"phase": "endgame", "per100": 9.0, "opp_per100": 6.0}, category="phases").chart
    assert phase is None or "Blunders" not in phase.title
    blunders = chart_of({"blunders": 40, "per100": 4.0, "opp_per100": 3.0, "games": 90}, category="blunders").chart
    assert blunders.title.startswith("Blunders per 100 moves") and values(blunders) == {
        "You": [4.0], "Your opponents": [3.0]}


def test_lost_positions_saved():
    c = chart_of({"lost_positions": 29, "saved": 4, "save_rate": 4 / 29}, category="conversion").chart
    assert values(c) == {"Games": [4.0, 25.0]} and c.value_format == "int" and "29 games" in c.note
    assert chart_of({"lost_positions": 3, "saved": 5}).chart is None  # impossible numbers: no picture


def test_the_board_names_the_moves_with_their_numbers_and_needs_your_move():
    item = chart_of({"fen": SICILIAN_FEN, "move": "e5", "best": "a6"}, category="positions")
    assert item.diagram.title == "You played 5...e5 here"
    assert item.diagram.caption == "Red: your move 5...e5. Green: Stockfish's choice, 5...a6."
    same = chart_of({"fen": SICILIAN_FEN, "move": "e5", "best": "e5"}, category="positions").diagram
    assert [a.kind for a in same.arrows] == ["played"] and "Green" not in same.caption
    only_best = chart_of({"fen": SICILIAN_FEN, "move": "Qxh7", "best": "a6"}, category="positions")
    assert only_best.diagram is None  # a board without your move would not show what the finding is about
    castle = chart_of({"fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 20", "move": "O-O", "best": "O-O-O"},
                      category="positions").diagram
    assert [(a.start, a.end, a.kind) for a in castle.arrows] == [("e1", "g1", "played"), ("e1", "c1", "best")]
    assert castle.title == "You played 20.O-O here" and castle.orientation == "white"


def test_numpy_counts_are_counts():
    import numpy as np

    games = [make_game(time_class="blitz", color="black", opening_family="Caro-Kann Defense") for _ in range(3)]
    games += [make_game(time_class="rapid", color="white") for _ in range(2)]
    right = ins("openings.a", {"color": "black", "family": "Caro-Kann Defense", "n": np.int64(3)})
    wrong = ins("openings.b", {"color": "black", "family": "Caro-Kann Defense", "n": np.int64(7)})
    clock = ins("time.c", {"time_class": "blitz", "n": np.int64(2)})
    fill.fill_formats(one_module(right, wrong, clock, key="openings"), ctx_of(games))
    assert right.formats == {"blitz": 3} and wrong.formats == {"blitz": 3, "rapid": 2} and clock.formats == {"blitz": 2}


def test_a_tactics_chart_names_the_kind_of_error():
    ev = {"count": 30, "per100": 1.2, "opp_count": 15, "opp_per100": 0.6, "other_errors_per100": 5.0,
          "opp_other_errors_per100": 5.1}
    c = chart_of(ev, id_="engine.weakness.hung-material", category="tactics").chart
    assert c.labels == ["Material left hanging", "Other errors"] and values(c)["Your opponents"] == [0.6, 5.1]
    assert chart_of(ev, id_="engine.strength.something-new", category="tactics").chart.labels[0] == "This kind of error"
