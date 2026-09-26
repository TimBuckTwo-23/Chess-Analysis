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
