import sys
import types

import pytest

from chess_insights import insights, pipeline
from chess_insights.models import Insight, ModuleResult
from factories import make_game


def ins(id_, kind="weakness", category="openings", severity=0.8, confidence=0.9, **kw):
    return Insight(id=id_, kind=kind, category=category, title=f"title {id_}", detail=f"detail {id_}",
                   severity=severity, confidence=confidence, **kw)


def mod(key, *items):
    return ModuleResult(key=key, title=key.title(), summary="", insights=list(items))


def test_rank_filters_low_confidence_and_observations():
    modules = [mod("a", ins("w1"), ins("w2", confidence=0.1), ins("o1", kind="observation"), ins("s1", kind="strength"))]
    strengths, weaknesses = insights.rank_insights(modules)
    assert [i.id for i in weaknesses] == ["w1"]
    assert [i.id for i in strengths] == ["s1"]


def test_rank_dedupes_by_id_keeping_highest_priority():
    modules = [mod("a", ins("dup", severity=0.3)), mod("b", ins("dup", severity=0.9))]
    _, weaknesses = insights.rank_insights(modules)
    assert len(weaknesses) == 1 and weaknesses[0].severity == 0.9


def test_rank_diversifies_categories_then_backfills():
    many = [ins(f"o{i}", category="openings", severity=0.9 - i * 0.01) for i in range(5)]
    other = ins("t1", category="time", severity=0.5)
    _, weaknesses = insights.rank_insights([mod("a", *many, other)], top_n=4, per_category=2)
    ids = [i.id for i in weaknesses]
    assert "t1" in ids and len(ids) == 4
    assert sum(1 for i in weaknesses if i.category == "openings") == 3  # 2 by quota + 1 backfill
    assert ids == sorted(ids, key=lambda x: -next(i.priority for i in weaknesses if i.id == x))


def test_study_plan_combines_specific_and_generic_actions_without_repeats():
    w = [
        ins("a", category="tactics", study=["Review your 5 hung-piece games"], example_games=["https://x/1"] * 7),
        ins("b", category="tactics", severity=0.5, study=[]),
    ]
    plan = insights.build_study_plan(w)
    assert plan[0].actions[0] == "Review your 5 hung-piece games"
    assert len(plan[0].games) == 5
    generic_a = set(plan[0].actions[1:])
    assert generic_a and not generic_a & set(plan[1].actions)  # generic tips are not repeated
    assert plan[0].category == "Tactics"


def test_headline():
    assert "No clear patterns" in insights.headline([], [])
    h = insights.headline([ins("s", kind="strength")], [ins("w")])
    assert h.startswith("Biggest opportunity: title w.") and "Biggest strength: title s." in h


def _install_fake_modules(monkeypatch):
    good = types.ModuleType("chess_insights.analysis.fake_good")
    good.analyze = lambda ctx: mod("fake_good", ins("fake.weakness.x"))
    bad = types.ModuleType("chess_insights.analysis.fake_bad")

    def boom(ctx):
        raise ValueError("kaboom")

    bad.analyze = boom
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_good", good)
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_bad", bad)
    return [("fake_good", "Good"), ("fake_bad", "Bad section")]


def test_pipeline_isolates_failing_modules(monkeypatch):
    modules = _install_fake_modules(monkeypatch)
    games = [make_game(), make_game(outcome="loss")]
    report = pipeline.run_analysis(list(reversed(games)), "tester", modules=modules, filters="rated blitz")
    assert [m.key for m in report.modules] == ["fake_good", "fake_bad"]
    assert "kaboom" in report.modules[1].summary and report.modules[1].title == "Bad section"
    assert [i.id for i in report.weaknesses] == ["fake.weakness.x"]
    assert report.study_plan and report.headline.startswith("Biggest opportunity")
    assert report.n_games == 2 and report.date_from <= report.date_to
    assert report.filters == "rated blitz"


def test_pipeline_with_no_games(monkeypatch):
    modules = _install_fake_modules(monkeypatch)
    report = pipeline.run_analysis([], "tester", modules=modules)
    assert report.n_games == 0 and report.date_from is None


def test_default_module_list_names_real_modules():
    names = [n for n, _ in pipeline.MODULES]
    assert names == ["results", "openings", "time_mgmt", "endings", "habits", "engine_stats", "mistakes"]


@pytest.mark.parametrize("argv", [["report", "someone", "--offline"], ["fetch", "someone"], ["demo", "--games", "5"]])
def test_cli_parser_accepts_commands(argv):
    from chess_insights.cli import build_parser

    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_cli_csv_helper():
    from chess_insights.cli import _csv

    assert _csv(["blitz", "rapid,bullet"]) == ["blitz", "rapid", "bullet"]
    assert _csv(None) is None


def test_check_planted_traits_matches_by_kind_category_keywords():
    from chess_insights.cli import check_planted_traits
    from chess_insights.models import Report

    found = ins("openings.weakness.black.caro-kann-defense", category="openings")
    found.title = "The Caro-Kann is costing you points as Black"
    report = Report(username="u", generated_at=None, filters="", n_games=1, date_from=None, date_to=None,
                    modules=[mod("openings", found)], strengths=[], weaknesses=[], study_plan=[])
    traits = [
        {"id": "t1", "category": "openings", "expect": "weakness", "description": "", "keywords": ["caro-kann"]},
        {"id": "t2", "category": "openings", "expect": "strength", "description": "", "keywords": ["italian"]},
    ]
    hits = check_planted_traits(report, traits)
    assert hits[0][1] is found and hits[1][1] is None


def _capture_module(monkeypatch, seen, returns=None):
    fake = types.ModuleType("chess_insights.analysis.fake_capture")

    def analyze(ctx):
        seen.append(ctx)
        return returns if returns is not None else mod("fake_capture")

    fake.analyze = analyze
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_capture", fake)
    return [("fake_capture", "Capture")]


def test_pipeline_isolates_a_module_that_returns_garbage(monkeypatch):
    modules = _install_fake_modules(monkeypatch)
    garbage = _capture_module(monkeypatch, [], returns="not a ModuleResult")
    report = pipeline.run_analysis([make_game()], "tester", modules=modules + garbage)
    assert [m.key for m in report.modules] == ["fake_good", "fake_bad", "fake_capture"]
    assert "could not be computed" in report.modules[2].summary
    assert [i.id for i in report.weaknesses] == ["fake.weakness.x"]


def test_pipeline_passes_options_and_sorts_ties_deterministically(monkeypatch):
    from datetime import datetime, timezone

    seen = []
    modules = _capture_module(monkeypatch, seen)
    t = datetime(2024, 5, 1, 12, tzinfo=timezone.utc)
    games = [make_game(game_id=f"g{i}", end_time=t) for i in (3, 1, 2)]
    pipeline.run_analysis(games, "tester", modules=modules, options={"tz": "Europe/Berlin"})
    pipeline.run_analysis(list(reversed(games)), "tester", modules=modules, options={"tz": "Europe/Berlin"})
    assert [g.game_id for g in seen[0].games] == [g.game_id for g in seen[1].games] == ["g1", "g2", "g3"]
    assert seen[0].options["tz"] == "Europe/Berlin"


def test_report_generated_at_is_timezone_aware_utc(monkeypatch):
    from datetime import timedelta

    report = pipeline.run_analysis([make_game()], "tester", modules=_capture_module(monkeypatch, []))
    assert report.generated_at.utcoffset() == timedelta(0)


def test_describe_filters_shows_the_last_included_day():
    from datetime import datetime, timezone

    from chess_insights.dataset import describe_filters

    text = describe_filters(since=datetime(2024, 1, 1, tzinfo=timezone.utc), until=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert "since 2024-01-01" in text and "until 2024-12-31" in text


def _nan_na_values():
    import math

    import pandas as pd

    return [None, pd.NA, pd.Series([pd.NA, pd.NA], dtype="Float64").mean(), math.nan, math.inf, "high", [0.5]]


@pytest.mark.parametrize("bad", _nan_na_values(), ids=["None", "pd.NA", "Float64-mean", "nan", "inf", "str", "list"])
@pytest.mark.parametrize("field", ["severity", "confidence"])
def test_one_insight_with_a_missing_score_does_not_sink_the_report(monkeypatch, bad, field):
    """severity/confidence None or pd.NA used to raise TypeError in ranking (no report at all);
    NaN even ranked first, because min(1.0, nan) is 1.0."""
    broken = ins("fake.weakness.broken", **{field: bad})
    good = ins("fake.weakness.good", severity=0.7)
    modules = _capture_module(monkeypatch, [], returns=mod("fake_capture", broken, good))
    report = pipeline.run_analysis([make_game()], "tester", modules=modules)
    assert [i.id for i in report.weaknesses] == ["fake.weakness.good"]
    assert [i.id for i in report.modules[0].insights] == ["fake.weakness.good"]
    assert report.study_plan and report.headline


@pytest.mark.parametrize(
    "change",
    [{"kind": "bogus"}, {"id": None}, {"title": None}, {"category": ["openings"]}, {"id": ""}],
    ids=["kind", "id-None", "title-None", "category-list", "id-empty"],
)
def test_malformed_insights_are_dropped_not_fatal(monkeypatch, change):
    broken = ins("fake.weakness.broken")
    for key, value in change.items():
        setattr(broken, key, value)
    items = [broken, ins("fake.weakness.good"), "not an insight"]
    modules = _capture_module(monkeypatch, [], returns=mod("fake_capture", *items))
    report = pipeline.run_analysis([make_game()], "tester", modules=modules)
    assert [i.id for i in report.weaknesses] == ["fake.weakness.good"]


def test_numpy_scores_are_kept_as_plain_floats(monkeypatch):
    import numpy as np

    item = ins("fake.weakness.np", severity=np.float64(0.6), confidence=np.float32(0.9))
    item.detail = None
    modules = _capture_module(monkeypatch, [], returns=mod("fake_capture", item))
    report = pipeline.run_analysis([make_game()], "tester", modules=modules)
    kept = report.weaknesses[0]
    assert type(kept.severity) is float and type(kept.confidence) is float and kept.detail == ""
