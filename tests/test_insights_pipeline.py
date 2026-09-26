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


def test_study_plan_groups_one_cause_into_one_item_with_at_most_three_actions():
    w = [
        ins("a", category="tactics", study=["Review your 5 hung-piece games"], example_games=["https://x/1"] * 7),
        ins("b", category="tactics", severity=0.5, study=[]),
    ]
    [item] = insights.build_study_plan(w)  # one cause: one item
    assert item.title == "title a" and item.findings == ["title a", "title b"] and item.insight_ids == ["a", "b"]
    assert item.actions[0] == "Review your 5 hung-piece games"
    assert 1 < len(item.actions) <= 3 and len(set(item.actions)) == len(item.actions)  # generic tips fill the rest
    assert item.games == ["https://x/1"]
    assert item.category.startswith("Tactics")


def test_study_plan_orders_items_by_effort_not_size():
    plan = insights.build_study_plan([
        ins("open", category="openings", severity=0.9, study=["Learn the Italian", "Replay the losses"]),
        ins("clock", category="time", severity=0.5, study=["Budget your clock", "Play with increment"]),
        ins("tilt", category="habits", severity=0.3, study=["Stop after a loss", "Take a break"]),
        ins("colour", category="color", severity=0.4, study=["Narrow your Black repertoire", "Replay"]),
    ])
    assert [p.insight_ids[0] for p in plan] == ["tilt", "clock", "open"]  # habits, clock, then the repertoire
    repertoire = plan[2]
    assert repertoire.insight_ids == ["open", "colour"]  # one repertoire job
    # the findings' own actions in turn: the first of each, then the second ...
    assert repertoire.actions == ["Learn the Italian", "Narrow your Black repertoire", "Replay the losses"]


def test_generic_tips_that_repeat_a_findings_own_action_are_left_out():
    w = [ins("late", category="habits", study=[
        "Avoid rated games between 00:00 and 04:00; play puzzles or unrated games then instead.",
    ])]
    [item] = insights.build_study_plan(w)
    assert item.actions[0].startswith("Avoid rated games between")
    assert not any("when tired" in a for a in item.actions)
    assert insights.similar_actions(
        "Replay your 5 most recent losses in the Ruy Lopez Opening as White and mark the move where you left known theory.",
        "Replay your losses in this opening and mark the exact move where you left known theory.",
    )
    assert not insights.similar_actions("Budget your clock", "Learn the Caro-Kann plans")


def test_practice_item_from_puzzles_even_without_weaknesses():
    mistakes_module = ModuleResult(
        key="mistakes", title="Positions", summary="", stats={"puzzles_exported": 40, "puzzle_file": "me-puzzles.pgn"}
    )
    actions = insights.practice_actions([mistakes_module])
    assert actions and "40 positions" in actions[0] and "me-puzzles.pgn" in actions[0]
    [item] = insights.build_study_plan([], practice=actions)
    assert item.title == "Practise your own mistakes as puzzles" and item.actions == actions and not item.insight_ids
    # with a repeated-position weakness, the puzzles join that item instead of making a second one
    rep = ins("mistakes.weakness.x", category="positions", study=["Set up the position", "Add it to notes", "Replay"])
    [item] = insights.build_study_plan([rep], practice=actions)
    assert item.insight_ids == ["mistakes.weakness.x"] and item.actions[-1] == actions[0] and len(item.actions) == 3
    assert insights.practice_actions([mod("mistakes")]) == []


def test_targets_carry_todays_number():
    slow = ins("time.weakness.slow-opening-blitz", category="time",
               evidence={"time_class": "blitz", "your_share": 0.5, "opponent_share": 0.26})
    text, baseline = insights.target_for(slow)
    assert text == "Use at most 30% of your clock on your first 15 moves in blitz (now 50%)."
    assert baseline["value"] == 0.5
    opening = ins("openings.weakness.white.ruy-lopez-opening",
                  evidence={"family": "Ruy Lopez Opening", "color": "white", "delta": -0.09})
    assert insights.target_for(opening)[0] == "Score in line with your rating in the Ruy Lopez Opening (now −9 per 100 games)."
    late = ins("habits.weakness.time-of-day-00-04", category="habits", evidence={"block": "00:00–04:00", "n": 100})
    assert insights.target_for(late)[0] == "No rated games started 00:00–04:00 (now 100 of your rated games)."
    assert insights.target_for(ins("x.weakness.y")) == ("", {})
    [item] = insights.build_study_plan([slow])
    assert item.target.startswith("Use at most 30%") and item.baseline["insight"] == slow.id


def test_rank_can_keep_every_claim():
    many = [ins(f"o{i}", category="openings", severity=0.9 - i * 0.01) for i in range(8)]
    _, top = insights.rank_insights([mod("a", *many)])
    _, everything = insights.rank_insights([mod("a", *many)], top_n=None)
    assert len(top) == 5 and [i.id for i in everything] == [f"o{i}" for i in range(8)]


def test_summary_lines_give_ratings_first_steps_and_a_strength():
    results = ModuleResult(key="results", title="Results", summary="", stats={"by_time_control": {
        "Blitz": {"rating": {"n": 221, "start": 1458, "current": 1379}},
        "Rapid": {"rating": {"n": 114, "start": 1531, "current": 1756}},
        "Bullet": {"rating": {"n": 5, "start": 1200, "current": 1268}},  # too few games to mention
    }})
    plan = insights.build_study_plan([ins("w", category="time")])
    lines = insights.summary_lines([results], plan, [ins("s", kind="strength")], [ins("w")])
    assert lines == [
        "Your ratings: blitz 1379 (−79 over these games), rapid 1756 (+225).",
        "Start with: 1. title w.",
        "Strength to build on: title s.",
    ]


def test_report_lists_every_claim_labels_games_and_marks_a_demo(monkeypatch):
    from datetime import datetime, timezone

    games = [make_game(game_id=f"g{i}", url=f"https://www.chess.com/game/live/{i}", outcome="loss",
                       end_time=datetime(2024, 5, 1 + i, tzinfo=timezone.utc)) for i in range(3)]
    many = [ins(f"fake.weakness.{i}", category="openings", severity=0.9 - i * 0.05,
                example_games=[g.url for g in games]) for i in range(7)]
    modules = _capture_module(monkeypatch, [], returns=mod("fake_capture", *many))
    report = pipeline.run_analysis(games, "tester", modules=modules, options={"demo": True})
    assert len(report.weaknesses) == 7  # nothing hidden beyond a top 5
    assert report.demo is True
    label = report.game_labels["https://www.chess.com/game/live/0"]
    assert label.startswith("Loss · ") and label.endswith("1 May")


def test_headline():
    assert "No clear patterns" in insights.headline([], [])
    h = insights.headline([ins("s", kind="strength")], [ins("w")])
    assert h.startswith("Biggest opportunity: title w.") and "Strength to build on: title s." in h
    plan = insights.build_study_plan([ins("w")])
    assert insights.headline([], [ins("w")], plan) == "Start with: title w."


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
    assert report.study_plan and report.headline.startswith("Start with")
    assert report.n_games == 2 and report.date_from <= report.date_to
    assert report.filters == "rated blitz"


def test_pipeline_with_no_games(monkeypatch):
    modules = _install_fake_modules(monkeypatch)
    report = pipeline.run_analysis([], "tester", modules=modules)
    assert report.n_games == 0 and report.date_from is None


def test_default_module_list_names_real_modules():
    names = [n for n, _ in pipeline.MODULES]
    assert names == ["results", "openings", "time_mgmt", "endings", "habits", "structure", "engine_stats", "mistakes"]


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


def test_same_opening_found_by_two_modules_is_merged_in_the_headline():
    a = ins("openings.weakness.black.caro-kann-defense", severity=0.6, study=["Replay your Caro-Kann losses"],
            example_games=["https://x/1"])
    b = ins("engine.weakness.openings-black-caro-kann-defense", severity=0.8, study=["Learn the Advance line"],
            example_games=["https://x/2"])
    other_colour = ins("openings.weakness.white.caro-kann-defense", severity=0.5)
    _, weaknesses = insights.rank_insights([mod("openings", a, other_colour), mod("engine", b)])
    merged = next(i for i in weaknesses if "black" in i.id)
    assert [i.id for i in weaknesses].count(merged.id) == 1 and len(weaknesses) == 2
    assert merged.id == b.id and merged.study == ["Learn the Advance line", "Replay your Caro-Kann losses"]
    assert merged.example_games == ["https://x/2", "https://x/1"]
    assert merged.evidence["also_found_by"] == [a.id]
    assert b.study == ["Learn the Advance line"] and "also_found_by" not in b.evidence  # module findings untouched


def test_hanging_piece_motif_and_hung_material_findings_are_one_topic():
    from chess_insights.insights import target_for, topic_key
    from chess_insights.models import Insight

    motif = Insight(id="tactics.weakness.motif-allowed.hangingPiece", kind="weakness", category="tactics",
                    title="t", detail="", severity=0.5, confidence=0.9,
                    evidence={"theme": "hangingPiece", "motif_kind": "allowed", "per100": 1.24, "opp_per100": 0.62})
    engine = Insight(id="engine.weakness.hung-material", kind="weakness", category="tactics", title="t", detail="",
                     severity=0.4, confidence=0.9)
    assert topic_key(motif) == topic_key(engine) == "hung-material:weakness"
    fork = Insight(id="tactics.weakness.motif-missed.fork", kind="weakness", category="tactics", title="t",
                   detail="", severity=0.5, confidence=0.9,
                   evidence={"theme": "fork", "motif_kind": "missed", "per100": 1.24, "opp_per100": 0.62})
    assert topic_key(fork) == fork.id
    sentence, baseline = target_for(fork)
    assert sentence == "Miss no more forks than your opponents: 0.62 per 100 moves (now 1.24)."
    assert baseline == {"metric": "forks missed per 100 moves", "value": 1.24}
    assert target_for(motif)[0].startswith("Allow no more hanging pieces than your opponents")


def test_a_format_views_own_puzzles_point_at_its_share_of_the_one_puzzle_file():
    """The --puzzles file holds every format's puzzles, costliest first: a view says how many are its format's."""
    stats = {"puzzles_exported": 63, "puzzle_file": "me-puzzles.pgn"}
    modules = [ModuleResult(key="mistakes", title="Positions", summary="", stats=stats)]
    counts = {"bullet": 100, "blitz": 63, "rapid": 21}
    [action] = actions = insights.practice_actions(modules, time_class="blitz", puzzle_file_formats=counts)
    assert action.startswith("Solve 10 of your own puzzles a day: 63 of the 184 puzzles in me-puzzles.pgn are from "
                             "your blitz games")
    assert actions.target == "Solve each of the 63 blitz puzzles in the file once before your next report."
    [item] = insights.build_study_plan([], practice=actions)
    assert item.target == actions.target and item.actions == [action]
    # none of the file's puzzles from this format (the file keeps the costliest 300): nothing to point at
    assert insights.practice_actions(modules, time_class="daily", puzzle_file_formats=counts) == []
    # counts unknown: the file is named as every format's
    [unknown] = insights.practice_actions(modules, time_class="blitz")
    assert "me-puzzles.pgn: it holds your costliest mistakes in every format, not only blitz" in unknown
    # no file: the view's own count, and how to export them
    no_file = [ModuleResult(key="mistakes", title="Positions", summary="", stats={"puzzles_exported": 63})]
    [export] = insights.practice_actions(no_file, time_class="blitz", puzzle_file_formats=counts)
    assert export.startswith("Solve 10 of your own puzzles a day: 63 positions from your blitz games where your move "
                             "cost a lot (export them with --puzzles).")
    # the main report is unchanged
    [main] = insights.practice_actions(modules)
    assert main.startswith("Solve 10 of your own puzzles a day: 63 positions from your games where your move cost a "
                           "lot, in me-puzzles.pgn.")
    [item] = insights.build_study_plan([], practice=insights.practice_actions(modules))
    assert item.target == "Solve every puzzle in the file once before your next report."
