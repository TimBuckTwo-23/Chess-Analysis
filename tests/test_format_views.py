"""Per-format views (Report.format_reports), the formats and engine counts on every report, and the coaching wiring."""

import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from chess_insights import coach, pipeline
from chess_insights.coach import CoachConfig
from chess_insights.models import Coaching, Drill, Explanation, Insight, ModuleResult
from factories import make_game, make_game_eval

T0 = datetime(2026, 6, 1, 18, tzinfo=timezone.utc)
OUTCOMES = [("win", "win", "resigned"), ("loss", "resigned", "win"), ("draw", "agreed", "agreed")]


def multi_format_games(n_bullet=20, n_blitz=70, n_rapid=65):
    """Games of three formats, interleaved in time; bullet below MIN_FORMAT_GAMES."""
    games, k = [], 0
    for tc, n in (("bullet", n_bullet), ("blitz", n_blitz), ("rapid", n_rapid)):
        for i in range(n):
            outcome, mine, theirs = OUTCOMES[(i + k) % 3]
            games.append(make_game(game_id=f"{tc}-{i}", time_class=tc, outcome=outcome, my_result_code=mine,
                                   opp_result_code=theirs, end_time=T0 - timedelta(hours=3 * i + k)))
        k += 1
    return games


def capture(monkeypatch, seen, result=None):
    """A fake analysis module that records every context it is given."""
    fake = types.ModuleType("chess_insights.analysis.fake_views")

    def analyze(ctx):
        seen.append(ctx)
        return result(ctx) if result else ModuleResult(key="fake_views", title="Fake", summary="")

    fake.analyze = analyze
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_views", fake)
    return [("fake_views", "Fake")]


def test_min_format_games_is_documented_and_sensible():
    assert 30 <= pipeline.MIN_FORMAT_GAMES <= 200


def test_views_only_for_formats_with_enough_games_each_with_only_its_games(monkeypatch):
    seen = []
    games = multi_format_games()
    evals = {g.game_id: make_game_eval(g.game_id, [], engine="Stockfish 16", depth=12)
             for g in games if g.game_id in {"bullet-0", "bullet-1", "blitz-0", "blitz-1", "blitz-2"}}
    report = pipeline.run_analysis(games, "tester", evals=evals, modules=capture(monkeypatch, seen),
                                   filters="rated + casual, all time controls, standard chess",
                                   engine_note="Stockfish 16 at depth 12 on the 5 most recent games: 2 bullet, 3 blitz.")
    assert report.formats == {"bullet": 20, "blitz": 70, "rapid": 65}
    assert list(report.formats) == ["bullet", "blitz", "rapid"]  # format order
    assert report.engine_formats == {"bullet": 2, "blitz": 3}
    assert report.time_class == ""
    assert list(report.format_reports) == ["blitz", "rapid"]  # bullet has fewer than MIN_FORMAT_GAMES games

    blitz, rapid = report.format_reports["blitz"], report.format_reports["rapid"]
    assert blitz.time_class == "blitz" and blitz.n_games == 70 and blitz.formats == {"blitz": 70}
    assert blitz.engine_formats == {"blitz": 3} and rapid.engine_formats == {}
    assert blitz.filters == "blitz only, rated + casual, standard chess"
    assert blitz.engine_note == "Stockfish 16 at depth 12 on 3 of your 70 blitz games (the most recent)."
    assert "None of your rapid games were in the engine sample" in rapid.engine_note
    assert "--engine-sample balanced" in rapid.engine_note
    assert blitz.format_reports == {} and rapid.format_reports == {}

    main_ctx, *view_ctxs = seen  # the module ran on all games, then once per view
    assert len(main_ctx.games) == 155
    by_view = {ctx.games[0].time_class: ctx for ctx in view_ctxs}
    assert set(by_view) == {"blitz", "rapid"}
    for tc, ctx in by_view.items():
        assert {g.time_class for g in ctx.games} == {tc}
        assert set(ctx.evals) <= {g.game_id for g in ctx.games}
    assert set(by_view["blitz"].evals) == {"blitz-0", "blitz-1", "blitz-2"} and by_view["rapid"].evals == {}


def test_a_view_tests_its_claims_on_its_own_games(monkeypatch):
    """A module's finding in a view comes from that view's games: here a claim that exists only in rapid."""

    def result(ctx):
        insights = []
        if {g.time_class for g in ctx.games} == {"rapid"}:
            insights.append(Insight(id="fake.weakness.rapid", kind="weakness", category="results", title="Rapid only",
                                    detail="", severity=0.8, confidence=0.9, evidence={"n": len(ctx.games)}))
        return ModuleResult(key="fake_views", title="Fake", summary="", insights=insights)

    report = pipeline.run_analysis(multi_format_games(), "tester", modules=capture(monkeypatch, [], result))
    assert report.weaknesses == [] and report.format_reports["blitz"].weaknesses == []
    [claim] = report.format_reports["rapid"].weaknesses
    assert claim.formats == {"rapid": 65}  # filled in from the view's games


def test_no_views_for_one_format_or_when_switched_off(monkeypatch):
    modules = capture(monkeypatch, [])
    one = pipeline.run_analysis([g for g in multi_format_games() if g.time_class == "blitz"], "t", modules=modules)
    assert one.formats == {"blitz": 70} and one.format_reports == {}
    off = pipeline.run_analysis(multi_format_games(), "t", modules=modules, options={"format_views": False})
    assert off.format_reports == {} and off.formats == {"bullet": 20, "blitz": 70, "rapid": 65}


def test_views_with_the_real_modules_on_every_format():
    report = pipeline.run_analysis(multi_format_games(n_bullet=60, n_blitz=61, n_rapid=62), "tester",
                                   options={"tz": "Etc/UTC"})
    assert list(report.format_reports) == ["bullet", "blitz", "rapid"]
    for tc, view in report.format_reports.items():
        assert view.n_games == report.formats[tc] and [m.key for m in view.modules] == [m.key for m in report.modules]
        assert all(m.stats.get("error") is None for m in view.modules), [m.summary for m in view.modules]
        for ins in (i for m in view.modules for i in m.insights):
            assert set(ins.formats) == {tc}


def test_view_filters_replace_the_time_control_part():
    assert pipeline.view_filters("rapid", "rated, blitz+rapid, standard chess, since 2025-01-01") == (
        "rapid only, rated, standard chess, since 2025-01-01"
    )
    assert pipeline.view_filters("blitz", "") == "blitz only"


def test_engine_note_of_a_view_when_the_engine_did_not_run(monkeypatch):
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []),
                                   engine_note="Engine analysis not run (add --engine ...).",
                                   options={"engine_sample": "balanced"})
    assert all(v.engine_note == "Engine analysis not run (add --engine ...)." for v in report.format_reports.values())


# --------------------------------------------------------------------------- coaching wiring
def explanation(tc):
    return Explanation(epd=f"epd-{tc}", fen="8/8/8/8/8/8/8/8 w - - 0 1", played="1.e4", best="1.d4", best_line=None,
                       refutation=None, time_class=tc)


@pytest.fixture
def fake_coach(monkeypatch):
    calls = {"build": [], "finish": []}

    def build(ctx, modules, cfg):
        calls["build"].append((ctx, cfg))
        return Coaching(explanations=[explanation("blitz"), explanation("rapid"), explanation("blitz")],
                        notes=["Opening explorer skipped (no token)."],
                        drills=[Drill(theme="fork", title="30 fork puzzles", link="https://lichess.org/training/fork")])

    def finish(report, cfg):
        calls["finish"].append(report)
        report.coaching.notes.append("finished")
        return report

    monkeypatch.setattr(coach, "build_coaching", build)
    monkeypatch.setattr(coach, "finish_coaching", finish)
    return calls


def test_coaching_runs_once_on_all_games_and_views_get_their_explanations(monkeypatch, fake_coach):
    cfg = CoachConfig(depth=18)
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []), coach=cfg)
    [(ctx, used)] = fake_coach["build"]
    assert used is cfg and len(ctx.games) == 155
    assert fake_coach["finish"] == [report]
    assert len(report.coaching.explanations) == 3 and report.coaching.drills
    blitz = report.format_reports["blitz"].coaching
    assert [e.time_class for e in blitz.explanations] == ["blitz", "blitz"]
    assert blitz.drills == [] and blitz.review == [] and blitz.motif_profile is None
    assert "finished" in blitz.notes  # copied after finish_coaching
    assert [e.time_class for e in report.format_reports["rapid"].coaching.explanations] == ["rapid"]


def test_no_coaching_without_a_config(monkeypatch, fake_coach):
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []))
    assert report.coaching is None and fake_coach["build"] == []
    assert all(v.coaching is None for v in report.format_reports.values())


@pytest.mark.parametrize("where", ["build", "finish", "garbage"])
def test_a_failing_coaching_leaves_a_note_and_the_report_builds(monkeypatch, where):
    def boom(*a, **k):
        raise RuntimeError("no Stockfish 16")

    if where == "build":
        monkeypatch.setattr(coach, "build_coaching", boom)
    elif where == "garbage":
        monkeypatch.setattr(coach, "build_coaching", lambda ctx, modules, cfg: "not coaching")
    else:
        monkeypatch.setattr(coach, "build_coaching", lambda ctx, modules, cfg: Coaching())
        monkeypatch.setattr(coach, "finish_coaching", boom)
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []), coach=CoachConfig())
    assert report.coaching is not None and report.format_reports
    assert any("could not be computed" in n for n in report.coaching.notes)


def test_a_finding_added_by_the_coaching_is_checked_like_a_modules(monkeypatch):
    def build(ctx, modules, cfg):
        modules[0].insights.append(Insight(id="engine.weakness.motif-fork", kind="weakness", category="tactics",
                                           title="You miss forks", detail="", severity=0.5, confidence=0.9))
        modules[0].insights.append(Insight(id="broken", kind="bogus", category="tactics", title="x", detail="",
                                           severity=0.5, confidence=0.9))
        return Coaching()

    monkeypatch.setattr(coach, "build_coaching", build)
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []), coach=CoachConfig())
    assert [i.id for i in report.weaknesses] == ["engine.weakness.motif-fork"]


@pytest.mark.parametrize("new_signature", [False, True])
def test_practice_actions_with_either_signature(monkeypatch, fake_coach, new_signature):
    seen = []
    if new_signature:
        def practice(modules, coaching=None):
            seen.append(coaching)
            return ["Solve 30 fork puzzles"] if coaching and coaching.drills else []
    else:
        def practice(modules):
            seen.append("old")
            return []
    monkeypatch.setattr(pipeline, "practice_actions", practice)
    report = pipeline.run_analysis(multi_format_games(), "t", modules=capture(monkeypatch, []), coach=CoachConfig())
    assert len(seen) == 1 + len(report.format_reports)
    if new_signature:
        assert all(c is report.coaching for c in seen)  # the views route to the main report's drills
        assert any("fork" in a for item in report.format_reports["blitz"].study_plan for a in item.actions)
    else:
        assert seen == ["old"] * 3


def test_engine_note_of_a_view_whose_format_was_not_sent_to_stockfish(monkeypatch):
    """With --engine-time-class the hint to balance the sample would not help: say which formats were sent."""
    games = multi_format_games()
    evals = {g.game_id: make_game_eval(g.game_id, [], engine="Stockfish 16", depth=12)
             for g in games if g.game_id in {"blitz-0", "blitz-1"}}
    note = "Stockfish 16 at depth 12 on the 2 most recent blitz games: all blitz."
    report = pipeline.run_analysis(games, "t", evals=evals, modules=capture(monkeypatch, []), engine_note=note,
                                   options={"engine_sample": "recent", "engine_time_classes": ["blitz"]})
    rapid = report.format_reports["rapid"].engine_note
    assert rapid.startswith("None of your rapid games were in the engine sample (Stockfish 16")
    assert rapid.endswith("Only blitz games were sent to Stockfish.") and "--engine-sample" not in rapid
    assert report.format_reports["blitz"].engine_note == (
        "Stockfish 16 at depth 12 on 2 of your 70 blitz games (the most recent).")


def test_view_engine_note_wording():
    from chess_insights.context import AnalysisContext

    games = multi_format_games(n_bullet=0, n_blitz=3, n_rapid=2)
    evals = {g.game_id: make_game_eval(g.game_id, [], engine="Stockfish 16", depth=12) for g in games}
    whole = AnalysisContext("t", games, evals)
    blitz = AnalysisContext("t", [g for g in games if g.time_class == "blitz"],
                            {k: v for k, v in evals.items() if k.startswith("blitz")})
    assert pipeline.view_engine_note("blitz", blitz, whole, "x") == "Stockfish 16 at depth 12 on all 3 of your blitz games."
    empty = AnalysisContext("t", blitz.games, {})
    assert pipeline.view_engine_note("blitz", empty, whole, "") == (
        "None of your blitz games were in the engine sample. A balanced engine sample (--engine-sample balanced, the "
        "default on GitHub) takes games from every format.")
    daily = AnalysisContext("t", [make_game(time_class="daily")], {})
    whole.options["engine_sample"] = "balanced"
    assert pipeline.view_engine_note("daily", daily, whole, "SF.") == (
        "None of your daily games were in the engine sample (SF). A balanced sample takes daily games only when "
        "asked to (--engine-time-class).")
