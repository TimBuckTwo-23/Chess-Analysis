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


def test_a_view_tests_its_findings_on_its_own_games_and_claims_only_what_the_main_report_claims(monkeypatch):
    """A module's finding in a view comes from that view's games. One that is a claim only there (rapid) is shown
    as an observation in that view, card, chart and board kept; one the main report also claims stays a claim."""
    from chess_insights.models import Chart, Diagram, Series

    def result(ctx):
        insights = [Insight(id="fake.weakness.everywhere", kind="weakness", category="results", title="Everywhere",
                            detail="Seen in every format.", severity=0.8, confidence=0.9)]
        if {g.time_class for g in ctx.games} == {"rapid"}:
            insights.append(Insight(
                id="fake.weakness.rapid", kind="weakness", category="habits", title="Rapid only",
                detail="You score 40% here.", severity=0.8, confidence=0.9, evidence={"n": len(ctx.games)},
                study=["Stop playing rapid at night."], chart=Chart("bar", "Score", ["You"], [Series("You", [0.4])]),
                diagram=Diagram("A board", "8/8/8/8/8/8/8/8 w - - 0 1")))
        return ModuleResult(key="fake_views", title="Fake", summary="", insights=insights)

    report = pipeline.run_analysis(multi_format_games(), "tester", modules=capture(monkeypatch, [], result))
    assert [i.id for i in report.weaknesses] == ["fake.weakness.everywhere"]
    rapid = report.format_reports["rapid"]
    assert [i.id for i in rapid.weaknesses] == ["fake.weakness.everywhere"]  # the main report claims it too
    assert [i.id for i in report.format_reports["blitz"].weaknesses] == ["fake.weakness.everywhere"]
    shown = next(i for m in rapid.modules for i in m.insights if i.id == "fake.weakness.rapid")
    assert shown.kind == "observation" and shown.title == "Rapid only"
    assert shown.detail == ("Seen in your rapid games only, and not strong enough across all your games to call it a "
                            "finding: it may be chance. You score 40% here.")
    assert shown.chart is not None and shown.diagram is not None and shown.formats == {"rapid": 65}
    assert shown.evidence["observation_because"] == "view-only" and shown.evidence["n"] == 65
    # the section's summary, written while it was a claim, says so
    fake = next(m for m in rapid.modules if m.key == "fake_views")
    assert fake.summary == ("One finding below shows in your rapid games only, not across all your games, so it is "
                            "shown as an observation.")
    assert next(m for m in report.modules if m.key == "fake_views").summary == ""
    # the view's study plan and headline have no item for it
    assert all("fake.weakness.rapid" not in item.insight_ids for item in rapid.study_plan)
    assert "Rapid only" not in rapid.headline and not any("Rapid only" in line for line in rapid.summary_lines)


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


def test_every_game_link_gets_a_label_in_the_report_and_its_views(monkeypatch):
    """Findings' games and boards, section boards (and their strips) and tables, and the coaching's explanations
    (their game, the games they repeat in, their board) and endgame boards: every linked game gets its label, in
    the main report and in the view of its format."""
    from chess_insights.models import Diagram, Frame, Strip, Table

    games = multi_format_games(n_bullet=0)
    blitz = [g for g in games if g.time_class == "blitz"]
    rapid = [g for g in games if g.time_class == "rapid"]
    url = {name: g.url for name, g in zip(
        ["example", "ins_board", "module_board", "table", "exp_game", "exp_board", "endgame"], blitz)}
    url |= {"exp_repeat": rapid[0].url, "rapid_game": rapid[1].url}
    unlinked = blitz[-1].url

    def module(ctx):
        ids = {g.url for g in ctx.games}
        keep = lambda u: u if u in ids else ""  # noqa: E731 - a view's module only links its own games
        board = Diagram("Your move", "8/8/8/8/8/8/8/8 w - - 0 1", link=keep(url["ins_board"]),
                        strips=[Strip("Next", [Frame("8/8/8/8/8/8/8/8 w - - 0 1")])])
        finding = Insight("fake.observation.x", "observation", "results", "A finding", "", 0.1, 0.5,
                          example_games=[u for u in [keep(url["example"])] if u], diagram=board)
        return ModuleResult(
            key="fake_views", title="Fake", summary="", insights=[finding],
            diagrams=[Diagram("A board", "8/8/8/8/8/8/8/8 w - - 0 1", link=keep(url["module_board"]))],
            tables=[Table("Games", ["Game", "n"], [[keep(url["table"]), 1], ["not a link", 2]], ["url", "int"])],
        )

    def build(ctx, modules, cfg):
        exp_blitz = explanation("blitz")
        exp_blitz.game_url = url["exp_game"]
        exp_blitz.diagram = Diagram("", "8/8/8/8/8/8/8/8 w - - 0 1", link=url["exp_board"])
        exp_rapid = explanation("rapid")
        exp_rapid.game_url, exp_rapid.games = url["rapid_game"], [url["exp_repeat"]]
        return Coaching(explanations=[exp_blitz, exp_rapid],
                        endgame_diagrams=[Diagram("Rook ending", "8/8/8/8/8/8/8/8 w - - 0 1", link=url["endgame"])])

    monkeypatch.setattr(coach, "build_coaching", build)
    monkeypatch.setattr(coach, "finish_coaching", lambda report, cfg: report)
    report = pipeline.run_analysis(games, "t", modules=capture(monkeypatch, [], result=module), coach=CoachConfig())
    assert set(report.game_labels) == set(url.values())
    assert unlinked not in report.game_labels
    assert all(label.split(" · ")[0] in ("Win", "Loss", "Draw") for label in report.game_labels.values())
    blitz_view, rapid_view = report.format_reports["blitz"], report.format_reports["rapid"]
    assert set(blitz_view.game_labels) == {url[k] for k in ("example", "ins_board", "module_board", "table",
                                                             "exp_game", "exp_board")}
    assert set(rapid_view.game_labels) == {url["exp_repeat"], url["rapid_game"]}  # its explanation's games


def test_linked_games_reads_links_on_strips_and_frames_when_they_carry_one():
    from types import SimpleNamespace

    from chess_insights.models import Diagram

    frame = SimpleNamespace(link="https://www.chess.com/game/live/7")
    strip = SimpleNamespace(link="https://www.chess.com/game/live/8", frames=[frame])
    board = Diagram("x", "8/8/8/8/8/8/8/8 w - - 0 1", strips=[strip])  # type: ignore[list-item]
    found = pipeline.linked_games([ModuleResult("m", "M", "", diagrams=[board])])
    assert found == {"https://www.chess.com/game/live/7", "https://www.chess.com/game/live/8"}


# --------------------------------------------------------------------------- deeper-search verdicts
ITALIAN_5 = "r1bqkb1r/pppp1pp1/2n2n1p/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 2 5"
SCOTCH_7 = "r1b1k2r/ppppqppp/2n2n2/2b5/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 5 7"
HABIT_ID = "mistakes.weakness.00f8d505e4"


def habit_finding(kind="weakness"):
    """The mistakes module's repeated-mistake finding, worded as it words it."""
    return Insight(
        id=HABIT_ID.replace("weakness", kind), kind=kind, category="positions",
        title="Italian Game, move 5: you played 5.d3 in 8 of 8 games; 5.d4 is better",
        detail="After 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O h6 you played 5.d3 in 8 of the 8 games that reached this "
               "position. Stockfish analysed one of them: the move cost you about 9 in 100 of your chances to win; "
               "Stockfish prefers 5.d4. You go wrong on about 1 move in 40 overall, so playing this one move in 8 of "
               "8 games is a habit, not bad luck. In the same games you went on to play 7.Be3 (7.b4 is better): fix "
               "the first move and the rest goes with it.",
        severity=0.5, confidence=0.9,
        evidence={"fen": ITALIAN_5, "move": "d3", "best": "d4", "errors": 8, "reached": 8},
        study=["Set up the position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O h6 and work out why 5.d4 beats 5.d3.",
               "Add the line to your repertoire notes so you recognise the position next time."],
    )


def deep_explanation(verdict, *, played="5.d3", best="5.d4", fen=ITALIAN_5, insight_id=HABIT_ID, games=(), depth=16):
    from chess_insights.models import Line

    return Explanation(epd=" ".join(fen.split()[:4]), fen=fen, played=played, best=best,
                       best_line=Line(fen=fen, depth=depth), refutation=Line(fen=fen, depth=depth + 2),
                       insight_id=insight_id, kind="repeated", verdict=verdict, games=list(games))


@pytest.mark.parametrize("verdict, words", [
    ("close",
     "A deeper Stockfish search (depth 16) rates 5.d3 about as good as 5.d4, so it is not listed as a weakness"),
    ("fine", "A deeper Stockfish search (depth 16) makes 5.d3 its own first choice, so it is not listed as a weakness"),
])
def test_a_repeated_mistake_the_deeper_search_clears_is_an_observation(verdict, words):
    finding = habit_finding()
    modules = [ModuleResult("mistakes", "Positions", "", insights=[finding])]
    assert pipeline.apply_deep_verdicts(modules, Coaching(explanations=[deep_explanation(verdict)])) == [HABIT_ID]
    assert finding.kind == "observation" and finding.id == HABIT_ID  # the Why card still links to it
    assert finding.detail.startswith(words + "; the numbers below come from the quicker game analysis. After 1.e4")
    assert "habit, not bad luck" not in finding.detail and "Stockfish prefers 5.d4." in finding.detail
    assert finding.detail.endswith("Stockfish prefers 5.d4. In the same games you went on to play 7.Be3 (7.b4 is "
                                   "better).")
    assert finding.title == "Italian Game, move 5: you played 5.d3 in 8 of 8 games"  # "5.d4 is better" would contradict
    assert finding.study == ["Add the line to your repertoire notes so you recognise the position next time."]
    assert finding.evidence["observation_because"] == "deep-check"
    assert finding.evidence["deep_check"] == {"verdict": verdict, "depth": 16, "best": "5.d4", "played": "5.d3"}
    assert insights_ranked(modules) == ([], [])
    assert modules[0].summary == "A deeper Stockfish search clears 5.d3, so it is shown as an observation."


def test_a_repeated_mistake_seen_in_one_view_only_is_not_called_a_habit_there():
    finding = habit_finding()
    modules = [ModuleResult("mistakes", "Positions", "You repeated 2 moves, 2 of them a habit.", insights=[finding])]
    assert pipeline.view_only_observations(modules, {"mistakes.weakness.other"}, "blitz") == [HABIT_ID]
    assert finding.kind == "observation" and finding.title.endswith("; 5.d4 is better")  # the title stays
    assert finding.detail.startswith("Seen in your blitz games only, and not strong enough across all your games to "
                                     "call it a finding: it may be chance. After 1.e4")
    assert "habit, not bad luck" not in finding.detail and "fix the first move" in finding.detail
    assert modules[0].summary == ("You repeated 2 moves, 2 of them a habit. One finding below shows in your blitz "
                                  "games only, not across all your games, so it is shown as an observation.")
    two = [ModuleResult("m", "M", "", insights=[habit_finding(), Insight("x.strength.y", "strength", "results", "Y",
                                                                         "", 0.5, 0.9)])]
    assert len(pipeline.view_only_observations(two, set(), "rapid")) == 2
    assert two[0].summary.startswith("2 findings below show in your rapid games only") and two[0].summary.endswith(
        "so they are shown as observations.")
    assert pipeline.view_only_observations(two, set(), "rapid") == []  # observations stay as they are


def insights_ranked(modules):
    from chess_insights.insights import rank_insights

    return rank_insights(modules)


def test_only_a_deeper_verdict_on_the_findings_own_move_clears_it():
    finding = habit_finding()
    modules = [ModuleResult("mistakes", "Positions", "", insights=[finding])]
    coaching = Coaching(explanations=[
        deep_explanation("error"),  # the deeper search confirms the mistake
        deep_explanation("fine", played="7.Qd4", best="7.Qd4", fen=SCOTCH_7),  # a follow-up folded into the habit
        deep_explanation("fine", insight_id=None),  # no finding behind it
        deep_explanation("fine", insight_id="mistakes.weakness.another"),
    ])
    assert pipeline.apply_deep_verdicts(modules, coaching) == []
    assert finding.kind == "weakness" and finding.title.endswith("5.d4 is better")
    assert pipeline.apply_deep_verdicts(modules, None) == []


def test_the_depth_comes_from_the_lines_else_the_source_else_the_settings():
    from chess_insights.models import Source

    exp = deep_explanation("close")
    assert pipeline._deep_depth(exp, Coaching(settings={"depth": 20})) == 16
    exp.best_line = exp.refutation = None
    exp.sources = [Source(name="Stockfish 16, depth 18")]
    assert pipeline._deep_depth(exp, Coaching(settings={"depth": 20})) == 18
    exp.sources = []
    assert pipeline._deep_depth(exp, Coaching(settings={"depth": 20})) == 20
    assert pipeline._deep_depth(exp, Coaching()) is None
    assert pipeline.deep_verdict_sentence(exp, "weakness", None).startswith(
        "A deeper Stockfish search rates 5.d3 about as good as 5.d4")


def habit_module(monkeypatch):
    """A fake mistakes section with the 5.d3 habit in every run (main report and each view)."""
    def result(ctx):
        return ModuleResult(key="mistakes", title="Positions", summary="", insights=[habit_finding()])

    fake = types.ModuleType("chess_insights.analysis.fake_mistakes")
    fake.analyze = result
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_mistakes", fake)
    return [("fake_mistakes", "Positions")]


def test_deeper_verdicts_apply_before_ranking_in_the_report_and_every_view(monkeypatch):
    games = multi_format_games()
    urls = [g.url for g in games if g.time_class == "blitz"][:3]
    monkeypatch.setattr(coach, "build_coaching", lambda ctx, modules, cfg: Coaching(
        explanations=[deep_explanation("close", games=urls)], settings={"depth": 16}))
    monkeypatch.setattr(coach, "finish_coaching", lambda report, cfg: report)
    report = pipeline.run_analysis(games, "t", modules=habit_module(monkeypatch), coach=CoachConfig())
    for view in [report, *report.format_reports.values()]:
        assert view.weaknesses == [] and all(HABIT_ID not in item.insight_ids for item in view.study_plan)
        [finding] = [i for m in view.modules for i in m.insights]
        assert finding.kind == "observation" and finding.detail.startswith("A deeper Stockfish search (depth 16)")
        assert "5.d3" not in view.headline
    # the blitz view keeps the explanation (its games are blitz games), linked to the view's own finding
    assert [e.insight_id for e in report.format_reports["blitz"].coaching.explanations] == [HABIT_ID]


# --------------------------------------------------------------------------- a view's explanations
def test_a_view_keeps_the_explanations_of_its_format_its_games_and_its_findings():
    games = multi_format_games(n_bullet=0)
    blitz = [g.url for g in games if g.time_class == "blitz"]
    rapid = [g.url for g in games if g.time_class == "rapid"]
    both = deep_explanation("error", played="7.Qd4", best="7.Qe2", fen=SCOTCH_7, insight_id=None,
                            games=[rapid[0], blitz[0]])  # repeated in two formats: time_class ""
    rapid_only = deep_explanation("error", played="7.Qd4", best="7.Qe2", fen=SCOTCH_7, insight_id=None,
                                  games=[rapid[1]])
    own = explanation("blitz")
    # the main report calls the 5.d3 habit an observation; the blitz view's own test calls it a weakness
    habit = deep_explanation("error", insight_id="mistakes.observation.00f8d505e4")
    elsewhere = deep_explanation("error", played="5.d4", insight_id="mistakes.observation.00f8d505e4")  # other move
    coaching = Coaching(explanations=[both, rapid_only, own, habit, elsewhere], notes=["n"], settings={"depth": 16})
    modules = [ModuleResult("mistakes", "Positions", "", insights=[habit_finding()])]

    view = pipeline.view_coaching(coaching, "blitz", games, modules)
    assert [e.insight_id for e in view.explanations[:1]] == [HABIT_ID]  # the view's finding's own id, first
    assert view.explanations[1:] == [both, own]
    assert habit.insight_id == "mistakes.observation.00f8d505e4"  # the main report's copy is untouched
    assert view.notes == ["n"] and view.settings == {"depth": 16} and view.drills == []
    rapid_view = pipeline.view_coaching(coaching, "rapid", games)
    assert rapid_view.explanations == [both, rapid_only]
    assert pipeline.view_coaching(coaching, "blitz").explanations == [own]  # without the games: time class only
    assert pipeline.view_coaching(None, "blitz") is None


# --------------------------------------------------------------------------- a view's puzzles
def test_a_views_puzzle_action_says_how_many_of_the_files_puzzles_are_its_formats(monkeypatch):
    games = multi_format_games(n_bullet=0)
    evals = {g.game_id: make_game_eval(g.game_id, []) for g in games}

    def result(ctx):  # the mistakes section's puzzle count: the file's in the main report, the format's in a view
        n = 84 if len({g.time_class for g in ctx.games}) > 1 else {"blitz": 63, "rapid": 21}[ctx.games[0].time_class]
        return ModuleResult(key="mistakes", title="Positions", summary="",
                            stats={"puzzles_exported": n, "puzzle_file": "me-puzzles.pgn"})

    fake = types.ModuleType("chess_insights.analysis.fake_puzzles")
    fake.analyze = result
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake_puzzles", fake)
    from chess_insights.analysis import mistakes

    # the file: the 84 costliest over all formats (two of the rapid view's 21 are not among them)
    in_file = ([g for g in games if g.time_class == "blitz"][:63] + [g for g in games if g.time_class == "rapid"][:19]
               + [make_game(time_class="bullet")] * 2)
    monkeypatch.setattr(mistakes, "build_puzzles", lambda gs, ev: [types.SimpleNamespace(game=g) for g in in_file])
    report = pipeline.run_analysis(games, "t", evals=evals, modules=[("fake_puzzles", "Positions")],
                                   options={"puzzle_file": "me-puzzles.pgn"})
    main = [a for item in report.study_plan for a in item.actions]
    assert any(a.startswith("Solve 10 of your own puzzles a day: 84 positions from your games") for a in main)
    [item] = report.format_reports["blitz"].study_plan
    assert item.actions == [
        "Solve 10 of your own puzzles a day: 63 of the 84 puzzles in me-puzzles.pgn are from your blitz games "
        "(positions where your move cost a lot, costliest first). Import the file into a Lichess study or any chess "
        "program."]
    assert item.target == "Solve each of the 63 blitz puzzles in the file once before your next report."
    assert "19 of the 84 puzzles" in report.format_reports["rapid"].study_plan[0].actions[0]


def test_puzzle_file_formats_needs_the_file_and_the_engine(monkeypatch):
    from chess_insights.analysis import mistakes
    from chess_insights.context import AnalysisContext

    games = multi_format_games(n_bullet=0, n_blitz=2, n_rapid=1)
    evals = {g.game_id: make_game_eval(g.game_id, []) for g in games}
    monkeypatch.setattr(mistakes, "build_puzzles", lambda gs, ev: [types.SimpleNamespace(game=g) for g in gs])
    assert pipeline.puzzle_file_formats(AnalysisContext("t", games, evals)) is None  # no --puzzles
    assert pipeline.puzzle_file_formats(AnalysisContext("t", games, {}, {"puzzle_file": "x.pgn"})) is None
    assert pipeline.puzzle_file_formats(AnalysisContext("t", games, evals, {"puzzle_file": "x.pgn"})) == {
        "blitz": 2, "rapid": 1}


def test_a_cleared_finding_board_loses_the_quick_checks_green_arrow():
    from chess_insights import visuals

    finding = habit_finding()
    finding.diagram = visuals.position_diagram(
        "Italian Game: 5.d3", ITALIAN_5, played="d3", best="d4",
        caption="After 1.e4 e5. Red: your move (in 8 of 8 games). Green: engine's choice.")
    modules = [ModuleResult("mistakes", "Positions", "", insights=[finding], diagrams=[finding.diagram])]
    pipeline.apply_deep_verdicts(modules, Coaching(explanations=[deep_explanation("fine")]))
    assert [a.kind for a in finding.diagram.arrows] == ["played"]
    assert "Green" not in finding.diagram.caption
    assert finding.diagram.caption.endswith("A deeper Stockfish check at depth 16 rates 5.d3 its own first choice.")
    assert modules[0].diagrams[0] is finding.diagram  # the section's copy of the board says the same
