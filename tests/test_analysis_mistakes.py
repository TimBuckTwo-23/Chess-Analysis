import io

import chess
import chess.pgn

from chess_insights.analysis import mistakes
from chess_insights.context import AnalysisContext
from factories import make_game, make_game_eval, make_ply_eval

LINE = ["e4", "e5", "Nf3", "Nc6", "Bc4"]  # black to move after 3.Bc4


def _game_with_error(n: int, reply: str = "Nd4", error: bool = True):
    game = make_game(color="black", moves_san=LINE + [reply, "Nxe5", "Qg5"], game_id=f"g{n}",
                     url=f"https://www.chess.com/game/live/{n}")
    plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=i % 2 == 1, san=s,
                           best_san=s, win_before=50, win_after=50) for i, s in enumerate(game.moves_san)]
    if error:
        plies[5] = make_ply_eval(ply=5, mover="black", is_user=True, san=reply, best_san="Nf6",
                                 win_before=48.0, win_after=33.0, judgement="blunder")
    return game, make_game_eval(game.game_id, plies)


def _ctx(specs):
    games, evals = [], {}
    for g, ev in specs:
        games.append(g)
        if ev is not None:
            evals[g.game_id] = ev
    return AnalysisContext("tester", games, evals=evals)


def test_format_line_and_move_label():
    assert mistakes.format_line(["e4", "e5", "Nf3"]) == "1.e4 e5 2.Nf3"
    assert mistakes.format_line(LINE, last_n=2) == "… 2...Nc6 3.Bc4"
    assert mistakes.format_line(LINE, last_n=3) == "… 2.Nf3 Nc6 3.Bc4"
    assert mistakes.format_line([]) == ""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    assert mistakes.move_label(fen, "Nf6") == "3...Nf6"
    assert mistakes.move_label(chess.STARTING_FEN, "e4") == "1.e4"


def test_collect_errors_records_position_before_the_move():
    g, ev = _game_with_error(1)
    [e] = mistakes.collect_errors([g], {g.game_id: ev})
    assert (e.ply, e.san, e.best_san, round(e.drop)) == (5, "Nd4", "Nf6", 15)
    board = chess.Board()
    for s in LINE:
        board.push_san(s)
    assert e.fen == board.fen() and e.epd == board.epd()
    assert e.move_label == "3...Nd4" and e.best_label == "3...Nf6"


def test_repeated_mistake_detected_with_reach_count_and_insight():
    specs = [_game_with_error(1), _game_with_error(2), _game_with_error(3, reply="Nf6", error=False)]
    unanalysed, _ = _game_with_error(4, reply="Nf6", error=False)
    ctx = _ctx(specs + [(unanalysed, None)])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games)
    assert (rep.errors, rep.reached, rep.best_san) == (2, 4, "Nf6")

    res = mistakes.analyze(ctx)
    assert res.key == "mistakes"
    [ins] = res.insights
    assert ins.kind == "weakness" and ins.category == "openings"
    assert ins.title == "You have played 3...Nd4 here 2 times; 3...Nf6 is better"
    assert "2 of the 4 games" in ins.detail
    assert set(ins.example_games) == {"https://www.chess.com/game/live/1", "https://www.chess.com/game/live/2"}
    assert 0 < ins.severity <= 1 and 0 < ins.confidence <= 1
    [table] = res.tables
    assert len(table.rows[0]) == len(table.columns) == len(table.formats)
    assert table.rows[0][0] == "1.e4 e5 2.Nf3 Nc6 3.Bc4"
    [diagram] = res.diagrams
    assert diagram.svg.startswith("<svg") and diagram.fen == ins.evidence["fen"]
    assert res.stats["repeated_positions"] == 1


def test_single_errors_are_not_repeated_but_become_puzzles():
    ctx = _ctx([_game_with_error(1), _game_with_error(2, reply="Nf6", error=False)])
    res = mistakes.analyze(ctx)
    assert res.insights == [] and res.stats["puzzles"] == 1
    assert "No position where you erred more than once" in res.summary


def test_without_engine_analysis():
    res = mistakes.analyze(AnalysisContext("tester", [make_game()]))
    assert "--engine" in res.summary and not res.insights


def test_illegal_moves_do_not_crash():
    g, ev = _game_with_error(1)
    g.moves_san[2] = "Ke4"  # illegal
    assert mistakes.analyze(_ctx([(g, ev), _game_with_error(2)])).key == "mistakes"


def test_puzzles_pgn_round_trips():
    g, ev = _game_with_error(1)
    puzzles = mistakes.build_puzzles([g], {g.game_id: ev})
    text = mistakes.puzzles_to_pgn(puzzles)
    game = chess.pgn.read_game(io.StringIO(text))
    assert game.headers["SetUp"] == "1" and game.headers["Site"] == g.url
    assert game.board().fen() == puzzles[0].fen
    [move] = list(game.mainline_moves())
    assert game.board().san(move) == "Nf6"
    assert "3...Nd4" in game.comment


def test_diagrams_render_in_html_and_markdown():
    from datetime import datetime, timezone

    from chess_insights.models import Diagram, ModuleResult, Report
    from chess_insights.report import render_html, render_markdown
    from chess_insights.report.html import lichess_analysis_url

    res = mistakes.analyze(_ctx([_game_with_error(1), _game_with_error(2)]))
    evil = Diagram(title="<b>x</b>", fen="not a fen", svg='<svg><script>alert(1)</script></svg>', caption="c")
    res.diagrams.append(evil)
    report = Report(username="tester", generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc), filters="", n_games=2,
                    date_from=None, date_to=None, modules=[res], strengths=[], weaknesses=[], study_plan=[])
    html = render_html(report)
    assert html.count('<figure class="board-fig">') == 2
    assert "<svg" in html and "alert(1)" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    fen = res.diagrams[0].fen
    assert lichess_analysis_url(fen) == "https://lichess.org/analysis/" + fen.replace(" ", "_")
    assert lichess_analysis_url("not a fen") is None
    md = render_markdown(report)
    assert "analyse on Lichess" in md and fen.split()[0] in md
