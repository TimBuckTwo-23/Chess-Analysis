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

    from chess_insights.models import Diagram, Report
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


# --------------------------------------------------------------------------- review fixes
def _game(n, moves, errors, color="black", **kw):
    """Game ``n`` with an error (48 -> 33 win %) at each ply in ``errors`` (ply -> (played, engine move))."""
    game = make_game(color=color, moves_san=moves, game_id=f"g{n}", url=f"https://www.chess.com/game/live/{n}", **kw)
    user_parity = 0 if color == "white" else 1
    plies = []
    for i, s in enumerate(moves):
        mine = i % 2 == user_parity
        if i in errors:
            plies.append(make_ply_eval(ply=i, mover=color, is_user=True, san=s, best_san=errors[i],
                                       win_before=48.0, win_after=33.0, judgement="blunder"))
        else:
            plies.append(make_ply_eval(ply=i, mover=color if mine else "white", is_user=mine, san=s, best_san=s,
                                       win_before=50, win_after=50))
    return game, make_game_eval(game.game_id, plies)


LOOP = ["Nb8", "Ng1", "Nc6", "Nf3"]  # both sides go back and forth: the position after 3.Bc4 comes back


def test_an_error_repeated_within_one_game_counts_once():
    twice = _game(1, LINE + LOOP + ["Nd4", "Nxe5"], {5: "Nf6", 9: "Nf6"})
    once = _game(2, LINE + ["Nd4", "Nxe5"], {5: "Nf6"})
    ctx = _ctx([twice, once])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games)
    assert (rep.errors, rep.reached) == (2, 2)  # games, not moves
    [ins] = mistakes.analyze(ctx).insights
    assert "2 of the 2 games" in ins.detail
    assert len(ins.example_games) == len(set(ins.example_games)) == 2


def test_reach_counts_find_transpositions_and_later_arrivals():
    a = _game(1, LINE + ["Nd4", "Nxe5"], {5: "Nf6"})
    b = _game(2, ["Nf3", "Nc6", "e4", "e5", "Bc4", "Nd4", "Nxe5"], {5: "Nf6"})  # another move order
    c, _ = _game(3, LINE + LOOP + ["Nf6", "Ng5"], {})  # not analysed: reaches it at ply 9 as well as ply 5
    d, _ = _game(4, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], {})  # never gets there
    ctx = _ctx([a, b, (c, None), (d, None)])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games)
    assert (rep.errors, rep.reached) == (2, 3)
    assert rep.games_reached == [a[0].url, b[0].url, c.url]
    # reached only at ply 9, after both knights went out and back: later than any analysed error
    late, _ = _game(5, ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nb8", "Nb1", "Nc6", "Bc4", "Nf6"], {})
    ctx = _ctx([a, b, (late, None)])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games)
    assert rep.reached == 3


def test_the_engines_own_move_is_not_an_error():
    """A big drop after the engine's preferred move means the position was already bad: not a mistake."""
    g, ev = _game_with_error(1)
    ev.plies[5].best_san = ev.plies[5].san
    assert mistakes.collect_errors([g], {g.game_id: ev}) == []


def test_unreadable_start_positions_do_not_crash():
    broken = make_game(color="black", game_id="bad", initial_fen="not a fen", moves_san=LINE)
    ctx = _ctx([_game_with_error(1), _game_with_error(2), (broken, None)])
    res = mistakes.analyze(ctx)
    assert res.stats["repeated_positions"] == 1
    evals = dict(ctx.evals)
    evals["bad"] = make_game_eval("bad", [make_ply_eval(ply=1, mover="black", is_user=True, san="e5", best_san="c5",
                                                          win_before=50, win_after=30)])
    assert len(mistakes.collect_errors(ctx.games, evals)) == 2


def test_puzzles_from_a_set_up_position_with_black_to_move():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 17"
    game = make_game(color="black", initial_fen=fen, moves_san=["Nd4", "Nxe5", "Qg5", "Nxf7"], game_id="setup")
    ev = make_game_eval("setup", [
        make_ply_eval(ply=0, mover="black", is_user=True, san="Nd4", best_san="Nf6", win_before=48, win_after=30),
        make_ply_eval(ply=2, mover="black", is_user=True, san="Qg5", best_san="Qf6", win_before=40, win_after=20),
    ])
    puzzles = mistakes.build_puzzles([game], {"setup": ev})
    assert [p.move_label for p in puzzles] == ["18...Qg5", "17...Nd4"]  # worst first
    text = mistakes.puzzles_to_pgn(puzzles)
    pgn = io.StringIO(text)
    games = [chess.pgn.read_game(pgn) for _ in range(2)]
    assert chess.pgn.read_game(pgn) is None
    for g, p in zip(games, puzzles):
        assert g.headers["SetUp"] == "1" and g.headers["FEN"] == p.fen and g.headers["Result"] == "*"
        assert not g.errors
        [move] = list(g.mainline_moves())
        assert g.board().san(move) == p.best_san and g.board().turn == chess.BLACK
    assert "18... Qf6 *" in text and "17... Nf6 *" in text


def test_chess960_puzzles_carry_the_variant_and_castle_correctly():
    fen = "rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w HAha - 0 1"  # kings on b1 / b8
    game = make_game(color="white", rules="chess960", initial_fen=fen, moves_san=["a3", "a6"], game_id="c960")
    ev = make_game_eval("c960", [
        make_ply_eval(ply=0, mover="white", is_user=True, san="a3", best_san="O-O-O", win_before=55, win_after=40),
    ])
    [puzzle] = mistakes.build_puzzles([game], {"c960": ev})
    text = mistakes.puzzles_to_pgn([puzzle])
    assert '[Variant "Chess960"]' in text
    g = chess.pgn.read_game(io.StringIO(text))
    assert not g.errors and g.board().chess960
    [move] = list(g.mainline_moves())
    board = g.board()
    assert board.san(move) == "O-O-O"
    board.push(move)
    assert board.king(chess.WHITE) == chess.C1 and board.piece_at(chess.D1) == chess.Piece.from_symbol("R")
    svg = mistakes._svg(puzzle)
    assert svg.startswith("<svg")


def test_diagram_arrows_show_where_the_king_goes_when_castling():
    standard = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert mistakes._arrow(standard, standard.parse_san("O-O")) == (chess.E1, chess.G1)
    c960 = chess.Board("rk5r/pppppppp/8/8/8/8/PPPPPPPP/RK5R w HAha - 0 1", chess960=True)
    assert mistakes._arrow(c960, c960.parse_san("O-O-O")) == (chess.B1, chess.C1)  # not b1 -> a1 (the rook)
    assert mistakes._arrow(c960, c960.parse_san("O-O")) == (chess.B1, chess.G1)
    assert mistakes._arrow(c960, c960.parse_san("a3")) == (chess.A2, chess.A3)
