import io
import math

import chess
import chess.pgn
import pytest

from chess_insights.analysis import mistakes
from chess_insights.context import AnalysisContext
from chess_insights.insights import MIN_CONFIDENCE, headline, rank_insights
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
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals)
    assert (rep.san, rep.errors, rep.reached, rep.analysed, rep.best_san) == ("Nd4", 2, 4, 2, "Nf6")

    res = mistakes.analyze(ctx)
    assert res.key == "mistakes"
    [ins] = res.insights
    # two games are not enough to call it a habit: shown, but as an observation that never ranks
    assert ins.kind == "observation" and ins.category == "positions" and ins.confidence < MIN_CONFIDENCE
    assert ins.title.endswith(": you played 3...Nd4 in 2 of 4 games; 3...Nf6 is better")
    assert "in 2 of the 4 games" in ins.detail and "Too few games" in ins.detail
    assert set(ins.example_games) == {"https://www.chess.com/game/live/1", "https://www.chess.com/game/live/2"}
    assert 0 < ins.severity <= 1
    table, puzzles = res.tables
    assert puzzles.title == "Your costliest mistakes (puzzles)" and len(puzzles.rows) == 2
    assert all(row[4].startswith("https://lichess.org/analysis/") for row in puzzles.rows)
    assert len(table.rows[0]) == len(table.columns) == len(table.formats)
    assert table.rows[0][0] == "1.e4 e5 2.Nf3 Nc6 3.Bc4"
    assert table.rows[0][2:6] == [4, 2, "3...Nd4", "3...Nf6"]
    [diagram] = res.diagrams
    assert diagram.svg.startswith("<svg") and diagram.fen == ins.evidence["fen"]
    assert res.stats["repeated_positions"] == 1 and res.stats["claims"] == 0


def test_single_errors_are_not_repeated_but_become_puzzles():
    ctx = _ctx([_game_with_error(1), _game_with_error(2, reply="Nf6", error=False)])
    res = mistakes.analyze(ctx)
    assert res.insights == [] and res.stats["puzzles"] == 1
    assert "No wrong move you played in more than one game" in res.summary
    assert "One of your mistakes is available as a puzzle." in res.summary


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
    twice = _game(1, LINE + LOOP + ["Nd4", "Nxe5"], {5: "Nf6", 9: "Nf6"})  # 3...Nb8 and, back there, 5...Nd4
    once = _game(2, LINE + ["Nd4", "Nxe5"], {5: "Nf6"})
    ctx = _ctx([twice, once])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals)
    assert (rep.san, rep.errors, rep.reached) == ("Nd4", 2, 2)  # games, not moves; ...Nb8 was played once
    [ins] = mistakes.analyze(ctx).insights
    assert "in 2 of the 2 games" in ins.detail
    assert len(ins.example_games) == len(set(ins.example_games)) == 2


def test_reach_counts_find_transpositions_and_later_arrivals():
    a = _game(1, LINE + ["Nd4", "Nxe5"], {5: "Nf6"})
    b = _game(2, ["Nf3", "Nc6", "e4", "e5", "Bc4", "Nd4", "Nxe5"], {5: "Nf6"})  # another move order
    c, _ = _game(3, LINE + LOOP + ["Nf6", "Ng5"], {})  # not analysed: reaches it at ply 9 as well as ply 5
    d, _ = _game(4, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], {})  # never gets there
    ctx = _ctx([a, b, (c, None), (d, None)])
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals)
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


# --------------------------------------------------------------------------- verifier findings: repeated mistakes
def test_different_one_off_errors_in_one_position_are_not_a_repeated_mistake():
    """Two unrelated errors (3...Nd4 once, 3...d6 once) where you play the engine's 3...Nf6 in 38 of 40 games:
    nothing is repeated, so nothing is claimed (it used to headline the report as 'played 3...d6 here 1 times')."""
    specs = [_game_with_error(1, "Nd4"), _game_with_error(2, "d6")]
    specs += [_game_with_error(n, "Nf6", error=False) for n in range(3, 41)]
    ctx = _ctx(specs)
    assert mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals) == []
    res = mistakes.analyze(ctx)
    assert res.insights == [] and res.stats["repeated_positions"] == 0
    assert [t.title for t in res.tables] == ["Your costliest mistakes (puzzles)"]  # the two errors, as puzzles
    strengths, weaknesses = rank_insights([res])
    assert weaknesses == [] and headline(strengths, weaknesses).startswith("No clear patterns")


def test_a_wrong_move_you_rarely_play_there_is_not_a_habit():
    """The same wrong move in 3 of 40 games, the engine's move in the other 37: at an ordinary error rate
    that is chance, so an observation at most, never a weakness."""
    # each of these games also has a one-off error later on (a different queen move each time)
    specs = [_game(n, LINE + ["Nd4", "Nxe5", queen], {5: "Nf6", 7: "Qh4"})
             for n, queen in enumerate(["Qg5", "Qf6", "Qe7"], 1)]
    specs += [_game_with_error(n, "Nf6", error=False) for n in range(4, 41)]
    ctx = _ctx(specs)
    [rep] = mistakes.assess(
        mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals),
        *mistakes.error_rate(ctx.games, ctx.evals),
    )
    assert (rep.errors, rep.reached) == (3, 40)
    assert rep.p_value > 0.01 and not rep.significant
    res = mistakes.analyze(ctx)
    assert [i.kind for i in res.insights] == ["observation"]
    assert rank_insights([res]) == ([], [])


def test_a_wrong_move_you_keep_playing_is_a_weakness():
    """3...Nd4 in 4 of the 5 games that reached the position (the engine's move once): a habit."""
    specs = [_game_with_error(n) for n in range(1, 5)] + [_game_with_error(5, "Nf6", error=False)]
    specs += [_game(n, ["d4", "d5", "c4", "e6", "Nc3", "Nf6"], {}, color="black") for n in range(6, 20)]
    ctx = _ctx(specs)
    res = mistakes.analyze(ctx)
    [ins] = res.insights
    assert ins.kind == "weakness" and ins.confidence >= MIN_CONFIDENCE
    assert ins.title.endswith(": you played 3...Nd4 in 4 of 5 games; 3...Nf6 is better")
    assert "habit" in ins.detail and ins.evidence["p_adjusted"] <= 0.01
    assert ins.id.startswith("mistakes.weakness.")
    _, weaknesses = rank_insights([res])
    assert [w.id for w in weaknesses] == [ins.id]
    assert res.stats["claims"] == 1


def test_unanalysed_games_with_the_same_move_count_as_repeats():
    """Reached and played-it come from the same games: 3...Nd4 in all 10 games, only 2 of them engine-analysed,
    is 10 of 10, not '2 of the 10 games that reached this position'."""
    games, evals = [], {}
    for n in range(1, 11):
        g, ev = _game_with_error(n)
        games.append(g)
        if n <= 2:
            evals[g.game_id] = ev
    other, _ = _game_with_error(11, "Nf6", error=False)  # reached it, played the right move, not analysed
    games.append(other)
    ctx = AnalysisContext("tester", games, evals=evals)
    [rep] = mistakes.find_repeated(mistakes.collect_errors(ctx.games, ctx.evals), ctx.games, ctx.evals)
    assert (rep.errors, rep.reached, rep.analysed) == (10, 11, 2)
    [ins] = mistakes.analyze(ctx).insights
    assert ins.kind == "weakness"
    assert ins.title.endswith(": you played 3...Nd4 in 10 of 11 games; 3...Nf6 is better")
    assert "Stockfish analysed 2 of them" in ins.detail
    assert ins.example_games[:2] == [games[1].url, games[0].url]  # analysed games first, most recent first


def test_a_move_the_engine_does_not_consistently_flag_is_left_out():
    """Flagged once (15 points) but a 2-point loss in two other analysed games: on average not an error."""
    specs = [_game_with_error(1)]
    for n in (2, 3):
        g, ev = _game_with_error(n)
        ev.plies[5].win_after = 46.0
        ev.plies[5].judgement = None
        specs.append((g, ev))
    ctx = _ctx(specs)
    events = mistakes.collect_errors(ctx.games, ctx.evals)
    assert len(events) == 1
    assert mistakes.find_repeated(events, ctx.games, ctx.evals) == []


def test_binomial_tail():
    assert mistakes.binomial_sf(0, 5, 0.3) == 1.0
    assert mistakes.binomial_sf(6, 5, 0.3) == 0.0
    assert mistakes.binomial_sf(2, 2, 0.1) == pytest.approx(0.01)
    assert mistakes.binomial_sf(3, 40, 0.025) == pytest.approx(1 - sum(
        math.comb(40, j) * 0.025**j * 0.975 ** (40 - j) for j in range(3)))


def test_puzzle_kpi_counts_what_is_exported(monkeypatch):
    monkeypatch.setattr(mistakes, "PUZZLE_LIMIT", 2)
    ctx = _ctx([_game_with_error(n, reply) for n, reply in enumerate(["Nd4", "d6", "Bc5", "h6"], 1)])
    res = mistakes.analyze(ctx)
    kpi = next(k for k in res.kpis if k.label == "Puzzles from your games")
    assert kpi.value == 2 and "the 2 costliest of 4" in kpi.hint
    assert res.stats["puzzles"] == 4 and res.stats["puzzles_exported"] == 2
    monkeypatch.setattr(mistakes, "PUZZLE_LIMIT", 300)
    monkeypatch.setattr(mistakes, "LICHESS_STUDY_CHAPTERS", 3)
    kpi = next(k for k in mistakes.analyze(ctx).kpis if k.label == "Puzzles from your games")
    assert kpi.value == 4 and "a Lichess study holds 3" in kpi.hint
    assert len(mistakes.build_puzzles(ctx.games, ctx.evals)) == 4


def test_a_follow_up_mistake_in_the_same_games_is_folded_into_the_first():
    from types import SimpleNamespace as NS

    games = [NS(game_id=g) for g in ("a", "b", "c")]

    def rep(ply, weight, played, significant=True):
        return NS(first=NS(ply=ply), played_in=played, weight=weight, significant=significant)

    first = rep(16, 30.0, games)  # 9.Nxg5 in games a, b, c
    follow = rep(20, 40.0, games)  # then 11.Kh1 in the same games
    elsewhere = rep(12, 50.0, [NS(game_id="d"), NS(game_id="e")])
    kept, folded = mistakes.fold_follow_ups([elsewhere, follow, first])
    assert kept == [elsewhere, first] and folded[id(first)] == [follow]
    # never folds a claim into an observation (that would drop or promote a claim)
    weak_root = rep(16, 30.0, games, significant=False)
    kept, folded = mistakes.fold_follow_ups([follow, weak_root])
    assert kept == [follow, weak_root] and not folded
