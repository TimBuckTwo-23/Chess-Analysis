"""Explanations checked against the chess review of the coaching layer: what changes hands and from when, what is
new about the move, what leads the text, the point of the line, the marks, the concept note and the verdict.

The positions are the review's (real games and Lichess puzzles); the lines are Stockfish 16's at depth 16, recorded
here so that no engine is needed. Concept terms need Stockfish 16's ``eval`` and are faked where a test needs them.
"""

from __future__ import annotations

import dataclasses

import chess
import pytest

from chess_insights.analysis.mistakes import move_label
from chess_insights.coach import concepts, explain, motifs
from chess_insights.coach.deep import DeepResult, make_line
from chess_insights.coach.explain import Comparison
from chess_insights.context import AnalysisContext
from chess_insights.models import ConceptDelta, CriticalPosition, Line, Motif

from factories import make_game

GATED = frozenset(motifs.GATED_THEMES)


def line(fen: str, ucis: list[str], cp: int, mate=None) -> Line:
    return make_line(fen, [chess.Move.from_uci(u) for u in ucis], cp, mate, 16)


def position(fen: str, played_uci: str, color: str, kind: str = "error", game=None, moves_before=(),
             drop: float = 20.0, best_san=None) -> CriticalPosition:
    board = chess.Board(fen)
    san = board.san(chess.Move.from_uci(played_uci))
    return CriticalPosition(
        game_id=game.game_id if game else "g1", url=game.url if game else "https://www.chess.com/game/live/1",
        ply=len(moves_before), fen=fen, played_uci=played_uci, played_san=san, move_label=move_label(fen, san),
        best_san=best_san, drop=drop, time_class="blitz", color=color, kind=kind, repeats=1,
        games=[game.url] if game else [], opening_family="", moves_before=list(moves_before),
    )


def explained(pos: CriticalPosition, best: Line, refutation: Line, ctx=None, tool=None, gated=GATED):
    games = explain._Games(ctx or AnalysisContext("t", []))
    result = DeepResult(epd=pos.epd, played_uci=pos.played_uci, best_line=best, refutation=refutation,
                        engine="Stockfish 16", depth=16)
    owned = tool is None
    tool = tool or explain._Concepts(None)  # board facts and material, no Stockfish
    try:
        return explain.explain_position(games, pos, result, tool, gated)
    finally:
        if owned:
            tool.close()


# The review's positions: FEN before your move, your move, your colour, the better line, the refutation.
BD3 = ("r4rk1/ppp1qpbp/2n2np1/8/3P4/5b2/PPPQBPPP/RNB2RK1 w - - 0 10", "e2d3", "white",
       ["e2f3", "f8e8", "c2c3", "a8d8", "d2c2", "c6b8", "c1e3", "f6d5", "f3d5", "d8d5"], 176,
       ["e2d3", "f3g4", "h2h3", "g4e6", "c2c3", "e7d7", "d2c2", "h7h6", "b1d2", "f8e8"], -449)
BEFORE_BXF3 = "r4rk1/ppp1qpbp/2n2np1/8/3P2b1/5N2/PPPQBPPP/RNB2RK1 b - - 5 9"  # before 9...Bxf3
NCXD4 = ("r1bqk2r/ppp2pbp/2n1p1p1/3pPn2/3P4/2NBBN2/PPPQ1PPP/R3K2R b KQkq - 4 8", "c6d4", "black",
         ["f5e3", "f2e3", "f7f6", "e5f6", "d8f6", "d3b5", "c8d7", "e1g1", "e8g8", "e3e4"], -46,
         ["c6d4", "f3d4", "f5e3", "f2e3", "g7e5", "d4f3", "e5g7", "h2h4", "c7c6", "h4h5"], -207)
NC6 = ("rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", "b8c6", "black",  # the 2...Nc6 habit
       ["c5d4", "g1f3", "e7e5", "c2c3", "g8f6", "c3d4", "f6e4", "f1d3", "f8b4", "b1d2"], -17,
       ["b8c6", "d4d5", "c6b8", "f2f4", "g7g6", "f1d3", "d7d6", "g1f3", "f8g7", "c2c4"], -152)
QE8 = ("7Q/6p1/6k1/6r1/7r/5K1p/8/8 w - - 16 72", "h8e8", "white",
       ["h8h4", "g5f5", "f3g4", "f5c5", "h4e7", "h3h2", "e7d6", "g6h7", "d6h2", "h7g8"], 23,
       ["h8e8", "g6h7", "e8e1", "h4h5", "e1e4", "g5g6", "f3e2", "h3h2", "e4h1", "h5h4"], -894)
QD1 = ("r2qrk2/ppp2Bpp/5n2/8/3n4/PQN1PPP1/1P3P2/R3K2R w KQ - 1 15", "b3d1", "white",
       ["b3a2", "d4f3", "e1e2", "e8e7", "f7d5", "f3e5", "a1d1", "d8c8", "d1d4", "c7c6"], 61,
       ["b3d1", "f8f7", "g3g4", "d4e6", "d1b3", "f7f8", "a1d1", "d8e7", "c3d5", "f6d5"], -224)
GXF3 = ("r2qr1k1/ppp2ppp/2n2n2/8/2BP4/PQN1PbP1/1P3PP1/R3K2R w KQ - 0 13", "g2f3", "white",
        ["c4f7", "g8h8", "g2f3", "c6d4", "b3d1", "e8e7", "f7a2", "d4e6", "d1d8", "a8d8"], 300,
        ["g2f3", "c6d4", "b3d1", "d4e6", "d1a4", "c7c6", "a1d1", "d8b6", "c4e6", "e8e6"], 76)
BG6 = ("8/pp1qn1r1/3p3k/2pP1p1b/2P1pP1N/2P1N2P/P4Q1K/6R1 b - - 1 33", "h5g6", "black",
       ["h6h7", "g1g7", "h7g7", "f2g1", "g7f7", "g1g5", "h5g6", "h4g6", "e7g6", "e3f5"], -325,
       ["h5g6", "h4f5", "e7f5", "e3g4", "h6h5", "g4f6", "h5h6", "f6d7", "e4e3", "f2e2"], -442)
QXD5 = ("5r1k/p1p2qpp/6b1/1P1QP3/8/4P1R1/PB4PP/3R2K1 b - - 0 26", "f7d5", "black",  # Lichess puzzle 023sA
        ["f7f2", "g1h1", "f2f1", "d1f1", "f8f1"], 1000,
        ["f7d5", "d1d5", "f8e8", "d5d7", "h8g8", "e5e6", "e8e6", "g3f3", "e6f6", "b2f6"], -811)
NF6 = ("r1bqk1nr/ppp1ppbp/2nP2p1/8/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 5", "g8f6", "black",
       ["d8d6", "f1b5", "c8d7", "c2c3", "e7e5", "d4e5", "d6d1", "e1d1", "a7a6", "b5c6"], 19,
       ["g8f6", "d6e7", "c6e7", "f1d3", "e8g8", "e1g1", "e7f5", "c2c3", "c7c5", "d4c5"], -82)
RH4 = ("3Q4/6p1/8/6rk/6r1/5K1p/8/8 b - - 13 70", "g4h4", "black",
       ["g5f5", "f3e3", "g4f4", "d8h8", "h5g6", "h8h3"], 469,
       ["g4h4", "d8h8", "h5g6", "h8h4", "g5h5", "h4e4", "g6h6"], -5)


def case(data, mate=None):
    fen, played, color, best, best_cp, ref, ref_cp = data
    return position(fen, played, color), line(fen, best, best_cp, mate), line(fen, ref, ref_cp)


# --------------------------------------------------------------------------- what changes hands
def test_trade_words_say_what_changes_hands():
    P, N, B, R, Q = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
    assert explain.trade_words([Q, P, P], [N, B], 5, yours=True) == "your queen and two pawns for a knight and a bishop"
    assert explain.trade_words([N, N], [P, B, P], 1, yours=True) == "a knight for two pawns"  # a knight for a bishop
    assert explain.trade_words([P], [], 1, yours=True) == "a pawn"
    assert explain.trade_words([Q], [R], 4, yours=False) == "the queen for a rook"
    assert explain.trade_words([N], [B], 0, yours=True) == ""  # an even trade: nothing changes hands
    assert explain.trade_words([R], [], 3, yours=True) == ""  # does not add up (a promotion on the way): no words
    assert explain.trade_words([Q, R], [N, B, P], 8, yours=False) == ""  # too many kinds to read: the amount


def test_a_recapture_takes_back_the_piece_and_wins_nothing():
    """9...Bxf3 took the knight: 10.Bxf3 takes it back (it wins nothing), and 10.Bd3 leaves White a knight down."""
    pos, best, ref = case(BD3)
    tool = explain._Concepts(None)
    cmp = tool.compare(pos.fen, best, ref, "white", BEFORE_BXF3)
    tool.close()
    assert (cmp.material_lost, cmp.material_missed, cmp.recapture, cmp.lost_words) == (3, 0, "piece", "a knight")
    text = explain._what_it_wins(cmp, explain._gap(best, ref), "10.Bxf3", "10.Bd3")
    assert text.startswith("10.Bxf3 takes back the piece, and your 10.Bd3 leaves you a knight down")
    assert "would have won" not in text
    # the whole explanation, with the game that leads to the position: the next check is about your own captures
    game = make_game(moves_san=["Bxf3", "Bd3"], color="white", initial_fen=BEFORE_BXF3)
    pos = dataclasses.replace(pos, game_id=game.game_id, moves_before=["Bxf3"])
    ex = explained(pos, best, ref, AnalysisContext("t", [game]))
    assert "10.Bxf3 takes back the piece, and your 10.Bd3 leaves you a knight down" in ex.text
    assert "Next time, look at your own checks, captures and threats first." in ex.text


def test_a_knight_for_two_pawns_and_the_count_of_attackers_and_defenders():
    pos, best, ref = case(NCXD4)
    ex = explained(pos, best, ref)
    assert ("9.Nxd4 takes the knight on d4, which two white pieces attack and only one of yours defends, and you "
            "lose a knight for two pawns") in ex.text
    assert "count the attackers and defenders" in ex.text
    # the quoted line reaches the point where the material has changed hands (the comparison point)
    assert "10.fxe3 Bxe5" in ex.text


def test_a_fair_trade_is_not_counted_as_a_pawn_or_piece_short_of_defenders():
    fen = "r4rk1/ppp1qpbp/2n2np1/8/3P2b1/5N2/PPPQBPPP/RNB2RK1 b - - 5 9"  # 9...Bxf3 10.Bxf3: a bishop for a knight
    pos = position(fen, "g4f3", "black")
    refutation = line(fen, ["g4f3", "e2f3", "f8e8", "c2c3", "c6d8"], -173)
    assert explain._count_point(pos, refutation, Comparison(), 0) is None


def test_a_pawn_won_in_a_gambit_line_is_not_won_and_the_habit_lesson_is_the_lost_time():
    """2...Nc6 3.d5 Nb8: the knight has to go back. At depth 16 Stockfish's line after 2...cxd4 has Black a pawn up
    (4.c3 Nf6 5.cxd4 Nxe4), but it rates it better for White: a gambit, not a pawn 2...cxd4 would have won."""
    pos, best, ref = case(NC6)
    tool = explain._Concepts(None)
    cmp = tool.compare(pos.fen, best, ref, "black")
    tool.close()
    assert cmp.material_missed == 0
    ex = explained(pos, best, ref)
    assert "would have won" not in ex.text
    assert ("3.d5 hits your knight with a pawn and it has to move again (3...Nb8), so you lose time; 2...cxd4 first "
            "takes the pawn that chases it") in ex.text
    assert ex.text.endswith(f"Next time, {explain.TEMPO_CHECK}.")


def test_material_taken_by_force_counts_even_in_a_drawish_ending():
    """72.Qxh4 takes a loose rook: Stockfish rates the ending +0.23 however much material it gives White."""
    pos, best, ref = case(QE8)
    ex = explained(pos, best, ref)
    assert ex.text.startswith("You had an undefended piece to take: 72.Qxh4 Rf5+ 73.Kg4")
    # the second sentence names the move you played, so the evaluation is not read as the better line's
    assert "After your 72.Qe8+, Stockfish rates the position −8.94 for you, against +0.23 after 72.Qxh4" in ex.text
    assert "72.Qxh4 would have won a rook" in ex.text


def test_a_loss_the_better_line_shares_is_counted_against_it_or_named_plainly():
    rel = Comparison(material_lost=2, lost_words="a bishop", lost_relative=True)
    assert explain._what_it_wins(rel, 3.0, "15.Qa2") == "you lose a bishop (2 pawns' worth more than after 15.Qa2)"
    # 33...Bg6 loses the queen, but 33...Kh7 is bad too: the difference between the lines does not back the loss,
    # the refutation's own verdict does, and the text says what changes hands without a comparison
    plain = Comparison(material_lost=7, lost_abs=7, lost_words="your queen and a pawn for a knight")
    assert explain._what_it_wins(plain, 1.17) == "you lose your queen and a pawn for a knight"
    assert explain._what_it_wins(dataclasses.replace(plain, lost_abs=0), 1.17) == ""


def test_a_fork_is_followed_until_the_queen_falls_and_the_chart_leaves_material_out(monkeypatch, fixtures_dir):
    pos, best, ref = case(BG6)
    table = concepts.parse_eval_table((fixtures_dir / "coach" / "sf16_eval_sicilian_5e5.txt").read_text())
    other = {k: (v[0] - 0.5, v[1] - 0.5) for k, v in table.items()}  # every term differs by half a pawn
    ref_end = concepts.comparison_point(pos.fen, ref.moves_uci)
    monkeypatch.setattr(concepts.ClassicalEval, "table", lambda self, fen: other if fen == ref_end.fen() else table)
    tool = explain._Concepts("/nonexistent/stockfish-16")
    ex = explained(pos, best, ref, tool=tool)
    tool.close()
    assert ex.text.startswith("After 33...Bg6, White has a fork: 34.Nhxf5+ Nxf5 35.Ng4+ Kh5 36.Nf6+ Kh6 37.Nxd7")
    assert "you lose your queen and a pawn for a knight" in ex.text
    assert "37.Nxd7" in [f.move for f in ex.diagram.strips[0].frames]
    # the two ends differ in material: no Material or piece-balance bar (and never "winning chances")
    assert ex.chart is not None
    assert not {"Material", "Piece balance", "Winning chances", "Piece placement"} & set(ex.chart.labels)
    assert all(c.term not in ("Material", "Imbalance", "Winnable") for c in ex.concepts)


# --------------------------------------------------------------------------- what is new, and what leads
def test_what_your_move_allowed_is_what_is_new():
    # 15.Qd1: the e3 pawn's pin stood before the move; Black takes the bishop the queen stopped guarding
    pos, best, ref = case(QD1)
    ex = explained(pos, best, ref)
    assert ex.text.startswith("After 15.Qd1, Black has an undefended piece to take: 15...Kxf7")
    assert "pin" not in {m.theme for m in ex.motifs} and "in line with your king" not in ex.text
    assert "you lose a bishop" in ex.text
    # 13.gxf3 Nxd4 pins nothing new: the same pin and ...Nxd4 come after 13.Bxf7+ too; the point is the check first
    pos, best, ref = case(GXF3)
    allowed, missed = explain.split_motifs(best, ref)
    assert "pin" not in {m.theme for m in allowed}
    ex = explained(pos, best, ref)
    assert ex.text.startswith("You had a fork: 13.Bxf7+ Kh8 14.gxf3")
    assert "After your 13.gxf3, Stockfish rates the position +0.76 for you, against +3.00 after 13.Bxf7+" in ex.text


def test_a_pattern_found_in_both_lines_is_not_blamed_on_the_move(monkeypatch):
    fork = Motif("fork", "refutation", 1, ["c7", "e8", "a8"], "opponent")
    monkeypatch.setattr(motifs, "detect_line", lambda line, role, previous=None: [fork] if role == "refutation" else [])
    monkeypatch.setattr(motifs, "carried_by", lambda line, side: [dataclasses.replace(fork, ply=3)]
                        if side == "second" else [])
    best = Line(fen=chess.STARTING_FEN, moves_uci=["e2e4"])
    assert explain.split_motifs(best, best) == ([], [])
    monkeypatch.setattr(motifs, "carried_by", lambda line, side: [Motif("fork", "", 3, ["c7", "g8", "a8"], "second")])
    assert explain.split_motifs(best, best)[0] == []  # two squares in common: the same fork
    monkeypatch.setattr(motifs, "carried_by", lambda line, side: [Motif("fork", "", 3, ["d6", "e8", "b7"], "second")])
    assert explain.split_motifs(best, best)[0] == [fork]  # another fork: still what the move allowed


def test_a_forced_mate_leads_the_text_and_its_strip_reaches_the_mate():
    pos, best, ref = case(QXD5, mate=3)
    ex = explained(pos, best, ref)
    assert ex.text.startswith("You had a back-rank mate in 3: 26...Qf2+ 27.Kh1 Qf1+ 28.Rxf1 Rxf1#.")
    assert "After your 26...Qxd5, Stockfish rates the position −8.11 for you, against mate in 3 for you" in ex.text
    assert "fork" not in ex.text.split(".")[0]
    assert ex.diagram.strips[1].frames[-1].move == "28...Rxf1#"
    # without the back-rank detector's name: plain "mate in 3"
    assert explained(pos, best, ref, gated=GATED - {"backRankMate"}).text.startswith("You had mate in 3: 26...Qf2+")


def test_a_far_advanced_pawn_leads_only_when_it_goes_on_to_promote():
    fen = "r1bqk1nr/ppp1ppbp/2nP2p1/8/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 5"
    ref = Line(fen=fen, moves_uci=["g8f6", "d6e7", "c6e7", "f1d3"])
    assert not explain._leads(Motif("advancedPawn", "refutation", 1, ["e7"], "opponent"), ref)
    promo = Line(fen="4k3/8/4P3/8/8/8/8/4K3 b - - 0 1", moves_uci=["e8d8", "e6e7", "d8c7", "e7e8q"])
    assert explain._leads(Motif("advancedPawn", "refutation", 1, ["e7"], "opponent"), promo)
    assert explain._leads(Motif("fork", "refutation", 1, ["e7"], "opponent"), ref)


def test_a_skipped_recapture_is_the_point():
    """5.exd6 took a pawn; 5...Nf6 leaves it (6.dxe7 Nxe7), 5...Qxd6 takes it back."""
    fen, played, color, best, best_cp, ref, ref_cp = NF6
    previous = "r1bqk1nr/ppp1ppbp/2np2p1/4P3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 5"
    game = make_game(moves_san=["exd6", "Nf6"], color="black", initial_fen=previous)
    pos = dataclasses.replace(position(fen, played, color), game_id=game.game_id, moves_before=["exd6"])
    ex = explained(pos, line(fen, best, best_cp), line(fen, ref, ref_cp), AnalysisContext("t", [game]))
    assert "far-advanced" not in ex.text and "promote" not in ex.text
    assert "5...Qxd6 takes back the pawn White has just taken" in ex.text
    assert ex.text.endswith(f"Next time, {explain.MISSED_CHECK}.")


def test_an_enemy_piece_on_a_hole_is_the_point_of_5e5():
    fen = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    pos = position(fen, "e6e5", "black")
    best = line(fen, ["g8f6", "d4c6", "b7c6", "e4e5", "f6d5", "c3e4"], -39)
    ref = line(fen, ["e6e5", "d4b5", "a7a6", "b5d6", "f8d6", "d1d6", "d8e7"], -157)
    ex = explained(pos, best, ref)
    assert "White's knight gets to d6 (7.Nd6+), a square no pawn of yours can cover" in ex.text
    assert ex.text.endswith(f"Next time, {explain.INVASION_CHECK}.")


# --------------------------------------------------------------------------- the pictures
def test_marks_go_only_where_the_pieces_stand_and_checkers_are_attackers():
    # 70...Rh4 71.Qh8+: the skewer's squares are read after 71.Qh8+, so the main board (before 70...Rh4, the queen
    # on d8 and the rook on g4) gets no mark; the frame of 71.Qh8+ does
    pos, best, ref = case(RH4)
    ex = explained(pos, best, ref)
    assert ex.diagram.marks == []
    first = ex.diagram.strips[0].frames[0]
    assert first.move == "71.Qh8+" and [(m.square, m.kind) for m in first.marks] == [
        ("h8", "attacker"), ("h5", "target"), ("h4", "target")]
    # a double check (25.Bxe6+): both checkers are attackers, only the king a target
    board = chess.Board("r5r1/1p1k1p2/p3b3/3Bb3/8/P2Q3q/1P3PP1/R4RK1 w - - 0 25")
    board.push_uci("d5e6")
    for theme, squares in (("doubleCheck", ["e6", "d3", "d7"]), ("discoveredCheck", ["d3", "d7", "e6"])):
        marks = explain.motif_marks(Motif(theme, "best", 0, squares, "you"), board, chess.WHITE)
        assert sorted((m.square, m.kind) for m in marks) == [("d3", "attacker"), ("d7", "target"), ("e6", "attacker")]
    # a discovered check by the rook: the knight that stepped aside is not marked
    board = chess.Board("4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1")
    board.push_uci("e4c5")
    marks = explain.motif_marks(Motif("discoveredCheck", "best", 0, ["e1", "e8", "c5"], "you"), board, chess.WHITE)
    assert [(m.square, m.kind) for m in marks] == [("e1", "attacker"), ("e8", "target")]


def test_a_pattern_whose_pieces_all_stand_is_marked_on_the_main_board_once():
    fen = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    pos = position(fen, "e6e5", "black")
    ref = line(fen, ["e6e5", "d4b5", "a7a6"], -150)
    best = line(fen, ["g8f6", "d4c6"], -40)
    # a (made-up) pattern at your move whose pieces all stood before it, and a second one sharing a square
    pin = Motif("pin", "refutation", 0, ["d4", "c6", "e8"], "opponent")
    fork = Motif("fork", "refutation", 0, ["d4", "c6", "b3"], "opponent")
    marks = explain._board_marks(pos, best, ref, [pin, fork], frozenset({"pin", "fork"}))
    assert [(m.square, m.kind) for m in marks] == [("d4", "attacker"), ("c6", "target"), ("e8", "target")]
    frames = explain._frame_marks([pin, fork], frozenset({"pin", "fork"}), best, 0, 2, "black")
    assert all(len({m.square for m in f}) == len(f) for f in frames)


# --------------------------------------------------------------------------- notes, advice and the verdict
def test_the_concept_note_teaches_only_what_your_move_made_worse():
    better_king = [ConceptDelta("King safety", 0.78, label="king safety"),
                   ConceptDelta("Mobility", -0.3, label="piece activity")]
    note, _ = explain.concept_note_for(better_king)
    assert note and "king" not in note["label"].lower()
    assert explain.concept_note_for([ConceptDelta("King safety", 0.78, label="king safety")]) == ({}, [])
    material = [ConceptDelta(t, -2.0, label=t.lower()) for t in ("Material", "Imbalance", "Winnable")]
    assert explain.concept_note_for(material) == ({}, [])
    worst_first = [ConceptDelta("Mobility", -0.3, label="piece activity"),
                   ConceptDelta("King safety", -1.2, label="king safety")]
    assert "king" in explain.concept_note_for(worst_first)[0]["label"].lower()
    assert "king" not in explain.concept_note_for(worst_first, king=False)[0].get("label", "").lower()


def test_king_safety_advice_only_for_a_move_that_touches_the_king():
    fen = "rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
    assert not explain._near_king(position(fen, "b8c6", "black"))  # 2...Nc6
    assert explain._near_king(position(fen, "e7e6", "black")) and explain._near_king(position(fen, "e8d8", "black"))
    castle = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 8"
    assert explain._near_king(position(castle, "e8g8", "black"))
    deltas = [ConceptDelta("King safety", -0.85, label="king safety"),
              ConceptDelta("Mobility", -0.43, label="piece activity")]
    cmp = Comparison(deltas=deltas)
    assert explain._check_next_time([], [], GATED, cmp, 1.3, False) == explain.CONCEPT_CHECKS["king safety"]
    assert explain._check_next_time([], [], GATED, cmp, 1.3, False, near_king=False) == \
        explain.CONCEPT_CHECKS["piece activity"]


@pytest.mark.parametrize("kind,best_first,ref_cp,drop,want", [
    ("error", "e6e5", -40, 20.0, "fine"),  # the deeper search's own first choice
    ("error", "g8f6", -60, 20.0, "close"),  # 2.9 win-% points behind: not even an inaccuracy
    ("error", "g8f6", -157, 20.0, "error"),
    ("choice", "g8f6", -157, 2.3, "close"),  # a choice point under the openings section's bar
    ("choice", "g8f6", -157, 20.0, "error"),
])
def test_the_verdict_of_the_deeper_search(kind, best_first, ref_cp, drop, want):
    fen = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    pos = position(fen, "e6e5", "black", kind=kind, drop=drop, best_san="Nf6")
    best = line(fen, [best_first, "d4b5", "a7a6"] if best_first == "e6e5" else [best_first, "d4c6", "b7c6"], -39)
    ref = line(fen, ["e6e5", "d4b5", "a7a6", "b5d6"], ref_cp)
    assert explained(pos, best, ref).verdict == want


def test_a_finished_trade_is_not_counted_from_before_its_last_capture():
    """10.hxg3 took back the bishop that took on g3: the trade is over (Black can't take on g3), so after 10...Bg4
    Black loses the c4 pawn, not "a bishop and a pawn"."""
    fen = "r2q1rk1/ppp2ppp/2n1bn2/8/2pP4/P1N1PNP1/1P3PP1/R2QKB1R b KQ - 0 10"
    previous = "r2q1rk1/ppp2ppp/2n1bn2/8/2pP4/P1N1PNb1/1P3PPP/R2QKB1R w KQ - 0 10"
    best = line(fen, ["c6a5", "d1c2", "h7h6", "a1d1", "c7c6", "f1e2", "d8c7", "f3e5"], -23)
    ref = line(fen, ["e6g4", "f1c4", "a8c8", "d1a4", "a7a6", "e1g1", "c6e7", "a4b4"], -236)
    tool = explain._Concepts(None)
    cmp = tool.compare(fen, best, ref, "black", previous)
    tool.close()
    assert explain.last_move(previous, fen)[1].uci() == "h2g3"
    assert (cmp.material_lost, cmp.recapture) == (1, "") and "bishop" not in cmp.lost_words
    game = make_game(moves_san=["hxg3", "Bg4"], color="black", initial_fen=previous)
    pos = dataclasses.replace(position(fen, "e6g4", "black"), game_id=game.game_id, moves_before=["hxg3"])
    ex = explained(pos, best, ref, AnalysisContext("t", [game]))
    assert "11.Bxc4 takes the pawn on c4, which one white piece attacks and none of yours defends." in ex.text
    assert "bishop" not in ex.text
