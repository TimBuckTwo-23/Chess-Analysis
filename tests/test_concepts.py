"""coach/concepts.py: Stockfish 16's classical eval table, concept deltas at the line ends, and board facts."""

from __future__ import annotations

import sys
import textwrap

import chess
import pytest

from chess_insights.coach import concepts
from chess_insights.coach.concepts import (
    ClassicalEval,
    board_facts,
    comparison_point,
    concept_deltas,
    fact_differences,
    material_phase,
    parse_eval_table,
)

AFTER_5E5 = "r1bqkbnr/pp1p1ppp/2n5/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"
IN_CHECK = "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3"


def _fixture(fixtures_dir, name: str) -> str:
    return (fixtures_dir / "coach" / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the eval table
def test_parse_eval_table_reads_the_total_column_of_stockfish_16(fixtures_dir):
    table = parse_eval_table(_fixture(fixtures_dir, "sf16_eval_sicilian_5e5.txt"))
    assert table is not None
    assert set(table) == set(concepts.TERMS)
    assert table["Material"] == (0.41, 0.30)  # "----" in the White / Black columns, numbers in Total
    assert table["Threats"] == (-0.86, -0.53)  # right after 5...e5 the pawn's attack on d4 counts for Black
    assert table["King safety"] == (0.12, 0.00)
    assert table["Winnable"] == (0.00, 0.05)


def test_parse_eval_table_handles_dashes_check_and_newer_versions(fixtures_dir):
    # sf_newer_eval_no_table.txt: the recorded Stockfish 16 output without its classical section, as Stockfish 16.1
    # and later print it (they dropped the classical evaluation)
    assert parse_eval_table(_fixture(fixtures_dir, "sf16_eval_in_check.txt")) is None
    assert parse_eval_table(_fixture(fixtures_dir, "sf_newer_eval_no_table.txt")) is None
    assert parse_eval_table("") is None
    text = textwrap.dedent("""
         Contributing terms for the classical eval:
        +------------+-------------+-------------+-------------+
        |    Term    |    White    |    Black    |    Total    |
        |            |   MG    EG  |   MG    EG  |   MG    EG  |
        +------------+-------------+-------------+-------------+
        |   Material |  ----  ---- |  ----  ---- |  ----  ---- |
        |   Mobility |  0.39  0.51 |  0.05  0.11 |  0.34  ---- |
        |    Unknown |  0.10  0.10 |  0.00  0.00 |  0.10  0.10 |
        +------------+-------------+-------------+-------------+
        |      Total |  ----  ---- |  ----  ---- |  0.10  0.09 |
    """)
    assert parse_eval_table(text) == {"Mobility": (0.34, 0.0)}


def test_material_phase_and_blend():
    assert material_phase(chess.Board()) == 24
    assert material_phase(chess.Board("4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1")) == 0
    assert material_phase(chess.Board("r3k3/8/8/8/8/8/8/3QK3 w - - 0 1")) == 6  # rook 2 + queen 4
    assert concepts.blend(1.0, 0.0, 24) == 1.0 and concepts.blend(1.0, 0.0, 0) == 0.0
    assert concepts.blend(0.4, 0.2, 12) == pytest.approx(0.3)


def test_concept_deltas_are_refutation_minus_best_from_your_side():
    best = {"Mobility": (0.40, 0.40), "King safety": (0.10, 0.00), "Material": (0.50, 0.50), "Space": (0.0, 0.0)}
    ref = {"Mobility": (0.10, 0.10), "King safety": (-0.20, 0.00), "Material": (0.50, 0.50), "Space": (0.05, 0.0)}
    white = concept_deltas(best, ref, "white", 24, 24)
    assert [(c.term, c.value) for c in white] == [("King safety", -0.30), ("Mobility", -0.30)]  # worst first
    assert white[1].label == "piece activity" and white[1].mg == -0.30 and white[1].source == "stockfish16"
    # the same numbers from Black's side: better for Black (the table is White's point of view)
    black = concept_deltas(best, ref, "black", 24, 24)
    assert {c.term: c.value for c in black} == {"Mobility": 0.30, "King safety": 0.30}
    # each end is blended with its own phase: an endgame at the refutation end reads the EG column
    eg = concept_deltas({"Mobility": (0.4, 0.0)}, {"Mobility": (0.4, 0.0)}, "white", 24, 0)
    assert [(c.term, c.value) for c in eg] == [("Mobility", -0.40)]
    # below MIN_DELTA: left out
    assert concept_deltas({"Space": (0.0, 0.0)}, {"Space": (0.1, 0.1)}, "white", 24, 24) == []


def test_material_terms_only_when_material_is_level_and_winnable_never():
    """With the same material at both ends, Stockfish's Material term is piece placement and Imbalance piece balance.
    With different material they only restate it on the evaluation's own scale (Material +1.86 for a line that
    loses the queen, read mid-combination): left out, the text names what changes hands. Winnable (scaling towards a
    draw) reads like the game's outcome ("winning chances +1.46" for a line that lost 79% of them): never shown."""
    best = {"Material": (0.5, 0.5), "Imbalance": (0.3, 0.3), "Winnable": (0.0, 0.0), "Mobility": (0.4, 0.4)}
    ref = {"Material": (0.0, 0.0), "Imbalance": (0.0, 0.0), "Winnable": (0.9, 0.9), "Mobility": (0.0, 0.0)}
    differs = concept_deltas(best, ref, "white", 20, 20)
    assert [c.term for c in differs] == ["Mobility"]
    level = {c.term: c.label for c in concept_deltas(best, ref, "white", 20, 20, material_level=True)}
    assert level == {"Material": "piece placement", "Imbalance": "piece balance", "Mobility": "piece activity"}
    assert "Winnable" not in level and "Winnable" in concepts.HIDDEN_TERMS


# --------------------------------------------------------------------------- where the lines are compared
def _uci(sans: list[str], fen: str = chess.STARTING_FEN) -> list[str]:
    board, out = chess.Board(fen), []
    for san in sans:
        out.append(board.push_san(san).uci())
    return out


def test_comparison_point_waits_for_a_settled_position():
    # after 3 plies (1.e4 d5 2.exd5) Black takes the pawn back: not settled, so the point moves on to 2...Qxd5
    end = comparison_point(chess.STARTING_FEN, _uci(["e4", "d5", "exd5", "Qxd5", "Nc3"]), plies=3)
    assert end is not None and end.fen().split()[0] == "rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR"
    # 1.e4 d5 is settled: exd5 Qxd5 is an even trade, no material is waiting to be won
    end = comparison_point(chess.STARTING_FEN, _uci(["e4", "d5", "exd5", "Qxd5", "Nc3"]), plies=2)
    assert end is not None and end.fen().split()[0] == "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR"
    # a check is never a comparison point (Stockfish prints no table in check): walk back when nothing follows
    end = comparison_point(chess.STARTING_FEN, _uci(["e4", "d5", "Bb5+"]), plies=3)
    assert end is not None and end.fen().split()[0] == "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR"
    assert comparison_point(chess.STARTING_FEN, [], plies=6) is None
    assert comparison_point(chess.STARTING_FEN, ["e2e5"], plies=6) is None  # an illegal move ends the line


# --------------------------------------------------------------------------- board facts
def test_board_facts_find_pawn_weaknesses_holes_and_the_king():
    # Black after ...e5 with the c-pawn gone (the 5...e5 structure): d5 and d6 are holes, d7 is backward
    board = chess.Board("r1bqkbnr/pp1p1ppp/2n5/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6")
    facts = board_facts(board, "black")
    assert "d5" in facts.holes and "d6" in facts.holes
    assert facts.bishop_pair and facts.opp_bishop_pair and facts.can_castle and not facts.castled
    assert facts.material == 0
    # isolated, doubled, backward
    board = chess.Board("4k3/8/8/8/3p4/2P5/1P2P1P1/4K3 w - - 0 1")
    facts = board_facts(board, "white")
    assert facts.isolated == ["e2", "g2"] and facts.doubled_files == []
    assert facts.backward == []  # b2-c3 chain: c3 is supported from behind
    board = chess.Board("4k3/8/8/2p5/8/1P1P4/1P6/4K3 w - - 0 1")
    facts = board_facts(board, "white")
    assert facts.doubled_files == ["b"]
    assert board_facts(chess.Board("4k3/8/8/2p1p3/8/3P4/8/4K3 w - - 0 1"), "white").isolated == ["d3"]
    # d3 has no pawn beside or behind it (e4 is ahead) and Black's c5 guards d4, its next square
    backward = board_facts(chess.Board("4k3/8/8/2p5/4P3/3P4/8/4K3 w - - 0 1"), "white")
    assert backward.backward == ["d3"] and backward.isolated == []
    # a king still on the e-file after move 12 with queens on is in the centre
    late = chess.Board("r1bq1rk1/pp3ppp/8/8/8/8/PP3PPP/R1BQK2R w - - 0 15")
    assert board_facts(late, "white").king_in_centre and not board_facts(late, "black").king_in_centre
    assert board_facts(late, "black").castled and not board_facts(late, "white").can_castle


def test_fact_differences_name_only_what_gets_worse_for_you():
    best = chess.Board("r1bqk2r/pp3ppp/2n2n2/2bpp3/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 5")
    worse = chess.Board("r1bqk2r/pp3ppp/2n2n2/3p4/3p4/8/PPPPPPPP/RNBQKBNR b Qkq - 0 5")
    facts = fact_differences(board_facts(best, "black"), board_facts(worse, "black"))
    assert facts[0] == "you lose the bishop pair"
    assert all("castle" not in f for f in facts)  # Black keeps both castling rights in both
    assert fact_differences(board_facts(best, "black"), board_facts(best, "black")) == []
    no_castle = chess.Board("r1bq3r/pp3ppp/2nk1n2/2bpp3/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 5")
    lost = fact_differences(board_facts(best, "black"), board_facts(no_castle, "black"))
    assert "you lose the right to castle" in lost
    assert len(fact_differences(board_facts(chess.Board(), "white"), board_facts(worse, "white"), limit=1)) <= 1


def test_the_bishop_pair_counts_against_the_opponents_and_holes_need_an_enemy_minor_piece():
    # 5.d3 Bc5 6.Be3 Bb6 7.Bxb6 axb6: both sides give up the bishop pair, so nothing changes between you
    both = chess.Board("r1bqk2r/1ppp1pp1/1pn2n1p/4p3/2B1P3/3P1N2/PPP2PPP/RN1Q1RK1 w kq - 0 8")
    start = chess.Board("r1bqkb1r/pppp1pp1/2n2n1p/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5")
    assert board_facts(both, "white").bishop_pair is False and board_facts(start, "white").bishop_pair
    assert fact_differences(board_facts(start, "white"), board_facts(both, "white")) == []
    # only your opponent keeps it
    opp_keeps = chess.Board("r1bqk2r/pppp1pp1/2n2n1p/2b1p3/4P3/3P1N2/PPP2PPP/RN1Q1RK1 w kq - 0 8")
    both_lose = chess.Board("r1bqk2r/1ppp1pp1/1pn2n1p/4p3/4P3/3P1N2/PPP2PPP/RN1Q1RK1 w kq - 0 8")
    assert fact_differences(board_facts(both_lose, "white"), board_facts(opp_keeps, "white")) == [
        "your opponent keeps the bishop pair"]
    # a new hole matters only while your opponent has a knight or bishop to put there
    # (c2-c4 and e2-e4 leave d3 and d4 for good)
    rooks = chess.Board("3rk3/8/8/8/2P1P3/8/PP1P1PPP/3RK3 w - - 0 30")
    rooks_before = chess.Board("3rk3/8/8/8/8/8/PPPPPPPP/3RK3 w - - 0 30")
    assert board_facts(rooks, "white").holes == ["d3", "d4"] and not board_facts(rooks, "white").opp_minor
    assert fact_differences(board_facts(rooks_before, "white"), board_facts(rooks, "white")) == []
    knight = chess.Board("3rk3/8/2n5/8/2P1P3/8/PP1P1PPP/3RK3 w - - 0 30")
    knight_before = chess.Board("3rk3/8/2n5/8/8/8/PPPPPPPP/3RK3 w - - 0 30")
    assert fact_differences(board_facts(knight_before, "white"), board_facts(knight, "white")) == [
        "d3 and d4 become holes in your camp (no pawn of yours can cover them)"]


def _facts_along(fen: str, best: list[str], refutation: list[str], color: str) -> list[str]:
    """fact_differences at the comparison points of two SAN lines from ``fen``, with the facts before the move."""
    ends = []
    for sans in (best, refutation):
        board = chess.Board(fen)
        ucis = []
        for san in sans:
            move = board.parse_san(san)
            ucis.append(move.uci())
            board.push(move)
        end = comparison_point(fen, ucis)
        assert end is not None
        ends.append(board_facts(end, color))
    return fact_differences(*ends, limit=10, start=board_facts(chess.Board(fen), color))


def test_no_castling_right_is_lost_when_there_was_none_to_lose():
    # 20...Kg7 steps off g8 with no castling rights left: the lines leave the king on g8 and on g7
    fen = "r4rk1/ppp4p/3p4/3P2pK/5qN1/3b3P/PPP3P1/R2Q3R b - - 1 20"
    best = ["Qf7+", "Kxg5", "Qe7+", "Nf6+", "Qxf6+", "Kg4", "Qf4+", "Kh5"]
    refutation = ["Kg7", "Qxd3", "Qf7+", "Kxg5", "Qe7+", "Kh5", "Qe8+", "Kh4"]
    assert "you lose the right to castle" not in _facts_along(fen, best, refutation, "black")
    # without the start the rule falls back on the best line's end, which has no right either
    b_end = board_facts(chess.Board("r4rk1/ppp4p/3p1q2/3P4/6K1/3b3P/PPP3P1/R2Q3R b - - 1 23"), "black")
    r_end = board_facts(chess.Board("r4r2/ppp1q1kp/3p4/3P3K/6N1/3Q3P/PPP3P1/R6R b - - 2 23"), "black")
    assert fact_differences(b_end, r_end) == []
    # a right you had, used to castle in the best line and lost by a king move in the refutation, is lost
    start = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 8"
    assert "you lose the right to castle" in _facts_along(start, ["O-O", "O-O"], ["Kf8", "O-O"], "black")
    assert "you lose the right to castle" not in _facts_along(start, ["Kf8", "O-O"], ["Ke7", "O-O"], "black")


def test_facts_already_true_before_your_move_are_not_blamed_on_it():
    # 7...Bd6 in a real game: Black's c-pawns were doubled before the move, and the best line (7...Nxe5 8.Nxe5 Bd6
    # 9.Qa4+ c6 10.Nxc4) wins one of them back: "you get doubled pawns" would be wrong
    fen = "r2qkb1r/ppp2ppp/2n1bn2/4B3/2pP4/P1N2N2/1P2PPPP/R2QKB1R b KQkq - 2 7"
    facts = _facts_along(fen, ["Nxe5", "Nxe5", "Bd6", "Qa4+", "c6", "Nxc4", "Bc7"],
                         ["Bd6", "Bxf6", "Qxf6", "d5", "Ne5", "dxe6", "O-O-O"], "black")
    assert facts == ["you lose the bishop pair"]
    # an isolated h5 pawn the best line gives up (40...Rf7 41.Rah1 Rg8 42.Rxh5) is not one 40...Rg7 gives you
    fen = "1k5r/1p5r/2q5/p1p1pBPp/PnPpPp2/3P1K2/2P2Q1R/R7 b - - 1 40"
    facts = _facts_along(fen, ["Rf7", "Rah1", "Rg8", "Rxh5", "Rxf5", "Rh6", "Rfxg5"],
                         ["Rg7", "Qh4", "Rhg8", "Rg1", "Qxa4", "g6", "Qa3"], "black")
    assert not any("isolated" in f for f in facts)
    # a new isolated pawn is still named (6...h6 7.Bxf6 gxf6: h6 and the doubled f-pawns)
    fen = "rnb1kb1r/pp1pqppp/5n2/2ppP1B1/8/5N2/PPP2PPP/RN1QKB1R b KQkq - 2 6"
    facts = _facts_along(fen, ["Nc6", "Be2", "Nxe5", "O-O", "h6", "Bxf6", "Qxf6"],
                         ["h6", "Bxf6", "gxf6", "Nc3", "fxe5", "Nxd5", "Qd6"], "black")
    assert "you get an isolated pawn on h6" in facts


def test_the_king_in_the_centre_and_holes_only_where_they_mean_something():
    late = "r1bq1rk1/pp3ppp/8/8/8/8/PP3PPP/R1BQK2R w - - 0 15"
    board = chess.Board(late)
    assert board_facts(board, "white").king_in_centre and board_facts(board, "white").king_central
    # an endgame (a queen against two bishops): the king belongs in the centre
    endgame = chess.Board("8/8/6bb/3Q4/P3p2p/3pk3/1P5P/K7 w - - 0 50")
    assert board_facts(endgame, "black").king_central and not board_facts(endgame, "black").king_in_centre
    # "stays" only for a king that was in the centre before your move; a king walked there "ends up" there
    stays = board_facts(board, "white")
    castled = board_facts(chess.Board("r1bq1rk1/pp3ppp/8/8/8/8/PP3PPP/R1BQ1RK1 w - - 0 15"), "white")
    assert "your king stays in the centre" in fact_differences(castled, stays, start=stays)
    assert "your king ends up in the centre" in fact_differences(castled, stays, start=castled)
    # the two ends may lie a move apart around move 12: a king central in both is no difference
    early = board_facts(chess.Board("r1bq1rk1/pp3ppp/8/8/8/8/PP3PPP/R1BQK2R w - - 0 12"), "white")
    assert not early.king_in_centre and fact_differences(early, stays) == []
    # holes in a bare endgame (one pawn left) say nothing
    few = chess.Board("8/8/4K3/p4Pb1/1B2n2k/3B4/8/8 b - - 0 50")
    one_pawn = chess.Board("8/p7/4K3/5Pb1/1B2n2k/3B4/8/8 b - - 0 50")
    assert board_facts(few, "black").holes and board_facts(few, "black").pawns == 1
    assert fact_differences(board_facts(one_pawn, "black"), board_facts(few, "black")) == []


def test_concept_labels_are_plain_words():
    assert all(concepts.LABELS[t] for t in concepts.TERMS)
    assert concepts.LABELS["Bishops"] == "bishop placement" and concepts.LABELS["Mobility"] == "piece activity"
    deltas = concept_deltas({"Bishops": (0.3, 0.3)}, {"Bishops": (0.0, 0.0)}, "white", 24, 24)
    assert deltas[0].label == "bishop placement"


# --------------------------------------------------------------------------- the eval helper
FAKE_NEWER = '''
import sys
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "uci":
        print("id name Stockfish 17"); print("uciok")
    elif cmd == "eval":
        print("NNUE evaluation        +0.25 (white side)")
        print("Final evaluation       +0.31 (white side)")
    elif cmd == "isready":
        print("readyok")
    elif cmd == "quit":
        break
    sys.stdout.flush()
'''


def test_a_stockfish_without_the_table_turns_concepts_off_with_a_note(tmp_path):
    script = tmp_path / "fake_sf17.py"
    script.write_text(FAKE_NEWER, encoding="utf-8")
    with ClassicalEval([sys.executable, str(script)]) as ce:
        assert ce.table(AFTER_5E5) is None
        assert ce.supported is False and ce.name == "Stockfish 17"
        assert ce.table(AFTER_5E5) is None  # no second process
        assert concepts.NEED_SF16.lower() in ce.note.lower() and "Stockfish 17" in ce.note
    assert ClassicalEval(None).table(AFTER_5E5) is None


def test_a_hung_engine_is_given_up_after_the_timeout(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import sys, time\nfor line in sys.stdin:\n    time.sleep(60)\n", encoding="utf-8")
    ce = ClassicalEval([sys.executable, str(script)], timeout=1.0)
    assert ce.table(AFTER_5E5) is None and ce.broken
    ce.close()


@pytest.mark.engine
def test_stockfish_16_eval_table_and_check(stockfish_path):
    with ClassicalEval(stockfish_path) as ce:
        assert ce.table(IN_CHECK) is None and ce.supported is None  # in check: no table, but no verdict yet
        table = ce.table(AFTER_5E5)
        if ce.supported is False:
            pytest.skip(f"{ce.name} has no classical eval table (concepts need Stockfish 16)")
        assert table is not None and table["Threats"] == (-0.86, -0.53)
        assert ce.supported is True and ce.note == ""


# --------------------------------------------------------------------------- settled comparison points
def test_static_exchange_count():
    see = concepts.see
    board = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    assert see(board, chess.Move.from_uci("d1d5")) == 3  # nothing guards the knight
    board = chess.Board("4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1")
    assert see(board, chess.Move.from_uci("d1d5")) == -6  # the pawn takes the queen back
    board = chess.Board("4k3/8/4p3/3p4/4P3/8/8/4K3 w - - 0 1")
    assert see(board, chess.Move.from_uci("e4d5")) == 0  # an even trade
    board = chess.Board("4k3/8/8/3r4/8/8/3R4/3RK3 w - - 0 1")
    assert see(board, chess.Move.from_uci("d2d5")) == 5  # two rooks against one: the rook falls
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert see(board, chess.Move.from_uci("e5d6")) == 1  # en passant takes a pawn
    assert concepts.settled(chess.Board()) and not concepts.settled(chess.Board(IN_CHECK))


def test_a_comparison_point_is_never_in_the_middle_of_a_fork():
    """33...Bg6? 34.Nhxf5+ Nxf5 35.Ng4+ Kh5 36.Nf6+ Kh6 37.Nxd7: the knight forks king and queen on move 36. Read
    after 36...Kh6 the refutation looked better for Black on material (Material +1.86); the comparison point waits
    until the queen has gone."""
    fen = "8/pp1qn1r1/3p3k/2pP1p1b/2P1pP1N/2P1N2P/P4Q1K/6R1 b - - 1 33"
    moves = ["h5g6", "h4f5", "e7f5", "e3g4", "h6h5", "g4f6", "h5h6", "f6d7", "e4e3", "f2e2"]
    board = chess.Board(fen)
    for uci in moves[:7]:
        board.push_uci(uci)
    assert not concepts.settled(board)  # 37.Nxd7 wins the queen
    n = concepts.comparison_ply(fen, moves)
    end = comparison_point(fen, moves)
    assert n is not None and n >= 8 and not end.pieces(chess.QUEEN, chess.BLACK)
    assert concepts.settled(end)


def test_captures_castling_and_holes_along_a_line():
    fen = "r1bqk2r/ppp2pbp/2n1p1p1/3pPn2/3P4/2NBBN2/PPPQ1PPP/R3K2R b KQkq - 4 8"  # 8...Ncxd4?
    line = _uci(["Ncxd4", "Nxd4", "Nxe3", "fxe3", "Bxe5"], fen)
    mine, theirs = concepts.captures(fen, line, "black")
    assert (mine, theirs) == ([chess.KNIGHT, chess.KNIGHT], [chess.PAWN, chess.BISHOP, chess.PAWN])
    start = "rn1qk2r/ppp2ppp/5b2/3P4/4P3/5Q1P/PPP2PP1/RN2KB1R b KQkq - 0 10"
    line = _uci(["Qd6", "c3", "Qb6", "Qe2", "c6", "Na3", "O-O"], start)
    assert concepts.castles_within(start, line, 6, "black") and not concepts.castles_within(start, line, 0, "black", 4)
    # 5...e5? 6.Ndb5 a6 7.Nd6+: the knight lands on d6, a square no black pawn can cover any more
    sicilian = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
    found = concepts.hole_invasion(sicilian, _uci(["e5", "Ndb5", "a6", "Nd6+", "Bxd6"], sicilian), "black")
    assert found == concepts.Invasion(ply=3, piece="knight", square="d6", san="7.Nd6+")
    assert concepts.hole_invasion(sicilian, _uci(["Nf6", "Nxc6", "bxc6", "e5"], sicilian), "black") is None


def test_the_king_is_not_left_in_the_centre_by_a_line_that_castles_next(monkeypatch):
    """10...Qd6 11.c3 Qb6 12.Qe2 c6 13.Na3 O-O: Black castles the move after the comparison point."""
    from chess_insights.coach import explain
    from chess_insights.coach.deep import make_line

    fen = "rn1qk2r/ppp2ppp/5b2/3P4/4P3/5Q1P/PPP2PP1/RN2KB1R b KQkq - 0 10"
    moves, b = [], chess.Board(fen)
    for san in ["Bxb2", "Qb3", "Bxa1", "c3", "O-O", "Be2"]:
        moves.append(b.push_san(san))
    best = make_line(fen, moves, 354, None, 16)
    moves, b = [], chess.Board(fen)
    for san in ["Qd6", "c3", "Qb6", "Qe2", "c6", "Na3", "O-O"]:
        moves.append(b.push_san(san))
    refutation = make_line(fen, moves, -265, None, 16)
    tool = explain._Concepts(None)
    cmp = tool.compare(fen, best, refutation, "black")
    assert not any("centre" in f for f in cmp.facts)
    monkeypatch.setattr(concepts, "castles_within", lambda *args, **kw: False)  # read one move early, it would be
    assert "your king stays in the centre" in tool.compare(fen, best, refutation, "black").facts
    tool.close()
