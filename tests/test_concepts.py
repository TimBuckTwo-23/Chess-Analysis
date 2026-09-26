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


def test_material_term_is_piece_placement_when_material_is_level():
    best, ref = {"Material": (0.5, 0.5)}, {"Material": (0.0, 0.0)}
    assert concept_deltas(best, ref, "white", 20, 20)[0].label == "material"
    assert concept_deltas(best, ref, "white", 20, 20, material_level=True)[0].label == "piece placement"


# --------------------------------------------------------------------------- where the lines are compared
def _uci(sans: list[str], fen: str = chess.STARTING_FEN) -> list[str]:
    board, out = chess.Board(fen), []
    for san in sans:
        out.append(board.push_san(san).uci())
    return out


def test_comparison_point_waits_for_a_quiet_position():
    # after 2 plies (1.e4 d5) White can take on d5: not quiet, so the point moves on to the first quiet position
    end = comparison_point(chess.STARTING_FEN, _uci(["e4", "d5", "exd5", "Qxd5", "Nc3"]), plies=2)
    assert end is not None and end.fen().split()[0] == "rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR"
    # a check is never a comparison point (Stockfish prints no table in check): walk back when nothing follows
    end = comparison_point(chess.STARTING_FEN, _uci(["e4", "d5", "Bb5+"]), plies=3)
    assert end is not None and end.fen().split()[0] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
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
