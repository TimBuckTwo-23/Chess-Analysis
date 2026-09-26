"""Motif detectors: one hand-built position each way per motif, the line roles, and the precision gate against
Lichess's puzzle themes (run with ``pytest -s`` to see the table)."""

import csv
import random
import time
from collections import Counter

import chess
import pytest

from chess_insights.coach import motifs
from chess_insights.coach.motifs import (
    CORE_THEMES, GATE_MIN_TAGGED, GATE_PRECISION, GATED_THEMES, THEME_LABELS, THEMES, detect_line, detect_moves,
    puzzle_motifs, theme_label, training_url,
)
from chess_insights.models import Line
from conftest import FIXTURES

SAMPLE = FIXTURES / "lichess_puzzles_sample.csv"


def themes(fen, moves, side="first", max_plies=8):
    return {m.theme for m in detect_moves(fen, moves, max_plies=max_plies) if m.side == side}


def motif(fen, moves, theme, max_plies=8):
    return next(m for m in detect_moves(fen, moves, max_plies=max_plies) if m.theme == theme)


# (theme, positive (fen, moves), negative (fen, moves)); White moves first in every line.
CASES = [
    ("fork",  # Nc7+ hits king and rook, the rook falls
     ("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", ["d5c7", "e8e7", "c7a8"]),
     ("rb2k3/8/8/3N4/8/8/8/4K3 w - - 0 1", ["d5c7", "b8c7", "e1d2"])),  # the knight lands en prise
    ("pin",  # Bb5 pins the loose knight to the king
     ("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", ["f1b5", "e8d8", "b5c6"]),
     ("4k3/1p6/2n5/8/8/8/8/4KB2 w - - 0 1", ["f1b5", "e8d8"])),  # pinned, but guarded and attacked by an equal
    ("skewer",  # Ra1+, the king steps aside, the rook behind it falls
     ("r7/8/8/k7/8/8/8/1R2K3 w - - 0 1", ["b1a1", "a5b5", "a1a8"]),
     ("r7/8/1n6/k7/8/8/8/1R2K3 w - - 0 1", ["b1a1", "a5b5", "a1a8"])),  # the rook behind is guarded
    ("hangingPiece",  # Qxe5: nothing guards the knight
     ("3qk3/8/8/4n3/8/8/4Q3/4K3 w - - 3 30", ["e2e5", "e8d7"]),
     ("3qk3/8/3p4/4n3/8/8/4Q3/4K3 w - - 3 30", ["e2e5", "d6e5"])),  # the d6 pawn guards it
    ("discoveredAttack",  # Nf5+ opens the long diagonal, Bxh8 wins the queen
     ("7q/8/7k/8/3N4/8/1B6/6K1 w - - 0 1", ["d4f5", "h6g6", "b2h8"]),
     ("7q/8/7k/8/8/N7/1B6/6K1 w - - 0 1", ["a3c4", "h6g6", "b2h8"])),  # the diagonal was open already
    ("backRankMate",
     ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8"]),
     ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", ["h1h8"])),  # mate on the back rank, but the king's escape is covered
    ("discoveredCheck",  # the knight steps off the e-file
     ("4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1", ["e4c5"]),
     ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", ["e1g1"])),  # the castling rook gives the check itself
    ("doubleCheck",
     ("4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1", ["e4f6"]),
     ("4k3/8/8/8/4N3/8/8/K3R3 w - - 0 1", ["e4c5"])),  # only the rook checks
    ("trappedPiece",  # g3 shuts the bishop in, Rxh2
     ("k7/p7/8/8/8/8/5PPb/5K1R w - - 0 1", ["g2g3", "a7a6", "h1h2"]),
     ("k7/p7/8/8/8/8/6Pb/5K1R w - - 0 1", ["g2g3", "a7a6", "h1h2"])),  # ...Bxg3 was safe (no f2 pawn)
    ("smotheredMate",
     ("6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1", ["g5f7"]),
     ("6rk/6p1/8/6N1/8/8/8/6K1 w - - 0 1", ["g5f7", "h8h7"])),  # h7 is free
    ("mateIn1",
     ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8"]),
     ("6k1/5pp1/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8", "g8h7"])),
    ("mateIn2",  # Ra8+ Rd8 Rxd8#
     ("6k1/3r1ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8", "d7d8", "a8d8"]),
     ("6k1/3r1ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8", "d7d8"])),
    ("deflection",  # Qe8+ drags the queen off the seventh rank, Rxa7
     ("6k1/r3q3/8/7Q/8/8/8/R5K1 w - - 0 1", ["h5e8", "e7e8", "a1a7"]),
     ("6k1/r3q3/8/7Q/8/8/8/R5K1 w - - 0 1", ["g1h1", "e7e5", "a1a7"])),  # the queen left of its own accord
    ("attraction",  # Rh8+ Kxh8 Qh5+
     ("6k1/5pp1/8/8/8/8/8/3Q2KR w - - 0 1", ["h1h8", "g8h8", "d1h5"]),
     ("6k1/5pp1/8/8/8/8/8/3Q2KR w - - 0 1", ["h1h8", "g8h8", "d1d2"])),  # the king is not hit on h8
    ("overloading",  # the queen guards d5 and h3: Rxd5 Qxd5 Nxh3
     ("6k1/3q4/8/3n4/8/7b/8/K2R2N1 w - - 0 1", ["d1d5", "d7d5", "g1h3"]),
     ("6k1/3q4/8/3n4/6p1/7b/8/K2R2N1 w - - 0 1", ["d1d5", "d7d5", "g1h3"])),  # g4 guards h3 (and blocks)
    ("advancedPawn",
     ("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1", ["e6e7"]),
     ("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1", ["e5e6"])),
]

# White rooks ladder the king up the board: mate in 3, 4 and 5.
LADDER_5 = ("8/8/8/8/4k3/R7/1R6/7K w - - 0 1",
            ["b2b4", "e4e5", "a3a5", "e5e6", "b4b6", "e6e7", "a5a7", "e7e8", "b6b8"])


@pytest.mark.parametrize("theme,positive,negative", CASES, ids=[c[0] for c in CASES])
def test_hand_built_positive_and_negative(theme, positive, negative):
    assert theme in themes(*positive)
    assert theme not in themes(*negative)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_mate_in_n_counts_the_mating_sides_moves(n):
    fen, moves = LADDER_5
    board = chess.Board(fen)
    for uci in moves[: 2 * (5 - n)]:
        board.push_uci(uci)
    line = moves[2 * (5 - n):]
    found = themes(board.fen(), line, max_plies=len(line))
    assert f"mateIn{n}" in found and {f"mateIn{k}" for k in range(1, 6)} & found == {f"mateIn{n}"}
    assert "backRankMate" not in found  # the escape squares are covered by a rook, not blocked by own pieces
    short = themes(board.fen(), line[:-1], max_plies=len(line))  # one move short: no mate
    assert not {t for t in short if t.startswith("mateIn")}


def test_squares_start_with_the_active_piece():
    fork = motif("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", ["d5c7", "e8e7", "c7a8"], "fork")
    assert fork.ply == 0 and fork.squares[0] == "c7" and set(fork.squares[1:]) == {"a8", "e8"}
    pin = motif("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", ["f1b5", "e8d8", "b5c6"], "pin")
    assert pin.squares == ["b5", "c6", "e8"]
    skewer = motif("r7/8/8/k7/8/8/8/1R2K3 w - - 0 1", ["b1a1", "a5b5", "a1a8"], "skewer")
    assert skewer.ply == 0 and skewer.squares == ["a1", "a5", "a8"]  # the check that set it up
    mate = motif("6k1/3r1ppp/8/8/8/8/8/R5K1 w - - 0 1", ["a1a8", "d7d8", "a8d8"], "backRankMate")
    assert mate.ply == 2 and mate.squares == ["d8", "g8"]


def test_pin_marks_the_piece_that_pins():
    # the piece on e2 pins the loose knight to its king; a bishop behind it on the file (it can't pin along a
    # file), or a rook behind the pinning queen, must not be marked instead
    for fen in ("4k3/8/4n3/8/8/8/4R3/4BK2 w - - 0 1", "4k3/8/4n3/8/8/8/4Q3/4RK2 w - - 0 1"):
        assert motif(fen, ["f1g1", "e8d8"], "pin").squares == ["e2", "e6", "e8"]


def test_a_promoted_pawn_forks_as_its_new_piece():
    # e8=Q hits two guarded minor pieces: a queen attacking a knight and a bishop is not a fork ...
    assert "fork" not in themes("8/4P3/p7/1n3p2/4b3/7K/k7/8 w - - 0 1", ["e7e8q", "a2b2", "h3h4"])
    # ... but e8=N+ forking king and queen is
    fork = motif("8/2q1P1k1/8/8/8/8/8/K7 w - - 0 1", ["e7e8n", "g7g6", "e8c7"], "fork")
    assert fork.squares[0] == "e8" and set(fork.squares[1:]) == {"c7", "g7"}


def test_double_check_and_mate_mark_the_moving_piece_first():
    # Nf6 is double check (knight and the rook behind it) and mate: the knight is the active piece
    found = {m.theme: m for m in detect_moves("3rkb2/3p1p2/8/8/4N3/8/8/K3R3 w - - 0 1", ["e4f6"])}
    assert found["doubleCheck"].squares == ["f6", "e1", "e8"]
    assert found["mateIn1"].squares == ["f6", "e1", "e8"]
    assert found["discoveredCheck"].squares == ["e1", "e8", "f6"]  # the uncovered checker first


def _flip_files(square):
    return chess.square(7 - chess.square_file(square), chess.square_rank(square))


def test_mirrored_boards_give_the_mirrored_motifs():
    """Every detector is blind to colour and to the side of the board: the same puzzle with the colours swapped
    (board mirrored top to bottom), or flipped left to right (when nobody can castle), gives the same themes,
    plies and sides, and the mirrored squares (the same active piece first; its targets in any order)."""

    def key(theme, ply, side, squares):
        return theme, ply, side, squares[:1], frozenset(squares)

    transforms = (("colours", chess.square_mirror, lambda b: b.mirror()),
                  ("files", _flip_files, lambda b: b.transform(chess.flip_horizontal)))
    mismatches = []
    for row in _rows():
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        board.push_uci(moves[0])
        solution = moves[1:]
        original = detect_moves(board.fen(), solution, max_plies=len(solution))
        for name, square_map, board_map in transforms:
            if name == "files" and board.castling_rights:
                continue
            expected = [key(m.theme, m.ply, m.side, [chess.square_name(square_map(chess.parse_square(s)))
                                                     for s in m.squares]) for m in original]
            moved = [chess.Move(square_map(mv.from_square), square_map(mv.to_square), mv.promotion).uci()
                     for mv in map(chess.Move.from_uci, solution)]
            got = [key(m.theme, m.ply, m.side, m.squares)
                   for m in detect_moves(board_map(board).fen(), moved, max_plies=len(solution))]
            if sorted(got) != sorted(expected):
                mismatches.append((name, row["PuzzleId"]))
    assert not mismatches, mismatches[:10]


def test_castling_in_either_notation_and_chess960():
    # O-O then Rf8 mate: e1g1 (standard) and e1h1 (king takes rook, as Chess960 engines write it) both work
    for castle in ("e1g1", "e1h1"):
        assert themes("7k/p5pp/8/8/8/8/8/4K2R w K - 0 1", [castle, "a7a6", "f1f8"]) == {"backRankMate", "mateIn2"}
    # a Chess960 position (king c1, rooks b1 and g1), with X-FEN or Shredder-FEN castling rights
    for rights in ("K", "G"):
        fen = f"7k/p5pp/8/8/8/8/8/1RK3R1 w {rights} - 0 1"
        assert themes(fen, ["c1g1", "a7a6", "f1f8"]) == {"backRankMate", "mateIn2"}
    assert "discoveredCheck" not in themes("5k2/8/8/8/8/8/8/4K2R w K - 0 1", ["e1h1"])  # the rook checks itself


def test_real_game_lines():
    """Two error positions from real chess.com games (tests/fixtures/real_chesscom_games.json), with short
    Stockfish lines, checked by hand."""
    # 19.Rd2? ...Rc5 traps the queen on c3: every square it can reach is covered, and ...Rxc2 wins it
    refutation = Line(fen="3r2k1/bpp1qppp/2n3b1/p2rp3/8/PPQPPN2/1B2BPPP/3RR1K1 w - - 2 19",
                      moves_uci=["d1d2", "d5c5", "c3c2", "c5c2", "d2c2"])
    [trapped] = detect_line(refutation, "refutation")
    assert (trapped.theme, trapped.ply, trapped.side) == ("trappedPiece", 1, "opponent")
    assert trapped.squares == ["c5", "c3"]
    # 50...Nxh4+ was the move: the rook on g3 pins the bishop to its king, so it can't take the knight
    best = Line(fen="8/p2R2p1/kp2R1K1/5nB1/1Pr2P1P/3p2r1/P7/8 b - - 8 50",
                moves_uci=["f5h4", "g6h7", "h4f3", "e6d6", "d3d2", "a2a3"])
    pin = next(m for m in detect_line(best, "best") if m.theme == "pin")
    assert (pin.ply, pin.side, pin.squares) == (0, "you", ["g3", "g5", "g6"])


def test_sides_and_one_motif_per_theme_and_side():
    # Black to move; Black's blunder ...Ra6 lets White fork king and rook
    found = detect_moves("r3k3/8/8/3N4/8/8/8/4K3 b - - 0 1", ["a8a6", "d5c7", "e8e7", "c7a6"])
    assert [(m.theme, m.side, m.ply) for m in found if m.theme == "fork"] == [("fork", "second", 1)]
    assert all(m.line == "" for m in found)
    assert len({(m.theme, m.side) for m in found}) == len(found)


def test_hanging_piece_is_exchange_aware():
    # ...Nxe5 took a knight: Qxe5 is a recapture, not a hanging piece
    assert "hangingPiece" not in themes("3qk3/8/2n5/4N3/8/8/4Q3/4K3 b - - 0 1", ["c6e5", "e2e5"], side="second")
    # without the move before, a capture straight after a capture that only gets back to level material is not
    # called hanging either (halfmove clock 0: the last move may have been a capture) ...
    assert "hangingPiece" not in themes("3qk3/8/8/4n3/8/8/4Q3/4K3 w - - 0 30", ["e2e5", "e8d7"])
    # ... but it is when the last move can't have been a capture
    assert "hangingPiece" in themes("3qk3/8/8/4n3/8/8/4Q3/4K3 w - - 3 30", ["e2e5", "e8d7"])
    # the material goes straight back: Qxe5 Qxa4 (the knight on a4 was loose too)
    fen = "k7/3q4/8/4n3/N7/8/4Q3/6K1 w - - 3 30"
    assert "hangingPiece" not in themes(fen, ["e2e5", "d7a4", "g1h1"])
    assert "hangingPiece" in themes(fen, ["e2e5", "d7d6", "g1h1"])


def test_previous_position_settles_recaptures():
    """With the position before the opponent's last move, the first move's capture is judged like any later one:
    a recapture after a trade is not a hanging piece, whatever the material; a piece left loose is."""
    line = ["e2e5", "e8d7", "e1d2"]
    # ...Nxe5 took a knight and White, a pawn up, recaptures: the material rule alone takes it for a free piece
    before = "q3k3/8/2n5/4N3/P7/8/4Q3/4K3 b - - 0 30"
    after = "q3k3/8/8/4n3/P7/8/4Q3/4K3 w - - 0 31"
    assert "hangingPiece" in themes(after, line)
    assert "hangingPiece" not in {m.theme for m in detect_moves(after, line, previous_fen=before)}
    assert "hangingPiece" not in {m.theme for m in detect_line(Line(fen=after, moves_uci=line), "best", before)}
    # ...h6 (a pawn move: halfmove clock 0) left the knight loose: the material rule is too careful here
    before = "q3k3/7p/8/4n3/8/8/4Q3/4K3 b - - 3 30"
    after = "q3k3/8/7p/4n3/8/8/4Q3/4K3 w - - 0 31"
    assert "hangingPiece" not in themes(after, line)
    assert "hangingPiece" in {m.theme for m in detect_moves(after, line, previous_fen=before)}
    # a previous position that doesn't lead here, or can't be read, is ignored
    for junk in (chess.STARTING_FEN, "not a fen"):
        assert {m.theme for m in detect_moves(after, line, previous_fen=junk)} == themes(after, line)


def test_detect_line_roles():
    best = Line(fen="r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", moves_uci=["d5c7", "e8e7", "c7a8"])
    [fork] = [m for m in detect_line(best, "best") if m.theme == "fork"]
    assert (fork.line, fork.side, fork.ply) == ("best", "you", 0)

    # the refutation starts with your move (...Ra6) and holds only the opponent's patterns
    refutation = Line(fen="r3k3/8/8/3N4/8/8/8/4K3 b - - 0 1", moves_uci=["a8a6", "d5c7", "e8e7", "c7a6"])
    found = detect_line(refutation, "refutation")
    assert {(m.theme, m.side, m.line) for m in found} >= {("fork", "opponent", "refutation")}
    assert all(m.side == "opponent" and m.ply % 2 == 1 for m in found)
    assert not [m for m in detect_line(refutation, "best") if m.theme == "fork"]

    # ...b6?? allows Ra8 mate
    mated = Line(fen="6k1/1p3ppp/8/8/8/8/8/R5K1 b - - 0 1", moves_uci=["b7b6", "a1a8"])
    assert {m.theme for m in detect_line(mated, "refutation")} == {"backRankMate", "mateIn1"}

    # a mate in 5 takes nine plies: detect_line looks for mates beyond the eight plies it scans for patterns
    assert "mateIn5" in {m.theme for m in detect_line(Line(fen=LADDER_5[0], moves_uci=LADDER_5[1]), "best")}
    assert "mateIn5" not in themes(*LADDER_5)  # detect_moves' default stops at eight
    # a move after the mate is illegal and ignored
    ladder = Line(fen=LADDER_5[0], moves_uci=LADDER_5[1] + ["h1g2"])
    assert "mateIn5" in {m.theme for m in detect_line(ladder, "best")}


def test_unknown_role_is_an_error():
    with pytest.raises(ValueError):
        detect_line(Line(fen=chess.STARTING_FEN, moves_uci=["e2e4"]), "played")


def test_bad_input_gives_no_motifs():
    assert detect_moves("not a fen", ["e2e4"]) == []
    assert detect_moves(chess.STARTING_FEN, []) == []
    assert detect_moves(chess.STARTING_FEN, ["e2e5", "e7e5"]) == []  # stops at the illegal first move
    assert detect_moves(chess.STARTING_FEN, ["zz"]) == []
    assert detect_line(Line(fen="", moves_uci=["e2e4"]), "best") == []
    assert puzzle_motifs(chess.STARTING_FEN, []) == []
    assert puzzle_motifs(chess.STARTING_FEN, ["e2e5"]) == []
    # no line from the engine, or junk in it: no motifs, never an exception
    assert detect_line(None, "best") == [] and detect_line(None, "refutation") == []
    assert detect_line(Line(fen=chess.STARTING_FEN, moves_uci=None), "best") == []
    assert detect_moves(chess.STARTING_FEN, ["e2e4", None, 5]) == []
    assert detect_moves(chess.STARTING_FEN, ["e2e4", "0000", "e7e5"]) == []  # a null move ends the line
    assert detect_moves(chess.STARTING_FEN, ["e4", "e5"]) == []  # SAN is not UCI
    assert puzzle_motifs(chess.STARTING_FEN, ["e2e4", None]) == []
    assert puzzle_motifs(chess.STARTING_FEN, [None]) == []


def test_labels_and_links():
    assert set(THEME_LABELS) == set(THEMES) and set(CORE_THEMES) <= set(THEMES)
    assert THEME_LABELS["backRankMate"] == "back-rank mate" and THEME_LABELS["mateIn2"] == "mate in 2"
    assert training_url("hangingPiece") == "https://lichess.org/training/hangingPiece"
    assert GATED_THEMES <= set(THEMES) and "overloading" not in GATED_THEMES
    assert theme_label("fork") == "fork" and theme_label("overloading") == "tactic"
    assert theme_label("unknownTheme") == "tactic"


# ---------------------------------------------------------------------------
# The precision gate on real Lichess puzzles (CC0)
# ---------------------------------------------------------------------------
def _rows():
    with open(SAMPLE, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _score(rows, find):
    tagged, hits, labelled = Counter(), Counter(), Counter()
    for row in rows:
        found = find(row)
        labels = set(row["Themes"].split())
        for theme in THEMES:
            labelled[theme] += theme in labels
            if theme in found:
                tagged[theme] += 1
                hits[theme] += theme in labels
    return tagged, hits, labelled


def _precision(tagged, hits, labelled, theme):
    return hits[theme] / tagged[theme] if tagged[theme] and labelled[theme] else None


def _table(title, tagged, hits, labelled):
    out = [title, f"{'theme':18} {'tagged':>6} {'precision':>9} {'lichess':>7} {'recall':>6}  gated"]
    for theme in THEMES:
        p = _precision(tagged, hits, labelled, theme)
        r = hits[theme] / labelled[theme] if labelled[theme] else None
        out.append(f"{theme:18} {tagged[theme]:6} {'-' if p is None else f'{p:.2f}':>9} {labelled[theme]:7} "
                   f"{'-' if r is None else f'{r:.2f}':>6}  {'yes' if theme in GATED_THEMES else ''}")
    return "\n".join(out)


@pytest.fixture(scope="module")
def puzzle_scores():
    rows = _rows()
    assert len(rows) > 1000
    return _score(rows, lambda row: {m.theme for m in puzzle_motifs(row["FEN"], row["Moves"].split())})


def test_precision_gate_on_lichess_puzzles(puzzle_scores):
    """Of the puzzles we tag with a core theme, at least 80% carry Lichess's tag. The solver's motifs only, on
    the position after the opponent's move (Moves[0]); recall is printed, not asserted."""
    tagged, hits, labelled = puzzle_scores
    print("\n" + _table(f"Motif detectors vs Lichess themes ({SAMPLE.name})", tagged, hits, labelled))
    for theme in CORE_THEMES:
        assert tagged[theme] >= GATE_MIN_TAGGED, theme
        assert _precision(tagged, hits, labelled, theme) >= GATE_PRECISION, theme


def test_gated_themes_are_exactly_those_that_pass(puzzle_scores):
    tagged, hits, labelled = puzzle_scores
    passing = {
        t for t in THEMES
        if tagged[t] >= GATE_MIN_TAGGED and (_precision(tagged, hits, labelled, t) or 0.0) >= GATE_PRECISION
    }
    assert GATED_THEMES == passing


def test_refutation_view_keeps_the_gate():
    """The coaching reads a refutation from the position before your move: that is a Lichess puzzle as stored
    (FEN before the opponent's blunder, then the punishment), so detect_line's opponent motifs must pass too."""
    rows = _rows()

    def find(row):
        line = Line(fen=row["FEN"], moves_uci=row["Moves"].split())
        return {m.theme for m in detect_line(line, "refutation")}

    tagged, hits, labelled = _score(rows, find)
    print("\n" + _table("Refutation view (detect_line, opponent's motifs)", tagged, hits, labelled))
    for theme in CORE_THEMES:
        assert tagged[theme] >= GATE_MIN_TAGGED and _precision(tagged, hits, labelled, theme) >= GATE_PRECISION, theme


def test_best_line_with_the_previous_position():
    """The best line when the caller passes the position before the opponent's last move (the puzzle's FEN):
    the recapture question is settled, so hangingPiece loses its guesswork, and nothing else changes."""
    rows = _rows()

    def find(row):
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        board.push_uci(moves[0])
        line = Line(fen=board.fen(), moves_uci=moves[1:])
        with_previous = {m.theme for m in detect_line(line, "best", previous_fen=row["FEN"])}
        assert with_previous - {"hangingPiece"} == {m.theme for m in detect_line(line, "best")} - {"hangingPiece"}
        return with_previous

    tagged, hits, labelled = _score(rows, find)
    print("\n" + _table("Best line with the previous position (detect_line, your motifs)", tagged, hits, labelled))
    assert _precision(tagged, hits, labelled, "hangingPiece") >= 0.95
    assert hits["hangingPiece"] / labelled["hangingPiece"] >= 0.95


def test_detect_line_speed():
    """The motif profile runs about 5,000 lines: an eight-ply line must take well under 5 ms."""
    rng = random.Random(20260926)
    lines = []
    for row in _rows()[:400]:
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        for uci in moves:
            board.push_uci(uci)
        while len(moves) < 8 and not board.is_game_over():
            move = rng.choice(sorted(board.legal_moves, key=chess.Move.uci))
            board.push(move)
            moves.append(move.uci())
        if len(moves) >= 8:
            lines.append(Line(fen=row["FEN"], moves_uci=moves[:8]))
    assert len(lines) > 200
    detect_line(lines[0], "best")
    runs = []
    for _ in range(3):  # best of three: the machine may be busy with other work
        start = time.perf_counter()
        for line in lines:
            detect_line(line, "best")
            detect_line(line, "refutation")
        runs.append((time.perf_counter() - start) / (2 * len(lines)))
    per_line = min(runs)
    print(f"\ndetect_line: {per_line * 1000:.2f} ms per 8-ply line")
    assert per_line < 0.005


def test_module_constants_consistent():
    assert motifs.MATE_PLIES >= 2 * motifs.MAX_MATE_IN
    assert motifs.LINE_PLIES == 8
