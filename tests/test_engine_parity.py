"""Parity of engine.py's Lichess formulas with lila / scalachess, vector by vector.

The reference functions below (``ref_*``) are a verbatim Python port of lila and scalachess
(``AccuracyPercent.scala``, ``Advice.scala``, ``WinPercent`` in ``eval.scala``,
``AccuracyCP.scala`` and ``Divider.scala``, read from ``master`` on 2026-09-25). The port
passes lila's own ``AccuracyPercentTest`` and scalachess's ``DividerTest``, whose vectors
are reproduced here, so this file needs nothing outside the repository at run time.

Where engine.py deliberately differs from Lichess, a test named ``test_deviation_*``
pins down exactly how.
"""

from __future__ import annotations

import io
import math
import random
from typing import Optional, Sequence

import chess
import chess.engine
import chess.pgn
import pytest

from chess_insights import engine
from chess_insights.engine import (
    CP_LOSS_CAP,
    MATE_CP,
    PositionEval,
    build_game_eval,
    divide,
    game_accuracy,
    game_phases,
    judge_positions,
    move_accuracy,
    win_percent,
)
from factories import make_game

# =========================================================================== reference (lila / scalachess)
REF_CEILING = 1000
REF_INITIAL = 15  # chess.eval.Eval.Cp.initial
REF_MULTIPLIER = -0.00368208


def ref_winning_chances(cp: float) -> float:  # [-1, 1], does not clamp cp
    return max(-1.0, min(1.0, 2 / (1 + math.exp(REF_MULTIPLIER * cp)) - 1))


def ref_win_percent(score) -> float:
    """WinPercent.fromScore: cp clamped to +-1000; ("mate", n) -> +-1000 by signum (n == 0 counts as negative)."""
    if isinstance(score, tuple):
        cp = REF_CEILING if score[1] > 0 else -REF_CEILING
    else:
        cp = max(-REF_CEILING, min(REF_CEILING, score))
    return 50 + 50 * ref_winning_chances(cp)


def ref_move_accuracy(before: float, after: float) -> float:
    if after >= before:
        return 100.0
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * (before - after)) + -3.166924740191411
    return max(0.0, min(100.0, raw + 1))


def _pstdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def ref_game_accuracy(cps: Sequence[int], start_white: bool = True, initial_cp: int = REF_INITIAL) -> dict:
    """AccuracyPercent.gameAccuracy on White-POV cps after each ply -> {"white": ..., "black": ...} (per colour)."""
    wp = [ref_win_percent(c) for c in [initial_cp, *cps]]
    size = max(2, min(8, len(cps) // 10))
    windows = [wp[:size]] * max(0, min(size, len(wp)) - 2)
    windows += [wp[i : i + size] for i in range(len(wp) - size + 1)] if len(wp) >= size else [wp]
    weights = [max(0.5, min(12.0, _pstdev(w))) for w in windows]
    per = {True: [], False: []}
    for i, weight in zip(range(len(wp) - 1), weights):
        white = (i % 2 == 0) == start_white
        acc = ref_move_accuracy(wp[i], wp[i + 1]) if white else ref_move_accuracy(wp[i + 1], wp[i])
        per[white].append((acc, weight))
    out = {}
    for colour, name in ((True, "white"), (False, "black")):
        pairs = per[colour]
        if not pairs:
            out[name] = None
            continue
        weighted = sum(a * w for a, w in pairs) / sum(w for _, w in pairs)
        harmonic = len(pairs) / sum(1 / max(1, a) for a, _ in pairs)
        out[name] = (weighted + harmonic) / 2
    return out


def ref_advice(prev, cur, mover_white: bool, played_is_best: bool = False) -> Optional[str]:
    """lila Advice (CpAdvice orElse MateAdvice), only when the move differs from the engine's best."""
    if played_is_best:
        return None
    if not isinstance(prev, tuple) and not isinstance(cur, tuple):  # CpAdvice: unclamped cp
        d = ref_winning_chances(cur) - ref_winning_chances(prev)
        delta = -d if mover_white else d
        return next((name for t, name in ((0.3, "blunder"), (0.2, "mistake"), (0.1, "inaccuracy")) if t <= delta), None)

    def pov(s):
        if isinstance(s, tuple):
            return ("mate", s[1] if mover_white else -s[1])
        return s if mover_white else -s

    p, c = pov(prev), pov(cur)
    p_cp = 0 if isinstance(p, tuple) else p
    c_cp = 0 if isinstance(c, tuple) else c
    if not isinstance(p, tuple) and isinstance(c, tuple) and c[1] < 0:  # MateCreated
        return "inaccuracy" if p_cp < -999 else "mistake" if p_cp < -700 else "blunder"
    if isinstance(p, tuple) and p[1] > 0 and (not isinstance(c, tuple) or c[1] < 0):  # MateLost
        return "inaccuracy" if c_cp > 999 else "mistake" if c_cp > 700 else "blunder"
    return None


def ref_acpl_diffs(cps: Sequence[int], mover_white: bool, start_white: bool = True) -> list[int]:
    """AccuracyCP: per-move cp loss (evals ceiled at +-1000 before differencing, floored at 0, no other cap)."""
    seq = ([REF_INITIAL] if mover_white == start_white else []) + list(cps)
    out = []
    for j in range(0, len(seq) - 1, 2):
        s1, s2 = (max(-REF_CEILING, min(REF_CEILING, c)) for c in seq[j : j + 2])
        out.append(max(0, (s2 - s1) * (-1 if mover_white else 1)))
    return out


_REF_REGIONS = [(0x0303 << (x + 8 * y), y + 1) for y in range(7) for x in range(7)]


def _ref_score(y: int, white: int, black: int) -> int:
    if white == 0:
        if black == 1:
            return 1 + y
        if black == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black in (3, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0
    if white == 1:
        return {0: 1 + (8 - y), 1: 5 + abs(4 - y), 2: 4 + (7 - y), 3: 5 + (7 - y)}.get(black, 0)
    if white == 2:
        if black == 0:
            return 2 + (y - 2) if y > 2 else 0
        return {1: 4 + (y - 1), 2: 7}.get(black, 0)
    if white == 3:
        if black == 0:
            return 3 + (y - 1) if y > 1 else 0
        return 5 + (y - 1) if black == 1 else 0
    if white == 4:
        return 3 + (y - 1) if black == 0 and y > 1 else 0
    return 0


def ref_divide(boards: Sequence[chess.Board]) -> tuple[Optional[int], Optional[int]]:
    def majors_minors(b):
        return chess.popcount(b.occupied & ~(b.kings | b.pawns))

    def sparse(b):
        return chess.popcount(0xFF & b.occupied_co[chess.WHITE]) < 4 or chess.popcount(
            (0xFF << 56) & b.occupied_co[chess.BLACK]
        ) < 4

    def mixed(b):
        w, bl = b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK]
        return sum(_ref_score(y, chess.popcount(w & r), chess.popcount(bl & r)) for r, y in _REF_REGIONS)

    mid = next((i for i, b in enumerate(boards) if majors_minors(b) <= 10 or sparse(b) or mixed(b) > 150), None)
    end = next((i for i, b in enumerate(boards) if majors_minors(b) <= 6), None) if mid is not None else None
    return (mid if mid is not None and (end is None or mid < end) else None), end


def ref_phase_of_ply(ply_after: int, middle: Optional[int], end: Optional[int]) -> str:
    """lila insight Phase.of: ``ply_after`` is the ply number AFTER the move (1 for White's first move)."""
    if middle is None:
        return "opening"
    if ply_after < middle:
        return "opening"
    if end is not None and ply_after >= end:
        return "endgame"
    return "middlegame"


# =========================================================================== lila / scalachess test vectors
# lila modules/analyse/src/test/AccuracyPercentTest.scala: (name, cps after each ply, white first,
# expected white, tolerance, expected black, tolerance)
ACCURACY_VECTORS = [
    ("two good moves", [15, 15], True, 100, 1, 100, 1),
    ("white blunders on first move", [-900, -900], True, 10, 5, 100, 1),
    ("black blunders on first move", [15, 900], True, 100, 1, 10, 5),
    ("both blunder on first move", [-900, 0], True, 10, 5, 10, 5),
    ("20 perfect moves", [15] * 20, True, 100, 1, 100, 1),
    ("20 perfect moves and a white blunder", [15] * 20 + [-900], True, 50, 5, 100, 1),
    ("21 perfect moves and a black blunder", [15] * 21 + [900], True, 100, 1, 50, 5),
    ("5 average moves (65 cpl) on each side", [-50, 15] * 5, True, 76, 8, 76, 8),
    ("50 average moves (65 cpl) on each side", [-50, 15] * 50, True, 76, 8, 76, 8),
    ("50 mediocre moves (150 cpl) on each side", [-135, 15] * 50, True, 54, 8, 54, 8),
    ("50 terrible moves (500 cpl) on each side", [-435, 15] * 50, True, 20, 8, 20, 8),
    ("black moves first, two good moves", [15, 15], False, 100, 1, 100, 1),
    ("black moves first, black blunders on first move", [900, 900], False, 100, 1, 10, 5),
    ("black moves first, white blunders on first move", [15, -900], False, 10, 5, 100, 1),
    ("black moves first, both blunder on first move", [900, 0], False, 10, 5, 10, 5),
]

# scalachess test-kit/src/test/scala/DividerTest.scala: (moves, middlegame bounds, endgame bounds or None).
# Bounds are for the Divider run on the positions BEFORE each move, as that test does.
DIVIDER_GAMES = [
    (
        "1. e3 g6 2. d4 Bg7 3. Nf3 Nf6 4. Bd3 O-O 5. O-O b6 6. c4 Bb7 7. Nbd2 d5 8. b3 Nbd7 9. Bb2 Re8 10. Qc2 dxc4 "
        "11. bxc4 c5 12. d5 e5 13. e4 h5 14. a4 Nf8 15. h3 Qd6 16. Nxe5 Rxe5 17. Nf3 N6d7 18. Nxe5 Bxe5 19. Bxe5 Nxe5 "
        "20. Be2 Bc8 21. f4 Ned7 22. e5 Qe7 23. Bf3 Rb8 24. Rae1 f5 25. d6 Qh4 26. e6 Nxe6 27. Rxe6 Nf6 28. Ree1 Qxf4 "
        "29. Bd5+ Nxd5 30. Rxf4 Nxf4 31. Qd2 g5 32. d7 Bb7 33. d8=Q+ Rxd8 34. Qxd8+ Kh7 35. Qc7+ Kh6 36. Qxb7 g4 "
        "37. Qc6+ Ng6 38. Re6 gxh3 39. Rxg6+ Kh7 40. Rh6+ Kg7 41. Qf6+ Kg8 42. Rh8#",
        (18, 40),
        (50, 65),
    ),
    (
        "1. e4 c5 2. Nf3 d6 3. Bc4 Nf6 4. d3 g6 5. c3 Bg7 6. Bg5 O-O 7. h3 Nc6 8. Nbd2 a6 9. Bb3 b5 10. Bc2 Bb7 "
        "11. O-O Nd7 12. Nh2 f6 13. Be3 e5 14. Ndf3 Ne7 15. Qd2 f5 16. Qe2 h6 17. Bd2 g5 18. g4 f4 19. Bb3+ d5 "
        "20. exd5 Bxd5 21. c4 Bf7 22. cxb5 axb5 23. Bxf7+ Rxf7 24. Bc3 Ng6 25. a3 b4 26. axb4 Rxa1 27. Rxa1 cxb4 "
        "28. Bxb4 Qb6 29. Bc3 Re7 30. Ra8+ Kh7 31. Kg2 Nc5 32. Qc2 Qb3 33. Qxb3 Nxb3 34. Rb8 Nc5 35. Rb5 Nxd3 "
        "36. Nf1 e4 37. Ng1 Nh4+ 38. Kh2 Bxc3 39. bxc3 Nxf2 40. Rd5 e3 41. Ne2 Nf3+ 42. Kg2 Ne1+ 43. Kh2 f3 "
        "44. Neg3 e2 45. Nd2 Ng2 46. Nxf3 e1=Q 47. Nxe1 Nxe1 48. c4 Nc2 49. c5 Ne4 50. c6 Nxg3 51. Kxg3 Re3+ "
        "52. Kf2 Rc3 53. Rd7+ Kg6 54. c7 Nb4 55. Rd6+ Kf7 56. Rxh6 Rxc7 57. Rh7+ Ke6 58. Rxc7 Nd3+ 59. Kf3 Nf4 "
        "60. Kg3 Ne2+ 61. Kf2 Nf4 62. Kg3 Ne2+ 63. Kh2 Nf4 64. Rc5 Kf6 65. Rf5+ Kg6 66. Kg3 Ne2+ 67. Kf3 Nd4+ "
        "68. Ke4 Nxf5 69. gxf5+ Kf6 70. h4 gxh4 71. Kf4 h3 72. Kg3 Kxf5 73. Kxh3 1/2-1/2",
        (17, 30),
        (65, 80),
    ),
    (
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nc6 5. Nc3 e5 6. Nb3 Nf6 7. f3 Be7 8. Be3 O-O 9. Qd2 b6 10. O-O-O Bb7 "
        "11. g4 Rc8 12. h4 a5 13. h5 Nb4 14. g5 Nd7 15. g6 Nc5 16. h6 Nxb3+ 17. axb3 fxg6 18. hxg7 Rxf3 19. Bxb6 Qxb6 "
        "20. Qh6 Qe3+ 21. Qxe3 Rxe3 22. Bd3 Nxd3+ 23. cxd3 Bg5 24. Kb1 Kxg7 25. Rh2 Bf4 26. Rh4 h5 27. Rhh1 Rf8 "
        "28. Rhg1 g5 29. Rg2 g4 30. Nb5 Rf6 31. Nc7 Rg3 32. Rxg3 Bxg3 33. Ne8+ Kf7 34. Nxf6 Kxf6 35. Rf1+ Kg5 "
        "36. Rf7 Ba6 37. Ra7 Bxd3+ 38. Kc1 Bf4+ 39. Kd1 Bxe4 40. Rxa5 g3 41. Ke2 g2 42. Ra1 h4 43. Kf2 h3 44. b4 Kg4 "
        "45. b5 h2 46. b6 h1=Q 47. Rg1 Be3+ 48. Kxe3 Qxg1+ 49. Kxe4 Qxb6 50. Kd5 g1=Q 51. Ke6 Qgf2 52. Kd5 Qf3+ "
        "53. Ke6 Qbb3+ 54. Ke7 Qff7+ 55. Kxd6 Qbd5# 0-1",
        (20, 28),
        (50, 65),
    ),
    (
        "1. f3 g6 2. e4 Bg7 3. d4 d5 4. Qe2 Bxd4 5. c3 Bg7 6. Nd2 Nc6 7. exd5 Qxd5 8. Ne4 Nf6 9. Nxf6+ Bxf6 "
        "10. Be3 O-O "
        "11. Rd1 Qxa2 12. Qd2 Rd8 13. Qc2 Rxd1+ 14. Kxd1 Bf5 15. Qc1 Rd8+ 16. Ke1 Ne5 17. Kf2 Bd3 18. Ne2 Ba6 "
        "19. Nd4 Bxf1 20. Kxf1 Nd3 21. Qc2 Qa1+ 22. Ke2 Qxh1 23. Qxd3 c5 24. Qb5 Qxg2+ 25. Kd3 Qf1+ 26. Kc2 Qxb5 "
        "27. Nxb5 a6 28. Na3 Rc8 29. Nc4 b5 30. Nd2 h5 31. Ne4 Be5 32. f4 Bd6 33. b3 f5 34. Ng5 e5 35. fxe5 Bxe5 "
        "36. Bg1 Bf4 37. Ne6 Bd6 38. h3 c4 39. b4 Re8 40. Nd4 Be5 41. Nf3 Bg7 42. Nd2 Re2 43. Kd1 Rg2 44. Bc5 Bxc3 "
        "45. Nf3 Rg3 46. Ke2 Rxh3 47. Ng5 Rh2+ 48. Kd1 Bf6 49. Ne6 Rh1+ 50. Ke2 Kf7 51. Nc7 Rh2+ 52. Kf3 g5 "
        "53. Nxa6 g4+ 54. Kg3 Be5# 0-1",
        (16, 34),
        (40, 47),
    ),
    (
        "1. c4 e5 2. Nc3 f5 3. d3 Nf6 4. g3 Bb4 5. Bd2 O-O 6. Bg2 c6 7. Nf3 d5 8. cxd5 Bxc3 9. Bxc3 e4 10. dxe4 Nxe4 "
        "11. dxc6 Nxc6 12. Qb3+ Kh8 13. Rd1 Qc7 14. O-O Nxc3 15. Qxc3 Be6 16. a3 Rac8 17. Nd4 Nxd4 18. Qxd4 b6 "
        "19. e4 fxe4 20. Bxe4 Rcd8 21. Qe3 Qe5 22. b4 Bc4 23. Rfe1 Qh5 24. Rxd8 Rxd8 25. Bf3 Qf5 26. Rd1 Rf8 "
        "27. Bg2 h6 28. Qd4 Qf6 29. Qxc4 Qxf2+ 30. Kh1 Qb2 31. Qd3 h5 32. h4 Qf6 33. Rf1 Qe7 34. Rxf8+ Qxf8 "
        "35. Qf3 Qe7 36. Qxh5+ Kg8 37. Bd5+ 1-0",
        (19, 26),
        (36, 48),
    ),
    (
        "1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. Bd2 Bg7 6. e4 Nxc3 7. Bxc3 O-O 8. Qd2 Nc6 9. Nf3 Bg4 10. d5 Bxf3 "
        "11. Bxg7 Kxg7 12. gxf3 Ne5 13. O-O-O c6 14. Qc3 f6 15. Bh3 cxd5 16. exd5 Nf7 17. f4 Qd6 18. Qd4 Rad8 "
        "19. Be6 Qb6 20. Qd2 Rd6 21. Rhe1 Nd8 22. f5 Nxe6 23. Rxe6 Qc7+ 24. Kb1 Rc8 25. Rde1 Rxe6 26. Rxe6 Rd8 "
        "27. Qe3 Rd7 28. d6 exd6 29. Qd4 Rf7 30. fxg6 hxg6 31. Rxd6 a6 32. a3 Qa5 33. f4 Qh5 34. Qd2 Qc5 35. Rd5 Qc4 "
        "36. Rd7 Qc6 37. Rd6 Qe4+ 38. Ka2 Re7 39. Qc1 a5 40. Qf1 a4 41. Rd1 Qc2 42. Rd4 Re2 43. Rb4 b5 44. Qh1 Re7 "
        "45. Qd5 Re1 46. Qd7+ Kh6 47. Qh3+ Kg7 48. Qd7+ 1/2-1/2",
        (19, 26),
        (36, 48),
    ),
    (
        "1. e4 e5 2. f4 d6 3. Nf3 exf4 4. Bc4 h6 5. O-O Bg4 6. d4 Nc6 7. Bxf4 Nf6 8. Nc3 Be7 9. e5 dxe5 10. dxe5 Nh5 "
        "11. Be3 O-O 12. h3 Bxf3 13. Qxf3 Nxe5 14. Bxf7+ Rxf7 15. Qxh5 Qf8",
        (19, 25),
        None,
    ),
]


def _divider_game(movetext: str):
    pgn = chess.pgn.read_game(io.StringIO(movetext))
    board = pgn.board()
    boards, sans = [board.copy()], []
    for move in pgn.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
        boards.append(board.copy())
    return sans, boards


# =========================================================================== win % and move accuracy
def test_win_percent_matches_scalachess_for_every_cp_and_mate():
    for cp in range(-3000, 3001, 7):
        assert win_percent(cp) == pytest.approx(ref_win_percent(cp), abs=1e-12)
    for n in (-12, -3, -1, 0, 1, 2, 30):
        assert win_percent(None, mate=n) == pytest.approx(ref_win_percent(("mate", n)), abs=1e-12)


def test_move_accuracy_matches_lila_on_a_grid():
    grid = [x / 4 for x in range(0, 401)]
    for before in grid[::8]:
        for after in grid[::5]:
            assert move_accuracy(before, after) == pytest.approx(ref_move_accuracy(before, after), abs=1e-12)


# =========================================================================== game accuracy
def _white_wp(cps, initial=REF_INITIAL):
    return [win_percent(c) for c in [initial, *cps]]


@pytest.mark.parametrize("name,cps,white_first,w,dw,b,db", ACCURACY_VECTORS, ids=[v[0] for v in ACCURACY_VECTORS])
def test_game_accuracy_lila_test_vectors(name, cps, white_first, w, dw, b, db):
    ref = ref_game_accuracy(cps, white_first)
    white = game_accuracy(_white_wp(cps), "white", white_first)
    black = game_accuracy(_white_wp(cps), "black", white_first)
    assert white == pytest.approx(ref["white"], abs=1e-9) and black == pytest.approx(ref["black"], abs=1e-9)
    assert abs(white - w) <= dw and abs(black - b) <= db  # lila's own tolerances


def test_game_accuracy_matches_lila_on_random_games():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(2, 160)
        cps, cp = [], rng.randint(-50, 80)
        for _ in range(n):
            cp += int(rng.gauss(0, 120)) if rng.random() < 0.8 else rng.choice((-900, 900, -3000, 3000))
            cps.append(cp)
        white_first = rng.random() < 0.8
        initial = rng.choice((REF_INITIAL, rng.randint(-300, 300)))
        ref = ref_game_accuracy(cps, white_first, initial)
        for colour in ("white", "black"):
            ours = game_accuracy(_white_wp(cps, initial), colour, white_first)
            assert (ours is None) == (ref[colour] is None)
            if ours is not None:
                assert ours == pytest.approx(ref[colour], abs=1e-9)


def test_deviation_accuracy_is_reported_per_colour():
    """lila returns no accuracy at all unless BOTH sides moved; we give the side that moved its value.
    (Irrelevant in practice: engine analysis needs at least MIN_PLIES plies.)"""
    assert ref_game_accuracy([15])["black"] is None
    assert game_accuracy(_white_wp([15]), "white") == pytest.approx(100.0)
    assert game_accuracy(_white_wp([15]), "black") is None
    assert game_accuracy(_white_wp([]), "white") is None


# =========================================================================== judgements (Advice)
SCORES = [0, 15, -40, 60, 100, -150, 220, 300, -350, 450, 699, 701, -701, 800, 999, 1000, 1001, -1001, 1500, -2000,
          2500, ("mate", 1), ("mate", 3), ("mate", -1), ("mate", -4)]


def _pos(score) -> PositionEval:
    return PositionEval(mate=score[1]) if isinstance(score, tuple) else PositionEval(cp=score)


def test_judgements_match_lila_advice_for_every_pair_of_scores():
    """Includes |cp| > 1000: lila's CpAdvice uses the UNclamped centipawns."""
    for prev in SCORES:
        for cur in SCORES:
            for mover_white in (True, False):
                ours = judge_positions(_pos(prev), _pos(cur), mover_white)
                assert ours == ref_advice(prev, cur, mover_white), (prev, cur, mover_white)


def test_judge_uses_unclamped_centipawns_like_lichess():
    # White drops from +20 pawns to +4.5 pawns: 0.319 winning chances (lila: blunder); clamped at +10 it would be 0.27
    assert ref_advice(2000, 450, True) == "blunder"
    assert judge_positions(PositionEval(cp=2000), PositionEval(cp=450), True) == "blunder"
    # ... and the same swing for Black (White-POV scores)
    assert judge_positions(PositionEval(cp=-2000), PositionEval(cp=-450), False) == "blunder"


# =========================================================================== phases (Divider)
@pytest.mark.parametrize("index", range(len(DIVIDER_GAMES)))
def test_divider_scalachess_test_games(index):
    movetext, mid_bounds, end_bounds = DIVIDER_GAMES[index]
    sans, boards = _divider_game(movetext)
    middle, end = divide(boards[:-1])  # DividerTest's convention: the positions before each move
    assert middle is not None and mid_bounds[0] <= middle <= mid_bounds[1]
    if end_bounds is None:
        assert end is None
    else:
        assert end is not None and end_bounds[0] <= end <= end_bounds[1]
    assert divide(boards) == ref_divide(boards)  # lila's production input: every position, final one included


@pytest.mark.parametrize("index", range(len(DIVIDER_GAMES)))
def test_game_phases_match_lila_phase_of_each_move(index):
    """Phase of a move = the phase of the position AFTER it (lila insight ``Phase.of(division, ply)``)."""
    sans, boards = _divider_game(DIVIDER_GAMES[index][0])
    middle, end = ref_divide(boards)
    expected = [ref_phase_of_ply(i + 1, middle, end) for i in range(len(sans))]
    assert game_phases(make_game(moves_san=sans)) == expected


def test_deviation_phases_when_the_middlegame_is_skipped():
    """Straight into an endgame (middlegame and endgame start on the same position): scalachess reports no
    middlegame, and lila then calls every move "opening". We call the moves from that position on "endgame"."""
    rook_ending = make_game(initial_fen="8/8/8/4k3/8/8/8/R3K3 w - - 0 1", moves_san=["Ra7", "Kd5", "Ke2", "Ke5"])
    boards = [chess.Board(rook_ending.initial_fen)]
    for san in rook_ending.moves_san:
        boards.append(boards[-1].copy())
        boards[-1].push_san(san)
    assert ref_divide(boards) == (None, 0)
    assert [ref_phase_of_ply(i + 1, None, 0) for i in range(4)] == ["opening"] * 4  # lila
    assert game_phases(rook_ending) == ["endgame"] * 4  # us


# =========================================================================== whole games from constructed evaluations
def _game_and_positions(sans, white_cps, bests=None, **kw):
    """A game of ``sans`` and White-POV evaluations of its N + 1 positions (int cp or ("mate", n))."""
    game = make_game(moves_san=sans, **kw)
    board = chess.Board(game.initial_fen or chess.STARTING_FEN)
    positions = []
    for i, score in enumerate(white_cps):
        best = None
        if bests is not None and bests[i] is not None:
            best = board.parse_san(bests[i]).uci()
        elif i < len(sans):
            best = next(m for m in sorted(board.legal_moves, key=lambda m: m.uci()) if board.san(m) != sans[i]).uci()
        mated = board.is_checkmate()
        if mated:
            positions.append(PositionEval(cp=-MATE_CP if board.turn == chess.WHITE else MATE_CP, checkmate=True))
        elif isinstance(score, tuple):
            positions.append(PositionEval(mate=score[1], best=best))
        else:
            positions.append(PositionEval(cp=score, best=best))
        if i < len(sans):
            board.push_san(sans[i])
    return game, positions


SICILIAN = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e5"]


def test_whole_game_judgements_accuracy_and_cp_loss_match_lila():
    rng = random.Random(3)
    for _ in range(40):
        cps = [REF_INITIAL]
        for _ in SICILIAN:
            cps.append(cps[-1] + int(rng.gauss(0, 250)) if rng.random() < 0.85 else rng.choice((-2500, 2500)))
        game, positions = _game_and_positions(SICILIAN, cps)
        ev = build_game_eval(game, positions, "Stockfish 16", 12)
        after = cps[1:]
        for i, p in enumerate(ev.plies):
            assert p.judgement == ref_advice(cps[i], cps[i + 1], p.mover == "white"), (i, cps)
            assert p.win_before == pytest.approx(ref_win_percent(cps[i] if p.mover == "white" else -cps[i]))
        ref = ref_game_accuracy(after)
        assert ev.my_accuracy == pytest.approx(ref["white"], abs=1e-9)
        assert ev.opp_accuracy == pytest.approx(ref["black"], abs=1e-9)
        # ACPL: lila's per-move loss, except that one move is charged at most CP_LOSS_CAP (see the deviation test)
        for colour in ("white", "black"):
            ours = [p.cp_loss for p in ev.plies if p.mover == colour]
            assert ours == [min(CP_LOSS_CAP, d) for d in ref_acpl_diffs(after, colour == "white")]


def test_the_engines_best_move_is_never_judged():
    """lila only judges a move with a variation, i.e. one that differs from the engine's best move."""
    cps = [REF_INITIAL, 20, 2000, -500, -480, -470, -460, -450, -440, -430, -420, -410, -400]
    bests = [None] * 13
    bests[2] = SICILIAN[2]  # 2.Nf3 is Stockfish's choice although the eval then collapses
    game, positions = _game_and_positions(SICILIAN, cps, bests)
    ev = build_game_eval(game, positions)
    assert ref_advice(2000, -500, True) == "blunder" and ev.plies[2].judgement is None
    assert ev.plies[2].best_san == "Nf3"


def test_mating_move_is_left_out_of_game_accuracy_like_lila():
    """fishnet drops the checkmate position (AnalysisBuilder: ``filterNot(_.exists(_.isCheckmate))``), so the
    mating move has no Info: no judgement and no part in the accuracy."""
    scholars = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
    cps = [REF_INITIAL, 30, 25, 40, 20, 60, ("mate", 1), None]
    game, positions = _game_and_positions(scholars, cps)
    assert positions[-1].checkmate
    ev = build_game_eval(game, positions)
    lila_cps = [30, 25, 40, 20, 60, MATE_CP]  # forceAsCp then ceiled; the checkmate eval is dropped
    ref = ref_game_accuracy(lila_cps)
    assert ev.my_accuracy == pytest.approx(ref["white"], abs=1e-9)
    assert ev.opp_accuracy == pytest.approx(ref["black"], abs=1e-9)
    mating = ev.plies[-1]
    assert mating.judgement is None and mating.accuracy == 100.0 and mating.tags == []


def test_deviation_the_start_position_is_evaluated_not_assumed():
    """lila assumes +15 cp for the start position; we use Stockfish's evaluation of it (needed anyway for the
    engine's best first move, and the only sensible choice for games from a set-up position). With the start
    position at +15 the numbers are identical to lila's (tests above); otherwise they equal lila's formula
    with our start evaluation as ``initialCp``."""
    cps = [60] + [40, 45, 30, 35, 20, 25, 10, 5, -10, -5, -20, -25]
    game, positions = _game_and_positions(SICILIAN, cps)
    ev = build_game_eval(game, positions)
    assert ev.my_accuracy == pytest.approx(ref_game_accuracy(cps[1:], True, initial_cp=60)["white"], abs=1e-9)
    assert ev.plies[0].win_before == pytest.approx(win_percent(60))


def test_deviation_cp_loss_is_capped():
    """lila's ACPL charges up to 2000 cp for one move (+1000 to -1000); we cap a move at CP_LOSS_CAP so a single
    hung queen in a won position doesn't dominate a player's average."""
    cps = [REF_INITIAL, 20, 900, -1500, -1400, -1400, -1400, -1400, -1400, -1400, -1400, -1400, -1400]
    game, positions = _game_and_positions(SICILIAN, cps)
    ev = build_game_eval(game, positions)
    assert ref_acpl_diffs(cps[1:], True)[1] == 1900  # White's 2.Nf3: +9 -> -15 pawns, ceiled to +10 -> -10
    assert ev.plies[2].cp_loss == CP_LOSS_CAP == 1000


def test_python_chess_mate_zero_is_the_side_to_move_being_mated():
    """python-chess reports a finished game as Mate(-0) for the side to move; ``PovScore.white()`` of that can be
    ``MateGiven`` (mate() == 0), which carries no sign. A live position reported as mate 0 is scored as lost for
    the side to move, never as a win."""
    board = chess.Board()

    class MatedEngine:
        def analyse(self, board, limit, **kw):
            return {"score": chess.engine.PovScore(chess.engine.Mate(-0), board.turn), "pv": []}

    white = engine.evaluate_position(MatedEngine(), board, chess.engine.Limit(depth=1), object())
    assert white.white_cp == -MATE_CP and white.mate is None and not white.checkmate
    board.push_san("e4")
    black = engine.evaluate_position(MatedEngine(), board, chess.engine.Limit(depth=1), object())
    assert black.white_cp == MATE_CP
