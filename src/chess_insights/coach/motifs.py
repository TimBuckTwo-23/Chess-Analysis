"""Tactical motif detectors on python-chess, named with Lichess's puzzle themes (C1). Owner: motifs.

Our own MIT code: Lichess's tagger (ornicar/lichess-puzzler, AGPL-3.0) was read for definitions only.

A line is a list of moves from a position; the side to move there is "first", the other "second". Each
detector looks at one move of a line and at the few moves around it, the way Lichess tags a puzzle solution:

* fork: a piece (not the king) lands where it attacks two enemy pieces worth more than itself (or loose ones
  that don't guard it), is not itself en prise, and its side moves again afterwards.
* pin: after a move, an enemy piece pinned to its king either can't take a piece it attacks (worth more than
  it, or loose) or can't step away from an attacker cheaper than itself.
* skewer: a line piece captures on a square that the enemy's previous move uncovered by moving a more valuable
  piece off that line.
* discoveredAttack: a line piece captures along a line that its own side's previous move opened.
* discoveredCheck / doubleCheck: a move gives check with a piece other than the one that moved / with two.
* hangingPiece: a side's first move of the line takes an undefended piece (not a pawn), not as a recapture of an
  equal trade, and keeps the material over the next two moves. When the move before the line is unknown (the
  first side's first move), a capture that may follow a capture (halfmove clock 0) and only gets back to level
  material counts as a recapture.
* trappedPiece: a piece that could not move anywhere safe (every square it can reach is attacked by something
  cheaper or not defended) is captured.
* backRankMate / smotheredMate / mateIn1..5: the line ends in mate; on the back rank behind the king's own
  pieces / by a knight with the king boxed in by its own pieces / within N moves of the mating side.
* deflection: a check or sacrifice pulls a defender away and the square it guarded is taken (or a pawn
  promotes there).
* attraction: a sacrifice lures a king, queen or rook onto a square that is then attacked (and the piece taken).
* overloading: a defender recaptures on one square and so drops the other piece it guarded. Lichess has no
  puzzles tagged with this theme any more, so its precision can't be measured and it is never named.
* advancedPawn: a pawn moves to its seventh or eighth rank.

``GATED_THEMES`` holds the themes whose detector tags at least 20 puzzles of
tests/fixtures/lichess_puzzles_sample.csv and agrees with Lichess's label on at least 80% of them
(tests/test_motifs.py measures it; scripts/motif_precision.py prints the table); the coaching names only those
and says "tactic" for the rest. The definitions follow Lichess's, so near-perfect agreement on Lichess puzzles
is expected: the gate shows we implement those definitions faithfully. An engine line runs on after the tactic
is over, and a pattern that turns up there counts too: continuing each puzzle with Stockfish to eight plies
drops precision on the core themes to 0.88-1.00 (scripts/motif_precision.py --extend 8), a lower bound since
some of those later patterns are real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chess

from ..models import Line, Motif

log = logging.getLogger(__name__)

CORE_THEMES = ("fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate")
THEMES = CORE_THEMES + (
    "discoveredCheck", "doubleCheck", "trappedPiece", "smotheredMate", "mateIn1", "mateIn2", "mateIn3", "mateIn4",
    "mateIn5", "deflection", "attraction", "overloading", "advancedPawn",
)
# Themes whose detector met the precision gate (>= 0.8 against Lichess's labels, with at least 20 puzzles tagged,
# on tests/fixtures/lichess_puzzles_sample.csv); the others are reported as "a tactic" rather than by name.
# Not gated: mateIn5 (only 2 puzzles in the sample) and overloading (Lichess no longer tags it, so there is
# nothing to measure against). tests/test_motifs.py fails when this set and the measurement disagree.
GATED_THEMES: frozenset[str] = frozenset({
    "fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate", "discoveredCheck", "doubleCheck",
    "trappedPiece", "smotheredMate", "mateIn1", "mateIn2", "mateIn3", "mateIn4", "deflection", "attraction",
    "advancedPawn",
})
GATE_PRECISION = 0.8
GATE_MIN_TAGGED = 20

# Plain words for the report ("you missed a fork").
THEME_LABELS: dict[str, str] = {
    "fork": "fork",
    "pin": "pin",
    "skewer": "skewer",
    "hangingPiece": "hanging piece",
    "discoveredAttack": "discovered attack",
    "backRankMate": "back-rank mate",
    "discoveredCheck": "discovered check",
    "doubleCheck": "double check",
    "trappedPiece": "trapped piece",
    "smotheredMate": "smothered mate",
    "mateIn1": "mate in 1",
    "mateIn2": "mate in 2",
    "mateIn3": "mate in 3",
    "mateIn4": "mate in 4",
    "mateIn5": "mate in 5",
    "deflection": "deflection",
    "attraction": "attraction",
    "overloading": "overloaded defender",
    "advancedPawn": "advanced pawn",
}

LINE_PLIES = 8  # detect_line looks this far along a line for patterns ...
MATE_PLIES = 10  # ... and this far for a mate (mate in 5 by the second side takes 10 plies)
MAX_MATE_IN = 5

# Piece values for "worth more than", in pawns; the king is worth more than anything when it is a target.
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 99}
_LINE_PIECES = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def training_url(theme: str) -> str:
    return f"https://lichess.org/training/{theme}"


def theme_label(theme: str) -> str:
    """The plain name of a gated theme ("fork", "back-rank mate"); "tactic" for a theme we can't name reliably."""
    return THEME_LABELS.get(theme, "tactic") if theme in GATED_THEMES else "tactic"


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------
def _value(piece: Optional[chess.Piece]) -> int:
    return VALUES[piece.piece_type] if piece else 0


def _defended(board: chess.Board, square: chess.Square) -> bool:
    """Whether the piece on ``square`` is guarded by its own side, counting a guard behind an enemy line piece
    that attacks it (the x-ray: after that piece captures, the guard recaptures)."""
    piece = board.piece_at(square)
    if piece is None:
        return False
    if board.attackers_mask(piece.color, square):
        return True
    for attacker in board.attackers(not piece.color, square):
        if board.piece_type_at(attacker) in _LINE_PIECES:
            probe = board.copy(stack=False)
            probe.remove_piece_at(attacker)
            if probe.attackers_mask(piece.color, square):
                return True
    return False


def _cheaper_attacker(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    own = VALUES[piece.piece_type]
    return any(_value(board.piece_at(a)) < own for a in board.attackers(not piece.color, square))


def _en_prise(board: chess.Board, square: chess.Square) -> bool:
    """Attacked, and either undefended or attacked by something cheaper: the piece can be won."""
    piece = board.piece_at(square)
    if piece is None or not board.attackers_mask(not piece.color, square):
        return False
    return not _defended(board, square) or _cheaper_attacker(board, square)


def _material(board: chess.Board, color: chess.Color) -> int:
    """Material of ``color`` minus the other side's, in pawns (kings not counted)."""
    total = 0
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += VALUES[piece_type] * (len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color)))
    return total


def _landing(board: chess.Board, move: chess.Move) -> chess.Square:
    """Where the moving piece ends up (for castling: the rook's square, the piece that can give check)."""
    if board.is_castling(move):
        rank = chess.square_rank(move.to_square)
        return chess.square(5 if board.is_kingside_castling(move) else 3, rank)
    return move.to_square


def _bit(square: chess.Square) -> int:
    return chess.BB_SQUARES[square]


# ---------------------------------------------------------------------------
# The walk along a line
# ---------------------------------------------------------------------------
@dataclass
class _Walk:
    boards: list[chess.Board]  # boards[i] = the position before move i; boards[-1] = after the last move
    moves: list[chess.Move] = field(default_factory=list)
    castles: list[bool] = field(default_factory=list)
    captured: list[Optional[chess.Piece]] = field(default_factory=list)  # what move i took (None: nothing)
    movers: list[chess.PieceType] = field(default_factory=list)  # the piece type that made move i

    @property
    def n(self) -> int:
        return len(self.moves)


def _walk(fen: str, moves_uci: list[str], limit: int) -> Optional[_Walk]:
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    walk = _Walk([board.copy(stack=False)])
    for uci in list(moves_uci)[:limit]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if not board.is_legal(move):
            break
        castle = board.is_castling(move)
        walk.castles.append(castle)
        walk.movers.append(board.piece_type_at(move.from_square) or chess.PAWN)
        if board.is_en_passant(move):
            walk.captured.append(chess.Piece(chess.PAWN, not board.turn))
        else:
            walk.captured.append(None if castle else board.piece_at(move.to_square))
        board.push(move)
        walk.moves.append(move)
        walk.boards.append(board.copy(stack=False))
    return walk


def _found(theme: str, ply: int, squares: list) -> tuple[str, int, list[str]]:
    names: list[str] = []
    for sq in squares:
        if sq is None:
            continue
        name = chess.square_name(sq)
        if name not in names:
            names.append(name)
    return theme, ply, names


# Each detector looks at move ``i`` of the walk and returns (theme, ply, squares) or None. ``ply`` is the move
# where the pattern shows on the board; ``squares`` are read on the board right after that move, the active
# piece first.
def _fork(w: _Walk, i: int):
    if w.movers[i] == chess.KING or i + 2 >= w.n:  # the fork has to be cashed in by a later move
        return None
    after = w.boards[i + 1]
    to = w.moves[i].to_square
    if _en_prise(after, to):
        return None
    own = VALUES[w.movers[i]]
    targets = []
    for sq in chess.SquareSet(after.attacks_mask(to) & after.occupied_co[after.turn]):
        piece_type = after.piece_type_at(sq)
        if piece_type == chess.PAWN:
            continue
        # worth more than the forking piece, or loose and not guarding it
        if VALUES[piece_type] > own or (not _defended(after, sq) and not after.attacks_mask(sq) & _bit(to)):
            targets.append(sq)
    if len(targets) < 2:
        return None
    return _found("fork", i, [to] + targets)


def _pin(w: _Walk, i: int):
    after = w.boards[i + 1]
    enemy = after.turn
    mover = not enemy
    king = after.king(enemy)
    if king is None:
        return None
    for sq in chess.SquareSet(after.occupied_co[enemy] & ~after.kings):
        ray = after.pin_mask(enemy, sq)
        if ray == chess.BB_ALL:
            continue
        pinned = VALUES[after.piece_type_at(sq)]
        pinner = next((a for a in chess.SquareSet(ray & after.occupied_co[mover])
                       if after.piece_type_at(a) in _LINE_PIECES), None)
        hit = None
        # the pin stops it from taking something worth more than itself, or a loose piece
        for target in chess.SquareSet(after.attacks_mask(sq) & after.occupied_co[mover] & ~ray):
            if VALUES[after.piece_type_at(target)] > pinned or not _defended(after, target):
                hit = target
                break
        # ... or it is attacked along the pin by something cheaper, or it is loose and can't step away
        if hit is None:
            for attacker in chess.SquareSet(after.attackers_mask(mover, sq) & ray):
                if VALUES[after.piece_type_at(attacker)] < pinned:
                    hit = attacker
                    break
                if (not _defended(after, sq) and not after.attacks_mask(sq) & _bit(attacker)
                        and any(not ray & _bit(m.to_square) for m in after.generate_pseudo_legal_moves(_bit(sq)))):
                    hit = attacker
                    break
        if hit is not None:
            return _found("pin", i, [pinner, sq, king])
    return None


def _skewer(w: _Walk, i: int):
    if i < 2 or w.movers[i] not in _LINE_PIECES or w.captured[i] is None:
        return None
    move, reply = w.moves[i], w.moves[i - 1]
    if reply.to_square == move.to_square:
        return None
    if not _bit(reply.from_square) & chess.between(move.from_square, move.to_square):
        return None  # the piece in front stepped off the line
    if w.boards[i + 1].is_checkmate():
        return None
    # the piece in front was worth more than the one behind, which is lost
    if VALUES[w.movers[i - 1]] <= VALUES[w.captured[i].piece_type] or not _en_prise(w.boards[i], move.to_square):
        return None
    return _found("skewer", i - 2, [move.from_square, reply.from_square, move.to_square])


def _discovered_attack(w: _Walk, i: int):
    if i < 2 or w.captured[i] is None:
        return None
    move, reply, opener = w.moves[i], w.moves[i - 1], w.moves[i - 2]
    if reply.to_square == move.to_square or w.castles[i - 2]:
        return None
    if not _bit(opener.from_square) & chess.between(move.from_square, move.to_square):
        return None
    if move.to_square == opener.to_square or move.from_square == opener.to_square:
        return None
    return _found("discoveredAttack", i - 2, [move.from_square, move.to_square, opener.to_square])


def _checks(w: _Walk, i: int):
    after = w.boards[i + 1]
    if not after.is_check():
        return []
    checkers = list(after.checkers())
    king = after.king(after.turn)
    landing = _landing(w.boards[i], w.moves[i])
    found = []
    if len(checkers) > 1:
        found.append(_found("doubleCheck", i, checkers + [king]))
    uncovered = [c for c in checkers if c != landing]
    if uncovered:
        found.append(_found("discoveredCheck", i, uncovered + [king, landing]))
    return found


def _hanging(w: _Walk, i: int):
    victim = w.captured[i]
    if victim is None or victim.piece_type in (chess.PAWN, chess.KING):
        return None
    before = w.boards[i]
    to = w.moves[i].to_square
    if _defended(before, to):
        return None
    if i >= 1:
        if w.moves[i - 1].to_square == to and _value(w.captured[i - 1]) >= VALUES[victim.piece_type]:
            return None  # a recapture: the other side just took something as valuable here
    elif before.halfmove_clock == 0 and _material(before, before.turn) + VALUES[victim.piece_type] <= 0:
        # The move before the line is unknown. It may have been a capture (the halfmove clock is 0) and taking
        # this piece only gets you back to level material: most likely the second half of a trade.
        return None
    if i + 3 <= w.n and _material(w.boards[i + 3], before.turn) < _material(w.boards[i + 1], before.turn):
        return None  # the material goes straight back
    return _found("hangingPiece", i, [w.moves[i].from_square, to])


def _is_trapped(board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None or piece.color != board.turn or piece.piece_type in (chess.PAWN, chess.KING):
        return False
    if board.is_check() or board.is_pinned(board.turn, square) or not _en_prise(board, square):
        return False
    for escape in list(board.generate_legal_moves(from_mask=_bit(square))):
        if _value(board.piece_at(escape.to_square)) >= VALUES[piece.piece_type]:
            return False  # it can take something as valuable as itself
        board.push(escape)
        safe = not _en_prise(board, escape.to_square)
        board.pop()
        if safe:
            return False
    return True


def _trapped(w: _Walk, i: int):
    if i < 2 or w.captured[i] is None or w.captured[i].piece_type == chess.PAWN:
        return None
    square = w.moves[i].to_square
    if w.moves[i - 1].to_square == square:
        square = w.moves[i - 1].from_square  # it ran to the square where it was taken
    if not _is_trapped(w.boards[i - 1].copy(stack=False), square):
        return None
    return _found("trappedPiece", i - 2, [w.moves[i - 2].to_square, square])


def _deflection(w: _Walk, i: int):
    if i < 2:
        return None
    move, reply, lure = w.moves[i], w.moves[i - 1], w.moves[i - 2]
    victim = w.captured[i]
    if victim is None and not move.promotion:
        return None
    if victim is not None and VALUES[victim.piece_type] > VALUES[w.movers[i]]:
        return None  # a cheap piece taking a dear one needs no deflection
    lure_took = w.captured[i - 2]
    if lure_took is not None and VALUES[lure_took.piece_type] >= VALUES[w.movers[i - 2]]:
        return None  # the luring move has to offer something (a check or a sacrifice), not just win material
    target = move.to_square
    if target in (reply.to_square, lure.to_square):
        return None
    if reply.to_square != lure.to_square and not w.boards[i - 1].is_check():
        return None  # the defender was forced away: it took the lure, or answered a check
    guarded = w.boards[i - 1].attacks_mask(reply.from_square)
    if not guarded & _bit(target):
        if not (move.promotion and chess.square_file(target) == chess.square_file(reply.from_square)
                and guarded & _bit(move.from_square)):
            return None
    if w.boards[i].attacks_mask(reply.to_square) & _bit(target):
        return None  # it still guards the square from where it went
    return _found("deflection", i - 2, [lure.to_square, reply.from_square, target])


def _attraction(w: _Walk, i: int):
    if i + 2 >= w.n:
        return None
    square = w.moves[i].to_square
    reply = w.moves[i + 1]
    if reply.to_square != square or w.movers[i + 1] not in (chess.KING, chess.QUEEN, chess.ROOK):
        return None
    follow = w.moves[i + 2]
    after = w.boards[i + 3]
    if not after.attackers_mask(not after.turn, square) & _bit(follow.to_square):
        return None  # the next move has to hit the lured piece
    if w.movers[i + 1] != chess.KING and not (i + 4 < w.n and w.moves[i + 4].to_square == square):
        return None  # ... and a lured queen or rook has to be taken there
    return _found("attraction", i, [square, reply.from_square])


def _overloading(w: _Walk, i: int):
    # take on X, the defender D recaptures on X, then take on Y that D was also guarding
    if i + 2 >= w.n or w.captured[i] is None:
        return None
    reply, follow = w.moves[i + 1], w.moves[i + 2]
    x, y = w.moves[i].to_square, follow.to_square
    if reply.to_square != x or y == x or w.captured[i + 2] is None or w.captured[i + 2].piece_type == chess.PAWN:
        return None
    if not w.boards[i + 1].attacks_mask(reply.from_square) & _bit(y):
        return None
    if w.boards[i + 2].attacks_mask(x) & _bit(y) or _defended(w.boards[i + 2], y):
        return None
    return _found("overloading", i, [reply.from_square, x, y])


def _advanced_pawn(w: _Walk, i: int):
    if w.movers[i] != chess.PAWN:
        return None
    rank = chess.square_rank(w.moves[i].to_square)
    if (rank if w.boards[i].turn == chess.WHITE else 7 - rank) < 6:
        return None
    return _found("advancedPawn", i, [w.moves[i].to_square])


def _mates(w: _Walk) -> list[tuple[str, int, list[str], int]]:
    """Mate themes when the line ends in mate: (theme, ply, squares, side index)."""
    if not w.n or not w.boards[-1].is_checkmate():
        return []
    last = w.n - 1
    board = w.boards[-1]
    loser = board.turn
    king = board.king(loser)
    checkers = list(board.checkers())
    found = []
    n_moves = last // 2 + 1  # the mating side's moves in the line
    if n_moves <= MAX_MATE_IN:
        found.append(_found(f"mateIn{n_moves}", last, checkers + [king]))
    if (any(board.piece_type_at(c) == chess.KNIGHT for c in checkers)
            and not chess.BB_KING_ATTACKS[king] & ~board.occupied_co[loser]):
        found.append(_found("smotheredMate", last, checkers + [king]))
    back = 0 if loser == chess.WHITE else 7
    if chess.square_rank(king) == back and any(chess.square_rank(c) == back for c in checkers):
        ahead = back + (1 if loser == chess.WHITE else -1)
        file = chess.square_file(king)
        front = [chess.square(f, ahead) for f in range(max(0, file - 1), min(7, file + 1) + 1)]
        if all(board.color_at(sq) == loser and not board.is_attacked_by(not loser, sq) for sq in front):
            found.append(_found("backRankMate", last, checkers + [king]))
    return [(theme, ply, squares, last % 2) for theme, ply, squares in found]


_PER_MOVE = (_fork, _pin, _skewer, _discovered_attack, _trapped, _deflection, _attraction, _overloading,
             _advanced_pawn)


def _detect(fen: str, moves_uci: list[str], max_plies: int, mate_plies: int) -> list[Motif]:
    walk = _walk(fen, moves_uci, max(max_plies, mate_plies))
    if walk is None or not walk.n:
        return []
    found: dict[tuple[str, int], Motif] = {}

    def add(theme: str, ply: int, squares: list[str], side: int) -> None:
        found.setdefault((theme, side), Motif(theme=theme, line="", ply=ply, squares=squares,
                                              side="first" if side == 0 else "second"))

    if walk.n <= mate_plies:
        for theme, ply, squares, side in _mates(walk):
            add(theme, ply, squares, side)
    for i in range(min(walk.n, max_plies)):
        side = i % 2
        for theme, ply, squares in _checks(walk, i):
            add(theme, ply, squares, side)
        if i < 2:  # a hanging piece is taken with the side's first move of the line
            hit = _hanging(walk, i)
            if hit:
                add(*hit, side)
        for detector in _PER_MOVE:
            hit = detector(walk, i)
            if hit:
                add(*hit, side)
    return sorted(found.values(), key=lambda m: (m.ply, THEMES.index(m.theme)))


def _safe_detect(fen: str, moves_uci: list[str], max_plies: int, mate_plies: int) -> list[Motif]:
    """``_detect``, but one odd position (a variant FEN, a board python-chess can't handle) costs only its own
    motifs, not the whole explanation step."""
    try:
        return _detect(fen, moves_uci, max_plies, mate_plies)
    except Exception as exc:  # noqa: BLE001
        log.debug("motif detection failed for %s %s: %s", fen, " ".join(moves_uci[:max_plies]), exc)
        return []


def detect_moves(fen: str, moves_uci: list[str], max_plies: int = 8) -> list[Motif]:
    """Motifs along ``moves_uci`` from ``fen``. ``Motif.side`` is "first" when the side to move at ``fen``
    carries it out and "second" otherwise; ``Motif.line`` is left empty.

    Looks at the first ``max_plies`` moves (a mate counts when those moves end in mate). At most one motif per
    theme and side, the earliest; sorted by ply. ``Motif.squares`` are read on the board right after the move
    at ``Motif.ply``, the active piece first (the forking piece, the pinner, the checker ...), then its targets;
    for a hanging piece they are the capture's from and to squares. Stops quietly at an illegal or malformed
    move; a FEN python-chess can't read gives no motifs.
    """
    return _safe_detect(fen, moves_uci, max_plies, max_plies)


def puzzle_motifs(fen: str, moves_uci: list[str]) -> list[Motif]:
    """The motifs a Lichess puzzle's solver carries out. ``fen`` is the puzzle's FEN (before the opponent's
    move), ``moves_uci`` its Moves column: the opponent's move, then the solution. Side "first" = the solver."""
    if not moves_uci:
        return []
    try:
        board = chess.Board(fen)
        board.push_uci(moves_uci[0])
    except ValueError:
        return []
    solution = list(moves_uci[1:])
    return [m for m in detect_moves(board.fen(), solution, max_plies=len(solution)) if m.side == "first"]


def detect_line(line: Line, role: str) -> list[Motif]:
    """Motifs along a coaching line. ``role`` is "best" (what you missed: yours are the patterns you carry out)
    or "refutation" (what your move allowed: the opponent's patterns). Sets ``line`` and ``side`` ("you" /
    "opponent").

    "You" is the side to move at ``line.fen``: the best line's first move is yours, the refutation's first move
    is the one you played. Only the role's patterns come back (yours along the best line, the opponent's along
    the refutation). Patterns are looked for in the first ``LINE_PLIES`` moves, a mate in the first
    ``MATE_PLIES``.
    """
    if role not in ("best", "refutation"):
        raise ValueError(f"role must be 'best' or 'refutation', not {role!r}")
    want = "first" if role == "best" else "second"
    out = []
    for motif in _safe_detect(line.fen, list(line.moves_uci), LINE_PLIES, MATE_PLIES):
        if motif.side != want:
            continue
        motif.line = role
        motif.side = "you" if want == "first" else "opponent"
        out.append(motif)
    return out
