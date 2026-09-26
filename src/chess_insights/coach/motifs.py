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
  equal trade, and keeps the material over the next two moves. For the first side's first move the move before
  is known only when the caller passes ``previous_fen``; without it, a capture that may follow a capture
  (halfmove clock 0) and only gets back to level material counts as a recapture. That guess misses trades when
  one side is already ahead: on tests/fixtures/lichess_puzzles_sample.csv hangingPiece scores 0.96 precision /
  0.79 recall without the previous position and 1.00 / 1.00 with it (scripts/motif_precision.py [--previous]).
* trappedPiece: a piece that could not move anywhere safe (every square it can reach is attacked by something
  cheaper or not defended) is captured.
* backRankMate / smotheredMate / mateIn1..5: the line ends in mate; on the back rank behind the king's own
  pieces / by a knight with the king boxed in by its own pieces / within N moves of the mating side.
* deflection: a check or sacrifice pulls a defender away and the square it guarded is taken (or a pawn
  promotes there).
* attraction: a sacrifice lures a king, queen or rook onto a square that is then attacked (and the piece taken).
* overloading: a defender recaptures on one square and so drops the other piece it guarded. Lichess has no
  puzzles tagged with this theme any more, so its precision can't be measured and it is never named.
* advancedPawn: a pawn moves to its seventh or eighth rank and is not taken straight away (a pawn that lands on
  the seventh and is captured on the next move is a trade, not a pawn that will promote).

``GATED_THEMES`` holds the themes whose detector tags at least 20 puzzles of
tests/fixtures/lichess_puzzles_sample.csv and agrees with Lichess's label on at least 80% of them
(tests/test_motifs.py measures it; scripts/motif_precision.py prints the table); the coaching names only those
and says "tactic" for the rest. The definitions follow Lichess's, so near-perfect agreement on Lichess puzzles
is expected: the gate shows we implement those definitions faithfully. An engine line runs on after the tactic
is over, and a pattern that turns up there counts too: continuing each puzzle with Stockfish 16's principal
variation (30,000 nodes) drops precision on the core themes to 0.89-1.00 at eight plies, the length the coaching
reads (scripts/motif_precision.py --extend 8), 0.84-1.00 at ten and 0.81-1.00 at twelve; a lower bound, since
some of those later patterns are real.

What a move allowed is what is new: along a refutation, a pin whose pinner, pinned piece and king already stood
that way before your move is not one your move allowed (``detect_line``), and the coaching drops a pattern that
the better line allows too (``carried_by``).
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

# The order of themes found at the same ply (the first one leads an explanation): taking an undefended piece is
# the plainer story than the pin that may come with it.
# A double check is always a discovered check too: the plainer and rarer name goes first.
_FIRST = ("fork", "hangingPiece", "pin", "skewer", "discoveredAttack", "backRankMate", "doubleCheck")
RANK = {theme: i for i, theme in enumerate(_FIRST + tuple(t for t in THEMES if t not in _FIRST))}

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


def _pinner(board: chess.Board, king: chess.Square, pinned: chess.Square) -> Optional[chess.Square]:
    """The line piece pinning ``pinned`` to ``king``: the first piece beyond the pinned one, away from the king.
    (``pin_mask`` is the whole line through the king, so it also holds pieces behind the king or the pinner.)"""
    for square in chess.SquareSet(chess.ray(king, pinned) & board.occupied & ~_bit(pinned)):
        if chess.between(king, square) & _bit(pinned) and not chess.between(pinned, square) & board.occupied:
            return square
    return None


def _first(squares: list[chess.Square], active: chess.Square) -> list[chess.Square]:
    """``squares`` with ``active`` (the piece that just moved) first when it is among them."""
    return ([active] + [s for s in squares if s != active]) if active in squares else squares


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
    previous_fen: Optional[str] = None  # the position before the move that led to boards[0], when known

    @property
    def n(self) -> int:
        return len(self.moves)


def _board(fen: str) -> chess.Board:
    """The position at ``fen``. A Chess960 start (castling rights that standard chess would drop, e.g. a king on
    f1 with rooks on b1 and g1) is read as Chess960 so its castling moves stay legal."""
    board = chess.Board(fen)
    parts = fen.split()
    if len(parts) > 2 and parts[2] != "-":
        board960 = chess.Board(fen, chess960=True)
        if board960.clean_castling_rights() != board.clean_castling_rights():
            return board960
    return board


def _walk(fen: str, moves_uci: list[str], limit: int) -> Optional[_Walk]:
    try:
        board = _board(fen)
    except ValueError:
        return None
    walk = _Walk([board.copy(stack=False)])
    for uci in list(moves_uci)[:limit]:
        try:
            # parse_uci takes castling as king-to-g1 or king-takes-rook (Chess960 engines), and checks legality
            move = board.parse_uci(uci)
        except (ValueError, TypeError, AttributeError):  # illegal, or not a UCI string (SAN, None ...): stop
            break
        if not move:  # the null move "0000"
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
    own = _value(after.piece_at(to))  # a promoted pawn forks as the piece it became
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
            return _found("pin", i, [_pinner(after, king, sq), sq, king])
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
        found.append(_found("doubleCheck", i, _first(checkers, landing) + [king]))
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
        last: Optional[tuple[chess.Square, int]] = (w.moves[i - 1].to_square, _value(w.captured[i - 1]))
    else:
        last = _previous(w.previous_fen, before) if w.previous_fen else None
    if last is not None:
        if last[0] == to and last[1] >= VALUES[victim.piece_type]:
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
    if i + 1 < w.n and w.moves[i + 1].to_square == w.moves[i].to_square and not w.moves[i].promotion:
        return None  # taken at once (6.dxe7 Nxe7): a trade, not a pawn about to promote
    return _found("advancedPawn", i, [w.moves[i].to_square])


def _mates(w: _Walk) -> list[tuple[str, int, list[str], int]]:
    """Mate themes when the line ends in mate: (theme, ply, squares, side index)."""
    if not w.n or not w.boards[-1].is_checkmate():
        return []
    last = w.n - 1
    board = w.boards[-1]
    loser = board.turn
    king = board.king(loser)
    checkers = _first(list(board.checkers()), _landing(w.boards[last], w.moves[last]))
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


def _previous(previous_fen: str, target: chess.Board) -> Optional[tuple[chess.Square, int]]:
    """The move that leads from ``previous_fen`` to ``target``: (its square, the value of what it took), or None
    when no legal move does (or ``previous_fen`` is unreadable: then the line is read as if it were not given)."""
    try:
        before = _board(previous_fen)
    except (ValueError, TypeError, AttributeError):
        return None
    for move in before.legal_moves:
        if before.is_en_passant(move):
            took = VALUES[chess.PAWN]
        else:
            took = 0 if before.is_castling(move) else _value(before.piece_at(move.to_square))
        before.push(move)
        same = before.board_fen() == target.board_fen() and before.turn == target.turn
        before.pop()
        if same:
            return move.to_square, took
    return None


def _detect(fen: str, moves_uci: list[str], max_plies: int, mate_plies: int,
            previous_fen: Optional[str] = None) -> list[Motif]:
    walk = _walk(fen, moves_uci, max(max_plies, mate_plies))
    if walk is None or not walk.n:
        return []
    walk.previous_fen = previous_fen
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
    return sorted(found.values(), key=lambda m: (m.ply, RANK[m.theme]))


def _safe_detect(fen: str, moves_uci: list[str], max_plies: int, mate_plies: int,
                 previous_fen: Optional[str] = None) -> list[Motif]:
    """``_detect``, but one odd position (a variant FEN, a board python-chess can't handle) costs only its own
    motifs, not the whole explanation step."""
    try:
        return _detect(fen, moves_uci, max_plies, mate_plies, previous_fen)
    except Exception as exc:  # noqa: BLE001
        log.debug("motif detection failed for %r %r: %s", fen, moves_uci, exc)
        return []


def detect_moves(fen: str, moves_uci: list[str], max_plies: int = 8, *,
                 previous_fen: Optional[str] = None) -> list[Motif]:
    """Motifs along ``moves_uci`` from ``fen``. ``Motif.side`` is "first" when the side to move at ``fen``
    carries it out and "second" otherwise; ``Motif.line`` is left empty.

    Looks at the first ``max_plies`` moves (a mate counts when those moves end in mate). At most one motif per
    theme and side, the earliest; sorted by ply. ``Motif.squares`` are read on the board right after the move
    at ``Motif.ply``, the active piece first (the forking piece, the pinner, the checker ...), then its targets;
    for a hanging piece they are the capture's from and to squares. Stops quietly at an illegal or malformed
    move; a FEN python-chess can't read gives no motifs.

    ``previous_fen`` is the position before the move that led to ``fen``, when known. It tells a recapture from
    a hanging piece on the first side's first move; without it, a capture right after a possible capture (the
    halfmove clock is 0) that only gets back to level material is taken for a recapture.
    """
    return _safe_detect(fen, moves_uci, max_plies, max_plies, previous_fen)


def puzzle_motifs(fen: str, moves_uci: list[str]) -> list[Motif]:
    """The motifs a Lichess puzzle's solver carries out. ``fen`` is the puzzle's FEN (before the opponent's
    move), ``moves_uci`` its Moves column: the opponent's move, then the solution. Side "first" = the solver."""
    if not moves_uci:
        return []
    try:
        board = chess.Board(fen)
        board.push_uci(moves_uci[0])
    except (ValueError, TypeError, AttributeError):  # an unreadable row gives no motifs
        return []
    solution = list(moves_uci[1:])
    return [m for m in detect_moves(board.fen(), solution, max_plies=len(solution)) if m.side == "first"]


def forcing_reach(fen: str, moves_uci: list[str], first: int) -> int:
    """``_forcing_reach``: the last ply index a pattern (or a capture) of the side moving at index ``first`` can sit
    at and still be forced."""
    return _forcing_reach(fen, list(moves_uci), first)


def _forcing_reach(fen: str, moves_uci: list[str], first: int) -> int:
    """The last ply index a pattern of the side moving at index ``first`` can sit at and still be forced.

    That side's first move always counts. A later move of theirs counts only when each of their moves before it
    was a check or a capture (the other side's replies are the engine's best defence, not forced moves). The
    reply to the first quiet move is still in reach: it is where a pattern set up by that move lands.
    """
    try:
        board = _board(fen)
    except (ValueError, TypeError):
        return first
    for i, uci in enumerate(moves_uci):
        try:
            move = board.parse_uci(uci)
        except (ValueError, TypeError, AttributeError):
            return max(i - 1, first)
        if i >= first and (i - first) % 2 == 0 and not (board.is_capture(move) or board.gives_check(move)):
            return i + 1
        board.push(move)
    return len(moves_uci)


_MATES = frozenset({"backRankMate", "smotheredMate"})


def _board_after(fen: str, moves_uci: list[str], plies: int) -> Optional[chess.Board]:
    try:
        board = _board(fen)
        for uci in moves_uci[:plies]:
            board.push(board.parse_uci(uci))
    except (ValueError, TypeError, AttributeError):
        return None
    return board


def _standing_pin(fen: str, moves_uci: list[str], motif: Motif) -> bool:
    """Whether ``motif`` (a pin) already stood at ``fen``: the same pinner, pinned piece and king on the same squares,
    and the piece pinned there to that king by that pinner."""
    if motif.theme != "pin" or len(motif.squares) != 3:
        return False
    start, after = _board_after(fen, moves_uci, 0), _board_after(fen, moves_uci, motif.ply + 1)
    if start is None or after is None:
        return False
    try:
        pinner, pinned, king = (chess.parse_square(s) for s in motif.squares)
    except ValueError:
        return False
    piece = after.piece_at(pinned)
    if piece is None or any(start.piece_at(sq) != after.piece_at(sq) for sq in (pinner, pinned, king)):
        return False
    return start.is_pinned(piece.color, pinned) and _pinner(start, king, pinned) == pinner


def carried_by(line: Optional[Line], side: str, previous_fen: Optional[str] = None, *,
               forcing_only: bool = True) -> list[Motif]:
    """The motifs along ``line`` carried out by ``side``: "first" (the side to move at ``line.fen``) or "second",
    within forcing reach (see ``detect_line``); ``Motif.line`` is left empty and ``Motif.side`` as detected.

    The coaching reads the other line with it: a pattern your opponent has along the better line too was not
    allowed by your move, and one you carry out after your move anyway was not missed."""
    if side not in ("first", "second"):
        raise ValueError(f"side must be 'first' or 'second', not {side!r}")
    if line is None or not line.fen or not line.moves_uci:
        return []
    moves = list(line.moves_uci)
    reach = _forcing_reach(line.fen, moves, 0 if side == "first" else 1) if forcing_only else len(moves)
    out = []
    for motif in _safe_detect(line.fen, moves, LINE_PLIES, MATE_PLIES, previous_fen):
        if motif.side != side:
            continue
        if motif.ply > reach and not motif.theme.startswith("mate") and motif.theme not in _MATES:
            continue  # only there if the other side cooperates
        out.append(motif)
    return out


def detect_line(line: Optional[Line], role: str, previous_fen: Optional[str] = None, *,
                forcing_only: bool = True) -> list[Motif]:
    """Motifs along a coaching line. ``role`` is "best" (what you missed: yours are the patterns you carry out)
    or "refutation" (what your move allowed: the opponent's patterns). Sets ``line`` and ``side`` ("you" /
    "opponent").

    "You" is the side to move at ``line.fen``: the best line's first move is yours, the refutation's first move
    is the one you played. Only the role's patterns come back (yours along the best line, the opponent's along
    the refutation). Patterns are looked for in the first ``LINE_PLIES`` moves, a mate in the first
    ``MATE_PLIES``. Along a refutation, a pin that already stood before your move (the same pinner, pinned piece
    and king on the same squares) is left out: your move did not allow it.

    ``previous_fen`` (optional) is the position before your opponent's last move, the one that led to
    ``line.fen``. Pass it when the game is at hand: along the best line it tells whether taking a piece on your
    first move wins a hanging piece or just completes a trade (see ``detect_moves``).

    ``forcing_only`` (default): an engine line is not a puzzle, so a pattern deep in it may only exist because
    the other side cooperates. Keep a pattern only when it is on the carrier's first move of the line, or every
    earlier move of the carrier was a check or a capture (``_forcing_reach``). Mates are forced by definition
    and always kept. (4.e4? in the Queen's Gambit Accepted: the best line 4.d5 Na5 5.Qa4+ has a fork only if
    Black plays 4...Na5, so it is not "a fork you missed".)
    """
    if role not in ("best", "refutation"):
        raise ValueError(f"role must be 'best' or 'refutation', not {role!r}")
    if line is None or not line.fen or not line.moves_uci:  # no line from the engine: nothing to name
        return []
    want = "first" if role == "best" else "second"
    out = []
    for motif in carried_by(line, want, previous_fen, forcing_only=forcing_only):
        if role == "refutation" and _standing_pin(line.fen, list(line.moves_uci), motif):
            continue  # the pin was there before your move
        motif.line = role
        motif.side = "you" if want == "first" else "opponent"
        out.append(motif)
    return out
