"""A "null world": games with no real strengths or weaknesses, plus optional planted ones.

Every result is drawn from the Elo expectation (with realistic rating noise and
White's first-move edge), and openings, terminations, clocks, schedules and
game lengths are independent of performance. Any strength/weakness an analysis
module reports on this data is a false positive. Used by
test_null_calibration.py to keep the false-claim rate low.

``planted`` injects realistic effects for power tests (test_power.py). Each one
changes only the games it concerns, by shifting their true expected score (or
their clocks / how they end):

``"opening": ("black", "Caro-Kann Defense", -0.15[, share])``
    points per game in that family with that colour; the family becomes the
    player's main choice with that colour (``share`` of those games, default 40%).
``"short_losses": ("white", "Italian Game", 0.40[, share])``
    probability that a loss in that family (mate or resignation) is over within
    22 moves (16-44 plies); the score is unchanged. Same ``share`` rule.
``"tilt_after_loss": -0.12``
    points per game for games started at most 15 minutes after a loss.
``"late_night": -0.10``
    points per game for games started 23:00-03:00 UTC.
``"time_trouble": (0.35, -0.10)``
    in that share of games the player's clock is forced below 10% of the
    starting time late in the game, and those games are shifted by the points.
``"flagging": 0.25``
    probability that a loss by mate or resignation becomes a loss on time.
``"colour": ("black", -0.08)``
    points per game with that colour (on top of White's normal edge).
``"play_on": 0.8``
    probability that a loss the player would have resigned is played on instead, 10-30
    more moves, to checkmate (a resignation habit: results are unchanged, but the player's
    losses get longer and end in mate more often).
``"baseline": 0.03``
    points per game added to every game. The null world anchors each game's true
    expectation to the *current* rating, so on its own a planted weakness makes the
    player underperform their rating on average (and the rating keeps sliding); a real
    player's rating has already priced their weaknesses in, which lifts every game a
    little relative to that rating. ``-share * effect`` (the share of games the effect
    touches) models this: the planted gap between affected and other games stays the
    same, the affected games' shortfall against the rating shrinks by ``share``.

``"late_castling": 0.5``
    share of games in which the player puts castling off until moves 11-16 (when they
    meant to castle earlier); the opponents' habits are unchanged.
``"early_queen": 0.3``
    share of games in which the player also moves the queen (not capturing) on one of
    moves 2-8.
``"slow_development": 0.5``
    share of games in which the player gets the knights and bishops out much later (all
    four by moves 24-40 instead of 7-20); the kingside pieces still come out in time for
    castling, so only development changes.
``"line_habits": True``
    the opening line, not the player, sets when each side castles, how soon it develops and
    how many early pawn moves it makes (``LINE_HABITS``: in the Ruy Lopez White castles on
    moves 4-6 and Black on 7-10, in the Caro-Kann Black castles on 8-13 ...), for the player
    and the opponent alike. The player's repertoire then shows up as a gap to the opponents
    that is no habit of the player's (combine with a main family, e.g. ``"opening":
    ("black", "Caro-Kann Defense", 0.0, 0.6)``, for a player who mostly plays one line).

Opening habits (castling, development, early queen and pawn moves) are the same for
both players: after the opening line, each side's first 16 moves come from a simple
policy whose habits are drawn from one distribution for the player and the opponent
alike, independent of the result (:func:`_structured_moves`), and random legal moves
follow. The structure effects above change only the player's habits.

Without ``planted`` every draw of the main random stream happens exactly as it always
did, so a seed gives the same results, openings (and their opening lines), clocks,
lengths, ratings and schedules as before; only the moves after the opening line differ
from older versions (they come from their own per-game stream, and have exactly as many
plies). The planted effects use their own random streams. The one other deliberate
change to the null world is its schedule: sessions start mostly in the evening and late
at night (UTC), like a real club player's, so that about a third of all games are
late-night games and a late-night effect is measurable.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import chess

from chess_insights.models import Game
from chess_insights.parse import fill_pregame_ratings

OPENINGS = {  # first plies, family, ECO — chosen independently of the result
    "white": [
        (["e4", "e5", "Nf3", "Nc6", "Bc4"], "Italian Game", "C50"),
        (["e4", "e5", "Nf3", "Nc6", "Bb5"], "Ruy Lopez Opening", "C60"),
        (["d4", "d5", "c4"], "Queen's Gambit", "D06"),
        (["e4", "c5", "Nf3"], "Sicilian Defense", "B27"),
        (["d4", "Nf6", "Bf4"], "Queen's Pawn Opening", "A45"),
    ],
    "black": [
        (["e4", "c6", "d4", "d5"], "Caro-Kann Defense", "B12"),
        (["e4", "e5", "Nf3", "Nc6"], "King's Pawn Opening", "C44"),
        (["d4", "d5", "c4", "e6"], "Queen's Gambit", "D30"),
        (["e4", "e6", "d4", "d5"], "French Defense", "C00"),
        (["d4", "Nf6", "c4", "g6"], "King's Indian Defense", "E60"),
    ],
}
RATING_NOISE = 75.0  # sd of (true strength difference - rating difference), both players' uncertainty
WHITE_EDGE = 0.04  # White scores ~4 points per 100 games more than Black at equal ratings
FORMATS = [("blitz", 300, 0, 0.45), ("blitz", 180, 2, 0.15), ("rapid", 600, 0, 0.25), ("bullet", 60, 0, 0.15)]
P_DRAW = 0.08

# When a new session starts (UTC hour; values >= 24 are after midnight): (weight, from, to).
SESSION_STARTS = ((0.20, 8.0, 17.0), (0.45, 17.0, 22.5), (0.35, 22.5, 25.5))
LATE_NIGHT_HOURS = frozenset({23, 0, 1, 2})  # games started 23:00-03:00 UTC
TILT_WINDOW = timedelta(minutes=15)
DEFAULT_MAIN_SHARE = 0.40  # share of a colour's games in a planted opening family
PLANT_KEYS = frozenset(
    {
        "opening",
        "short_losses",
        "tilt_after_loss",
        "late_night",
        "time_trouble",
        "flagging",
        "colour",
        "baseline",
        "play_on",
        "late_castling",
        "early_queen",
        "slow_development",
        "line_habits",
    }
)
STRUCTURE_KEYS = ("late_castling", "early_queen", "slow_development")


def _random_moves(rng: random.Random, start: list[str], target_plies: int) -> list[str]:
    """Random legal moves after ``start`` (same draws as ``board.is_game_over()`` + ``board.san()``, just faster)."""
    board = chess.Board()
    moves = []
    for san in start:
        board.push_san(san)
        moves.append(san)
    while len(moves) < target_plies:
        legal = list(board.legal_moves)
        if (
            not legal
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            break
        moves.append(board.san_and_push(rng.choice(legal)))
    return moves


# Opening habits, drawn per game for each side from the SAME distribution (player and opponent alike). A
# side's habits are targets, met whatever the opening line did first: when it castles, by which move all four
# knights and bishops are out, and how many of its first 10 moves are pawn moves. So the opening lines, which
# differ between the player's colours, don't give either side a head start.
STRUCTURED_PLIES = 32  # the habit policy plays each side's first 16 moves; random legal moves after that
P_CASTLE = 0.8  # the side means to castle in the opening ...
CASTLE_AT = (5, 13)  # ... on this own move (uniform), or as soon after it as castling is legal
LATE_CASTLE_AT = (11, 16)  # planted late castling
P_EARLY_QUEEN = 0.15  # the side moves its queen (not capturing) on one of these own moves
EARLY_QUEEN_MOVES = (2, 8)
QUEEN_FREE_FROM = 9  # otherwise the queen stays at home until this move (unless nothing else is legal)
DEVELOPED_BY = (7, 20)  # own move by which all four knights and bishops are out (uniform), developed evenly
SLOW_DEVELOPED_BY = (24, 40)  # planted slow development
PAWN_MOVES = (2, 6)  # pawn moves among the first 10 moves (uniform), spread evenly
STRUCTURE_TRIES = 20  # attempts at a game that lasts as many plies as the original one
_HOME_FILES = {chess.KNIGHT: (1, 6), chess.BISHOP: (2, 5)}  # b/g knights, c/f bishops


def _draw_habits(srng: random.Random) -> dict[str, Any]:
    """One side's opening habits (the same draws, in the same order, for both players)."""
    return {
        "castle_at": srng.randint(*CASTLE_AT) if srng.random() < P_CASTLE else None,
        "queen_at": srng.randint(*EARLY_QUEEN_MOVES) if srng.random() < P_EARLY_QUEEN else None,
        "developed_by": srng.randint(*DEVELOPED_BY),
        "pawn_moves": srng.randint(*PAWN_MOVES),
    }


# Planted "line_habits": the opening line sets each side's targets, whoever plays it (book-like move orders):
# family -> side -> {habit: (from, to)} for the own move it castles on, the move by which all four knights and
# bishops are out, and the pawn moves among its first 10 moves. Whether a side castles at all stays P_CASTLE.
LINE_HABITS: dict[str, dict[chess.Color, dict[str, tuple[int, int]]]] = {
    "Italian Game": {chess.WHITE: {"castle_at": (4, 7), "developed_by": (6, 12)},
                     chess.BLACK: {"castle_at": (5, 8), "developed_by": (6, 12)}},
    "Ruy Lopez Opening": {chess.WHITE: {"castle_at": (4, 6), "developed_by": (7, 14)},
                          chess.BLACK: {"castle_at": (7, 11), "developed_by": (8, 16), "pawn_moves": (3, 6)}},
    "Queen's Gambit": {chess.WHITE: {"castle_at": (8, 12), "developed_by": (8, 16)},
                       chess.BLACK: {"castle_at": (5, 7), "developed_by": (6, 12)}},
    "Sicilian Defense": {chess.WHITE: {"castle_at": (6, 9), "developed_by": (6, 12)},
                         chess.BLACK: {"castle_at": (9, 14), "developed_by": (10, 22), "pawn_moves": (4, 7)}},
    "Queen's Pawn Opening": {chess.WHITE: {"castle_at": (6, 9), "developed_by": (7, 14)},
                             chess.BLACK: {"castle_at": (5, 8), "developed_by": (6, 12)}},
    "Caro-Kann Defense": {chess.WHITE: {"castle_at": (6, 9), "developed_by": (7, 14)},
                          chess.BLACK: {"castle_at": (9, 14), "developed_by": (10, 20), "pawn_moves": (3, 6)}},
    "King's Pawn Opening": {chess.WHITE: {"castle_at": (4, 7), "developed_by": (6, 12)},
                            chess.BLACK: {"castle_at": (5, 9), "developed_by": (7, 14)}},
    "French Defense": {chess.WHITE: {"castle_at": (6, 9), "developed_by": (7, 14)},
                       chess.BLACK: {"castle_at": (9, 13), "developed_by": (12, 24), "pawn_moves": (3, 6)}},
    "King's Indian Defense": {chess.WHITE: {"castle_at": (7, 11), "developed_by": (7, 14), "pawn_moves": (3, 6)},
                              chess.BLACK: {"castle_at": (4, 6), "developed_by": (5, 10)}},
}


def _line_habits(lrng: random.Random, habits: dict[chess.Color, dict[str, Any]], family: str) -> None:
    """Planted "line_habits": both sides' targets from the opening line (own stream; same draws for any line)."""
    for color in (chess.WHITE, chess.BLACK):
        spec = LINE_HABITS.get(family, {}).get(color, {})
        for key, default in (("castle_at", CASTLE_AT), ("developed_by", DEVELOPED_BY), ("pawn_moves", PAWN_MOVES)):
            value = lrng.randint(*spec.get(key, default))
            if key != "castle_at" or habits[color]["castle_at"] is not None:
                habits[color][key] = value


def _plant_structure(prng: random.Random, habits: dict[str, Any], planted: dict[str, Any]) -> None:
    """Apply the planted structure effects to the player's habits (the same draws whatever they decide)."""
    if "late_castling" in planted:
        late, move = prng.random() < planted["late_castling"], prng.randint(*LATE_CASTLE_AT)
        if late and habits["castle_at"] is not None:
            habits["castle_at"] = max(habits["castle_at"], move)
    if "early_queen" in planted:
        early, move = prng.random() < planted["early_queen"], prng.randint(*EARLY_QUEEN_MOVES)
        if early and habits["queen_at"] is None:
            habits["queen_at"] = move
    if "slow_development" in planted:
        slow, move = prng.random() < planted["slow_development"], prng.randint(*SLOW_DEVELOPED_BY)
        if slow:
            habits["developed_by"] = move


def _at_home(board: chess.Board, square: int, color: chess.Color) -> bool:
    """A knight or bishop of ``color`` still on one of its starting squares."""
    piece = board.piece_at(square)
    return (
        piece is not None
        and piece.color == color
        and piece.piece_type in _HOME_FILES
        and chess.square_rank(square) == (0 if color == chess.WHITE else 7)
        and chess.square_file(square) in _HOME_FILES[piece.piece_type]
    )


def _pick(srng: random.Random, groups: dict[str, list[chess.Move]], weights: dict[str, float]) -> Optional[chess.Move]:
    """A move from one of the non-empty groups, chosen by weight."""
    kinds = [k for k in weights if groups.get(k) and weights[k] > 0]
    if not kinds:
        return None
    kind = srng.choices(kinds, weights=[weights[k] for k in kinds])[0]
    return srng.choice(groups[kind])


def _habit_move(
    srng: random.Random, board: chess.Board, habits: dict[str, Any], state: dict[str, Any]
) -> Optional[chess.Move]:
    """The next move of a side playing by ``habits`` in the opening (None when it has no legal move).

    Castling and the early queen move happen on their move (or as soon after as they are legal). Otherwise
    the side develops a knight or bishop when it is behind its development schedule, makes a pawn move when
    it is behind its pawn schedule, and moves an already developed piece (a rook once castled, the queen
    after move 8) when it is on schedule for both; when a group has no legal move it falls back on the
    others. A side that means to castle develops its kingside pieces first, and frees its king's bishop
    (e- or g-pawn) before its castling move.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
    number = board.fullmove_number
    color = board.turn
    castle_at = habits["castle_at"]
    if castle_at is not None and number >= castle_at and not state["castled"]:
        castles = sorted(board.generate_castling_moves(), key=lambda m: not board.is_kingside_castling(m))
        if castles:
            state["castled"] = True
            return castles[0]
    if habits["queen_at"] is not None and not state["queen"] and number >= habits["queen_at"]:
        quiet = [m for m in legal if board.piece_type_at(m.from_square) == chess.QUEEN and not board.is_capture(m)]
        if quiet:
            state["queen"] = True
            return srng.choice(quiet)
    wants_castle = castle_at is not None and not state["castled"]
    home_rank = 0 if color == chess.WHITE else 7
    kingside = [chess.square(f, home_rank) for f in (5, 6)]  # f-bishop, g-knight
    groups: dict[str, list[chess.Move]] = {"prepare": [], "develop": [], "pawn": [], "free": [], "other": []}
    bishop_blocked = _at_home(board, kingside[0], color)
    for m in legal:
        piece = board.piece_type_at(m.from_square)
        if piece == chess.PAWN:
            groups["pawn"].append(m)
            if chess.square_file(m.from_square) in (4, 6):  # e- or g-pawn: frees the king's bishop
                groups["free"].append(m)
        elif piece in (chess.KNIGHT, chess.BISHOP):
            if _at_home(board, m.from_square, color):
                groups["develop"].append(m)
                if m.from_square in kingside:
                    groups["prepare"].append(m)
                    bishop_blocked = bishop_blocked and m.from_square != kingside[0]
            else:
                groups["other"].append(m)
        elif (piece == chess.ROOK and not wants_castle) or (piece == chess.QUEEN and number >= QUEEN_FREE_FROM):
            groups["other"].append(m)  # the king (and castling) only as planned, or when nothing else is legal
    if not bishop_blocked:
        groups["free"] = []
    # castling soon, and the kingside not ready: develop the king's knight and bishop, free the bishop
    if wants_castle and number >= castle_at - 3:
        move = _pick(srng, groups, {"prepare": 2.0, "free": 1.0})
        if move is not None:
            return move
    developed = 4 - sum(_at_home(board, sq, color) for sq in (chess.square(f, home_rank) for f in (1, 2, 5, 6)))
    dev_need = min(4, math.floor(4 * number / habits["developed_by"] + 0.5)) - developed
    pawn_need = math.floor(habits["pawn_moves"] * min(number, 10) / 10 + 0.5) - state["pawns"]
    if dev_need > 0 or pawn_need > 0:
        weights = {"develop": max(dev_need, 0), "pawn": max(pawn_need, 0)}
    else:
        weights = {"other": 1.0}
    if wants_castle and dev_need > 0:
        weights["prepare"] = 2.0 * dev_need  # kingside pieces first
    move = _pick(srng, groups, weights)
    if move is None:  # the wanted group has no legal move: whatever keeps closest to the schedules
        move = _pick(srng, groups, {"other": 1.0}) or _pick(
            srng, groups, {"develop": max(dev_need, 0) + 0.5, "pawn": max(pawn_need, 0) + 0.5}
        )
    return move if move is not None else srng.choice(legal)


def _random_legal(srng: random.Random, board: chess.Board) -> Optional[chess.Move]:
    """A uniformly random legal move (None if there is none): a random pseudo-legal move, redrawn while illegal.

    The same distribution as ``srng.choice(list(board.legal_moves))``, without generating every legal move.
    """
    candidates = list(board.generate_pseudo_legal_moves())
    while candidates:
        k = srng.randrange(len(candidates))
        if board.is_legal(candidates[k]):
            return candidates[k]
        candidates[k] = candidates[-1]
        candidates.pop()
    return None


def _game_ended(board: chess.Board) -> bool:
    """The end-of-game rule of _random_moves (checkmate and stalemate aside)."""
    return (
        board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or (board.halfmove_clock >= 16 and board.is_fivefold_repetition())  # 16 quiet plies at least
    )


def _structured_moves(
    srng: random.Random, line: list[str], plies: int, habits: dict[chess.Color, dict[str, Any]]
) -> Optional[list[str]]:
    """``plies`` moves: the opening line, each side's first moves by its habits, then random legal moves.

    None when no attempt lasts that long without the game ending (the caller keeps its original moves).
    """
    for _ in range(STRUCTURE_TRIES):
        board = chess.Board()
        moves = []
        for san in line[:plies]:
            moves.append(board.san_and_push(board.parse_san(san)))
        state = {c: {"castled": False, "queen": False, "pawns": 0} for c in (chess.WHITE, chess.BLACK)}
        for ply, san in enumerate(moves):  # the opening line's pawn moves count toward the pawn schedule
            if san[0].islower():
                state[chess.WHITE if ply % 2 == 0 else chess.BLACK]["pawns"] += 1
        while len(moves) < plies and not _game_ended(board):
            if len(moves) < STRUCTURED_PLIES:
                move = _habit_move(srng, board, habits[board.turn], state[board.turn])
                if move is not None and board.piece_type_at(move.from_square) == chess.PAWN:
                    state[board.turn]["pawns"] += 1
            else:
                move = _random_legal(srng, board)
            if move is None:
                break
            moves.append(board.san_and_push(move))
        if len(moves) == plies:
            return moves
    return None


def _continue(srng: random.Random, moves: list[str], plies: int) -> Optional[list[str]]:
    """``moves`` followed by random legal moves up to ``plies`` plies (as _play_on stops: only without a legal
    move); None when no attempt gets that far."""
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    for _ in range(STRUCTURE_TRIES):
        trial = board.copy()
        out = list(moves)
        while len(out) < plies:
            move = _random_legal(srng, trial)
            if move is None:
                break
            out.append(trial.san_and_push(move))
        if len(out) == plies:
            return out
    return None


def _clocks(rng: random.Random, plies: int, base: int, inc: int, flag_side: int | None) -> list[float]:
    remaining = [float(base), float(base)]
    budget = base / 40.0 + inc
    out = []
    for i in range(plies):
        side = i % 2
        spent = min(remaining[side] + inc, rng.lognormvariate(0, 0.8) * budget * 0.8)
        remaining[side] = max(0.0, remaining[side] - spent + inc)
        out.append(round(remaining[side], 1))
    if flag_side is not None and plies:
        last = max(i for i in range(plies) if i % 2 == flag_side) if any(i % 2 == flag_side for i in range(plies)) else None
        if last is not None:
            out[last] = 0.0
    return out


def _force_low_clock(prng: random.Random, clocks: list[float], side: int, base: int) -> None:
    """From a point late in the game on, keep ``side``'s clock below the time-trouble line (10% / 5 s)."""
    mine = [i for i in range(len(clocks)) if i % 2 == side]
    if not mine:
        return
    line = max(0.10 * base, 5.0)
    first = int(len(mine) * prng.uniform(0.6, 0.95))
    low = line * prng.uniform(0.3, 0.9)
    tail = mine[first:] or mine[-1:]
    for k, i in enumerate(tail):
        clocks[i] = round(min(clocks[i], low * (1.0 - 0.5 * k / len(tail))), 1)


def _play_on(prng: random.Random, moves: list[str], clocks: list[float], base: int, inc: int) -> None:
    """Extend a resigned game by 10-30 random moves and clocks, in place (the planted stream only)."""
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    for _ in range(prng.randint(20, 60)):
        legal = list(board.legal_moves)
        if not legal:
            break
        moves.append(board.san_and_push(prng.choice(legal)))
        prev = clocks[-2] if len(clocks) >= 2 else float(base)
        clocks.append(round(max(0.0, prev - prng.uniform(0, 0.02) * base) + inc, 1))


def _next_session_start(srng: random.Random, after: datetime) -> datetime:
    """Start of the next session: a later day, mostly in the evening or late at night (UTC)."""
    _, lo, hi = srng.choices(SESSION_STARTS, weights=[s[0] for s in SESSION_STARTS])[0]
    days = srng.choices([1, 2, 3], weights=[65, 25, 10])[0]
    evening = (after - timedelta(hours=6)).date()  # a session ending at 01:00 belongs to the evening before
    start = datetime(evening.year, evening.month, evening.day, tzinfo=timezone.utc) + timedelta(
        days=days, hours=srng.uniform(lo, hi)
    )
    while start < after + timedelta(hours=2):
        start += timedelta(days=1)
    return start


def _shares(planted: dict[str, Any], color: str) -> Optional[list[float]]:
    """Opening-family weights for ``color`` when a planted family is played more often (None: uniform)."""
    main: dict[str, float] = {}
    for key in ("opening", "short_losses"):
        spec = planted.get(key)
        if spec and spec[0] == color:
            main[spec[1]] = spec[3] if len(spec) > 3 else DEFAULT_MAIN_SHARE
    if not main:
        return None
    names = [family for _, family, _ in OPENINGS[color]]
    unknown = set(main) - set(names)
    if unknown or sum(main.values()) > 1.0:
        raise ValueError(f"bad planted opening families for {color}: {main}")
    rest = (1.0 - sum(main.values())) / max(1, len(names) - len(main))
    return [main.get(name, rest) for name in names]


def null_games(
    n: int = 400, seed: int = 0, username: str = "nullplayer", planted: Optional[dict[str, Any]] = None
) -> list[Game]:
    planted = dict(planted or {})
    unknown = set(planted) - PLANT_KEYS
    if unknown:
        raise ValueError(f"unknown planted effects: {sorted(unknown)}")
    rng = random.Random(seed)  # every draw of the null world, in its original order
    srng = random.Random(f"schedule-{seed}")  # session start times
    prng = random.Random(f"planted-{seed}")  # planted effects only
    weights = {c: _shares(planted, c) for c in ("white", "black")}
    rating = {"blitz": 1500, "rapid": 1560, "bullet": 1420}
    t = datetime(2025, 1, 3, 18, 0, tzinfo=timezone.utc)
    games: list[Game] = []
    prev: Optional[Game] = None
    for i in range(n):
        tc, base, inc, _ = rng.choices(FORMATS, weights=[f[3] for f in FORMATS])[0]
        color = rng.choice(["white", "black"])
        my_pre = rating[tc]
        opp_pre = int(rng.gauss(my_pre, 110))
        e = 1.0 / (1.0 + 10 ** ((opp_pre - my_pre) / 400.0))  # what the ratings predict
        # Real life: ratings are noisy (true strength differs by ~RATING_NOISE) and White
        # scores a bit better than Black. Results come from the *true* expectation.
        true_diff = opp_pre - my_pre + rng.gauss(0, RATING_NOISE)
        e_true = 1.0 / (1.0 + 10 ** (true_diff / 400.0)) + (WHITE_EDGE / 2 if color == "white" else -WHITE_EDGE / 2)

        # planted effects decided before the result (own random stream)
        family_choice = None
        if weights[color] is not None:
            family_choice = prng.choices(OPENINGS[color], weights=weights[color])[0]
        trouble = False
        if planted:
            family = family_choice[1] if family_choice else None
            shift = 0.0
            spec = planted.get("opening")
            if spec and spec[0] == color and spec[1] == family:
                shift += spec[2]
            spec = planted.get("colour")
            if spec and spec[0] == color:
                shift += spec[1]
            if "tilt_after_loss" in planted and prev is not None and prev.outcome == "loss":
                if t - prev.end_time <= TILT_WINDOW:
                    shift += planted["tilt_after_loss"]
            if "late_night" in planted and t.hour in LATE_NIGHT_HOURS:
                shift += planted["late_night"]
            spec = planted.get("time_trouble")
            if spec and prng.random() < spec[0]:
                trouble = True
                shift += spec[1]
            e_true += shift + planted.get("baseline", 0.0)

        p_win = max(0.0, min(1.0 - P_DRAW, e_true - P_DRAW / 2))
        u = rng.random()
        outcome = "win" if u < p_win else ("draw" if u < p_win + P_DRAW else "loss")
        score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
        change = round(16 * (score - e))
        rating[tc] = my_pre + change

        line, family, eco = rng.choice(OPENINGS[color])
        if family_choice is not None:
            line, family, eco = family_choice
        plies = max(8, min(160, int(rng.gauss(70, 25))))  # length independent of the result
        spec = planted.get("short_losses")
        quick_loss = bool(
            spec and outcome == "loss" and spec[0] == color and spec[1] == family and prng.random() < spec[2]
        )
        if quick_loss:
            plies = prng.randint(16, 44)
        random_moves = _random_moves(rng, line, plies)  # the main stream's draws, exactly as before
        # the moves themselves: both sides play the opening by habits from the same distribution (own streams)
        srng_game = random.Random(f"structure-{seed}-{i}")
        habits = {chess.WHITE: _draw_habits(srng_game), chess.BLACK: _draw_habits(srng_game)}
        if planted.get("line_habits"):
            _line_habits(random.Random(f"line-habits-{seed}-{i}"), habits, family)
        if any(k in planted for k in STRUCTURE_KEYS):
            mine = habits[chess.WHITE if color == "white" else chess.BLACK]
            _plant_structure(random.Random(f"structure-planted-{seed}-{i}"), mine, planted)
        moves = _structured_moves(srng_game, line, len(random_moves), habits) or random_moves
        if outcome == "draw":
            my_code = opp_code = rng.choice(["agreed", "repetition", "insufficient", "stalemate"])
            termination = {"agreed": "agreement"}.get(my_code, my_code)
            flag_side = None
        else:
            reason = rng.choices(["resigned", "checkmated", "timeout", "abandoned"], weights=[70, 14, 14, 2])[0]
            if quick_loss and reason not in ("resigned", "checkmated"):
                reason = "resigned"
            if (
                outcome == "loss"
                and reason in ("resigned", "checkmated")
                and "flagging" in planted
                and prng.random() < planted["flagging"]
            ):
                reason = "timeout"
            termination = {"resigned": "resignation", "checkmated": "checkmate", "timeout": "timeout", "abandoned": "abandoned"}[reason]
            my_code, opp_code = ("win", reason) if outcome == "win" else (reason, "win")
            loser_is_white = (outcome == "loss") == (color == "white")
            flag_side = (0 if loser_is_white else 1) if reason == "timeout" else None
        clocks = _clocks(rng, len(moves), base, inc, flag_side)
        if trouble:
            _force_low_clock(prng, clocks, 0 if color == "white" else 1, base)
        if "play_on" in planted and outcome == "loss" and termination == "resignation":
            if prng.random() < planted["play_on"]:  # the game is labelled a mate, as the player plays on to the end
                # the extension of the original moves draws from the planted stream and adds the clocks exactly as
                # before; the game's own moves are then extended by as many plies
                extended = list(random_moves)
                _play_on(prng, extended, clocks, base, inc)
                moves = (moves is not random_moves and _continue(srng_game, moves, len(extended))) or extended
                termination, my_code = "checkmate", "checkmated"

        # schedule: sessions of quick re-queues, independent of results
        duration = timedelta(seconds=min(2 * base + 80 * inc, 60 + len(moves) * base / 60))
        start = t
        end = start + duration
        if rng.random() < 0.75:
            t = end + timedelta(minutes=rng.uniform(0.5, 8))
        else:
            rng.uniform(3, 30)  # the old break length: drawn so that every later draw stays the same
            t = _next_session_start(srng, end)
        game = Game(
            game_id=f"null-{seed}-{i}",
            url=f"https://www.chess.com/game/live/{9_000_000 + seed * 10_000 + i}",
            username=username,
            color=color,
            opponent=f"opp{rng.randint(1, 5000)}",
            outcome=outcome,
            my_result_code=my_code,
            opp_result_code=opp_code,
            termination=termination,
            time_class=tc,
            time_control=f"{base}+{inc}" if inc else str(base),
            base_seconds=base,
            increment=inc,
            rules="chess",
            rated=True,
            end_time=end,
            start_time=start,
            my_rating=my_pre + change,
            opp_rating=opp_pre - change,
            eco=eco,
            opening=family,
            opening_family=family,
            my_accuracy=None,
            opp_accuracy=None,
            initial_fen=None,
            moves_san=moves,
            clocks=clocks,
            pgn="",
        )
        games.append(game)
        prev = game
    fill_pregame_ratings(games)
    return games
