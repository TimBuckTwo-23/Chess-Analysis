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

Without ``planted`` every random draw happens exactly as it always did, so a
seed gives the same games, results, openings, clocks and lengths as before; the
planted effects use their own random stream. The one deliberate change to the
null world is its schedule: sessions start mostly in the evening and late at
night (UTC), like a real club player's, so that about a third of all games are
late-night games and a late-night effect is measurable.
"""

from __future__ import annotations

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
PLANT_KEYS = frozenset({"opening", "short_losses", "tilt_after_loss", "late_night", "time_trouble", "flagging", "colour"})


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
            spec = planted.get("opening")
            if spec and spec[0] == color and spec[1] == family:
                e_true += spec[2]
            spec = planted.get("colour")
            if spec and spec[0] == color:
                e_true += spec[1]
            if "tilt_after_loss" in planted and prev is not None and prev.outcome == "loss":
                if t - prev.end_time <= TILT_WINDOW:
                    e_true += planted["tilt_after_loss"]
            if "late_night" in planted and t.hour in LATE_NIGHT_HOURS:
                e_true += planted["late_night"]
            spec = planted.get("time_trouble")
            if spec and prng.random() < spec[0]:
                trouble = True
                e_true += spec[1]

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
        moves = _random_moves(rng, line, plies)
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
