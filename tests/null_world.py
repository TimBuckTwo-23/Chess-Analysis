"""A "null world": games with no real strengths or weaknesses.

Every result is drawn from the Elo expectation, and openings, terminations,
clocks and schedules are independent of performance. Any strength/weakness an
analysis module reports on this data is a false positive. Used by
test_null_calibration.py to keep the false-claim rate low.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

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


def _random_moves(rng: random.Random, start: list[str], target_plies: int) -> list[str]:
    board = chess.Board()
    moves = []
    for san in start:
        board.push_san(san)
        moves.append(san)
    while len(moves) < target_plies and not board.is_game_over():
        legal = list(board.legal_moves)
        move = rng.choice(legal)
        moves.append(board.san(move))
        board.push(move)
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


def null_games(n: int = 400, seed: int = 0, username: str = "nullplayer") -> list[Game]:
    rng = random.Random(seed)
    rating = {"blitz": 1500, "rapid": 1560, "bullet": 1420}
    t = datetime(2025, 1, 3, 18, 0, tzinfo=timezone.utc)
    games: list[Game] = []
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
        p_draw = 0.08
        p_win = max(0.0, min(1.0 - p_draw, e_true - p_draw / 2))
        u = rng.random()
        outcome = "win" if u < p_win else ("draw" if u < p_win + p_draw else "loss")
        score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
        change = round(16 * (score - e))
        rating[tc] = my_pre + change

        line, family, eco = rng.choice(OPENINGS[color])
        plies = max(8, min(160, int(rng.gauss(70, 25))))  # length independent of the result
        moves = _random_moves(rng, line, plies)
        if outcome == "draw":
            my_code = opp_code = rng.choice(["agreed", "repetition", "insufficient", "stalemate"])
            termination = {"agreed": "agreement"}.get(my_code, my_code)
            flag_side = None
        else:
            reason = rng.choices(["resigned", "checkmated", "timeout", "abandoned"], weights=[70, 14, 14, 2])[0]
            termination = {"resigned": "resignation", "checkmated": "checkmate", "timeout": "timeout", "abandoned": "abandoned"}[reason]
            my_code, opp_code = ("win", reason) if outcome == "win" else (reason, "win")
            loser_is_white = (outcome == "loss") == (color == "white")
            flag_side = (0 if loser_is_white else 1) if reason == "timeout" else None
        clocks = _clocks(rng, len(moves), base, inc, flag_side)

        # schedule: sessions of quick re-queues, independent of results
        duration = timedelta(seconds=min(2 * base + 80 * inc, 60 + len(moves) * base / 60))
        start = t
        end = start + duration
        t = end + (timedelta(minutes=rng.uniform(0.5, 8)) if rng.random() < 0.75 else timedelta(hours=rng.uniform(3, 30)))
        games.append(
            Game(
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
        )
    fill_pregame_ratings(games)
    return games
