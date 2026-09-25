"""Builders for Game / PlyEval objects in tests (no parser or network needed)."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from typing import Any

from chess_insights.models import Game, GameEval, PlyEval

_counter = itertools.count(1)

# A plausible 20-ply Italian Game used as the default move list.
DEFAULT_MOVES = [
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6",
    "O-O", "O-O", "Re1", "a6", "Bb3", "Ba7", "h3", "h6", "Nbd2", "Re8",
]


def make_game(**overrides: Any) -> Game:
    """A valid Game with sensible defaults; override any field by keyword.

    Handy extras (not Game fields):
      * ``minutes_ago`` / ``end_time`` to place the game in time
      * ``with_clocks=True`` to synthesise clocks from base_seconds (default True for live games)
    """
    n = next(_counter)
    minutes_ago = overrides.pop("minutes_ago", None)
    with_clocks = overrides.pop("with_clocks", True)
    base_end = datetime(2024, 6, 1, 18, 0, tzinfo=timezone.utc) + timedelta(minutes=15 * n)
    if minutes_ago is not None:
        base_end = datetime(2024, 12, 31, 23, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)

    moves = overrides.pop("moves_san", list(DEFAULT_MOVES))
    defaults: dict[str, Any] = dict(
        game_id=f"game-{n}",
        url=f"https://www.chess.com/game/live/{1000000 + n}",
        username="tester",
        color="white",
        opponent=f"opp{n % 7}",
        outcome="win",
        my_result_code="win",
        opp_result_code="resigned",
        termination="resignation",
        time_class="blitz",
        time_control="300",
        base_seconds=300,
        increment=0,
        rules="chess",
        rated=True,
        end_time=base_end,
        start_time=base_end - timedelta(minutes=8),
        my_rating=1500,
        opp_rating=1500,
        eco="C50",
        opening="Italian Game: Giuoco Pianissimo",
        opening_family="Italian Game",
        my_accuracy=None,
        opp_accuracy=None,
        initial_fen=None,
        moves_san=moves,
        clocks=[],
        pgn="",
    )
    defaults.update(overrides)
    if not defaults["clocks"] and with_clocks and defaults["time_class"] != "daily" and defaults["base_seconds"]:
        base = float(defaults["base_seconds"])
        clocks = []
        remaining = {0: base, 1: base}
        for i in range(len(defaults["moves_san"])):
            side = i % 2
            remaining[side] = max(0.0, remaining[side] - 5.0 + float(defaults["increment"] or 0))
            clocks.append(round(remaining[side], 1))
        defaults["clocks"] = clocks
    elif not defaults["clocks"]:
        defaults["clocks"] = [None] * len(defaults["moves_san"])

    # keep outcome/result codes consistent when only `outcome` was overridden
    if "outcome" in overrides and "my_result_code" not in overrides:
        defaults["my_result_code"] = {"win": "win", "draw": "agreed", "loss": "resigned"}[defaults["outcome"]]
    if "outcome" in overrides and "opp_result_code" not in overrides:
        defaults["opp_result_code"] = {"win": "resigned", "draw": "agreed", "loss": "win"}[defaults["outcome"]]
    if "outcome" in overrides and "termination" not in overrides:
        defaults["termination"] = {"win": "resignation", "draw": "agreement", "loss": "resignation"}[
            defaults["outcome"]
        ]
    return Game(**defaults)


def make_ply_eval(**overrides: Any) -> PlyEval:
    defaults: dict[str, Any] = dict(
        ply=0,
        mover="white",
        is_user=True,
        san="e4",
        best_san="e4",
        cp_before=20,
        cp_after=20,
        mate_before=None,
        mate_after=None,
        win_before=51.8,
        win_after=51.8,
        accuracy=100.0,
        cp_loss=0,
        judgement=None,
        phase="opening",
        clock_after=None,
        time_spent=None,
        tags=[],
    )
    defaults.update(overrides)
    return PlyEval(**defaults)


def make_game_eval(game_id: str, plies: list[PlyEval], **overrides: Any) -> GameEval:
    defaults: dict[str, Any] = dict(
        game_id=game_id, engine="Stockfish 16", depth=10, plies=plies, my_accuracy=None, opp_accuracy=None
    )
    defaults.update(overrides)
    return GameEval(**defaults)


def all_tables(mr):
    """A module's tables, including those a chart carries under "Show the numbers" (Chart.table)."""
    return list(mr.tables) + [c.table for c in mr.charts if getattr(c, "table", None) is not None]
