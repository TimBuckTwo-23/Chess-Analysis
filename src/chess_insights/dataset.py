"""Game collections: filtering and conversion to a pandas DataFrame."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

import pandas as pd

from .models import Game

FRAME_COLUMNS = [
    "game_id",
    "url",
    "color",
    "opponent",
    "outcome",
    "score",
    "expected_score",
    "termination",
    "my_result_code",
    "opp_result_code",
    "time_class",
    "time_control",
    "base_seconds",
    "increment",
    "rules",
    "rated",
    "end_time",
    "start_time",
    "my_rating",
    "opp_rating",
    "rating_diff",
    "eco",
    "opening",
    "opening_family",
    "my_accuracy",
    "opp_accuracy",
    "plies",
    "full_moves",
    "has_clocks",
]


def to_frame(games: Iterable[Game]) -> pd.DataFrame:
    """One row per game, sorted by ``end_time`` (tz-aware UTC). Columns: FRAME_COLUMNS."""
    rows = []
    for g in games:
        rows.append(
            {
                "game_id": g.game_id,
                "url": g.url,
                "color": g.color,
                "opponent": g.opponent,
                "outcome": g.outcome,
                "score": g.score,
                "expected_score": g.expected_score,
                "termination": g.termination,
                "my_result_code": g.my_result_code,
                "opp_result_code": g.opp_result_code,
                "time_class": g.time_class,
                "time_control": g.time_control,
                "base_seconds": g.base_seconds,
                "increment": g.increment,
                "rules": g.rules,
                "rated": g.rated,
                "end_time": g.end_time,
                "start_time": g.start_time,
                "my_rating": g.my_rating,
                "opp_rating": g.opp_rating,
                "rating_diff": g.rating_diff,
                "eco": g.eco,
                "opening": g.opening,
                "opening_family": g.opening_family,
                "my_accuracy": g.my_accuracy,
                "opp_accuracy": g.opp_accuracy,
                "plies": g.plies,
                "full_moves": g.full_moves,
                "has_clocks": any(c is not None for c in g.clocks),
            }
        )
    df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    if not df.empty:
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        for col in ("my_rating", "opp_rating", "rating_diff", "base_seconds"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
        for col in ("expected_score", "my_accuracy", "opp_accuracy"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = df.sort_values("end_time", kind="stable").reset_index(drop=True)
    return df


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def filter_games(
    games: Iterable[Game],
    *,
    time_classes: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[str]] = ("chess",),
    rated: Optional[bool] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    min_plies: int = 0,
) -> list[Game]:
    """Return matching games sorted by end_time.

    Defaults keep standard chess only (variants distort every statistic) and
    both rated and casual games.
    """
    since, until = _as_utc(since), _as_utc(until)
    out = []
    for g in games:
        if time_classes and g.time_class not in time_classes:
            continue
        if rules and g.rules not in rules:
            continue
        if rated is not None and g.rated != rated:
            continue
        if since and g.end_time < since:
            continue
        if until and g.end_time >= until:
            continue
        if g.plies < min_plies:
            continue
        out.append(g)
    out.sort(key=lambda g: g.end_time)
    return out


def describe_filters(
    *,
    time_classes: Optional[Sequence[str]] = None,
    rules: Optional[Sequence[str]] = ("chess",),
    rated: Optional[bool] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> str:
    parts = []
    parts.append({None: "rated + casual", True: "rated", False: "casual"}[rated])
    parts.append("+".join(time_classes) if time_classes else "all time controls")
    if rules:
        parts.append("standard chess" if list(rules) == ["chess"] else "/".join(rules))
    if since:
        parts.append(f"since {since:%Y-%m-%d}")
    if until:
        parts.append(f"until {until:%Y-%m-%d}")
    return ", ".join(parts)
