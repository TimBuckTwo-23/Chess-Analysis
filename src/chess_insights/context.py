"""The single object every analysis module receives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .models import Game, GameEval


@dataclass
class AnalysisContext:
    username: str
    games: list[Game]  # already filtered, sorted by end_time ascending
    evals: dict[str, GameEval] = field(default_factory=dict)  # game_id -> engine analysis (may be empty)
    options: dict[str, Any] = field(default_factory=dict)  # free-form knobs (thresholds, min sample sizes)
    _df: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def df(self) -> pd.DataFrame:
        """One row per game (see dataset.to_frame for the columns). Cached."""
        if self._df is None:
            from .dataset import to_frame

            self._df = to_frame(self.games)
        return self._df

    @property
    def games_by_id(self) -> dict[str, Game]:
        return {g.game_id: g for g in self.games}

    def opt(self, key: str, default: Any) -> Any:
        return self.options.get(key, default)
