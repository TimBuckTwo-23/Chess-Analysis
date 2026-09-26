"""Settings for the coaching layer (``--coach`` and friends)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

DEFAULT_COACH_DEPTH = 20
DEFAULT_COACH_MAX = 150
DEFAULT_PROFILE_DEPTH = 10
DEFAULT_DRILL_RATING = (1200, 1600)
DEFAULT_DRILL_SIZE = 30
DEFAULT_PRACTICE_MINUTES = 20  # per day
DEFAULT_LLM_MODEL = "claude-opus-5-5"
REVIEW_STEPS_DAYS = (1, 3, 7, 21)


@dataclass
class CoachConfig:
    """Everything the coaching layer needs besides the games and evals.

    The layer never creates a claim: it explains positions and routes to practice. Every external source is
    optional; with ``offline`` (or no network) the report still builds, and ``Coaching.notes`` says what was
    skipped.
    """

    stockfish: Optional[str] = None  # path; None = engine.find_stockfish()
    depth: int = DEFAULT_COACH_DEPTH  # --coach-depth: the re-search of critical positions
    max_positions: int = DEFAULT_COACH_MAX  # --coach-max: costliest first
    multipv: int = 3
    workers: int = 0  # 0 = CPUs - 1
    hash_mb: int = 64
    cache_dir: Optional[Path] = None  # <cache>/<user>/coach (deep lines), never the evals cache
    sources_cache: Optional[Path] = None  # <cache>/sources (explorer, cloud eval, tablebase, wikibooks)
    profile: bool = True  # the motif profile over every error, both sides (a cheap PV per error)
    profile_depth: int = DEFAULT_PROFILE_DEPTH
    offline: bool = False  # no network sources at all
    lichess_token: Optional[str] = field(default=None, repr=False)  # LICHESS_TOKEN / --lichess-token (explorer)
    puzzle_db: Optional[Path] = None  # filtered Lichess puzzle subset from `chess-insights puzzles-db`
    drill_rating: tuple[int, int] = DEFAULT_DRILL_RATING
    drill_size: int = DEFAULT_DRILL_SIZE
    out_stem: Optional[Path] = None  # report path stem: drill PGNs are written as <stem>-drill-<theme>.pgn
    practice_minutes: int = DEFAULT_PRACTICE_MINUTES
    llm: bool = False  # --coach-llm (needs ANTHROPIC_API_KEY)
    llm_model: str = DEFAULT_LLM_MODEL
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    maia: bool = False  # --maia (optional extra: pip install maia2)
    previous: Optional[dict[str, Any]] = None  # the previous report's JSON (progress, reviews due)
    today: Optional[date] = None  # None = the report's date
    rating_map: dict[str, Any] = field(default_factory=dict)  # overrides for coach.rating_map
