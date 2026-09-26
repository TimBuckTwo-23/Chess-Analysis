"""`chess-insights puzzles-db`: download the Lichess puzzle database (CC0) and keep a filtered subset (C2).
Owner: drills."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable, Optional

from ..models import DrillPuzzle

PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
SUBSET_NAME = "lichess_puzzles_subset.csv"


def subset_path(cache_dir: Path) -> Path:
    """Where the filtered subset lives: <cache>/puzzles/lichess_puzzles_subset.csv."""
    return Path(cache_dir) / "puzzles" / SUBSET_NAME


def download(
    cache_dir: Path,
    *,
    url: str = PUZZLE_DB_URL,
    progress: Optional[Callable[[int], None]] = None,
    today: Optional[date] = None,
) -> Path:
    """Stream the database, keep rating 800-2200, Popularity >= 80, NbPlays >= 300 (PuzzleId, FEN, Moves,
    Rating, Themes, OpeningTags) and record the download date. Returns the subset path."""
    raise NotImplementedError


def load_subset(path: Path) -> list[DrillPuzzle]:
    """Read a subset written by ``download`` (or a test fixture with the Lichess columns)."""
    return []
