"""Positional concepts at the ends of the two lines: Stockfish 16's classical eval terms and board facts (C1).
Owner: coach-core."""

from __future__ import annotations

from typing import Optional

TERMS = ("Material", "Imbalance", "Pawns", "Knights", "Bishops", "Rooks", "Queens", "Mobility", "King safety",
         "Threats", "Passed", "Space", "Winnable")
LABELS = {
    "Material": "material", "Imbalance": "piece balance", "Pawns": "pawn structure", "Knights": "knights",
    "Bishops": "bishops", "Rooks": "rooks", "Queens": "queen", "Mobility": "piece activity",
    "King safety": "king safety", "Threats": "threats", "Passed": "passed pawns", "Space": "space",
    "Winnable": "winning chances",
}
MIN_DELTA = 0.15  # pawns


def parse_eval_table(text: str) -> Optional[dict[str, tuple[float, float]]]:
    """Stockfish 16 ``eval`` output -> {term: (MG, EG)} totals from White's side; None without the table."""
    return None
