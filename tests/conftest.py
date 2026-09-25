import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _find_stockfish():
    for cand in (
        os.environ.get("CHESS_INSIGHTS_STOCKFISH"),
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def stockfish_path():
    path = _find_stockfish()
    if not path:
        pytest.skip("Stockfish binary not found (set CHESS_INSIGHTS_STOCKFISH)")
    return path
