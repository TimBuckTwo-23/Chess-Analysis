"""``python -m chess_insights`` (same as the ``chess-insights`` command).

The guard matters on Windows and macOS: ``demo`` generates games in worker processes that are
started with *spawn*, which re-imports the main module in every worker.
"""

import multiprocessing
import sys

from .cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
