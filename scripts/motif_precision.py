"""Precision and recall of the motif detectors against Lichess's puzzle themes.

    python scripts/motif_precision.py [--csv tests/fixtures/lichess_puzzles_sample.csv]
                                      [--refutation] [--extend PLIES --stockfish PATH --nodes N]

Default: the gate measurement of tests/test_motifs.py. A Lichess puzzle's FEN is the position before the
opponent's move; the solution is Moves[1:]. We run ``detect_moves`` on the position after Moves[0] and count the
motifs carried out by the solver. Precision of a theme = of the puzzles we tag with it, the share Lichess tagged
with it; recall = of the puzzles Lichess tagged, the share we found.

``--refutation`` runs the puzzle the way the coaching reads a refutation: from the FEN before the opponent's
move, counting the second side's motifs (the recapture check then sees the move before).

``--extend PLIES`` stresses the detectors the way an engine line does: the solution is continued with
Stockfish's principal variation until the line is PLIES long, so patterns that turn up after the tactic is over
count against precision (a lower bound: some of them are real motifs Lichess had no reason to tag).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chess_insights.coach.motifs import GATED_THEMES, THEMES, detect_moves, puzzle_motifs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def score(rows, *, refutation=False, extend=0, engine=None, nodes=30000):
    tagged, hits, labelled = Counter(), Counter(), Counter()
    for row in rows:
        moves = row["Moves"].split()
        board = chess.Board(row["FEN"])
        if extend and engine is not None:
            end = board.copy()
            for uci in moves:
                end.push_uci(uci)
            if not end.is_game_over() and len(moves) - 1 < extend:
                info = engine.analyse(end, chess.engine.Limit(nodes=nodes))
                moves = moves + [m.uci() for m in info.get("pv", [])][: extend - (len(moves) - 1)]
        if refutation:
            found = {m.theme for m in detect_moves(board.fen(), moves, max_plies=len(moves)) if m.side == "second"}
        else:
            found = {m.theme for m in puzzle_motifs(board.fen(), moves)}
        labels = set(row["Themes"].split())
        for theme in THEMES:
            labelled[theme] += theme in labels
            if theme in found:
                tagged[theme] += 1
                hits[theme] += theme in labels
    return tagged, hits, labelled


def table(tagged, hits, labelled) -> str:
    out = [f"{'theme':18} {'tagged':>6} {'precision':>9} {'lichess':>7} {'recall':>6}  gated"]
    for theme in THEMES:
        p = f"{hits[theme] / tagged[theme]:.2f}" if tagged[theme] and labelled[theme] else "-"
        r = f"{hits[theme] / labelled[theme]:.2f}" if labelled[theme] else "-"
        gated = "yes" if theme in GATED_THEMES else ""
        out.append(f"{theme:18} {tagged[theme]:6} {p:>9} {labelled[theme]:7} {r:>6}  {gated}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=str(ROOT / "tests" / "fixtures" / "lichess_puzzles_sample.csv"))
    ap.add_argument("--refutation", action="store_true")
    ap.add_argument("--extend", type=int, default=0, help="continue each solution to this many plies (Stockfish)")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--nodes", type=int, default=30000)
    args = ap.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish) if args.extend else None
    try:
        if engine is not None:
            engine.configure({"Threads": 1, "Hash": 16})
        start = time.perf_counter()
        result = score(rows, refutation=args.refutation, extend=args.extend, engine=engine, nodes=args.nodes)
        print(f"{len(rows)} puzzles in {time.perf_counter() - start:.1f}s")
    finally:
        if engine is not None:
            engine.quit()
    print(table(*result))


if __name__ == "__main__":
    main()
