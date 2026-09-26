"""Opening names for any position, from lichess-org/chess-openings (CC0), bundled as ``data/openings.tsv``.

The bundle is the five upstream TSVs (a.tsv ... e.tsv: eco, name, pgn) in one file, with two columns computed
once with python-chess: ``uci`` (the moves in UCI) and ``epd`` (the position after them). Looking a position up
by EPD finds its name even when it was reached by another move order.

Refresh the bundle from a checkout of https://github.com/lichess-org/chess-openings::

    python -m chess_insights.coach.sources.openings_db a.tsv b.tsv c.tsv d.tsv e.tsv \\
        -o src/chess_insights/data/openings.tsv
"""

from __future__ import annotations

import csv
import functools
import io
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional, Sequence

import chess

SOURCE_NAME = "lichess-org/chess-openings"
SOURCE_URL = "https://github.com/lichess-org/chess-openings"
LICENSE = "CC0"
COLUMNS = ("eco", "name", "pgn", "uci", "epd")


@dataclass(frozen=True)
class NamedPosition:
    """A named opening position and how deep into a move list it was found."""

    eco: str
    name: str
    pgn: str  # the moves that define it in chess-openings, "1. e4 c5 2. Nf3"
    ply: int = 0  # plies of the looked-up move list that lead to it (0 = not from a move list)

    @property
    def family(self) -> str:
        """'Sicilian Defense' for 'Sicilian Defense: Open, Lowenthal Variation'."""
        return self.name.split(":", 1)[0].strip()


def normalize_epd(fen_or_epd: str) -> Optional[str]:
    """The EPD python-chess writes for a FEN or EPD (en passant only when a capture is legal); None if invalid.

    FENs from different tools disagree on the en passant square (always after a double step, or only when a
    capture is possible), so both the bundle and every lookup go through python-chess.
    """
    parts = str(fen_or_epd or "").split()
    if len(parts) < 4:
        return None
    try:
        return chess.Board(" ".join(parts[:4]) + " 0 1").epd()
    except ValueError:
        return None


def position_key(board: chess.Board) -> tuple:
    """A cheap hashable key for the position (what an EPD says: pieces, side to move, castling, a usable en passant
    square). About 25 times faster than ``board.epd()``, which matters when every game's opening is walked."""
    ep = board.ep_square if board.ep_square is not None and board.has_legal_en_passant() else None
    return (
        board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.turn, board.clean_castling_rights(), ep,
    )


class OpeningsDB:
    """EPD -> (eco, name) over the bundled chess-openings positions."""

    def __init__(self, rows: Iterable[dict[str, str]]) -> None:
        self.by_epd: dict[str, NamedPosition] = {}
        self.max_plies = 0
        for row in rows:
            epd = (row.get("epd") or "").strip()
            if not epd:
                continue
            plies = len((row.get("uci") or "").split())
            self.max_plies = max(self.max_plies, plies)
            pos = NamedPosition(eco=row.get("eco", ""), name=row.get("name", ""), pgn=row.get("pgn", ""))
            # the upstream files hold each position once; if a future version repeats one, keep the shortest line
            old = self.by_epd.get(epd)
            if old is None or len(pos.pgn) < len(old.pgn):
                self.by_epd[epd] = pos
        self._keys: Optional[dict[tuple, str]] = None

    def __len__(self) -> int:
        return len(self.by_epd)

    @property
    def keys(self) -> dict[tuple, str]:
        """position_key -> EPD for every named position (built on first use, about a third of a second)."""
        if self._keys is None:
            self._keys = {position_key(chess.Board(epd + " 0 1")): epd for epd in self.by_epd}
        return self._keys

    def lookup(self, fen_or_epd: str) -> Optional[NamedPosition]:
        """The named position with this EPD (move counters are ignored), or None."""
        epd = normalize_epd(fen_or_epd)
        return self.by_epd.get(epd) if epd else None

    def named_plies(self, moves_san: Sequence[str], max_plies: Optional[int] = None) -> list[int]:
        """Every ply count k (1-based) whose position after ``moves_san[:k]`` is a named opening position.

        Stops at the deepest named line in the bundle (no position further in can have a name) or at the first
        illegal move.
        """
        limit = min(len(moves_san), self.max_plies if max_plies is None else max_plies)
        keys = self.keys
        board = chess.Board()
        found = []
        for k, san in enumerate(moves_san[:limit], start=1):
            try:
                board.push_san(san)
            except ValueError:
                break
            if position_key(board) in keys:
                found.append(k)
        return found

    def last_named_ply(self, moves_san: Sequence[str]) -> int:
        """Plies up to the last named position of the game (0 when even the first move has no name)."""
        found = self.named_plies(moves_san)
        return found[-1] if found else 0

    def last_named_many(self, move_lists: Sequence[Sequence[str]]) -> list[int]:
        """``last_named_ply`` for many games at once. The games are walked in sorted order on one board, so the
        moves games have in common (most of an opening) are played and looked up once."""
        order = sorted(range(len(move_lists)), key=lambda i: tuple(move_lists[i][: self.max_plies]))
        keys = self.keys
        out = [0] * len(move_lists)
        board = chess.Board()
        path: list[str] = []  # the moves on the board now
        named: list[bool] = []  # named[j]: the position after path[: j + 1] has a name
        for i in order:
            moves = move_lists[i][: self.max_plies]
            common = 0
            while common < len(path) and common < len(moves) and path[common] == moves[common]:
                common += 1
            while len(path) > common:
                board.pop()
                path.pop()
                named.pop()
            for san in moves[common:]:
                try:
                    board.push_san(san)
                except ValueError:
                    break
                path.append(san)
                named.append(position_key(board) in keys)
            out[i] = next((j + 1 for j in range(len(named) - 1, -1, -1) if named[j]), 0)
        return out

    def nearest(self, moves_san: Sequence[str]) -> Optional[NamedPosition]:
        """The deepest named position along ``moves_san`` from the start: the position itself when it has a name,
        else the nearest named one going back along the moves. ``ply`` says how many moves in it is."""
        found = self.named_plies(moves_san)
        if not found:
            return None
        k = found[-1]
        board = chess.Board()
        for san in moves_san[:k]:
            board.push_san(san)
        pos = self.by_epd[self.keys[position_key(board)]]
        return NamedPosition(pos.eco, pos.name, pos.pgn, ply=k)


def _read_bundle() -> list[dict[str, str]]:
    text = resources.files("chess_insights.data").joinpath("openings.tsv").read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


@functools.lru_cache(maxsize=1)
def load() -> OpeningsDB:
    """The bundled chess-openings table, read once per process."""
    return OpeningsDB(_read_bundle())


def lookup(fen_or_epd: str) -> Optional[NamedPosition]:
    """Shortcut: ``load().lookup(fen_or_epd)``."""
    return load().lookup(fen_or_epd)


def eco_name(fen_or_epd: str) -> Optional[tuple[str, str]]:
    """EPD (or FEN) -> (eco, name), e.g. ('B20', 'Sicilian Defense'); None for a position without a name."""
    pos = lookup(fen_or_epd)
    return (pos.eco, pos.name) if pos else None


# --------------------------------------------------------------------------- building the bundle
def pgn_moves(pgn: str) -> list[str]:
    """'1. e4 c5 2. Nf3' -> ['e4', 'c5', 'Nf3'] (move numbers dropped)."""
    return [tok for tok in pgn.split() if not tok[0].isdigit()]


def build_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Upstream rows (eco, name, pgn) with the ``uci`` and ``epd`` columns computed; unreadable lines are skipped."""
    out = []
    for row in rows:
        board = chess.Board()
        ucis = []
        try:
            for san in pgn_moves(row["pgn"]):
                move = board.parse_san(san)
                ucis.append(move.uci())
                board.push(move)
        except (KeyError, ValueError):
            continue
        out.append(
            {"eco": row["eco"], "name": row["name"], "pgn": row["pgn"], "uci": " ".join(ucis), "epd": board.epd()}
        )
    return out


def write_bundle(sources: Sequence[Path], out: Path) -> int:
    """Merge upstream TSVs into one bundle file; returns the number of positions written."""
    rows: list[dict[str, str]] = []
    for path in sources:
        with open(path, newline="", encoding="utf-8") as fh:
            rows += build_rows(csv.DictReader(fh, delimiter="\t"))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Bundle lichess-org/chess-openings TSVs with uci and epd columns.")
    parser.add_argument("sources", nargs="+", type=Path, help="a.tsv b.tsv c.tsv d.tsv e.tsv")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    n = write_bundle(args.sources, args.out)
    print(f"wrote {n} positions to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
