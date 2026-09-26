"""`chess-insights puzzles-db`: download the Lichess puzzle database (CC0) and keep a filtered subset (C2).
Owner: drills.

The database holds about six million puzzles (a few hundred MB compressed); a report needs a few hundred of them.
``download`` streams it once, keeps the puzzles a club player can use (rating 800-2200, well liked, played often)
and writes them to the cache as a plain CSV with Lichess's column names, next to a small JSON file that records
where and when it came from. The report only reads that subset (``iter_rows`` / ``load_subset``); it never
downloads anything by itself.

A Lichess puzzle's FEN is the position *before* the opponent's move: ``Moves[0]`` is that move, and the solution
starts at ``Moves[1]``. ``to_puzzle`` pushes the first move, so a ``DrillPuzzle`` holds the position you solve.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import chess

from ..models import DrillPuzzle

PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
SUBSET_NAME = "lichess_puzzles_subset.csv"
LICENSE = "CC0"
SOURCE_NAME = "Lichess puzzle database"
MIN_RATING, MAX_RATING = 800, 2200
MIN_POPULARITY = 80  # Lichess's -100..100 vote score
MIN_PLAYS = 300
COLUMNS = ("PuzzleId", "FEN", "Moves", "Rating", "Popularity", "NbPlays", "Themes", "OpeningTags")
REQUIRED = ("PuzzleId", "FEN", "Moves", "Rating")
CHUNK_BYTES = 1 << 16
PROGRESS_EVERY = 100_000  # rows read between progress callbacks
TIMEOUT = 60  # seconds without data before the download gives up
USER_AGENT = "chess-insights (+https://github.com/TimBuckTwo-23/Chess-Analysis)"


class PuzzleDbError(RuntimeError):
    """The download failed (network, HTTP status, a broken or truncated file); nothing was replaced."""


@dataclass(frozen=True)
class PuzzleRow:
    """One row of the subset, as read (cheap: no board is built until ``to_puzzle``)."""

    puzzle_id: str
    fen: str  # before the opponent's first move
    moves: tuple[str, ...]  # UCI; moves[0] is the opponent's move
    rating: int
    popularity: int = 0
    nb_plays: int = 0
    themes: tuple[str, ...] = ()
    opening_tags: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return puzzle_url(self.puzzle_id)

    @property
    def solver(self) -> str:
        """The colour that solves the puzzle: the side NOT to move in the Lichess FEN."""
        parts = self.fen.split()
        return "black" if len(parts) > 1 and parts[1] == "w" else "white"

    def cells(self) -> list[Any]:
        return [self.puzzle_id, self.fen, " ".join(self.moves), self.rating, self.popularity, self.nb_plays,
                " ".join(self.themes), " ".join(self.opening_tags)]


def puzzle_url(puzzle_id: str) -> str:
    return f"https://lichess.org/training/{puzzle_id}"


def subset_path(cache_dir: Path) -> Path:
    """Where the filtered subset lives: <cache>/puzzles/lichess_puzzles_subset.csv."""
    return Path(cache_dir) / "puzzles" / SUBSET_NAME


def meta_path(subset: Path) -> Path:
    """The sidecar JSON next to a subset: download date, URL, licence, row count."""
    return Path(subset).with_suffix(".json")


def read_meta(subset: Path) -> dict[str, Any]:
    """The sidecar of ``subset`` ({} when missing or unreadable)."""
    try:
        data = json.loads(meta_path(subset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- reading
def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_row(row: dict[str, Any]) -> Optional[PuzzleRow]:
    """A PuzzleRow from a CSV row with Lichess's column names; None when a required field is missing or broken.

    Popularity and NbPlays count as 0 when the columns are absent (a hand-made fixture), but a present,
    non-numeric value makes the row malformed.
    """
    if not isinstance(row, dict) or any(not row.get(k) for k in REQUIRED):
        return None
    moves = tuple(str(row["Moves"]).split())
    rating = _int(row["Rating"])
    popularity = _int(row.get("Popularity") or 0)
    plays = _int(row.get("NbPlays") or 0)
    fen = str(row["FEN"]).strip()
    if len(moves) < 2 or rating is None or popularity is None or plays is None or len(fen.split()) < 4:
        return None
    return PuzzleRow(
        puzzle_id=str(row["PuzzleId"]).strip(),
        fen=fen,
        moves=moves,
        rating=rating,
        popularity=popularity,
        nb_plays=plays,
        themes=tuple(str(row.get("Themes") or "").split()),
        opening_tags=tuple(str(row.get("OpeningTags") or "").split()),
    )


def keep(row: PuzzleRow) -> bool:
    """The subset's filter: rating 800-2200, Popularity >= 80, NbPlays >= 300."""
    return MIN_RATING <= row.rating <= MAX_RATING and row.popularity >= MIN_POPULARITY and row.nb_plays >= MIN_PLAYS


def iter_rows(
    path: Path,
    rating: Optional[tuple[int, int]] = None,
    themes: Optional[Iterable[str]] = None,
    tagged: bool = False,
) -> Iterator[PuzzleRow]:
    """Every well-formed row of a subset (or a fixture with the Lichess columns), in file order.

    Filters, checked on the raw text before a row is parsed (a pass over a full-size subset stays quick):
    ``rating`` = (low, high); ``themes`` and/or ``tagged``: only rows with one of those themes or (``tagged``)
    with opening tags. Malformed rows are skipped; a missing file yields nothing.
    """
    wanted = set(themes) if themes is not None else None
    selective = wanted is not None or tagged
    try:
        handle = open(path, newline="", encoding="utf-8")
    except OSError:
        return
    with handle:
        reader = csv.reader(handle)
        header = next(reader, None) or []
        col = {name: i for i, name in enumerate(header)}
        at_rating, at_themes, at_tags = col.get("Rating"), col.get("Themes"), col.get("OpeningTags")
        for cells in reader:
            if len(cells) > len(header):
                continue
            if rating is not None and at_rating is not None and at_rating < len(cells):
                value = _int(cells[at_rating])
                if value is None or not rating[0] <= value <= rating[1]:
                    continue
            if selective:
                row_themes = cells[at_themes].split() if at_themes is not None and at_themes < len(cells) else []
                has_tags = at_tags is not None and at_tags < len(cells) and bool(cells[at_tags].strip())
                if not ((tagged and has_tags) or (wanted and wanted.intersection(row_themes))):
                    continue
            parsed = parse_row(dict(zip(header, cells)))
            if parsed is not None and (rating is None or rating[0] <= parsed.rating <= rating[1]):
                yield parsed


def to_puzzle(row: PuzzleRow) -> Optional[DrillPuzzle]:
    """The puzzle as you solve it: the position after the opponent's first move, the rest as the solution.

    None when the FEN or any move is not legal (a corrupt row).
    """
    try:
        board = chess.Board(row.fen)
        first = chess.Move.from_uci(row.moves[0])
        if first not in board.legal_moves:
            return None
        board.push(first)
        fen = board.fen()
        uci, san = [], []
        for text in row.moves[1:]:
            move = chess.Move.from_uci(text)
            if move not in board.legal_moves:
                return None
            san.append(board.san(move))
            uci.append(text)
            board.push(move)
    except ValueError:
        return None
    return DrillPuzzle(
        puzzle_id=row.puzzle_id,
        fen=fen,
        solution_uci=uci,
        solution_san=san,
        rating=row.rating,
        themes=list(row.themes),
        url=row.url,
        opening_tags=list(row.opening_tags),
    )


def load_subset(path: Path) -> list[DrillPuzzle]:
    """Read a subset written by ``download`` (or a test fixture with the Lichess columns), in file order.

    Rows that are malformed or not legal chess are skipped; a missing file gives an empty list. For a full-size
    subset prefer ``iter_rows`` and convert only the rows you keep: building every board takes a while.
    """
    puzzles = []
    for row in iter_rows(path):
        puzzle = to_puzzle(row)
        if puzzle is not None:
            puzzles.append(puzzle)
    return puzzles


# --------------------------------------------------------------------------- download
def _decompressed(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Decompress a zstd stream chunk by chunk (several frames are fine); PuzzleDbError if it is cut short."""
    import zstandard

    dctx = zstandard.ZstdDecompressor()
    obj = dctx.decompressobj()
    seen = pending = False  # pending: a frame has started and not ended yet
    try:
        for chunk in chunks:
            while chunk:
                seen = pending = True
                out = obj.decompress(chunk)
                if out:
                    yield out
                if obj.eof:  # the frame ended: anything left over starts the next one
                    pending = False
                    chunk = obj.unused_data
                    obj = dctx.decompressobj()
                else:
                    chunk = b""
    except zstandard.ZstdError as exc:
        raise PuzzleDbError(f"the puzzle database is not a valid .zst file ({exc})") from exc
    if not seen or pending:
        raise PuzzleDbError("the puzzle database download was cut short (incomplete .zst file)")


def _lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Text lines (each with its newline) from a stream of byte chunks."""
    rest = b""
    for chunk in chunks:
        rest += chunk
        *complete, rest = rest.split(b"\n")
        for line in complete:
            yield line.decode("utf-8", errors="replace") + "\n"
    if rest:
        yield rest.decode("utf-8", errors="replace")


def _atomic_write(path: Path, write: Callable[[Any], None]) -> None:
    """``write(handle)`` into a temporary file next to ``path``, then move it into place in one step."""
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            write(handle)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def download(
    cache_dir: Path,
    *,
    url: str = PUZZLE_DB_URL,
    progress: Optional[Callable[[int], None]] = None,
    today: Optional[date] = None,
    session: Any = None,
    timeout: float = TIMEOUT,
) -> Path:
    """Stream the database, keep rating 800-2200, Popularity >= 80, NbPlays >= 300 (PuzzleId, FEN, Moves,
    Rating, Popularity, NbPlays, Themes, OpeningTags) and record the download date. Returns the subset path.

    The subset and its sidecar JSON (``meta_path``: date, URL, licence, rows) are each written atomically: a
    failed or interrupted download raises PuzzleDbError and leaves any earlier subset as it was. Malformed rows
    are skipped. ``progress`` is called with the number of rows read so far, every 100,000 rows and once at the
    end. ``session`` is a ``requests.Session`` (or anything with the same ``get``); the default is a new one.
    """
    if session is None:
        import requests

        session = requests.Session()
    target = subset_path(cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = session.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except Exception as exc:  # noqa: BLE001 — requests' errors, or a broken session
        raise PuzzleDbError(f"could not download {url} ({type(exc).__name__}: {exc})") from exc
    status = getattr(response, "status_code", 200)
    if status != 200:
        raise PuzzleDbError(f"could not download {url} (HTTP {status})")
    expected = _int((getattr(response, "headers", None) or {}).get("Content-Length"))
    received = 0

    def chunks() -> Iterator[bytes]:
        nonlocal received
        try:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    received += len(chunk)
                    yield chunk
        except Exception as exc:  # noqa: BLE001 — a dropped connection mid-stream
            raise PuzzleDbError(f"the download of {url} failed ({type(exc).__name__}: {exc})") from exc

    counts = {"read": 0, "kept": 0}

    def write_subset(handle: Any) -> None:
        reader = csv.reader(_lines(_decompressed(chunks())))
        header = next(reader, None)
        if not header or any(k not in header for k in REQUIRED):
            raise PuzzleDbError(f"{url} does not look like the Lichess puzzle database (header {header!r})")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for cells in reader:
            counts["read"] += 1
            if progress and counts["read"] % PROGRESS_EVERY == 0:
                progress(counts["read"])
            row = parse_row(dict(zip(header, cells))) if len(cells) == len(header) else None
            if row is not None and keep(row):
                writer.writerow(row.cells())
                counts["kept"] += 1
        if expected is not None and received < expected:
            raise PuzzleDbError(f"the download of {url} was cut short ({received} of {expected} bytes)")

    try:
        _atomic_write(target, write_subset)
    except csv.Error as exc:
        raise PuzzleDbError(f"the puzzle database could not be read as CSV ({exc})") from exc
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if progress:
        progress(counts["read"])
    meta = {
        "source": SOURCE_NAME,
        "url": url,
        "license": LICENSE,
        "downloaded": (today or datetime.now(timezone.utc).date()).isoformat(),
        "rows": counts["kept"],
        "rows_read": counts["read"],
        "filters": {"rating": [MIN_RATING, MAX_RATING], "min_popularity": MIN_POPULARITY, "min_plays": MIN_PLAYS},
    }
    _atomic_write(meta_path(target), lambda handle: json.dump(meta, handle, indent=1))
    return target
