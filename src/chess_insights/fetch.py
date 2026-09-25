"""Download a player's complete game history into a local cache, and read it back offline.

Cache layout (shared with ``synth.write_archives``)::

    <cache_dir>/<username>/profile.json                    optional snapshot
    <cache_dir>/<username>/stats.json                      optional snapshot
    <cache_dir>/<username>/games/<YYYY-MM>.json            one file per month:
        {"username": ..., "month": "YYYY-MM", "fetched_at": ISO-8601, "etag": str|null,
         "last_modified": str|null, "complete": bool, "games": [<raw chess.com game objects>]}

A month is ``complete`` once it was fetched more than a day after it ended;
complete months are never downloaded again, so re-running a sync only costs a
couple of requests.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .api import ChessComClient, PlayerNotFound, normalize_username, parse_archive_url
from .parse import game_url_key


def expand_path(path: Path | str) -> Path:
    """``~`` and environment variables (``$HOME``, ``%USERPROFILE%``) expanded: Windows shells expand neither."""
    return Path(os.path.expanduser(os.path.expandvars(str(path))))


DEFAULT_CACHE_DIR = expand_path(os.environ.get("CHESS_INSIGHTS_CACHE") or "~/.chess-insights-cache")

YearMonth = tuple[int, int]


def month_key(ym: YearMonth) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def parse_month(text: str) -> YearMonth:
    """'2024-03' or '2024/03' -> (2024, 3)."""
    y, m = text.replace("/", "-").split("-")[:2]
    ym = (int(y), int(m))
    if not 1 <= ym[1] <= 12:
        raise ValueError(f"bad month: {text!r}")
    return ym


def month_end(ym: YearMonth) -> datetime:
    y, m = ym
    nxt = (y + 1, 1) if m == 12 else (y, m + 1)
    return datetime(nxt[0], nxt[1], 1, tzinfo=timezone.utc)


def _months_since(joined: Optional[int], now: datetime) -> list[YearMonth]:
    if not joined:
        return []
    start = datetime.fromtimestamp(int(joined), tz=timezone.utc)
    ym, last, out = (start.year, start.month), (now.year, now.month), []
    while ym <= last:
        out.append(ym)
        ym = (ym[0] + 1, 1) if ym[1] == 12 else (ym[0], ym[1] + 1)
    return out


def _in_range(ym: YearMonth, since: Optional[YearMonth], until: Optional[YearMonth]) -> bool:
    return (since is None or ym >= since) and (until is None or ym <= until)


def _replace(src: str, dst: Path, attempts: int = 5) -> None:
    """os.replace, retried briefly: on Windows a virus scanner or indexer can hold the target open for a moment."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.1 * 2**i)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        _replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_text(path: Path | str) -> str:
    """A user-supplied text file (PGN / JSON) as str, whatever Windows tool wrote it.

    UTF-8 with or without BOM, UTF-16 with BOM (Windows PowerShell 5 ``>`` / ``Out-File``),
    else cp1252 (legacy Windows programs); undecodable bytes never raise.
    """
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


class GameStore:
    """Per-player month files under ``cache_dir``."""

    def __init__(self, cache_dir: Path | str, username: str) -> None:
        self.username = normalize_username(username)
        self.root = expand_path(cache_dir) / self.username
        self.games_dir = self.root / "games"

    def month_path(self, ym: YearMonth) -> Path:
        return self.games_dir / f"{month_key(ym)}.json"

    def months(self) -> list[YearMonth]:
        if not self.games_dir.exists():
            return []
        out = []
        for p in self.games_dir.glob("*.json"):
            try:
                out.append(parse_month(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def load_month(self, ym: YearMonth) -> Optional[dict[str, Any]]:
        path = self.month_path(ym)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return None  # treat a corrupt file as missing; it will be re-downloaded
        return data if isinstance(data, dict) else None

    def save_month(
        self,
        ym: YearMonth,
        games: list[dict[str, Any]],
        *,
        etag: Optional[str],
        fetched_at: datetime,
        complete: bool,
        last_modified: Optional[str] = None,
    ) -> None:
        _atomic_write_json(
            self.month_path(ym),
            {
                "username": self.username,
                "month": month_key(ym),
                "fetched_at": fetched_at.isoformat(),
                "etag": etag,
                "last_modified": last_modified,
                "complete": complete,
                "games": games,
            },
        )

    def save_snapshot(self, name: str, data: Any) -> None:
        _atomic_write_json(self.root / f"{name}.json", data)

    def load_snapshot(self, name: str) -> Optional[Any]:
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None


@dataclass
class SyncSummary:
    username: str
    months_listed: int = 0
    months_downloaded: int = 0
    months_cached: int = 0
    months_not_modified: int = 0
    months_missing: int = 0
    games_total: int = 0
    requests: int = 0
    account_status: Optional[str] = None  # profile "status": basic, premium, closed, closed:fair_play_violations ...


def sync(
    username: str,
    client: ChessComClient,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    since: Optional[YearMonth] = None,
    until: Optional[YearMonth] = None,
    snapshots: bool = True,
    refresh_all: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    now: Optional[Callable[[], datetime]] = None,
) -> SyncSummary:
    """Bring the local cache up to date with chess.com. Safe to interrupt and re-run.

    ``refresh_all`` re-validates complete months too (cheap conditional requests);
    chess.com adds data to old games later, e.g. accuracies after a Game Review.
    """
    now = now or (lambda: datetime.now(timezone.utc))
    say = progress or (lambda _msg: None)
    store = GameStore(cache_dir, username)
    summary = SyncSummary(username=store.username)
    start_requests = client.requests_made

    profile = client.profile(username)  # also raises PlayerNotFound early for unknown users
    status = profile.get("status") if isinstance(profile, dict) else None
    summary.account_status = status
    closed = str(status or "").startswith("closed")
    if closed:
        say(f"note: the chess.com account {store.username!r} is closed ({status}); using whatever games it still lists")
    if snapshots:
        store.save_snapshot("profile", profile)
        try:
            store.save_snapshot("stats", client.stats(username))
        except Exception as exc:  # stats are a nice-to-have
            say(f"warning: could not fetch stats: {exc}")

    try:
        months = [parse_archive_url(u) for u in client.archives(username)]
    except PlayerNotFound:
        if closed:
            raise PlayerNotFound(f"the chess.com account {store.username!r} is closed ({status}) and has no game archive")
        # chess.com occasionally 404s the archive list of a real player; walk the
        # months from the join date instead (empty months return an empty list).
        say("warning: archive list unavailable, scanning month by month from the join date")
        months = _months_since(profile.get("joined"), now())
    months = [ym for ym in months if _in_range(ym, since, until)]
    summary.months_listed = len(months)
    say(f"{store.username}: {len(months)} monthly archives")

    for i, ym in enumerate(months, 1):
        cached = store.load_month(ym)
        if cached and cached.get("complete") and not refresh_all:
            summary.months_cached += 1
            summary.games_total += len(cached.get("games", []))
            continue
        fetched_at = now()
        result = client.month(
            username, ym[0], ym[1], etag=(cached or {}).get("etag"), last_modified=(cached or {}).get("last_modified")
        )
        if result.missing:
            summary.months_missing += 1
            summary.games_total += len((cached or {}).get("games", []))
            say(f"  [{i}/{len(months)}] {month_key(ym)}: not available right now (HTTP 404), will retry next run")
            continue
        # chess.com files games by a US-time month end, so wait a day past the UTC month end.
        # 410 = "will never be available": don't ask again.
        complete = fetched_at >= month_end(ym) + timedelta(days=1) or result.gone
        if (result.not_modified or result.gone) and cached is not None:
            games = cached.get("games", [])
            summary.months_not_modified += 1
        else:
            games = result.games
            summary.months_downloaded += 1
        store.save_month(
            ym, games, etag=result.etag, last_modified=result.last_modified, fetched_at=fetched_at, complete=complete
        )
        summary.games_total += len(games)
        say(f"  [{i}/{len(months)}] {month_key(ym)}: {len(games)} games")

    summary.requests = client.requests_made - start_requests
    return summary


def _game_keys(g: dict[str, Any]) -> set[str]:
    """Identity of a raw game: its uuid and its era-independent URL key (either one matching = same game)."""
    keys = {k for k in (g.get("uuid"), game_url_key(g.get("url"))) if k}
    return keys or {json.dumps(g, sort_keys=True, default=str)}


def load_games(
    username: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    since: Optional[YearMonth] = None,
    until: Optional[YearMonth] = None,
) -> list[dict[str, Any]]:
    """All cached raw game objects for ``username`` (no network), de-duplicated.

    ``since`` / ``until`` select *archive* months. chess.com files a game under the month it
    ended in US time, so a game that ended early on the 1st (UTC) sits in the previous month's
    archive: callers filtering by date should load one month either side and filter on
    ``end_time`` (the CLI does).
    """
    store = GameStore(cache_dir, username)
    seen: set[str] = set()
    games: list[dict[str, Any]] = []
    for ym in store.months():
        if not _in_range(ym, since, until):
            continue
        data = store.load_month(ym) or {}
        for g in data.get("games") or []:
            if not isinstance(g, dict):
                continue
            keys = _game_keys(g)
            if keys & seen:
                continue
            seen |= keys
            games.append(g)
    return games


def _games_in(data: Any) -> list[dict[str, Any]]:
    """Raw games in any JSON shape people save: a month archive, a list of games or of archives,
    a single game, or a dict of archives keyed by URL."""
    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for item in data:
            out.extend(_games_in(item))
        return out
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("games"), list):
        return [g for g in data["games"] if isinstance(g, dict)]
    if "white" in data and "black" in data:
        return [data]
    if isinstance(data.get("game"), dict) and "white" in data["game"]:
        return [data["game"]]
    out = []
    for value in data.values():
        if isinstance(value, (dict, list)):
            out.extend(_games_in(value))
    return out


def load_json_files(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Read raw games from JSON files (see :func:`_games_in` for the accepted shapes; any text encoding)."""
    out: list[dict[str, Any]] = []
    for p in paths:
        out.extend(_games_in(json.loads(read_text(expand_path(p)))))
    return out
