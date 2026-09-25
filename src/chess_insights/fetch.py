"""Download a player's complete game history into a local cache, and read it back offline.

Cache layout (shared with ``synth.write_archives``)::

    <cache_dir>/<username>/profile.json                    optional snapshot
    <cache_dir>/<username>/stats.json                      optional snapshot
    <cache_dir>/<username>/games/<YYYY-MM>.json            one file per month:
        {"username": ..., "month": "YYYY-MM", "fetched_at": ISO-8601, "etag": str|null,
         "complete": bool, "games": [<raw chess.com game objects>]}

A month is ``complete`` once it was fetched more than a day after it ended;
complete months are never downloaded again, so re-running a sync only costs a
couple of requests.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .api import ChessComClient, normalize_username, parse_archive_url

DEFAULT_CACHE_DIR = Path(os.environ.get("CHESS_INSIGHTS_CACHE", "~/.chess-insights-cache")).expanduser()

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


def _in_range(ym: YearMonth, since: Optional[YearMonth], until: Optional[YearMonth]) -> bool:
    return (since is None or ym >= since) and (until is None or ym <= until)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class GameStore:
    """Per-player month files under ``cache_dir``."""

    def __init__(self, cache_dir: Path | str, username: str) -> None:
        self.username = normalize_username(username)
        self.root = Path(cache_dir).expanduser() / self.username
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
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None  # treat a corrupt file as missing; it will be re-downloaded

    def save_month(
        self,
        ym: YearMonth,
        games: list[dict[str, Any]],
        *,
        etag: Optional[str],
        fetched_at: datetime,
        complete: bool,
    ) -> None:
        _atomic_write_json(
            self.month_path(ym),
            {
                "username": self.username,
                "month": month_key(ym),
                "fetched_at": fetched_at.isoformat(),
                "etag": etag,
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
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)


@dataclass
class SyncSummary:
    username: str
    months_listed: int = 0
    months_downloaded: int = 0
    months_cached: int = 0
    months_not_modified: int = 0
    games_total: int = 0
    requests: int = 0


def sync(
    username: str,
    client: ChessComClient,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    since: Optional[YearMonth] = None,
    until: Optional[YearMonth] = None,
    snapshots: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    now: Optional[Callable[[], datetime]] = None,
) -> SyncSummary:
    """Bring the local cache up to date with chess.com. Safe to interrupt and re-run."""
    now = now or (lambda: datetime.now(timezone.utc))
    say = progress or (lambda _msg: None)
    store = GameStore(cache_dir, username)
    summary = SyncSummary(username=store.username)
    start_requests = client.requests_made

    if snapshots:
        store.save_snapshot("profile", client.profile(username))
        try:
            store.save_snapshot("stats", client.stats(username))
        except Exception as exc:  # stats are a nice-to-have
            say(f"warning: could not fetch stats: {exc}")

    months = [parse_archive_url(u) for u in client.archives(username)]
    months = [ym for ym in months if _in_range(ym, since, until)]
    summary.months_listed = len(months)
    say(f"{store.username}: {len(months)} monthly archives")

    for i, ym in enumerate(months, 1):
        cached = store.load_month(ym)
        if cached and cached.get("complete"):
            summary.months_cached += 1
            summary.games_total += len(cached.get("games", []))
            continue
        fetched_at = now()
        result = client.month(username, ym[0], ym[1], etag=(cached or {}).get("etag"))
        complete = fetched_at >= month_end(ym) + timedelta(days=1)
        if result.not_modified and cached is not None:
            games = cached.get("games", [])
            summary.months_not_modified += 1
        else:
            games = result.games
            summary.months_downloaded += 1
        store.save_month(ym, games, etag=result.etag, fetched_at=fetched_at, complete=complete)
        summary.games_total += len(games)
        say(f"  [{i}/{len(months)}] {month_key(ym)}: {len(games)} games")

    summary.requests = client.requests_made - start_requests
    return summary


def load_games(
    username: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    since: Optional[YearMonth] = None,
    until: Optional[YearMonth] = None,
) -> list[dict[str, Any]]:
    """All cached raw game objects for ``username`` (no network), de-duplicated."""
    store = GameStore(cache_dir, username)
    seen: set[str] = set()
    games: list[dict[str, Any]] = []
    for ym in store.months():
        if not _in_range(ym, since, until):
            continue
        data = store.load_month(ym) or {}
        for g in data.get("games", []):
            key = g.get("uuid") or g.get("url") or json.dumps(g, sort_keys=True)[:200]
            if key in seen:
                continue
            seen.add(key)
            games.append(g)
    return games


def load_json_files(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Read raw games from arbitrary JSON files: a month archive ({"games": [...]}) or a list of games."""
    out: list[dict[str, Any]] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            out.extend(data.get("games", []))
        elif isinstance(data, list):
            out.extend(data)
    return out
