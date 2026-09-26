"""The HTTP layer every public source goes through: polite, cached, and quiet when a source is unavailable.

* One request at a time in the whole process (a module-level lock), and at least ``MIN_INTERVAL`` seconds between
  two requests to the same host.
* A disk cache: ``<sources_cache>/<source>/<sha1 of the URL>.json`` holding the URL, the HTTP status, the
  retrieval date and the answer. Each source has its own time to live (``TTL_DAYS``); the tablebase never
  changes, so its answers are kept for good. 404 answers are cached too ("not in the database" is an answer).
* ``offline`` (or no cache folder) means cache only: never a request.
* HTTP 429: wait ``RATE_LIMIT_WAIT`` seconds once (what Lichess asks for), then skip that source for the rest of
  the run. 401/403: the source needs a token (the opening explorer does). A timeout or no connection: skip the
  source for the rest of the run, so the report never waits on it twice.
* Every failure becomes one line in the notes list (each line once) and ``None`` for the caller.

The cache files use the same layout as the recorded responses in ``tests/fixtures/sources`` (``url``,
``status``, ``content_type``, ``retrieved``, ``body``), so a recorded answer can be dropped into a cache folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import quote, urlsplit

import chess
import requests

from ...api import DEFAULT_USER_AGENT, header_safe

MIN_INTERVAL = 1.0  # seconds between two requests to one host
RATE_LIMIT_WAIT = 60.0  # after an HTTP 429 (Lichess: "wait a full minute")
TIMEOUT = (5.0, 15.0)  # connect, read (seconds)
MAX_FAILURES = 3  # other errors (HTTP 5xx, unreadable answers) before a source is skipped for the run

# Plain-language names, used in notes and credits
NAMES = {
    "explorer": "Lichess opening explorer",
    "cloud_eval": "Lichess cloud eval",
    "tablebase": "Lichess tablebase",
    "wikibooks": "Wikibooks",
}
# Days a cached answer stays fresh (None = for good)
TTL_DAYS: dict[str, Optional[int]] = {"explorer": 30, "cloud_eval": 30, "tablebase": None, "wikibooks": 90}
TOKEN_NOTE = (
    "The Lichess opening explorer (what masters and players at your level play) needs a free Lichess personal "
    "access token: set LICHESS_TOKEN or pass --lichess-token. Skipped this time."
)

_LOCK = threading.Lock()  # one request at a time, across every source and thread
_LAST_REQUEST: dict[str, float] = {}  # host -> clock() when its last request finished

# What a Fetcher uses when none is passed in; tests replace these (recorded answers, no real waiting).
session_factory: Callable[[], Any] = requests.Session


def default_sleep(seconds: float) -> None:
    time.sleep(seconds)


def reset_throttle() -> None:
    """Forget when each host was last called (tests)."""
    with _LOCK:
        _LAST_REQUEST.clear()


def normalize_fen(fen: str) -> Optional[str]:
    """The FEN python-chess writes for ``fen`` (en passant only when legal), or None when it is not a position."""
    try:
        return chess.Board(fen).fen()
    except (ValueError, TypeError):
        return None


def quote_fen(fen: str) -> str:
    """A FEN as a query value, the way the Lichess APIs are called: spaces as %20, slashes kept."""
    return quote(fen, safe="/")


def cache_path(cache_dir: Path, source: str, url: str) -> Path:
    """``<cache_dir>/<source>/<sha1 of url>.json``."""
    return Path(cache_dir) / source / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"


@dataclass
class Fetched:
    """One answer from a source (fresh or from the cache)."""

    url: str
    status: int  # 200, or 404 when the source has nothing for this request
    data: Any  # parsed JSON body; None for a 404
    retrieved: str  # ISO date of the retrieval, "2026-09-26"
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.data is not None


def _user_agent() -> str:
    contact = header_safe(os.environ.get("CHESS_INSIGHTS_CONTACT") or "")
    return f"{DEFAULT_USER_AGENT} contact: {contact}" if contact else DEFAULT_USER_AGENT


class Fetcher:
    """GET JSON from the public sources with the rules above. One Fetcher per coaching step and run.

    ``session``, ``sleep``, ``clock`` and ``now`` are injectable for tests (or replace the module's
    ``session_factory`` and ``default_sleep``). ``max_requests`` caps the network requests per source in this
    run (cache hits are free); past the cap a source answers from the cache only.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]],
        *,
        offline: bool = False,
        notes: Optional[list[str]] = None,
        lichess_token: Optional[str] = None,
        session: Optional[Any] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        now: Optional[Callable[[], datetime]] = None,
        timeout: Union[float, tuple[float, float]] = TIMEOUT,
        min_interval: float = MIN_INTERVAL,
        rate_limit_wait: float = RATE_LIMIT_WAIT,
        max_requests: Optional[dict[str, int]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.offline = bool(offline)
        self.notes = notes if notes is not None else []
        self.lichess_token = (lichess_token or "").strip() or None
        self._session = session
        self._sleep_fn = sleep
        self._clock = clock
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.timeout = timeout
        self.min_interval = min_interval
        self.rate_limit_wait = rate_limit_wait
        self.max_requests = dict(max_requests or {})
        self.requests_made: dict[str, int] = {}
        self.disabled: set[str] = set()  # sources skipped for the rest of this run
        self._waited_429: set[str] = set()
        self._failures: dict[str, int] = {}

    @classmethod
    def from_config(cls, cfg: Any, notes: Optional[list[str]] = None, **kwargs: Any) -> "Fetcher":
        """A Fetcher for a ``CoachConfig`` (sources_cache, offline, lichess_token; LICHESS_TOKEN as a fallback)."""
        token = getattr(cfg, "lichess_token", None) or os.environ.get("LICHESS_TOKEN")
        return cls(
            getattr(cfg, "sources_cache", None),
            offline=getattr(cfg, "offline", False),
            notes=notes,
            lichess_token=token,
            **kwargs,
        )

    # ------------------------------------------------------------------ notes
    def note(self, text: str) -> None:
        """Add ``text`` to the notes once."""
        if text not in self.notes:
            self.notes.append(text)

    @property
    def can_request(self) -> bool:
        """False when offline or without a cache folder: answers come from the cache only."""
        return not self.offline and self.cache_dir is not None

    # ------------------------------------------------------------------ cache
    def _read_cache(self, source: str, url: str) -> Optional[tuple[Fetched, bool]]:
        """(answer, still fresh) from the disk cache, or None."""
        if self.cache_dir is None:
            return None
        path = cache_path(self.cache_dir, source, url)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            status = int(doc["status"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        stamp = str(doc.get("retrieved") or doc.get("recorded") or "")
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
        except ValueError:
            when = None
        ttl = TTL_DAYS.get(source, 30)
        fresh = when is not None and (ttl is None or self._now() - when <= timedelta(days=ttl))
        data = doc.get("body") if status == 200 else None
        retrieved = when.date().isoformat() if when else ""
        return Fetched(url=url, status=status, data=data, retrieved=retrieved, from_cache=True), fresh

    def _write_cache(self, source: str, url: str, status: int, content_type: str, body: Any, when: datetime) -> None:
        if self.cache_dir is None:
            return
        path = cache_path(self.cache_dir, source, url)
        doc = {"url": url, "status": status, "content_type": content_type, "retrieved": when.isoformat(), "body": body}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as exc:
            self.note(f"Could not write the sources cache ({type(exc).__name__}); answers were used but not kept.")

    # ------------------------------------------------------------------ network
    def _sleep(self, seconds: float) -> None:
        (self._sleep_fn or default_sleep)(seconds)

    def _session_get(self, url: str, headers: dict[str, str]) -> Any:
        if self._session is None:
            self._session = session_factory()
        return self._session.get(url, headers=headers, timeout=self.timeout)

    def _throttled_get(self, url: str, headers: dict[str, str]) -> Any:
        """One request, with the process-wide lock held and the per-host interval respected."""
        host = urlsplit(url).netloc
        with _LOCK:
            last = _LAST_REQUEST.get(host)
            if last is not None:
                wait = self.min_interval - (self._clock() - last)
                if wait > 0:
                    self._sleep(wait)
            try:
                return self._session_get(url, headers)
            finally:
                _LAST_REQUEST[host] = self._clock()

    def _fail(self, source: str, text: str, disable: bool) -> None:
        self.note(text)
        if disable:
            self.disabled.add(source)

    def get(
        self,
        source: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        cache_only: bool = False,
        miss_note: Optional[str] = None,
    ) -> Optional[Fetched]:
        """The JSON answer for ``url`` (fresh from the cache, or fetched), or None when unavailable.

        ``cache_only`` answers from the cache whatever its age and never makes a request; ``miss_note`` is the
        note to leave when that finds nothing.
        """
        name = NAMES.get(source, source)
        cached = self._read_cache(source, url)
        if cached is not None and (cached[1] or cache_only or not self.can_request or source in self.disabled):
            return cached[0]  # fresh, or stale but better than nothing when no request may be made
        if cache_only:
            if miss_note:
                self.note(miss_note)
            return None
        if self.offline:
            self.note(f"{name} not consulted (offline): only answers saved by earlier runs are used.")
            return None
        if self.cache_dir is None:
            self.note(f"{name} not consulted: no cache folder is set for online sources.")
            return None
        if source in self.disabled:
            return None  # its note is already there
        cap = self.max_requests.get(source)
        if cap is not None and self.requests_made.get(source, 0) >= cap:
            self.note(f"{name}: request limit for one report ({cap}) reached; the rest will follow in later runs.")
            return cached[0] if cached else None

        hdrs = {"User-Agent": _user_agent(), "Accept": "application/json"}
        hdrs.update(headers or {})
        while True:
            self.requests_made[source] = self.requests_made.get(source, 0) + 1
            try:
                resp = self._throttled_get(url, hdrs)
            except requests.exceptions.Timeout:
                self._fail(source, f"{name} did not answer in time; skipped for the rest of this report.", True)
                return cached[0] if cached else None
            except requests.RequestException as exc:
                text = f"{name} could not be reached ({type(exc).__name__}); skipped for the rest of this report."
                self._fail(source, text, True)
                return cached[0] if cached else None

            status = int(getattr(resp, "status_code", 0) or 0)
            if status == 429:
                if source not in self._waited_429:
                    self._waited_429.add(source)
                    self.note(
                        f"{name} was busy (HTTP 429); waited {self.rate_limit_wait:.0f} s once before asking again."
                    )
                    self._sleep(self.rate_limit_wait)
                    continue
                self._fail(
                    source, f"{name} is still rate limited (HTTP 429); skipped for the rest of this report.", True
                )
                return cached[0] if cached else None
            break

        content_type = str(getattr(resp, "headers", {}).get("Content-Type", "") or "")
        when = self._now()
        if status in (401, 403):
            text = f"{name} refused the request (HTTP {status}); skipped for this report."
            if source == "explorer":
                text = TOKEN_NOTE
            if source == "explorer" and self.lichess_token:
                text = (
                    f"The Lichess opening explorer refused the token (HTTP {status}): check LICHESS_TOKEN. "
                    "Skipped this time."
                )
            self._fail(source, text, True)
            return cached[0] if cached else None
        if status == 404:
            try:
                body = resp.json()
            except ValueError:
                body = None
            self._write_cache(source, url, 404, content_type, body, when)
            return Fetched(url=url, status=404, data=None, retrieved=when.date().isoformat())
        if status != 200:
            self._failures[source] = self._failures.get(source, 0) + 1
            self._fail(source, f"{name} was unavailable (HTTP {status}); some facts are left out.",
                       self._failures[source] >= MAX_FAILURES)
            return cached[0] if cached else None
        try:
            body = resp.json()
        except ValueError:
            self._failures[source] = self._failures.get(source, 0) + 1
            self._fail(source, f"{name} sent an unreadable answer; some facts are left out.",
                       self._failures[source] >= MAX_FAILURES)
            return cached[0] if cached else None
        self._write_cache(source, url, 200, content_type, body, when)
        return Fetched(url=url, status=200, data=body, retrieved=when.date().isoformat())
