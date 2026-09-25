"""Minimal client for the chess.com Published-Data API (PubAPI).

Docs: https://www.chess.com/news/view/published-data-api

Etiquette the API asks for (and that keeps you from being blocked):
* send a descriptive User-Agent with a way to contact you;
* make requests **serially** — parallel requests are what trigger HTTP 429;
* back off and retry on 429 / 5xx.

All data is public; no authentication is involved.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

from . import __version__

BASE_URL = "https://api.chess.com/pub"
DEFAULT_USER_AGENT = f"chess-insights/{__version__} (+https://github.com/TimBuckTwo-23/chess-insights)"
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
_ARCHIVE_RE = re.compile(r"/games/(\d{4})/(\d{2})/?$")


class ChessComError(RuntimeError):
    """Any failure talking to chess.com."""


class PlayerNotFound(ChessComError):
    """The username does not exist or the account is closed (HTTP 404 / 410)."""


class RateLimited(ChessComError):
    """Still rate limited (HTTP 429) after all retries."""


@dataclass
class MonthResult:
    year: int
    month: int
    games: list[dict[str, Any]]
    etag: Optional[str]
    last_modified: Optional[str]
    not_modified: bool = False  # True when the server answered 304 to a conditional request
    missing: bool = False  # True on 404: chess.com sometimes 404s a listed month transiently


def normalize_username(username: str) -> str:
    """chess.com URLs use the lowercase username; reject anything that isn't a plain handle."""
    name = (username or "").strip()
    if not _USERNAME_RE.match(name):
        raise ValueError(f"not a valid chess.com username: {username!r}")
    return name.lower()


def parse_archive_url(url: str) -> tuple[int, int]:
    """'https://api.chess.com/pub/player/x/games/2024/03' -> (2024, 3)."""
    m = _ARCHIVE_RE.search(url)
    if not m:
        raise ValueError(f"not a monthly archive URL: {url!r}")
    return int(m.group(1)), int(m.group(2))


class ChessComClient:
    """Serial, polite, retrying PubAPI client.

    ``session`` and ``sleep`` are injectable for tests.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        contact: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        min_interval: float = 0.25,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = BASE_URL,
    ) -> None:
        contact = contact or os.environ.get("CHESS_INSIGHTS_CONTACT")
        ua = user_agent or DEFAULT_USER_AGENT
        if contact:
            ua = f"{ua} contact: {contact}"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept": "application/json"})
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_interval = min_interval
        self.timeout = timeout
        self._sleep = sleep
        self._last_request = 0.0
        self.base_url = base_url.rstrip("/")
        self.requests_made = 0

    # ------------------------------------------------------------------ core
    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http"):
            return path_or_url
        return f"{self.base_url}/{path_or_url.lstrip('/')}"

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def request(
        self, path_or_url: str, *, etag: Optional[str] = None, not_found_ok: bool = False
    ) -> tuple[Optional[Any], requests.Response]:
        """GET JSON with retries. Returns (data, response); data is None on 304 or tolerated 404."""
        url = self._url(path_or_url)
        headers = {"If-None-Match": etag} if etag else {}
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                self.requests_made += 1
            except requests.RequestException as exc:  # connection reset, DNS, timeout
                last_error = exc
                self._sleep(self.backoff_base * 2**attempt)
                continue

            if resp.status_code == 304:
                return None, resp
            if resp.status_code == 200:
                try:
                    return resp.json(), resp
                except ValueError as exc:
                    raise ChessComError(f"invalid JSON from {url}") from exc
            if resp.status_code in (404, 410):
                if not_found_ok:
                    return None, resp
                raise PlayerNotFound(f"{url} -> HTTP {resp.status_code}")
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RateLimited(f"{url} -> HTTP {resp.status_code}")
                self._sleep(self._retry_after(resp, attempt))
                continue
            if resp.status_code == 403:
                raise ChessComError(
                    f"{url} -> HTTP 403. chess.com's CDN rejects requests without a descriptive "
                    "User-Agent (pass --contact you@example.com), and some networks block api.chess.com."
                )
            raise ChessComError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")

        if isinstance(last_error, ChessComError):
            raise last_error
        raise ChessComError(f"{url} failed after {self.max_retries + 1} attempts: {last_error}")

    def _retry_after(self, resp: requests.Response, attempt: int) -> float:
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
        return self.backoff_base * 2**attempt

    # ------------------------------------------------------------ endpoints
    def profile(self, username: str) -> dict[str, Any]:
        data, _ = self.request(f"player/{normalize_username(username)}")
        return data or {}

    def stats(self, username: str) -> dict[str, Any]:
        data, _ = self.request(f"player/{normalize_username(username)}/stats")
        return data or {}

    def archives(self, username: str) -> list[str]:
        """Monthly archive URLs, oldest first. Empty list for a player with no games."""
        data, _ = self.request(f"player/{normalize_username(username)}/games/archives")
        urls = list((data or {}).get("archives", []))
        return sorted(urls, key=parse_archive_url)

    def month(self, username: str, year: int, month: int, etag: Optional[str] = None) -> MonthResult:
        """Games of one month. A 404 yields an empty result with ``missing=True`` (retry later)."""
        path = f"player/{normalize_username(username)}/games/{year:04d}/{month:02d}"
        data, resp = self.request(path, etag=etag, not_found_ok=True)
        return MonthResult(
            year=year,
            month=month,
            games=list((data or {}).get("games", [])),
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
            not_modified=resp.status_code == 304,
            missing=resp.status_code in (404, 410),
        )
