"""Minimal client for the chess.com Published-Data API (PubAPI).

Docs: https://www.chess.com/news/view/published-data-api

Etiquette the API asks for (and that keeps you from being blocked):
* send a descriptive User-Agent with a way to contact you;
* make requests **serially** — parallel requests are what trigger HTTP 429;
* back off and retry on 429 / 5xx.

All data is public; no authentication is involved.
"""

from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Optional, Union
from urllib.parse import quote

import requests
import urllib3

from . import __version__

BASE_URL = "https://api.chess.com/pub"
DEFAULT_USER_AGENT = f"chess-insights/{__version__} (+https://github.com/TimBuckTwo-23/chess-insights)"
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
# "https://www.chess.com/member/Hikaru", "chess.com/member/hikaru/", ".../stats/live/blitz/Hikaru"
_PROFILE_URL_RE = re.compile(r"^(?:https?://)?(?:www\.)?chess\.com/(?:member|stats/[a-z]+/[a-z0-9]+)/([^/?#\s]+)/?(?:[?#].*)?$", re.I)
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
    gone: bool = False  # True on 410: chess.com says the data will never be available


def normalize_username(username: str) -> str:
    """chess.com URLs use the lowercase username (mixed case gets a 301); reject anything that isn't a handle.

    A pasted profile URL (``https://www.chess.com/member/Hikaru``) or ``@Hikaru`` is accepted too.
    """
    name = str(username or "").strip()
    m = _PROFILE_URL_RE.match(name)
    if m:
        name = m.group(1)
    name = name.lstrip("@")
    if not _USERNAME_RE.match(name):
        raise ValueError(
            f"{username!r} is not a chess.com username (letters, digits, '_' and '-' only, "
            "e.g. the part after chess.com/member/)"
        )
    return name.lower()


def header_safe(text: str) -> str:
    """``text`` as an HTTP header value that can always be sent.

    Python's HTTP client only sends Latin-1 and rejects control characters, so a contact such as
    ``Łukasz <l@example.pl>`` or one with a trailing newline (pasted from a file) would crash or fail
    every request. Accents are dropped (``é`` -> ``e``), other non-ASCII characters are
    percent-encoded (``Ł`` -> ``%C5%81``), control characters and runs of whitespace become one space.
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return "".join(ch if ch.isascii() else quote(ch, safe="") for ch in text)


def _is_connect_failure(exc: BaseException) -> bool:
    """True when no connection could be made at all (DNS failure, refused, connect timeout).

    Retrying those five times only makes a user without internet wait minutes for the error; a
    connection that breaks mid-response (reset, truncated body, read timeout) is worth retrying.
    """
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        reason = getattr(exc.args[0] if exc.args else None, "reason", None)
        return isinstance(reason, urllib3.exceptions.NewConnectionError)  # includes NameResolutionError
    return False


def parse_archive_url(url: str) -> tuple[int, int]:
    """'https://api.chess.com/pub/player/x/games/2024/03' -> (2024, 3)."""
    m = _ARCHIVE_RE.search(url)
    if not m:
        raise ValueError(f"not a monthly archive URL: {url!r}")
    return int(m.group(1)), int(m.group(2))


class ChessComClient:
    """Serial, polite, retrying PubAPI client.

    ``notify`` receives a one-line message before every retry wait, so a slow or busy chess.com
    never looks like a hang. ``session`` and ``sleep`` are injectable for tests.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        contact: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        max_retries: int = 5,
        connect_retries: int = 1,
        backoff_base: float = 1.0,
        max_retry_after: float = 60.0,
        max_total_wait: float = 180.0,
        min_interval: float = 0.25,
        timeout: Union[float, tuple[float, float]] = (10.0, 30.0),
        sleep: Callable[[float], None] = time.sleep,
        notify: Optional[Callable[[str], None]] = None,
        base_url: str = BASE_URL,
    ) -> None:
        contact = header_safe(contact or os.environ.get("CHESS_INSIGHTS_CONTACT") or "")
        ua = header_safe(user_agent or DEFAULT_USER_AGENT) or DEFAULT_USER_AGENT
        if contact:
            ua = f"{ua} contact: {contact}"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept": "application/json"})
        self.max_retries = max_retries
        self.connect_retries = connect_retries  # retries when no connection can be made at all
        self.backoff_base = backoff_base
        self.max_retry_after = max_retry_after
        self.max_total_wait = max_total_wait  # give up rather than wait longer than this in one request
        self.min_interval = min_interval
        self.timeout = timeout  # (connect, read) seconds
        self._sleep = sleep
        self._notify = notify
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
        self,
        path_or_url: str,
        *,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        not_found_ok: bool = False,
    ) -> tuple[Optional[Any], requests.Response]:
        """GET JSON with retries. Returns (data, response); data is None on 304 or tolerated 404/410.

        ``etag`` / ``last_modified`` make the request conditional (chess.com's ETags are weak,
        ``W/"..."``, and are sent back unchanged). Retries 429, 5xx, broken connections and
        truncated/invalid bodies with backoff (honouring ``Retry-After``), but gives up after
        ``connect_retries`` retries when no connection can be made at all (no internet, DNS
        failure, blocked host), and never waits more than ``max_total_wait`` seconds in total
        or sleeps after the last attempt. Each wait is announced through ``notify``.
        """
        url = self._url(path_or_url)
        headers = {"If-None-Match": etag} if etag else {"If-Modified-Since": last_modified} if last_modified else {}
        last_error: Optional[BaseException] = None
        attempts = self.max_retries + 1
        made = connect_failures = 0
        waited = 0.0
        for attempt in range(attempts):
            self._throttle()
            delay = self.backoff_base * 2**attempt  # wait before the next attempt, unless the server says otherwise
            made += 1
            try:
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                self.requests_made += 1
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError) as exc:
                raise ChessComError(
                    f"could not connect securely to api.chess.com ({type(exc).__name__}). A proxy, VPN or "
                    "antivirus that inspects HTTPS traffic is the usual cause: its certificate must be trusted "
                    "(set REQUESTS_CA_BUNDLE to your company's CA file) or the proxy configured (HTTPS_PROXY)."
                ) from exc
            except (
                requests.exceptions.InvalidHeader,
                requests.exceptions.InvalidURL,
                requests.exceptions.InvalidSchema,
                requests.exceptions.MissingSchema,
            ) as exc:  # a bad request is bad every time: retrying only makes the user wait
                raise ChessComError(f"could not send the request to {url} ({type(exc).__name__}: {exc})") from exc
            except requests.RequestException as exc:  # connection reset, DNS, timeout, truncated body
                last_error = exc
                if _is_connect_failure(exc):
                    connect_failures += 1
                    if connect_failures > self.connect_retries:
                        break
                    reason = f"could not connect to api.chess.com ({type(exc).__name__})"
                else:
                    reason = f"connection problem ({type(exc).__name__})"
            else:
                if resp.status_code == 304:
                    return None, resp
                if resp.status_code == 200:
                    try:
                        return resp.json(), resp
                    except ValueError as exc:  # cut-off or non-JSON body (captive portal, CDN hiccup)
                        last_error = ChessComError(f"invalid JSON from {url}")
                        last_error.__cause__ = exc
                        reason = "incomplete answer"
                elif resp.status_code in (404, 410):
                    if not_found_ok:
                        return None, resp
                    raise PlayerNotFound(f"{url} -> HTTP {resp.status_code}")
                elif resp.status_code == 429 or resp.status_code >= 500:
                    last_error = RateLimited(f"{url} -> HTTP {resp.status_code}")
                    delay = self._retry_after(resp, attempt)
                    reason = "too many requests (HTTP 429)" if resp.status_code == 429 else f"server error (HTTP {resp.status_code})"
                elif resp.status_code == 403:
                    raise ChessComError(
                        f"{url} -> HTTP 403. chess.com's CDN rejects requests without a descriptive "
                        "User-Agent (pass --contact you@example.com), and some networks block api.chess.com."
                    )
                else:
                    raise ChessComError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")

            if attempt == attempts - 1 or waited + delay > self.max_total_wait:
                break  # never sleep after the last attempt, nor past the total wait budget
            if self._notify:
                self._notify(f"chess.com: {reason}; retrying in {delay:.0f}s (retry {attempt + 1})")
            self._sleep(delay)
            waited += delay

        if isinstance(last_error, ChessComError):
            raise last_error
        raise ChessComError(
            f"could not reach api.chess.com after {made} attempt{'s' if made != 1 else ''} "
            f"({type(last_error).__name__}: {last_error}). "
            "Check your internet connection; some school and company networks block api.chess.com."
        ) from last_error

    def _retry_after(self, resp: requests.Response, attempt: int) -> float:
        """Seconds to wait: ``Retry-After`` (seconds or an HTTP date), capped; else exponential backoff."""
        header = (resp.headers.get("Retry-After") or "").strip()
        wait: Optional[float] = None
        if header:
            try:
                wait = float(header)
            except ValueError:
                try:
                    when = parsedate_to_datetime(header)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    wait = (when - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, IndexError, OverflowError):
                    wait = None
        if wait is None or math.isnan(wait):
            wait = self.backoff_base * 2**attempt
        return min(max(0.0, wait), self.max_retry_after)

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
        urls = [u for u in ((data or {}).get("archives") or []) if isinstance(u, str) and _ARCHIVE_RE.search(u)]
        return sorted(set(urls), key=parse_archive_url)

    def month(
        self,
        username: str,
        year: int,
        month: int,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> MonthResult:
        """Games of one month. A 404 yields an empty result with ``missing=True`` (retry later).

        On a 304 the response may omit ``ETag`` / ``Last-Modified``; the values sent are kept.
        """
        path = f"player/{normalize_username(username)}/games/{year:04d}/{month:02d}"
        data, resp = self.request(path, etag=etag, last_modified=last_modified, not_found_ok=True)
        not_modified = resp.status_code == 304
        games = (data or {}).get("games", []) if isinstance(data, dict) else []
        return MonthResult(
            year=year,
            month=month,
            games=list(games),
            etag=resp.headers.get("ETag") or (etag if not_modified else None),
            last_modified=resp.headers.get("Last-Modified") or (last_modified if not_modified else None),
            not_modified=not_modified,
            missing=resp.status_code == 404,
            gone=resp.status_code == 410,
        )
