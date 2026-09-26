"""Recorded HTTP answers for the public-source tests, and a guard that fails a test on any real network request.

Every ``*.json`` file here is an answer recorded from the real service on 2026-09-26 (``url``, ``status``,
``content_type``, ``recorded``, ``body``), copied unchanged from the recording run (two tablebase files were
renamed after what they contain: ``tablebase_kpk_drawn`` and ``tablebase_kpk_lost``). Answers that could not be
recorded (the opening explorer needs a token; a batched Wikibooks existence query; a cloud-eval 404) are built in
the tests themselves and say so.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Callable, Optional

import requests

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict[str, Any]:
    """One recorded answer by file stem: ``load("tablebase_krk")``."""
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status: int, body: Any, content_type: str = "application/json") -> None:
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, str):
            return json.loads(self._body)  # raises ValueError for HTML, like requests does
        return self._body


def response(doc: dict[str, Any]) -> FakeResponse:
    return FakeResponse(int(doc["status"]), doc["body"], doc.get("content_type") or "application/json")


class RecordedSession:
    """Answers GETs from recorded files (matched on the exact URL). Unknown URLs fail the test, unless
    ``fallback(url)`` returns a FakeResponse for them.

    ``queue`` maps a URL to several answers in order (e.g. a 429 and then a 200).
    """

    def __init__(self, *names: str, fallback: Optional[Callable[[str], Optional[FakeResponse]]] = None) -> None:
        self.by_url: dict[str, list[FakeResponse]] = {}
        for name in names:
            self.add(load(name))
        self.fallback = fallback
        self.calls: list[tuple[str, dict[str, str]]] = []

    def add(self, doc: dict[str, Any], times: int = 1) -> "RecordedSession":
        self.by_url.setdefault(doc["url"], []).extend([response(doc)] * times)
        return self

    def get(self, url: str, headers: Optional[dict[str, str]] = None, timeout: Any = None) -> FakeResponse:
        self.calls.append((url, dict(headers or {})))
        answers = self.by_url.get(url)
        if answers:
            return answers.pop(0) if len(answers) > 1 else answers[0]
        if self.fallback is not None:
            got = self.fallback(url)
            if got is not None:
                return got
        raise AssertionError(f"no recorded answer for {url}")


class TimeoutSession:
    """Every request times out (a simulated timeout: nothing was recorded)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, headers: Any = None, timeout: Any = None) -> Any:
        self.calls.append(url)
        raise requests.exceptions.ReadTimeout(f"simulated read timeout for {url}")


def block_network(monkeypatch: Any) -> list[str]:
    """Make every real connection attempt fail and record it; the caller asserts the list stays empty."""
    attempts: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(args[:2]))
        raise OSError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts
