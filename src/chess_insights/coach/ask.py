"""`chess-insights ask USERNAME "question"`: answers from the last report's JSON, through the verifier (C4).
Owner: llm."""

from __future__ import annotations

from typing import Any

from .config import CoachConfig


def answer(question: str, report_json: dict[str, Any], cfg: CoachConfig) -> str:
    """A grounded answer to ``question`` (moves and numbers checked against the report), or an explanation of
    why none can be given (no API key, the llm extra not installed, nothing relevant in the report)."""
    raise NotImplementedError
