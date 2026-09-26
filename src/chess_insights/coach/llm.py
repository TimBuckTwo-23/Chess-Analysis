"""Optional LLM coach: rewrite explanations and draft a weekly plan from a verified facts packet (C4). Owner: llm."""

from __future__ import annotations

from ..models import Report
from .config import CoachConfig


def annotate(report: Report, cfg: CoachConfig) -> None:
    """Replace template texts that pass the verifier; fill ``coaching.weekly_plan`` and ``coaching.llm``."""
    return None
