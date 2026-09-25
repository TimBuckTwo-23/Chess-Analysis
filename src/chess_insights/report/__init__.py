"""Report writers: self-contained HTML, GitHub Markdown and JSON.

Renderers only ever see a :class:`~chess_insights.models.Report` and are generic
over module contents (KPIs, charts, tables, insights).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from ..models import Report
from .html import format_value, render_html
from .json_export import to_dict, to_json
from .markdown import render_markdown

__all__ = ["write_report", "render_html", "render_markdown", "to_dict", "to_json", "format_value", "FORMATS"]

# format name -> (file extension, renderer)
FORMATS: dict[str, tuple[str, Callable[[Report], str]]] = {
    "html": ("html", render_html),
    "md": ("md", render_markdown),
    "json": ("json", to_json),
}
_ALIASES = {"htm": "html", "markdown": "md"}
_REPORT_SUFFIXES = {".html", ".htm", ".md", ".markdown", ".json"}


def _normalise_formats(formats: Sequence[str]) -> list[str]:
    wanted: list[str] = []
    for fmt in formats:
        key = str(fmt).strip().lower().lstrip(".")
        key = _ALIASES.get(key, key)
        if key not in FORMATS:
            raise ValueError(f"unknown report format {fmt!r}; choose from {', '.join(FORMATS)}")
        if key not in wanted:
            wanted.append(key)
    return wanted


def write_report(report: Report, out: Path | str, formats: Sequence[str] = ("html", "md", "json")) -> list[Path]:
    """Write ``<out>.html`` / ``<out>.md`` / ``<out>.json`` and return the paths written.

    ``out`` is a path stem; a report-type suffix (``report.html``) is stripped so
    every format lands next to it. Parent directories are created. Unknown format
    names raise ``ValueError`` before anything is written.
    """
    wanted = _normalise_formats(formats)
    stem = Path(out)
    if stem.suffix.lower() in _REPORT_SUFFIXES:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in wanted:
        ext, render = FORMATS[key]
        path = stem.parent / f"{stem.name}.{ext}"
        text = render(report)
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths
