"""Report writers: self-contained HTML, GitHub Markdown and JSON.

Renderers only ever see a :class:`~chess_insights.models.Report` and are generic
over module contents (KPIs, charts, tables, insights, boards). Boards are drawn from
FEN + annotations by :mod:`.boards` (inline SVG, one piece sprite per page); a report
with ``format_reports`` gets one view per format, and ``Report.coaching`` its own
sections ("Why these moves go wrong", "Practice").
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Callable, Sequence

from ..models import Report
from .html import format_value, render_html
from .json_export import to_dict, to_json
from .markdown import render_markdown

__all__ = ["write_report", "report_stem", "render_html", "render_markdown", "to_dict", "to_json", "format_value", "FORMATS"]

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


def report_stem(report: Report, out: Path | str) -> Path:
    """The path stem the report files get for ``out``.

    ``out`` is normally a stem; a report-type suffix (``report.html``) is stripped so every
    format lands next to it. A folder (an existing directory, or a path ending in a separator)
    gets the files inside it, named after the player: ``reports/`` -> ``reports/<player>``.
    """
    from ..fetch import safe_file_stem

    text = str(out)
    stem = Path(out)
    if text.endswith(tuple(sep for sep in (os.sep, os.altsep) if sep)) or stem.is_dir():
        name = str(report.username or "").strip().lower() or "chess-insights"
        return stem / safe_file_stem(name)
    if stem.suffix.lower() in _REPORT_SUFFIXES:
        stem = stem.with_suffix("")
    return stem


def _clean_text(text: str) -> bytes:
    """UTF-8 bytes of ``text``, with any lone surrogate (from broken input) as U+FFFD instead of a crash."""
    if not text.endswith("\n"):
        text += "\n"
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace").encode("utf-8")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file in the same folder, so a failure never leaves a truncated report.
    (Not ``tempfile.mkstemp``: its owner-only permissions would carry over to the report.)"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_report(report: Report, out: Path | str, formats: Sequence[str] = ("html", "md", "json")) -> list[Path]:
    """Write ``<out>.html`` / ``<out>.md`` / ``<out>.json`` and return the paths written.

    See :func:`report_stem` for how ``out`` becomes the file stem; parent directories are
    created. Every format is rendered before anything is written, and each file is replaced
    atomically, so an error leaves earlier reports intact. Unknown format names raise
    ``ValueError`` before anything is written.
    """
    wanted = _normalise_formats(formats)
    stem = report_stem(report, out)
    rendered = [(FORMATS[key][0], FORMATS[key][1](report)) for key in wanted]
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext, text in rendered:
        path = stem.parent / f"{stem.name}.{ext}"
        _write_atomic(path, _clean_text(text))
        paths.append(path)
    return paths
