"""GitHub-flavoured Markdown report, readable in a repo or on GitHub mobile.

Numbers are formatted exactly like the HTML report (shared helpers in
``report/html.py``). Every string that comes from data is escaped so it cannot
inject HTML, links or table structure; only http(s) URLs become links.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from ..models import VALUE_FORMATS, Chart, Insight, Kpi, ModuleResult, Report, StudyItem, Table
from .html import (
    GLYPH_MEANINGS,
    KIND_LABELS,
    METHOD_NOTES,
    MISSING,
    category_label,
    chart_data,
    chart_table,
    confidence_label,
    date_range_text,
    format_generated,
    format_value,
    games_text,
    glyph_for,
    infer_format,
    insight_kind,
    is_missing,
    is_numeric_format,
    plan_titles,
    report_title,
    safe_url,
    text_or_empty,
    url_link_text,
    valid_urls,
)

_SPECIAL = re.compile(r"([\\`*\[\]<>|~])")
_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")  # intraword _ is literal in GFM
_ENTITY = re.compile(r"&(?=#?[A-Za-z0-9]+;)")
_BLOCK_START = re.compile(r"^(#{1,6}\s|[-+=]\s|>|\d+[.)]\s)")


def md_text(value: Any) -> str:
    """Escape one line of inline text: collapses newlines, neutralises markup, HTML and pipes."""
    text = " ".join(text_or_empty(value).split())
    text = _SPECIAL.sub(r"\\\1", text)
    text = _UNDERSCORE.sub(r"\\_", text)
    text = _ENTITY.sub("&amp;", text)
    if _BLOCK_START.match(text):  # would start a heading, list or quote at the start of a line
        text = "\\" + text
    return text


def md_link(url: Any, label: str) -> str:
    """``[label](url)`` for http(s) URLs; anything else is shown as escaped text, never linked."""
    safe = safe_url(url)
    if not safe:
        return md_text(url)
    target = safe.replace("(", "%28").replace(")", "%29").replace("<", "%3C").replace(">", "%3E")
    return f"[{md_text(label)}]({target})"


def _games_line(urls: Iterable[Any], label: str) -> str:
    links = valid_urls(urls)
    if not links:
        return ""
    return f"{label}: " + " · ".join(md_link(u, f"Game {i}") for i, u in enumerate(links, 1))


def _cell(value: Any, fmt: Optional[str]) -> str:
    if not is_missing(value) and (fmt == "url" or (fmt not in VALUE_FORMATS and infer_format(value) == "url")):
        safe = safe_url(value)
        return md_link(safe, url_link_text(safe) + " ↗") if safe else md_text(value)
    return md_text(format_value(value, fmt)) or MISSING


def md_table(table: Table) -> list[str]:
    """Pipe table lines (numeric columns right-aligned); an italic note when there are no rows."""
    columns = [text_or_empty(c) for c in (table.columns or [])]
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in (table.rows or [])]
    ncol = max([len(columns)] + [len(r) for r in rows])
    if ncol == 0 or not rows:
        return ["_No rows yet._"]
    columns += [""] * (ncol - len(columns))
    rows = [r + [None] * (ncol - len(r)) for r in rows]
    fmts = [f if f else None for f in (table.formats or [])]
    fmts += [None] * (ncol - len(fmts))
    numeric = [is_numeric_format(fmts[c], (r[c] for r in rows)) for c in range(ncol)]
    lines = [
        "| " + " | ".join(md_text(c) for c in columns) + " |",
        "|" + "|".join("---:" if numeric[c] else "---" for c in range(ncol)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_cell(v, fmts[c]) for c, v in enumerate(r)) + " |")
    return lines


def _insight_line(ins: Insight, *, show_study: bool, with_kind: bool) -> list[str]:
    glyph = glyph_for(ins)
    label = f"{KIND_LABELS[insight_kind(ins)]} · {category_label(ins.category)}: " if with_kind else ""
    head = f"- `{glyph}` **{md_text(label)}{md_text(ins.title)}**"
    if text_or_empty(ins.detail):
        head += f" — {md_text(ins.detail)}"
    head += f" _({confidence_label(ins.confidence)})_"
    games = _games_line(ins.example_games, "review")
    if games:
        head += f" · {games}"
    lines = [head]
    if show_study:
        lines += [f"  - {md_text(s)}" for s in (ins.study or []) if text_or_empty(s)]
    return lines


def _study_plan(items: Sequence[StudyItem]) -> list[str]:
    lines = ["## Study plan", "", "Most important first. Work through it from the top.", ""]
    if not items:
        return lines + ["_Nothing to study yet: no weakness stood out clearly enough in these games._", ""]
    for i, item in enumerate(items, 1):
        marker = f"{i}. "
        pad = " " * len(marker)
        title = f"{marker}**{md_text(item.title)}**"
        if text_or_empty(item.category):
            title += f" — _{md_text(item.category)}_"
        lines += [title, ""]
        if text_or_empty(item.why):
            lines += [pad + md_text(item.why), ""]
        actions = [a for a in (item.actions or []) if text_or_empty(a)]
        if actions:
            lines += [f"{pad}- [ ] {md_text(a)}" for a in actions] + [""]
        games = _games_line(item.games, "Review these games")
        if games:
            lines += [pad + games, ""]
    return lines


def _kpi_line(kpi: Kpi) -> str:
    fmt = kpi.format if kpi.format in VALUE_FORMATS else None
    value = _cell(kpi.value, fmt)
    line = f"- {md_text(kpi.label)}: **{value}**" if value != MISSING else f"- {md_text(kpi.label)}: {MISSING}"
    if text_or_empty(kpi.hint):
        line += f" — {md_text(kpi.hint)}"
    return line


def _chart(chart: Chart) -> list[str]:
    title = text_or_empty(chart.title) or "Chart"
    lines = [f"#### {md_text(title)}", ""]
    data = chart_data(chart)
    if not data.has_values:
        lines += ["_No data to chart yet._", ""]
    else:
        lines += md_table(chart_table(chart)) + [""]
    notes = []
    if data.reference is not None:
        notes.append(f"Reference line: {format_value(data.reference, data.fmt)}.")
    if text_or_empty(chart.note):
        notes.append(md_text(chart.note))
    if notes:
        lines += ["_" + " ".join(notes) + "_", ""]
    return lines


def _module(module: ModuleResult) -> list[str]:
    title = text_or_empty(module.title) or text_or_empty(module.key) or "Section"
    lines = [f"## {md_text(title)}", ""]
    if text_or_empty(module.summary):
        lines += [md_text(module.summary), ""]
    if module.kpis:
        lines += [_kpi_line(k) for k in module.kpis] + [""]
    for chart in module.charts or []:
        lines += _chart(chart)
    for table in module.tables or []:
        if text_or_empty(table.title):
            lines += [f"#### {md_text(table.title)}", ""]
        lines += md_table(table) + [""]
        if text_or_empty(table.note):
            lines += [f"_{md_text(table.note)}_", ""]
    if module.insights:
        lines += ["### Findings", ""]
        ranked = sorted(module.insights, key=lambda i: -(i.priority if i.priority == i.priority else 0.0))
        for ins in ranked:
            lines += _insight_line(ins, show_study=True, with_kind=True)
        lines.append("")
    return lines


def render_markdown(report: Report) -> str:
    """The whole report as GitHub-flavoured Markdown."""
    lines = [f"# {md_text(report_title(report))}", ""]
    meta = [f"**{games_text(report.n_games)}**"]
    span = date_range_text(report.date_from, report.date_to)
    if span:
        meta.append(span)
    if text_or_empty(report.filters):
        meta.append(md_text(report.filters))
    lines.append(" · ".join(meta))
    if text_or_empty(report.engine_note):
        lines += ["", f"_{md_text(report.engine_note)}_"]
    headline = text_or_empty(report.headline) or (
        "No games matched the filters, so there is nothing to analyse yet."
        if not report.n_games
        else "No clear patterns yet. Play more games, or widen the filters, for reliable insights."
    )
    lines += ["", f"> {md_text(headline)}", ""]

    lines += _study_plan(report.study_plan or [])

    covered = plan_titles(report)
    for heading, insights, empty in (
        ("Strengths", report.strengths or [], "No clear strengths yet."),
        ("Weaknesses", report.weaknesses or [], "No clear weaknesses yet."),
    ):
        lines += [f"## {heading}", ""]
        if not insights:
            lines += [f"_{empty} More games make patterns reliable enough to report._", ""]
            continue
        for ins in insights:
            lines += _insight_line(ins, show_study=text_or_empty(ins.title) not in covered, with_kind=False)
        lines.append("")
    key = " · ".join(f"`{g}` {m.lower()}" for g, m in GLYPH_MEANINGS.items())
    lines += [f"_Marks: {key}._", ""]

    for module in report.modules or []:
        lines += ["---", ""] + _module(module)

    lines += ["---", "", "### How this report works", ""]
    lines += [f"- {md_text(note)}" for note in METHOD_NOTES]
    generated = format_generated(report.generated_at)
    if generated:
        lines += ["", f"_Generated {generated} by chess-insights._"]
    return "\n".join(lines).rstrip() + "\n"
