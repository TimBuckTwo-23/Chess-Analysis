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
    lichess_analysis_url,
    DEMO_NOTE,
    GLYPH_MEANINGS,
    KIND_LABELS,
    MAX_GAME_LINKS,
    METHOD_NOTES,
    MISSING,
    NO_GAME_LINKS,
    game_link_text,
    headline_lines,
    is_game_url,
    plan_numbers,
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
    report_title,
    safe_url,
    text_or_empty,
    url_link_text,
    valid_urls,
)

_SPECIAL = re.compile(r"([\\`*\[\]<>|~$])")  # $ starts GitHub maths
_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")  # intraword _ is literal in GFM
_ENTITY = re.compile(r"&(?=#?[A-Za-z0-9]+;)")
_HASH = re.compile(r"(?:(?<=\s)|^)#")  # headings, and the optional closing #s of a heading
# a list bullet ("- x", "+ x"), or a line of only - / = (thematic break "---", setext underline "===")
_LEADING_MARK = re.compile(r"^(?:[-+](?=\s|$)|[-=](?=[-=\s]*$))")
_ORDERED = re.compile(r"^(\d{1,9})([.)])(?=\s|$)")  # "1. e4": escape the dot, not the digit
# A web address where GitHub would autolink one: at the start, after a space or after ( * _ ~.
# Escaping it character by character would leave the backslashes in the link, so it is written
# as an explicit link instead.
_BARE_URL = re.compile(r"(?:(?<=[\s(*_~])|^)(?:(?i:https?://)|www\.(?=[A-Za-z0-9]))[^\s<>]+")
_URL_TRAILING = ".,:;!?*_~'\""  # punctuation that ends a sentence rather than the address (as GitHub)
_FTP = re.compile(r"(?<![A-Za-z0-9])((?i:ftp)):(?=//)")  # GitHub autolinks ftp:// too; only http(s) may link


def _escape(text: str) -> str:
    text = _SPECIAL.sub(r"\\\1", text)
    text = _FTP.sub(r"\1\\:", text)
    text = _UNDERSCORE.sub(r"\\_", text)
    text = _ENTITY.sub("&amp;", text)
    text = _HASH.sub(r"\\#", text)
    text = _LEADING_MARK.sub(lambda m: "\\" + m.group(0), text)
    return _ORDERED.sub(r"\1\\\2", text)


def _url_end(run: str) -> int:
    """Length of the address in ``run``: trailing punctuation and unbalanced ")" are not part of it."""
    end, opens, closes = len(run), run.count("("), run.count(")")
    while end:
        ch = run[end - 1]
        if ch == ")" and opens < closes:
            closes -= 1
        elif ch not in _URL_TRAILING:
            break
        end -= 1
    return end


def md_text(value: Any) -> str:
    """Escape one line of inline text: collapses newlines, neutralises markup, HTML and pipes.
    Bare http(s) / www addresses become plain links whose text and target are the address itself."""
    text = " ".join(text_or_empty(value).split())
    out, pos = [], 0
    for m in _BARE_URL.finditer(text):
        run = m.group(0)[: _url_end(m.group(0))]
        target = run if run.lower().startswith("http") else "http://" + run  # www. links are http, as on GitHub
        if not run or not safe_url(target):
            continue
        # escaping a piece on its own may add a backslash more than needed, never one fewer
        out += [_escape(text[pos : m.start()]), md_link(target, run)]
        pos = m.start() + len(run)
    out.append(_escape(text[pos:]))
    return "".join(out)


def md_code(value: Any) -> str:
    """Inline code span that no backtick in the text can close early (backslashes are literal in code)."""
    text = " ".join(text_or_empty(value).split())
    if not text:
        return ""
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def md_link(url: Any, label: str) -> str:
    """``[label](url)`` for http(s) URLs; anything else is shown as escaped text, never linked."""
    safe = safe_url(url)
    if not safe:
        return md_text(url)
    if NO_GAME_LINKS.get() and is_game_url(safe):  # a demo report's games are made up: no links to real ones
        return md_text(label)
    target = safe
    # "|" would split a table cell even inside a link; the rest could end the destination early
    for char, code in (("\\", "%5C"), ("(", "%28"), (")", "%29"), ("<", "%3C"), (">", "%3E"), ("|", "%7C"), ("`", "%60")):
        target = target.replace(char, code)
    target = _ENTITY.sub("&amp;", target)  # "&amp;" in a destination is decoded: keep the address as it is
    return f"[{_escape(' '.join(text_or_empty(label).split()))}]({target})"


def _games_line(urls: Iterable[Any], label: str, labels: Optional[dict[str, str]] = None) -> str:
    links = valid_urls(urls, MAX_GAME_LINKS)
    if not links:
        return ""
    return f"{label}: " + " · ".join(md_link(u, game_link_text(u, labels, i)) for i, u in enumerate(links, 1))


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


def _insight_line(
    ins: Insight,
    *,
    show_study: bool,
    with_kind: bool,
    labels: Optional[dict[str, str]] = None,
    plan_no: Optional[int] = None,
) -> list[str]:
    glyph = glyph_for(ins)
    label = f"{md_text(f'{KIND_LABELS[insight_kind(ins)]} · {category_label(ins.category)}')}: " if with_kind else ""
    head = f"- `{glyph}` **{label}{md_text(ins.title)}**"
    if text_or_empty(ins.detail):
        head += f" — {md_text(ins.detail)}"
    head += f" _({confidence_label(ins.confidence)})_"
    if plan_no:
        head += f" · what to do: study plan item {plan_no}"
    else:
        games = _games_line(ins.example_games, "review", labels)
        if games:
            head += f" · {games}"
    lines = [head]
    if show_study and not plan_no:
        lines += [f"  - {md_text(s)}" for s in (ins.study or []) if text_or_empty(s)]
    return lines


def _study_plan(items: Sequence[StudyItem], labels: Optional[dict[str, str]] = None) -> list[str]:
    lines = [
        "## Study plan",
        "",
        "Easiest changes first. Related findings share one item, with at most three actions.",
        "",
    ]
    if not items:
        return lines + ["_Nothing to study yet: no weakness stood out clearly enough in these games._", ""]
    for i, item in enumerate(items, 1):
        marker = f"{i}. "
        pad = " " * len(marker)
        title = f"{marker}**{md_text(item.title)}**"
        if text_or_empty(item.category):
            title += f" — _{md_text(item.category)}_"
        lines += [title, ""]
        findings = [t for t in (getattr(item, "findings", None) or []) if text_or_empty(t)]
        if len(findings) > 1:
            lines += [pad + "Also covers: " + " · ".join(md_text(t) for t in findings[1:]), ""]
        if text_or_empty(item.why):
            lines += [pad + md_text(item.why), ""]
        actions = [a for a in (item.actions or []) if text_or_empty(a)]
        if actions:
            lines += [f"{pad}- [ ] {md_text(a)}" for a in actions] + [""]
        target = text_or_empty(getattr(item, "target", ""))
        if target:
            lines += [f"{pad}_Target for your next report:_ {md_text(target)}", ""]
        games = _games_line(item.games, "Review", labels)
        if games:
            lines += [pad + games, ""]
    return lines


def _glance(report: Report, sections: dict[str, str], plan: dict[str, int]) -> list[str]:
    lines = ["## At a glance", ""]
    for heading, insights, empty in (
        ("Weaknesses", report.weaknesses or [], "No clear weaknesses yet."),
        ("Strengths", report.strengths or [], "No clear strengths yet."),
    ):
        lines += [f"**{heading}** ({len(insights)})" if insights else f"**{heading}**", ""]
        if not insights:
            lines += [f"_{empty} More games make patterns reliable enough to report._", ""]
            continue
        for ins in insights:
            iid = text_or_empty(ins.id)
            where = [sections.get(iid, "")] + ([f"plan item {plan[iid]}"] if iid in plan else [])
            where_text = " · ".join(w for w in where if w)
            lines.append(f"- `{glyph_for(ins)}` {md_text(ins.title)}" + (f" — _{md_text(where_text)}_" if where_text else ""))
        lines.append("")
    key = " · ".join(f"`{g}` {m.lower()}" for g, m in GLYPH_MEANINGS.items())
    lines += [f"_Marks: {key}. Most important first; each one is explained in its section._", ""]
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
    full = chart.table if isinstance(getattr(chart, "table", None), Table) else None
    if not data.has_values and full is None:
        lines += ["_No data to chart yet._", ""]
    else:  # the chart's own full table when it has one (more columns than the series)
        lines += md_table(full or chart_table(chart)) + [""]
        if full is not None and text_or_empty(full.note):
            lines += [f"_{md_text(full.note)}_", ""]
    notes = []
    if data.reference is not None:
        notes.append(f"Reference line: {format_value(data.reference, data.fmt)}.")
    if text_or_empty(chart.note):
        notes.append(md_text(chart.note))
    if notes:
        lines += ["_" + " ".join(notes) + "_", ""]
    return lines


def _module(
    module: ModuleResult, labels: Optional[dict[str, str]] = None, plan: Optional[dict[str, int]] = None
) -> list[str]:
    title = text_or_empty(module.title) or text_or_empty(module.key) or "Section"
    lines = [f"## {md_text(title)}", ""]
    if text_or_empty(module.summary):
        lines += [md_text(module.summary), ""]
    if module.kpis:
        lines += [_kpi_line(k) for k in module.kpis] + [""]
    for chart in module.charts or []:
        lines += _chart(chart)
    for d in getattr(module, "diagrams", None) or []:
        links = [
            md_link(u, t) for u, t in ((d.link, (labels or {}).get(d.link) or "game"), (lichess_analysis_url(d.fen), "analyse on Lichess")) if u
        ]
        lines += [f"- **{md_text(d.title)}**: {md_text(d.caption)} {md_code(d.fen)} " + " · ".join(links), ""]
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
            lines += _insight_line(
                ins, show_study=True, with_kind=True, labels=labels, plan_no=(plan or {}).get(text_or_empty(ins.id))
            )
        lines.append("")
    return lines


def render_markdown(report: Report) -> str:
    """The whole report as GitHub-flavoured Markdown."""
    token = NO_GAME_LINKS.set(bool(getattr(report, "demo", False)))
    try:
        return _render(report)
    finally:
        NO_GAME_LINKS.reset(token)


def _render(report: Report) -> str:
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
    if getattr(report, "demo", False):
        lines += ["", f"**{md_text(DEMO_NOTE)}**"]
    lines.append("")
    for line in headline_lines(report):
        lines += [f"> {md_text(line)}", ">"]
    lines[-1] = ""

    labels = dict(getattr(report, "game_labels", None) or {})
    plan = plan_numbers(report)
    sections: dict[str, str] = {}
    for module in report.modules or []:
        for ins in module.insights or []:
            sections.setdefault(text_or_empty(ins.id), text_or_empty(module.title) or text_or_empty(module.key))
    lines += _glance(report, sections, plan)
    lines += _study_plan(report.study_plan or [], labels)

    for module in report.modules or []:
        lines += ["---", ""] + _module(module, labels, plan)

    lines += ["---", "", "### How this report works", ""]
    lines += [f"- {md_text(note)}" for note in METHOD_NOTES]
    generated = format_generated(report.generated_at)
    if generated:
        lines += ["", f"_Generated {generated} by chess-insights._"]
    return "\n".join(lines).rstrip() + "\n"
