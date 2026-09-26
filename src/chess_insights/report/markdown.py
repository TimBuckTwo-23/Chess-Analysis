"""GitHub-flavoured Markdown report, readable in a repo or on GitHub mobile.

Numbers are formatted exactly like the HTML report (shared helpers in
``report/html.py``). Every string that comes from data is escaped so it cannot
inject HTML, links or table structure; only http(s) URLs become links.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

from ..models import (
    VALUE_FORMATS,
    Chart,
    Coaching,
    Diagram,
    Drill,
    DrillPuzzle,
    Explanation,
    Insight,
    Kpi,
    ModuleResult,
    Motif,
    OpeningFacts,
    Report,
    ReviewItem,
    Source,
    Strip,
    StudyItem,
    Table,
)
from . import boards
from .html import (
    EXPLANATION_KINDS,
    WHY_OPEN,
    clean_formats,
    format_chip_text,
    format_text,
    format_name,
    format_views,
    line_text,
    maia_text,
    motif_words,
    peer_label,
    tablebase_text,
    training_url,
    view_coaching,
    view_scope_text,
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
    to_number,
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


def chart_text(chart: Chart) -> str:
    """A small chart as one line of numbers: "Bullet +4%, Blitz +7%" or "Bullet: You 47%, Expected 50%; ..."."""
    data = chart_data(chart)
    if not data.has_values:
        return ""
    fmt = data.fmt
    if len(data.series) == 1:
        body = ", ".join(
            f"{label} {format_value(v, fmt)}" for label, v in zip(data.labels, data.series[0][1]) if v is not None
        )
    else:
        body = "; ".join(
            f"{label}: " + ", ".join(f"{name} {format_value(vals[i], fmt)}" for name, vals in data.series if vals[i] is not None)
            for i, label in enumerate(data.labels)
        )
    if data.reference is not None:
        body += f" (reference {format_value(data.reference, fmt)})"
    title = text_or_empty(chart.title)
    return f"{title}: {body}" if title else body


def _position_lines(d: Diagram, labels: Optional[dict[str, str]] = None, *, indent: str = "", title: bool = True,
                    why: str = "", strips: bool = True) -> list[str]:
    """A board in text: title, caption, the FEN in code with a Lichess link, the arrows in words, the lines.
    ``title=False`` (an explanation, which names the move and format itself) gives just "Position"."""
    links = [
        md_link(u, t)
        for u, t in ((d.link, (labels or {}).get(d.link) or "game"), (lichess_analysis_url(d.fen), "analyse on Lichess"))
        if u
    ]
    tc = text_or_empty(getattr(d, "time_class", ""))
    head = f"**{md_text(d.title)}**" if title and text_or_empty(d.title) else "Position"
    if tc and title:
        head += f" ({md_text(format_name(tc).lower())} game)"
    parts = [md_text(d.caption)] if text_or_empty(d.caption) else []
    parts.append(md_code(d.fen) or "")
    line = f"{indent}- {head}: " + " ".join(p for p in parts if p)
    if links:
        line += " · " + " · ".join(links)
    if why:
        line += f" · {why}"
    out = [line]
    board = boards.parse_board(d.fen)
    arrows = boards.arrows_text(board, d.arrows or []) if board is not None else ""
    if arrows:
        out.append(f"{indent}  - Arrows: {md_text(arrows)}")
    for strip in (getattr(d, "strips", None) or []) if strips else []:
        if isinstance(strip, Strip):
            moves = boards.compact_labels([getattr(f, "move", "") for f in strip.frames or []])
            if moves:
                out.append(f"{indent}  - {md_text(strip.title) or 'Line'}: {md_code(moves)}")
    return out


def _insight_line(
    ins: Insight,
    *,
    show_study: bool,
    with_kind: bool,
    labels: Optional[dict[str, str]] = None,
    plan_no: Optional[int] = None,
    view_tc: str = "",
    why: str = "",
) -> list[str]:
    glyph = glyph_for(ins)
    label = f"{md_text(f'{KIND_LABELS[insight_kind(ins)]} · {category_label(ins.category)}')}: " if with_kind else ""
    head = f"- `{glyph}` **{label}{md_text(ins.title)}**"
    if text_or_empty(ins.detail):
        head += f" — {md_text(ins.detail)}"
    formats = clean_formats(getattr(ins, "formats", None))
    fmt = format_chip_text(formats) if formats and list(formats) != [view_tc] else ""
    head += f" _({confidence_label(ins.confidence)}{' · ' + md_text(fmt) if fmt else ''})_"
    if plan_no:
        head += f" · what to do: study plan item {plan_no}"
    else:
        games = _games_line(ins.example_games, "review", labels)
        if games:
            head += f" · {games}"
    if why:
        head += f" · {why}"
    lines = [head]
    if isinstance(getattr(ins, "chart", None), Chart):
        text = chart_text(ins.chart)
        if text:
            lines.append(f"  - Chart: {md_text(text)}")
    if isinstance(getattr(ins, "diagram", None), Diagram):
        lines += _position_lines(ins.diagram, labels, indent="  ")
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
    report_formats = clean_formats(getattr(report, "formats", None))
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
            formats = clean_formats(getattr(ins, "formats", None))
            if formats and set(formats) != set(report_formats or formats):
                where.append(format_text(formats))  # only when the finding is about some formats, not all
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
    module: ModuleResult,
    labels: Optional[dict[str, str]] = None,
    plan: Optional[dict[str, int]] = None,
    *,
    view_tc: str = "",
    why_by_id: Optional[dict[str, int]] = None,
    why_by_epd: Optional[dict[str, int]] = None,
) -> list[str]:
    title = text_or_empty(module.title) or text_or_empty(module.key) or "Section"
    lines = [f"## {md_text(title)}", ""]
    if text_or_empty(module.summary):
        lines += [md_text(module.summary), ""]
    if module.kpis:
        lines += [_kpi_line(k) for k in module.kpis] + [""]
    for chart in module.charts or []:
        lines += _chart(chart)
    diagrams = [d for d in getattr(module, "diagrams", None) or [] if isinstance(d, Diagram)]
    for d in diagrams:
        n = (why_by_epd or {}).get(boards.epd(d.fen))
        lines += _position_lines(d, labels, why=f"why: explanation {n} below" if n else "")
    if diagrams:
        lines.append("")
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
            n = (why_by_id or {}).get(text_or_empty(ins.id))
            lines += _insight_line(
                ins, show_study=True, with_kind=True, labels=labels, plan_no=(plan or {}).get(text_or_empty(ins.id)),
                view_tc=view_tc, why=f"why: explanation {n} below" if n else "",
            )
        lines.append("")
    return lines


def render_markdown(report: Report) -> str:
    """The whole report as GitHub-flavoured Markdown: the main report, then one part per format."""
    token = NO_GAME_LINKS.set(bool(getattr(report, "demo", False)))
    try:
        return _render(report)
    finally:
        NO_GAME_LINKS.reset(token)


def _formats_line(label: str, formats: Any) -> str:
    counts = clean_formats(formats)
    if not counts:
        return ""
    return f"{label}: " + " · ".join(f"{md_text(format_name(tc))} {n:,}" for tc, n in counts.items())


def _body(report: Report, coaching: Any) -> list[str]:
    """Headline, glance lists, study plan, sections and coaching of one report (the main one or one format's)."""
    lines: list[str] = []
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

    exps = [e for e in (getattr(coaching, "explanations", None) or []) if isinstance(e, Explanation)]
    why_by_id: dict[str, int] = {}
    why_by_epd: dict[str, int] = {}
    for k, e in enumerate(exps, 1):
        if text_or_empty(e.insight_id):
            why_by_id.setdefault(text_or_empty(e.insight_id), k)
        why_by_epd.setdefault(boards.epd(e.epd or e.fen), k)
    modules = list(report.modules or [])
    keys = [text_or_empty(m.key) for m in modules]
    at = next((keys.index(k) + 1 for k in ("mistakes", "engine_stats") if k in keys), len(modules))
    view_tc = text_or_empty(getattr(report, "time_class", "")).lower()
    for i, module in enumerate(modules):
        lines += ["---", ""] + _module(module, labels, plan, view_tc=view_tc, why_by_id=why_by_id, why_by_epd=why_by_epd)
        if i + 1 == at:
            lines += _coaching(coaching, labels, view_tc)
    if at == 0:
        lines += _coaching(coaching, labels, view_tc)
    return lines


def _render(report: Report) -> str:
    lines = [f"# {md_text(report_title(report))}", ""]
    meta = [f"**{games_text(report.n_games)}**"]
    span = date_range_text(report.date_from, report.date_to)
    if span:
        meta.append(span)
    if text_or_empty(report.filters):
        meta.append(md_text(report.filters))
    lines.append(" · ".join(meta))
    for extra in (_formats_line("Formats", report.formats), _formats_line("Engine-analysed games", report.engine_formats)):
        if extra:
            lines += ["", extra]
    if text_or_empty(report.engine_note):
        lines += ["", f"_{md_text(report.engine_note)}_"]
    if getattr(report, "demo", False):
        lines += ["", f"**{md_text(DEMO_NOTE)}**"]
    views = format_views(report)
    if views:
        names = ", ".join(md_text(format_name(tc)) for tc, _ in views)
        lines += ["", f"_This is every format together. The same analysis on each format's games alone follows the "
                      f"main report: {names}._"]
    lines.append("")
    lines += _body(report, getattr(report, "coaching", None))

    for tc, sub in views:
        lines += ["---", "", f"# {md_text(view_scope_text(sub))}", ""]
        if list(clean_formats(sub.engine_formats)) not in ([], [tc]):
            lines += [_formats_line("Engine-analysed games", sub.engine_formats), ""]
        if text_or_empty(sub.engine_note):
            lines += [f"_{md_text(sub.engine_note)}_", ""]
        lines += _body(sub, view_coaching(sub, report, tc))

    lines += ["---", "", "### How this report works", ""]
    lines += [f"- {md_text(note)}" for note in METHOD_NOTES]
    generated = format_generated(report.generated_at)
    if generated:
        lines += ["", f"_Generated {generated} by chess-insights._"]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- coaching
def _theme_links(themes: Iterable[Any]) -> str:
    out, seen = [], set()
    for theme in themes:
        key = text_or_empty(theme)
        if not key or key in seen:
            continue
        seen.add(key)
        url = training_url(key)
        out.append(md_link(url, motif_words(key)) if url else md_text(motif_words(key)))
    return ", ".join(out)


def _sources_line(sources: Iterable[Any]) -> str:
    out, seen = [], set()
    for src in sources:
        if not isinstance(src, Source) or not text_or_empty(src.name):
            continue
        key = (text_or_empty(src.name), text_or_empty(src.url))
        if key in seen:
            continue
        seen.add(key)
        name = md_link(src.url, src.name) if safe_url(src.url) else md_text(src.name)
        extra = ", ".join(x for x in (text_or_empty(src.license), f"retrieved {text_or_empty(src.retrieved)}"
                                      if text_or_empty(src.retrieved) else "") if x)
        out.append(name + (f" ({md_text(extra)})" if extra else ""))
    return " · ".join(out)


def _moves_text(stats: Iterable[Any], played: str) -> str:
    out = []
    for m in list(stats)[:4]:
        san = text_or_empty(getattr(m, "san", ""))
        if not san:
            continue
        bits = [format_value(getattr(m, "share", None), "pct")]
        score = getattr(m, "score", None)
        if score is not None:
            bits.append(f"score {format_value(score, 'pct')}")
        out.append(f"{san}{' (yours)' if san == played else ''} {', '.join(bits)}")
    return md_text("; ".join(out))


def _opening_lines(op: Any, played: str) -> list[str]:
    if not isinstance(op, OpeningFacts):
        return []
    name = " ".join(x for x in (text_or_empty(op.eco), text_or_empty(op.name)) if x)
    out = [f"- Opening: {md_text(name)}" if name else "- Opening databases"]
    if op.masters:
        out.append(f"  - Masters: {_moves_text(op.masters, played)}")
    if op.peers:
        out.append(f"  - {md_text(peer_label(op.peer_groups or []))}: {_moves_text(op.peers, played)}")
    ranks = []
    if to_number(op.played_rank_masters):
        ranks.append(f"the masters' choice number {int(to_number(op.played_rank_masters))}")
    if to_number(op.played_rank_peers):
        ranks.append(f"number {int(to_number(op.played_rank_peers))} among {peer_label(op.peer_groups or [])}")
    if ranks and played:
        out.append(f"  - Your move {md_text(played)} is {md_text(' and '.join(ranks))}.")
    for line in (op.cloud_lines or [])[:3]:
        t = line_text(line)
        if t:
            out.append(f"  - Engine line (Lichess cloud evaluation): {md_code(t)}")
    if safe_url(op.master_game):
        out.append(f"  - {md_link(op.master_game, 'A master game from this position')}")
    if text_or_empty(op.wiki_text):
        credit = md_link(op.wiki_url, "Wikibooks, CC BY-SA 4.0") if safe_url(op.wiki_url) else "Wikibooks, CC BY-SA 4.0"
        out.append(f"  - “{md_text(op.wiki_text)}” ({credit})")
    return out if len(out) > 1 else []


def _explanation(e: Explanation, k: int, labels: Optional[dict[str, str]], view_tc: str) -> list[str]:
    played, best = text_or_empty(e.played), text_or_empty(e.best)
    title = md_text(played or "Your move") + (f", better {md_text(best)}" if best and best != played else "")
    kicker = [EXPLANATION_KINDS.get(text_or_empty(e.kind), "Position")]
    tc = text_or_empty(e.time_class).lower()
    if tc and tc != view_tc:
        kicker.append(format_name(tc))
    drop = to_number(e.drop)
    if drop:
        kicker.append(f"{format_value(drop / 100.0, 'pct')} winning chances lost")
    if int(to_number(e.repeats) or 0) > 1:
        kicker.append(f"you played it in {int(e.repeats)} games")
    lines = [f"### {k}. {title}", "", f"_{md_text(' · '.join(kicker))}_", ""]
    d = e.diagram if isinstance(e.diagram, Diagram) else None
    if d is not None:  # the full lines follow, so the strips are not repeated
        lines += _position_lines(d, labels, title=False, strips=False)
    elif boards.parse_board(e.fen) is not None:
        lines.append(f"- Position: {md_code(e.fen)} · {md_link(lichess_analysis_url(e.fen), 'analyse on Lichess')}")
    if lines[-1]:
        lines.append("")
    if text_or_empty(e.text):
        lines += [md_text(e.text) + (" _(Wording by the AI coach, checked against the engine lines.)_"
                                     if text_or_empty(e.text_source) == "llm" else ""), ""]
    items = []
    for line, label in ((e.refutation, f"What {played or 'your move'} allows"), (e.best_line, f"Better {best}" if best else "Better")):
        t = line_text(line)
        if t:
            items.append(f"- {md_text(label)}: {md_code(t)}")
    motifs = [m for m in e.motifs or [] if isinstance(m, Motif)]
    allowed = [m.theme for m in motifs if m.line == "refutation"]
    missed = [m.theme for m in motifs if m.line == "best"]
    other = [m.theme for m in motifs if m.line not in ("refutation", "best")]
    for label, themes in ((f"What {played or 'your move'} allowed", allowed), ("What you missed", missed),
                          ("Patterns", other), ("Practise", [t for t in e.drill_themes or [] if t not in allowed + missed + other])):
        links = _theme_links(themes)
        if links:
            items.append(f"- {md_text(label)}: {links}")
    items += [f"- {md_text(f)}" for f in e.facts or [] if text_or_empty(f)]
    if isinstance(e.chart, Chart):
        t = chart_text(e.chart)
        if t:
            items.append(f"- {md_text(t)}")
    else:
        concepts = [f"{text_or_empty(c.label) or text_or_empty(c.term)} {format_value(c.value, 'signed_float2')}"
                    for c in e.concepts or [] if to_number(getattr(c, "value", None)) is not None]
        if concepts:
            items.append(f"- Where the lines end (pawns, from your side): {md_text(', '.join(concepts))}")
    items += _opening_lines(e.opening, re.sub(r"^\d+\.(?:\.\.)?\s*", "", played))
    tb = tablebase_text(e.tablebase)
    if tb:
        items.append(f"- Endgame tablebase: {md_text(tb)}.")
    maia = maia_text(e.maia, e)
    if maia:
        items.append(f"- How findable: {md_text(maia)}")
    sources = _sources_line(list(e.sources or []) + list(getattr(e.opening, "sources", None) or []))
    if sources:
        items.append(f"- Sources: {sources}")
    games = _games_line(([e.game_url] if text_or_empty(e.game_url) else []) + list(e.games or []), "Review", labels)
    if games:
        items.append(f"- {games}")
    return lines + items + ([""] if items else [])


def _review_lines(items: Iterable[Any]) -> list[str]:
    out = []
    for it in items:
        if not isinstance(it, ReviewItem):
            continue
        bits = [f"due {md_text(it.due)}" if text_or_empty(it.due) else "", md_code(it.fen)]
        if safe_url(it.url):
            bits.append(md_link(it.url, "open"))
        out.append(f"- [ ] {md_text(it.title) or 'Position'} — " + " · ".join(b for b in bits if b))
    return out


def _coaching(coaching: Any, labels: Optional[dict[str, str]], view_tc: str = "") -> list[str]:
    """The coaching sections: why your moves go wrong (the engine's lines), then practice."""
    if not isinstance(coaching, Coaching):
        return []
    lines: list[str] = []
    exps = [e for e in coaching.explanations or [] if isinstance(e, Explanation)]
    why: list[str] = []
    for k, e in enumerate(exps, 1):
        if k == WHY_OPEN + 1:
            why += [f"_{len(exps) - WHY_OPEN} more:_", ""]
        why += _explanation(e, k, labels, view_tc)
    if isinstance(coaching.motif_chart, Chart) or isinstance(coaching.motif_profile, Table):
        why += ["### Patterns in the mistakes", ""]
        if isinstance(coaching.motif_chart, Chart):
            why += _chart(coaching.motif_chart)
        if isinstance(coaching.motif_profile, Table) and coaching.motif_profile is not getattr(coaching.motif_chart, "table", None):
            why += md_table(coaching.motif_profile) + [""]
    for table, heading in ((coaching.theory_exit, "Where you leave opening theory"),
                           (coaching.endgames, "Endgames checked with the tablebase")):
        if isinstance(table, Table):
            why += [f"### {heading}", ""] + md_table(table) + [""]
            if text_or_empty(table.note):
                why += [f"_{md_text(table.note)}_", ""]
    notes = [text_or_empty(n) for n in coaching.notes or [] if text_or_empty(n)]
    if why or notes:
        lines += ["---", "", "## Why these moves go wrong", ""]
        if exps:
            lines += [f"The engine's lines for {len(exps)} position{'s' if len(exps) != 1 else ''} from your games: "
                      "what your move allowed, what was better, and the pattern to practise.", ""]
        lines += why
        if notes:
            lines += ["_Notes:_", ""] + [f"- {md_text(n)}" for n in notes] + [""]

    practice: list[str] = []
    progress = []
    for p in coaching.progress or []:
        text = text_or_empty(getattr(p, "text", ""))
        if text:
            mark = {True: "improved: ", False: "not yet: "}.get(getattr(p, "improved", None), "")
            progress.append(f"- {mark}{md_text(text)}")
    if progress:
        practice += ["### Since your last report", ""] + progress + [""]
    due = _review_lines(coaching.review_due or [])
    if due:
        practice += ["### Due for review", ""] + due + [""]
    drills = [d for d in coaching.drills or [] if isinstance(d, Drill)]
    if drills:
        practice += ["### Puzzle packs", ""]
        for d in drills:
            bits = [md_text(d.reason)] if text_or_empty(d.reason) else []
            if isinstance(d.rating_range, (list, tuple)) and len(d.rating_range) == 2:
                bits.append(f"puzzles rated {d.rating_range[0]}–{d.rating_range[1]}")
            if text_or_empty(d.file):
                bits.append(f"saved as {md_code(d.file)}")
            if safe_url(d.link):
                bits.append(md_link(d.link, "practise on Lichess"))
            practice.append(f"- **{md_text(d.title) or md_text(motif_words(d.theme))}** — " + " · ".join(bits))
            for p in [p for p in d.puzzles or [] if isinstance(p, DrillPuzzle)][:3]:
                name = md_link(p.url, f"puzzle {p.puzzle_id}") if safe_url(p.url) else md_text(f"puzzle {p.puzzle_id}")
                practice.append(f"  - {name} (rated {p.rating}): {md_code(p.fen)}")
        practice.append("")
    plan = [p for p in coaching.weekly_plan or [] if text_or_empty(getattr(p, "item", ""))]
    if plan:
        practice += ["### Your week", ""] + md_table(Table(
            "", ["What", "Minutes a week", "Check again", "Note"],
            [[p.item, p.minutes_per_week, p.recheck_date or None, p.note or None] for p in plan],
            ["text", "int", "text", "text"])) + [""]
    upcoming = _review_lines(coaching.review or [])
    if upcoming:
        practice += ["### Coming up for review", ""] + upcoming + [""]
    if practice:
        lines += ["---", "", "## Practice", ""] + practice
    return lines
