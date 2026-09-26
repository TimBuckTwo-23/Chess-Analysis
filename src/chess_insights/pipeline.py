"""Run every analysis module and assemble the Report.

    games + evals ─ run_modules ─ coach.build_coaching ─ apply_deep_verdicts ─ build_report (fill, rank, plan)
                                                        ─ coach.finish_coaching
                   └─ the same per format (Report.format_reports): run_modules ─ apply_deep_verdicts
                      ─ view_only_observations ─ build_report

Two checks keep the page to the main report's claims:

* ``apply_deep_verdicts``: a repeated-mistake weakness whose move the coaching's deeper Stockfish search clears
  (``Explanation.verdict`` "close" or "fine") is shown as an observation, in the main report and every view.
* ``view_only_observations``: a format view tests its findings on that format's games alone, so each view is
  another set of chances for a false claim (about 0.6 per page with three views, against 0.1-0.2 for the main
  report alone, on null data). A view's strength or weakness that is not also a claim of the main report is
  shown as an observation in that view, with a first sentence saying so.

Every step is isolated: a failing module, coaching step or format view logs a warning and the report still builds.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import re
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, get_args

from .context import AnalysisContext
from .fill import fill
from .insights import build_study_plan, headline, practice_actions, rank_insights, summary_lines
from .models import Coaching, Diagram, Explanation, Game, GameEval, Insight, InsightKind, ModuleResult, Report
from .visuals import FORMAT_NAMES, FORMAT_ORDER, format_counts, format_text

if TYPE_CHECKING:
    from .coach import CoachConfig

log = logging.getLogger(__name__)

# A format gets its own view of the report (Report.format_reports) from this many games, when the games cover two
# or more formats. Below it a view would have too few games for most tests to say anything (the modules need 8
# games per opening and colour, 25 per split), and its sections would be mostly "not enough data".
# (visuals.MIN_FORMAT_GAMES is the much smaller bar for a format's own bar in a chart split by format.)
MIN_FORMAT_GAMES = 60

# (module path under chess_insights.analysis, section title used if the module crashes)
MODULES: list[tuple[str, str]] = [
    ("results", "Results & rating"),
    ("openings", "Openings"),
    ("time_mgmt", "Clock & time management"),
    ("endings", "How your games end"),
    ("habits", "Habits & tilt"),
    ("structure", "Development & king safety"),
    ("engine_stats", "Engine review"),
    ("mistakes", "Positions you keep getting wrong"),
]


_KINDS = frozenset(get_args(InsightKind))


def _finite(value: Any) -> float:
    """``value`` as a finite float; ValueError for None, pd.NA, NaN, inf, strings ..."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _check_insight(ins: Any) -> Insight:
    """Validate (and normalise in place) one insight so ranking and rendering can't crash on it.

    severity / confidence become plain floats (a numpy or pandas scalar is fine; None, pd.NA, NaN
    are not: NaN would even rank first, since min(1.0, nan) is 1.0).
    """
    if not isinstance(ins, Insight):
        raise ValueError(f"not an Insight: {type(ins).__name__}")
    if ins.kind not in _KINDS:
        raise ValueError(f"unknown kind {ins.kind!r}")
    for name in ("id", "category", "title"):
        if not isinstance(getattr(ins, name), str) or not getattr(ins, name):
            raise ValueError(f"{name} must be a non-empty string, got {getattr(ins, name)!r}")
    ins.severity = _finite(ins.severity)
    ins.confidence = _finite(ins.confidence)
    ins.detail = "" if ins.detail is None else str(ins.detail)
    ins.study = [str(a) for a in (ins.study or []) if a]
    ins.example_games = [str(u) for u in (ins.example_games or []) if u]
    if not isinstance(ins.evidence, dict):
        ins.evidence = {}
    return ins


def _clean_insights(result: ModuleResult, name: str) -> None:
    """Drop (with a warning) the insights of one module that would break ranking or rendering."""
    kept = []
    for ins in result.insights or []:
        try:
            kept.append(_check_insight(ins))
        except ValueError as exc:
            log.warning("analysis module %s: dropped insight %r: %s", name, getattr(ins, "id", ins), exc)
    result.insights = kept


def run_modules(ctx: AnalysisContext, modules: Optional[list[tuple[str, str]]] = None) -> list[ModuleResult]:
    """Run each module in isolation: one failing module never sinks the whole report."""
    results: list[ModuleResult] = []
    for name, title in modules or MODULES:
        try:
            mod = importlib.import_module(f"chess_insights.analysis.{name}")
            result = mod.analyze(ctx)
            if not isinstance(result, ModuleResult):
                raise TypeError(f"analyze() returned {type(result).__name__}, not ModuleResult")
            _clean_insights(result, name)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 — isolate module bugs
            log.warning("analysis module %s failed: %s", name, exc)
            log.debug("%s", traceback.format_exc())
            results.append(
                ModuleResult(
                    key=name,
                    title=title,
                    summary=f"This section could not be computed ({type(exc).__name__}: {exc}).",
                    stats={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return results


def _time_control_text(g: Game) -> str:
    if g.time_class == "daily":
        return "daily"
    if not g.base_seconds:
        return g.time_class
    base = g.base_seconds / 60
    return f"{base:g}+{g.increment}"


def game_label(g: Game, year: Optional[int] = None) -> str:
    """Link text for a game: 'Loss · 5+0 · vs 1512 · 12 Aug' (the year only when it differs from ``year``)."""
    result = {"win": "Win", "draw": "Draw", "loss": "Loss"}.get(g.outcome, g.outcome)
    opponent = f"vs {g.opp_rating}" if g.opp_rating else (f"vs {g.opponent}" if g.opponent else "")
    end = g.end_time.astimezone(timezone.utc) if g.end_time.tzinfo else g.end_time
    day = f"{end.day} {end:%b}" + (f" {end.year}" if year is not None and end.year != year else "")
    return " · ".join(p for p in (result, _time_control_text(g), opponent, day) if p)


def _diagram_links(d: Any) -> Iterator[Any]:
    """The game a board links to, and any links its strips or their frames carry."""
    if d is None:
        return
    yield getattr(d, "link", None)
    for strip in getattr(d, "strips", None) or []:
        yield getattr(strip, "link", None)
        for frame in getattr(strip, "frames", None) or []:
            yield getattr(frame, "link", None)


def _table_cells(table: Any) -> Iterator[Any]:
    for row in getattr(table, "rows", None) or []:
        if isinstance(row, (list, tuple)):
            yield from row


def linked_games(
    modules: Iterable[ModuleResult], coaching: Optional[Coaching] = None, extra: Iterable[str] = ()
) -> set[str]:
    """Every link the report shows that could be a game: findings (their games and board), each section's boards
    and tables, the coaching's explanations (their game, the games they repeat in, their board), endgame boards and
    tables, and ``extra`` (the study plan's games). Other links are harmless: only games get a label."""
    wanted: list[Any] = list(extra)
    for m in modules:
        for ins in m.insights or []:
            wanted += ins.example_games or []
            wanted += _diagram_links(getattr(ins, "diagram", None))
            wanted += _table_cells(getattr(getattr(ins, "chart", None), "table", None))
        for d in getattr(m, "diagrams", None) or []:
            wanted += _diagram_links(d)
        for table in [*(m.tables or []), *(getattr(c, "table", None) for c in m.charts or [])]:
            wanted += _table_cells(table)
    if coaching is not None:
        for e in getattr(coaching, "explanations", None) or []:
            wanted += [getattr(e, "game_url", None), *(getattr(e, "games", None) or [])]
            wanted += _diagram_links(getattr(e, "diagram", None))
        for d in getattr(coaching, "endgame_diagrams", None) or []:
            wanted += _diagram_links(d)
        for name in ("endgames", "theory_exit", "motif_profile"):
            wanted += _table_cells(getattr(coaching, name, None))
    return {u for u in wanted if isinstance(u, str) and u}


def game_labels(
    games: list[Game],
    modules: list[ModuleResult],
    extra: Iterable[str] = (),
    *,
    coaching: Optional[Coaching] = None,
) -> dict[str, str]:
    """Labels for every game the report links to (``linked_games``): "Loss · 5+0 · vs 1512 · 12 Aug"."""
    wanted = linked_games(modules, coaching, extra)
    year = max((g.end_time.year for g in games), default=None)
    return {g.url: game_label(g, year) for g in games if g.url and g.url in wanted}


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    """Whether ``fn`` takes the keyword argument ``name`` (a stand-in with an older signature may not)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # a builtin or a mock without a signature
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _practice(
    modules: list[ModuleResult],
    coaching: Optional[Coaching],
    time_class: str = "",
    puzzle_file_formats: Optional[dict[str, int]] = None,
) -> list[str]:
    """``insights.practice_actions``, with the coaching when it takes it (its drills become plan actions), and in a
    format view with how many of the puzzle file's puzzles are from that format."""
    kwargs: dict[str, Any] = {}
    if _accepts(practice_actions, "coaching"):
        kwargs["coaching"] = coaching
    if time_class and _accepts(practice_actions, "time_class"):
        kwargs["time_class"] = time_class
        kwargs["puzzle_file_formats"] = puzzle_file_formats
    return practice_actions(modules, **kwargs)


def build_report(
    ctx: AnalysisContext,
    modules: list[ModuleResult],
    *,
    filters: str = "",
    engine_note: str = "",
    top_n: Optional[int] = None,
    coaching: Optional[Coaching] = None,
    practice_coaching: Optional[Coaching] = None,
    time_class: str = "",
    puzzle_file_formats: Optional[dict[str, int]] = None,
) -> Report:
    """``top_n``: how many strengths and weaknesses the report lists (None: every one that passed the rule).

    Every finding first gets its formats and a picture where its module gave none (``fill``). ``coaching`` goes
    on the report; the study plan's practice actions come from ``practice_coaching`` (default: ``coaching``), so a
    format view can point at drills that live on the main report. ``time_class`` marks a one-format view;
    ``puzzle_file_formats`` (a view only) counts the puzzle file's puzzles per format, as the file holds every
    format's.
    """
    fill(modules, ctx)
    strengths, weaknesses = rank_insights(modules, top_n=top_n)
    practice = _practice(modules, practice_coaching if practice_coaching is not None else coaching,
                         time_class, puzzle_file_formats)
    plan = build_study_plan(weaknesses, practice=practice)
    games = ctx.games
    return Report(
        username=ctx.username,
        generated_at=datetime.now(timezone.utc),
        filters=filters,
        n_games=len(games),
        date_from=games[0].end_time if games else None,
        date_to=games[-1].end_time if games else None,
        modules=modules,
        strengths=strengths,
        weaknesses=weaknesses,
        study_plan=plan,
        engine_note=engine_note,
        headline=headline(strengths, weaknesses, plan),
        summary_lines=summary_lines(modules, plan, strengths, weaknesses),
        game_labels=game_labels(games, modules, (u for item in plan for u in item.games), coaching=coaching),
        demo=bool(ctx.opt("demo", False)),
        time_class=time_class,
        formats=format_counts(games),
        engine_formats=format_counts(g for g in games if g.game_id in ctx.evals),
        coaching=coaching,
    )


# --------------------------------------------------------------------------- coaching
def _coaching_failed(exc: Exception, what: str) -> str:
    log.warning("%s failed: %s", what, exc)
    log.debug("%s", traceback.format_exc())
    return f"The coaching could not be computed ({type(exc).__name__}: {exc})."


def build_coaching(ctx: AnalysisContext, modules: list[ModuleResult], cfg: "CoachConfig") -> Coaching:
    """``coach.build_coaching``, isolated like a module: a failure leaves a Coaching with a note, never an error.

    The coaching may add findings to a section (only motif findings that passed the claim rule); they are
    checked like every module's before ranking.
    """
    try:
        from . import coach

        coaching = coach.build_coaching(ctx, modules, cfg)
        if not isinstance(coaching, Coaching):
            raise TypeError(f"build_coaching() returned {type(coaching).__name__}, not Coaching")
    except Exception as exc:  # noqa: BLE001 — the report must build without its coaching
        coaching = Coaching(notes=[_coaching_failed(exc, "coaching")])
    for m in modules:
        _clean_insights(m, m.key)
    return coaching


def finish_coaching(report: Report, cfg: "CoachConfig", games: Optional[list[Game]] = None) -> None:
    """``coach.finish_coaching`` (progress, Maia, the LLM coach), isolated: a failure becomes a note. ``games``: the
    report's games, which progress needs to tell the games since the previous report from the earlier ones."""
    try:
        from . import coach

        if games is not None and _accepts(coach.finish_coaching, "games"):
            coach.finish_coaching(report, cfg, games=games)
        else:
            coach.finish_coaching(report, cfg)
    except Exception as exc:  # noqa: BLE001
        note = _coaching_failed(exc, "finishing the coaching")
        if report.coaching is not None:
            report.coaching.notes.append(note)


# --------------------------------------------------------------------------- claim checks
CLAIM_KINDS = ("strength", "weakness")
DEEP_CLEARED = ("close", "fine")  # Explanation.verdict values that clear the move the finding is about

_KIND_SEGMENT = re.compile(r"\.(strength|weakness|observation)\.")


def claim_ids(modules: Iterable[ModuleResult]) -> set[str]:
    """The ids of every strength and weakness in the sections (the lists at the top show a subset of them)."""
    return {i.id for m in modules for i in m.insights or [] if i.kind in CLAIM_KINDS}


def _kindless(insight_id: str) -> str:
    """The id without its kind: a finding that is a weakness in one view and an observation in another
    (mistakes.weakness.x / mistakes.observation.x) is about the same thing."""
    return _KIND_SEGMENT.sub(".*.", str(insight_id or ""), count=1)


def _epd(fen: Any) -> str:
    return " ".join(str(fen or "").split()[:4])


def _san(label: Any) -> str:
    """"5...e5" / "5.d3" / "d3" -> "e5" / "d3"."""
    return str(label or "").replace("…", "...").split(".")[-1].strip()


def _position_key(ins: Insight) -> Optional[tuple[str, str]]:
    """(EPD, SAN) of a finding about one move in one position (the mistakes module's evidence), else None."""
    evidence = ins.evidence or {}
    fen, move = evidence.get("fen"), evidence.get("move")
    return (_epd(fen), _san(move)) if fen and move else None


def explains(exp: Explanation, ins: Insight) -> bool:
    """Whether ``exp`` explains the finding ``ins``: its insight id is the finding's (whatever the kind in it) and,
    for a finding about one move in one position, it is about that position and move (a follow-up move folded into
    an earlier habit carries the habit's id, but its verdict is about another move)."""
    if not getattr(exp, "insight_id", None) or _kindless(exp.insight_id) != _kindless(ins.id):
        return False
    key = _position_key(ins)
    return key is None or key == (_epd(exp.epd or exp.fen), _san(exp.played))


def _deep_depth(exp: Explanation, coaching: Coaching) -> Optional[int]:
    """The depth of the deeper search behind ``exp`` (its lines, its source, else the coaching's setting)."""
    depths = [line.depth for line in (exp.best_line, exp.refutation) if line is not None and line.depth]
    if depths:
        return min(depths)
    for source in exp.sources or []:
        found = re.search(r"depth (\d+)", str(getattr(source, "name", "")))
        if found:
            return int(found.group(1))
    depth = (coaching.settings or {}).get("depth")
    return depth if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0 else None


def deep_verdict_sentence(exp: Explanation, kind: str, depth: Optional[int]) -> str:
    """"A deeper Stockfish search (depth 16) rates 5.d3 about as good as 5.d4, so it is not listed as a weakness;
    ..." (or "makes 5.d3 its own first choice" when the deeper search picks the move played)."""
    at = f" (depth {depth})" if depth else ""
    if exp.verdict == "fine" or not exp.best or _san(exp.best) == _san(exp.played):
        what = f"makes {exp.played} its own first choice"
    else:
        what = f"rates {exp.played} about as good as {exp.best}"
    return (f"A deeper Stockfish search{at} {what}, so it is not listed as a {kind}; the numbers below come from the "
            "quicker game analysis.")


# The mistakes module's words for a habit and for the moves that follow it: not true of a move the deeper search
# clears.
_HABIT_SENTENCE = re.compile(
    r"\s*(You go wrong on about 1 move in \d+ overall, so playing this one move in \d+ of \d+ games is a habit, not "
    r"bad luck\.|That is a habit, not bad luck\.)"
)
_FIX_THE_FIRST = ": fix the first move and the rest goes with it."
_BETTER_SUFFIX = re.compile(r";\s*[^;]+ is better$")


def _observe(ins: Insight, first: str, because: str, **evidence: Any) -> None:
    """Make ``ins`` an observation: the same card, chart and board, with ``first`` as its detail's first sentence
    and the reason in its evidence (``observation_because``). A repeated-mistake finding loses the sentence that
    calls the move a habit, not bad luck: that is the claim it no longer makes."""
    ins.kind = "observation"
    detail = _HABIT_SENTENCE.sub("", ins.detail or "") if _position_key(ins) is not None else ins.detail or ""
    ins.detail = f"{first} {detail}".strip()
    ins.evidence = {**(ins.evidence or {}), "observation_because": because, **evidence}


def _say_in_summary(module: ModuleResult, text: str) -> None:
    """Add ``text`` to a section's summary, which was written when its findings were still claims."""
    module.summary = f"{(module.summary or '').rstrip()} {text}".strip()


def _count_words(n: int, one: str, many: str) -> str:
    return one if n == 1 else many.replace("{n}", str(n))


def apply_deep_verdicts(modules: list[ModuleResult], coaching: Optional[Coaching]) -> list[str]:
    """Findings the coaching's deeper search contradicts become observations; returns their ids.

    A strength or weakness that an explanation explains (``explains``: its insight id, and the same position and
    move) whose ``verdict`` is "close" or "fine" rests on a verdict of the quicker game analysis that the deeper
    search does not confirm. It stays in its section as an observation, its detail starting with the deeper
    check's result in words; for the mistakes module's findings, the "is better" in the title and the habit
    sentence go, as they would contradict it, and the section's summary (written while it was a claim) says which
    moves the deeper search clears. A check of the finding's premise, not a new claim: nothing is ever promoted.
    Run it before ranking, on the main report's modules and on every view's (the same insight ids).
    """
    if coaching is None:
        return []
    cleared = [e for e in coaching.explanations or []
               if isinstance(e, Explanation) and getattr(e, "verdict", "error") in DEEP_CLEARED and e.insight_id]
    changed: list[str] = []
    for m in modules:
        moves: list[str] = []
        for ins in m.insights or []:
            if ins.kind not in CLAIM_KINDS:
                continue
            exp = next((e for e in cleared if explains(e, ins)), None)
            if exp is None:
                continue
            depth = _deep_depth(exp, coaching)
            kind = ins.kind
            if _position_key(ins) is not None:
                ins.title = _BETTER_SUFFIX.sub("", ins.title) or ins.title
                ins.detail = (ins.detail or "").replace(_FIX_THE_FIRST, ".")
                ins.study = [a for a in ins.study or [] if f" beats {exp.played}" not in a]
            _observe(ins, deep_verdict_sentence(exp, kind, depth), "deep-check",
                     deep_check={"verdict": exp.verdict, "depth": depth, "best": exp.best, "played": exp.played})
            _clear_board(ins.diagram, exp, depth)
            changed.append(ins.id)
            moves.append(str(exp.played))
        if moves:
            _say_in_summary(m, f"A deeper Stockfish search clears {', '.join(moves)}, so "
                               + _count_words(len(moves), "it is shown as an observation.",
                                              "they are shown as observations."))
    return changed


_GREEN_SENTENCE = re.compile(r"\s*Green:[^.]*\.")


def _clear_board(diagram: Optional[Diagram], exp: Explanation, depth: Optional[int]) -> None:
    """A cleared finding's board loses the quick analysis's green "better move" arrow and says what the deeper
    check found instead (the board object may also be drawn in its section: it changes there too)."""
    if diagram is None:
        return
    diagram.arrows = [a for a in diagram.arrows or [] if a.kind != "best"]
    at = f" at depth {depth}" if depth else ""
    what = ("its own first choice" if exp.verdict == "fine" else "about as good as its first choice")
    caption = _GREEN_SENTENCE.sub("", diagram.caption or "").strip()
    diagram.caption = (caption + " " if caption else "") + f"A deeper Stockfish check{at} rates {exp.played} {what}."


def view_only_sentence(time_class: str) -> str:
    name = FORMAT_NAMES.get(time_class, time_class or "these").lower()
    return (f"Seen in your {name} games only, and not strong enough across all your games to call it a finding: "
            "it may be chance.")


def view_only_observations(modules: list[ModuleResult], main_claims: Iterable[str], time_class: str) -> list[str]:
    """A format view's strengths and weaknesses that are not claims of the main report (by id) become observations
    in the view; returns their ids.

    Each view tests its findings on its own format's games: with three views that is three more sets of chances
    for a false claim on the same page. A view keeps a claim only when the main report makes it too, so the page
    claims nothing the main report doesn't; the rest keep their card, chart, board and title, with a first
    sentence saying the pattern shows in this format only and may be chance (and the section's summary says how
    many). Run it before the view's ranking and study plan.
    """
    main = set(main_claims)
    name = FORMAT_NAMES.get(time_class, time_class or "these").lower()
    changed: list[str] = []
    for m in modules:
        n = 0
        for ins in m.insights or []:
            if ins.kind in CLAIM_KINDS and ins.id not in main:
                _observe(ins, view_only_sentence(time_class), "view-only", view_only=time_class)
                changed.append(ins.id)
                n += 1
        if n:
            _say_in_summary(m, _count_words(
                n, f"One finding below shows in your {name} games only, not across all your games, so it is shown "
                   "as an observation.",
                f"{{n}} findings below show in your {name} games only, not across all your games, so they are shown "
                "as observations."))
    return changed


def _safely(what: str, fn: Callable[[], list[str]]) -> list[str]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a check that fails leaves the findings as they are
        log.warning("%s failed: %s", what, exc)
        log.debug("%s", traceback.format_exc())
        return []


def view_coaching(
    coaching: Optional[Coaching],
    time_class: str,
    games: Iterable[Game] = (),
    modules: Iterable[ModuleResult] = (),
) -> Optional[Coaching]:
    """A format view's copy of the coaching: the explanations of that format's positions. Drills, the motif
    profile and the review schedule stay on the main report (they are practice for every format).

    An explanation belongs to the view when it is of that format (``time_class``), when one of its games
    (``game_url``, ``games``: URLs, looked up in ``games``) is, or when it explains one of the view's findings
    (``modules``; ``explains``). A position repeated in several formats has no single time class, but its
    explanation still belongs in each of their views. The explanations of the view's findings come first (a
    renderer may show only the first few), with the finding's own id in the view (a view may call the same
    finding a weakness where the main report calls it an observation)."""
    if coaching is None:
        return None
    formats = {g.url: g.time_class for g in games if g.url}
    findings = [i for m in modules for i in m.insights or []]
    linked: list[Explanation] = []
    rest: list[Explanation] = []
    for e in coaching.explanations or []:
        finding = next((i for i in findings if explains(e, i)), None)
        if finding is not None:
            linked.append(e if e.insight_id == finding.id else replace(e, insight_id=finding.id))
        elif e.time_class == time_class or any(formats.get(u) == time_class for u in [e.game_url, *(e.games or [])]
                                               if u):
            rest.append(e)
    return Coaching(
        explanations=linked + rest,
        notes=list(coaching.notes),
        settings=dict(coaching.settings),
        llm=dict(coaching.llm),
    )


# --------------------------------------------------------------------------- format views
def view_filters(time_class: str, filters: str) -> str:
    """"blitz only, rated, standard chess" from the report's filter text (its time-control part replaced)."""

    def is_time_part(part: str) -> bool:
        return part == "all time controls" or all(p in FORMAT_ORDER for p in part.split("+"))

    parts = [p for p in (filters or "").split(", ") if p and not is_time_part(p)]
    return ", ".join([format_text({time_class: 1}), *parts])


def view_engine_note(time_class: str, view: AnalysisContext, whole: AnalysisContext, engine_note: str) -> str:
    """The engine note for one format's view: how many of its games Stockfish analysed (or why none)."""
    if not whole.evals:
        return engine_note  # not run, or skipped: the same reason holds for every format
    if not view.evals:
        sample = engine_note.strip().rstrip(".")
        note = f"None of your {time_class} games were in the engine sample" + (f" ({sample})." if sample else ".")
        listed = [tc for tc in whole.opt("engine_time_classes", None) or [] if isinstance(tc, str)]
        if listed and time_class not in {tc.lower() for tc in listed}:
            sent = format_text({tc.lower(): 1 for tc in listed}).removesuffix(" only")
            note += f" Only {sent} games were sent to Stockfish."
        elif whole.opt("engine_sample", "recent") != "balanced":
            note += (" A balanced engine sample (--engine-sample balanced, the default on GitHub) takes games from "
                     "every format.")
        elif time_class == "daily":
            note += " A balanced sample takes daily games only when asked to (--engine-time-class)."
        return note
    first = next(iter(view.evals.values()))
    depth = f" at depth {first.depth}" if first.depth else ""
    k, n = len(view.evals), len(view.games)
    if k >= n:
        return f"{first.engine}{depth} on all {n} of your {time_class} games."
    # both samples take a format's games newest first
    return f"{first.engine}{depth} on {k} of your {n} {time_class} games (the most recent)."


def puzzle_file_formats(ctx: AnalysisContext) -> Optional[dict[str, int]]:
    """Puzzles per format in the --puzzles file (``options["puzzle_file"]``), which holds every format's costliest
    mistakes (``mistakes.build_puzzles`` on all the report's games); None without the file or the engine."""
    if not ctx.opt("puzzle_file", "") or not ctx.evals:
        return None
    try:
        from .analysis.mistakes import build_puzzles

        return format_counts(e.game for e in build_puzzles(ctx.games, ctx.evals))
    except Exception as exc:  # noqa: BLE001 — the views then point at the file without counts
        log.warning("counting the puzzle file's formats failed: %s", exc)
        return None


def build_format_views(
    ctx: AnalysisContext,
    modules: Optional[list[tuple[str, str]]] = None,
    *,
    filters: str = "",
    engine_note: str = "",
    coaching: Optional[Coaching] = None,
    min_games: int = MIN_FORMAT_GAMES,
    main_claims: Optional[Iterable[str]] = None,
) -> dict[str, Report]:
    """The same analysis on each format's games alone, for every format with at least ``min_games`` games.

    Only when the games cover two or more formats (a one-format report is already its own view). Each view is a
    full Report: its findings are tested on that format's games only, its engine sections use that format's
    engine-analysed games, and it has no views of its own. Keys in format order.

    A view claims nothing the main report doesn't: its findings go through ``apply_deep_verdicts`` (the same
    deeper-search verdicts as the main report) and ``view_only_observations`` against ``main_claims`` (the ids of
    the main report's strengths and weaknesses; None: found by running the modules on all the games) before its
    ranking and study plan.
    """
    counts = format_counts(ctx.games)
    if len(counts) < 2:
        return {}
    if main_claims is None:
        whole = run_modules(ctx, modules)
        apply_deep_verdicts(whole, coaching)
        main_claims = claim_ids(whole)
    main_claims = set(main_claims)
    in_file = puzzle_file_formats(ctx)
    views: dict[str, Report] = {}
    for tc, n in counts.items():
        if n < min_games:
            continue
        games = [g for g in ctx.games if g.time_class == tc]
        ids = {g.game_id for g in games}
        sub = AnalysisContext(
            username=ctx.username,
            games=games,
            evals={gid: ev for gid, ev in ctx.evals.items() if gid in ids},
            options=dict(ctx.options),
        )
        try:
            results = run_modules(sub, modules)
            _safely("the deeper-search verdicts", lambda: apply_deep_verdicts(results, coaching))
            _safely("the view-only check", lambda: view_only_observations(results, main_claims, tc))
            views[tc] = build_report(
                sub,
                results,
                filters=view_filters(tc, filters),
                engine_note=view_engine_note(tc, sub, ctx, engine_note),
                coaching=view_coaching(coaching, tc, ctx.games, results),
                practice_coaching=coaching,
                time_class=tc,
                puzzle_file_formats=in_file,
            )
        except Exception as exc:  # noqa: BLE001 — a view is extra; the main report stands without it
            log.warning("the %s view could not be built: %s", tc, exc)
            log.debug("%s", traceback.format_exc())
    return views


def run_analysis(
    games: list[Game],
    username: str,
    *,
    evals: Optional[dict[str, GameEval]] = None,
    options: Optional[dict[str, Any]] = None,
    filters: str = "",
    engine_note: str = "",
    modules: Optional[list[tuple[str, str]]] = None,
    coach: Optional["CoachConfig"] = None,
) -> Report:
    """Games (already filtered) -> Report.

    ``coach``: run the coaching layer (explanations, drills ...) with these settings; None skips it. Its deeper
    search's verdicts apply to the findings before ranking (``apply_deep_verdicts``).
    ``options["format_views"]``: False skips the per-format views (default: built whenever two or more formats
    are present, each from ``MIN_FORMAT_GAMES`` games).
    """
    ctx = AnalysisContext(
        username=username,
        games=sorted(games, key=lambda g: (g.end_time, g.game_id)),  # ties: same order whatever the source order
        evals=evals or {},
        options=options or {},
    )
    results = run_modules(ctx, modules)
    coaching = build_coaching(ctx, results, coach) if coach is not None else None
    _safely("the deeper-search verdicts", lambda: apply_deep_verdicts(results, coaching))
    report = build_report(ctx, results, filters=filters, engine_note=engine_note, coaching=coaching)
    if coach is not None:
        finish_coaching(report, coach, ctx.games)
    if ctx.opt("format_views", True):
        report.format_reports = build_format_views(
            ctx, modules, filters=filters, engine_note=engine_note, coaching=report.coaching,
            main_claims=claim_ids(results),
        )
    return report
