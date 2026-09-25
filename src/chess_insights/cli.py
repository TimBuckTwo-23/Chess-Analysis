"""Command line entry point: ``chess-insights fetch | report | demo``.

Exit codes: 0 success; 1 nothing to analyse (no games, no games in the filter window) or the
report could not be written; 2 usage error (bad option, unknown chess.com player, missing
input file or Stockfish); 3 chess.com could not be reached or refused the request; 130 Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import __version__
from .dataset import describe_filters, filter_games
from .fetch import DEFAULT_CACHE_DIR, expand_path, load_games, load_json_files, read_text, sync
from .models import TIME_CLASSES, Game, Report
from .parse import games_from_pgn, merge_games, parse_games, parse_pgn_headers, split_pgn

EXIT_NO_GAMES = 1
EXIT_USAGE = 2
EXIT_NETWORK = 3

YearMonth = tuple[int, int]


class UserError(Exception):
    """A problem the user can fix; printed as one friendly line (no traceback)."""

    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_console() -> None:
    """Never crash on printing: a Windows console or a redirected file may not be UTF-8 (cp1252, cp437 ...).

    Characters the console can't show (→, Ł, −, CJK usernames ...) print as '?' instead of raising
    UnicodeEncodeError half-way through the summary. The report files are always UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "errors", None) == "strict" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):  # already detached / closed
                pass


# --------------------------------------------------------------------------- argument types
def _csv(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Accept both repeated flags and comma lists: --time-class blitz --time-class rapid,bullet."""
    if not values:
        return None
    out = [v.strip() for item in values for v in item.split(",") if v.strip()]
    return out or None


def _time_classes(text: str) -> str:
    bad = [v for v in (_csv([text]) or []) if v.lower() not in TIME_CLASSES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown time class {', '.join(bad)!s}; choose from {', '.join(TIME_CLASSES)}")
    return text.lower()


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


@dataclass(frozen=True)
class Period:
    """A --since / --until value: a year, a month or a day."""

    year: int
    month: Optional[int] = None
    day: Optional[int] = None

    def start(self, tz: tzinfo) -> datetime:
        return datetime(self.year, self.month or 1, self.day or 1, tzinfo=tz)

    def end(self, tz: tzinfo) -> datetime:
        """Exclusive end: the first moment after the period."""
        if self.day is not None:
            nxt = datetime(self.year, self.month or 1, self.day) + _ONE_DAY
            return nxt.replace(tzinfo=tz)
        if self.month is not None:
            return datetime(self.year + (self.month == 12), self.month % 12 + 1, 1, tzinfo=tz)
        return datetime(self.year + 1, 1, 1, tzinfo=tz)

    @property
    def first_month(self) -> YearMonth:
        return self.year, self.month or 1

    @property
    def last_month(self) -> YearMonth:
        return self.year, self.month or 12


_ONE_DAY = timedelta(days=1)
_PERIOD_RE = re.compile(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?$")


def _period(text: str) -> Period:
    m = _PERIOD_RE.match(text.strip())
    bad = argparse.ArgumentTypeError(f"expected YYYY-MM (or YYYY, or YYYY-MM-DD), got {text!r}")
    if not m:
        raise bad
    year, month, day = (int(g) if g else None for g in m.groups())
    try:
        datetime(year, month or 1, day or 1)
    except ValueError:
        raise bad from None
    return Period(year, month, day)


def _timezone_name(text: str) -> str:
    if text.strip().upper() in ("UTC", "Z", "GMT"):
        return "UTC"
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(text.strip())
    except Exception:  # noqa: BLE001 — ZoneInfoNotFoundError, ValueError for odd strings
        hint = ""
        try:
            import tzdata  # noqa: F401
        except ImportError:
            hint = " (if the name is right, install the time zone database: pip install tzdata)"
        raise argparse.ArgumentTypeError(
            f"unknown time zone {text!r}: use an IANA name such as America/New_York or Europe/London{hint}"
        ) from None
    return text.strip()


def _tz(name: str) -> tzinfo:
    if name == "UTC":
        return timezone.utc
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _formats(text: str) -> tuple[str, ...]:
    from .report import _normalise_formats

    try:
        return tuple(_normalise_formats([f for f in text.split(",") if f.strip()]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _rules(text: str) -> Optional[list[str]]:
    values = [v.lower() for v in (_csv([text]) or [])]
    return None if not values or "all" in values else values


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="show debug logging")

    p = argparse.ArgumentParser(
        prog="chess-insights",
        description="Download your chess.com games and find your strengths, weaknesses and what to study.",
        parents=[common],
    )
    p.set_defaults(verbose=False)
    p.add_argument("--version", action="version", version=f"chess-insights {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{fetch,report,demo}")

    def add_source_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"game cache (default {DEFAULT_CACHE_DIR})")
        sp.add_argument("--since", type=_period, help="first month to include: YYYY-MM (or YYYY, or YYYY-MM-DD)")
        sp.add_argument("--until", type=_period, help="last month to include: YYYY-MM (or YYYY, or YYYY-MM-DD)")
        sp.add_argument("--contact", help="email/URL added to the User-Agent, as chess.com asks API users to do")

    f = sub.add_parser("fetch", parents=[common], help="download/refresh your game archive from chess.com")
    f.add_argument("username", help="your chess.com username (or profile URL)")
    add_source_args(f)
    f.add_argument("--refresh-all", action="store_true", help="re-check old months too (picks up new Game Review accuracies)")

    r = sub.add_parser("report", parents=[common], help="analyse your games and write an HTML/Markdown/JSON report")
    r.add_argument("username", help="your chess.com username (or profile URL)")
    add_source_args(r)
    r.add_argument("--offline", action="store_true", help="use the local cache only (no chess.com requests)")
    r.add_argument("--pgn", nargs="+", help="analyse PGN file(s) or folder(s) instead of the chess.com API")
    r.add_argument("--json", nargs="+", dest="json_files", help="analyse chess.com JSON archive file(s) or folder(s)")
    add_analysis_args(r)

    d = sub.add_parser("demo", parents=[common], help="generate a synthetic player with known weaknesses and analyse them")
    d.add_argument("--games", type=_positive_int, default=400, help="number of synthetic games (default 400)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--cache-dir", default=None, help="where to write the synthetic archive (default: a temporary folder)")
    d.add_argument("--no-engine-synth", action="store_true", help="generate games without Stockfish (faster, less realistic)")
    add_analysis_args(d, default_out="reports/demo")
    return p


def add_analysis_args(sp: argparse.ArgumentParser, default_out: Optional[str] = None) -> None:
    sp.add_argument(
        "--time-class", action="append", type=_time_classes, help="bullet, blitz, rapid, daily (repeat or comma-separate)"
    )
    sp.add_argument("--rules", default="chess", help="variant filter, default 'chess' (use 'all' for everything)")
    sp.add_argument("--rated-only", action="store_true", help="skip casual (unrated) games")
    sp.add_argument(
        "--tz", type=_timezone_name, default="UTC",
        help="IANA time zone, e.g. America/New_York: time-of-day stats and --since/--until boundaries (default UTC)",
    )
    sp.add_argument("--engine", action="store_true", help="run Stockfish analysis (slower, much deeper insights)")
    sp.add_argument("--stockfish", help="path to the Stockfish program or its folder (default: auto-detect)")
    sp.add_argument("--depth", type=_positive_int, default=12, help="Stockfish depth per position (default 12)")
    sp.add_argument("--engine-games", type=_positive_int, default=150, help="analyse the N most recent games (default 150)")
    sp.add_argument("--workers", type=_non_negative_int, default=0, help="parallel engine processes (default: CPUs - 1)")
    sp.add_argument("--puzzles", action="store_true", help="with --engine: export your mistakes as a PGN puzzle file")
    sp.add_argument("--out", default=default_out, help="output path stem (default reports/<username>)")
    sp.add_argument("--formats", type=_formats, default=("html", "md", "json"), help="comma list of html, md, json")


# --------------------------------------------------------------------------- helpers
def _username(text: str) -> str:
    from .api import normalize_username

    try:
        return normalize_username(text)
    except ValueError as exc:
        raise UserError(str(exc)) from None


def _archive_window(args: argparse.Namespace) -> tuple[Optional[YearMonth], Optional[YearMonth]]:
    """Archive months to fetch / load for --since / --until: one extra month either side.

    chess.com files a game under the month it ended in US time, so a game that ended early on
    the 1st (UTC) is in the previous month's archive; the exact date filter runs on end_time.
    """

    def shift(ym: YearMonth, k: int) -> YearMonth:
        n = ym[0] * 12 + ym[1] - 1 + k
        return n // 12, n % 12 + 1

    since = shift(args.since.first_month, -1) if args.since else None
    until = shift(args.until.last_month, +1) if args.until else None
    return since, until


def _check_period(args: argparse.Namespace) -> None:
    if args.since and args.until and args.since.start(timezone.utc) >= args.until.end(timezone.utc):
        raise UserError("--since must be before --until")


def _client(args: argparse.Namespace):
    from . import api

    return api.ChessComClient(contact=args.contact)


def _expand_inputs(paths: Iterable[str], suffix: str) -> list[Path]:
    """Files as given; a folder means every ``*<suffix>`` file in it. Missing paths are a usage error."""
    out: list[Path] = []
    for raw in paths:
        path = expand_path(raw)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() == suffix and p.is_file())
            if not found:
                raise UserError(f"no {suffix} files in folder {raw}")
            out.extend(found)
        elif path.is_file():
            out.append(path)
        else:
            raise UserError(f"file not found: {raw}")
    return out


def _players(pgn_texts: Iterable[str]) -> Counter:
    names: Counter = Counter()
    for text in pgn_texts:
        for chunk in split_pgn(text):
            h = parse_pgn_headers(chunk)
            names.update(n for n in (h.get("White"), h.get("Black")) if n)
    return names


def _raw_players(raws: Iterable[dict]) -> Counter:
    names: Counter = Counter()
    for raw in raws:
        for side in ("white", "black"):
            player = raw.get(side) if isinstance(raw, dict) else None
            name = player.get("username") if isinstance(player, dict) else None
            if name:
                names[str(name)] += 1
    return names


def _not_in_file(username: str, names: Counter, what: str) -> UserError:
    top = ", ".join(f"{n} ({c})" for n, c in names.most_common(5)) or "none"
    return UserError(f"none of the games in the {what} were played by {username!r}; players found: {top}", EXIT_NO_GAMES)


def _out_stem(args: argparse.Namespace, username: str) -> Path:
    if args.out:
        return expand_path(args.out)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", username.lower()).strip("._") or "report"
    return Path("reports") / safe


def resolve_stockfish_arg(value: Optional[str]) -> Optional[str]:
    """--stockfish value -> executable path, or None. A folder is searched for a stockfish* program."""
    from .engine import find_stockfish

    if not value:
        return find_stockfish(None)
    path = expand_path(value)
    if path.is_dir():
        cands = sorted(path.glob("stockfish*")) + sorted(path.glob("*/stockfish*"))
        for cand in cands:
            if cand.is_file() and (cand.suffix.lower() == ".exe" or not cand.suffix):
                found = find_stockfish(str(cand))
                if found:
                    return found
        return None
    return find_stockfish(str(path))


def _prepare_engine(args: argparse.Namespace) -> None:
    """Check engine options before any download or parsing, so mistakes surface in the first second."""
    args.stockfish_path = None
    if getattr(args, "puzzles", False) and not args.engine:
        _say("note: --puzzles needs --engine (puzzles are built from Stockfish's verdicts); no puzzle file will be written.")
    if not args.engine:
        return
    args.stockfish_path = resolve_stockfish_arg(args.stockfish)
    if args.stockfish and not args.stockfish_path:
        raise UserError(
            f"Stockfish not found at {args.stockfish!r}. Pass the Stockfish program itself or its folder "
            r"(on Windows: the .exe in the unzipped download, e.g. C:\Tools\stockfish\stockfish-windows-x86-64-avx2.exe)."
        )
    if not args.stockfish_path:
        _say(
            "warning: --engine was given but Stockfish was not found (install it, add it to PATH, or pass "
            "--stockfish PATH); continuing without engine analysis."
        )


# --------------------------------------------------------------------------- commands
def cmd_fetch(args: argparse.Namespace) -> int:
    username = _username(args.username)
    _check_period(args)
    since, until = _archive_window(args)
    client = _client(args)
    cache_dir = expand_path(args.cache_dir)
    s = sync(
        username,
        client,
        cache_dir,
        since=since,
        until=until,
        refresh_all=getattr(args, "refresh_all", False),
        progress=_say,
    )
    extra = f", {s.months_not_modified} unchanged" if s.months_not_modified else ""
    _say(
        f"done: {s.games_total} games in {s.months_listed} months "
        f"({s.months_downloaded} downloaded, {s.months_cached} cached{extra}, {s.requests} requests) -> {cache_dir}"
    )
    if s.months_missing:
        plural = "month" if s.months_missing == 1 else "months"
        _say(
            f"warning: {s.months_missing} {plural} could not be downloaded right now (chess.com answered 404); "
            "run the command again later to fill them in."
        )
    if s.months_listed == 0:
        _say("no monthly archives in the chosen period." if args.since or args.until else f"chess.com lists no games for {username!r} yet.")
    return 0


def _load_raw_games(args: argparse.Namespace) -> list[Game]:
    """Games from --pgn / --json / the chess.com cache (refreshed first unless --offline).

    Raises UserError(EXIT_NO_GAMES) with a specific explanation when there are none.
    """
    if args.pgn:
        files = _expand_inputs(args.pgn, ".pgn")
        texts = [read_text(p) for p in files]
        games = [g for text in texts for g in games_from_pgn(text, args.username)]
        if not games:
            raise _not_in_file(args.username, _players(texts), "PGN file(s)")
        return merge_games(games)  # repeats across files dropped, chronological
    if args.json_files:
        files = _expand_inputs(args.json_files, ".json")
        try:
            raws = load_json_files(files)
        except ValueError as exc:
            raise UserError(f"not a JSON file: {exc}") from None
        games = parse_games(raws, args.username)
        if not games:
            raise _not_in_file(args.username, _raw_players(raws), "JSON file(s)")
        return games

    username = _username(args.username)
    cache_dir = expand_path(args.cache_dir)
    since, until = _archive_window(args)
    if not args.offline:
        from .api import ChessComError, PlayerNotFound

        try:
            cmd_fetch(args)
        except PlayerNotFound:
            raise
        except ChessComError as exc:
            cached = load_games(username, cache_dir, since=since, until=until)
            if not cached:
                raise
            _say(f"warning: could not update from chess.com ({exc})")
            _say(f"continuing with the {len(cached)} games already downloaded to {cache_dir}")
    games = parse_games(load_games(username, cache_dir, since=since, until=until), username)
    if not games and (since or until):
        # nothing in the period: load everything so the filter step can say what there is instead
        games = parse_games(load_games(username, cache_dir), username)
    if not games:
        if args.offline:
            raise UserError(
                f"no games for {username!r} in the cache {cache_dir} "
                f"(run `chess-insights fetch {username}` first, or leave out --offline)",
                EXIT_NO_GAMES,
            )
        raise UserError(f"chess.com has no finished games for {username!r} in this period.", EXIT_NO_GAMES)
    return games


def _available(games: list[Game]) -> str:
    """What the unfiltered games contain, to explain an empty selection."""
    tc = Counter(g.time_class for g in games)
    rules = Counter(g.rules for g in games)
    first, last = min(g.end_time for g in games), max(g.end_time for g in games)
    parts = [
        "time classes: " + ", ".join(f"{k} {v}" for k, v in tc.most_common()),
        "variants: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()),
        f"rated {sum(g.rated for g in games)}, casual {sum(not g.rated for g in games)}",
        f"played {first:%Y-%m-%d} to {last:%Y-%m-%d} (UTC)",
    ]
    return "; ".join(parts)


def analyse_and_write(games: list[Game], username: str, args: argparse.Namespace, cache_dir: Optional[Path]) -> Report:
    from .pipeline import run_analysis
    from .report import write_report

    tz = _tz(args.tz)
    time_classes = _csv(args.time_class)
    rules = _rules(args.rules)
    since = args.since.start(tz) if getattr(args, "since", None) else None
    until = args.until.end(tz) if getattr(args, "until", None) else None
    rated = True if args.rated_only else None
    selected = filter_games(games, time_classes=time_classes, rules=rules, rated=rated, since=since, until=until)
    filters = describe_filters(time_classes=time_classes, rules=rules, rated=rated, since=since, until=until)
    _say(f"{len(selected)} of {len(games)} games match: {filters}")
    if not selected:
        raise UserError(f"nothing to analyse. Your games have {_available(games)}.", EXIT_NO_GAMES)

    evals: dict = {}
    engine_note = "Engine analysis not run (add --engine for accuracy, blunder and phase insights)."
    if args.engine:
        evals, engine_note = run_engine(selected, username, args, cache_dir)

    report = run_analysis(
        selected, username, evals=evals, options={"tz": args.tz}, filters=filters, engine_note=engine_note
    )
    out = _out_stem(args, username)
    try:
        paths = write_report(report, out, formats=args.formats)
        if getattr(args, "puzzles", False) and evals:
            paths.append(write_puzzles(selected, evals, out))
    except OSError as exc:
        raise UserError(f"could not write the report to {out}: {exc}", EXIT_NO_GAMES) from None
    print_summary(report)
    for p in paths:
        _say(f"wrote {p}")
    return report


def _puzzle_path(out: Path) -> Path:
    out = Path(out)
    if out.suffix.lower() in (".html", ".htm", ".md", ".markdown", ".json"):
        out = out.with_suffix("")
    return out.parent / f"{out.name}-puzzles.pgn"


def write_puzzles(games: list[Game], evals: dict, out: Path) -> Path:
    from .analysis.mistakes import build_puzzles, puzzles_to_pgn

    path = _puzzle_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    puzzles = build_puzzles(games, evals)
    path.write_text(puzzles_to_pgn(puzzles), encoding="utf-8")
    _say(f"{len(puzzles)} puzzles from your own mistakes (import into a Lichess study or any chess GUI)")
    return path


def run_engine(games: list[Game], username: str, args: argparse.Namespace, cache_dir: Optional[Path]):
    from .engine import EngineConfig, analyze_games, engine_name

    path = getattr(args, "stockfish_path", None) or resolve_stockfish_arg(args.stockfish)
    if not path:
        return {}, "Engine analysis skipped: Stockfish not found (install it or pass --stockfish PATH)."
    cfg = EngineConfig(path=path, depth=args.depth, workers=args.workers)
    eval_dir = expand_path(cache_dir) / username.lower() / "evals" if cache_dir else None
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            _say(f"  engine: {done}/{total} games ({time.time() - t0:.0f}s)")

    evals = analyze_games(games, cfg, cache_dir=eval_dir, max_games=args.engine_games, progress=progress)
    note = f"{engine_name(path)} at depth {args.depth} on the {len(evals)} most recent games."
    return evals, note


def print_summary(report: Report) -> None:
    out = sys.stdout
    print(f"\n== {report.username}: {report.n_games} games ({report.filters}) ==", file=out)
    if report.headline:
        print(report.headline, file=out)
    if report.strengths:
        print("\nStrengths", file=out)
        for ins in report.strengths:
            print(f"  + {ins.title}", file=out)
    if report.weaknesses:
        print("\nWeaknesses", file=out)
        for ins in report.weaknesses:
            print(f"  - {ins.title}", file=out)
    if report.study_plan:
        print("\nStudy plan", file=out)
        for i, item in enumerate(report.study_plan, 1):
            print(f"  {i}. {item.title}", file=out)
            for action in item.actions[:3]:
                print(f"       * {action}", file=out)
    print("", file=out, flush=True)


def _display_name(games: list[Game], fallback: str) -> str:
    """The player's name as chess.com spells it (display case), from the most recent game."""
    return games[-1].username if games and games[-1].username else fallback


def cmd_report(args: argparse.Namespace) -> int:
    _check_period(args)
    _prepare_engine(args)
    games = _load_raw_games(args)
    username = _display_name(games, args.username)
    cache_dir = expand_path(args.cache_dir) if args.cache_dir else None
    analyse_and_write(games, username, args, cache_dir)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .synth import DEFAULT_PERSONA, PLANTED_TRAITS, generate_archives, write_archives

    _prepare_engine(args)
    args.since = args.until = None
    temp = args.cache_dir is None
    cache_dir = Path(tempfile.mkdtemp(prefix="chess-insights-demo-")) if temp else expand_path(args.cache_dir)
    try:
        persona = DEFAULT_PERSONA
        _say(f"generating {args.games} synthetic games for {persona.username!r} (seed {args.seed}) ...")
        t0 = time.time()
        archives = generate_archives(
            args.games,
            seed=args.seed,
            persona=persona,
            engine_path=None if args.no_engine_synth else "auto",
            workers=args.workers,
            progress=lambda done, total: (done == total or done % 50 == 0) and _say(f"  {done}/{total} games"),
        )
        write_archives(archives, cache_dir, persona.username)
        _say(f"generated in {time.time() - t0:.0f}s" + ("" if temp else f" -> {cache_dir}"))

        games = parse_games(load_games(persona.username, cache_dir), persona.username)
        report = analyse_and_write(games, persona.username, args, cache_dir)
    finally:
        if temp:
            shutil.rmtree(cache_dir, ignore_errors=True)

    hits = check_planted_traits(report, PLANTED_TRAITS)
    print("Planted traits vs. what the analysis found:")
    for trait, match in hits:
        mark = "FOUND " if match else "missed"
        print(f"  [{mark}] ({trait['expect']}) {trait['description']}")
        if match:
            print(f"           -> {match.title}")
    print(f"\n{sum(1 for _, m in hits if m)}/{len(hits)} planted traits recovered.")
    return 0


def check_planted_traits(report: Report, traits: list[dict]) -> list[tuple[dict, Optional[object]]]:
    """Match each planted trait to the best insight of the right kind/category mentioning one of its keywords."""
    from .insights import all_insights

    pool = sorted(all_insights(report.modules), key=lambda i: -i.priority)
    out = []
    for trait in traits:
        keywords = [k.lower() for k in trait.get("keywords", [])]
        match = next(
            (
                i
                for i in pool
                if i.kind == trait["expect"]
                and (not trait.get("category") or i.category == trait["category"])
                and (not keywords or any(k in f"{i.id} {i.title} {i.detail}".lower() for k in keywords))
            ),
            None,
        )
        out.append((trait, match))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    _safe_console()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # usage errors (2), --help / --version (0): return instead of exiting
        return exc.code if isinstance(exc.code, int) else 1
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return {"fetch": cmd_fetch, "report": cmd_report, "demo": cmd_demo}[args.command](args)
    except KeyboardInterrupt:
        _say("interrupted")
        return 130
    except UserError as exc:
        _say(f"error: {exc}")
        return exc.code
    except Exception as exc:  # noqa: BLE001 — friendly top-level error
        from .api import ChessComError, PlayerNotFound

        if isinstance(exc, PlayerNotFound):
            name = getattr(args, "username", "")
            if "closed" in str(exc):
                _say(f"error: {exc}")
            else:
                _say(
                    f"error: chess.com has no player {name!r}. Check the spelling: use the name from your "
                    "profile address, chess.com/member/NAME."
                )
            return EXIT_USAGE
        if isinstance(exc, ChessComError):
            _say(f"error: chess.com request failed: {exc}")
            _say("(see Troubleshooting in the README; --offline uses games downloaded earlier)")
            return EXIT_NETWORK
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
