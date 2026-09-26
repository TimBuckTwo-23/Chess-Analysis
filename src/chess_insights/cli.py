"""Command line entry point: ``chess-insights fetch | report | demo | puzzles-db | ask``.

Exit codes: 0 success; 1 nothing to analyse (no games, no games in the filter window), the
report / game cache could not be written, or no earlier report to answer from (``ask``); 2 usage
error (bad option, unknown chess.com player, missing input file or Stockfish); 3 chess.com (or, for
``puzzles-db``, database.lichess.org) could not be reached or refused the request; 130 Ctrl+C.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing
import os
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
from .coach.config import (
    DEFAULT_COACH_DEPTH,
    DEFAULT_COACH_MAX,
    DEFAULT_DRILL_RATING,
    DEFAULT_LLM_MODEL,
    DEFAULT_PRACTICE_MINUTES,
    CoachConfig,
)
from .engine import ENGINE_SAMPLES
from .fetch import (
    DEFAULT_CACHE_DIR,
    GameStore,
    expand_path,
    load_games,
    load_json_files,
    month_key,
    read_text,
    safe_file_stem,
    sync,
)
from .models import TIME_CLASSES, Coaching, Game, Report
from .parse import parse_games, parse_games_from_pgns, parse_pgn_headers, split_pgn

EXIT_NO_GAMES = 1
EXIT_USAGE = 2
EXIT_NETWORK = 3

YearMonth = tuple[int, int]


class UserError(Exception):
    """A problem the user can fix; printed as one friendly line (no traceback)."""

    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


class CacheWriteError(UserError):
    """The game cache folder could not be written (not a folder, read-only, disk full, locked ...)."""

    def __init__(self, cache_dir: Path, exc: OSError) -> None:
        super().__init__(
            f"could not write to the game cache {cache_dir}: {exc}. Choose another folder with --cache-dir.",
            EXIT_NO_GAMES,
        )


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
        raise argparse.ArgumentTypeError(f"unknown time class {', '.join(bad)}; choose from {', '.join(TIME_CLASSES)}")
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


_ONE_DAY = timedelta(days=1)
_PERIOD_RE = re.compile(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?$")


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


def _period(text: str) -> Period:
    m = _PERIOD_RE.match(text.strip())
    bad = argparse.ArgumentTypeError(f"expected YYYY-MM (or YYYY, or YYYY-MM-DD), got {text!r}")
    if not m:
        raise bad
    year, month, day = (int(g) if g else None for g in m.groups())
    if not 1000 <= year <= 9998:  # Period.end() of 9999 would be year 10000, which datetime can't hold
        raise argparse.ArgumentTypeError(f"year out of range in {text!r} (1000 to 9998)")
    try:
        datetime(year, month or 1, day or 1)
    except ValueError:
        raise bad from None
    return Period(year, month, day)


def _timezone_name(text: str) -> str:
    if text.strip().upper() in ("UTC", "Z", "GMT"):
        return "UTC"
    import zoneinfo

    try:
        zoneinfo.ZoneInfo(text.strip())
    except Exception:  # noqa: BLE001 — ZoneInfoNotFoundError, ValueError for odd strings
        # Windows has no system time zone database: Python needs the tzdata package there.
        hint = "" if zoneinfo.available_timezones() else " (no time zone database found: pip install tzdata)"
        raise argparse.ArgumentTypeError(
            f"unknown time zone {text!r}: use an IANA name such as America/New_York or Europe/London{hint}"
        ) from None
    return text.strip()


def _tz(name: Optional[str]) -> tzinfo:
    if name is None or name == "UTC":  # unknown time zone: date boundaries fall back to UTC
        return timezone.utc
    from zoneinfo import ZoneInfo

    return ZoneInfo(name)


def _engine_games(text: str) -> int:
    """--engine-games: a positive number, or "all"."""
    if text.strip().lower() == "all":
        return 10**9
    return _positive_int(text)


def _formats(text: str) -> tuple[str, ...]:
    from .report import _normalise_formats

    try:
        formats = tuple(_normalise_formats([f for f in text.split(",") if f.strip()]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not formats:  # --formats "" would run the whole analysis and write nothing
        raise argparse.ArgumentTypeError("no report format given; choose from html, md, json")
    return formats


_RATING_RANGE_RE = re.compile(r"^\s*(\d{3,4})\s*[-:]\s*(\d{3,4})\s*$")


def _drill_rating(text: str) -> tuple[int, int]:
    """--drill-rating: a puzzle rating window such as 1200-1600."""
    m = _RATING_RANGE_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected a rating range such as 1200-1600, got {text!r}")
    low, high = int(m.group(1)), int(m.group(2))
    if not 400 <= low < high <= 3500:
        raise argparse.ArgumentTypeError(f"expected two ratings from 400 to 3500, the lower first, got {text!r}")
    return low, high


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
    sub = p.add_subparsers(dest="command", required=True, metavar="{fetch,report,demo,puzzles-db,ask}")

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

    pz = sub.add_parser(
        "puzzles-db", parents=[common],
        help="download the Lichess puzzle database (CC0) once and keep the subset the drills use (--coach)",
    )
    pz.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"cache folder (default {DEFAULT_CACHE_DIR})")

    a = sub.add_parser("ask", parents=[common], help="ask a question about your last report (needs ANTHROPIC_API_KEY)")
    a.add_argument("username", help="your chess.com username (the report's name)")
    a.add_argument("question", help='your question in quotes, e.g. "why do I lose with the Alapin?"')
    a.add_argument("--out", default=None, help="the report's path stem, as given to `report --out` (default reports/<username>)")
    a.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help=f"cache folder (default {DEFAULT_CACHE_DIR})")
    a.add_argument("--offline", action="store_true", help="no Lichess or Wikibooks requests")
    a.add_argument("--llm-model", default=None, help="Claude model (default: the coaching layer's)")
    return p


def add_analysis_args(sp: argparse.ArgumentParser, default_out: Optional[str] = None) -> None:
    sp.add_argument(
        "--time-class", action="append", type=_time_classes, help="bullet, blitz, rapid, daily (repeat or comma-separate)"
    )
    sp.add_argument("--rules", default="chess", help="variant filter, default 'chess' (use 'all' for everything)")
    sp.add_argument("--rated-only", action="store_true", help="skip casual (unrated) games")
    sp.add_argument(
        "--tz", type=_timezone_name, default=None,
        help="your IANA time zone, e.g. America/New_York (recommended): local time-of-day stats, the late-night "
        "check and --since/--until boundaries. Without it, times are shown in UTC and tested more strictly",
    )
    sp.add_argument("--engine", action="store_true", help="run Stockfish analysis (slower, much deeper insights)")
    sp.add_argument("--stockfish", help="path to the Stockfish program or its folder (default: auto-detect)")
    sp.add_argument("--depth", type=_positive_int, default=12, help="Stockfish depth per position (default 12)")
    sp.add_argument(
        "--engine-games", type=_engine_games, default=150,
        help="analyse the N most recent games (default 150), or 'all'",
    )
    sp.add_argument(
        "--engine-time-class", action="append", type=_time_classes, metavar="TIME_CLASS",
        help="send only these formats to Stockfish (bullet, blitz, rapid, daily); the report still uses every game",
    )
    sp.add_argument(
        "--engine-sample", choices=ENGINE_SAMPLES, default="recent",
        help="which games Stockfish analyses: the most recent (default), or 'balanced': the same number of each "
        "format (bullet, blitz, rapid; daily only with --engine-time-class), most recent in each",
    )
    sp.add_argument("--workers", type=_non_negative_int, default=0, help="parallel engine processes (default: CPUs - 1)")
    sp.add_argument("--puzzles", action="store_true", help="with --engine: export your mistakes as a PGN puzzle file")

    c = sp.add_argument_group("coaching (with --engine)")
    c.add_argument("--coach", action="store_true",
                   help="explain your costliest and repeated mistakes, and add puzzle drills and a review schedule")
    c.add_argument("--coach-depth", type=_positive_int, metavar="N", default=DEFAULT_COACH_DEPTH,
                   help=f"Stockfish depth for the positions the coaching explains (default {DEFAULT_COACH_DEPTH})")
    c.add_argument("--coach-max", type=_positive_int, metavar="N", default=DEFAULT_COACH_MAX,
                   help=f"explain at most N positions, costliest first (default {DEFAULT_COACH_MAX})")
    c.add_argument("--no-motif-profile", action="store_true",
                   help="skip the tactical-motif profile over every error (saves a quick engine pass)")
    c.add_argument("--lichess-token", default=None, metavar="TOKEN",
                   help="Lichess personal access token for the opening explorer (default: the LICHESS_TOKEN "
                   "environment variable)")
    c.add_argument("--puzzle-db", default=None, metavar="PATH",
                   help="the filtered Lichess puzzle file for drills (default: the one `chess-insights puzzles-db` "
                   "keeps in the cache folder)")
    c.add_argument("--drill-rating", type=_drill_rating, metavar="LO-HI", default=DEFAULT_DRILL_RATING,
                   help=f"puzzle rating window for drills, e.g. 1300-1700 (default {DEFAULT_DRILL_RATING[0]}-"
                   f"{DEFAULT_DRILL_RATING[1]})")
    c.add_argument("--practice-minutes", type=_positive_int, metavar="MIN", default=DEFAULT_PRACTICE_MINUTES,
                   help=f"minutes of practice a day the plan is sized for (default {DEFAULT_PRACTICE_MINUTES})")
    c.add_argument("--coach-llm", action="store_true",
                   help="let Claude rewrite the explanations and draft a weekly plan, every move and number checked "
                   "(needs ANTHROPIC_API_KEY)")
    c.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, metavar="MODEL", help=f"Claude model for --coach-llm (default {DEFAULT_LLM_MODEL})")
    c.add_argument("--maia", action="store_true",
                   help="how findable the better move was at your level (needs pip install maia2)")
    c.add_argument("--previous", default=None, metavar="PATH",
                   help="the previous report's JSON, for progress and puzzles due for review (default: the JSON "
                   "report already at the output path)")
    sp.add_argument(
        "--out", default=default_out,
        help="output path stem, e.g. reports/me writes reports/me.html ...; an existing folder (or a path "
        "ending in / or \\) gets <username>.html inside it (default reports/<username>)",
    )
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

    return api.ChessComClient(contact=args.contact, notify=lambda msg: _say(f"  {msg}"))


_WILDCARD = re.compile(r"[*?[]")


def _expand_inputs(paths: Iterable[str], suffix: str) -> list[Path]:
    """Files as given; a folder means every ``*<suffix>`` file in it; a wildcard (``C:\\games\\*.pgn``:
    cmd.exe and PowerShell pass it unexpanded) is expanded here. Missing paths are a usage error."""
    out: list[Path] = []
    for raw in paths:
        path = expand_path(raw)
        if not path.exists() and _WILDCARD.search(str(path)):
            matches = [Path(m) for m in sorted(glob.glob(str(path)))]
            files = [m for m in matches if m.is_file()]
            for folder in (m for m in matches if m.is_dir()):
                files.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() == suffix and p.is_file()))
            if not files:
                raise UserError(f"no files match {raw}")
            out.extend(files)
        elif path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() == suffix and p.is_file())
            if not found:
                raise UserError(f"no {suffix} files in folder {raw}")
            out.extend(found)
        elif path.is_file():
            out.append(path)
        else:
            raise UserError(f"file not found: {raw} (give a {suffix} file, a folder of them, or a wildcard like *{suffix})")
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


_SEPARATORS = tuple(sep for sep in (os.sep, os.altsep) if sep)  # "\\" and "/" on Windows, "/" elsewhere


def _out_stem(args: argparse.Namespace, username: str) -> Path:
    """--out as a path stem; an existing folder, or a path ending in a slash, gets ``<username>.*`` inside it."""
    default_name = safe_file_stem(username.lower())
    if args.out:
        out = expand_path(args.out)
        if str(args.out).rstrip('"').endswith(_SEPARATORS) or out.is_dir():
            return out / default_name
        return out
    return Path("reports") / default_name


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
    if getattr(args, "coach", False) and not args.engine:
        _say("note: --coach needs --engine (the coaching explains Stockfish's verdicts); the report has no coaching.")
    elif getattr(args, "coach_llm", False) and not getattr(args, "coach", False):
        _say("note: --coach-llm works with --coach; no LLM coaching.")
    if getattr(args, "coach", False) and args.engine and getattr(args, "coach_llm", False):
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            _say("note: --coach-llm needs ANTHROPIC_API_KEY in the environment; the explanations use the built-in "
                 "wording.")
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
    try:
        s = sync(
            username,
            client,
            cache_dir,
            since=since,
            until=until,
            refresh_all=getattr(args, "refresh_all", False),
            progress=_say,
        )
    except OSError as exc:
        raise CacheWriteError(cache_dir, exc) from None
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
    if args.pgn or args.json_files:
        from .api import normalize_username

        try:  # "@Name" / a pasted profile URL; other sites' names are used as given
            name = normalize_username(args.username)
        except ValueError:
            name = args.username.strip()
    if args.pgn:
        files = _expand_inputs(args.pgn, ".pgn")
        texts = [read_text(p) for p in files]
        # one pass over all files: repeats across files dropped, chronological
        games = parse_games_from_pgns([chunk for text in texts for chunk in split_pgn(text)], name)
        if not games:
            raise _not_in_file(args.username, _players(texts), "PGN file(s)")
        return games
    if args.json_files:
        raws = []
        for path in _expand_inputs(args.json_files, ".json"):
            try:
                raws.extend(load_json_files([path]))
            except ValueError as exc:
                raise UserError(f"{path} is not valid JSON ({exc})") from None
        games = parse_games(raws, name)
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
        except (ChessComError, CacheWriteError) as exc:
            # a closed or renamed account, no internet, an unwritable cache: games downloaded earlier still count
            cached = load_games(username, cache_dir, since=since, until=until)
            if not cached:
                raise
            if isinstance(exc, CacheWriteError):
                _say(f"warning: {exc}")
            else:
                if isinstance(exc, PlayerNotFound) and "closed" not in str(exc):
                    exc = f"chess.com no longer has a player {username!r}"
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

    if args.tz is None and not getattr(args, "demo", False):
        _say(
            "tip: add --tz with your time zone (e.g. --tz America/New_York) for time-of-day results in your local "
            "time and the late-night check"
        )
    evals: dict = {}
    engine_note = "Engine analysis not run (add --engine for accuracy, blunder and phase insights)."
    if args.engine:
        evals, engine_note = run_engine(selected, username, args, cache_dir)

    out = _out_stem(args, username)
    options = {"tz": args.tz, "engine_sample": getattr(args, "engine_sample", "recent")}
    if getattr(args, "puzzles", False) and evals:
        options["puzzle_file"] = _puzzle_path(out).name  # written next to the report, after it
    if getattr(args, "demo", False):
        options["demo"] = True
    coach = coach_config(args, username, cache_dir, out)
    report = run_analysis(
        selected, username, evals=evals, options=options, filters=filters, engine_note=engine_note, coach=coach
    )
    try:
        paths = write_report(report, out, formats=args.formats)
        if getattr(args, "puzzles", False) and evals:
            paths.append(write_puzzles(selected, evals, out, report.coaching))
    except OSError as exc:
        raise UserError(f"could not write the report to {out}: {exc}", EXIT_NO_GAMES) from None
    print_summary(report)
    for p in paths + _drill_files(report.coaching):
        _say(f"wrote {_absolute(p)}")
    return report


def _drill_files(coaching: Optional[Coaching]) -> list[Path]:
    """The puzzle packs the coaching wrote next to the report."""
    files = [Path(d.file) for d in (coaching.drills if coaching else []) if d.file]
    return [f for f in files if f.exists()]


def _stem(out: Path) -> Path:
    """The report path without a report-type suffix: reports/me.html -> reports/me."""
    out = Path(out)
    return out.with_suffix("") if out.suffix.lower() in (".html", ".htm", ".md", ".markdown", ".json") else out


def _read_report_json(path: Path) -> Optional[dict]:
    """A report's JSON as a dict, or None (with a warning) when it can't be read."""
    try:
        data = json.loads(read_text(path))
    except (OSError, ValueError) as exc:
        _say(f"warning: could not read the previous report {path} ({exc}); no progress comparison this time.")
        return None
    if not isinstance(data, dict):
        _say(f"warning: {path} is not a chess-insights JSON report; no progress comparison this time.")
        return None
    return data


def _puzzle_db(args: argparse.Namespace, cache_dir: Optional[Path]) -> Optional[Path]:
    """--puzzle-db, else the subset `chess-insights puzzles-db` keeps in the cache folder (when it exists)."""
    if getattr(args, "puzzle_db", None):
        path = expand_path(args.puzzle_db)
        if not path.is_file():
            _say(f"warning: no puzzle file at {path}; the drills fall back to Lichess's theme pages.")
            return None
        return path
    from .coach.puzzles_db import subset_path

    for folder in dict.fromkeys(f for f in (cache_dir, expand_path(str(DEFAULT_CACHE_DIR))) if f is not None):
        candidate = subset_path(folder)
        if candidate.is_file():
            return candidate
    return None


def coach_config(
    args: argparse.Namespace, username: str, cache_dir: Optional[Path], out: Path
) -> Optional[CoachConfig]:
    """The coaching settings from the command line (None without --coach and --engine).

    Keys come from the environment (LICHESS_TOKEN, ANTHROPIC_API_KEY) unless given; they are never printed.
    The previous report (--previous, default the JSON already at the output path) is read here, before the new
    report overwrites it.
    """
    if not (getattr(args, "coach", False) and getattr(args, "engine", False)):
        return None
    user = safe_file_stem(username.lower())
    stem = _stem(out)
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    previous_path = expand_path(args.previous) if getattr(args, "previous", None) else stem.parent / f"{stem.name}.json"
    previous = None
    if previous_path.is_file():
        previous = _read_report_json(previous_path)
    elif getattr(args, "previous", None):
        _say(f"warning: no previous report at {previous_path}; no progress comparison this time.")
    return CoachConfig(
        stockfish=getattr(args, "stockfish_path", None),
        depth=args.coach_depth,
        max_positions=args.coach_max,
        workers=args.workers,
        cache_dir=cache_dir / user / "coach" if cache_dir else None,
        sources_cache=cache_dir / "sources" if cache_dir else None,
        profile=not args.no_motif_profile,
        offline=bool(getattr(args, "offline", False)),
        lichess_token=(args.lichess_token or os.environ.get("LICHESS_TOKEN", "").strip() or None),
        puzzle_db=_puzzle_db(args, cache_dir),
        drill_rating=tuple(args.drill_rating),  # type: ignore[arg-type]
        out_stem=stem,
        practice_minutes=args.practice_minutes,
        llm=bool(args.coach_llm and key),
        llm_model=args.llm_model,
        anthropic_api_key=key if args.coach_llm else None,
        maia=args.maia,
        previous=previous,
    )


def _absolute(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _puzzle_path(out: Path) -> Path:
    out = _stem(out)
    return out.parent / f"{out.name}-puzzles.pgn"


def write_puzzles(games: list[Game], evals: dict, out: Path, coaching: Optional[Coaching] = None) -> Path:
    """The --puzzles PGN; with the coaching, each puzzle gets the engine's line, its motifs and the explanation."""
    from .analysis.mistakes import LICHESS_STUDY_CHAPTERS, build_puzzles, puzzles_to_pgn

    path = _puzzle_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    puzzles = build_puzzles(games, evals)
    text = None
    if coaching is not None:
        try:
            from .coach.puzzles import puzzles_pgn

            text = puzzles_pgn(puzzles, coaching)
        except Exception as exc:  # noqa: BLE001 — the plain puzzle file is still worth writing
            logging.getLogger(__name__).warning("puzzle file without the coaching's lines: %s", exc)
    path.write_text(text if text is not None else puzzles_to_pgn(puzzles), encoding="utf-8")
    note = "import into a Lichess study or any chess GUI"
    if len(puzzles) > LICHESS_STUDY_CHAPTERS:
        note += (
            f"; a Lichess study holds {LICHESS_STUDY_CHAPTERS} chapters, and the file is sorted costliest first, "
            f"so the first {LICHESS_STUDY_CHAPTERS} are the ones to start with"
        )
    _say(f"{len(puzzles)} puzzles from your own mistakes ({note})")
    return path


def _joined(words: Sequence[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + f" and {words[-1]}"


def describe_engine_sample(
    engine: str, depth: int, games: Sequence[Game], evals: dict, sample: str = "recent",
    time_classes: Optional[Sequence[str]] = None,
) -> str:
    """"Stockfish 16 at depth 12 on 300 games: 100 bullet, 100 blitz, 100 rapid (most recent in each)."."""
    from .visuals import format_counts

    counts = format_counts(g for g in games if g.game_id in evals)
    n = sum(counts.values())
    if not n:
        return (f"{engine} at depth {depth}: none of the selected games could be analysed (the engine needs "
                "games of standard chess or Chess960 with at least 10 moves).")
    mix = f"all {next(iter(counts))}" if len(counts) == 1 else ", ".join(f"{k} {tc}" for tc, k in counts.items())
    if sample == "balanced":
        return f"{engine} at depth {depth} on {n} games: {mix} (most recent in each)."
    scope = ""
    if time_classes:
        scope = " " + _joined([tc for tc in ("bullet", "blitz", "rapid", "daily") if tc in set(time_classes)])
    return f"{engine} at depth {depth} on the {n} most recent{scope} games: {mix}."


def run_engine(games: list[Game], username: str, args: argparse.Namespace, cache_dir: Optional[Path]):
    from .engine import EngineConfig, analyze_games, engine_name

    path = getattr(args, "stockfish_path", None) or resolve_stockfish_arg(args.stockfish)
    if not path:
        return {}, "Engine analysis skipped: Stockfish not found (install it or pass --stockfish PATH)."
    cfg = EngineConfig(path=path, depth=args.depth, workers=args.workers)
    eval_dir = expand_path(cache_dir) / safe_file_stem(username.lower()) / "evals" if cache_dir else None
    time_classes = _csv(getattr(args, "engine_time_class", None))
    sample = getattr(args, "engine_sample", "recent")
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            _say(f"  engine: {done}/{total} games ({time.time() - t0:.0f}s)")

    evals = analyze_games(
        games, cfg, cache_dir=eval_dir, max_games=args.engine_games, progress=progress,
        time_classes=time_classes, sample=sample,
    )
    return evals, describe_engine_sample(engine_name(path), args.depth, games, evals, sample, time_classes)


def print_summary(report: Report) -> None:
    out = sys.stdout
    print(f"\n== {report.username}: {report.n_games} games ({report.filters}) ==", file=out)
    for line in report.summary_lines or ([report.headline] if report.headline else []):
        print(line, file=out)
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
    if len(report.formats) > 1:
        mix = ", ".join(f"{n} {tc}" for tc, n in report.formats.items())
        views = ", ".join(report.format_reports) or "none (each format needs more games)"
        print(f"\nFormats: {mix}. Separate views in the report: {views}.", file=out)
    if report.coaching is not None:
        c = report.coaching
        parts = [f"{len(c.explanations)} positions explained"]
        if c.drills:
            parts.append(f"{len(c.drills)} puzzle packs")
        if c.review_due:
            parts.append(f"{len(c.review_due)} puzzles due for review")
        print(f"\nCoaching: {', '.join(parts)}.", file=out)
        for note in c.notes[:5]:
            print(f"  ({note})", file=out)
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
    try:
        cache_dir = Path(tempfile.mkdtemp(prefix="chess-insights-demo-")) if temp else expand_path(args.cache_dir)
    except OSError as exc:
        raise CacheWriteError(Path(tempfile.gettempdir()), exc) from None
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
        store = GameStore(cache_dir, persona.username)
        try:
            for ym in store.months():  # months left in this folder by an earlier demo run
                if month_key(ym) not in archives:
                    store.month_path(ym).unlink()
            write_archives(archives, cache_dir, persona.username)
        except OSError as exc:
            raise CacheWriteError(cache_dir, exc) from None
        _say(f"generated in {time.time() - t0:.0f}s" + ("" if temp else f" -> {cache_dir}"))

        games = parse_games([g for key in sorted(archives) for g in archives[key]], persona.username)
        if args.tz is None:
            args.tz = "Etc/UTC"  # the synthetic player's clock times are UTC
        args.demo = True
        report = analyse_and_write(games, persona.username, args, cache_dir)
    finally:
        if temp:
            shutil.rmtree(cache_dir, ignore_errors=True)

    print_trait_check(report, PLANTED_TRAITS)
    return 0


def print_trait_check(report: Report, traits: list[dict]) -> None:
    """Planted traits found (in the report's lists, or only in a section), missed or not testable; then the
    claims that match no planted trait."""
    engine_ran = any(m.key == "engine_stats" and m.stats.get("games") for m in report.modules)
    listed = {id(i) for i in list(report.strengths or []) + list(report.weaknesses or [])}
    hits = check_planted_traits(report, traits)
    print("Planted traits vs. what the analysis found:")
    tested = found = 0
    for trait, match in hits:
        if match is None and trait.get("needs_engine") and not engine_ran:
            mark = "not tested: needs --engine"
        else:
            tested += 1
            found += match is not None
            mark = "missed" if match is None else "FOUND" if id(match) in listed else "FOUND, section only"
        print(f"  [{mark}] ({trait['expect']}) {trait['description']}")
        if match is not None:
            print(f"           -> {match.title}")
    print(f"\n{found}/{tested} planted traits recovered" + (f" ({len(hits) - tested} not tested)." if tested < len(hits) else "."))
    unplanted = [
        i for i in list(report.weaknesses or []) + list(report.strengths or []) if not any(trait_matches(t, i) for t in traits)
    ]
    if unplanted:
        print("Claims that match no planted trait (side effects of how the games were generated, or chance: about")
        print("0.2 false claims per report are expected on players without real strengths or weaknesses):")
        for ins in unplanted:
            print(f"  - ({ins.kind}) {ins.title}")


def check_planted_traits(report: Report, traits: list[dict]) -> list[tuple[dict, Optional[object]]]:
    """Match each planted trait to the best insight of the right kind/category mentioning one of its keywords."""
    from .insights import all_insights

    pool = sorted(all_insights(report.modules), key=lambda i: -i.priority)
    return [(trait, next((i for i in pool if trait_matches(trait, i)), None)) for trait in traits]


def trait_matches(trait: dict, ins: object) -> bool:
    """An insight of the trait's kind and category that mentions one of its keywords."""
    keywords = [k.lower() for k in trait.get("keywords", [])]
    return (
        getattr(ins, "kind", None) == trait["expect"]
        and (not trait.get("category") or getattr(ins, "category", None) == trait["category"])
        and (
            not keywords
            or any(k in f"{getattr(ins, 'id', '')} {getattr(ins, 'title', '')} {getattr(ins, 'detail', '')}".lower() for k in keywords)
        )
    )


def cmd_puzzles_db(args: argparse.Namespace) -> int:
    """Download the Lichess puzzle database once and keep the filtered subset the drills read."""
    from .coach import puzzles_db

    cache_dir = expand_path(args.cache_dir)
    _say(f"downloading the Lichess puzzle database (CC0) and keeping the drill subset in {cache_dir} ...")
    last = [time.time()]

    def progress(n: int) -> None:
        if time.time() - last[0] >= 10:
            last[0] = time.time()
            _say(f"  {n:,} ...")

    try:
        path = puzzles_db.download(cache_dir, progress=progress)
    except KeyboardInterrupt:
        raise
    except NotImplementedError:
        raise UserError("the puzzle database download is not available in this version yet.", EXIT_NO_GAMES) from None
    except Exception as exc:  # noqa: BLE001 — one friendly line; the report works without the database
        import requests

        network = isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError))
        where = "could not download the Lichess puzzle database" if network else "could not build the puzzle subset"
        raise UserError(
            f"{where} ({type(exc).__name__}: {exc}). Reports still work: the drills link to Lichess's puzzle "
            "themes instead.", EXIT_NETWORK if network else EXIT_NO_GAMES,
        ) from None
    _say(f"wrote {_absolute(Path(path))}")
    return 0


def _report_json_path(args: argparse.Namespace, username: str) -> Path:
    """Where `report` wrote this player's JSON: --out's stem (a folder gets <username>.json), else reports/."""
    stem = _stem(_out_stem(args, username))
    return stem.parent / f"{stem.name}.json"


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question from the last report's JSON (through the coaching layer's verifier)."""
    from .api import normalize_username
    from .coach import ask

    try:
        name = normalize_username(args.username)
    except ValueError:
        name = args.username.strip()
    path = _report_json_path(args, name)
    if not path.is_file():
        raise UserError(
            f"no report for {name!r} at {path}: run `chess-insights report {name} --engine --coach` first "
            "(with --formats including json), or point --out at the report.", EXIT_NO_GAMES,
        )
    report_json = _read_report_json(path)
    if report_json is None:
        raise UserError(f"could not read the report {path}.", EXIT_NO_GAMES)
    cache_dir = expand_path(args.cache_dir) if args.cache_dir else None
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    cfg = CoachConfig(
        cache_dir=cache_dir / safe_file_stem(name.lower()) / "coach" if cache_dir else None,
        sources_cache=cache_dir / "sources" if cache_dir else None,
        offline=args.offline,
        lichess_token=os.environ.get("LICHESS_TOKEN", "").strip() or None,
        llm=key is not None,
        llm_model=args.llm_model or DEFAULT_LLM_MODEL,
        anthropic_api_key=key,
    )
    try:
        answer = ask.answer(args.question, report_json, cfg)
    except NotImplementedError:
        raise UserError("`ask` is not available in this version yet.", EXIT_NO_GAMES) from None
    print(answer, flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    _safe_console()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # usage errors (2), --help / --version (0): return instead of exiting
        return exc.code if isinstance(exc.code, int) else 1
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        commands = {"fetch": cmd_fetch, "report": cmd_report, "demo": cmd_demo, "puzzles-db": cmd_puzzles_db, "ask": cmd_ask}
        return commands[args.command](args)
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
                if args.command == "report":
                    _say("(no games of it were downloaded before; --pgn / --json analyse files you saved yourself)")
                else:
                    _say(f"(games downloaded earlier stay in the cache: `chess-insights report {name} --offline`)")
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
        if isinstance(exc, OSError):  # a file or folder problem is the user's to fix, not a crash
            logging.getLogger(__name__).debug("file error", exc_info=True)
            _say(f"error: {exc}")
            return EXIT_NO_GAMES
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
