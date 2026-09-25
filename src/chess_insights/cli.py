"""Command line entry point: ``chess-insights fetch | report | demo``."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .dataset import describe_filters, filter_games
from .fetch import DEFAULT_CACHE_DIR, load_games, load_json_files, month_end, parse_month, sync
from .models import Game, Report
from .parse import games_from_pgn, parse_games


def _csv(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    """Accept both repeated flags and comma lists: --time-class blitz --time-class rapid,bullet."""
    if not values:
        return None
    out = [v.strip() for item in values for v in item.split(",") if v.strip()]
    return out or None


def _month_start(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    y, m = parse_month(text)
    return datetime(y, m, 1, tzinfo=timezone.utc)


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chess-insights",
        description="Download your chess.com games and find your strengths, weaknesses and what to study.",
    )
    p.add_argument("--version", action="version", version=f"chess-insights {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_source_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"game cache (default {DEFAULT_CACHE_DIR})")
        sp.add_argument("--since", help="first month to include, YYYY-MM")
        sp.add_argument("--until", help="last month to include, YYYY-MM")
        sp.add_argument("--contact", help="email/URL added to the User-Agent, as chess.com asks API users to do")

    f = sub.add_parser("fetch", help="download/refresh your game archive from chess.com")
    f.add_argument("username")
    add_source_args(f)
    f.add_argument("--refresh-all", action="store_true", help="re-check old months too (picks up new Game Review accuracies)")

    r = sub.add_parser("report", help="analyse your games and write an HTML/Markdown/JSON report")
    r.add_argument("username")
    add_source_args(r)
    r.add_argument("--offline", action="store_true", help="use the local cache only (no chess.com requests)")
    r.add_argument("--pgn", nargs="+", type=Path, help="analyse PGN file(s) instead of the chess.com API")
    r.add_argument("--json", nargs="+", type=Path, dest="json_files", help="analyse chess.com JSON archive file(s)")
    add_analysis_args(r)

    d = sub.add_parser("demo", help="generate a synthetic player with known weaknesses and analyse them")
    d.add_argument("--games", type=int, default=400, help="number of synthetic games (default 400)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--cache-dir", type=Path, default=None, help="where to write the synthetic archive (default: temp dir)")
    d.add_argument("--no-engine-synth", action="store_true", help="generate games without Stockfish (faster, less realistic)")
    add_analysis_args(d, default_out="reports/demo")
    return p


def add_analysis_args(sp: argparse.ArgumentParser, default_out: Optional[str] = None) -> None:
    sp.add_argument("--time-class", action="append", help="bullet, blitz, rapid, daily (repeat or comma-separate)")
    sp.add_argument("--rules", default="chess", help="variant filter, default 'chess' (use 'all' for everything)")
    sp.add_argument("--rated-only", action="store_true")
    sp.add_argument("--tz", default="UTC", help="IANA time zone for time-of-day stats, e.g. America/New_York")
    sp.add_argument("--engine", action="store_true", help="run Stockfish analysis (slower, much deeper insights)")
    sp.add_argument("--stockfish", help="path to the Stockfish binary (default: auto-detect)")
    sp.add_argument("--depth", type=int, default=12, help="Stockfish depth per position (default 12)")
    sp.add_argument("--engine-games", type=int, default=150, help="analyse the N most recent games (default 150)")
    sp.add_argument("--workers", type=int, default=0, help="parallel engine processes (default: CPUs - 1)")
    sp.add_argument("--puzzles", action="store_true", help="with --engine: export your mistakes as a PGN puzzle file")
    sp.add_argument("--out", type=Path, default=Path(default_out) if default_out else None, help="output path stem (default reports/<username>)")
    sp.add_argument("--formats", default="html,md,json", help="comma list of html, md, json")


# --------------------------------------------------------------------------- commands
def cmd_fetch(args: argparse.Namespace) -> int:
    from .api import ChessComClient

    client = ChessComClient(contact=args.contact)
    s = sync(
        args.username,
        client,
        args.cache_dir,
        since=parse_month(args.since) if args.since else None,
        until=parse_month(args.until) if args.until else None,
        refresh_all=getattr(args, "refresh_all", False),
        progress=_say,
    )
    _say(
        f"done: {s.games_total} games in {s.months_listed} months "
        f"({s.months_downloaded} downloaded, {s.months_cached} cached, {s.requests} requests) -> {args.cache_dir}"
    )
    return 0


def _load_raw_games(args: argparse.Namespace) -> list[Game]:
    if args.pgn:
        games: list[Game] = []
        for path in args.pgn:
            games.extend(games_from_pgn(Path(path).read_text(encoding="utf-8", errors="replace"), args.username))
        return games
    if args.json_files:
        return parse_games(load_json_files(args.json_files), args.username)
    if not args.offline:
        cmd_fetch(args)
    since = parse_month(args.since) if args.since else None
    until = parse_month(args.until) if args.until else None
    return parse_games(load_games(args.username, args.cache_dir, since=since, until=until), args.username)


def analyse_and_write(games: list[Game], username: str, args: argparse.Namespace, cache_dir: Optional[Path]) -> Report:
    from .pipeline import run_analysis
    from .report import write_report

    time_classes = _csv(args.time_class)
    rules = None if args.rules == "all" else _csv([args.rules])
    since = _month_start(getattr(args, "since", None))
    until_text = getattr(args, "until", None)
    until = month_end(parse_month(until_text)) if until_text else None
    rated = True if args.rated_only else None
    selected = filter_games(games, time_classes=time_classes, rules=rules, rated=rated, since=since, until=until)
    filters = describe_filters(time_classes=time_classes, rules=rules, rated=rated, since=since, until=until)
    _say(f"{len(selected)} of {len(games)} games match: {filters}")

    evals, engine_note = {}, "Engine analysis not run (add --engine for accuracy, blunder and phase insights)."
    if args.engine and selected:
        evals, engine_note = run_engine(selected, username, args, cache_dir)

    report = run_analysis(
        selected, username, evals=evals, options={"tz": args.tz}, filters=filters, engine_note=engine_note
    )
    out = args.out or Path("reports") / username.lower()
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    paths = write_report(report, out, formats=formats)
    if getattr(args, "puzzles", False) and evals:
        paths.append(write_puzzles(selected, evals, out))
    print_summary(report)
    for p in paths:
        _say(f"wrote {p}")
    return report


def write_puzzles(games: list[Game], evals: dict, out: Path) -> Path:
    from .analysis.mistakes import build_puzzles, puzzles_to_pgn

    path = Path(str(Path(out).with_suffix("")) + "-puzzles.pgn")
    path.parent.mkdir(parents=True, exist_ok=True)
    puzzles = build_puzzles(games, evals)
    path.write_text(puzzles_to_pgn(puzzles), encoding="utf-8")
    _say(f"{len(puzzles)} puzzles from your own mistakes (import into a Lichess study or any chess GUI)")
    return path


def run_engine(games: list[Game], username: str, args: argparse.Namespace, cache_dir: Optional[Path]):
    from .engine import EngineConfig, analyze_games, engine_name, find_stockfish

    path = find_stockfish(args.stockfish)
    if not path:
        note = "Engine analysis skipped: Stockfish not found (install it or pass --stockfish PATH)."
        _say(note)
        return {}, note
    cfg = EngineConfig(path=path, depth=args.depth, workers=args.workers)
    eval_dir = Path(cache_dir) / username.lower() / "evals" if cache_dir else None
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
    print("", file=out)


def cmd_report(args: argparse.Namespace) -> int:
    games = _load_raw_games(args)
    if not games:
        _say(f"no games found for {args.username!r} (try `chess-insights fetch {args.username}` first)")
        return 1
    analyse_and_write(games, args.username, args, args.cache_dir)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .synth import DEFAULT_PERSONA, PLANTED_TRAITS, generate_archives, write_archives

    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="chess-insights-demo-"))
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
    _say(f"generated in {time.time() - t0:.0f}s -> {cache_dir}")

    games = parse_games(load_games(persona.username, cache_dir), persona.username)
    report = analyse_and_write(games, persona.username, args, cache_dir)

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
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return {"fetch": cmd_fetch, "report": cmd_report, "demo": cmd_demo}[args.command](args)
    except KeyboardInterrupt:
        _say("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 — friendly top-level error
        from .api import ChessComError, PlayerNotFound

        if isinstance(exc, PlayerNotFound):
            _say(f"chess.com has no player {args.username!r} (or the account is closed).")
            return 2
        if isinstance(exc, ChessComError):
            _say(f"chess.com request failed: {exc}")
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
