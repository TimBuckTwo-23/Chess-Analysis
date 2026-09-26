"""End-to-end CLI tests: chess.com is replaced by a fake session serving recorded fixture data.

Everything runs offline, in-process, on a handful of games (a few seconds in total).
"""

from __future__ import annotations

import copy
import functools
import io
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from chess_insights import api, cli, parse, pipeline
from chess_insights.dataset import filter_games
from chess_insights.models import Insight, ModuleResult

BASE = "https://api.chess.com/pub/player/testerbob"
BOUNDARY_END = int(datetime(2024, 3, 1, 3, 0, tzinfo=timezone.utc).timestamp())


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = "" if payload is None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        resp = self.routes.get(url, FakeResponse(404, {"code": 0, "message": "User not found."}))
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture(scope="module")
def sample_games(fixtures_dir):
    return json.loads((fixtures_dir / "chesscom_archive_sample.json").read_text(encoding="utf-8"))["games"]


@pytest.fixture
def boundary_game(sample_games):
    """Filed in the February archive (US month end) but ended on 1 March, 03:00 UTC."""
    g = copy.deepcopy(sample_games[0])
    g["uuid"] = "boundary-0000-11ee-8000-000000000001"
    g["url"] = g["url"].replace("105000000001", "105000009999")
    g["end_time"] = BOUNDARY_END
    return g


@pytest.fixture
def routes(sample_games, boundary_game):
    joined = int(datetime(2023, 12, 5, tzinfo=timezone.utc).timestamp())
    return {
        BASE: FakeResponse(200, {"username": "testerbob", "status": "basic", "joined": joined}),
        f"{BASE}/stats": FakeResponse(200, {"chess_blitz": {"last": {"rating": 1500}}}),
        f"{BASE}/games/archives": FakeResponse(200, {"archives": [f"{BASE}/games/2024/02", f"{BASE}/games/2024/03"]}),
        f"{BASE}/games/2024/02": FakeResponse(200, {"games": [boundary_game]}, {"ETag": 'W/"feb"'}),
        f"{BASE}/games/2024/03": FakeResponse(200, {"games": sample_games}, {"ETag": 'W/"mar"'}),
    }


@pytest.fixture
def chesscom(monkeypatch, routes):
    """Point every ChessComClient the CLI creates at the fake session; returns the session."""
    session = FakeSession(routes)
    real = api.ChessComClient
    monkeypatch.setattr(
        api, "ChessComClient", functools.partial(real, session=session, sleep=lambda s: None, min_interval=0, max_retries=1)
    )
    return session


@pytest.fixture
def run(tmp_path, capsys):
    """run(*argv) -> (exit code, stdout, stderr); --cache-dir defaults to a temp dir."""

    def _run(*argv, cache=True):
        argv = list(argv)
        if cache and argv[0] in ("report", "fetch") and "--cache-dir" not in argv:
            argv += ["--cache-dir", str(tmp_path / "cache")]
        code = cli.main(argv)
        out, err = capsys.readouterr()
        return code, out, err

    return _run


def report_json(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- online / offline
def test_report_online_writes_every_format(chesscom, run, tmp_path):
    out = tmp_path / "out" / "tb"
    code, stdout, stderr = run("report", "TesterBob", "--contact", "me@example.com", "--out", str(out))
    assert code == 0, stderr
    for ext in ("html", "md", "json"):
        assert out.with_suffix(f".{ext}").exists()
    assert "== TesterBob: " in stdout  # display name as spelled in the games
    assert "games match" in stderr and "wrote" in stderr
    assert f"wrote {out.with_suffix('.html').resolve()}" in stderr  # absolute paths: easy to find and open
    assert "tip: add --tz" in stderr  # no time zone given: say what it would add
    assert report_json(out)["n_games"] == 8  # 7 March games (standard chess) + the boundary game
    assert f"{BASE}/games/2024/03" in chesscom.calls


def test_report_offline_never_touches_the_network(chesscom, run, tmp_path):
    assert run("fetch", "testerbob")[0] == 0
    chesscom.calls.clear()
    chesscom.routes.clear()  # any request would now 404
    out = tmp_path / "off"
    code, _, stderr = run("report", "testerbob", "--offline", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert chesscom.calls == [] and report_json(out)["n_games"] == 8


def test_offline_without_a_cache_says_how_to_fetch(run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--offline", "--out", str(tmp_path / "x"))
    assert code == 1 and "chess-insights fetch testerbob" in stderr


def test_profile_url_is_accepted_as_username(chesscom, run, tmp_path):
    code, _, stderr = run("report", "https://www.chess.com/member/TesterBob", "--out", str(tmp_path / "u"), "--formats", "json")
    assert code == 0, stderr
    assert (tmp_path / "u.json").exists()


# --------------------------------------------------------------------------- errors -> exit codes
def test_unknown_user_exit_2_with_a_friendly_message(chesscom, run):
    code, _, stderr = run("report", "no-such-player-xyz")
    assert code == 2
    assert "no player 'no-such-player-xyz'" in stderr and "chess.com/member/" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("name", ["bad name", "a/b", "x" * 60])
def test_invalid_username_exit_2(run, name):
    code, _, stderr = run("report", name, "--offline")
    assert code == 2 and "not a chess.com username" in stderr


def test_http_403_exit_3_explains_user_agent(chesscom, run):
    chesscom.routes[BASE] = FakeResponse(403)
    code, _, stderr = run("fetch", "testerbob")
    assert code == 3 and "User-Agent" in stderr and "--contact" in stderr


def test_network_down_exit_3(chesscom, run):
    chesscom.routes[BASE] = requests.ConnectionError("Name or service not known")
    code, _, stderr = run("report", "testerbob")
    assert code == 3 and "api.chess.com" in stderr and "Traceback" not in stderr


def test_network_down_falls_back_to_the_cache(chesscom, run, tmp_path):
    assert run("fetch", "testerbob")[0] == 0
    chesscom.routes[BASE] = requests.ConnectionError("Name or service not known")
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "fb"), "--formats", "json")
    assert code == 0, stderr
    assert "already downloaded" in stderr and report_json(tmp_path / "fb")["n_games"] == 8


def test_empty_history(chesscom, run):
    chesscom.routes[f"{BASE}/games/archives"] = FakeResponse(200, {"archives": []})
    code, _, stderr = run("report", "testerbob")
    assert code == 1 and "no games" in stderr.lower()


def test_report_write_failure_is_friendly(chesscom, run, tmp_path):
    (tmp_path / "blocker").write_text("a file, not a folder")
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "blocker" / "r"))
    assert code == 1 and "could not write" in stderr and "Traceback" not in stderr


# --------------------------------------------------------------------------- fetch
def test_fetch_prints_a_summary(chesscom, run, tmp_path):
    code, _, stderr = run("fetch", "TesterBob")
    assert code == 0
    assert "done: 11 games in 2 months" in stderr  # raw game objects in the archives, before parsing
    assert (tmp_path / "cache" / "testerbob" / "games" / "2024-03.json").exists()


def test_fetch_reports_months_that_could_not_be_downloaded(chesscom, run):
    chesscom.routes[f"{BASE}/games/2024/02"] = FakeResponse(404, {"message": "An internal error has occurred"})
    code, _, stderr = run("fetch", "testerbob")
    assert code == 0 and "1 month" in stderr and "run the command again later" in stderr


# --------------------------------------------------------------------------- filters
def _expected(sample_games, boundary_game, **filters):
    games = parse.parse_games([boundary_game, *sample_games], "testerbob")
    return len(filter_games(games, **filters))


def test_time_class_rated_and_rules_filters(chesscom, run, tmp_path, sample_games, boundary_game):
    out = tmp_path / "f"
    code, _, stderr = run("report", "testerbob", "--time-class", "Blitz,rapid", "--time-class", "BULLET",
                          "--rated-only", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    want = _expected(sample_games, boundary_game, time_classes=["blitz", "rapid", "bullet"], rated=True)
    assert report_json(out)["n_games"] == want
    code, _, _ = run("report", "testerbob", "--rules", "all", "--out", str(out), "--formats", "json")
    assert report_json(out)["n_games"] == _expected(sample_games, boundary_game, rules=None) == 9


def test_unknown_time_class_is_rejected_with_the_valid_choices(run):
    code, _, stderr = run("report", "testerbob", "--offline", "--time-class", "blits")
    assert code == 2 and "blits" in stderr and "bullet, blitz, rapid, daily" in stderr


def test_since_includes_games_filed_in_the_previous_months_archive(chesscom, run, tmp_path):
    out = tmp_path / "s"
    code, _, stderr = run("report", "testerbob", "--since", "2024-03", "--until", "2024-03", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] == 8  # includes the 1 March 03:00 UTC game from the February archive
    assert "since 2024-03-01" in stderr and "until 2024-03-31" in stderr


def test_since_until_accept_years_and_days(chesscom, run, tmp_path):
    out = tmp_path / "d"
    code, _, stderr = run("report", "testerbob", "--since", "2024-03-02", "--until", "2024", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    games = report_json(out)
    assert games["n_games"] == 7 and "until 2024-12-31" in stderr


def test_since_uses_the_report_time_zone(chesscom, run, tmp_path):
    out = tmp_path / "tz"
    # 1 March 03:00 UTC is still 29 February in New York
    code, _, stderr = run("report", "testerbob", "--since", "2024-03", "--tz", "America/New_York",
                          "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] == 7


@pytest.mark.parametrize("value", ["2024-13", "March", "24-03", "2024-03-40"])
def test_bad_since_is_a_usage_error(run, value):
    code, _, stderr = run("report", "testerbob", "--offline", "--since", value)
    assert code == 2 and "YYYY-MM" in stderr and "Traceback" not in stderr


def test_since_after_until_is_rejected(run):
    code, _, stderr = run("report", "testerbob", "--offline", "--since", "2024-05", "--until", "2024-01")
    assert code == 2 and "--since" in stderr


def test_no_games_in_the_filter_window(chesscom, run, tmp_path):
    assert run("fetch", "testerbob")[0] == 0
    code, _, stderr = run("report", "testerbob", "--since", "2030-01", "--out", str(tmp_path / "none"))
    assert code == 1
    assert "0 of 9 games match" in stderr and "blitz" in stderr  # says what there is instead
    assert not (tmp_path / "none.html").exists()


def test_unknown_time_zone_is_rejected(run):
    code, _, stderr = run("report", "testerbob", "--offline", "--tz", "Mars/Olympus_Mons")
    assert code == 2 and "time zone" in stderr


def test_unknown_format_is_rejected_before_any_work(chesscom, run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--formats", "html,pdf", "--out", str(tmp_path / "p"))
    assert code == 2 and "pdf" in stderr
    assert chesscom.calls == []


def test_verbose_flag_works_after_the_subcommand(run):
    code, _, _ = run("report", "testerbob", "--offline", "-v")
    assert code == 1  # parsed fine (and then: no cached games)


# --------------------------------------------------------------------------- other sources
def test_pgn_source_with_bom_crlf_and_a_folder(run, tmp_path, fixtures_dir):
    text = (fixtures_dir / "sample_games.pgn").read_text(encoding="utf-8")
    folder = tmp_path / "pgns"
    folder.mkdir()
    (folder / "a.pgn").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    out = tmp_path / "pgn"
    code, _, stderr = run("report", "testerbob", "--pgn", str(folder), "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] == len(filter_games(parse.games_from_pgn(text, "testerbob")))


def test_pgn_without_the_player_lists_who_is_in_it(run, tmp_path, fixtures_dir):
    code, _, stderr = run("report", "testerbobb", "--pgn", str(fixtures_dir / "sample_games.pgn"))
    assert code == 1 and "TesterBob" in stderr and "testerbobb" in stderr


def test_missing_input_file_exit_2(run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--pgn", str(tmp_path / "nope.pgn"))
    assert code == 2 and "nope.pgn" in stderr and "Traceback" not in stderr


def test_json_source(run, tmp_path, fixtures_dir):
    out = tmp_path / "j"
    code, _, stderr = run("report", "@TesterBob", "--json", str(fixtures_dir / "chesscom_archive_sample.json"),
                          "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] == 7


# --------------------------------------------------------------------------- engine options
def test_puzzles_without_engine_explains(chesscom, run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--puzzles", "--out", str(tmp_path / "pz"), "--formats", "json")
    assert code == 0 and "--puzzles needs --engine" in stderr
    assert not (tmp_path / "pz-puzzles.pgn").exists()


def test_bad_stockfish_path_fails_before_downloading(chesscom, run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--engine", "--stockfish", str(tmp_path / "stockfish.exe"))
    assert code == 2 and "Stockfish" in stderr and chesscom.calls == []


def test_stockfish_folder_is_searched(tmp_path):
    exe = tmp_path / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    exe.chmod(0o755)
    assert cli.resolve_stockfish_arg(str(tmp_path / "stockfish")) == str(exe)


def test_missing_stockfish_warns_and_continues(chesscom, run, tmp_path, monkeypatch):
    from chess_insights import engine

    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: None)
    out = tmp_path / "ne"
    code, _, stderr = run("report", "testerbob", "--engine", "--out", str(out), "--formats", "json")
    assert code == 0 and "Stockfish was not found" in stderr
    assert "Stockfish not found" in report_json(out)["engine_note"]


# --------------------------------------------------------------------------- Windows console and paths
def test_non_utf8_console_does_not_crash(chesscom, run, tmp_path, monkeypatch):
    """A cp1252 console (or output redirected to a file on Windows) can't encode '→', 'Ł' or '−'."""
    fake = types.ModuleType("chess_insights.analysis.fake")
    fake.analyze = lambda ctx: ModuleResult(
        key="fake", title="Fake", summary="",
        insights=[Insight(id="x.weakness.y", kind="weakness", category="openings", title="Réti → Łódź −3 × “x” …",
                          detail="", severity=0.9, confidence=0.9, study=["Study Nimzo–Larsen ½"])],
    )
    monkeypatch.setitem(sys.modules, "chess_insights.analysis.fake", fake)
    monkeypatch.setattr(pipeline, "MODULES", [("fake", "Fake")])
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\r\n")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\r\n")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = cli.main(["report", "testerbob", "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "e")])
    stdout.flush()
    assert code == 0
    text = stdout.buffer.getvalue().decode("cp1252")
    assert "Réti ? ?ód? ?3 × “x” …" in text


def test_tilde_in_cache_dir_is_expanded(chesscom, run, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    code, _, stderr = run("fetch", "testerbob", "--cache-dir", "~/ci-cache", cache=False)
    assert code == 0, stderr
    assert (home / "ci-cache" / "testerbob" / "games" / "2024-03.json").exists()
    assert not (tmp_path / "~").exists()


def test_python_dash_m_entry_points():
    src = str(Path(__file__).resolve().parent.parent / "src")
    for module in ("chess_insights", "chess_insights.cli"):
        res = subprocess.run([sys.executable, "-m", module, "--version"], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": src}, timeout=60)
        assert res.returncode == 0 and "chess-insights" in res.stdout, res.stderr


# --------------------------------------------------------------------------- demo
def test_demo_with_a_tiny_synthetic_player(run, tmp_path):
    pytest.importorskip("chess_insights.synth")
    out = tmp_path / "demo"
    code, stdout, stderr = run("demo", "--games", "12", "--no-engine-synth", "--workers", "1", "--out", str(out),
                               "--formats", "json", "--cache-dir", str(tmp_path / "demo-cache"))
    assert code == 0, stderr
    assert "planted traits recovered" in stdout
    assert "[not tested: needs --engine]" in stdout  # no engine: the engine-only traits are not "missed"
    data = report_json(out)
    assert data["n_games"] > 0 and data["demo"] is True
    assert "tip: add --tz" not in stderr  # the demo player's time zone is known


def test_trait_check_says_where_each_trait_was_found(capsys):
    from chess_insights.models import Report

    def ins(id_, kind, category, title):
        return Insight(id=id_, kind=kind, category=category, title=title, detail="", severity=0.8, confidence=0.9)

    listed = ins("openings.weakness.black.caro-kann-defense", "weakness", "openings", "The Caro-Kann costs you")
    section_only = ins("habits.weakness.after-a-loss", "weakness", "habits", "You play worse right after a loss")
    stray = ins("openings.weakness.white.ruy-lopez-opening", "weakness", "openings", "The Ruy Lopez costs you")
    modules = [ModuleResult(key="openings", title="Openings", summary="", insights=[listed, stray]),
               ModuleResult(key="habits", title="Habits", summary="", insights=[section_only]),
               ModuleResult(key="engine_stats", title="Engine", summary="", stats={"games": 0})]
    report = Report(username="u", generated_at=None, filters="", n_games=1, date_from=None, date_to=None,
                    modules=modules, strengths=[], weaknesses=[listed, stray], study_plan=[])
    traits = [
        {"id": "t1", "category": "openings", "expect": "weakness", "description": "Caro", "keywords": ["caro-kann"]},
        {"id": "t2", "category": "habits", "expect": "weakness", "description": "Tilt", "keywords": ["after a loss"]},
        {"id": "t3", "category": "phases", "expect": "weakness", "description": "Endgame", "keywords": ["endgame"],
         "needs_engine": True},
        {"id": "t4", "category": "results", "expect": "strength", "description": "Rapid", "keywords": ["rapid"]},
    ]
    cli.print_trait_check(report, traits)
    out = capsys.readouterr().out
    assert "[FOUND] (weakness) Caro" in out and "[FOUND, section only] (weakness) Tilt" in out
    assert "[not tested: needs --engine] (weakness) Endgame" in out and "[missed] (strength) Rapid" in out
    assert "2/3 planted traits recovered (1 not tested)." in out
    assert "match no planted trait" in out and "- (weakness) The Ruy Lopez costs you" in out

    # the Engine review's key is engine_stats.KEY ("engine"): when it analysed games, engine traits are tested
    from chess_insights.analysis import engine_stats

    modules[-1] = ModuleResult(key=engine_stats.KEY, title="Engine", summary="", stats={"games": 90})
    cli.print_trait_check(report, traits)
    out = capsys.readouterr().out
    assert "not tested" not in out and "[missed] (weakness) Endgame" in out
    assert "2/4 planted traits recovered." in out


def test_demo_rejects_a_non_positive_game_count(run):
    code, _, stderr = run("demo", "--games", "0")
    assert code == 2 and "--games" in stderr


def test_engine_cache_goes_under_the_expanded_cache_dir(chesscom, run, tmp_path, monkeypatch):
    from chess_insights import engine

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_analyze(games, cfg, cache_dir=None, max_games=None, progress=None, time_classes=None, sample="recent"):
        seen["cache_dir"] = Path(cache_dir)
        return {}

    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: "/opt/stockfish")
    monkeypatch.setattr(engine, "analyze_games", fake_analyze)
    monkeypatch.setattr(engine, "engine_name", lambda path: "Stockfish 16")
    code, _, stderr = run("report", "testerbob", "--engine", "--cache-dir", "~/cc", "--out", str(tmp_path / "e"),
                          "--formats", "json", cache=False)
    assert code == 0, stderr
    assert seen["cache_dir"] == home / "cc" / "testerbob" / "evals"


# --------------------------------------------------------------------------- README accuracy
def test_every_documented_command_and_option_parses():
    import re
    import shlex

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    commands = [
        m.strip() for m in re.findall(r"chess-insights ((?:fetch|report|demo|puzzles-db|ask)\b[^`#\n]*)", readme)
    ]
    options = re.findall(r"^\| `(--[^`]+|-v)` \|", readme, re.M)
    assert len(commands) >= 4 and len(options) >= 8
    parser = cli.build_parser()
    for text in commands + [f"report someone {o}" for o in options]:
        if "..." in text or "--help" in text:
            continue
        argv = [a.strip('"') for a in shlex.split(text.replace("YOUR_USERNAME", "someone"), posix=False)]
        parser.parse_args(argv)  # raises SystemExit on anything undocumented or malformed


def test_invalid_json_file_names_the_file(run, tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text('{"games": [', encoding="utf-8")
    code, _, stderr = run("report", "testerbob", "--json", str(bad))
    assert code == 2 and "broken.json" in stderr and "Traceback" not in stderr


def test_default_report_name_is_safe_on_windows():
    import argparse

    args = argparse.Namespace(out=None)
    assert cli._out_stem(args, "Aux") == Path("reports") / "_aux"
    assert cli._out_stem(args, "TesterBob") == Path("reports") / "testerbob"


def test_missing_time_zone_database_suggests_tzdata(run, monkeypatch):
    """Windows ships no IANA database; without the tzdata package every name is unknown."""
    import zoneinfo

    def no_db(key):
        raise zoneinfo.ZoneInfoNotFoundError(key)

    monkeypatch.setattr(zoneinfo, "ZoneInfo", no_db)
    monkeypatch.setattr(zoneinfo, "available_timezones", lambda: set())
    code, _, stderr = run("report", "testerbob", "--offline", "--tz", "Europe/London")
    assert code == 2 and "pip install tzdata" in stderr


# --------------------------------------------------------------------------- verifier regressions
def _club_pgn(date="2024.05.04"):
    """Three games between the same two players on one day, as an over-the-board / ChessBase PGN (no Link, no time)."""
    games = [(1, "1-0", "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"), (2, "0-1", "1. f3 e5 2. g4 Qh4#"), (3, "1/2-1/2", "1. d4 d5")]
    return "".join(
        f'[Event "Club night"]\n[Site "Club"]\n[Date "{date}"]\n[Round "{rnd}"]\n[White "Me"]\n[Black "Friend"]\n'
        f'[Result "{res}"]\n\n{moves} {res}\n\n'
        for rnd, res, moves in games
    )


def test_contact_with_any_characters_is_sendable(chesscom, run):
    code, _, stderr = run("fetch", "testerbob", "--contact", "Łukasz Nowak <lukasz@example.pl>\n")
    assert code == 0, stderr
    ua = chesscom.headers["User-Agent"]
    ua.encode("latin-1")  # what http.client needs; used to raise UnicodeEncodeError
    assert ua.isascii() and ua.isprintable() and "lukasz@example.pl" in ua


@pytest.mark.parametrize("command", ["fetch", "report", "demo"])
def test_unwritable_cache_dir_is_a_friendly_error(chesscom, run, tmp_path, command):
    """A file (or read-only folder, full disk ...) as --cache-dir used to end in a NotADirectoryError traceback."""
    if command == "demo":
        pytest.importorskip("chess_insights.synth")
    blocker = tmp_path / "afile"
    blocker.write_text("not a folder")
    argv = {
        "fetch": ["fetch", "testerbob"],
        "report": ["report", "testerbob", "--out", str(tmp_path / "r")],
        "demo": ["demo", "--games", "3", "--no-engine-synth", "--workers", "1", "--out", str(tmp_path / "d")],
    }[command]
    code, _, stderr = run(*argv, "--cache-dir", str(blocker))
    assert code == 1, stderr
    assert "could not write to the game cache" in stderr and "--cache-dir" in stderr and "Traceback" not in stderr


def test_report_uses_the_cached_games_when_the_cache_cannot_be_updated(chesscom, run, tmp_path, monkeypatch):
    from chess_insights import fetch

    assert run("fetch", "testerbob")[0] == 0

    def locked(path, data):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process", str(path))

    monkeypatch.setattr(fetch, "_atomic_write_json", locked)
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "lk"), "--formats", "json")
    assert code == 0, stderr
    assert "could not write to the game cache" in stderr and "already downloaded" in stderr
    assert report_json(tmp_path / "lk")["n_games"] == 8


def test_other_file_errors_are_one_line_not_a_traceback(chesscom, run, tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "somewhere")

    monkeypatch.setattr(cli, "load_games", denied)
    code, _, stderr = run("report", "testerbob", "--offline")
    assert code == 1 and "Permission denied" in stderr and "Traceback" not in stderr


def test_retries_are_announced_instead_of_a_silent_wait(chesscom, run):
    chesscom.routes[BASE] = requests.exceptions.ConnectTimeout("connect timed out")
    code, _, stderr = run("report", "testerbob")
    assert code == 3
    assert "retrying in" in stderr and "ConnectTimeout" in stderr
    assert len(chesscom.calls) == 2  # no connection at all: one retry, not five


def test_demo_cache_dir_is_not_mixed_with_an_earlier_run(run, tmp_path):
    """A second demo into the same --cache-dir used to analyse the old run's months too (63 of 63 games)."""
    pytest.importorskip("chess_insights.synth")
    cache = tmp_path / "democache"
    common = ["--no-engine-synth", "--workers", "1", "--formats", "json", "--cache-dir", str(cache)]
    assert run("demo", "--games", "30", "--out", str(tmp_path / "d1"), *common)[0] == 0
    code, _, stderr = run("demo", "--games", "3", "--out", str(tmp_path / "d2"), *common)
    assert code == 0, stderr
    assert report_json(tmp_path / "d2")["n_games"] <= 3
    code, _, stderr = run("report", "demo_player", "--offline", "--cache-dir", str(cache), "--out", str(tmp_path / "d3"),
                          "--formats", "json", cache=False)
    assert code == 0, stderr
    assert report_json(tmp_path / "d3")["n_games"] == report_json(tmp_path / "d2")["n_games"]


@pytest.mark.parametrize("value", ["", ",", " "])
def test_empty_formats_is_rejected_before_any_work(chesscom, run, tmp_path, value):
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "e"), "--formats", value)
    assert code == 2 and "no report format" in stderr and chesscom.calls == []


def test_out_pointing_at_a_folder_writes_inside_it(chesscom, run, tmp_path):
    existing = tmp_path / "Desktop"
    existing.mkdir()
    code, _, stderr = run("report", "TesterBob", "--out", str(existing), "--formats", "json,md")
    assert code == 0, stderr
    assert sorted(p.name for p in existing.iterdir()) == ["testerbob.json", "testerbob.md"]
    assert not (tmp_path / "Desktop.json").exists()
    code, _, stderr = run("report", "TesterBob", "--out", str(tmp_path / "new") + os.sep, "--formats", "json")
    assert code == 0, stderr
    assert (tmp_path / "new" / "testerbob.json").exists()


@pytest.mark.parametrize("option", ["--since", "--until"])
@pytest.mark.parametrize("value", ["9999", "9999-12", "9999-12-31", "0999"])
def test_far_away_years_are_a_usage_error(chesscom, run, option, value):
    for command in ("report", "fetch"):
        code, _, stderr = run(command, "testerbob", option, value)
        assert code == 2 and "out of range" in stderr and "Traceback" not in stderr


def test_the_latest_possible_until_still_works(chesscom, run, tmp_path):
    code, _, stderr = run("report", "testerbob", "--until", "9998-12-31", "--out", str(tmp_path / "u"), "--formats", "json")
    assert code == 0, stderr


def _close_account(chesscom):
    chesscom.routes[BASE] = FakeResponse(200, {"username": "testerbob", "status": "closed:fair_play_violations"})
    del chesscom.routes[f"{BASE}/games/archives"]  # 404 from now on


def test_closed_account_report_uses_the_games_downloaded_earlier(chesscom, run, tmp_path):
    assert run("fetch", "testerbob")[0] == 0
    _close_account(chesscom)
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "cl"), "--formats", "json")
    assert code == 0, stderr
    assert "closed" in stderr and "already downloaded" in stderr
    assert report_json(tmp_path / "cl")["n_games"] == 8
    code, _, stderr = run("fetch", "testerbob")
    assert code == 2 and "--offline" in stderr  # fetch can't refresh it, but says how to use the cache


def test_closed_account_without_cached_games_explains(chesscom, run):
    _close_account(chesscom)
    code, _, stderr = run("report", "testerbob")
    assert code == 2 and "closed" in stderr and "--pgn" in stderr


def test_renamed_or_deleted_account_falls_back_to_the_cache(chesscom, run, tmp_path):
    assert run("fetch", "testerbob")[0] == 0
    del chesscom.routes[BASE]  # the profile now 404s
    code, _, stderr = run("report", "testerbob", "--out", str(tmp_path / "gone"), "--formats", "json")
    assert code == 0, stderr
    assert "no longer has a player" in stderr and report_json(tmp_path / "gone")["n_games"] == 8


def test_json_file_of_wrapped_games(run, tmp_path, fixtures_dir):
    """The project's own edge-case fixture ({games: [{case, game: {...}}]}) used to find no players at all."""
    out = tmp_path / "edge"
    code, _, stderr = run("report", "erik", "--json", str(fixtures_dir / "real_chesscom_edge_cases.json"), "--out", str(out),
                          "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] == 3


def test_pgn_wildcard_is_expanded_like_a_shell_would(run, tmp_path, fixtures_dir):
    """cmd.exe and PowerShell pass *.pgn through unexpanded."""
    folder = tmp_path / "pgns"
    folder.mkdir()
    (folder / "a.pgn").write_text((fixtures_dir / "sample_games.pgn").read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "notes.txt").write_text("not a pgn")
    out = tmp_path / "wild"
    code, _, stderr = run("report", "testerbob", "--pgn", str(folder / "*.pgn"), "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    assert report_json(out)["n_games"] > 0
    code, _, stderr = run("report", "testerbob", "--pgn", str(folder / "*.xyz"))
    assert code == 2 and "no files match" in stderr


def test_club_pgn_without_links_or_times(run, tmp_path):
    """Same-day games between the same players were merged into one, and dated 1970 (so --since dropped them)."""
    pgn = tmp_path / "club.pgn"
    pgn.write_text(_club_pgn(), encoding="utf-8")
    out = tmp_path / "club"
    code, _, stderr = run("report", "Me", "--pgn", str(pgn), "--since", "2024", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    data = report_json(out)
    assert data["n_games"] == 3 and data["date_from"].startswith("2024-05-04")


def test_engine_games_accepts_all():
    args = cli.build_parser().parse_args(["report", "someone", "--engine-games", "all"])
    assert args.engine_games >= 10**9
    assert cli.build_parser().parse_args(["report", "someone", "--engine-games", "40"]).engine_games == 40


# --------------------------------------------------------------------------- engine sample and coaching (C0-C4)
@pytest.fixture
def fake_engine(monkeypatch):
    """Stockfish "found", analysis replaced: records what analyze_games was asked for; returns no evals."""
    from chess_insights import engine

    seen = {}

    def fake_analyze(games, cfg, cache_dir=None, max_games=None, progress=None, time_classes=None, sample="recent"):
        seen.update(time_classes=time_classes, sample=sample, max_games=max_games, depth=cfg.depth)
        return {}

    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: "/opt/stockfish")
    monkeypatch.setattr(engine, "analyze_games", fake_analyze)
    monkeypatch.setattr(engine, "engine_name", lambda path: "Stockfish 16")
    return seen


@pytest.fixture
def fake_coaching(monkeypatch):
    """coach.build_coaching replaced: records the CoachConfig it gets."""
    from chess_insights import coach
    from chess_insights.models import Coaching

    seen = []

    def build(ctx, modules, cfg):
        seen.append(cfg)
        return Coaching(notes=["fake coaching ran"])

    monkeypatch.setattr(coach, "build_coaching", build)
    monkeypatch.setattr(coach, "finish_coaching", lambda report, cfg: report)
    return seen


def test_engine_sample_options_reach_the_engine(chesscom, run, tmp_path, fake_engine):
    code, _, stderr = run("report", "testerbob", "--engine", "--engine-sample", "balanced", "--engine-time-class",
                          "blitz,rapid", "--engine-games", "90", "--out", str(tmp_path / "e"), "--formats", "json")
    assert code == 0, stderr
    assert fake_engine == {"time_classes": ["blitz", "rapid"], "sample": "balanced", "max_games": 90, "depth": 12}
    run("report", "testerbob", "--engine", "--out", str(tmp_path / "e"), "--formats", "json")
    assert fake_engine["sample"] == "recent" and fake_engine["time_classes"] is None  # the defaults


def test_engine_sample_choices_are_checked(run):
    code, _, stderr = run("report", "someone", "--engine-sample", "random")
    assert code == 2 and "balanced" in stderr
    code, _, stderr = run("report", "someone", "--engine-time-class", "hyperbullet")
    assert code == 2 and "unknown time class" in stderr


def test_engine_note_names_the_formats_analysed():
    from factories import make_game

    games = ([make_game(time_class="bullet") for _ in range(4)] + [make_game(time_class="blitz") for _ in range(3)]
             + [make_game(time_class="rapid") for _ in range(3)])
    every = {g.game_id: None for g in games}
    assert cli.describe_engine_sample("Stockfish 16", 12, games, every, "balanced") == (
        "Stockfish 16 at depth 12 on 10 games: 4 bullet, 3 blitz, 3 rapid (most recent in each)."
    )
    assert cli.describe_engine_sample("Stockfish 16", 12, games, every) == (
        "Stockfish 16 at depth 12 on the 10 most recent games: 4 bullet, 3 blitz, 3 rapid."
    )
    bullet_only = {g.game_id: None for g in games if g.time_class == "bullet"}
    assert cli.describe_engine_sample("Stockfish 16", 10, games, bullet_only) == (
        "Stockfish 16 at depth 10 on the 4 most recent games: all bullet."
    )
    slow = {g.game_id: None for g in games if g.time_class != "bullet"}
    assert cli.describe_engine_sample("Stockfish 16", 12, games, slow, time_classes=["rapid", "blitz"]) == (
        "Stockfish 16 at depth 12 on the 6 most recent blitz and rapid games: 3 blitz, 3 rapid."
    )
    none = cli.describe_engine_sample("Stockfish 16", 12, games, {})
    assert "none of the selected games" in none
    # engine.MIN_PLIES counts half-moves: 10 plies are five moves each, not ten moves
    from chess_insights import engine

    assert engine.MIN_PLIES == 10 and "at least 10 plies (five moves each)" in none and "10 moves" not in none
    assert cli.describe_engine_sample("Stockfish 16", 12, games, bullet_only, "balanced") == (
        "Stockfish 16 at depth 12 on 4 games: all bullet (the most recent; no other format to balance with)."
    )


def test_coach_flags_reach_the_coaching_config(chesscom, run, tmp_path, monkeypatch, fake_engine, fake_coaching):
    monkeypatch.setenv("LICHESS_TOKEN", "lip_secret_token_123")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = tmp_path / "subset.csv"
    db.write_text("PuzzleId,FEN,Moves,Rating,Themes,OpeningTags\n", encoding="utf-8")
    out = tmp_path / "out" / "me"
    out.parent.mkdir()
    out.with_suffix(".json").write_text('{"username": "testerbob", "study_plan": []}', encoding="utf-8")
    code, stdout, stderr = run(
        "report", "testerbob", "--engine", "--workers", "3", "--coach", "--coach-depth", "18", "--coach-max", "40",
        "--coach-seconds", "2.5", "--no-motif-profile", "--puzzle-db", str(db), "--drill-rating", "1300-1700", "--practice-minutes", "30",
        "--coach-llm", "--llm-model", "claude-test", "--maia", "--out", str(out), "--formats", "json",
    )
    assert code == 0, stderr
    [cfg] = fake_coaching
    assert (cfg.depth, cfg.max_positions, cfg.profile, cfg.workers) == (18, 40, False, 3)
    assert cfg.search_seconds == 2.5
    assert cfg.stockfish == "/opt/stockfish"
    assert cfg.lichess_token == "lip_secret_token_123"
    assert "lip_secret_token_123" not in stdout + stderr and "lip_secret_token_123" not in out.with_suffix(".json").read_text(encoding="utf-8")
    assert cfg.puzzle_db == db and cfg.drill_rating == (1300, 1700) and cfg.practice_minutes == 30
    assert cfg.llm is False and cfg.anthropic_api_key is None and "ANTHROPIC_API_KEY" in stderr  # no key: a note
    assert cfg.llm_model == "claude-test" and cfg.maia is True and cfg.offline is False
    assert cfg.previous == {"username": "testerbob", "study_plan": []}  # read before the report replaced it
    assert cfg.out_stem == out
    cache = tmp_path / "cache"
    assert cfg.cache_dir == cache / "testerbob" / "coach" and cfg.sources_cache == cache / "sources"
    assert report_json(out)["coaching"]["notes"] == ["fake coaching ran"]
    assert "Coaching: 0 positions explained." in stdout


def test_coach_llm_with_a_key_and_offline(chesscom, run, tmp_path, monkeypatch, fake_engine, fake_coaching):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-456")
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    out = tmp_path / "me"
    code, _, _ = run("report", "testerbob", "--engine", "--coach", "--out", str(out), "--formats", "json")
    assert code == 0
    code, stdout, stderr = run("report", "testerbob", "--offline", "--engine", "--coach", "--coach-llm",
                               "--lichess-token", "lip_given", "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    first, second = fake_coaching
    assert first.llm is False and first.anthropic_api_key is None and first.lichess_token is None
    assert first.previous is None  # nothing at the output path yet
    assert (first.depth, first.max_positions, first.profile) == (20, 150, True)  # the defaults
    assert first.search_seconds is None  # deep.py's own caps
    assert first.drill_rating is None and first.practice_minutes == 20  # None: the drills follow your rating
    # --offline sends nothing to Claude: the key is not passed on, and the AI coach only leaves its note in the report
    assert second.llm is True and second.anthropic_api_key is None and second.offline is True
    assert "--offline sends nothing to Claude" in stderr and "needs ANTHROPIC_API_KEY" not in stderr
    assert second.lichess_token == "lip_given"
    assert second.previous is not None and second.previous["username"].lower() == "testerbob"
    assert "sk-ant-secret-456" not in stdout + stderr and "lip_given" not in stdout + stderr


def test_coach_needs_the_engine(chesscom, run, tmp_path, fake_coaching):
    code, _, stderr = run("report", "testerbob", "--coach", "--out", str(tmp_path / "c"), "--formats", "json")
    assert code == 0 and "--coach needs --engine" in stderr
    assert fake_coaching == [] and report_json(tmp_path / "c")["coaching"] is None


def test_previous_that_is_missing_or_broken_is_a_warning(chesscom, run, tmp_path, fake_engine, fake_coaching):
    broken = tmp_path / "old.json"
    broken.write_text("{not json", encoding="utf-8")
    for previous, message in ((tmp_path / "missing.json", "no previous report"), (broken, "could not read")):
        code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--previous", str(previous),
                              "--out", str(tmp_path / "p"), "--formats", "json")
        assert code == 0 and message in stderr
    assert [cfg.previous for cfg in fake_coaching] == [None, None]


def test_the_default_puzzle_db_is_the_subset_in_the_cache(chesscom, run, tmp_path, fake_engine, fake_coaching):
    from chess_insights.coach.puzzles_db import subset_path

    subset = subset_path(tmp_path / "cache")
    subset.parent.mkdir(parents=True)
    subset.write_text("PuzzleId,FEN,Moves,Rating,Themes,OpeningTags\n", encoding="utf-8")
    code, _, _ = run("report", "testerbob", "--engine", "--coach", "--out", str(tmp_path / "d"), "--formats", "json")
    assert code == 0 and fake_coaching[0].puzzle_db == subset


@pytest.mark.parametrize("text", ["1600-1200", "12-16", "abc", "1200", "2300-2600", "400-700", "2200-2500"])
def test_bad_drill_rating_is_a_usage_error(run, text):
    code, _, stderr = run("report", "someone", "--drill-rating", text)
    assert code == 2 and "--drill-rating" in stderr


def test_a_drill_rating_the_puzzles_cannot_satisfy_says_what_they_hold(run):
    """The puzzle collection keeps puzzles rated 800-2200: a window outside it would silently give no packs."""
    code, _, stderr = run("report", "someone", "--drill-rating", "2300-2600")
    assert code == 2 and "rated 800 to 2200" in stderr and "leave --drill-rating out" in stderr


def test_drill_rating_is_cut_to_the_puzzle_collection_and_defaults_to_your_level(
        chesscom, run, tmp_path, fake_engine, fake_coaching):
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--drill-rating", "2000-2600",
                          "--out", str(tmp_path / "d"), "--formats", "json")
    assert code == 0, stderr
    assert fake_coaching[-1].drill_rating == (2000, 2200)
    assert "note: the drill puzzles are rated 800 to 2200; the drills use 2000-2200 (from --drill-rating 2000-2600)" \
        in stderr
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--drill-rating", "1300-1700",
                          "--out", str(tmp_path / "d"), "--formats", "json")
    assert code == 0 and fake_coaching[-1].drill_rating == (1300, 1700) and "drills use" not in stderr
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--out", str(tmp_path / "d"), "--formats",
                          "json")
    assert code == 0 and fake_coaching[-1].drill_rating is None  # your level (coach.drills.drill_window)
    assert "puzzle rating window" in cli.build_parser()._subparsers._group_actions[0].choices["report"].format_help()


def test_offline_report_skips_the_ai_coach_with_a_note(chesscom, run, tmp_path, monkeypatch, fake_engine):
    """--offline --coach-llm with a key and the SDK: nothing goes to Claude, and the report says why (the real
    finish_coaching; only the coaching's first stage is faked)."""
    from chess_insights import coach
    from chess_insights.coach import llm
    from chess_insights.models import Coaching

    monkeypatch.setattr(coach, "build_coaching", lambda ctx, modules, cfg: Coaching())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-offline")
    monkeypatch.setattr(cli, "llm_sdk_installed", lambda: True)
    monkeypatch.setattr(llm, "load_sdk", lambda: pytest.fail("--offline must not load the SDK"))
    monkeypatch.setattr(llm, "make_client", lambda *a: pytest.fail("--offline must not build a client"))
    out = tmp_path / "off"
    assert run("fetch", "testerbob")[0] == 0  # the games --offline reads
    code, stdout, stderr = run("report", "testerbob", "--offline", "--engine", "--coach", "--coach-llm",
                               "--out", str(out), "--formats", "json")
    assert code == 0, stderr
    notes = report_json(out)["coaching"]["notes"]
    assert llm.OFFLINE_NOTE in notes
    assert "sk-ant-secret-offline" not in stdout + stderr + out.with_suffix(".json").read_text(encoding="utf-8")
    help_text = cli.build_parser()._subparsers._group_actions[0].choices["report"].format_help()
    assert "sends nothing to Claude" in " ".join(help_text.split())


def test_drill_files_are_listed_from_the_report_folder(chesscom, run, tmp_path, monkeypatch, fake_engine):
    """Drill.file is a name next to the report: the console lists the packs actually written, whatever the working
    folder."""
    from chess_insights import coach
    from chess_insights.models import Coaching, Drill

    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "me-drill-fork.pgn").write_text("[Event \"fork\"]\n", encoding="utf-8")
    drills = [Drill(theme="fork", title="1 fork puzzle", link="", file="me-drill-fork.pgn"),
              Drill(theme="pin", title="1 pin puzzle", link="", file="me-drill-pin.pgn"),  # named, not written
              Drill(theme="skewer", title="1 skewer puzzle", link="", file="")]
    monkeypatch.setattr(coach, "build_coaching", lambda ctx, modules, cfg: Coaching(drills=drills))
    monkeypatch.setattr(coach, "finish_coaching", lambda report, cfg: report)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--out", str(folder / "me"),
                          "--formats", "json")
    assert code == 0, stderr
    wrote = [line for line in stderr.splitlines() if line.startswith("wrote ")]
    assert f"wrote {(folder / 'me-drill-fork.pgn').resolve()}" in wrote
    assert not any("drill-pin" in line or "drill-skewer" in line for line in wrote)
    assert cli._drill_files(Coaching(drills=drills), folder / "me.html") == [folder / "me-drill-fork.pgn"]
    assert cli._drill_files(None, folder / "me") == []


def test_a_failed_puzzle_file_write_leaves_the_earlier_file_whole(tmp_path, monkeypatch):
    """The puzzle PGN is written like the report files (temporary file, then rename): an interrupted or failed
    write never leaves a truncated file for the study plan to point at."""
    from chess_insights.coach import puzzles
    from chess_insights.models import Coaching

    monkeypatch.setattr(puzzles, "puzzles_pgn", lambda events, coaching: "[Event \"new\"]\n")
    path = cli.write_puzzles([], {}, tmp_path / "me", Coaching())
    assert path.read_text(encoding="utf-8") == "[Event \"new\"]\n"
    path.write_text("[Event \"last week\"]\n", encoding="utf-8")

    def refuse(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        cli.write_puzzles([], {}, tmp_path / "me", Coaching())
    assert path.read_text(encoding="utf-8") == "[Event \"last week\"]\n"
    assert [p.name for p in tmp_path.iterdir()] == ["me-puzzles.pgn"]  # no temporary file left behind


def test_puzzle_export_uses_the_coachings_lines(tmp_path, monkeypatch):
    from chess_insights.coach import puzzles
    from chess_insights.models import Coaching

    calls = []
    monkeypatch.setattr(puzzles, "puzzles_pgn", lambda events, coaching: calls.append(coaching) or "[Event \"x\"]\n")
    path = cli.write_puzzles([], {}, tmp_path / "me", Coaching())
    assert calls and path.name == "me-puzzles.pgn" and path.read_text(encoding="utf-8").startswith("[Event")

    def broken(events, coaching):
        raise RuntimeError("no lines")

    monkeypatch.setattr(puzzles, "puzzles_pgn", broken)
    assert cli.write_puzzles([], {}, tmp_path / "me.html", Coaching()).read_text(encoding="utf-8") == ""  # plain file
    assert cli.write_puzzles([], {}, tmp_path / "plain", None).exists()


def test_puzzles_db_command(run, tmp_path, monkeypatch):
    from chess_insights.coach import puzzles_db

    def failing_download(cache_dir, progress=None, **kw):  # never the network in tests
        raise puzzles_db.PuzzleDbError("the download was cut short")

    monkeypatch.setattr(puzzles_db, "download", failing_download)
    code, _, stderr = run("puzzles-db", "--cache-dir", str(tmp_path))
    assert code == 1 and "error:" in stderr and "Traceback" not in stderr

    seen = {}

    def fake_download(cache_dir, progress=None, **kw):
        seen["cache_dir"] = cache_dir
        path = puzzles_db.subset_path(cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PuzzleId\n", encoding="utf-8")
        return path

    monkeypatch.setattr(puzzles_db, "download", fake_download)
    code, _, stderr = run("puzzles-db", "--cache-dir", str(tmp_path))
    assert code == 0 and seen["cache_dir"] == tmp_path and "wrote" in stderr

    def offline(cache_dir, progress=None, **kw):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(puzzles_db, "download", offline)
    code, _, stderr = run("puzzles-db", "--cache-dir", str(tmp_path))
    assert code == 3 and "could not download" in stderr and "drills" in stderr


def test_ask_answers_from_the_last_report(run, tmp_path, monkeypatch):
    from chess_insights.coach import ask

    monkeypatch.chdir(tmp_path)
    code, _, stderr = run("ask", "testerbob", "why do I lose?")
    assert code == 1 and "no report" in stderr and "reports" in stderr

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "testerbob.json").write_text('{"username": "TesterBob"}', encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, stdout, _ = run("ask", "TesterBob", "why do I lose?")
    assert code == 0 and "ANTHROPIC_API_KEY" in stdout  # no key: a clear message instead of an answer

    seen = {}

    def fake_answer(question, report_json, cfg):
        seen.update(question=question, report=report_json, cfg=cfg)
        return "Because of 5...e5."

    monkeypatch.setattr(ask, "answer", fake_answer)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    code, stdout, _ = run("ask", "testerbob", "why do I lose?")
    assert code == 0 and stdout.strip() == "Because of 5...e5."
    assert seen["question"] == "why do I lose?" and seen["report"] == {"username": "TesterBob"}
    assert seen["cfg"].llm is True and seen["cfg"].anthropic_api_key == "sk-ant-secret" and not seen["cfg"].offline
    assert "sk-ant-secret" not in stdout
    code, stdout, _ = run("ask", "testerbob", "why do I lose?", "--offline")  # --offline: the key stays here
    assert code == 0 and seen["cfg"].offline is True and seen["cfg"].llm is False
    assert seen["cfg"].anthropic_api_key is None and "sk-ant-secret" not in stdout
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "me.json").write_text('{"username": "x"}', encoding="utf-8")
    code, _, _ = run("ask", "testerbob", "and now?", "--out", str(tmp_path / "elsewhere" / "me"))
    assert code == 0 and seen["report"] == {"username": "x"}


def test_ask_offline_quotes_the_report_and_sends_nothing(run, tmp_path, monkeypatch):
    """`ask --offline` answers from the report's own words, with a key set and the SDK there: no request at all."""
    from chess_insights.coach import llm

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    report = {"username": "TesterBob", "summary_lines": ["You score best in rapid."], "study_plan": [
        {"title": "Stop losing on time in bullet", "why": "you lose on time often", "actions": [], "target": ""}]}
    (tmp_path / "reports" / "testerbob.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(llm, "load_sdk", lambda: pytest.fail("--offline must not load the SDK"))
    monkeypatch.setattr(llm, "make_client", lambda *a: pytest.fail("--offline must not build a client"))
    code, stdout, stderr = run("ask", "testerbob", "why do I lose on time in bullet?", "--offline")
    assert code == 0, stderr
    assert stdout.startswith("Offline (--offline): nothing was sent to Claude.")
    assert "Stop losing on time in bullet" in stdout and "sk-ant-secret" not in stdout + stderr


def test_format_views_are_listed_in_the_summary(capsys):
    from chess_insights.models import Report

    report = Report(username="u", generated_at=datetime.now(timezone.utc), filters="", n_games=300, date_from=None,
                    date_to=None, modules=[], strengths=[], weaknesses=[], study_plan=[],
                    formats={"bullet": 20, "blitz": 180, "rapid": 100})
    report.format_reports = {"blitz": report, "rapid": report}
    cli.print_summary(report)
    assert "Formats: 20 bullet, 180 blitz, 100 rapid. Separate views in the report: blitz, rapid." in capsys.readouterr().out


def test_puzzle_db_may_be_the_folder_puzzles_db_wrote_to(chesscom, run, tmp_path, fake_engine, fake_coaching):
    """The GitHub workflow keeps the database in its own cached folder and passes that folder."""
    from chess_insights.coach.puzzles_db import subset_path

    folder = tmp_path / "puzzle-cache"
    folder.mkdir()
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--puzzle-db", str(folder),
                          "--out", str(tmp_path / "f"), "--formats", "json")
    assert code == 0 and "no puzzle file" in stderr and fake_coaching[-1].puzzle_db is None  # a failed download
    subset = subset_path(folder)
    subset.parent.mkdir(parents=True)
    subset.write_text("PuzzleId,FEN,Moves,Rating,Themes,OpeningTags\n", encoding="utf-8")
    code, _, stderr = run("report", "testerbob", "--engine", "--coach", "--puzzle-db", str(folder),
                          "--out", str(tmp_path / "f"), "--formats", "json")
    assert code == 0 and "no puzzle file" not in stderr and fake_coaching[-1].puzzle_db == subset


def test_engine_time_classes_reach_the_analysis_options(chesscom, run, tmp_path, fake_engine, monkeypatch):
    seen = {}
    real = pipeline.run_analysis

    def spy(*args, **kwargs):
        seen.update(kwargs["options"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_analysis", spy)
    code, _, stderr = run("report", "testerbob", "--engine", "--engine-time-class", "blitz", "--engine-time-class",
                          "rapid", "--out", str(tmp_path / "t"), "--formats", "json")
    assert code == 0, stderr
    assert seen["engine_time_classes"] == ["blitz", "rapid"] and seen["engine_sample"] == "recent"


def test_ask_with_an_unreadable_report_says_so_once(run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "testerbob.json").write_text("{half a report", encoding="utf-8")
    code, _, stderr = run("ask", "testerbob", "why?")
    assert code == 1 and "could not read the report" in stderr
    assert "previous" not in stderr and "progress" not in stderr  # not the --previous wording


def test_coach_llm_warns_once_without_a_key_and_says_how_to_install_the_sdk(
        chesscom, run, tmp_path, monkeypatch, fake_engine, fake_coaching):
    args = ("report", "testerbob", "--engine", "--coach", "--coach-llm", "--out", str(tmp_path / "me"),
            "--formats", "json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, stdout, stderr = run(*args)
    assert code == 0, stderr
    assert (stdout + stderr).count("--coach-llm") == 1 and "ANTHROPIC_API_KEY" in stderr  # one warning
    assert fake_coaching[-1].llm is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-789")
    monkeypatch.setattr(cli, "llm_sdk_installed", lambda: False)
    code, stdout, stderr = run(*args)
    assert code == 0, stderr
    assert (stdout + stderr).count("--coach-llm") == 1 and 'pip install "chess-insights[llm]"' in stderr
    assert "sk-ant-secret-789" not in stdout + stderr
    assert fake_coaching[-1].llm is True  # the report says it too (the coaching adds a note)

    monkeypatch.setattr(cli, "llm_sdk_installed", lambda: True)
    code, stdout, stderr = run(*args)
    assert code == 0 and "--coach-llm" not in stdout + stderr and fake_coaching[-1].llm is True


def test_llm_sdk_installed_looks_for_the_package_without_importing_it(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "anthropic" else object())
    assert cli.llm_sdk_installed() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert cli.llm_sdk_installed() is True
