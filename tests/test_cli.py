"""End-to-end CLI tests: chess.com is replaced by a fake session serving recorded fixture data.

Everything runs offline, in-process, on a handful of games (a few seconds in total).
"""

from __future__ import annotations

import copy
import functools
import io
import json
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
    code, _, stderr = run("report", "testerbob", "--json", str(fixtures_dir / "chesscom_archive_sample.json"),
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
                             env={"PYTHONPATH": src, "PATH": ""}, timeout=60)
        assert res.returncode == 0 and "chess-insights" in res.stdout, res.stderr


# --------------------------------------------------------------------------- demo
def test_demo_with_a_tiny_synthetic_player(run, tmp_path):
    pytest.importorskip("chess_insights.synth")
    out = tmp_path / "demo"
    code, stdout, stderr = run("demo", "--games", "12", "--no-engine-synth", "--workers", "1", "--out", str(out),
                               "--formats", "json", "--cache-dir", str(tmp_path / "demo-cache"))
    assert code == 0, stderr
    assert "planted traits recovered" in stdout
    assert report_json(out)["n_games"] > 0


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

    def fake_analyze(games, cfg, cache_dir=None, max_games=None, progress=None):
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
    commands = [m.strip() for m in re.findall(r"chess-insights ((?:fetch|report|demo)\b[^`#\n]*)", readme)]
    options = re.findall(r"^\| `(--[^`]+|-v)` \|", readme, re.M)
    assert len(commands) >= 4 and len(options) >= 8
    parser = cli.build_parser()
    for text in commands + [f"report someone {o}" for o in options]:
        if "..." in text or "--help" in text:
            continue
        argv = [a.strip('"') for a in shlex.split(text.replace("YOUR_USERNAME", "someone"), posix=False)]
        parser.parse_args(argv)  # raises SystemExit on anything undocumented or malformed
