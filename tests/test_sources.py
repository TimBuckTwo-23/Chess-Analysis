"""Public-source clients (coach/sources): the HTTP layer, each client on recorded answers, the bundled opening
names, the rating map and the concept notes. No test makes a real network request."""

from __future__ import annotations

import importlib.util
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import chess
import pytest

from chess_insights.coach import rating_map
from chess_insights.coach.sources import (
    cloud_eval, concept_notes, http, lichess_explorer, openings_db, tablebase, wikibooks,
)
from chess_insights.coach.sources.http import TOKEN_NOTE, Fetcher

_spec = importlib.util.spec_from_file_location(
    "recorded_http", Path(__file__).parent / "fixtures" / "sources" / "recorded_http.py"
)
recorded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorded)  # type: ignore[union-attr]

HABITS = recorded.load("habit_fens")
NOW = datetime(2026, 9, 26, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test on any real connection attempt; keep one test's throttle from slowing the next."""
    attempts = recorded.block_network(monkeypatch)
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    http.reset_throttle()
    yield
    http.reset_throttle()
    assert not attempts, f"a test tried to reach the network: {attempts}"


class Clock:
    """A fake monotonic clock that sleeping advances."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def fetcher(tmp_path, session=None, clock=None, **kw) -> Fetcher:
    clock = clock or Clock()
    return Fetcher(tmp_path / "sources", session=session, sleep=clock.sleep, clock=clock, now=lambda: NOW, **kw)


# --------------------------------------------------------------------------- URLs
def test_url_builders_match_the_recorded_requests():
    assert cloud_eval.url(HABITS["sicilian_5e5"]["before"]) == recorded.load("cloud_eval_sicilian_5e5_before")["url"]
    masters = recorded.load("explorer_masters_sicilian_5e5")["url"]
    assert lichess_explorer.masters_url(HABITS["sicilian_5e5"]["before"]) == masters
    assert (
        lichess_explorer.lichess_url(HABITS["sicilian_5e5"]["before"], [1200, 1400, 1600])
        == recorded.load("explorer_lichess_sicilian_5e5")["url"]
    )
    assert tablebase.url("8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1") == recorded.load("tablebase_rook_ending")["url"]
    title = wikibooks.page_title(["d4", "d5", "c4", "dxc4"])
    assert title == "Chess_Opening_Theory/1. d4/1...d5/2. c4/2...dxc4"
    assert wikibooks.extract_url(title) == recorded.load("wikibooks_qga")["url"]
    missing = recorded.load("wikibooks_missing")["url"]
    assert wikibooks.extract_url(wikibooks.page_title("d4 d5 c4 dxc4 Nc3 Nc6 e4".split())) == missing
    assert wikibooks.page_url(title) == "https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._d4/1...d5/2._c4/2...dxc4"


# --------------------------------------------------------------------------- the HTTP layer
def test_answers_are_cached_on_disk_with_their_date_and_ttl(tmp_path):
    session = recorded.RecordedSession("cloud_eval_sicilian_5e5_before")
    doc = recorded.load("cloud_eval_sicilian_5e5_before")
    f = fetcher(tmp_path, session)
    got = f.get("cloud_eval", doc["url"])
    assert got.ok and not got.from_cache and got.retrieved == "2026-09-26"
    path = http.cache_path(tmp_path / "sources", "cloud_eval", doc["url"])
    assert path.parent.name == "cloud_eval" and re.fullmatch(r"[0-9a-f]{40}\.json", path.name)
    stored = json.loads(path.read_text())
    assert stored["url"] == doc["url"] and stored["status"] == 200 and stored["retrieved"].startswith("2026-09-26")
    assert stored["body"] == doc["body"]

    again = fetcher(tmp_path, session).get("cloud_eval", doc["url"])
    assert again.from_cache and again.data == doc["body"] and len(session.calls) == 1

    # 31 days later the cloud eval answer is stale and asked again; a tablebase answer never is
    later = Fetcher(tmp_path / "sources", session=session, sleep=lambda s: None, now=lambda: NOW + timedelta(days=31))
    assert not later.get("cloud_eval", doc["url"]).from_cache and len(session.calls) == 2
    tb = recorded.load("tablebase_krk")
    tb_session = recorded.RecordedSession("tablebase_krk")
    fetcher(tmp_path, tb_session).get("tablebase", tb["url"])
    years = Fetcher(tmp_path / "sources", session=tb_session, now=lambda: NOW + timedelta(days=3650))
    assert years.get("tablebase", tb["url"]).from_cache and len(tb_session.calls) == 1


def test_a_recorded_file_can_serve_as_a_cache_entry(tmp_path):
    """The cache uses the recorded layout (``recorded`` is read as the retrieval date)."""
    doc = recorded.load("tablebase_rook_ending")
    path = http.cache_path(tmp_path / "sources", "tablebase", doc["url"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc))
    got = fetcher(tmp_path, offline=True).get("tablebase", doc["url"])
    assert got.ok and got.from_cache and got.retrieved == "2026-09-26"


def test_offline_never_requests_and_notes_once(tmp_path):
    session = recorded.RecordedSession()
    notes: list[str] = []
    f = fetcher(tmp_path, session, offline=True, notes=notes)
    url = recorded.load("cloud_eval_sicilian_5e5_before")["url"]
    assert f.get("cloud_eval", url) is None
    assert f.get("cloud_eval", url) is None
    assert session.calls == []
    assert notes == ["Lichess cloud eval not consulted (offline): only answers saved by earlier runs are used."]


def test_no_cache_folder_means_no_requests(tmp_path):
    session = recorded.RecordedSession()
    notes: list[str] = []
    f = Fetcher(None, session=session, notes=notes)
    assert f.get("tablebase", recorded.load("tablebase_krk")["url"]) is None
    assert session.calls == [] and len(notes) == 1 and "no cache folder" in notes[0]


def test_429_waits_a_minute_once_then_skips_the_source(tmp_path):
    busy = recorded.load("cloud_eval_sicilian_2nc6_Nc6")  # a real 429 page from lichess.org
    assert busy["status"] == 429
    session = recorded.RecordedSession().add(busy, times=2)
    clock = Clock()
    notes: list[str] = []
    f = fetcher(tmp_path, session, clock=clock, notes=notes)
    assert f.get("cloud_eval", busy["url"]) is None
    assert clock.sleeps.count(60.0) == 1 and len(session.calls) == 2
    assert "cloud_eval" in f.disabled
    # the rest of the run: no more requests to that source
    other = recorded.load("cloud_eval_sicilian_5e5_before")
    assert f.get("cloud_eval", other["url"]) is None and len(session.calls) == 2
    assert sum("429" in n for n in notes) == 2 and len(notes) == len(set(notes))


def test_429_then_success_after_the_wait(tmp_path):
    busy = recorded.load("cloud_eval_sicilian_2nc6_Nc6")
    ok = dict(recorded.load("cloud_eval_sicilian_2nc6_before"), url=busy["url"])  # same URL, answered after a minute
    session = recorded.RecordedSession().add(busy).add(ok)
    clock = Clock()
    f = fetcher(tmp_path, session, clock=clock)
    got = f.get("cloud_eval", busy["url"])
    assert got is not None and got.ok and clock.sleeps.count(60.0) == 1
    assert "cloud_eval" not in f.disabled


def test_401_from_the_explorer_says_a_token_is_needed(tmp_path):
    doc = recorded.load("explorer_masters_sicilian_5e5")  # the real answer without a token
    assert doc["status"] == 401
    session = recorded.RecordedSession("explorer_masters_sicilian_5e5")
    notes: list[str] = []
    f = fetcher(tmp_path, session, notes=notes, lichess_token="lip_expired")
    assert lichess_explorer.masters(f, HABITS["sicilian_5e5"]["before"]) is None
    assert session.calls[0][1]["Authorization"] == "Bearer lip_expired"
    assert any("refused the token" in n for n in notes) and "explorer" in f.disabled
    # the token never reaches the cache or the notes
    assert not any("lip_expired" in n for n in notes)
    assert not list((tmp_path / "sources").rglob("*.json"))


def test_without_a_token_the_explorer_is_never_asked(tmp_path):
    session = recorded.RecordedSession()
    notes: list[str] = []
    f = fetcher(tmp_path, session, notes=notes)
    assert lichess_explorer.masters(f, HABITS["sicilian_5e5"]["before"]) is None
    assert lichess_explorer.lichess(f, HABITS["sicilian_5e5"]["before"], [1200, 1400, 1600]) is None
    assert session.calls == [] and notes == [TOKEN_NOTE]


def test_a_timeout_skips_the_source_for_the_run(tmp_path):
    session = recorded.TimeoutSession()
    notes: list[str] = []
    f = fetcher(tmp_path, session, notes=notes)
    assert tablebase.probe(f, "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1") is None
    assert tablebase.probe(f, "8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1") is None
    assert len(session.calls) == 1
    assert notes == ["Lichess tablebase did not answer in time; skipped for the rest of this report."]


def test_cloud_eval_404_means_not_in_the_database(tmp_path):
    # Simulated: the recording run hit the rate limit before its 404 case, so this body is Lichess's documented one.
    fen = "8/8/8/3k4/8/2K5/8/7Q w - - 0 1"
    session = recorded.RecordedSession(fallback=lambda url: recorded.FakeResponse(404, {"error": "Not found"}))
    f = fetcher(tmp_path, session)
    result = cloud_eval.evaluate(f, fen)
    assert result is not None and not result.found and result.lines == []
    assert cloud_eval.evaluate(fetcher(tmp_path, session), fen).found is False  # cached 404
    assert len(session.calls) == 1


def test_requests_are_spaced_per_host_and_capped(tmp_path):
    clock = Clock()
    session = recorded.RecordedSession("tablebase_krk", "tablebase_rook_ending", "cloud_eval_sicilian_5e5_before")
    notes: list[str] = []
    f = fetcher(tmp_path, session, clock=clock, notes=notes, max_requests={"tablebase": 2})
    tablebase.probe(f, "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1")
    cloud_eval.evaluate(f, HABITS["sicilian_5e5"]["before"])  # another host: no wait
    assert clock.sleeps == []
    tablebase.probe(f, "8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1")  # same host, no time passed: waits 1 s
    assert clock.sleeps == [1.0]
    assert tablebase.probe(f, "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1") is None  # past the cap
    assert len(session.calls) == 3 and any("request limit" in n for n in notes)
    ua = session.calls[0][1]["User-Agent"]
    assert ua.startswith("chess-insights/") and "github.com" in ua


def test_one_request_at_a_time_across_threads(tmp_path):
    active, peak = [0], [0]
    lock = threading.Lock()

    class SlowSession:
        def get(self, url, headers=None, timeout=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return recorded.FakeResponse(200, {"category": "draw", "moves": []})

    def worker(i: int) -> None:
        f = Fetcher(tmp_path / "sources", session=SlowSession(), min_interval=0.0)
        f.get("tablebase", f"https://tablebase.lichess.org/standard?fen=test{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


# --------------------------------------------------------------------------- the clients
def test_cloud_eval_lines_are_parsed_for_the_side_to_move(tmp_path):
    f = fetcher(tmp_path, recorded.RecordedSession("cloud_eval_sicilian_5e5_before"))
    result = cloud_eval.evaluate(f, HABITS["sicilian_5e5"]["before"])
    assert result.found and result.depth == 50 and len(result.lines) == 3
    best = result.lines[0]
    assert best.moves_san[:3] == ["a6", "Bf4", "d6"] and best.moves_uci[0] == "a7a6"
    assert best.cp_end == -26  # +0.26 for White, Black to move
    board = chess.Board(best.fen)
    for uci in best.moves_uci:
        board.push_uci(uci)
    assert best.fen_end == board.fen()
    assert result.source.name == "Lichess cloud eval, depth 50" and result.source.retrieved == "2026-09-26"
    # a 429 page is not a cloud answer
    busy = fetcher(tmp_path, recorded.RecordedSession("cloud_eval_sicilian_2nc6_Nc6"))
    assert cloud_eval.evaluate(busy, HABITS["sicilian_2nc6"]["Nc6"]) is None


def test_tablebase_answers_are_parsed():
    lost = tablebase.parse(recorded.load("tablebase_kpk_lost")["body"], "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1")
    assert lost.category == "loss" and lost.outcome == "loss" and lost.dtz == -4 and lost.dtm == -24
    assert lost.best.san == "Kd8" and lost.best.category == "win" and lost.best.outcome_for_mover == "loss"
    drawn = tablebase.parse(recorded.load("tablebase_kpk_drawn")["body"], "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1")
    assert drawn.outcome == "draw" and [m.san for m in drawn.moves] == ["Kd1", "Kf1"]
    rook = tablebase.parse(recorded.load("tablebase_rook_ending")["body"], "8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 1")
    assert rook.outcome == "draw" and rook.best.san == "f5"
    assert rook.move("Rd1").outcome_for_mover == "loss" and rook.move("d5a5").outcome_for_mover == "loss"
    facts = rook.as_dict("d5d1")
    assert facts["result"] == "draw" and facts["best"] == "f5" and facts["played"] == "Rd1"
    assert facts["played_result"] == "loss"
    krk = tablebase.parse(recorded.load("tablebase_krk")["body"], "8/8/8/4k3/8/8/8/R3K3 w Q - 0 1")
    assert krk.outcome == "win" and krk.dtz is None and all(m.outcome_for_mover == "win" for m in krk.moves)
    assert tablebase.outcome("cursed-win") == "draw" and tablebase.outcome("blessed-loss") == "draw"
    assert tablebase.outcome("maybe-win") is None and tablebase.outcome("unknown") is None
    assert tablebase.parse({"error": "x"}, "") is None


def test_tablebase_is_only_asked_with_7_pieces_or_fewer(tmp_path):
    session = recorded.RecordedSession("tablebase_too_many", "tablebase_kpk_drawn")
    f = fetcher(tmp_path, session)
    assert tablebase.probe(f, chess.STARTING_FEN) is None
    assert session.calls == []
    assert tablebase.parse(recorded.load("tablebase_too_many")["body"], chess.STARTING_FEN).outcome is None
    result = tablebase.probe(f, "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1")
    assert result.outcome == "draw" and result.source.name == "Lichess tablebase"
    assert tablebase.pieces("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1") == 3


def test_explorer_answers_are_parsed():
    # Constructed in the shape of the lichess-org/api "OpeningExplorerMasters" schema: no token was available to
    # record a real answer, so the numbers here are made up.
    body = {
        "white": 600, "draws": 300, "black": 100,
        "moves": [
            {"uci": "c5d4", "san": "cxd4", "averageRating": 2450, "white": 540, "draws": 280, "black": 90},
            {"uci": "e7e6", "san": "e6", "averageRating": 2400, "white": 50, "draws": 15, "black": 5},
            {"uci": "d7d5", "san": "d5", "averageRating": 2380, "white": 10, "draws": 5, "black": 5},
        ],
        "topGames": [{"uci": "c5d4", "id": "abcd1234", "winner": "black"}],
        "opening": {"eco": "B21", "name": "Sicilian Defense: Smith-Morra Gambit"},
    }
    result = lichess_explorer.parse(body, HABITS["sicilian_2nc6"]["before"], "masters")
    assert [m.san for m in result.top(3)] == ["cxd4", "e6", "d5"] and result.total == 1000
    assert result.moves[0].share == pytest.approx(0.91)
    assert result.moves[0].score == pytest.approx((90 + 140) / 910)  # Black to move: Black's wins + half the draws
    assert result.rank("e6") == 2 and result.rank("c5d4") == 1 and result.rank("Nc6") is None
    assert result.top_game == "https://lichess.org/abcd1234" and result.eco == "B21"


def test_explorer_uses_cached_answers_without_a_token(tmp_path):
    move = {"uci": "c5d4", "san": "cxd4", "white": 3, "draws": 0, "black": 1}
    body = {"white": 3, "draws": 0, "black": 1, "moves": [move]}  # constructed (no token to record one)
    fen = HABITS["sicilian_2nc6"]["before"]
    url = lichess_explorer.masters_url(fen)
    path = http.cache_path(tmp_path / "sources", "explorer", url)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"url": url, "status": 200, "retrieved": "2026-09-20T10:00:00+00:00", "body": body}))
    session = recorded.RecordedSession()
    result = lichess_explorer.masters(fetcher(tmp_path, session), fen)
    assert result.moves[0].san == "cxd4" and session.calls == []
    assert result.source.retrieved == "2026-09-20" and "lichess.org/analysis" in result.source.url


def test_wikibooks_snippet_walks_up_to_the_nearest_page(tmp_path):
    moves = "d4 d5 c4 dxc4 Nc3 Nc6 e4".split()
    missing = recorded.load("wikibooks_missing")["body"]["query"]["pages"]["-1"]
    qga = recorded.load("wikibooks_qga")["body"]["query"]["pages"]["81401"]
    # Constructed: one batched existence query (not recorded). The page for 2...dxc4 and the one for 4.e4 are
    # taken from the recorded answers; the three pages in between are marked missing for the test.
    pages = {"81401": {k: v for k, v in qga.items() if k != "extract"}}
    for i, k in enumerate([7, 6, 5], start=1):
        title = wikibooks.page_title(moves[:k]).replace("_", " ")
        pages[f"-{i}"] = {"ns": 0, "title": title, "missing": ""} if k != 7 else dict(missing)
    for i, k in enumerate([3, 2, 1], start=4):
        title = wikibooks.page_title(moves[:k]).replace("_", " ")
        pages[str(90000 + i)] = {"pageid": 90000 + i, "ns": 0, "title": title}
    titles = [wikibooks.page_title(moves[:k]) for k in range(7, 0, -1)]
    batched = {"url": wikibooks.exists_url(titles), "status": 200, "content_type": "",
               "body": {"batchcomplete": "", "query": {"pages": pages}}}
    session = recorded.RecordedSession("wikibooks_qga").add(batched)
    snip = wikibooks.snippet(fetcher(tmp_path, session), moves)
    assert snip.plies == 4 and snip.url.endswith("/2._c4/2...dxc4")
    assert snip.text.startswith("The Queen's Gambit Accepted has a rich heritage")
    # two sentences: "3. Qa4+" is a move number, not a full stop; the third sentence is left out
    assert "3. Qa4+" in snip.text and snip.text.endswith("this is unnecessary.")
    assert "Black does better" not in snip.text
    assert snip.source.name == "Wikibooks, CC BY-SA 4.0" and snip.source.license == "CC BY-SA 4.0"
    assert len(session.calls) == 2


def test_wikibooks_missing_page_gives_nothing(tmp_path):
    moves = "d4 d5 c4 dxc4 Nc3 Nc6 e4".split()
    doc = recorded.load("wikibooks_missing")
    title = wikibooks.page_title(moves).replace("_", " ")
    assert not wikibooks._exists(wikibooks._pages(doc["body"])[title])
    # every question answered with the recorded "missing" page: nothing is quoted, no extract is asked for
    session = recorded.RecordedSession(fallback=lambda url: recorded.response(dict(doc, url=url)))
    assert wikibooks.snippet(fetcher(tmp_path, session), moves) is None
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "name, first",
    [
        ("wikibooks_sicilian", "1...c5 is the Sicilian defence, a counter-attacking, asymmetric opening."),
        ("wikibooks_sicilian_nf3_nc6", "Black develops the knight to its natural square c6"),
    ],
)
def test_wikibooks_extracts_keep_two_sentences(name, first):
    page = next(iter(wikibooks._pages(recorded.load(name)["body"]).values()))
    text = wikibooks.first_sentences(page["extract"])
    assert text.startswith(first) and "==" not in text
    assert len(re.findall(r"[.!?](?:\s|$)", text)) == 2


def test_first_sentences_ignores_move_numbers_and_cut_fragments():
    assert wikibooks.first_sentences("== x ==\nWhite plays 3. Qa4+ here. Then e5. More text.") == \
        "White plays 3. Qa4+ here. Then e5."
    assert wikibooks.first_sentences("Only one sentence. The main approach is 2.", 3) == "Only one sentence."
    assert wikibooks.first_sentences("") == ""


# --------------------------------------------------------------------------- bundled opening names
def test_bundled_openings_table():
    db = openings_db.load()
    assert len(db) == 3815 and db.max_plies == 36
    text = (Path(openings_db.__file__).resolve().parents[2] / "data" / "openings.tsv").read_text(encoding="utf-8")
    header, *rows = text.splitlines()
    assert header.split("\t") == ["eco", "name", "pgn", "uci", "epd"]
    for row in rows[::200]:  # the precomputed columns replay
        eco, name, pgn, uci, epd = row.split("\t")
        board = chess.Board()
        for san, move in zip(openings_db.pgn_moves(pgn), uci.split()):
            assert board.parse_san(san).uci() == move
            board.push_uci(move)
        assert board.epd() == epd


def test_epd_to_name():
    assert openings_db.lookup("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -").name == "Sicilian Defense"
    # an en passant square that no pawn can use, and move counters, don't matter
    named = openings_db.lookup("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")
    assert (named.eco, named.name) == ("B20", "Sicilian Defense")
    assert openings_db.eco_name(HABITS["qga_4e4"]["before"]) == openings_db.eco_name(
        "r1bqkbnr/ppp1pppp/2n5/8/2pP4/2N5/PP2PPPP/R1BQKBNR w KQkq -"
    )
    assert openings_db.lookup(HABITS["sicilian_2nc6"]["before"]).name == "Sicilian Defense: Smith-Morra Gambit"
    # the plan's habit positions: the habit leaves named theory, the better move stays in it
    assert openings_db.eco_name(HABITS["sicilian_5e5"]["e5"]) is None
    assert openings_db.eco_name(HABITS["sicilian_5e5"]["a6"]) == ("B46", "Sicilian Defense: Taimanov Variation")
    assert openings_db.eco_name(HABITS["qga_4e4"]["e4"]) is None
    assert openings_db.eco_name(HABITS["qga_4e4"]["Nf3"])[0] == "D07"
    assert openings_db.lookup("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1") is None
    assert openings_db.lookup("not a fen") is None


def test_nearest_named_position_along_the_moves():
    db = openings_db.load()
    near = db.nearest("e4 c5 Nf3 Nc6 Nc3 e6 d4 cxd4 Nxd4 e5".split())
    assert near.ply == 9 and near.name == "Sicilian Defense: Taimanov Variation"  # reached by transposition
    assert near.family == "Sicilian Defense"
    qga = db.nearest("d4 d5 c4 dxc4 Nc3 Nc6 e4".split())
    assert qga.ply == 6 and qga.eco == "D07"
    assert db.nearest([]) is None and db.nearest(["e4", "zz"]).ply == 1
    assert db.last_named_ply("e4 c5 d4 Nc6 d5".split()) == 3


def test_bundle_builder_adds_uci_and_epd(tmp_path):
    src = tmp_path / "a.tsv"
    src.write_text("eco\tname\tpgn\nB20\tSicilian Defense\t1. e4 c5\nX00\tBroken\t1. e5\n", encoding="utf-8")
    out = tmp_path / "openings.tsv"
    assert openings_db.write_bundle([src], out) == 1
    assert out.read_text().splitlines()[1].split("\t")[3:] == [
        "e2e4 c7c5", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -"
    ]


# --------------------------------------------------------------------------- rating map
def test_rating_map_blitz_example_from_the_plan():
    assert rating_map.to_lichess(900, "blitz") == 1360 and rating_map.to_lichess(1000, "blitz") == 1425
    assert rating_map.to_lichess(949, "blitz") == 1390
    assert rating_map.groups_for(949, "blitz") == [1200, 1400, 1600]
    assert 900 in rating_map.CHECKED["blitz"]
    assert "chessgoals.com" in rating_map.SOURCE_URL and "July 2026" in rating_map.SOURCE_NAME


def test_rating_groups_and_overrides():
    assert rating_map.rating_group(1399) == 1200 and rating_map.rating_group(1400) == 1400
    assert rating_map.rating_group(600) == 0 and rating_map.rating_group(2700) == 2500
    assert rating_map.peer_groups(2300) == [2200, 2500] and rating_map.peer_groups(2600) == [2500]
    assert rating_map.group_label([1200, 1400, 1600]) == "1200–1799" and rating_map.group_label([2500]) == "2500+"
    for tc in ("blitz", "rapid", "bullet"):
        values = [rating_map.to_lichess(r, tc) for r in range(300, 2800, 100)]
        assert values == sorted(values)  # monotone, also past the table's ends
    assert rating_map.to_lichess(949, "blitz", {"blitz": {"900": 1300, "1000": 1400}}) == 1350
    assert rating_map.to_lichess(949, "daily") is None and rating_map.to_lichess(None) is None
    assert rating_map.describe(949).startswith("your chess.com blitz 949 is about 1390 on Lichess")


# --------------------------------------------------------------------------- concept notes
def _words(text: str) -> str:
    text = text.lower().replace("’", "").replace("'", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def test_concept_notes_cite_real_chapters():
    contents = json.loads((Path(__file__).parent / "fixtures" / "sources" / "gutenberg_contents.json").read_text())
    data = concept_notes.load()
    wanted = {"king_safety", "development", "pawn_structure", "holes_outposts", "bishop_pair", "open_files",
              "passed_pawns", "piece_activity", "space"}
    assert wanted <= set(data["concepts"])
    for key in data["concepts"]:
        note = concept_notes.concept_note(key)
        sentences = re.findall(r"[^.!?]+[.!?]", note["text"])
        assert 2 <= len(sentences) <= 3, key
        assert "you" in note["text"].lower() or "your" in note["text"].lower(), key
        assert note["cites"], key
        for cite in note["cites"]:
            book = contents[str(cite["ebook"])]
            toc = _words(" ".join(book["contents"]))
            assert _words(cite["section"]) in toc, (key, cite["section"])
            chapter = _words(cite["chapter"].replace("Chapter", ""))  # "vi the middle game"
            assert chapter in toc, (key, cite["chapter"])
            assert cite["url"] == book["url"] == f"https://www.gutenberg.org/ebooks/{cite['ebook']}"


def test_concept_note_lookup_by_term_label_or_fact():
    assert concept_notes.concept_note("King safety")["key"] == "king_safety"  # Stockfish 16 term
    assert concept_notes.concept_note("piece activity")["key"] == "piece_activity"  # concepts.LABELS value
    assert concept_notes.concept_note("Mobility")["key"] == "piece_activity"
    assert concept_notes.concept_note("isolated_pawns")["key"] == "pawn_structure"  # a python-chess fact key
    assert concept_notes.concept_note("bishop pair lost")["key"] == "bishop_pair"
    assert concept_notes.concept_note("king in the centre after move 12")["key"] == "king_safety"
    assert concept_notes.concept_note("Material") is None and concept_notes.concept_note("") is None
    note = concept_notes.concept_note("Passed")
    assert note["credit"].startswith("Capablanca, Chess Fundamentals (Project Gutenberg #33870)")
    sources = concept_notes.note_sources(note)
    assert sources and all(s.url.startswith("https://www.gutenberg.org/ebooks/") for s in sources)


def test_cloud_eval_with_white_to_move_keeps_the_sign(tmp_path):
    f = fetcher(tmp_path, recorded.RecordedSession("cloud_eval_qga_4e4_before"))
    result = cloud_eval.evaluate(f, HABITS["qga_4e4"]["before"])
    assert [ln.moves_san[0] for ln in result.lines] == ["d5", "Nf3", "e3"]  # 4.e4, the habit, is not among them
    assert [ln.cp_end for ln in result.lines] == [76, 55, 40]


def test_every_recorded_explorer_answer_is_a_401(tmp_path):
    for name in ("explorer_masters_sicilian_5e5", "explorer_masters_qga_4e4", "explorer_lichess_sicilian_5e5"):
        doc = recorded.load(name)
        assert doc["status"] == 401 and "Authorization Required" in doc["body"]
    notes: list[str] = []
    f = fetcher(tmp_path, recorded.RecordedSession("explorer_masters_qga_4e4"), notes=notes, lichess_token="x")
    assert lichess_explorer.masters(f, HABITS["qga_4e4"]["before"]) is None
    assert len(notes) == 1 and "LICHESS_TOKEN" in notes[0]
