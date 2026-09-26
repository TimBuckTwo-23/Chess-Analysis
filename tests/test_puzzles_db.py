"""The Lichess puzzle database subset (coach/puzzles_db.py): streamed, filtered, written atomically, read back.

Every download here is a small .csv.zst built in the test and served by a fake session: nothing touches the
network (requests' own transport is disabled for the whole module).
"""

import csv
import io
import json
from datetime import date

import chess
import pytest
import zstandard

from chess_insights.coach import puzzles_db
from chess_insights.coach.puzzles_db import PuzzleDbError

HEADER = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
GOOD = [  # PuzzleId, Rating, Popularity, NbPlays, Themes, OpeningTags
    ("aaaa1", 800, 80, 300, "fork short", "Italian_Game"),
    ("aaaa2", 2200, 95, 5000, "pin middlegame", ""),
    ("aaaa3", 1500, 100, 12345, "mateIn2 mate", "Italian_Game Italian_Game_Two_Knights_Defense"),
]
FILTERED = [
    ("bbbb1", 799, 99, 9999, "fork", ""),
    ("bbbb2", 2201, 99, 9999, "fork", ""),
    ("bbbb3", 1500, 79, 9999, "fork", ""),
    ("bbbb4", 1500, 99, 299, "fork", ""),
]


def row(pid, rating, pop, plays, themes, tags, fen=FEN, moves="f1c4 g8f6 f3g5"):
    return f"{pid},{fen},{moves},{rating},75,{pop},{plays},{themes},https://lichess.org/xyz#5,{tags}"


def database(rows, header=HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def zst(data: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(data)


class FakeResponse:
    def __init__(self, body: bytes, status=200, length=None, fail_after=None):
        self.body, self.status_code, self.fail_after = body, status, fail_after
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self.closed = False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.body), 97):  # odd chunks: rows and frames split anywhere
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("connection reset")
            yield self.body[i : i + 97]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def refuse(*a, **k):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(requests.Session, "request", refuse)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", refuse)


def good_body():
    rows = [row(*r) for r in GOOD + FILTERED] + [
        row("cccc1", "not-a-number", 90, 900, "fork", ""),  # malformed rating
        "cccc2,only,three",  # too few columns
        row("cccc3", 1500, 90, 900, "fork", "", moves="e2e4"),  # no solution
        row("cccc4", 1500, "", 900, "fork", "") + ",extra",  # too many columns
        row("cccc5", 1500, 90, 900, "fork", "", fen="garbage"),
    ]
    return zst(database(rows))


def test_download_keeps_the_filtered_columns_and_records_where_it_came_from(tmp_path):
    progress = []
    session = FakeSession(FakeResponse(good_body()))
    path = puzzles_db.download(tmp_path, url="https://example.test/p.csv.zst", progress=progress.append,
                               today=date(2026, 9, 26), session=session)
    assert path == puzzles_db.subset_path(tmp_path) == tmp_path / "puzzles" / "lichess_puzzles_subset.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(puzzles_db.COLUMNS)
    assert [r["PuzzleId"] for r in rows] == ["aaaa1", "aaaa2", "aaaa3"]
    assert rows[2]["OpeningTags"] == "Italian_Game Italian_Game_Two_Knights_Defense"
    assert rows[0]["Moves"] == "f1c4 g8f6 f3g5" and rows[0]["NbPlays"] == "300"
    meta = json.loads(puzzles_db.meta_path(path).read_text())
    assert meta["downloaded"] == "2026-09-26" and meta["url"] == "https://example.test/p.csv.zst"
    assert meta["license"] == "CC0" and meta["rows"] == 3 and meta["rows_read"] == 12
    assert puzzles_db.read_meta(path) == meta
    assert progress[-1] == 12
    url, kw = session.calls[0]
    assert kw["stream"] is True and "chess-insights" in kw["headers"]["User-Agent"]
    assert session.response.closed
    assert not list(path.parent.glob("*.part"))


def test_several_zstd_frames_are_read_as_one_stream(tmp_path):
    body = database([row(*r) for r in GOOD])
    split = body.index(b"\n", len(body) // 2) + 1
    session = FakeSession(FakeResponse(zst(body[:split]) + zst(body[split:])))
    path = puzzles_db.download(tmp_path, session=session, today=date(2026, 1, 1))
    assert [r.puzzle_id for r in puzzles_db.iter_rows(path)] == ["aaaa1", "aaaa2", "aaaa3"]


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(b"", status=429),
        FakeResponse(b"", status=500),
        FakeResponse(zst(database([row(*r) for r in GOOD]))[:-7]),  # cut short: incomplete frame
        FakeResponse(b"this is not zstd at all" * 20),
        FakeResponse(zst(database([row(*r) for r in GOOD])), length=10**6),  # fewer bytes than announced
        FakeResponse(good_body(), fail_after=200),  # connection dropped mid-stream
        FakeResponse(zst(database([row(*r) for r in GOOD], header="a,b,c"))),  # not the puzzle database
    ],
    ids=["429", "500", "truncated", "not-zstd", "short", "dropped", "wrong-file"],
)
def test_a_failed_download_raises_and_keeps_the_old_subset(tmp_path, response):
    old = puzzles_db.subset_path(tmp_path)
    old.parent.mkdir(parents=True)
    old.write_text("PuzzleId,FEN,Moves,Rating\nkeep,x,y,1\n")
    with pytest.raises(PuzzleDbError):
        puzzles_db.download(tmp_path, session=FakeSession(response), today=date(2026, 1, 1))
    assert old.read_text() == "PuzzleId,FEN,Moves,Rating\nkeep,x,y,1\n"
    assert not list(old.parent.glob("*.part")) and not puzzles_db.meta_path(old).exists()


def test_a_session_error_becomes_a_puzzle_db_error(tmp_path):
    class Broken:
        def get(self, *a, **k):
            raise OSError("no route to host")

    with pytest.raises(PuzzleDbError, match="no route to host"):
        puzzles_db.download(tmp_path, session=Broken())


def test_load_subset_pushes_the_first_move_and_keeps_the_solution(fixtures_dir):
    path = fixtures_dir / "lichess_puzzles_drill_subset.csv"
    puzzles = puzzles_db.load_subset(path)
    assert len(puzzles) == 2000
    first = puzzles[0]
    # 00Mke: FEN before the opponent's d6d7, solution a2f2 d4f2 f3f2 h2h1 f2f3
    assert first.puzzle_id == "00Mke" and first.url == "https://lichess.org/training/00Mke"
    assert first.fen == "6k1/p2P1p2/8/6pp/P2Q4/5qPP/r4P1K/3R4 b - - 0 42"
    assert first.solution_uci == ["a2f2", "d4f2", "f3f2", "h2h1", "f2f3"]
    assert first.solution_san == ["Rxf2+", "Qxf2", "Qxf2+", "Kh1", "Qf3+"]
    assert first.rating == 1160 and first.themes == ["crushing", "endgame", "long"]
    for p in puzzles:  # every puzzle is legal chess from its position
        board = chess.Board(p.fen)
        for uci, san in zip(p.solution_uci, p.solution_san, strict=True):
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves and board.san(move) == san
            board.push(move)
    tagged = [p for p in puzzles if p.opening_tags]
    assert tagged and all("_" in t or t.isalpha() for p in tagged for t in p.opening_tags)


def test_load_subset_skips_malformed_and_illegal_rows(tmp_path):
    path = tmp_path / "subset.csv"
    lines = [HEADER] + [row(*GOOD[0])] + [
        row("bad1", 1500, 90, 900, "fork", "", moves="f1c4 a1a8"),  # illegal solution move
        row("bad2", 1500, 90, 900, "fork", "", moves="e1e8 g8f6"),  # illegal first move
        row("bad3", 1500, 90, 900, "fork", "", fen="8/8/8/8 w - - 0 1"),  # broken FEN
        row("bad4", "x", 90, 900, "fork", ""),
        "bad5,,,,,,,,,",
    ]
    path.write_text("\n".join(lines) + "\n")
    assert [p.puzzle_id for p in puzzles_db.load_subset(path)] == ["aaaa1"]
    assert len(list(puzzles_db.iter_rows(path))) == 4  # well-formed rows: legality is checked by to_puzzle
    assert puzzles_db.load_subset(tmp_path / "missing.csv") == []
    assert puzzles_db.read_meta(tmp_path / "missing.csv") == {}


def test_rows_know_their_solver_and_filter():
    r = puzzles_db.parse_row(dict(zip(HEADER.split(","), next(csv.reader(io.StringIO(row(*GOOD[0])))))))
    assert r.solver == "black" and puzzles_db.keep(r)  # White moves first, Black solves
    assert r.opening_tags == ("Italian_Game",) and r.themes == ("fork", "short")
    assert puzzles_db.parse_row({"PuzzleId": "x", "FEN": FEN, "Moves": "a b", "Rating": "1500"}).popularity == 0


def test_iter_rows_filters_before_parsing(fixtures_dir):
    path = fixtures_dir / "lichess_puzzles_drill_subset.csv"
    every = list(puzzles_db.iter_rows(path))
    window = list(puzzles_db.iter_rows(path, rating=(1200, 1600)))
    assert window == [r for r in every if 1200 <= r.rating <= 1600]
    forks_or_tagged = list(puzzles_db.iter_rows(path, rating=(1200, 1600), themes={"fork"}, tagged=True))
    assert forks_or_tagged == [r for r in window if "fork" in r.themes or r.opening_tags]
    assert list(puzzles_db.iter_rows(path, themes=set())) == []
