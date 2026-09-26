"""Opening facts (coach/openings_info): explained positions and choice points get what the public sources say,
the openings section gets a table and boards, and the theory-exit table. Recorded answers only."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from chess_insights.analysis import openings
from chess_insights.coach import openings_info
from chess_insights.coach.config import CoachConfig
from chess_insights.coach.sources import http, wikibooks
from chess_insights.coach.sources.http import TOKEN_NOTE
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, Explanation
from factories import DEFAULT_MOVES, make_game, make_game_eval, make_ply_eval

_spec = importlib.util.spec_from_file_location(
    "recorded_http", Path(__file__).parent / "fixtures" / "sources" / "recorded_http.py"
)
recorded = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorded)  # type: ignore[union-attr]

HABITS = recorded.load("habit_fens")
SMITH_MORRA_TAKES = "e4 c5 d4 cxd4 Nf3 Nc6 c3 d5".split()
SMITH_MORRA_NC6 = "e4 c5 d4 Nc6 d5 Nb8 c4 e6".split()
HABIT_5E5 = "e4 c5 Nf3 Nc6 Nc3 e6 d4 cxd4 Nxd4 e5 Ndb5 a6 Nd6+ Bxd6 Qxd6 Qf6".split()

# Wikibooks pages treated as existing in the constructed existence answers below; the page ids of the two pages
# we quote come from the recorded answers (37063: 1...c5, 42318: 2...Nc6), the others are made up for the test.
EXISTING = {
    "Chess Opening Theory/1. e4": 1001,
    "Chess Opening Theory/1. e4/1...c5": 37063,
    "Chess Opening Theory/1. e4/1...c5/2. Nf3": 1002,
    "Chess Opening Theory/1. e4/1...c5/2. Nf3/2...Nc6": 42318,
}


def existence_answer(url: str):
    """Constructed answer to the batched Wikibooks existence query (not recorded): pages in EXISTING exist."""
    query = parse_qs(urlsplit(url).query)
    if "extracts" in query.get("prop", []) or "titles" not in query:
        return None
    pages = {}
    for i, title in enumerate(query["titles"][0].split("|"), start=1):
        name = title.replace("_", " ")
        pages[str(EXISTING[name]) if name in EXISTING else f"-{i}"] = (
            {"pageid": EXISTING[name], "ns": 0, "title": name} if name in EXISTING
            else {"ns": 0, "title": name, "missing": ""}
        )
    return recorded.FakeResponse(200, {"batchcomplete": "", "query": {"pages": pages}})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = recorded.block_network(monkeypatch)
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    monkeypatch.setattr(http, "default_sleep", lambda s: None)
    http.reset_throttle()
    yield
    assert not attempts, f"a test tried to reach the network: {attempts}"


@pytest.fixture
def session(monkeypatch):
    s = recorded.RecordedSession(
        "cloud_eval_sicilian_2nc6_before", "cloud_eval_sicilian_5e5_before",
        "wikibooks_sicilian", "wikibooks_sicilian_nf3_nc6",
        fallback=existence_answer,
    )
    monkeypatch.setattr(http, "session_factory", lambda: s)
    return s


def sicilian_games():
    """13 games of 2...cxd4 and 11 of 2...Nc6 after 1.e4 c5 2.d4 (a choice point), and one 5...e5 habit game."""
    games = [
        make_game(color="black", moves_san=list(SMITH_MORRA_TAKES), opening_family="Sicilian Defense",
                  opening="Sicilian Defense: Smith-Morra Gambit", outcome="win" if i % 2 else "loss",
                  time_class="blitz" if i % 3 else "rapid", my_rating=949)
        for i in range(13)
    ]
    games += [
        make_game(color="black", moves_san=list(SMITH_MORRA_NC6), opening_family="Sicilian Defense",
                  opening="Sicilian Defense: Smith-Morra Gambit", outcome="loss", time_class="bullet", my_rating=652)
        for _ in range(11)
    ]
    habit = make_game(color="black", moves_san=list(HABIT_5E5), opening_family="Sicilian Defense",
                      opening="Sicilian Defense: Taimanov Variation", outcome="loss", time_class="blitz", my_rating=949)
    return games + [habit], habit


def habit_explanation(game) -> Explanation:
    return Explanation(
        epd=" ".join(HABITS["sicilian_5e5"]["before"].split()[:4]), fen=HABITS["sicilian_5e5"]["before"],
        played="5...e5", best="5...a6", best_line=None, refutation=None, kind="repeated", repeats=10, drop=18.0,
        time_class="blitz", color="black", game_url=game.url, games=[game.url],
    )


def run(tmp_path, games, explanations, **cfg_kw):
    ctx = AnalysisContext("tester", sorted(games, key=lambda g: g.end_time))
    modules = [openings.analyze(ctx)]
    coaching = Coaching(explanations=explanations)
    cfg = CoachConfig(sources_cache=tmp_path / "sources", **cfg_kw)
    openings_info.annotate(ctx, coaching, modules, cfg)
    return coaching, modules[0]


def test_facts_for_choice_points_and_repeated_mistakes(tmp_path, session):
    games, habit = sicilian_games()
    e = habit_explanation(habit)
    coaching, mod = run(tmp_path, games, [e])

    # the explained habit position: name, cloud lines, Wikibooks, credited
    facts = e.opening
    assert facts.name == "Sicilian Defense: Taimanov Variation" and facts.eco == "B45"
    assert [ln.moves_san[0] for ln in facts.cloud_lines] == ["a6", "Qc7", "Bc5"]
    assert facts.cloud_lines[0].cp_end == -26
    assert facts.wiki_text.startswith("Black develops the knight to its natural square c6")
    assert facts.wiki_url.endswith("/2._Nf3/2...Nc6")
    assert facts.masters == [] and facts.peers == [] and facts.peer_groups == [1200, 1400, 1600]
    names = [s.name for s in e.sources]
    assert "lichess-org/chess-openings" in names and "Wikibooks, CC BY-SA 4.0" in names
    assert any(n.startswith("Lichess cloud eval") for n in names)

    # the table under the openings tables
    table = next(t for t in mod.tables if t.title == "What stronger players play here")
    assert table.columns == ["Position", "You play", "Engine (pawns for you)", "Opening",
                             "Theory (Wikibooks, CC BY-SA 4.0)", "Wikibooks page", "Your games"]
    choice, habit_row = table.rows
    assert choice[0] == "1.e4 c5 2.d4" and choice[1].startswith("2...cxd4 (13 games, you score")
    assert "2...Nc6 (11 games" in choice[1]
    assert choice[2] == "cxd4 −0.1 · e6 −0.6 · g6 −0.8"
    assert choice[3] == "Sicilian Defense: Smith-Morra Gambit (B21)"
    assert choice[4].startswith("After 1.e4 c5: 1...c5 is the Sicilian defence")
    assert habit_row[0] == "1.e4 c5 2.Nf3 Nc6 3.Nc3 e6 4.d4 cxd4 5.Nxd4" and habit_row[1] == "5...e5 (10 games)"
    assert choice[-1] == "11 bullet · 8 blitz · 5 rapid" and habit_row[-1] == "1 blitz"  # formats behind each row
    assert table.formats[-2:] == ["url", "text"] and table.key_columns == [0, 1, 2]
    assert "Lichess opening explorer needs a token" in table.note

    # a board for each: what the engine prefers green, your move red (your other choices grey)
    choice_board, habit_board = mod.diagrams[-2:]
    assert choice_board.orientation == "black" and choice_board.title.startswith("After 1.e4 c5 2.d4: cxd4")
    assert [(a.start, a.end, a.kind) for a in choice_board.arrows] == [("c5", "d4", "best"), ("b8", "c6", "neutral")]
    assert [(a.start, a.end, a.kind) for a in habit_board.arrows] == [("a7", "a6", "best"), ("e6", "e5", "played")]
    assert "Your games: 11 bullet, 8 blitz, 5 rapid." in choice_board.caption
    assert habit_board.time_class == "blitz"
    # the cloud's best line as a strip of small boards (recorded: c5d4 g1f3 e7e5 c2c3 ...)
    (strip,) = choice_board.strips
    assert strip.title == "The engine's line (Lichess cloud: −0.1 for you)"
    assert [f.move for f in strip.frames] == ["2...cxd4", "3.Nf3", "3...e5", "4.c3"]
    assert strip.frames[0].last_move == "c5d4"

    # the explorer was never asked (no token), and said so once
    assert coaching.notes.count(TOKEN_NOTE) == 1
    assert not any("explorer.lichess.org" in url for url, _ in session.calls)


def test_theory_exit_rows_and_chart(tmp_path, session):
    games, habit = sicilian_games()
    coaching, mod = run(tmp_path, games, [])
    table = coaching.theory_exit
    assert table.title == "Where you leave opening theory"
    assert table.columns == ["Line", "Games", "Theory ends (move)", "You leave first", "Formats"]  # no engine games
    (row,) = table.rows
    # cxd4 games: named up to ply 6 (move 3); Nc6 games: up to ply 3 (move 2); the habit game: ply 9 (move 5)
    assert row[0] == "Sicilian Defense (Black)" and row[1] == 25
    assert row[2] == pytest.approx((13 * 3 + 11 * 2 + 5) / 25)
    # the first move out of theory: White's 4.c3 after 3...Nc6, your 2...Nc6, your 5...e5
    assert row[3] == pytest.approx(12 / 25)
    assert row[4] == "11 bullet · 9 blitz · 5 rapid"
    assert "bullet" in table.note and "blitz" in table.note and "rapid" in table.note
    chart = next(c for c in mod.charts if c.title == "Where you leave opening theory")
    assert chart.table is table and chart.labels == ["Sicilian Defense (Black)"]


def test_theory_exit_with_engine_games():
    white = [make_game(color="white", opening_family="Italian Game", time_class="blitz") for _ in range(8)]
    other = [make_game(color="white", opening_family="Ruy Lopez", moves_san="e4 e5 Nf3 Nc6 Bb5 a6".split())
             for _ in range(5)]
    evals = {
        g.game_id: make_game_eval(g.game_id, [
            make_ply_eval(ply=p, mover="white" if p % 2 == 0 else "black", is_user=p % 2 == 0,
                          judgement="inaccuracy" if p in (18, 19) else None)
            for p in range(20)
        ])
        for g in white[:4]
    }
    table, chart = openings_info.theory_exit(white + other, evals)
    assert [r[0] for r in table.rows] == ["Italian Game (White)"]  # the Ruy Lopez has too few games
    (row,) = table.rows
    # DEFAULT_MOVES is named up to 9.h3 (ply 17); Black's 9...h6 leaves theory, so you never leave first
    assert DEFAULT_MOVES[16] == "h3"
    assert row[1:] == [8, 9.0, 0.0, 4, 10.0, 1.0, "8 blitz"]
    assert table.formats == ["text", "int", "float1", "pct", "int", "float1", "float1", "text"]
    assert [s.name for s in chart.series] == ["Theory ends (move)", "Your first inaccuracy (move)"]
    assert "engine columns from 4 (4 blitz)" in table.note


def test_offline_without_a_cache_keeps_names_only(tmp_path, monkeypatch):
    session = recorded.RecordedSession()
    monkeypatch.setattr(http, "session_factory", lambda: session)
    games, habit = sicilian_games()
    e = habit_explanation(habit)
    coaching, mod = run(tmp_path, games, [e], offline=True)
    assert session.calls == []
    assert e.opening.name == "Sicilian Defense: Taimanov Variation" and not e.opening.cloud_lines
    assert not any(t.title == "What stronger players play here" for t in mod.tables)
    assert not any(d.title.startswith("After 1.e4 c5 2.d4") for d in mod.diagrams)
    assert any("offline" in n for n in coaching.notes) and len(coaching.notes) == len(set(coaching.notes))
    assert coaching.theory_exit is not None  # bundled names: works offline


def test_second_run_is_served_from_the_cache(tmp_path, session):
    games, habit = sicilian_games()
    run(tmp_path, games, [habit_explanation(habit)])
    first = len(session.calls)
    assert first > 0
    e = habit_explanation(habit)
    coaching, mod = run(tmp_path, games, [e], offline=True)
    assert len(session.calls) == first
    assert e.opening.wiki_text and e.opening.cloud_lines
    assert any(t.title == "What stronger players play here" for t in mod.tables)


def test_choice_points_can_be_read_from_the_table():
    games, _ = sicilian_games()
    ctx = AnalysisContext("tester", games)
    mod = openings.analyze(ctx)
    points = openings_info._choice_positions_from_table([mod])
    assert [p.moves for p in points] == [["e4", "c5", "d4"]]
    (p,) = points
    assert p.color == "black" and [(o.san, o.games) for o in p.options] == [("cxd4", 13), ("Nc6", 11)]
    assert p.fen == HABITS["sicilian_2nc6"]["before"]


def test_peer_groups_follow_the_format():
    games, _ = sicilian_games()
    cfg = CoachConfig()
    assert openings_info.peer_groups(games, "blitz", cfg)[0] == [1200, 1400, 1600]
    groups, how = openings_info.peer_groups(games, "", cfg)  # mixed / bullet positions: your blitz rating
    assert groups == [1200, 1400, 1600] and "blitz 949 is about 1390" in how
    assert openings_info.peer_groups([], "blitz", cfg) == ([], "")


def test_positions_past_move_15_get_no_opening_facts(tmp_path, session):
    late = Explanation(epd="", fen="8/5k2/8/3R4/5P2/5K2/8/r7 w - - 0 40", played="40.Rd1", best="40.f5",
                       best_line=None, refutation=None)
    games, _ = sicilian_games()
    run(tmp_path, games, [late])
    assert late.opening is None


def test_wikibooks_titles_for_the_choice_point():
    assert wikibooks.page_title(["e4", "c5", "d4"]) == "Chess_Opening_Theory/1. e4/1...c5/2. d4"


def test_a_position_shown_once_when_explained_and_a_choice_point(tmp_path, session):
    games, habit = sicilian_games()
    fen = HABITS["sicilian_2nc6"]["before"]
    nc6 = next(g for g in games if g.moves_san[3] == "Nc6")
    e = Explanation(epd=" ".join(fen.split()[:4]), fen=fen, played="2...Nc6", best="2...cxd4", best_line=None,
                    refutation=None, kind="repeated", repeats=11, color="black", game_url=nc6.url)
    coaching, mod = run(tmp_path, games, [e])
    table = next(t for t in mod.tables if t.title == "What stronger players play here")
    assert [r[0] for r in table.rows] == ["1.e4 c5 2.d4"]
    assert e.opening.name == "Sicilian Defense: Smith-Morra Gambit" and e.opening.cloud_lines
    assert sum(d.fen == fen for d in mod.diagrams) == 1


def test_cached_explorer_answers_fill_the_masters_and_peers_columns(tmp_path, session):
    # Constructed explorer answers (no token was available to record real ones), put in the cache as an earlier
    # run with a token would have left them: without a token the explorer is read from the cache only.
    import json

    from chess_insights.coach.sources import lichess_explorer

    fen = HABITS["sicilian_2nc6"]["before"]
    masters = {"white": 50, "draws": 30, "black": 20, "moves": [
        {"uci": "c5d4", "san": "cxd4", "white": 45, "draws": 28, "black": 17},
        {"uci": "e7e6", "san": "e6", "white": 3, "draws": 1, "black": 2},
        {"uci": "d7d5", "san": "d5", "white": 2, "draws": 1, "black": 1},
    ], "topGames": [{"id": "Mast3rGm", "uci": "c5d4"}]}
    peers = {"white": 500, "draws": 50, "black": 450, "moves": [
        {"uci": "c5d4", "san": "cxd4", "white": 300, "draws": 30, "black": 300},
        {"uci": "b8c6", "san": "Nc6", "white": 150, "draws": 15, "black": 100},
    ]}
    for url, body in [(lichess_explorer.masters_url(fen), masters),
                      (lichess_explorer.lichess_url(fen, [1200, 1400, 1600]), peers)]:
        path = http.cache_path(tmp_path / "sources", "explorer", url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "status": 200, "retrieved": "2026-09-25T12:00:00+00:00", "body": body}))
    games, _ = sicilian_games()
    coaching, mod = run(tmp_path, games, [])
    table = next(t for t in mod.tables if t.title == "What stronger players play here")
    assert table.columns[:5] == ["Position", "You play", "Masters play", "Players above you play",
                                 "Engine (pawns for you)"]
    assert table.columns[-2:] == ["Master game", "Your games"]
    (row,) = table.rows
    assert row[2].startswith("cxd4 90% (scores 34%) · e6 6.0%")  # Black to move: (17 + 28 / 2) / 90
    assert row[3].startswith("cxd4 63% (scores 50%) · Nc6 27%")
    assert "2...cxd4 (13 games, you score" in row[1] and "masters' #1" in row[1]
    assert "not in the masters' top 3" in row[1]
    # a masters game opens through Lichess's import route (masters ids are not Lichess game ids), Black at the bottom
    assert row[-2] == "https://lichess.org/import/master/Mast3rGm/black"
    assert "rating groups 1200, 1400, 1600; your chess.com blitz 949 is about 1390 on Lichess" in table.note
    board = next(d for d in mod.diagrams if d.fen == fen)
    assert board.title == "After 1.e4 c5 2.d4: cxd4 is what masters choose"
    assert "Lichess 1200–1799: cxd4 63%" in board.caption
    assert board.link == "https://lichess.org/import/master/Mast3rGm/black"
    assert not any("explorer.lichess.org" in url for url, _ in session.calls)


def test_peer_groups_prefer_a_checked_conversion():
    """Blitz 949 was checked against ChessGoals, rapid 1132 is only an estimate: rapid positions use blitz too."""
    games = [make_game(time_class="blitz", my_rating=949), make_game(time_class="rapid", my_rating=1132)]
    groups, how = openings_info.peer_groups(games, "rapid", CoachConfig())
    assert groups == [1200, 1400, 1600] and how.startswith("your chess.com blitz 949")
    # a rapid-only player gets the estimate, and the note says it is one
    groups, how = openings_info.peer_groups(games[1:], "rapid", CoachConfig())
    assert groups and "rough estimate" in how


def test_choice_table_with_odd_cells_and_wiki_titles_with_escapes():
    from chess_insights.models import ModuleResult, OpeningFacts, Table

    table = Table(title="Your choices at key moves", columns=["Position", "Your move", "Games", "Score"],
                  rows=[["1.e4 c5 2.d4", "2...cxd4", "n/a", 0.5], ["1.e4 c5 2.d4", "2...Nc6", 11, None]])
    (p,) = openings_info._choice_positions_from_table([ModuleResult(key="openings", title="", summary="",
                                                                    tables=[table])])
    assert [(o.san, o.games) for o in p.options] == [("cxd4", 0), ("Nc6", 11)]
    facts = OpeningFacts(wiki_text="Text.", wiki_url=wikibooks.page_url(wikibooks.page_title(["e4", "d5", "e5"])))
    assert openings_info._wiki_text(facts) == "After 1.e4 d5 2.e5: Text."
    promo = OpeningFacts(wiki_text="T.", wiki_url="https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._e8%3DQ")
    assert openings_info._wiki_text(promo) == "After 1.e8=Q: T."


def test_an_unexpected_failure_leaves_the_theory_exit(tmp_path, monkeypatch):
    """A source that breaks in an unexpected way costs its facts (one note), not the theory-exit table."""

    class Broken:
        def get(self, url, headers=None, timeout=None):
            raise ValueError("Invalid URL")  # e.g. what the HTTP stack raises for a malformed URL

    monkeypatch.setattr(http, "session_factory", lambda: Broken())
    games, habit = sicilian_games()
    e = habit_explanation(habit)
    coaching, mod = run(tmp_path, games, [e])
    assert e.opening is not None and e.opening.name and not e.opening.cloud_lines
    assert any("could not be reached (ValueError)" in n for n in coaching.notes)
    assert coaching.theory_exit is not None

    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(openings_info, "gather", boom)
    coaching, mod = run(tmp_path / "again", games, [habit_explanation(habit)])
    assert coaching.notes.count("Opening facts for a position skipped (RuntimeError: bug).") == 1
    assert coaching.theory_exit is not None


def test_an_explanation_without_its_epd_still_finds_its_moves(tmp_path, session):
    games, habit = sicilian_games()
    e = habit_explanation(habit)
    e.epd = ""  # only the FEN is set
    run(tmp_path, games, [e])
    assert e.opening.wiki_url.endswith("/2._Nf3/2...Nc6")  # the Wikibooks walk needs the moves
