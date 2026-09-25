"""Parser checks against real, recorded chess.com API responses (tests/fixtures/real_chesscom_games.json)."""

import json

import chess
import pytest

from chess_insights import parse
from chess_insights.dataset import filter_games


@pytest.fixture(scope="module")
def raws(fixtures_dir):
    return json.loads((fixtures_dir / "real_chesscom_games.json").read_text())["games"]


def players(raw):
    return [raw[s]["username"] for s in ("white", "black") if raw.get(s, {}).get("username")]


def test_every_real_game_parses_from_both_sides(raws):
    for raw in raws:
        for user in players(raw):
            g = parse.parse_game(raw, user)
            assert g is not None
            res = parse.parse_pgn_headers(raw.get("pgn") or "").get("Result")
            if res in ("1-0", "0-1", "1/2-1/2"):
                expected = "draw" if res == "1/2-1/2" else ("win" if (res == "1-0") == (g.color == "white") else "loss")
                assert g.outcome == expected, g.url


def test_real_games_replay_to_the_reported_final_position(raws):
    checked = 0
    for raw in raws:
        g = parse.parse_game(raw, raw["white"]["username"])
        if g.rules not in ("chess", "chess960") or not raw.get("pgn"):
            continue
        board = chess.Board(g.initial_fen or chess.STARTING_FEN, chess960=g.rules == "chess960")
        for san in g.moves_san:
            board.push_san(san)
        # compare placement, side to move and castling (chess.com always writes an e.p. square after a double push)
        assert board.fen().split()[:3] == raw["fen"].split()[:3], g.url
        checked += 1
    assert checked >= 20


def test_caruana_archive_month(raws):
    caruana = parse.parse_games(raws[:17], "fabianocaruana")
    assert len(caruana) == 17
    g = next(g for g in caruana if g.url.endswith("4861120901"))
    assert (g.color, g.outcome, g.termination, g.time_class) == ("black", "win", "resignation", "blitz")
    assert (g.my_accuracy, g.opp_accuracy) == pytest.approx((89.3367519583493, 85.16453042379537))
    assert g.opening == "English Opening: Four Knights Quiet Line"
    assert g.eco == "A28"
    assert len(g.clocks) == g.plies == 38 and all(c is not None for c in g.clocks)
    assert g.clocks[:4] == [179.5, 179.0, 178.6, 178.4]  # per ply, both sides: 1. c4 {0:02:59.5} 1... e5 {0:02:59} ...
    assert g.initial_fen is None  # initial_setup "" means the standard position


def test_chess960_uses_pgn_fen(raws):
    raw = next(r for r in raws if r.get("rules") == "chess960")
    g = parse.parse_game(raw, raw["white"]["username"])
    headers = parse.parse_pgn_headers(raw["pgn"])
    assert g.initial_fen == headers["FEN"]


def test_daily_games_have_no_clocks_and_a_start_time(raws):
    dailies = [parse.parse_game(r, r["white"]["username"]) for r in raws if r.get("time_class") == "daily"]
    assert dailies
    for g in dailies:
        assert all(c is None for c in g.clocks)
        assert g.start_time is not None and g.start_time <= g.end_time


def test_variants_parse_but_are_filtered_by_default(raws):
    variants = [parse.parse_game(r, r["white"]["username"]) for r in raws if r.get("rules") in ("oddschess", "bughouse")]
    assert {g.rules for g in variants} == {"oddschess", "bughouse"}
    bughouse = next(g for g in variants if g.rules == "bughouse")
    assert bughouse.plies == 0  # bughouse games have no PGN
    assert filter_games(variants) == []


# --------------------------------------------------------------------------- real edge cases
@pytest.fixture(scope="module")
def edge(fixtures_dir):
    cases = json.loads((fixtures_dir / "real_chesscom_edge_cases.json").read_text(encoding="utf-8"))["games"]
    return {c["case"]: c for c in cases}


def test_current_games_endpoint_shape_does_not_crash_the_import(edge, raws):
    """/games (current daily games) has URL strings for white/black and no result codes."""
    current = edge["current_games_endpoint"]["game"]
    assert parse.parse_game(current, "afgano29") is None  # unfinished: no outcome yet
    games = parse.parse_games([current, *raws[:3]], raws[0]["white"]["username"])
    assert len(games) >= 1  # the rest of the history still parses


def test_unfinished_game_is_not_counted_as_a_draw():
    pgn = '[Event "Let\'s Play!"]\n[White "me"]\n[Black "you"]\n[Result "*"]\n\n1. e4 e5 *\n'
    raw = {"url": "https://www.chess.com/game/daily/1", "pgn": pgn, "end_time": 1700000000, "time_class": "daily",
           "white": {"username": "me", "rating": 1500}, "black": {"username": "you", "rating": 1500}}
    assert parse.parse_game(raw, "me") is None


def test_2009_daily_games(edge):
    zero = parse.parse_game(edge["zero_move_2009"]["game"], "erik")
    assert (zero.outcome, zero.termination, zero.plies, zero.time_class) == ("win", "timeout", 0, "daily")
    assert zero.initial_fen is None  # initial_setup is the standard FEN
    them = parse.parse_game(edge["zero_move_2009"]["game"], "pachi")
    assert (them.outcome, them.color) == ("loss", "white")


def test_thematic_start_keeps_the_fen_and_black_moves_first(edge):
    g = parse.parse_game(edge["thematic_2009"]["game"], "erik")
    assert g.rules == "chess" and g.initial_fen and not g.white_to_move_first
    board = chess.Board(g.initial_fen)
    for san in g.moves_san:
        board.push_san(san)
    assert board.fen().split()[:3] == edge["thematic_2009"]["game"]["fen"].split()[:3]
    assert g.outcome == "loss" and g.moves_san[0] == "exf4"


def test_2020_schema_without_uuid_and_old_url_form(edge):
    raw = edge["schema_2020"]["game"]
    g = parse.parse_game(raw, "cromat18")
    assert g.game_id == raw["url"]
    assert g.eco == "C40" and g.opening_family  # no "eco" field in 2020, but the PGN has ECO / ECOUrl
    assert (g.outcome, g.termination) == ("win", "checkmate")


def test_same_game_from_different_eras_is_deduplicated(edge):
    """chess.com added uuids and moved /live/game/<id> to /game/live/<id>; a re-fetch must not double count."""
    old = edge["schema_2020"]["game"]
    new = dict(old, url=old["url"].replace("/live/game/", "/game/live/"), uuid="0f0e0d0c-0000-11eb-8000-000000000001")
    assert len(parse.parse_games([old, new], "cromat18")) == 1
    # live and daily ids are separate id spaces
    daily = dict(old, url=old["url"].replace("/live/game/", "/game/daily/"))
    assert len(parse.parse_games([old, daily], "cromat18")) == 2


def test_month_boundary_games_keep_their_utc_end_time(edge):
    for case in ("month_boundary_2009", "month_boundary_2024"):
        c = edge[case]
        g = parse.parse_game(c["game"], c["game"]["black"]["username"])
        # listed in the previous month's archive, but ended on the 1st (UTC)
        assert f"{g.end_time:%Y-%m}" > c["archive_month"] and g.end_time.day == 1


def test_one_ply_game_with_integer_accuracies(edge):
    g = parse.parse_game(edge["one_ply_2026"]["game"], "NandoPro1")
    assert (g.plies, g.outcome, g.termination, g.color) == (1, "draw", "agreement", "black")
    assert (g.my_accuracy, g.opp_accuracy) == (0.0, 100.0)


def test_crazyhouse_pgn_import_is_not_standard_chess(edge):
    raw = edge["crazyhouse"]["game"]
    assert parse.parse_game(raw, "vladalekic").rules == "crazyhouse"
    (g,) = parse.games_from_pgn(raw["pgn"], "vladalekic")
    assert g.rules == "crazyhouse"
    assert filter_games([g]) == []


def test_games_without_url_or_uuid_are_not_collapsed():
    base = {"time_control": "180", "time_class": "blitz", "rules": "chess",
            "white": {"username": "me", "rating": 1500, "result": "win"},
            "black": {"username": "a", "rating": 1500, "result": "resigned"}}
    raws = [dict(base, end_time=1700000000), dict(base, end_time=1700000600)]
    games = parse.parse_games(raws, "me")
    assert len(games) == 2 and all(g.game_id for g in games) and games[0].game_id != games[1].game_id
    assert parse.parse_games(raws, "me")[0].game_id == games[0].game_id  # stable across runs


def test_ties_in_end_time_sort_deterministically(raws):
    a, b = dict(raws[0], end_time=1700000000), dict(raws[1], end_time=1700000000)
    user = raws[0]["white"]["username"]
    assert b["white"]["username"] == user
    ids1 = [g.game_id for g in parse.parse_games([a, b], user)]
    ids2 = [g.game_id for g in parse.parse_games([b, a], user)]
    assert ids1 == ids2


# --------------------------------------------------------------------------- whole recorded corpus (research dump)
CORPUS_DIR = "/tmp/claude-0/research/pubapi/gh"


def _corpus_games():
    import glob
    import os

    if not os.path.isdir(CORPUS_DIR):
        pytest.skip("research corpus not available on this machine")
    out = []

    def walk(obj):
        if isinstance(obj, dict):
            if "white" in obj and "black" in obj and ("url" in obj or "pgn" in obj):
                out.append(obj)
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(json.loads(v) if isinstance(v, str) and v.startswith("{") else v)

    for f in sorted(glob.glob(os.path.join(CORPUS_DIR, "**", "*.json"), recursive=True)):
        try:
            with open(f, encoding="utf-8") as fh:
                walk(json.load(fh))
        except ValueError:
            continue
    return out


def test_whole_recorded_corpus_parses_from_both_sides():
    from chess_insights.models import TERMINATIONS, TIME_CLASSES

    raws = _corpus_games()
    assert len(raws) > 1000
    parsed = 0
    for raw in raws:
        for side in ("white", "black"):
            player = raw[side]
            user = player.get("username") if isinstance(player, dict) else str(player).rsplit("/", 1)[-1]
            g = parse.parse_game(raw, user)
            if g is None:  # only games without any outcome (unfinished / unknown result codes) are dropped
                codes = {str(raw[s].get("result")) for s in ("white", "black") if isinstance(raw[s], dict)}
                result = parse.parse_pgn_headers(raw.get("pgn") or "").get("Result")
                assert not codes & parse._KNOWN_CODES and result not in parse._FINAL_RESULTS, raw.get("url")
                continue
            parsed += 1
            assert g.game_id and g.termination in TERMINATIONS and g.time_class in TIME_CLASSES
            assert len(g.clocks) == len(g.moves_san)
            res = parse.parse_pgn_headers(raw.get("pgn") or "").get("Result")
            if res in ("1-0", "0-1", "1/2-1/2"):
                expected = "draw" if res == "1/2-1/2" else ("win" if (res == "1-0") == (g.color == "white") else "loss")
                assert g.outcome == expected, raw.get("url")
            if g.time_class == "daily":
                assert all(c is None for c in g.clocks)
    assert parsed > 2000
    # and as whole histories, for every player with games in the corpus
    names = {r[s]["username"] for r in raws for s in ("white", "black") if isinstance(r[s], dict)}
    assert sum(len(parse.parse_games(raws, n)) for n in sorted(names)[:200]) > 0
