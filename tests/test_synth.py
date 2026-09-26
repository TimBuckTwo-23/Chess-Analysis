"""Synthetic chess.com archives: format fidelity, coherence of every game, determinism, planted traits."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import chess
import pytest

from chess_insights import fetch, parse, synth
from chess_insights.models import CATEGORIES, DRAW_CODES, LOSS_CODES

USER = synth.DEFAULT_PERSONA.username
HEADER_ORDER = (
    "Event Site Date Round White Black Result CurrentPosition Timezone ECO ECOUrl UTCDate UTCTime WhiteElo BlackElo "
    "TimeControl Termination StartTime EndDate EndTime Link"
).split()
GAME_KEYS = set("url pgn time_control end_time rated tcn uuid initial_setup fen time_class rules white black eco".split())
PLAYER_KEYS = {"rating", "result", "@id", "username", "uuid"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-1[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
TERMINATION_RE = {
    "checkmated": re.compile(r" won by checkmate$"),
    "resigned": re.compile(r" won by resignation$"),
    "timeout": re.compile(r" won on time$"),
    "agreed": re.compile(r"^Game drawn by agreement$"),
    "repetition": re.compile(r"^Game drawn by repetition$"),
    "stalemate": re.compile(r"^Game drawn by stalemate$"),
    "insufficient": re.compile(r"^Game drawn by insufficient material$"),
    "50move": re.compile(r"^Game drawn by 50-move rule$"),
    "timevsinsufficient": re.compile(r"^Game drawn by timeout vs insufficient material$"),
}


def all_games(archives: dict[str, list[dict]]) -> list[dict]:
    return [g for month in archives.values() for g in month]


def check_archives(archives: dict[str, list[dict]], expected: int) -> None:
    games = all_games(archives)
    assert len(games) == expected
    assert list(archives) == sorted(archives)
    for key, month in archives.items():
        assert [g["end_time"] for g in month] == sorted(g["end_time"] for g in month)
        for g in month:
            assert datetime.fromtimestamp(g["end_time"], tz=timezone.utc).strftime("%Y-%m") == key
    assert len({g["url"] for g in games}) == len({g["uuid"] for g in games}) == expected
    parsed = parse.parse_games(games, USER)
    assert len(parsed) == expected  # every game round-trips through the parser for the player
    for raw in games:
        check_game(raw)


def check_game(raw: dict) -> None:
    assert GAME_KEYS <= raw.keys()
    assert ("start_time" in raw) == (raw["time_class"] == "daily")
    assert set(raw["white"]) == set(raw["black"]) == PLAYER_KEYS
    assert UUID_RE.match(raw["uuid"]) and UUID_RE.match(raw["white"]["uuid"])
    assert raw["rules"] == "chess" and raw["initial_setup"] == chess.STARTING_FEN
    if "accuracies" in raw:
        assert all(0 <= raw["accuracies"][c] <= 100 for c in ("white", "black"))

    headers = parse.parse_pgn_headers(raw["pgn"])
    assert list(headers) == HEADER_ORDER
    assert headers["Link"] == raw["url"] and headers["ECOUrl"] == raw["eco"]
    assert (int(headers["WhiteElo"]), int(headers["BlackElo"])) == (raw["white"]["rating"], raw["black"]["rating"])
    assert headers["TimeControl"] == raw["time_control"]
    line = next(o for o in synth.OPENINGS.values() if o.url == raw["eco"])
    assert headers["ECO"] == line.eco

    g = parse.parse_game(raw, USER)
    assert g is not None and g.time_class == raw["time_class"]
    assert tuple(g.moves_san[: len(line.moves)]) == line.moves

    board = chess.Board()
    for san in g.moves_san:
        board.push_san(san)  # raises on an illegal move
    assert board.fen(en_passant="fen") == raw["fen"] == headers["CurrentPosition"]
    assert synth.encode_tcn(board.move_stack) == raw["tcn"]

    check_result(raw, headers, board)
    check_clocks(raw, g, board)


def check_result(raw: dict, headers: dict, board: chess.Board) -> None:
    result = headers["Result"]
    w, b = raw["white"]["result"], raw["black"]["result"]
    to_move = "white" if board.turn == chess.WHITE else "black"
    if result == "1/2-1/2":
        assert w == b and w in DRAW_CODES
        code = w
    else:
        winner, loser = ("white", "black") if result == "1-0" else ("black", "white")
        assert raw[winner]["result"] == "win" and raw[loser]["result"] in LOSS_CODES
        assert headers["Termination"].startswith(raw[winner]["username"] + " won")
        code = raw[loser]["result"]
        if code in ("checkmated", "timeout"):
            assert to_move == loser  # mated / flagged while on move
    assert TERMINATION_RE[code].search(headers["Termination"]), (code, headers["Termination"])
    assert board.is_checkmate() == (code == "checkmated")
    assert board.is_stalemate() == (code == "stalemate")
    if code == "insufficient":
        assert board.is_insufficient_material()
    if code == "repetition":
        assert board.is_repetition(3)
    if code == "50move":
        assert board.halfmove_clock >= 100
    if code == "timevsinsufficient":
        assert board.has_insufficient_material(not board.turn)
    if code == "agreed":
        assert board.fullmove_number >= 25
    if code in ("resigned", "timeout", "agreed"):
        assert not board.is_game_over(claim_draw=False)


def check_clocks(raw: dict, g, board: chess.Board) -> None:
    elapsed = raw["end_time"] - int(g.start_time.timestamp())
    if raw["time_class"] == "daily":
        # archived daily games store time spent / 10 in [%clk]; the parser drops them
        assert raw["time_control"] == "1/86400" and raw["start_time"] == int(g.start_time.timestamp())
        spent = parse.parse_movetext(raw["pgn"])[1]
        assert len(spent) == g.plies and all(c is not None and 0 <= c < 8640 for c in spent)
        assert elapsed >= 10 * sum(spent) - len(spent)
        return
    assert len(g.clocks) == g.plies and all(c is not None for c in g.clocks)
    base, inc = g.base_seconds, g.increment
    spent = 0.0
    last = {0: float(base), 1: float(base)}
    for i, clk in enumerate(g.clocks):
        side = i % 2
        assert 0 < clk <= last[side] + inc + 1e-9, (raw["url"], i)  # clocks only ever grow by the increment
        spent += last[side] + inc - clk
        last[side] = clk
    flagged = {raw["white"]["result"], raw["black"]["result"]} & {"timeout", "timevsinsufficient"}
    if flagged:  # the side on move used up its whole remaining clock
        remaining = last[0 if board.turn == chess.WHITE else 1]
        assert abs(elapsed - (spent + remaining)) <= 1.1, raw["url"]
    else:
        assert elapsed >= spent - 1.0


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def small_archives():
    return synth.generate_archives(12, seed=11, engine_path=None, workers=1)


@pytest.fixture(scope="module")
def medium_archives():
    return synth.generate_archives(150, seed=7, engine_path=None, workers=2)


# --------------------------------------------------------------------------- format and coherence
def test_small_heuristic_run_is_valid_chesscom_data(small_archives):
    check_archives(small_archives, 12)


def test_medium_run_is_coherent_and_varied(medium_archives):
    check_archives(medium_archives, 150)
    games = parse.parse_games(all_games(medium_archives), USER)
    time_classes = {g.time_class for g in games}
    assert {"blitz", "rapid", "bullet"} <= time_classes
    assert {g.termination for g in games} >= {"resignation", "timeout", "checkmate"}
    assert len({g.opponent for g in games}) > 100
    assert 0.25 < sum(1 for g in all_games(medium_archives) if "accuracies" in g) / 150 < 0.45


@pytest.mark.engine
def test_engine_run_is_valid_and_worker_independent(stockfish_path):
    one = synth.generate_archives(6, seed=5, engine_path=stockfish_path, workers=1)
    check_archives(one, 6)
    two = synth.generate_archives(6, seed=5, engine_path=stockfish_path, workers=2)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


def test_same_seed_same_output_regardless_of_workers():
    a = synth.generate_archives(8, seed=3, engine_path=None, workers=1)
    b = synth.generate_archives(8, seed=3, engine_path=None, workers=3)
    c = synth.generate_archives(8, seed=4, engine_path=None, workers=1)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_window_and_edge_cases():
    assert synth.generate_archives(0) == {}
    end = datetime(2025, 3, 1, tzinfo=timezone.utc)
    one = synth.generate_archives(1, seed=1, engine_path=None, end=end, months=1)
    games = all_games(one)
    assert len(games) == 1 and list(one) == ["2025-02"]
    assert datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp() <= games[0]["end_time"] < end.timestamp()
    naive = synth.generate_archives(3, seed=1, engine_path=None, end=datetime(2025, 3, 1), months=2)
    assert set(naive) <= {"2025-01", "2025-02"}
    with pytest.raises(FileNotFoundError):
        synth.generate_archives(2, engine_path="/nonexistent/stockfish")


def test_progress_is_reported_per_game():
    calls = []
    synth.generate_archives(5, seed=2, engine_path=None, workers=1, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(i, 5) for i in range(1, 6)]


def test_write_archives_round_trips_through_the_cache(tmp_path, small_archives):
    root = synth.write_archives(small_archives, tmp_path, USER)
    assert root == tmp_path / USER.lower() and (root / "games").is_dir()
    loaded = fetch.load_games(USER, tmp_path)
    assert json.dumps(sorted(loaded, key=lambda g: g["url"]), sort_keys=True) == json.dumps(
        sorted(all_games(small_archives), key=lambda g: g["url"]), sort_keys=True
    )
    store = fetch.GameStore(tmp_path, USER)
    assert [fetch.month_key(ym) for ym in store.months()] == list(small_archives)
    assert all(store.load_month(ym)["complete"] for ym in store.months())


# --------------------------------------------------------------------------- building blocks
def test_opening_book_is_real_theory_with_chesscom_names():
    for key, line in synth.OPENINGS.items():
        board = chess.Board()
        for san in line.moves:
            board.push_san(san)
        assert 6 <= len(line.moves) <= 12, key
        assert re.fullmatch(r"[A-E]\d\d", line.eco)
        opening, family = parse.opening_from_eco_url(line.url)
        assert opening and family, key
    for key in synth.CARO_KANN_LINES:
        assert parse.opening_from_eco_url(synth.OPENINGS[key].url)[1] == "Caro-Kann Defense"
    for key in synth.ITALIAN_LINES:
        assert parse.opening_from_eco_url(synth.OPENINGS[key].url)[1] == "Italian Game"


def test_tcn_matches_real_chesscom_encoding(fixtures_dir):
    raws = json.loads((fixtures_dir / "real_chesscom_games.json").read_text(encoding="utf-8"))["games"]
    checked = 0
    for raw in raws:
        if raw.get("rules") != "chess" or not raw.get("tcn") or "/game/live/" not in raw["url"]:
            continue  # (a 2015 daily game still encodes castling as king-takes-rook)
        board = chess.Board()
        for san in parse.parse_movetext(raw["pgn"])[0]:
            board.push_san(san)
        assert synth.encode_tcn(board.move_stack) == raw["tcn"], raw["url"]
        checked += 1
    assert checked >= 5
    # promotions fold piece and direction into the target character (decoder from chess.com's web client)
    chars = synth._TCN_CHARS
    for fen, san in (("8/P6k/8/8/8/8/6K1/8 w - - 0 1", "a8=N"), ("1n5k/P7/8/8/8/8/6K1/8 w - - 0 1", "axb8=Q"),
                     ("8/6K1/8/8/8/8/5p1k/6N1 b - - 0 1", "fxg1=R")):
        board = chess.Board(fen)
        move = board.parse_san(san)
        code = synth.encode_tcn([move])
        frm, to = chars.index(code[0]), chars.index(code[1])
        assert to > 63 and "qnrbkp"[(to - 64) // 3] == chess.piece_symbol(move.promotion)
        assert (frm, frm + (-8 if frm < 16 else 8) + (to - 1) % 3 - 1) == (move.from_square, move.to_square)


def test_planted_traits_contract():
    ids = [t["id"] for t in synth.PLANTED_TRAITS]
    assert len(ids) == len(set(ids)) == 8
    for trait in synth.PLANTED_TRAITS:
        assert trait["category"] in CATEGORIES
        assert trait["expect"] in ("weakness", "strength")
        assert trait["description"] and trait["keywords"]
        assert all(k == k.lower() and k.strip() for k in trait["keywords"])


def test_plan_matches_the_persona():
    """The calendar/repertoire plan (cheap, no games played) reflects the persona's mix and habits."""
    persona = synth.DEFAULT_PERSONA
    end = synth.DEFAULT_END
    plans = synth._plan_games(2000, 1, persona, synth._months_before(end, 12), end)
    n = len(plans)
    share = lambda pred: sum(1 for p in plans if pred(p)) / n  # noqa: E731
    assert share(lambda p: p.time_class == "blitz") == pytest.approx(0.60, abs=0.03)
    assert share(lambda p: p.time_class == "rapid") == pytest.approx(0.28, abs=0.03)
    assert share(lambda p: p.time_class == "daily") == pytest.approx(0.02, abs=0.01)
    black_e4 = [p for p in plans if p.color == "black" and synth.OPENINGS[p.opening].moves[0] == "e4"]
    caro = sum(p.opening in synth.CARO_KANN_LINES for p in black_e4) / len(black_e4)
    assert 0.33 <= caro <= 0.43
    white = [p for p in plans if p.color == "white"]
    assert sum(p.opening in synth.ITALIAN_LINES for p in white) / len(white) == pytest.approx(0.45, abs=0.04)
    assert 0.15 <= share(lambda p: p.late) <= 0.32
    assert 0.15 <= share(lambda p: p.tilt) <= 0.35
    assert all(synth.DEFAULT_END > p.planned_start >= synth._months_before(end, 12) for p in plans)


# --------------------------------------------------------------------------- planted traits (statistical smoke test)
def test_caro_kann_underperformance_is_visible(medium_archives):
    games = parse.parse_games(all_games(medium_archives), USER)
    diff = lambda gs: sum(g.score - g.expected_score for g in gs) / len(gs)  # noqa: E731
    caro = [g for g in games if g.color == "black" and g.opening_family == "Caro-Kann Defense"]
    others = [g for g in games if not (g.color == "black" and g.opening_family == "Caro-Kann Defense")]
    assert len(caro) >= 12
    assert diff(caro) < -0.08
    assert diff(caro) < diff(others) - 0.1


def test_blitz_clock_habits_are_visible(medium_archives):
    games = [g for g in parse.parse_games(all_games(medium_archives), USER) if g.time_class == "blitz"]
    on_time = lambda gs: sum(g.termination == "timeout" for g in gs) / len(gs)  # noqa: E731
    losses, wins = [g for g in games if g.outcome == "loss"], [g for g in games if g.outcome == "win"]
    assert on_time(losses) >= 0.10 and on_time(losses) > 1.5 * on_time(wins)  # flags far more often than opponents
    used_me = used_opp = trouble_me = trouble_opp = 0.0
    for g in games:
        mine = [c for i, c in enumerate(g.clocks) if g.is_my_ply(i)]
        theirs = [c for i, c in enumerate(g.clocks) if not g.is_my_ply(i)]
        trouble_me += min(mine, default=g.base_seconds) < 0.1 * g.base_seconds
        trouble_opp += min(theirs, default=g.base_seconds) < 0.1 * g.base_seconds
        if len(mine) >= 15 and len(theirs) >= 15:
            used_me += g.base_seconds - mine[14] + 15 * g.increment
            used_opp += g.base_seconds - theirs[14] + 15 * g.increment
    assert used_me > 1.5 * used_opp  # about twice the opponents' time on the first 15 moves
    assert trouble_me > 1.3 * trouble_opp  # ... so it ends up in time trouble far more often
