"""engine.py: Lichess formulas, phases, Stockfish analysis of real short games, caching and workers."""

import concurrent.futures
import json
import os
from concurrent.futures.process import BrokenProcessPool

import chess
import chess.engine
import pytest

from chess_insights import engine
from chess_insights.engine import (
    CP_LOSS_CAP,
    MATE_CP,
    EngineConfig,
    analyze_game,
    analyze_games,
    cache_file_name,
    engine_name,
    find_stockfish,
    game_accuracy,
    game_eval_from_dict,
    game_eval_to_dict,
    game_phases,
    judge,
    move_accuracy,
    win_percent,
    windows_candidates,
)
from chess_insights.parse import parse_games
from factories import make_game, make_game_eval, make_ply_eval

SCHOLARS_MATE = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]
# White sets up Qxf7# (after 5...Nf6??) and then plays 4.Qf3 instead.
MISSED_MATE = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qf3", "Bc5", "Nh3", "d6"]
QUEENS_GAMBIT = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"]
SICILIAN = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]
ILLEGAL = ["e4", "e5", "Ke3", "Ke6", "Nf3", "Nf6", "Nc3", "Nc6", "d3", "d6"]


def white_pov(cps):
    """Lichess test helper: the start position (cp 15) plus White's cp after each ply, as win %."""
    return [win_percent(15)] + [win_percent(c) for c in cps]


# --------------------------------------------------------------------------- win % / accuracy / judgements
def test_win_percent_reference_values_and_symmetry():
    assert win_percent(0) == 50.0
    assert win_percent(None) == 50.0
    for cp, expected in ((100, 59.10), (300, 75.11), (500, 86.31), (1000, 97.545)):
        assert win_percent(cp) == pytest.approx(expected, abs=0.01)
    for cp in (1, 37, 150, 420, 999, 5000):
        assert win_percent(cp) + win_percent(-cp) == pytest.approx(100.0)
    assert all(win_percent(a) < win_percent(b) for a, b in ((-300, -100), (-100, 0), (0, 80), (80, 999)))


def test_win_percent_clamps_and_maps_mates_to_the_ceiling():
    assert win_percent(5000) == win_percent(1000) == win_percent(MATE_CP)
    assert win_percent(-5000) == win_percent(-1000)
    assert win_percent(None, mate=3) == win_percent(1000)  # distance to mate is ignored, like Lichess
    assert win_percent(None, mate=1) == win_percent(None, mate=12)
    assert win_percent(-50, mate=4) == win_percent(1000)  # a mate overrides the cp value
    assert win_percent(None, mate=-2) == pytest.approx(100.0 - win_percent(None, mate=2))
    assert win_percent(None, mate=0) == win_percent(-1000)  # mate 0: this side is checkmated


def test_move_accuracy_bounds_reference_and_monotonic():
    assert move_accuracy(50, 50) == 100.0
    assert move_accuracy(40, 70) == 100.0  # improving your chances is never penalised
    assert move_accuracy(97.5, 2.5) == 0.0
    for drop, expected in ((5, 80.82), (10, 64.58), (15, 51.52), (30, 25.77)):
        assert move_accuracy(60, 60 - drop) == pytest.approx(expected, abs=0.01)
    accs = [move_accuracy(90, 90 - d) for d in range(0, 91)]
    assert all(0.0 <= a <= 100.0 for a in accs)
    assert all(a >= b for a, b in zip(accs, accs[1:]))


def test_judge_thresholds():
    assert judge(60, 55.01) is None
    assert judge(60, 55) == "inaccuracy"
    assert judge(60, 50.01) == "inaccuracy"
    assert judge(60, 50) == "mistake"
    assert judge(60, 45) == "blunder"
    assert judge(20, 80) is None


def test_judge_mate_rules():
    # a mate against the mover appears: blunder, milder if the mover was already lost
    assert judge(win_percent(0), win_percent(-1000), mate_after=-3) == "blunder"
    assert judge(win_percent(-800), win_percent(-1000), mate_after=-3) == "mistake"
    assert judge(win_percent(-1000), win_percent(-1000), mate_after=-3) == "inaccuracy"
    # the mover had a forced mate and lost it: milder if still crushing
    assert judge(win_percent(1000), win_percent(1000), mate_before=2) == "inaccuracy"
    assert judge(win_percent(1000), win_percent(800), mate_before=2) == "mistake"
    assert judge(win_percent(1000), win_percent(300), mate_before=2) == "blunder"
    assert judge(win_percent(1000), win_percent(-1000), mate_before=2, mate_after=-4) == "blunder"
    # no advice while a mate is kept, or when you were already being mated
    assert judge(win_percent(1000), win_percent(1000), mate_before=2, mate_after=5) is None
    assert judge(win_percent(-1000), win_percent(-1000), mate_before=-3, mate_after=-2) is None
    assert judge(win_percent(200), win_percent(1000), mate_after=4) is None


def test_game_accuracy_flat_and_lichess_vectors():
    flat = white_pov([15] * 20)
    assert game_accuracy(flat, "white") == pytest.approx(100.0)
    assert game_accuracy(flat, "black") == pytest.approx(100.0)
    # lila AccuracyPercentTest: "50 average moves (65 cpl)" and "(150 cpl)" -> 76 +- 8 and 54 +- 8
    avg = white_pov([-50, 15] * 50)
    assert game_accuracy(avg, "white") == pytest.approx(77.38, abs=0.05)
    assert game_accuracy(avg, "black") == pytest.approx(77.38, abs=0.05)
    assert game_accuracy(white_pov([-135, 15] * 50), "white") == pytest.approx(55.0, abs=0.05)


def test_game_accuracy_volatile_sequence_and_edge_cases():
    blunder = white_pov([15] * 20 + [-900])  # 20 perfect moves, then a White blunder
    white = game_accuracy(blunder, "white")
    assert white == pytest.approx(50.0, abs=5.0)
    assert game_accuracy(blunder, "black") == pytest.approx(100.0, abs=1.0)
    # one bad move weighs more in a volatile game than the plain mean of move accuracies would suggest
    per_move = [move_accuracy(blunder[i], blunder[i + 1]) for i in range(0, len(blunder) - 1, 2)]
    assert white < sum(per_move) / len(per_move)
    # Black moves first (a set-up position)
    black_first = white_pov([900, 900])
    assert game_accuracy(black_first, "black", white_moves_first=False) == pytest.approx(12.30, abs=0.05)
    assert game_accuracy(black_first, "white", white_moves_first=False) == pytest.approx(100.0)
    assert game_accuracy([], "white") is None
    assert game_accuracy([50.0], "white") is None
    assert game_accuracy(white_pov([15]), "black") is None  # Black never moved


# --------------------------------------------------------------------------- phases
def test_game_phases_opening_middlegame_endgame():
    start = game_phases(make_game())
    assert len(start) == 20 and start[0] == "opening"
    # White's back rank empties to 3 pieces (Ra1, Ke1, Rh1) after Qd2 -> middlegame from ply 13
    developed = "e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 Be3 Be6 Qd2 Qd7 O-O O-O".split()
    phases = game_phases(make_game(moves_san=developed))
    assert phases[:13] == ["opening"] * 13 and phases[13:] == ["middlegame"] * 3
    # queens and knights traded -> middlegame
    traded = "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nxd4 Qxd4 Qf6 Qxf6 Nxf6 Bc4 Bc5 Be3 Bxe3 fxe3 Nxe4".split()
    assert game_phases(make_game(moves_san=traded))[-1] == "middlegame"
    rook_ending = make_game(initial_fen="8/8/8/4k3/8/8/8/R3K3 w - - 0 1", moves_san=["Ra7", "Kd5", "Ke2", "Ke5"])
    assert game_phases(rook_ending) == ["endgame"] * 4


def test_game_phases_never_raise():
    assert game_phases(make_game(moves_san=ILLEGAL)) == ["opening"] * len(ILLEGAL)
    assert game_phases(make_game(moves_san=[])) == []
    assert game_phases(make_game(initial_fen="not a fen", moves_san=["e4"])) == ["opening"]


def test_divide_matches_scalachess_on_boards():
    boards = [chess.Board()]
    assert engine.divide(boards) == (None, None)
    lone = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
    assert engine.divide([lone]) == (None, 0)  # middlegame skipped: straight into the endgame


# --------------------------------------------------------------------------- config / discovery / cache names
def test_engine_config_key_and_limit():
    assert EngineConfig().key() == "sf-d12"
    assert EngineConfig(depth=None, nodes=200000).key() == "sf-n200000"
    assert EngineConfig(depth=None).key() == "sf-d12"
    assert EngineConfig(depth=None, movetime=0.5).key() == "sf-t0.5"
    assert EngineConfig(depth=None).limit().depth == 12
    lim = EngineConfig(depth=None, nodes=5000).limit()
    assert lim.nodes == 5000 and lim.depth is None
    assert EngineConfig(workers=3).worker_count() == 3
    assert EngineConfig(workers=0).worker_count() == max(1, (os.cpu_count() or 2) - 1)


def test_find_stockfish_explicit_env_and_windows_dirs(tmp_path, monkeypatch):
    assert find_stockfish(str(tmp_path / "missing")) is None
    fake = tmp_path / "sf"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    assert find_stockfish(str(fake)) == str(fake)
    monkeypatch.setenv("CHESS_INSIGHTS_STOCKFISH", str(fake))
    assert find_stockfish() == str(fake)
    exe = tmp_path / "Program Files" / "Stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    assert windows_candidates({"ProgramFiles": str(tmp_path / "Program Files")}) == [str(exe)]
    assert windows_candidates({}) == []


def test_engine_name_falls_back_to_the_file_name(tmp_path):
    not_an_engine = tmp_path / "my-engine"
    not_an_engine.write_text("not executable")
    assert engine_name(str(not_an_engine)) == "my-engine"


def test_cache_file_names_are_safe_and_distinct():
    uuid = "0a1b2c3d-0000-11ef-8000-000000000001"
    assert cache_file_name(uuid) == f"{uuid}.json"
    a = cache_file_name("https://www.chess.com/game/live/123")
    b = cache_file_name("https://www.chess.com/game/live/123?x")
    assert a != b and "/" not in a and ":" not in a


def test_dict_round_trip_through_json():
    plies = [
        make_ply_eval(ply=0, tags=["missed_mate"], mate_before=2, win_before=97.54, clock_after=291.5, time_spent=8.5),
        make_ply_eval(ply=1, mover="black", is_user=False, judgement="blunder", cp_after=-340, accuracy=12.25),
    ]
    ev = make_game_eval("g1", plies, my_accuracy=81.5, opp_accuracy=None, depth=None)
    back = game_eval_from_dict(json.loads(json.dumps(game_eval_to_dict(ev))))
    assert back == ev
    assert game_eval_from_dict(game_eval_to_dict(make_game_eval("g2", []))).plies == []


# --------------------------------------------------------------------------- orchestration without Stockfish
class FakeEngine:
    id = {"name": "Fake 1"}

    def __init__(self, log):
        self.log = log

    def close(self):
        self.log.append("closed")


def test_in_process_run_isolates_failures_restarts_and_always_closes(monkeypatch, tmp_path):
    log = []
    monkeypatch.setattr(engine, "open_engine", lambda cfg: (log.append("opened"), FakeEngine(log))[1])
    good1, bad, good2 = (make_game(moves_san=QUEENS_GAMBIT) for _ in range(3))
    crashed = set()

    def fake_analyze(game, eng, cfg):
        if game is bad:
            raise RuntimeError("corrupt record")
        if game is good2 and game.game_id not in crashed:  # engine dies once: restarted, game retried
            crashed.add(game.game_id)
            raise chess.engine.EngineTerminatedError("engine died")
        return make_game_eval(game.game_id, [], engine=eng.id["name"])

    monkeypatch.setattr(engine, "analyze_game", fake_analyze)
    cfg = EngineConfig(path="/not/used", depth=6, workers=1)
    progress = []
    res = analyze_games([good1, bad, good2], cfg, cache_dir=tmp_path, progress=lambda d, t: progress.append((d, t)))
    assert set(res) == {good1.game_id, good2.game_id}
    assert progress[-1] == (3, 3)
    assert log == ["opened", "closed", "opened", "closed"]

    def interrupted(game, eng, cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "analyze_game", interrupted)
    log.clear()
    with pytest.raises(KeyboardInterrupt):
        analyze_games([make_game(moves_san=SICILIAN)], cfg)
    assert log == ["opened", "closed"]


class BrokenPool:
    """Stands in for ProcessPoolExecutor when worker processes die (or can't start)."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, *args):
        fut = concurrent.futures.Future()
        fut.set_exception(BrokenProcessPool("worker died"))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_broken_worker_pool_falls_back_to_one_engine_in_process(monkeypatch):
    log = []
    monkeypatch.setattr(engine, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(engine, "open_engine", lambda cfg: (log.append("opened"), FakeEngine(log))[1])
    monkeypatch.setattr(engine, "analyze_game", lambda game, eng, cfg: make_game_eval(game.game_id, []))
    games = [make_game(moves_san=QUEENS_GAMBIT) for _ in range(3)]
    res = analyze_games(games, EngineConfig(path="/not/used", workers=2))
    assert list(res) == [g.game_id for g in games]
    assert log == ["opened", "closed"]


def test_analyze_games_without_stockfish(monkeypatch):
    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: None)
    assert analyze_games([], EngineConfig()) == {}
    assert analyze_games([make_game(moves_san=SICILIAN[:6])], EngineConfig()) == {}  # too short: nothing to do
    with pytest.raises(FileNotFoundError):
        analyze_games([make_game(moves_san=SICILIAN)], EngineConfig())


# --------------------------------------------------------------------------- real Stockfish (tiny, depth 6)
@pytest.fixture(scope="module")
def sf(stockfish_path):
    cfg = EngineConfig(path=stockfish_path, depth=6, hash_mb=16)
    eng = engine.open_engine(cfg)
    yield eng, cfg
    eng.close()


class CountingEngine:
    def __init__(self, inner):
        self.inner, self.id, self.calls = inner, inner.id, 0

    def analyse(self, board, limit, **kw):
        self.calls += 1
        return self.inner.analyse(board, limit, **kw)


@pytest.mark.engine
def test_engine_name_reads_the_uci_id(stockfish_path):
    assert engine_name(stockfish_path).startswith("Stockfish")


@pytest.mark.engine
def test_scholars_mate_loss(sf):
    eng, cfg = sf
    game = make_game(color="black", outcome="loss", termination="checkmate", moves_san=SCHOLARS_MATE)
    counting = CountingEngine(eng)
    ev = analyze_game(game, counting, cfg)
    assert counting.calls == len(SCHOLARS_MATE)  # N + 1 positions, the final checkmate needs no engine
    assert ev.engine.startswith("Stockfish") and ev.depth == 6
    assert [p.ply for p in ev.plies] == list(range(7))
    assert [p.mover for p in ev.plies] == ["white", "black"] * 3 + ["white"]
    assert [p.is_user for p in ev.plies] == [False, True] * 3 + [False]
    for a, b in zip(ev.plies, ev.plies[1:]):  # each position evaluated once: after(i) == before(i + 1)
        assert (a.cp_after, a.mate_after) == (b.cp_before, b.mate_before)
        assert a.win_after == pytest.approx(100.0 - b.win_before)

    allowing = ev.plies[5]  # 3...Nf6??
    assert allowing.san == "Nf6" and allowing.is_user
    assert allowing.judgement == "blunder" and "allowed_mate" in allowing.tags
    assert allowing.mate_after == 1 and allowing.cp_after == MATE_CP  # White (the opponent) mates in 1
    assert allowing.win_after == pytest.approx(win_percent(-1000))
    assert allowing.cp_loss == CP_LOSS_CAP or allowing.cp_loss >= 500
    assert allowing.best_san and allowing.best_san != "Nf6"

    mating = ev.plies[6]  # 4.Qxf7#
    assert mating.best_san == "Qxf7#" and mating.accuracy == pytest.approx(100.0)
    assert mating.judgement is None and "missed_mate" not in mating.tags
    assert mating.cp_after == MATE_CP and mating.mate_after is None and mating.cp_loss == 0

    assert all(p.phase == "opening" for p in ev.plies)
    assert [p.clock_after for p in ev.plies] == game.clocks
    assert all(p.time_spent == pytest.approx(5.0) for p in ev.plies)
    assert ev.my_accuracy is not None and ev.opp_accuracy is not None and ev.my_accuracy < ev.opp_accuracy


@pytest.mark.engine
def test_missed_mate_in_one(sf):
    eng, cfg = sf
    game = make_game(color="white", moves_san=MISSED_MATE, time_class="daily", time_control="1/86400",
                     base_seconds=86400, with_clocks=False)
    ev = analyze_game(game, eng, cfg)
    miss = ev.plies[6]  # 4.Qf3 instead of 4.Qxf7#
    assert miss.san == "Qf3" and miss.is_user and miss.best_san == "Qxf7#"
    assert miss.mate_before == 1
    assert "missed_mate" in miss.tags and "missed_tactic" in miss.tags
    assert miss.judgement in ("mistake", "blunder")
    assert miss.win_before == pytest.approx(win_percent(1000))
    assert all(p.clock_after is None and p.time_spent is None for p in ev.plies)  # daily game: no clocks


@pytest.mark.engine
def test_chess960_and_illegal_games(sf, fixtures_dir):
    eng, cfg = sf
    raw = json.loads((fixtures_dir / "chesscom_archive_sample.json").read_text())["games"]
    games = parse_games(raw, raw[0]["white"]["username"])
    c960 = next(g for g in games if g.rules == "chess960")
    ev = analyze_game(c960, eng, cfg)
    assert ev is not None and len(ev.plies) == c960.plies
    assert analyze_game(make_game(moves_san=ILLEGAL), eng, cfg) is None
    assert analyze_game(make_game(moves_san=["e4", "--", "d4"]), eng, cfg) is None  # null moves aren't chess


@pytest.mark.engine
def test_analyze_games_parallel_cache_and_reuse(stockfish_path, tmp_path, monkeypatch):
    italian = make_game(moves_san=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6"])
    gambit = make_game(color="black", outcome="loss", moves_san=QUEENS_GAMBIT)
    sicilian = make_game(moves_san=SICILIAN)
    illegal = make_game(moves_san=ILLEGAL)
    variant = make_game(rules="crazyhouse", moves_san=SICILIAN)
    short = make_game(moves_san=SICILIAN[:8])
    corrupt = make_game(moves_san=QUEENS_GAMBIT)
    corrupt.clocks = None  # breaks inside the worker: that game is lost, the run is not
    good = [italian, gambit, sicilian]
    games = [italian, gambit, sicilian, illegal, variant, short, corrupt]
    cfg = EngineConfig(path=stockfish_path, depth=6, hash_mb=16, workers=2)

    progress = []
    first = analyze_games(games, cfg, cache_dir=tmp_path, progress=lambda d, t: progress.append((d, t)))
    assert list(first) == [g.game_id for g in good]
    assert progress[-1] == (5, 5)  # candidates: the three good games, the illegal one and the corrupt one
    files = sorted(p.name for p in (tmp_path / "sf-d6").glob("*.json"))
    assert files == sorted(cache_file_name(g.game_id) for g in good)
    assert first[gambit.game_id].plies[1].is_user and not first[gambit.game_id].plies[0].is_user

    def no_engine(*args, **kwargs):
        raise AssertionError("cached games must not start an engine")

    monkeypatch.setattr(engine, "_analyze_uncached", no_engine)
    monkeypatch.setattr(engine, "open_engine", no_engine)
    second = analyze_games(games[:-1], cfg, cache_dir=tmp_path)
    assert second == first
    assert list(analyze_games(good, cfg, cache_dir=tmp_path, max_games=2)) == [gambit.game_id, sicilian.game_id]

    # the same game seen from the other player's side reuses the cache with the perspective swapped
    flipped = make_game(game_id=gambit.game_id, color="white", moves_san=QUEENS_GAMBIT)
    ev = analyze_games([flipped], cfg, cache_dir=tmp_path)[gambit.game_id]
    assert ev.my_accuracy == first[gambit.game_id].opp_accuracy
    assert [p.is_user for p in ev.plies[:2]] == [True, False]

    # a cache file written for different moves is ignored (the game gets re-analysed)
    changed = make_game(game_id=sicilian.game_id, moves_san=SICILIAN[:-1] + ["e6"])
    with pytest.raises(AssertionError, match="must not start an engine"):
        analyze_games([changed], cfg, cache_dir=tmp_path)
