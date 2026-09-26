"""engine.py: Lichess formulas, phases, per-ply semantics, Stockfish runs, caching and workers.

Formula-by-formula parity with lila / scalachess lives in test_engine_parity.py.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_insights import engine
from chess_insights.engine import (
    CP_LOSS_CAP,
    MATE_CP,
    EngineConfig,
    PositionEval,
    analyze_game,
    analyze_games,
    build_game_eval,
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
# White sets up Qxf7# (after 3...Nf6??) and then plays 4.Qf3 instead.
MISSED_MATE = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qf3", "Bc5", "Nh3", "d6"]
QUEENS_GAMBIT = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"]
SICILIAN = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e5"]
ILLEGAL = ["e4", "e5", "Ke3", "Ke6", "Nf3", "Nf6", "Nc3", "Nc6", "d3", "d6"]
SRC = Path(engine.__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent


def white_pov(cps):
    """Lichess test helper: the start position (cp 15) plus White's cp after each ply, as win %."""
    return [win_percent(15)] + [win_percent(c) for c in cps]


# --------------------------------------------------------------------------- a scripted stand-in for Stockfish
class ScriptedEngine:
    """Scores the position after k plies with ``scores[k]`` (White's POV: cp, or ("mate", n)) and gives
    ``bests[k]`` (SAN) as its best move. Counts calls; closing it is recorded."""

    def __init__(self, scores, bests=None, name="Scripted 1"):
        self.scores, self.bests = scores, dict(bests or {})
        self.id = {"name": name}
        self.calls, self.closed = 0, False

    def analyse(self, board, limit, **kw):
        self.calls += 1
        k = len(board.move_stack)
        s = self.scores[k]
        score = chess.engine.Mate(s[1]) if isinstance(s, tuple) else chess.engine.Cp(s)
        info = {"score": chess.engine.PovScore(score, chess.WHITE)}
        if self.bests.get(k):
            info["pv"] = [board.parse_san(self.bests[k])]
        return info

    def close(self):
        self.closed = True


def scripted(game, scores, bests=None):
    return analyze_game(game, ScriptedEngine(scores, bests), EngineConfig(depth=12))


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
    avg = white_pov([-50, 15] * 50)
    assert game_accuracy(avg, "white") == pytest.approx(77.38, abs=0.05)
    assert game_accuracy(avg, "black") == pytest.approx(77.38, abs=0.05)
    assert game_accuracy(white_pov([-135, 15] * 50), "white") == pytest.approx(55.0, abs=0.05)


def test_game_accuracy_volatile_sequence_and_edge_cases():
    blunder = white_pov([15] * 20 + [-900])  # 20 perfect moves, then a White blunder
    white = game_accuracy(blunder, "white")
    assert white == pytest.approx(50.0, abs=5.0)
    assert game_accuracy(blunder, "black") == pytest.approx(100.0, abs=1.0)
    per_move = [move_accuracy(blunder[i], blunder[i + 1]) for i in range(0, len(blunder) - 1, 2)]
    assert white < sum(per_move) / len(per_move)
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
    # White's back rank drops to 3 pieces (Ra1, Ke1, Rh1) with 7.Qd2: that move is the first middlegame move
    developed = "e4 e5 Nf3 Nc6 Bc4 Bc5 Nc3 Nf6 d3 d6 Be3 Be6 Qd2 Qd7 O-O O-O".split()
    phases = game_phases(make_game(moves_san=developed))
    assert phases[:12] == ["opening"] * 12 and phases[12:] == ["middlegame"] * 4
    traded = "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nxd4 Qxd4 Qf6 Qxf6 Nxf6 Bc4 Bc5 Be3 Bxe3 fxe3 Nxe4".split()
    assert game_phases(make_game(moves_san=traded))[-1] == "middlegame"
    rook_ending = make_game(initial_fen="8/8/8/4k3/8/8/8/R3K3 w - - 0 1", moves_san=["Ra7", "Kd5", "Ke2", "Ke5"])
    assert game_phases(rook_ending) == ["endgame"] * 4


def test_game_phases_never_raise():
    assert game_phases(make_game(moves_san=ILLEGAL)) == ["opening"] * len(ILLEGAL)
    assert game_phases(make_game(moves_san=[])) == []
    assert game_phases(make_game(initial_fen="not a fen", moves_san=["e4"])) == ["opening"]


def test_divide_matches_scalachess_on_boards():
    assert engine.divide([chess.Board()]) == (None, None)
    lone = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
    assert engine.divide([lone]) == (None, 0)  # middlegame skipped: straight into the endgame


# --------------------------------------------------------------------------- per-ply semantics (scripted engine)
# White-POV scores of the 13 positions of SICILIAN, and what each ply should be tagged with.
SCRIPT = [15, 20, ("mate", 3), 200, 180, 300, 300, -100, -100, -450, 500, 0, 0]
SCRIPT_BESTS = {6: "Qxd4", 9: "Nxe4"}  # 4.Nxd4 missed 4.Qxd4; after 5.Nc3?? Black wins the e4 pawn with a capture
SCRIPT_TAGS = {1: ["allowed_mate"], 2: ["missed_mate"], 6: ["missed_tactic"], 8: ["hung_material"],
               9: ["missed_tactic", "thrown_win"], 10: ["thrown_win"]}  # 5...a6?? missed 5...Nxe4 too


def test_ply_evals_line_up_with_positions_and_use_the_movers_point_of_view():
    game = make_game(color="black", moves_san=SICILIAN)
    ev = scripted(game, SCRIPT, SCRIPT_BESTS)
    assert [p.ply for p in ev.plies] == list(range(12)) and [p.san for p in ev.plies] == SICILIAN
    assert [p.mover for p in ev.plies] == ["white", "black"] * 6
    assert [p.is_user for p in ev.plies] == [False, True] * 6
    for i, p in enumerate(ev.plies):
        before, after = SCRIPT[i], SCRIPT[i + 1]
        white_cp = lambda s: (MATE_CP if s[1] > 0 else -MATE_CP) if isinstance(s, tuple) else s  # noqa: E731
        assert (p.cp_before, p.cp_after) == (white_cp(before), white_cp(after))  # White's POV, position i then i+1
        assert p.mate_before == (before[1] if isinstance(before, tuple) else None)
        sign = 1 if p.mover == "white" else -1
        assert p.win_before == pytest.approx(win_percent(sign * p.cp_before))  # the mover's POV
        assert p.win_after == pytest.approx(win_percent(sign * p.cp_after))
        assert p.cp_loss == min(CP_LOSS_CAP, max(0, sign * (p.cp_before - p.cp_after)))
        assert p.accuracy == pytest.approx(move_accuracy(p.win_before, p.win_after))
    assert ev.plies[6].best_san == "Qxd4" and ev.plies[9].best_san == "Nxe4"  # best move of the position BEFORE
    assert ev.plies[0].best_san is None  # no PV from the engine
    for a, b in zip(ev.plies, ev.plies[1:]):  # one evaluation per position: after(i) == before(i + 1)
        assert (a.cp_after, a.mate_after) == (b.cp_before, b.mate_before)


def test_tags_on_constructed_positions():
    ev = scripted(make_game(moves_san=SICILIAN), SCRIPT, SCRIPT_BESTS)
    assert {p.ply: p.tags for p in ev.plies if p.tags} == SCRIPT_TAGS
    judgements = [p.judgement for p in ev.plies]
    assert judgements[1] == "blunder"  # 1...c5 allows a forced mate (MateCreated from an even position)
    assert judgements[2] == "blunder"  # 2.Nf3 throws away mate in 3 for +2 (MateLost, not crushing)
    assert judgements[6] == judgements[8] == judgements[9] == judgements[10] == "blunder"
    assert judgements[0] is None and judgements[3] is None and judgements[11] is None


def test_mate_signs_for_black_and_mate_in_n():
    # White-POV mate scores: negative = Black mates. 2...d6 gives Black mate in 2; 3...cxd4?? lets it go.
    scores = [15, 20, 25, 30, ("mate", -2), ("mate", -1), -300, ("mate", -3), -2000, -2000, -2000, -2000, -2000]
    ev = scripted(make_game(moves_san=SICILIAN), scores)
    doomed = ev.plies[4]  # 3.d4: White is being mated before and after
    assert doomed.mover == "white" and (doomed.mate_before, doomed.mate_after) == (-2, -1)
    assert doomed.tags == [] and doomed.judgement is None  # already lost: nothing new was allowed
    lost = ev.plies[5]  # 3...cxd4: Black had mate in 1 (mover POV +1) and played for +3 pawns
    assert lost.mover == "black" and lost.mate_before == -1 and lost.tags == ["missed_mate"]
    assert lost.win_before == pytest.approx(win_percent(MATE_CP)) and lost.win_after == pytest.approx(win_percent(300))
    assert lost.judgement == "blunder"  # MateLost, and +3 is not crushing (<= 700 cp)
    allowed = ev.plies[6]  # 4.Nxd4: from -3 pawns into a forced mate
    assert allowed.mover == "white" and allowed.mate_after == -3 and allowed.tags == ["allowed_mate"]
    assert allowed.win_after == pytest.approx(win_percent(-MATE_CP)) and allowed.judgement == "blunder"
    still_crushing = ev.plies[7]  # 4...Nf6: mate in 3 let go, but still +20 pawns
    assert still_crushing.tags == ["missed_mate"] and still_crushing.judgement == "inaccuracy"


def test_moves_the_engine_itself_prefers_get_no_judgement_and_no_tags():
    """The engine's own best move can't be a mistake (Lichess only judges moves that differ from it), so an eval
    that collapses after it must not produce allowed_mate / thrown_win / missed_mate tags either."""
    scores = [15, 20, 500, ("mate", -3), -300, 0, 0, 0, 0, 0, 0, 0, 0]
    ev = scripted(make_game(moves_san=SICILIAN), scores, {2: "Nf3", 3: "d6"})
    nf3, d6 = ev.plies[2], ev.plies[3]
    assert nf3.best_san == "Nf3" and nf3.judgement is None and nf3.tags == []
    assert d6.best_san == "d6" and d6.judgement is None and d6.tags == []


def test_judgement_uses_unclamped_centipawns():
    """Lichess's CpAdvice compares winning chances of the raw centipawns: +20 -> +4.5 pawns loses 0.319 (a blunder);
    with both clamped at +10 pawns it would be 0.27 (a mistake). Stored cps stay clamped at +-MATE_CP."""
    scores = [15, 20, 25, 2000, 2000, 450, -2000, -2000, -450, -450, -450, -450, -450]
    ev = scripted(make_game(moves_san=SICILIAN), scores)
    white, black = ev.plies[4], ev.plies[7]  # 3.d4 and 4...Nf6
    assert (white.cp_before, white.cp_after) == (MATE_CP, 450)
    assert white.judgement == "blunder" and black.judgement == "blunder"
    assert [p.judgement for p in ev.plies[2:4]] == [None, None]


def test_checkmate_on_the_board_and_game_accuracy():
    scores = [15, 30, 25, 40, 20, 60, ("mate", 1), None]
    ev = scripted(make_game(color="black", moves_san=SCHOLARS_MATE), scores)
    mating = ev.plies[6]
    assert mating.cp_after == MATE_CP and mating.mate_after is None and mating.judgement is None
    assert mating.accuracy == 100.0 and mating.tags == [] and mating.cp_loss == 0
    wp = [win_percent(c) for c in (15, 30, 25, 40, 20, 60, MATE_CP)]  # the checkmate position is left out
    assert ev.opp_accuracy == pytest.approx(game_accuracy(wp, "white"))
    assert ev.my_accuracy == pytest.approx(game_accuracy(wp, "black"))


def test_time_spent_with_increment_first_moves_and_gaps():
    # 3+2: chess.com's clock after a move already includes the increment; the first moves are close to free
    clocks = [182.0, 181.5, 175.0, 170.5, None, 168.0, 160.0, 165.0, 150.0, 140.0, 141.0, 130.0]
    game = make_game(moves_san=SICILIAN, time_control="180+2", base_seconds=180, increment=2, clocks=clocks)
    spent = [p.time_spent for p in scripted(game, [0] * 13).plies]
    assert spent[:4] == [0.0, pytest.approx(0.5), 9.0, pytest.approx(13.0)]
    assert spent[4] is None and spent[6] is None  # no clock for ply 4: neither it nor ply 6 can be measured
    assert spent[5] == pytest.approx(4.5) and spent[7] == pytest.approx(5.0)
    assert spent[9] == pytest.approx(27.0)  # 165 - 140 + 2
    assert spent[8] == pytest.approx(12.0) and spent[10] == pytest.approx(11.0) and spent[11] == pytest.approx(12.0)
    assert all(s is None or s >= 0 for s in spent)
    daily = make_game(moves_san=SICILIAN, time_class="daily", time_control="1/86400", base_seconds=86400)
    assert all(p.time_spent is None for p in scripted(daily, [0] * 13).plies)


def test_black_to_move_first_from_a_set_up_position():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 3"
    moves = ["Nf6", "Nc3", "Bc5", "Bc4", "d6", "d3", "O-O", "O-O", "h6", "h3"]
    game = make_game(color="black", initial_fen=fen, moves_san=moves)
    ev = scripted(game, [15, -20, 0, 10, 5, 20, 30, 25, 20, 30, 40])
    assert [p.mover for p in ev.plies[:2]] == ["black", "white"] and ev.plies[0].is_user
    assert ev.my_accuracy == pytest.approx(game_accuracy([win_percent(c) for c in
                                                           (15, -20, 0, 10, 5, 20, 30, 25, 20, 30, 40)],
                                                          "black", white_moves_first=False))


def test_build_game_eval_rejects_positions_that_do_not_fit_the_game():
    game = make_game(moves_san=QUEENS_GAMBIT)
    assert build_game_eval(game, [PositionEval(cp=0)] * 10) is None  # 11 positions needed
    assert build_game_eval(make_game(moves_san=ILLEGAL), [PositionEval(cp=0)] * 11) is None
    ev = build_game_eval(game, [PositionEval(cp=0, best="e2e5")] * 11)  # an illegal "best move" is ignored
    assert ev is not None and all(p.best_san is None for p in ev.plies)


# --------------------------------------------------------------------------- config / discovery / cache names
def test_engine_config_key_and_limit():
    assert EngineConfig().key() == "sf-d12"
    assert EngineConfig(depth=None, nodes=200000).key() == "sf-n200000"
    assert EngineConfig(depth=None).key() == "sf-d12"
    assert EngineConfig(depth=None, movetime=0.5).key() == "sf-t0.5"
    assert EngineConfig(threads=2, hash_mb=16).key() == "sf-d12-th2-h16"  # settings that change the numbers
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


WINDOWS_RESERVED = ("CON", "con", "PRN", "AUX", "NUL", "COM1", "lpt9", "CON.x")


def test_cache_file_names_are_safe_and_distinct_on_windows_and_macos():
    uuid = "0a1b2c3d-0000-11ef-8000-000000000001"
    assert cache_file_name(uuid) == f"{uuid}.json"
    ids = ["https://www.chess.com/game/live/123", "https://www.chess.com/game/live/123?x", "pgn:2024.01.01:a:b",
           "ABC", "abc", "Abc", "a b", "a_b", "...", "", "x" * 300, *WINDOWS_RESERVED]
    names = [cache_file_name(i) for i in ids]
    assert len({n.lower() for n in names}) == len(ids)  # distinct even on case-insensitive file systems
    for name in names:
        stem = name[: -len(".json")]
        assert name.endswith(".json") and 5 < len(name) <= 120
        assert not set(name) & set('<>:"/\\|?* ') and not stem.endswith((".", " "))
        assert stem.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                                                  *(f"LPT{i}" for i in range(10))}


def test_engine_labels_become_directory_names(tmp_path):
    cfg = EngineConfig(depth=12)
    assert engine.cache_root(tmp_path, cfg, "Stockfish 16") == tmp_path / "stockfish-16" / "sf-d12"
    assert engine.cache_root(tmp_path, cfg, "Stockfish 16.1") != engine.cache_root(tmp_path, cfg, "Stockfish 16")
    assert engine.cache_root(tmp_path, cfg, "Stockfish dev-20240101-abc:1").parent.name.isascii()


def test_dict_round_trip_through_json():
    plies = [
        make_ply_eval(ply=0, tags=["missed_mate"], mate_before=2, win_before=97.54, clock_after=291.5, time_spent=8.5),
        make_ply_eval(ply=1, mover="black", is_user=False, judgement="blunder", cp_after=-340, accuracy=12.25),
    ]
    ev = make_game_eval("g1", plies, my_accuracy=81.5, opp_accuracy=None, depth=None)
    back = game_eval_from_dict(json.loads(json.dumps(game_eval_to_dict(ev))))
    assert back == ev
    assert game_eval_from_dict(game_eval_to_dict(make_game_eval("g2", []))).plies == []


def test_position_evals_round_trip_through_json():
    evs = [PositionEval(cp=35, best="e2e4"), PositionEval(mate=-3, best="g8f6"), PositionEval(cp=-1000, checkmate=True),
           PositionEval(cp=0)]
    assert [PositionEval.from_json(json.loads(json.dumps(e.to_json()))) for e in evs] == evs
    with pytest.raises((TypeError, ValueError, KeyError)):
        PositionEval.from_json({"cp": "lots"})


# --------------------------------------------------------------------------- orchestration with fake engines
class FakeEngine:
    """Any position: +20 cp for the side to move, best move = first legal move. Records open / close."""

    delay = 0.0  # seconds per search

    def __init__(self, log, name="Fake 1"):
        self.log, self.id = log, {"name": name}
        self.calls = 0
        log.append("opened")

    def analyse(self, board, limit, **kw):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        move = min(board.legal_moves, key=lambda m: m.uci())
        return {"score": chess.engine.PovScore(chess.engine.Cp(20), board.turn), "pv": [move]}

    def close(self):
        self.log.append("closed")


@pytest.fixture
def fake_engines(monkeypatch):
    """open_engine / engine_name replaced: returns the list of (event) log and a dict to change the name."""
    log, state = [], {"name": "Fake 1"}
    monkeypatch.setattr(engine, "open_engine", lambda cfg: FakeEngine(log, state["name"]))
    monkeypatch.setattr(engine, "engine_name", lambda path: state["name"])
    return log, state


def cfg_fake(**kw):
    return EngineConfig(**{"path": "/fake/stockfish", "depth": 6, "workers": 1, **kw})


def test_in_process_run_isolates_failures_restarts_and_always_closes(fake_engines, monkeypatch, tmp_path):
    log, _ = fake_engines
    good1, bad, good2 = (make_game(moves_san=QUEENS_GAMBIT) for _ in range(3))
    crashed = set()
    real = engine.evaluate_game

    def flaky(game, eng, cfg):
        if game is bad:
            raise RuntimeError("corrupt record")
        if game is good2 and game.game_id not in crashed:  # the engine dies once: restarted, game retried
            crashed.add(game.game_id)
            raise chess.engine.EngineTerminatedError("engine died")
        return real(game, eng, cfg)

    monkeypatch.setattr(engine, "evaluate_game", flaky)
    progress = []
    res = analyze_games(
        [good1, bad, good2], cfg_fake(), cache_dir=tmp_path, progress=lambda d, t: progress.append((d, t))
    )
    assert set(res) == {good1.game_id, good2.game_id} and res[good1.game_id].engine == "Fake 1"
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert log == ["opened", "closed", "opened", "closed"]

    def interrupted(game, eng, cfg):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "evaluate_game", interrupted)
    log.clear()
    with pytest.raises(KeyboardInterrupt):
        analyze_games([make_game(moves_san=SICILIAN)], cfg_fake())
    assert log == ["opened", "closed"]


def test_worker_threads_analyse_in_parallel_and_close_every_engine(fake_engines, tmp_path, monkeypatch):
    log, _ = fake_engines
    monkeypatch.setattr(FakeEngine, "delay", 0.003)
    games = [make_game(moves_san=QUEENS_GAMBIT) for _ in range(7)]
    progress = []
    res = analyze_games(games, cfg_fake(workers=3), cache_dir=tmp_path, progress=lambda d, t: progress.append((d, t)))
    assert list(res) == [g.game_id for g in games]  # oldest first, like the input
    assert [d for d, _ in progress] == list(range(1, 8)) and {t for _, t in progress} == {7}
    assert log.count("opened") == 3 and log.count("closed") == 3
    serial = analyze_games(games, cfg_fake(), cache_dir=None)
    assert serial == res  # same numbers whichever engine analysed which game
    assert not [t for t in threading.enumerate() if t.name.startswith("stockfish-worker")]


def test_worker_threads_stop_and_close_engines_on_keyboard_interrupt(fake_engines):
    log, _ = fake_engines
    games = [make_game(moves_san=QUEENS_GAMBIT) for _ in range(30)]

    def stop(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        analyze_games(games, cfg_fake(workers=3), progress=stop)
    assert log.count("opened") == log.count("closed") >= 1
    deadline = time.time() + 2
    while time.time() < deadline and [t for t in threading.enumerate() if t.name.startswith("stockfish-worker")]:
        time.sleep(0.01)
    assert not [t for t in threading.enumerate() if t.name.startswith("stockfish-worker")]


def test_an_engine_that_cannot_start_fails_every_game_once_not_per_retry(monkeypatch, caplog):
    starts = []

    def broken(cfg):
        starts.append(1)
        raise FileNotFoundError("no such file: /fake/stockfish")

    monkeypatch.setattr(engine, "open_engine", broken)
    monkeypatch.setattr(engine, "engine_name", lambda path: "stockfish")
    games = [make_game(moves_san=QUEENS_GAMBIT) for _ in range(6)]
    assert analyze_games(games, cfg_fake(workers=2)) == {}
    assert len(starts) <= 2  # one start attempt per worker, not one per game
    assert len([r for r in caplog.records if r.levelname == "WARNING" and "no such file" in r.getMessage()]) == 1


def test_analyze_games_without_stockfish(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: None)
    assert analyze_games([], EngineConfig()) == {}
    assert analyze_games([make_game(moves_san=SICILIAN[:6])], EngineConfig()) == {}  # too short: nothing to do
    with pytest.raises(FileNotFoundError):
        analyze_games([make_game(moves_san=SICILIAN)], EngineConfig())


def test_cached_games_load_without_stockfish(fake_engines, monkeypatch, tmp_path):
    games = [make_game(moves_san=SICILIAN) for _ in range(2)]
    first = analyze_games(games, cfg_fake(), cache_dir=tmp_path)
    monkeypatch.setattr(engine, "find_stockfish", lambda explicit=None: None)
    again = analyze_games(games, EngineConfig(depth=6), cache_dir=tmp_path)
    assert again == first and again[games[0].game_id].engine == "Fake 1"
    with pytest.raises(FileNotFoundError):  # a game that isn't cached still needs the engine
        analyze_games(games + [make_game(moves_san=SICILIAN)], EngineConfig(depth=6), cache_dir=tmp_path)


def test_max_games_takes_the_most_recent_analysable_games(fake_engines):
    old, mid, new = (make_game(moves_san=SICILIAN, minutes_ago=m) for m in (300, 200, 100))
    newest_illegal = make_game(moves_san=ILLEGAL, minutes_ago=10)
    newest_short = make_game(moves_san=SICILIAN[:4], minutes_ago=5)
    newest_variant = make_game(moves_san=SICILIAN, rules="crazyhouse", minutes_ago=1)
    progress = []
    res = analyze_games([new, old, newest_illegal, mid, newest_short, newest_variant], cfg_fake(), max_games=2,
                        progress=lambda d, t: progress.append((d, t)))
    assert list(res) == [mid.game_id, new.game_id]
    assert progress[-1] == (2, 2)
    assert analyze_games([old, new], cfg_fake(), max_games=0) == {}


def test_balanced_sample_replaces_a_game_that_does_not_replay(fake_engines, tmp_path):
    """--engine-sample balanced: the same number of each format; an illegal game gives way to the next of its format."""
    bullet = [make_game(moves_san=SICILIAN, time_class="bullet", minutes_ago=m) for m in (1, 2, 3, 4, 5, 6)]
    rapid = [make_game(moves_san=SICILIAN, time_class="rapid", minutes_ago=m) for m in (100, 200, 300)]
    broken = make_game(moves_san=ILLEGAL, time_class="rapid", minutes_ago=50)  # the newest rapid game
    games = bullet + rapid + [broken]
    res = analyze_games(games, cfg_fake(), cache_dir=tmp_path, max_games=4, sample="balanced")
    by_id = {g.game_id: g for g in games}
    assert sorted(by_id[i].time_class for i in res) == ["bullet", "bullet", "rapid", "rapid"]
    assert broken.game_id not in res and {rapid[0].game_id, rapid[1].game_id} <= set(res)
    assert list(res) == sorted(res, key=lambda i: by_id[i].end_time)  # oldest first, as before
    recent = analyze_games(games, cfg_fake(), cache_dir=tmp_path, max_games=4)  # the default: bullet only
    assert {by_id[i].time_class for i in recent} == {"bullet"}
    only_rapid = analyze_games(games, cfg_fake(), cache_dir=tmp_path, max_games=4, time_classes=["rapid"])
    assert set(only_rapid) == {g.game_id for g in rapid}


def test_cache_is_keyed_by_engine_version(fake_engines, tmp_path):
    log, state = fake_engines
    game = make_game(moves_san=SICILIAN)
    assert analyze_games([game], cfg_fake(), cache_dir=tmp_path)[game.game_id].engine == "Fake 1"
    log.clear()
    assert analyze_games([game], cfg_fake(), cache_dir=tmp_path)[game.game_id].engine == "Fake 1"
    assert log == []  # cached
    state["name"] = "Fake 2"  # Stockfish upgraded: its numbers differ, so the old analysis is not reused
    assert analyze_games([game], cfg_fake(), cache_dir=tmp_path)[game.game_id].engine == "Fake 2"
    assert log == ["opened", "closed"]
    assert analyze_games([game], cfg_fake(depth=8), cache_dir=tmp_path) and log.count("opened") == 2  # other limit


def test_cache_stores_engine_output_and_rebuilds_game_specific_fields(fake_engines, tmp_path):
    log, _ = fake_engines
    game = make_game(color="black", moves_san=SICILIAN)
    first = analyze_games([game], cfg_fake(), cache_dir=tmp_path)[game.game_id]
    log.clear()
    # the same game re-parsed with other clocks, and seen from White's side: nothing stale comes back
    other = make_game(game_id=game.game_id, color="white", moves_san=SICILIAN, clocks=[290.0 - i for i in range(12)])
    again = analyze_games([other], cfg_fake(), cache_dir=tmp_path)[game.game_id]
    assert log == []
    assert [p.clock_after for p in again.plies] == other.clocks
    assert [p.time_spent for p in again.plies][2:4] == [pytest.approx(2.0)] * 2
    assert [p.is_user for p in again.plies[:2]] == [True, False]
    assert (again.my_accuracy, again.opp_accuracy) == (first.opp_accuracy, first.my_accuracy)
    assert [p.cp_after for p in again.plies] == [p.cp_after for p in first.plies]


def test_corrupt_truncated_and_foreign_cache_files_are_ignored(fake_engines, tmp_path):
    log, _ = fake_engines
    games = [make_game(moves_san=SICILIAN) for _ in range(4)]
    first = analyze_games(games, cfg_fake(), cache_dir=tmp_path)
    files = sorted(tmp_path.rglob("*.json"))
    assert len(files) == 4
    payloads = [json.loads(f.read_text()) for f in files]
    files[0].write_text("{not json")
    files[1].write_text(json.dumps([1, 2, 3]))
    data = payloads[2]
    key = "positions" if "positions" in data else "eval"
    if key == "positions":
        data["positions"] = data["positions"][:-3]
    else:
        data["eval"]["plies"] = data["eval"]["plies"][:-3]
    files[2].write_text(json.dumps(data))
    files[3].write_bytes(b"\xff\xfe\x00garbage")
    log.clear()
    again = analyze_games(games, cfg_fake(), cache_dir=tmp_path)
    assert again == first and log == ["opened", "closed"]
    assert all(len(ev.plies) == 12 for ev in again.values())
    assert not [p for p in tmp_path.rglob("*") if p.name.startswith(".tmp")]  # atomic writes leave no temp files


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
    no_king = make_game(initial_fen="8/8/8/8/8/8/4P3/4K3 w - - 0 1", moves_san=["e4", "Kd2"] * 5)
    assert analyze_game(no_king, eng, cfg) is None  # would crash Stockfish


@pytest.mark.engine
def test_analyze_games_parallel_cache_and_reuse(stockfish_path, tmp_path, monkeypatch):
    italian = make_game(moves_san=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6"])
    gambit = make_game(color="black", outcome="loss", moves_san=QUEENS_GAMBIT)
    sicilian = make_game(moves_san=SICILIAN)
    illegal = make_game(moves_san=ILLEGAL)
    variant = make_game(rules="crazyhouse", moves_san=SICILIAN)
    short = make_game(moves_san=SICILIAN[:8])
    broken = make_game(moves_san=QUEENS_GAMBIT)
    good = [italian, gambit, sicilian]
    games = [italian, gambit, sicilian, illegal, variant, short, broken]
    cfg = EngineConfig(path=stockfish_path, depth=6, hash_mb=16, workers=2)
    real = engine.evaluate_game

    def evaluate(game, eng, cfg):
        if game is broken:
            raise RuntimeError("fails inside a worker: that game is lost, the run is not")
        return real(game, eng, cfg)

    monkeypatch.setattr(engine, "evaluate_game", evaluate)
    progress = []
    first = analyze_games(games, cfg, cache_dir=tmp_path, progress=lambda d, t: progress.append((d, t)))
    assert list(first) == [g.game_id for g in good]
    assert progress[-1] == (4, 4)  # the three good games and the broken one; illegal / variant / short: skipped
    root = engine.cache_root(tmp_path, cfg, engine_name(stockfish_path))
    assert sorted(p.name for p in root.glob("*.json")) == sorted(cache_file_name(g.game_id) for g in good)
    assert first[gambit.game_id].plies[1].is_user and not first[gambit.game_id].plies[0].is_user

    def no_engine(*args, **kwargs):
        raise AssertionError("cached games must not start an engine")

    monkeypatch.setattr(engine, "_analyze_uncached", no_engine)
    monkeypatch.setattr(engine, "open_engine", no_engine)
    second = analyze_games(games[:-1], cfg, cache_dir=tmp_path)
    assert second == first
    assert list(analyze_games(good, cfg, cache_dir=tmp_path, max_games=2)) == [gambit.game_id, sicilian.game_id]

    flipped = make_game(game_id=gambit.game_id, color="white", moves_san=QUEENS_GAMBIT)
    ev = analyze_games([flipped], cfg, cache_dir=tmp_path)[gambit.game_id]
    assert ev.my_accuracy == first[gambit.game_id].opp_accuracy
    assert [p.is_user for p in ev.plies[:2]] == [True, False]

    changed = make_game(game_id=sicilian.game_id, moves_san=SICILIAN[:-1] + ["Nc6"])
    with pytest.raises(AssertionError, match="must not start an engine"):
        analyze_games([changed], cfg, cache_dir=tmp_path)


def _stockfish_descendants() -> list[int]:
    """Live (non-zombie) Stockfish processes started by this process or its children (Linux only)."""
    parents, names = {}, {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        state, ppid = stat[stat.rindex(")") + 2 :].split()[:2]
        if state != "Z":
            parents[int(d.name)], names[int(d.name)] = int(ppid), name
    ours, frontier = set(), {os.getpid()}
    while frontier:
        frontier = {pid for pid, ppid in parents.items() if ppid in frontier} - ours
        ours |= frontier
    return sorted(pid for pid in ours if "stockfish" in names[pid].lower())


@pytest.mark.engine
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_keyboard_interrupt_leaves_no_stockfish_running(stockfish_path):
    before = _stockfish_descendants()
    games = [make_game(moves_san=SICILIAN * 1) for _ in range(6)]

    def interrupt(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        analyze_games(games, EngineConfig(path=stockfish_path, depth=8, hash_mb=16, workers=3), progress=interrupt)
    assert _stockfish_descendants() == before


@pytest.mark.engine
def test_parallel_run_from_a_script_without_a_main_guard(stockfish_path, tmp_path):
    """Windows and macOS start worker processes with "spawn", which re-runs an unguarded script in every
    worker. Analysis must work (and print nothing alarming) when called from such a script."""
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(SRC)!r}, {str(TESTS)!r}]\n"
        "from chess_insights.engine import EngineConfig, analyze_games\n"
        "from factories import make_game\n"
        f"games = [make_game(moves_san={QUEENS_GAMBIT!r}) for _ in range(3)]\n"
        f"res = analyze_games(games, EngineConfig(path={stockfish_path!r}, depth=4, hash_mb=16, workers=2))\n"
        "print('RESULT', len(res))\n"
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["RESULT", "3"]
    assert "Traceback" not in out.stderr and "Error" not in out.stderr


@pytest.mark.engine
def test_ctrl_c_during_a_parallel_run_is_quiet_and_leaves_nothing_running(stockfish_path, tmp_path):
    """What a user sees after Ctrl+C: the KeyboardInterrupt, nothing else (no asyncio "Future exception was
    never retrieved" noise from the engines being killed mid-search), and the interpreter exits promptly."""
    script = tmp_path / "interrupted.py"
    script.write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(SRC)!r}, {str(TESTS)!r}]\n"
        "from chess_insights.engine import EngineConfig, analyze_games\n"
        "from factories import make_game\n"
        f"games = [make_game(moves_san={SICILIAN!r}) for _ in range(8)]\n"
        "def stop(done, total):\n"
        "    raise KeyboardInterrupt\n"
        "try:\n"
        f"    cfg = EngineConfig(path={stockfish_path!r}, depth=10, hash_mb=16, workers=3)\n"
        "    analyze_games(games, cfg, progress=stop)\n"
        "except KeyboardInterrupt:\n"
        "    print('INTERRUPTED')\n"
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.split() == ["INTERRUPTED"]
    assert out.stderr == ""


# --------------------------------------------------------------------------- verifier findings
QUEEN_UP = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"
REPETITION = ["Kd1", "Kd8", "Ke1", "Ke8", "Kd1", "Kd8", "Ke1", "Ke8"]  # the start position comes back a third time


def test_a_threefold_repetition_draw_ends_on_a_drawn_position_without_a_search():
    """Black, a queen down, holds the draw by repetition (chess.com ends the game there). Stockfish would score
    the final position +10 (it doesn't treat the root as drawn), charging the drawing move 1000 cp."""
    game = make_game(color="black", outcome="draw", termination="repetition", initial_fen=QUEEN_UP,
                     moves_san=REPETITION)
    # after 4.Ke1 Stockfish (given the moves) sees that ...Ke8 repeats: 0; it scores the final position +10
    scripted_engine = ScriptedEngine([1000] * 7 + [0, 1000], {k: "Qa7" for k in range(0, 9, 2)})
    ev = analyze_game(game, scripted_engine, EngineConfig(depth=12))
    assert scripted_engine.calls == 8  # the drawn final position is not searched
    last = ev.plies[-1]
    assert (last.san, last.cp_before, last.cp_after, last.cp_loss) == ("Ke8", 0, 0, 0)
    assert last.accuracy == 100.0 and last.judgement is None
    assert ev.plies[-2].judgement == "blunder"  # White's 4.Ke1 let a won game be drawn
    # an analysis cached before this rule (the final position searched) is corrected when it is read
    searched = [PositionEval(cp=1000, best="a1a7")] * 9
    assert build_game_eval(game, searched).plies[-1].cp_loss == 0
    # the same moves in a game that went on being scored (nobody claimed the draw): the search stands
    unclaimed = make_game(color="black", outcome="loss", initial_fen=QUEEN_UP, moves_san=REPETITION)
    assert build_game_eval(unclaimed, searched).plies[-1].cp_after == 1000


def test_fifty_move_and_seventy_five_move_draws():
    fifty = make_game(color="white", outcome="draw", termination="50move",
                      initial_fen="4k3/8/8/8/8/8/8/Q3K3 w - - 98 80", moves_san=["Kd1", "Kd8"])
    assert build_game_eval(fifty, [PositionEval(cp=900)] * 3).plies[-1].cp_after == 0
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 149 80")
    board.push_san("Kd1")
    assert engine.ends_drawn_by_rule(make_game(outcome="win"), board)  # 75 moves: over whatever the result
    mate = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        mate.push_san(san)
    assert not engine.ends_drawn_by_rule(make_game(outcome="draw"), mate)


def test_hung_material_needs_the_reply_to_win_more_than_the_move_did():
    """A capture answered by a recapture is a trade, not material left hanging; a capture of a defended pawn
    with the queen is."""
    trade = make_game(moves_san=["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6", "Nf3", "Bf5"])
    ev = scripted(trade, [30, 40, 50, -400] + [-400] * 7, {2: "Nc3", 3: "Qxd5"})
    assert ev.plies[2].judgement == "blunder" and "hung_material" not in ev.plies[2].tags
    bad = make_game(moves_san=["e4", "e5", "Qh5", "Nc6", "Qxe5+", "Nxe5", "d4", "Ng6", "Nf3", "Nf6"])
    ev = scripted(bad, [30, 30, 0, 0, 0, -900] + [-900] * 5, {4: "Bc4", 5: "Nxe5"})
    assert ev.plies[4].judgement == "blunder" and "hung_material" in ev.plies[4].tags
    board = chess.Board("4k3/1P6/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert engine.material_gain(board, board.parse_san("exd6")) == 1  # en passant
    assert engine.material_gain(board, board.parse_san("b8=Q+")) == 8  # a promotion adds a queen for a pawn


def test_an_unwritable_cache_is_reported_once(fake_engines, tmp_path, caplog):
    not_a_dir = tmp_path / "cache"
    not_a_dir.write_text("x")
    games = [make_game(moves_san=SICILIAN) for _ in range(4)]
    assert len(analyze_games(games, cfg_fake(), cache_dir=not_a_dir)) == 4  # the analysis itself is unaffected
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "could not cache" in r.getMessage()]
    assert len(warnings) == 1
