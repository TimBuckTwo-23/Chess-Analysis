"""coach/deep.py: whole lines for critical positions and errors, workers, the cache, and the C1 gate."""

from __future__ import annotations

import dataclasses
import json
import time

import chess
import chess.engine
import pytest

from chess_insights import engine as sf
from chess_insights.analysis.mistakes import move_label
from chess_insights.coach import concepts, deep
from chess_insights.coach.config import CoachConfig
from chess_insights.context import AnalysisContext
from chess_insights.models import CriticalPosition

from factories import make_game, make_game_eval, make_ply_eval

HABITS = {  # the coaching plan's three habit positions: moves before, your move, your colour, the refutation's reply
    "sicilian_5e5": ("e4 c5 Nf3 Nc6 Nc3 e6 d4 cxd4 Nxd4", "e5", "black", "Ndb5"),
    "qga_4e4": ("d4 d5 c4 dxc4 Nc3 Nc6", "e4", "white", "Qxd4"),
    "sicilian_2nc6": ("e4 c5 d4", "Nc6", "black", "d5"),
}


def critical(before: str, san: str, color: str, **kw) -> CriticalPosition:
    board = chess.Board()
    for m in before.split():
        board.push_san(m)
    move = board.parse_san(san)
    fields = dict(
        game_id="g1", url="https://www.chess.com/game/live/1", ply=len(before.split()), fen=board.fen(),
        played_uci=move.uci(), played_san=san, move_label=move_label(board.fen(), san), best_san=None, drop=20.0,
        time_class="blitz", color=color, moves_before=before.split(),
    )
    fields.update(kw)
    return CriticalPosition(**fields)


# --------------------------------------------------------------------------- fake engines
class FakeEngine:
    """MultiPV = the legal moves in UCI order, each followed by the first legal reply; +50 cp for the side to move
    on the first line, 10 less on each next one. Records open / close and the positions searched."""

    def __init__(self, log, name="Fake 1"):
        self.log, self.id, self.options = log, {"name": name}, {}
        log.append("opened")

    def analyse(self, board, limit, multipv=None, game=None, info=None):
        self.log.append(("search", board.epd(), multipv))
        infos = []
        for i, move in enumerate(sorted(board.legal_moves, key=lambda m: m.uci())[: multipv or 1]):
            after = board.copy()
            after.push(move)
            reply = min(after.legal_moves, key=lambda m: m.uci()) if not after.is_game_over() else None
            infos.append({
                "score": chess.engine.PovScore(chess.engine.Cp(50 - 10 * i), board.turn),
                "pv": [move] + ([reply] if reply else []),
                "depth": limit.depth,
            })
        return infos if multipv is not None else infos[0]

    def configure(self, options):
        pass

    def close(self):
        self.log.append("closed")


@pytest.fixture
def fake_engines(monkeypatch):
    log: list = []
    monkeypatch.setattr(sf, "open_engine", lambda cfg: FakeEngine(log))
    monkeypatch.setattr(sf, "find_stockfish", lambda explicit=None: "/fake/stockfish")
    monkeypatch.setattr(sf, "engine_name", lambda path: "Fake 1")
    deep._PROFILE_MEMO.clear()
    return log


def searches(log) -> list:
    return [x for x in log if isinstance(x, tuple) and x[0] == "search"]


def test_both_lines_start_before_your_move_and_are_scored_from_your_side(fake_engines, tmp_path):
    pos = critical("e4", "e5", "black")
    notes: list[str] = []
    res = deep.analyse_positions([pos], CoachConfig(workers=1, cache_dir=tmp_path), notes)
    r = res[(pos.epd, pos.played_uci)]
    assert notes == [] and r.engine == "Fake 1" and r.depth == 20
    assert r.best_line.fen == pos.fen and r.refutation.fen == pos.fen
    assert r.best_line.cp_end == 50  # +50 for Black, the side to move: from your side
    assert r.refutation.moves_uci[0] == "e7e5" and r.refutation.moves_san[0] == "e5"
    assert r.refutation.cp_end == -50  # +50 for White after your move: -0.50 for you
    assert len(r.alternatives) == 2 and [a.cp_end for a in r.alternatives] == [40, 30]
    end = chess.Board(pos.fen)
    for u in r.refutation.moves_uci:
        end.push_uci(u)
    assert r.refutation.fen_end == end.fen()
    # MultiPV before the move, a single line after it
    assert [s[2] for s in searches(fake_engines)] == [3, None]


def test_results_are_cached_per_engine_and_depth_and_reused(fake_engines, tmp_path, monkeypatch):
    positions = [critical("e4", "e5", "black"), critical("d4", "d5", "black")]
    cfg = CoachConfig(workers=2, cache_dir=tmp_path)
    first = deep.analyse_positions(positions, cfg, [])
    files = sorted((tmp_path / "fake-1" / "d20").glob("*.json"))
    assert len(files) == 2
    data = json.loads(files[0].read_text())
    assert data["format"] == deep.CACHE_FORMAT and data["multipv"] == 3 and data["seconds"] == deep.SEARCH_SECONDS

    attempts: list = []

    def no_engine(cfg):
        attempts.append(cfg)
        raise RuntimeError("no engine here")

    monkeypatch.setattr(sf, "open_engine", no_engine)
    again = deep.analyse_positions(positions, cfg, [])
    assert {k: dataclasses.asdict(v) for k, v in again.items()} == {k: dataclasses.asdict(v) for k, v in first.items()}
    assert deep.analyse_positions(positions, dataclasses.replace(cfg, multipv=1), []).keys() == first.keys()
    assert attempts == []  # MultiPV 3 results serve a MultiPV 1 request too
    # a single-PV file (a profile run at the same depth) is not enough for MultiPV 3, a corrupt one is ignored
    for broken in (json.dumps({**data, "multipv": 1}), "{not json"):
        files[0].write_text(broken)
        notes: list[str] = []
        res = deep.analyse_positions(positions, cfg, notes)
        assert len(res) == 1 and attempts and notes == ["Deep analysis failed for 1 of 2 positions."]
        attempts.clear()
    assert not any(p.name.endswith(".part") for p in tmp_path.rglob("*"))


def test_one_search_per_position_and_move(fake_engines):
    a = critical("e4 e5 Nf3 Nc6", "Bb5", "white")
    b = dataclasses.replace(a, fen=" ".join(a.fen.split()[:4] + ["4", "9"]), game_id="g2")  # other move counters
    res = deep.analyse_positions([a, b], CoachConfig(workers=2), [])
    assert list(res) == [(a.epd, a.played_uci)] and len(searches(fake_engines)) == 2


def test_no_stockfish_gives_a_note_and_nothing(monkeypatch):
    monkeypatch.setattr(sf, "find_stockfish", lambda explicit=None: None)
    notes: list[str] = []
    assert deep.analyse_positions([critical("e4", "e5", "black")], CoachConfig(), notes) == {}
    assert notes and "Stockfish not found" in notes[0]
    assert deep.analyse_positions([], CoachConfig(), notes) == {} and len(notes) == 1


def test_failures_restarts_and_every_engine_closed(fake_engines, monkeypatch):
    positions = [critical("e4", "e5", "black"), critical("d4", "d5", "black"), critical("c4", "e5", "black")]
    real = deep.search
    died: set = set()

    def flaky(eng, task, label=""):
        if task.played_uci == "d7d5":
            raise RuntimeError("bad position")
        if task.fen == positions[0].fen and task.fen not in died:
            died.add(task.fen)  # the engine dies once: restarted, the position retried
            raise chess.engine.EngineTerminatedError("engine died")
        return real(eng, task, label)

    monkeypatch.setattr(deep, "search", flaky)
    for workers in (1, 3):
        died.clear()
        fake_engines.clear()
        notes: list[str] = []
        res = deep.analyse_positions(positions, CoachConfig(workers=workers), notes)
        assert set(res) == {(p.epd, p.played_uci) for p in positions if p.played_uci != "d7d5"}
        assert notes == ["Deep analysis failed for 1 of 3 positions."]
        assert fake_engines.count("opened") == fake_engines.count("closed") >= 2


def test_ctrl_c_stops_the_run_and_closes_the_engines(fake_engines, monkeypatch):
    positions = [critical("e4", "e5", "black"), critical("d4", "d5", "black"), critical("c4", "e5", "black")]

    def interrupted(eng, task, label=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(deep, "search", interrupted)  # one engine: the search runs in the main thread
    with pytest.raises(KeyboardInterrupt):
        deep.analyse_positions(positions, CoachConfig(workers=1), [])
    assert fake_engines.count("opened") == fake_engines.count("closed") == 1

    monkeypatch.undo()
    monkeypatch.setattr(sf, "open_engine", lambda cfg: FakeEngine(fake_engines))
    monkeypatch.setattr(sf, "find_stockfish", lambda explicit=None: "/fake/stockfish")
    monkeypatch.setattr(sf, "engine_name", lambda path: "Fake 1")

    def ctrl_c(what, total):
        def report(done):
            raise KeyboardInterrupt  # Ctrl+C reaches the main thread while the workers search

        return report

    monkeypatch.setattr(deep, "_progress", ctrl_c)
    fake_engines.clear()
    with pytest.raises(KeyboardInterrupt):
        deep.analyse_positions(positions, CoachConfig(workers=3), [])
    assert fake_engines.count("opened") == fake_engines.count("closed") >= 1


def test_time_caps_come_from_the_config_when_it_has_them():
    assert deep._config_seconds(CoachConfig()) == (deep.SEARCH_SECONDS, deep.REFUTATION_SECONDS)

    @dataclasses.dataclass
    class Capped(CoachConfig):
        search_seconds: float | None = 6.0

    assert deep._config_seconds(Capped()) == (6.0, 3.0)
    assert deep._config_seconds(Capped(search_seconds=0)) == (None, None)  # no cap
    assert deep._config_seconds(Capped(search_seconds=None)) == (deep.SEARCH_SECONDS, deep.REFUTATION_SECONDS)
    assert deep._limit(20, None) == chess.engine.Limit(depth=20)
    assert deep._limit(20, 8.0) == chess.engine.Limit(depth=20, time=8.0)


def test_short_searches_are_counted_in_a_note(fake_engines, monkeypatch):
    real = deep.search

    def shallow(eng, task, label=""):
        r = real(eng, task, label)
        r.best_line.depth = 17
        return r

    monkeypatch.setattr(deep, "search", shallow)
    notes: list[str] = []
    deep.analyse_positions([critical("e4", "e5", "black")], CoachConfig(workers=1), notes)
    assert notes == ["1 of 1 positions were searched to less than depth 20 (the time cap: 8 s before your move, "
                     "4 s after it)."]


def test_rebase_moves_a_line_to_other_move_counters_and_cuts_it():
    pos = critical("e4 e5 Nf3 Nc6", "Bb5", "white")
    line = deep.make_line(pos.fen, [chess.Move.from_uci(u) for u in ("f1b5", "a7a6", "b5a4", "g8f6")], 30, None, 20)
    moved = deep.rebase(line, " ".join(pos.fen.split()[:4] + ["0", "12"]), plies=3)
    assert moved.moves_uci == ["f1b5", "a7a6", "b5a4"] and moved.cp_end == 30 and moved.fen.endswith(" 12")
    assert moved.fen_end.endswith(" 13")
    bad = deep.make_line(pos.fen, [chess.Move.from_uci("e1e8")], 0, None, 1)
    assert bad.moves_uci == [] and bad.fen_end == pos.fen


# --------------------------------------------------------------------------- the motif-profile pass
def _game_with_errors():
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4", "Nxe5", "Qg5", "Nxf7", "Qxg2"]
    game = make_game(moves_san=moves, color="white")
    plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=i % 2 == 0, san=m, best_san=m)
             for i, m in enumerate(moves)]
    plies[5] = make_ply_eval(ply=5, mover="black", is_user=False, san="Nd4", best_san="Nf6", win_before=48.0,
                             win_after=38.0)  # the opponent's error
    plies[6] = make_ply_eval(ply=6, mover="white", is_user=True, san="Nxe5", best_san="Nxd4", win_before=62.0,
                             win_after=35.0)  # yours
    plies[8] = make_ply_eval(ply=8, mover="white", is_user=True, san="Nxf7", best_san="O-O", win_before=40.0,
                             win_after=36.0)  # 4 points: not an error
    return game, make_game_eval(game.game_id, plies)


def test_profile_lines_cover_both_sides_and_are_kept_for_the_run(fake_engines, tmp_path, monkeypatch):
    game, ev = _game_with_errors()
    chess960 = make_game(moves_san=list(game.moves_san), rules="chess960")
    ctx = AnalysisContext("tester", [game, chess960], evals={game.game_id: ev, chess960.game_id: dataclasses.replace(
        ev, game_id=chess960.game_id)})
    cfg = CoachConfig(workers=2, cache_dir=tmp_path)
    notes: list[str] = []
    errors = deep.profile_lines(ctx, cfg, notes)
    assert [(e.ply, e.side, round(e.drop)) for e in errors] == [(5, "opponent", 10), (6, "you", 27)]
    assert all(e.game_id == game.game_id and e.time_class == "blitz" for e in errors)
    for e in errors:
        assert e.best_line.fen == e.fen and e.refutation.moves_uci[0] == e.played_uci
        assert e.best_line.depth == cfg.profile_depth
    assert (tmp_path / "fake-1" / f"d{cfg.profile_depth}").is_dir()
    assert {s[2] for s in searches(fake_engines)} == {1, None}  # a single line before and after the move
    n = len(searches(fake_engines))
    assert deep.profile_lines(ctx, cfg, notes) == errors and len(searches(fake_engines)) == n  # kept for the run
    assert notes == []
    assert deep.profile_lines(AnalysisContext("tester", []), cfg, notes) == []


# --------------------------------------------------------------------------- the C1 gate (Stockfish 16)
@pytest.fixture(scope="module")
def habit_lines(stockfish_path):
    """The three habit positions at depth 20, MultiPV 3, without a time cap (about 20 s on two engines)."""
    positions = {name: critical(before, san, color) for name, (before, san, color, _) in HABITS.items()}
    saved = deep.SEARCH_SECONDS, deep.REFUTATION_SECONDS
    deep.SEARCH_SECONDS = deep.REFUTATION_SECONDS = None
    try:
        started = time.monotonic()
        res = deep.analyse_positions(list(positions.values()), CoachConfig(stockfish=stockfish_path, workers=2), [])
        print(f"\ndeep analysis of the 3 habit positions: {time.monotonic() - started:.1f} s on 2 engines")
    finally:
        deep.SEARCH_SECONDS, deep.REFUTATION_SECONDS = saved
    return {name: (positions[name], res.get((p.epd, p.played_uci))) for name, p in positions.items()}


@pytest.mark.engine
@pytest.mark.parametrize("name", list(HABITS))
def test_gate_refutations_of_the_three_habit_moves_at_depth_20(habit_lines, name):
    pos, result = habit_lines[name]
    assert result is not None and result.refutation is not None
    assert result.refutation.moves_san[0] == HABITS[name][1]
    assert result.refutation.moves_san[1] == HABITS[name][3], (
        f"depth 20 refutes {pos.move_label} with {result.refutation.moves_san[1]} (plan, depth 22: {HABITS[name][3]})"
    )
    assert result.refutation.depth == 20 and result.best_line.depth == 20
    assert result.refutation.cp_end < result.best_line.cp_end  # worse for you than the engine's move


@pytest.mark.engine
def test_gate_concepts_for_5e5_compare_line_ends_not_the_move(habit_lines, stockfish_path):
    """Right after 5...e5 Stockfish's terms favour Black (the pawn hits d4); where the lines have played out, the
    refutation leaves Black worse on piece activity and king safety than the engine's move and than 5...a6."""
    pos, result = habit_lines["sicilian_5e5"]
    with concepts.ClassicalEval(stockfish_path) as ce:
        board = chess.Board(pos.fen)
        board.push_uci(pos.played_uci)
        right_after = ce.table(board.fen())
        if ce.supported is False:
            pytest.skip("concepts need Stockfish 16")
        assert right_after["Threats"][0] < -0.5  # White's point of view: a plus for Black right after the move
        ref_end = concepts.comparison_point(pos.fen, result.refutation.moves_uci)
        a6 = next(x for x in [result.best_line, *result.alternatives] if x.moves_san[0] == "a6")
        for other in (result.best_line, a6):
            end = concepts.comparison_point(pos.fen, other.moves_uci)
            deltas = {c.term: c.value for c in concepts.concept_deltas(
                ce.table(end.fen()), ce.table(ref_end.fen()), "black",
                concepts.material_phase(end), concepts.material_phase(ref_end))}
            assert deltas.get("Mobility", 0) <= -concepts.MIN_DELTA, (other.moves_san[0], deltas)
            assert deltas.get("King safety", 0) <= -concepts.MIN_DELTA, (other.moves_san[0], deltas)
            assert deltas.get("Threats", 0) >= 0  # the threat that looked good right after the move is gone
