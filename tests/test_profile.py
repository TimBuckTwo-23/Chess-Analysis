"""The motif profile (coach/profile.py): counts of the patterns you and your opponents miss and allow, and the only
claims the coaching layer makes, which must be as honest as every other finding (few false claims on data without
a real difference, and real differences found)."""

import dataclasses
import math
import random

import pytest

from chess_insights.coach import CoachConfig, deep, motifs, profile
from chess_insights.coach.deep import ProfileError
from chess_insights.coach.profile import GameCounts, motif_claims
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, Line, ModuleResult, Motif
from factories import make_game, make_game_eval, make_ply_eval

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
GATED = frozenset({"fork", "pin", "skewer", "hangingPiece", "discoveredAttack", "backRankMate"})

# Per mistake or blunder: how often each pattern is in the best line (missed) or the refutation (allowed).
MISSED = {"fork": 0.12, "pin": 0.06, "skewer": 0.03, "hangingPiece": 0.30, "discoveredAttack": 0.05,
          "backRankMate": 0.02}
ALLOWED = {"fork": 0.10, "pin": 0.05, "skewer": 0.03, "hangingPiece": 0.35, "discoveredAttack": 0.04,
           "backRankMate": 0.03}


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def simulate(n_games: int, seed: int, planted=None, error_boost: float = 1.0) -> list[GameCounts]:
    """Engine-analysed games in which both sides share the same pattern rates, unless ``planted`` says otherwise.

    Realistic clustering: a game-level error propensity shared by both sides (wild games have errors on both
    sides), a per-side factor, and a per-game proneness to each pattern (positions rich in forks). ``planted``:
    {(side, kind, theme): rate}, side 0 = you. ``error_boost`` multiplies your error rate only.
    """
    rng = random.Random(seed)
    planted = planted or {}
    games = []
    for i in range(n_games):
        moves = rng.randint(12, 60)
        wild = rng.gammavariate(2.0, 0.5)
        gc = GameCounts(f"g{i}", rng.choice(["bullet", "blitz", "rapid"]), (moves, moves - rng.randint(0, 1)))
        for side in (0, 1):
            lam = moves * 0.10 * wild * rng.gammavariate(4.0, 0.25) * (error_boost if side == 0 else 1.0)
            errors = _poisson(rng, lam)
            gc.errors[side] = errors
            for kind, base in (("missed", MISSED), ("allowed", ALLOWED)):
                for theme, rate in base.items():
                    prone = min(1.0, planted.get((side, kind, theme), rate) * rng.gammavariate(2.0, 0.5))
                    hits = sum(1 for _ in range(errors) if rng.random() < prone)
                    if hits:
                        gc.counts.setdefault((kind, theme), [0, 0])[side] += hits
        games.append(gc)
    return games


# --------------------------------------------------------------------------- honesty of the claims
def test_no_false_motif_claims_when_both_sides_share_the_same_rates():
    runs, n = 200, 300
    claims = [motif_claims(simulate(n, seed), GATED)[0] for seed in range(runs)]
    mean = sum(len(c) for c in claims) / runs
    assert mean <= 0.05, [[i.id for i in c] for c in claims if c]


def test_more_errors_of_every_kind_is_not_a_motif_finding():
    # You make 60% more mistakes and blunders than your opponents, with the same mix of patterns: that is the
    # blunder-rate finding, not "you miss forks more often" (the pattern's share of your errors must stand out).
    claims = [motif_claims(simulate(300, seed, error_boost=1.6), GATED)[0] for seed in range(30)]
    assert sum(len(c) for c in claims) / len(claims) <= 0.1, [[i.id for i in c] for c in claims if c]


def test_a_real_difference_is_found():
    # you miss forks twice as often as your opponents, per error: realistic at 300 analysed games
    planted = {(0, "missed", "fork"): 0.24}
    found, opposite = 0, 0
    for seed in range(20):
        claims, stats = motif_claims(simulate(300, 100 + seed, planted), GATED,
                                     formats={"bullet": 100, "blitz": 100, "rapid": 100})
        ids = [i.id for i in claims]
        found += "tactics.weakness.motif-missed.fork" in ids
        opposite += "tactics.strength.motif-missed.fork" in ids
    assert found >= 16, found
    assert opposite == 0


def test_a_claim_carries_formats_a_chart_split_by_format_and_a_drill():
    games = simulate(300, 101, {(0, "missed", "fork"): 0.30})
    formats = {"bullet": 0, "blitz": 0, "rapid": 0}
    for g in games:
        formats[g.time_class] += 1
    claims, stats = motif_claims(games, GATED, formats=formats)
    [fork] = [i for i in claims if i.id == "tactics.weakness.motif-missed.fork"]
    assert fork.kind == "weakness" and fork.category == "tactics"
    assert fork.title == "You miss forks more often than your opponents"
    assert fork.confidence >= 0.5 and 0 < fork.severity <= 1
    assert fork.evidence["p_adjusted"] <= 0.01 and fork.evidence["share_p_adjusted"] <= 0.01
    assert fork.formats == formats
    assert "300 engine-analysed games (bullet, blitz and rapid)" in fork.detail
    chart = fork.chart
    assert chart.labels == ["All games", "Bullet", "Blitz", "Rapid"]
    assert [s.name for s in chart.series] == ["You", "Opponents"]
    you, them = chart.series[0].values, chart.series[1].values
    assert you[0] > them[0] > 0
    assert chart.table is not None and len(chart.table.rows) == 4
    assert any("lichess.org/training/fork" in a for a in fork.study)
    assert stats["fork:missed"]["claim"] == "weakness"


def test_minimum_samples_before_any_test():
    games = simulate(40, 7, {(0, "missed", "backRankMate"): 0.5})
    tests = {(t.theme, t.kind) for t in profile.motif_tests(games, GATED)}
    for t in profile.motif_tests(games, GATED):
        assert t.games_with >= profile.MIN_MOTIF_GAMES and t.events >= profile.MIN_MOTIF_EVENTS
    assert ("backRankMate", "allowed") not in tests  # rare: never enough games
    assert profile.motif_tests(games, frozenset()) == []  # nothing gated, nothing tested


def test_only_gated_patterns_are_named():
    games = simulate(300, 3, {(0, "missed", "fork"): 0.40})
    assert motif_claims(games, frozenset({"pin"}))[0] == []  # the fork detector did not pass the gate
    table = profile.profile_table(games, {"pin"}, {"blitz": 300})
    assert [r[0] for r in table.rows] == ["Pin"]


# --------------------------------------------------------------------------- counting
def _records_ctx(n=3):
    games, evals = [], {}
    for k in range(n):
        g = make_game(color="white" if k % 2 == 0 else "black", time_class=["blitz", "rapid", "bullet"][k % 3])
        plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i),
                               san=g.moves_san[i]) for i in range(g.plies)]
        games.append(g)
        evals[g.game_id] = make_game_eval(g.game_id, plies)
    return AnalysisContext(username="tester", games=games, evals=evals)


def _line(fen=START, moves=("e2e4", "e7e5", "g1f3")):
    return Line(fen=fen, moves_uci=list(moves))


def test_collect_counts_each_error_once_per_pattern_and_side():
    ctx = _records_ctx(2)
    from chess_insights.analysis.engine_stats import join

    records = join(ctx)
    g0, g1 = ctx.games
    best, refute = _line(), _line(moves=("d2d4", "d7d5"))
    found = {
        id(best): [Motif("fork", "best", 0, ["e4"], "you"), Motif("fork", "best", 2, ["f3"], "you"),
                   Motif("pin", "best", 1, ["e5"], "opponent")],  # the opponent's pin is not what you missed
        id(refute): [Motif("skewer", "refutation", 1, [], "opponent"), Motif("fork", "refutation", 0, [], "you")],
    }
    errors = [
        ProfileError(g0.game_id, 4, "you", "blitz", START, "e2e4", 30.0, best, refute),
        ProfileError(g0.game_id, 5, "opponent", "blitz", START, "e2e4", 20.0, best, None),
        ProfileError(g0.game_id, 5, "opponent", "blitz", START, "e2e4", 20.0, best, None),  # duplicate
        ProfileError("not-analysed", 3, "you", "blitz", START, "e2e4", 20.0, best, refute),
        ProfileError(g1.game_id, 6, "you", "rapid", START, "e2e4", 20.0, None, refute),
    ]
    games, events = profile.collect(records, errors, lambda line, role: found.get(id(line), []))
    c0, c1 = games
    assert c0.errors == [1, 1] and c1.errors == [1, 0]
    assert c0.count("missed", "fork") == (1, 1) and c0.count("missed", "pin") == (0, 0)
    assert c0.count("allowed", "skewer") == (1, 0) and c0.count("allowed", "fork") == (0, 0)
    assert c1.count("allowed", "skewer") == (1, 0) and c1.count("missed", "fork") == (0, 0)
    assert c0.moves == (10, 10)
    assert len(events) == 4


def test_a_line_the_detectors_fail_on_counts_without_a_pattern():
    ctx = _records_ctx(1)
    from chess_insights.analysis.engine_stats import join

    [g] = ctx.games
    good, broken = _line(), _line(moves=("d2d4",))

    def detect(line, role):
        if line is broken:
            raise ValueError("odd line")
        return [Motif("fork", role, 0, ["e4"], "you" if role == "best" else "opponent")]

    errors = [ProfileError(g.game_id, 4, "you", "blitz", START, "e2e4", 30.0, good, broken)]
    notes = []
    [c], events = profile.collect(join(ctx), errors, detect, notes)
    assert c.count("missed", "fork") == (1, 0) and c.count("allowed", "fork") == (0, 0) and len(events) == 1
    assert notes == ["Motif profile: the pattern detectors failed on 1 engine line (counted without a pattern)."]


def test_table_and_chart_are_per_100_moves_with_practice_links():
    g = GameCounts("a", "blitz", (50, 50), [4, 2], {("missed", "fork"): [2, 1], ("allowed", "pin"): [1, 0],
                                                     ("missed", "mateIn2"): [3, 3]})
    h = GameCounts("b", "blitz", (50, 50), [1, 1], {("missed", "fork"): [1, 0]})
    table = profile.profile_table([g, h], GATED, {"blitz": 2})
    assert table.columns[:5] == ["Pattern", "You missed", "Opponents missed", "You allowed", "Opponents allowed"]
    assert table.rows[0] == ["Fork", 3.0, 1.0, 0.0, 0.0, "https://lichess.org/training/fork"]
    assert [r[0] for r in table.rows] == ["Fork", "Pin"]  # mateIn2 is not gated here
    assert "2 engine-analysed games (2 blitz)" in table.note
    chart = profile.profile_chart([g, h], GATED, {"blitz": 2})
    assert chart.labels == ["Fork · missed", "Fork · allowed", "Pin · missed", "Pin · allowed"]
    assert chart.series[0].values == [3.0, 0.0, 0.0, 1.0] and chart.series[1].values == [1.0, 0.0, 0.0, 0.0]
    assert chart.table.rows[0][:3] == ["Fork", "missed", 3]
    assert profile.profile_table([g, h], frozenset(), {}) is None


def test_names_of_patterns():
    assert profile.motif_name("fork", plural=True) == "forks"
    assert profile.motif_name("hangingPiece") == "hanging piece"
    assert profile.motif_name("someNewTheme", plural=True) == "some new themes"


# --------------------------------------------------------------------------- the coaching step
def _planted_world(n_games=120, seed=5, p_you=0.8, p_opp=0.15):
    """Games, evals and profiled errors where you miss forks far more often than your opponents (by default)."""
    rng = random.Random(seed)
    games, evals, errors, registry = [], {}, [], {}
    for k in range(n_games):
        g = make_game(color="white" if k % 2 else "black", time_class=["blitz", "rapid", "bullet"][k % 3],
                      game_id=f"pg{k}", url=f"https://www.chess.com/game/live/{9000 + k}")
        plies = [make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i),
                               san=g.moves_san[i]) for i in range(g.plies)]
        for i in (4, 5, 6, 7):  # your errors are errors in the game analysis too (the puzzle export's selection)
            if g.is_my_ply(i):
                plies[i] = dataclasses.replace(plies[i], win_before=60.0, win_after=40.0,
                                               best_san="h4" if g.moves_san[i] != "h4" else "a4")
        games.append(g)
        evals[g.game_id] = make_game_eval(g.game_id, plies)
        for side, p_fork in (("you", p_you), ("opponent", p_opp)):
            for e in range(2):
                ply = 4 + 2 * e + (0 if (side == "you") == (g.color == "white") else 1)
                fen = START if ply % 2 == 0 else AFTER_E4
                best = Line(fen=fen, moves_uci=["g1f3", "b8c6", "f1c4"] if fen == START else ["e7e5", "g1f3"])
                refutation = Line(fen=fen, moves_uci=["d2d4", "d7d5"] if fen == START else ["d7d5", "e4d5"])
                fork = Motif("fork", "best", 0, ["f3", "e5", "d4"], "you")
                registry[id(best)] = [fork] if rng.random() < p_fork else []
                registry[id(refutation)] = [Motif("pin", "refutation", 1, [], "opponent")] if rng.random() < 0.2 else []
                played = "d2d4" if fen == START else "d7d5"
                errors.append(ProfileError(g.game_id, ply, side, g.time_class, fen, played, 20.0 + k % 7, best,
                                           refutation))
    ctx = AnalysisContext(username="tester", games=games, evals=evals)
    return ctx, errors, registry


def test_annotate_adds_the_profile_and_claims_to_the_engine_review(monkeypatch):
    ctx, errors, registry = _planted_world()
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork", "pin"}))
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())

    assert coaching.motif_profile is not None and coaching.motif_profile in engine.tables
    assert coaching.motif_chart is not None and coaching.motif_chart in engine.charts
    assert "(40 bullet, 40 blitz, 40 rapid)" in coaching.motif_profile.note  # the formats analysed, in order
    [claim] = [i for i in engine.insights if i.id == "tactics.weakness.motif-missed.fork"]
    assert claim.formats == {"bullet": 40, "blitz": 40, "rapid": 40}
    assert claim.example_games and all(u.startswith("https://www.chess.com/game/live/") for u in claim.example_games)
    diagram = claim.diagram
    assert diagram is not None and diagram.fen in (START, AFTER_E4)
    assert [a.kind for a in diagram.arrows] == ["played", "best"]
    assert diagram.strips and diagram.strips[0].frames[0].marks[0].kind == "attacker"
    assert diagram.orientation in ("white", "black") and diagram.link == claim.example_games[0]
    counts = coaching.settings["motif_profile"]["counts"]
    assert counts["fork"]["you_missed"] > 3 * counts["fork"]["opp_missed"] > 0
    assert engine.stats["motif_profile"]["tests"]["fork:missed"]["claim"] == "weakness"
    # the puzzle export gets your errors' best lines and their named patterns
    assert set(coaching.puzzle_lines) == {f"{e.game_id}:{e.ply}" for e in errors if e.side == "you"}
    assert any(v == ["fork"] for v in coaching.puzzle_themes.values())


def test_annotate_names_nothing_before_the_detectors_pass_the_gate(monkeypatch):
    ctx, errors, registry = _planted_world(40)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset())
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())
    assert coaching.motif_profile is None and not engine.tables and not engine.insights
    assert any("precision check" in n for n in coaching.notes)
    assert coaching.settings["motif_profile"]["unnamed_themes"] == ["fork", "pin"]


def test_annotate_without_lines_or_engine_games_leaves_a_note(monkeypatch):
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    coaching = Coaching()
    profile.annotate(AnalysisContext(username="x", games=[]), coaching, [engine], CoachConfig())
    assert coaching.notes and not engine.tables
    ctx, _, _ = _planted_world(5)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: [])
    coaching = Coaching()
    profile.annotate(ctx, coaching, [engine], CoachConfig())  # the stub on this branch returns []
    assert any("no engine lines" in n for n in coaching.notes) and coaching.motif_profile is None


@pytest.mark.parametrize("kind", ["missed", "allowed"])
def test_event_diagram_shows_the_right_line(kind):
    g = make_game(color="white")
    best, refutation = Line(START, ["g1f3", "b8c6"]), Line(START, ["d2d4", "d7d5", "c2c4"])
    e = ProfileError(g.game_id, 0, "you", "blitz", START, "d2d4", 25.0, best, refutation)
    ev = profile.MotifEvent(g, e, kind, Motif("fork", "best" if kind == "missed" else "refutation", 1, ["c6"], ""))
    d = profile.event_diagram(ev)
    assert [(a.start, a.end, a.kind) for a in d.arrows] == [("d2", "d4", "played"), ("g1", "f3", "best")]
    frames = d.strips[0].frames
    assert [f.move for f in frames] == (["1.Nf3", "1...Nc6"] if kind == "missed" else ["1.d4", "1...d5", "2.c4"])
    assert frames[1].marks and frames[1].marks[0].square == "c6"
    assert "It cost you 25 percentage points" in d.caption and d.time_class == "blitz"
    assert d.caption.startswith("You played 1.d4 (red); 1.Nf3 (green) was better.")


def test_event_diagram_of_an_opponents_miss_keeps_your_side_at_the_bottom():
    g = make_game(color="black")
    best, refutation = Line(START, ["g1f3", "b8c6"]), Line(START, ["d2d4", "d7d5"])
    e = ProfileError(g.game_id, 0, "opponent", "rapid", START, "d2d4", 18.0, best, refutation)
    d = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 0, ["f3"], "you")))
    assert d.title == "A fork your opponent missed: 1.Nf3 instead of 1.d4"
    assert d.caption.startswith("Your opponent played 1.d4 (red); 1.Nf3 (green) was better. It cost your opponent 18")
    assert d.orientation == "black"
    d = profile.event_diagram(profile.MotifEvent(g, e, "allowed", Motif("fork", "refutation", 1, [], "opponent")))
    assert d.title == "The fork your opponent allowed after 1.d4"


def test_the_strip_reaches_the_pattern():
    g = make_game(color="white")
    moves = ["g1f3", "b8c6", "b1c3", "g8f6", "e2e4", "e7e5", "f3e5", "c6e5"]
    e = ProfileError(g.game_id, 0, "you", "blitz", START, "a2a3", 20.0, Line(START, moves), None)
    d = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 6, ["e5", "f7", "c6"], "you")))
    frames = d.strips[0].frames
    assert len(frames) == 7 and frames[6].move == "4.Nxe5" and [m.square for m in frames[6].marks] == ["e5", "f7", "c6"]
    short = profile.event_diagram(profile.MotifEvent(g, e, "missed", Motif("fork", "best", 0, ["f3"], "you")))
    assert len(short.strips[0].frames) == 4  # a few moves of context after an early pattern


def test_a_strength_shows_one_of_your_opponents_misses(monkeypatch):
    ctx, errors, registry = _planted_world(p_you=0.15, p_opp=0.8)
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: registry.get(id(line), []))
    monkeypatch.setattr(motifs, "GATED_THEMES", frozenset({"fork", "pin"}))
    engine = ModuleResult(key="engine", title="Engine review", summary="")
    profile.annotate(ctx, Coaching(), [engine], CoachConfig())
    [claim] = [i for i in engine.insights if i.id == "tactics.strength.motif-missed.fork"]
    assert claim.kind == "strength" and claim.title == "You miss forks less often than your opponents"
    assert claim.chart is not None and claim.diagram is not None
    assert claim.diagram.title.startswith("A fork your opponent missed")
    assert claim.diagram.link in claim.example_games


# --------------------------------------------------------------------------- the previous position and the puzzle lines
def _one_game_with_errors(errors_at=(4, 6, 8)):
    """One game of yours (White, the default moves) whose moves at ``errors_at`` lost 20 win-% points."""
    import chess

    g = make_game(color="white", game_id="prev1", url="https://www.chess.com/game/live/77")
    plies = []
    for i, san in enumerate(g.moves_san):
        worse = i in errors_at
        plies.append(make_ply_eval(ply=i, mover="white" if i % 2 == 0 else "black", is_user=g.is_my_ply(i), san=san,
                                   best_san="h4" if worse else san, win_before=60.0 if worse else 51.8,
                                   win_after=40.0 if worse else 51.8))
    board, fens = chess.Board(), []
    for san in g.moves_san:
        fens.append(board.fen())
        board.push_san(san)
    ctx = AnalysisContext(username="tester", games=[g], evals={g.game_id: make_game_eval(g.game_id, plies)})
    return ctx, g, fens


def test_best_lines_are_read_with_the_position_before_the_last_move():
    ctx, g, fens = _one_game_with_errors()
    records = profile.join(ctx)
    best = Line(fen=fens[4], moves_uci=["f1c4"])
    refutation = Line(fen=fens[4], moves_uci=["b1c3", "d7d5"])
    calls = []

    def three(line, role, previous_fen=None):
        calls.append((role, previous_fen))
        return []

    # computed from the game (one replay) when the error does not carry it ...
    profile.collect(records, [ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation)],
                    three)
    assert calls == [("best", fens[3]), ("refutation", None)]  # the position before 2...Nc6, the opponent's move
    # ... or taken from it (deep.profile_errors sets it)
    calls.clear()
    carried = ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation, previous_fen="x")
    profile.collect(records, [carried], three)
    assert calls[0] == ("best", "x")
    # an opponent's error: the position before your last move
    calls.clear()
    profile.collect(records, [ProfileError(g.game_id, 5, "opponent", "blitz", fens[5], "f8c5", 20.0,
                                           Line(fen=fens[5], moves_uci=["g8f6"]), None)], three)
    assert calls == [("best", fens[4])]
    # a detector that takes only (line, role) still works
    two_calls = []
    profile.collect(records, [ProfileError(g.game_id, 4, "you", "blitz", fens[4], "b1c3", 20.0, best, refutation)],
                    lambda line, role: two_calls.append(role) or [])
    assert two_calls == ["best", "refutation"]


def test_deep_records_the_previous_position_of_every_error():
    ctx, g, fens = _one_game_with_errors()
    found = deep.profile_errors(ctx)
    assert [(e.ply, e.previous_fen) for e in found] == [(4, fens[3]), (6, fens[5]), (8, fens[7])]
    assert deep.previous_fens(g, [0, 1, 4, 999]) == {1: fens[0], 4: fens[3]}


def test_puzzle_lines_are_kept_only_for_the_puzzle_exports_errors(monkeypatch):
    """Not every profiled error of yours (about 1,500 in a real run): the ones mistakes.build_puzzles exports, plus
    the explained positions."""
    from chess_insights.coach import puzzles

    ctx, g, fens = _one_game_with_errors(errors_at=(4, 6, 8))
    errors = [ProfileError(g.game_id, ply, "you", "blitz", fens[ply], "a2a3", 20.0,
                           Line(fen=fens[ply], moves_uci=["h2h4"]), Line(fen=fens[ply], moves_uci=["a2a3"]))
              for ply in (2, 4, 6, 8)]  # ply 2 is no error in the game analysis: no puzzle
    monkeypatch.setattr(deep, "profile_lines", lambda ctx, cfg, notes: errors)
    monkeypatch.setattr(motifs, "detect_line", lambda line, role: [])
    coaching = Coaching()
    profile.annotate(ctx, coaching, [], CoachConfig())
    assert set(coaching.puzzle_lines) == {f"{g.game_id}:{p}" for p in (4, 6, 8)}
    # at most PUZZLE_LIMIT puzzles, costliest first: here one
    real = puzzles.build_puzzles
    monkeypatch.setattr(puzzles, "build_puzzles", lambda games, evals: real(games, evals, limit=1))
    coaching = Coaching()
    profile.annotate(ctx, coaching, [], CoachConfig())
    assert len(coaching.puzzle_lines) == 1
