"""analysis/engine_stats.py: aggregates over hand-built engine evaluations (no Stockfish needed)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from chess_insights.analysis import engine_stats
from chess_insights.context import AnalysisContext
from chess_insights.models import CATEGORIES, VALUE_FORMATS, ModuleResult
from factories import make_game, make_game_eval, make_ply_eval

SANS = ["e4", "Nf6", "Qd1", "O-O", "exd5", "Rb8", "Bc4", "Kh8", "Ne2", "a6"]


# --------------------------------------------------------------------------- helpers
def assert_consistent(mr: ModuleResult) -> None:
    for chart in mr.charts:
        assert chart.value_format in VALUE_FORMATS
        assert chart.series and all(len(s.values) == len(chart.labels) for s in chart.series), chart.title
    for table in mr.tables:
        assert table.formats is None or len(table.formats) == len(table.columns)
        assert table.formats is None or set(table.formats) <= set(VALUE_FORMATS)
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
        assert len(set(table.columns)) == len(table.columns), table.title
    assert all(k.format in VALUE_FORMATS for k in mr.kpis)
    for ins in mr.insights:
        assert ins.id.startswith(f"{mr.key}.{ins.kind}.") and ins.category in CATEGORIES
        assert 0.0 <= ins.severity <= 1.0 and 0.0 <= ins.confidence <= 1.0
        assert 2 <= len(ins.study) <= 4 and ins.title and ins.detail
        assert len(ins.example_games) <= 5 and all(u.startswith("https://") for u in ins.example_games)
        assert len(set(ins.example_games)) == len(ins.example_games)
    assert len({i.id for i in mr.insights}) == len(mr.insights)
    json.dumps(mr.stats, default=str, allow_nan=False)


def run(pairs, games=None, **options) -> ModuleResult:
    games = games if games is not None else [g for g, _ in pairs]
    ctx = AnalysisContext(
        "tester", sorted(games, key=lambda g: g.end_time), evals={ev.game_id: ev for _, ev in pairs}, options=options
    )
    mr = engine_stats.analyze(ctx)
    assert (mr.key, mr.title) == ("engine", "Engine review")
    assert_consistent(mr)
    return mr


def default_phase(i: int, n: int) -> str:
    return "opening" if i < n // 3 else "middlegame" if i < 2 * n // 3 else "endgame"


def analysed(spec=lambda i, user: {}, *, n=40, color="white", my_acc=80.0, opp_acc=80.0, **game_kw):
    """A game of ``n`` plies plus its GameEval; ``spec(ply, is_user)`` overrides PlyEval fields."""
    game = make_game(color=color, moves_san=[SANS[i % len(SANS)] for i in range(n)], **game_kw)
    plies = []
    for i in range(n):
        mover = "white" if i % 2 == 0 else "black"
        fields = dict(
            ply=i, mover=mover, is_user=mover == color, san=game.moves_san[i], best_san=None,
            cp_before=0, cp_after=0, win_before=50.0, win_after=50.0, accuracy=90.0, cp_loss=10,
            phase=default_phase(i, n),
        )
        fields.update(spec(i, mover == color))
        plies.append(make_ply_eval(**fields))
    return game, make_game_eval(game.game_id, plies, my_accuracy=my_acc, opp_accuracy=opp_acc)


def insight(mr: ModuleResult, id_: str):
    return next((i for i in mr.insights if i.id == id_), None)


def table(mr: ModuleResult, title: str):
    return next(t for t in mr.tables if t.title == title)


def claims(mr: ModuleResult) -> list[str]:
    return [i.id for i in mr.insights if i.kind != "observation"]


# --------------------------------------------------------------------------- empty and tiny input
def test_no_engine_analysis_explains_how_to_enable_it():
    mr = run([], games=[make_game()])
    assert "--engine" in mr.summary and not mr.insights and not mr.tables and not mr.kpis


def test_evals_for_other_games_or_too_short_games():
    g, ev = analysed()
    mr = run([(g, ev)], games=[make_game()])
    assert "Not enough data" in mr.summary and not mr.insights
    short, short_ev = analysed(n=3)
    assert "Not enough data" in run([(short, short_ev)]).summary


def test_single_game_is_a_snapshot_without_claims():
    mr = run([analysed(lambda i, u: {"judgement": "blunder"} if u and i == 10 else {})])
    assert mr.summary.startswith("Not enough data") and claims(mr) == []
    assert mr.kpis[0].value == 1


def test_missing_accuracies_out_of_range_plies_and_daily_games():
    g, ev = analysed(my_acc=None, opp_acc=None, time_class="daily", time_control="1/86400", base_seconds=86400)
    ev.plies.append(make_ply_eval(ply=999, judgement="blunder"))  # beyond the game: ignored
    mr = run([(g, ev)])
    assert mr.stats["player"]["accuracy"] is None and mr.stats["player"]["moves"] == 20
    assert mr.stats["player"]["blunders_per100"] == 0.0
    assert all(t.title != "Blunders when short of time" for t in mr.tables)


# --------------------------------------------------------------------------- per-100 rates and KPIs
def test_per_100_rates_acpl_and_time_class_table():
    def spec(i, user):
        if user and i in (0, 2):
            return {"judgement": "blunder", "cp_loss": 300}
        if user and i == 4:
            return {"judgement": "mistake", "cp_loss": 150}
        if not user and i == 1:
            return {"judgement": "blunder", "cp_loss": 300}
        if not user and i in (3, 5, 7):
            return {"judgement": "inaccuracy", "cp_loss": 60}
        return {}

    pairs = [analysed(spec, my_acc=70.0, opp_acc=76.0) for _ in range(8)]
    pairs += [analysed(spec, my_acc=74.0, opp_acc=80.0, time_class="rapid", time_control="600", base_seconds=600)
              for _ in range(2)]
    mr = run(pairs)
    kpi = {k.label: k.value for k in mr.kpis}
    assert kpi["Games analysed"] == 10
    assert kpi["Blunders /100 moves"] == pytest.approx(10.0)  # 2 of 20 moves per game
    assert kpi["Mistakes /100 moves"] == pytest.approx(5.0)
    assert kpi["Inaccuracies /100 moves"] == pytest.approx(0.0)
    assert mr.stats["opponents"]["blunders_per100"] == pytest.approx(5.0)
    assert mr.stats["opponents"]["inaccuracies_per100"] == pytest.approx(15.0)
    assert kpi["Average centipawn loss"] == round((2 * 300 + 150 + 17 * 10) / 20)
    assert kpi["Your accuracy"] == pytest.approx(70.8)
    rows = {r[0]: r for r in table(mr, "By time control").rows}
    assert list(rows) == ["Blitz", "Rapid"]
    assert rows["Blitz"][1:4] == [8, 70.0, 76.0] and rows["Rapid"][1:4] == [2, 74.0, 80.0]
    assert rows["Blitz"][6] == pytest.approx(10.0) and rows["Blitz"][7] == pytest.approx(5.0)


def test_blunder_rate_weakness_and_equal_rates_make_no_claims():
    def worse(i, user):
        return {"judgement": "blunder"} if (user and i in (6, 12, 20)) or (not user and i == 9) else {}

    mr = run([analysed(worse) for _ in range(20)])
    ins = insight(mr, "engine.weakness.blunder-rate")
    assert ins and ins.category == "blunders" and ins.evidence["per100"] == pytest.approx(15.0)
    assert ins.evidence["opp_per100"] == pytest.approx(5.0)

    def equal(i, user):
        return {"judgement": "blunder" if i % 10 in (4, 5) else "mistake" if i % 10 in (6, 7) else None}

    same = run([analysed(equal) for _ in range(30)])
    assert claims(same) == []


# --------------------------------------------------------------------------- phases
def phase_game(user_errors: dict, opp_errors: dict, **kw):
    """60-ply game: plies 0-19 opening, 20-39 middlegame, 40-59 endgame; N errors on the first N moves per phase."""
    def spec(i, user):
        ph = default_phase(i, 60)
        start = {"opening": 0, "middlegame": 20, "endgame": 40}[ph]
        k = (i - start) // 2  # this side's k-th move in the phase
        wanted = (user_errors if user else opp_errors).get(ph, 0)
        return {"judgement": "mistake", "cp_loss": 200} if k < wanted else {}

    return analysed(spec, n=60, **kw)


def test_phase_comparison_picks_the_endgame():
    pairs = [phase_game({"opening": 1, "middlegame": 1, "endgame": 3}, {"opening": 1, "middlegame": 1, "endgame": 1})
             for _ in range(20)]
    mr = run(pairs)
    weak = [i for i in mr.insights if i.category == "phases"]
    assert [i.id for i in weak] == ["engine.weakness.phase-endgame"]
    assert weak[0].evidence["per100"] == pytest.approx(30.0) and weak[0].evidence["opp_per100"] == pytest.approx(10.0)
    assert "endgame" in weak[0].title and 2 <= len(weak[0].study) <= 4
    rows = {r[0]: r for r in table(mr, "Game phases").rows}
    assert rows["Endgame"][1] == 200 and rows["Endgame"][3] == pytest.approx(30.0)
    chart = next(c for c in mr.charts if c.title.startswith("Mistakes and blunders"))
    assert chart.labels == ["Opening", "Middlegame", "Endgame"]
    assert chart.series[0].values == [rows[label][3] for label in chart.labels]
    assert chart.series[1].values == [rows[label][6] for label in chart.labels]


def test_phase_strength_and_minimum_moves():
    pairs = [phase_game({"middlegame": 1, "endgame": 1}, {"middlegame": 4, "endgame": 1}) for _ in range(20)]
    mr = run(pairs)
    assert [i.id for i in mr.insights if i.category == "phases"] == ["engine.strength.phase-middlegame"]
    few = run(pairs[:12])  # 120 of your moves per phase: below the 150 minimum
    assert [i.id for i in few.insights if i.category == "phases"] == []


# --------------------------------------------------------------------------- time pressure
def pressure_game(low_blunders: int, ok_blunder: bool, opp_low_blunders=None):
    """You spend 25 s a move from a 300 s clock: short of time (< 45 s) from ply 22 on.

    With ``opp_low_blunders`` set, your opponent burns the clock the same way (short of time from ply 23)
    and blunders on their first ``opp_low_blunders`` moves when short of time; otherwise they keep 290 s.
    """
    n = 40
    clocks = []
    for i in range(n):
        burns = i % 2 == 0 or opp_low_blunders is not None
        clocks.append(max(5.0, 300.0 - 25.0 * (i // 2 + 1)) if burns else 290.0)

    def spec(i, user):
        if user and i >= 22 and (i - 22) // 2 < low_blunders:
            return {"judgement": "blunder"}
        if user and ok_blunder and i == 4:
            return {"judgement": "blunder"}
        if not user and opp_low_blunders and i >= 23 and (i - 23) // 2 < opp_low_blunders:
            return {"judgement": "blunder"}
        return {}

    return analysed(spec, n=n, clocks=clocks, base_seconds=300, time_control="300", outcome="loss")


def test_time_pressure_blunders_vs_your_opponents():
    pairs = [pressure_game(2, ok_blunder=(k % 4 == 0), opp_low_blunders=int(k % 4 == 0)) for k in range(20)]
    mr = run(pairs)
    ins = insight(mr, "engine.weakness.time-pressure-blunders-blitz")
    assert ins and ins.category == "time"
    assert ins.evidence["moves_low"] == 9 * 20 and ins.evidence["opp_moves_low"] == 9 * 20
    assert ins.evidence["per100_low"] == pytest.approx(100 * 40 / 180)
    assert ins.evidence["per100_ok"] == pytest.approx(100 * 5 / 220)
    assert ins.evidence["opp_per100_low"] == pytest.approx(100 * 5 / 180)
    row = table(mr, "Blunders when short of time").rows[0]
    assert row[:2] == ["Blitz", 180] and row[4] == 180 and row[5] == pytest.approx(100 * 5 / 180)


def test_time_pressure_that_hits_everyone_is_only_an_observation():
    same = run([pressure_game(2, ok_blunder=(k % 4 == 0), opp_low_blunders=2) for k in range(20)])
    assert insight(same, "engine.weakness.time-pressure-blunders-blitz") is None
    assert insight(same, "engine.observation.time-pressure-blunders-blitz") is not None
    alone = run([pressure_game(2, ok_blunder=(k % 4 == 0)) for k in range(20)])  # opponents never short of time
    assert insight(alone, "engine.weakness.time-pressure-blunders-blitz") is None
    assert "never that short of time" in insight(alone, "engine.observation.time-pressure-blunders-blitz").detail
    calm = run([pressure_game(0, ok_blunder=True) for _ in range(20)])
    assert not [i for i in calm.insights if i.category == "time"]


# --------------------------------------------------------------------------- conversion
def test_conversion_rates_and_thrown_games():
    pairs = []
    for k in range(20):  # you reach +85%: won 8, drew 4, lost 8
        outcome = "win" if k < 8 else "draw" if k < 12 else "loss"
        pairs.append(analysed(lambda i, u: {"win_before": 90.0} if u and i == 10 else {}, outcome=outcome))
    for k in range(20):  # your opponent reaches 85%+ (you at 10%): they won 19
        outcome = "loss" if k < 19 else "draw"
        pairs.append(analysed(lambda i, u: {"win_before": 90.0} if not u and i == 11 else {}, outcome=outcome))
    mr = run(pairs)
    stats = mr.stats["conversion"]
    assert (stats["reached"], stats["converted"], stats["rate"]) == (20, 8, 0.4)
    assert (stats["opp_reached"], stats["opp_converted"], stats["opp_rate"]) == (20, 19, 0.95)
    ins = insight(mr, "engine.weakness.conversion")
    assert ins and ins.category == "conversion"
    thrown_losses = {g.url for g, _ in pairs[12:20]}
    assert ins.example_games and set(ins.example_games) <= thrown_losses  # losses come first
    res = insight(mr, "engine.observation.resilience")
    assert res and res.evidence["saved"] == 1 and res.evidence["lost_positions"] == 20
    rows = table(mr, "Winning and losing positions").rows
    assert rows[0][1:5] == [20, 8, 4, 8] and rows[0][5] == pytest.approx(0.4)
    assert rows[1][1:5] == [20, 0, 1, 19] and rows[1][5] == pytest.approx(0.05)


def test_conversion_counts_each_game_once_for_whoever_got_there_first():
    def swing(i, user):  # you are winning at ply 10, then your opponent is winning at ply 21
        if user and i == 10:
            return {"win_before": 90.0}
        if not user and i == 21:
            return {"win_before": 92.0}
        return {}

    pairs = [analysed(swing, outcome="loss") for _ in range(12)]
    pairs += [analysed(lambda i, u: {"win_before": 95.0} if not u and i == 11 else {}, outcome="loss")
              for _ in range(12)]
    stats = run(pairs).stats["conversion"]
    assert (stats["reached"], stats["converted"]) == (12, 0)  # the swings count as your thrown wins only
    assert (stats["opp_reached"], stats["opp_converted"]) == (12, 12)


def test_conversion_needs_ten_games_each_way():
    pairs = [analysed(lambda i, u: {"win_before": 90.0} if u and i == 10 else {}, outcome="draw") for _ in range(9)]
    pairs += [analysed(lambda i, u: {"win_before": 95.0} if not u and i == 11 else {}, outcome="loss")
              for _ in range(12)]
    mr = run(pairs)
    assert insight(mr, "engine.weakness.conversion") is None and mr.stats["conversion"]["reached"] == 9


# --------------------------------------------------------------------------- tactics
def test_missed_tactics_and_costliest_examples():
    pairs = []
    for k in range(20):
        def spec(i, user, k=k):
            if user and i in (8, 16):
                return {"tags": ["missed_tactic"], "judgement": "blunder", "win_before": 60.0 + k, "win_after": 30.0}
            if not user and i == 9 and k % 2 == 0:
                return {"tags": ["missed_tactic"], "win_before": 60.0, "win_after": 40.0}
            if user and i == 20 and k < 3:
                return {"tags": ["missed_mate"]}
            return {}

        pairs.append(analysed(spec))
    mr = run(pairs)
    ins = insight(mr, "engine.weakness.missed-tactics")
    assert ins and ins.category == "tactics" and ins.evidence["count"] == 40 and ins.evidence["opp_count"] == 10
    assert ins.example_games[0] == pairs[-1][0].url  # the costliest miss first
    assert insight(mr, "engine.weakness.hung-material") is None
    mates = insight(mr, "engine.observation.missed-mates")
    assert mates and mates.evidence["missed_mates"] == 3
    row = table(mr, "Tactics").rows[0]
    assert row[1:] == [40, pytest.approx(10.0), 10, pytest.approx(2.5)]


# --------------------------------------------------------------------------- blunder anatomy
def test_blunder_anatomy_by_piece_and_move_number():
    assert [engine_stats.piece_of(s) for s in ("Qd1", "O-O", "O-O-O", "exd5", "e8=Q+", "Nbd2", "Kh1")] == [
        "Q", "K", "K", "P", "P", "N", "K"
    ]
    buckets_by_ply = [engine_stats.move_bucket(p) for p in (0, 19, 20, 79, 80, 200)]
    assert buckets_by_ply == ["1–10", "1–10", "11–20", "31–40", "41+", "41+"]

    def spec(i, user):  # your plies use SANS[even]: e4, Qd1, exd5, Bc4, Ne2 in turn
        return {"judgement": "blunder"} if user and i in (2, 12, 22, 32) else {}  # every one a queen move

    mr = run([analysed(spec) for _ in range(10)])
    pieces = {r[0]: r for r in table(mr, "Your blunders by piece moved").rows}
    assert pieces["Queen"][1:3] == [40, 40] and pieces["Queen"][4] == pytest.approx(1.0)
    assert pieces["Pawn"][1] == 80  # e4 and exd5
    ins = insight(mr, "engine.observation.blunder-anatomy")
    assert ins and "Queen moves" in ins.title and ins.evidence["blunders"] == 40
    buckets = table(mr, "Blunders by move number")
    chart = next(c for c in mr.charts if c.title.startswith("Blunders per 100 moves, by move number"))
    assert chart.series[0].values == [r[2] for r in buckets.rows]
    assert chart.series[1].values == [r[4] for r in buckets.rows]
    assert buckets.rows[0][1:3] == [100, pytest.approx(20.0)]  # moves 1-10: 2 blunders in 10 of your moves


# --------------------------------------------------------------------------- opening outcome
def test_opening_outcome_eval_after_move_ten():
    def at_ply_20(cp):
        return lambda i, user: {"cp_after": cp} if i == 19 else {}

    caro = [analysed(at_ply_20(120), color="black", opening_family="Caro-Kann Defense") for _ in range(11)]
    caro.append(analysed(at_ply_20(900), color="black", opening_family="Caro-Kann Defense"))  # capped at -500
    italian = [analysed(at_ply_20(20), opening_family="Italian Game") for _ in range(6)]
    extra = [
        analysed(at_ply_20(-600), n=18, opening_family="Italian Game"),  # ended before move 10: left out
        analysed(at_ply_20(-600), opening_family="Italian Game", initial_fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    ]
    mr = run(caro + italian + extra)
    rows = {(r[0], r[1]): r for r in table(mr, "Where your openings leave you (engine eval after move 10)").rows}
    caro_row = rows[("Caro-Kann Defense", "Black")]
    assert caro_row[2] == 12 and caro_row[3] == pytest.approx((11 * -120 - 500) / 12) and caro_row[4] == 12
    assert rows[("Italian Game", "White")][2:4] == [6, pytest.approx(20.0)]
    ins = insight(mr, "engine.weakness.openings-black-caro-kann-defense")
    assert ins and ins.category == "openings" and "Caro-Kann" in ins.title
    assert not any(i.id.startswith("engine.") and "italian" in i.id for i in mr.insights)  # 6 games: table only


# --------------------------------------------------------------------------- accuracy trend and by result
def test_accuracy_trend_and_accuracy_by_result():
    pairs = []
    start = datetime(2024, 1, 10, 12, tzinfo=timezone.utc)
    for month, (mine, theirs, count) in enumerate([(70.0, 75.0, 4), (74.0, 75.0, 3), (99.0, 50.0, 2), (80.0, 76.0, 3)]):
        for k in range(count):
            outcome = ("win", "loss", "draw")[k % 3]
            end = start + timedelta(days=31 * month, hours=k)
            pairs.append(analysed(my_acc=mine, opp_acc=theirs, outcome=outcome, end_time=end))
    mr = run(pairs)
    chart = next(c for c in mr.charts if c.kind == "line")
    assert chart.labels == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert chart.series[0].values == [70.0, 74.0, None, 80.0]  # March has only 2 games
    assert chart.series[1].values == [75.0, 75.0, None, 76.0]
    ins = insight(mr, "engine.observation.accuracy-by-result")
    assert ins and ins.kind == "observation"
    assert ins.evidence["games"] == 12
    wins = [a.my_accuracy for g, a in pairs if g.outcome == "win"]
    assert ins.evidence["win"] == pytest.approx(sum(wins) / len(wins))


# --------------------------------------------------------------------------- statistics: games are the unit
def clustered_world(seed: int, *, n_games: int = 120, n: int = 60, shape: float = 0.3, my_factor: float = 1.0):
    """Games whose errors come in bursts (one bad game holds several blunders), as real games do.

    Each side of each game gets its own error intensity from a Gamma(shape) distribution (small shape = heavy
    clustering); you and your opponents share it unless ``my_factor`` scales yours.
    """
    import random

    from chess_insights.models import PlyEval

    rng = random.Random(seed)
    pairs = []
    for _ in range(n_games):
        color = rng.choice(("white", "black"))
        game = make_game(color=color, moves_san=[SANS[i % len(SANS)] for i in range(n)],
                         outcome=rng.choice(("win", "draw", "loss")))
        rates = {}
        for side in ("white", "black"):
            base = rng.gammavariate(shape, 1.0 / shape)  # mean 1
            rates[side] = min(9.0, base * (my_factor if side == color else 1.0))  # error probability <= 0.9
        plies = []
        for i in range(n):
            mover = "white" if i % 2 == 0 else "black"
            lam = rates[mover]
            judgement = None
            u = rng.random()
            if u < 0.03 * lam:
                judgement = "blunder"
            elif u < 0.06 * lam:
                judgement = "mistake"
            elif u < 0.10 * lam:
                judgement = "inaccuracy"
            tags = []
            if judgement == "blunder" and rng.random() < 0.5:
                tags.append("missed_tactic")
            if judgement in ("blunder", "mistake") and rng.random() < 0.4:
                tags.append("hung_material")
            plies.append(PlyEval(
                ply=i, mover=mover, is_user=mover == color, san=game.moves_san[i], best_san=None, cp_before=0,
                cp_after=0, mate_before=None, mate_after=None, win_before=50.0, win_after=50.0,
                accuracy=90.0, cp_loss=150 if judgement else 10, judgement=judgement,
                phase=default_phase(i, n), clock_after=None, time_spent=None, tags=tags,
            ))
        pairs.append((game, make_game_eval(game.game_id, plies, my_accuracy=80.0, opp_accuracy=80.0)))
    return pairs


def test_clustered_errors_do_not_produce_false_claims():
    """You and your opponents err at the same rate, but in bursts: counting moves as independent observations
    (or deflating them by a fixed factor) makes chance differences look significant."""
    runs = 30
    false_claims = []
    for seed in range(runs):
        mr = engine_stats.analyze(AnalysisContext("tester", [g for g, _ in clustered_world(seed)],
                                                  evals={ev.game_id: ev for _, ev in clustered_world(seed)}))
        false_claims.append(len(claims(mr)))
    assert sum(false_claims) / runs <= 0.2, false_claims


def test_clustered_errors_still_detect_a_real_difference():
    found = 0
    runs = 12
    for seed in range(runs):
        pairs = clustered_world(1000 + seed, n_games=150, my_factor=2.0)
        mr = engine_stats.analyze(AnalysisContext("tester", [g for g, _ in pairs], evals={ev.game_id: ev for _, ev in pairs}))
        found += insight(mr, "engine.weakness.blunder-rate") is not None
    assert found >= 0.75 * runs


def test_rate_test_uses_games_as_the_unit():
    """Same totals, different spread over games: bursts (all errors in a few games) are weaker evidence."""
    spread = [engine_stats.Tally(moves=30, blunders=2) for _ in range(20)]
    bursty = [engine_stats.Tally(moves=30, blunders=20 if i < 2 else 0) for i in range(20)]
    opp = [engine_stats.Tally(moves=30, blunders=1) for _ in range(20)]
    even = engine_stats.clustered_rate_test(spread, opp, lambda t: t.blunders)
    burst = engine_stats.clustered_rate_test(bursty, opp, lambda t: t.blunders)
    assert even.mean == pytest.approx(burst.mean) == pytest.approx(1 / 30)
    assert even.n == burst.n == 20  # games, not moves
    assert burst.p_value > 0.1 > even.p_value
    empty = engine_stats.clustered_rate_test([], [], lambda t: t.blunders)
    assert empty.p_value == 1.0 and empty.n == 0
