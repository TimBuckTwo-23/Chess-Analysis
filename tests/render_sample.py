"""A realistic report for the renderer tests and screenshots: findings with charts and boards, format views and
coaching with lines of play. Built with the shared ``visuals`` helpers, as the analysis modules and the coach do.

The positions are the three habit positions from the coaching plan (the Open Sicilian with 5...e5, the Queen's
Gambit Accepted with 4.e4 and 2...Nc6 against 2.d4); the drill puzzles are real Lichess puzzles (CC0).
"""

from __future__ import annotations

from datetime import datetime, timezone

import chess

from chess_insights import visuals
from chess_insights.models import (
    Arrow,
    Chart,
    Coaching,
    ConceptDelta,
    Diagram,
    Drill,
    DrillPuzzle,
    Explanation,
    Insight,
    Kpi,
    Line,
    Mark,
    ModuleResult,
    Motif,
    MoveStat,
    OpeningFacts,
    PlanEntry,
    ProgressItem,
    Report,
    ReviewItem,
    Source,
    StudyItem,
    Table,
)

GAME = "https://www.chess.com/game/live/{}"
FORMATS = {"bullet": 652, "blitz": 949, "rapid": 1130}
ENGINE_FORMATS = {"bullet": 100, "blitz": 100, "rapid": 100}

SICILIAN = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
QGA = "r1bqkbnr/ppp1pppp/2n5/8/2pP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"
SICILIAN_2 = "rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
ROOK_END = "8/8/4k3/8/8/3K4/8/R7 w - - 0 60"

# Real Lichess puzzles (CC0): (PuzzleId, FEN before the opponent's move, Moves, Rating, Themes)
PUZZLES = [
    ("01243", "r7/pp1Qp1B1/8/3kb3/4q3/8/PP3PP1/5K2 b - - 4 34", "e5d6 d7b7 d5e6 b7e4", 1527, "crushing deflection endgame fork master short skewer"),
    ("00KNK", "r3r1k1/p1p4p/3b4/4p1qN/8/1P1b1Q2/P2P1PP1/B5KR b - - 2 22", "e5e4 h5f6 g8f8 f6h7 f8e7 a1f6 g5f6 f3f6", 2002, "advantage deflection discoveredCheck doubleCheck fork middlegame veryLong"),
    ("00O9Z", "5QR1/5p1p/1b3qp1/p6k/P2P4/8/1P2rPPP/5RK1 w - - 7 32", "g8h8 f6f2 f1f2 e2e1 f2f1 b6d4 g1h1 e1f1", 1615, "backRankMate deflection endgame fork master mate mateIn4 pin sacrifice veryLong"),
    ("014It", "1k1r4/p1p3pp/Pp3r2/3pR3/Q2N4/8/1PPq1PPP/4R1K1 w - - 3 25", "e5e8 d2f2 g1h1 f2f1 e1f1 f6f1", 1369, "backRankMate long mate mateIn3 middlegame sacrifice"),
    ("00InW", "8/2p5/pp1p4/3PnN2/PPP1Pp2/5P1p/5K1k/8 b - - 3 46", "e5f3 f2f3 h2g1 f3f4 h3h2 f5g3", 1450, "crushing defensiveMove endgame hangingPiece knightEndgame long"),
    ("01M1X", "4r3/4bpkp/3p2pN/4P1P1/1p2qn1P/5N2/1P1Q1P2/3R2K1 w - - 0 30", "e5d6 e4f3 d2d4 f7f6", 1864, "crushing hangingPiece middlegame short"),
]


def drill_puzzle(pid: str, fen: str, moves: str, rating: int, themes: str) -> DrillPuzzle:
    """A Lichess puzzle as the coach stores it: the opponent's first move is pushed, the rest is the solution."""
    board = chess.Board(fen)
    first, *solution = moves.split()
    board.push_uci(first)
    san, b = [], board.copy()
    for uci in solution:
        move = chess.Move.from_uci(uci)
        san.append(b.san(move))
        b.push(move)
    return DrillPuzzle(pid, board.fen(), solution, san, rating, themes.split(), f"https://lichess.org/training/{pid}")


def _line(fen: str, uci: list[str], cp: int | None = None, mate: int | None = None) -> Line:
    board = chess.Board(fen)
    san = []
    for u in uci:
        move = chess.Move.from_uci(u)
        san.append(board.san(move))
        board.push(move)
    return Line(fen=fen, moves_uci=uci, moves_san=san, cp_end=cp, mate_end=mate, fen_end=board.fen(), depth=20)


def _explanation(
    fen: str, played: str, best: str, refutation: list[str], best_line: list[str], *, cp_ref: int, cp_best: int,
    color: str, time_class: str, kind: str = "error", repeats: int = 1, insight_id: str | None = None,
    motifs: list[Motif] | None = None, text: str = "", opening: OpeningFacts | None = None, game: int = 1,
    with_diagram: bool = True,
) -> Explanation:
    ref, bl = _line(fen, refutation, cp_ref), _line(fen, best_line, cp_best)
    diagram = None
    if with_diagram:
        diagram = visuals.position_diagram(
            f"{played} (better {best})", fen, orientation=color, played=played.split(".")[-1],
            best=best.split(".")[-1], link=GAME.format(game), time_class=time_class,
        )
        diagram.strips = [
            visuals.line_strip(f"What {played} allows", fen, refutation, max_frames=5,
                               captions=[""] * (min(5, len(refutation)) - 1) + [f"{cp_ref / 100:+.2f} for you"]),
            visuals.line_strip(f"Better: {best}", fen, best_line, max_frames=4),
        ]
    concepts = [
        ConceptDelta("Mobility", -0.31, -0.35, -0.2, "piece activity"),
        ConceptDelta("King safety", -0.19, -0.22, 0.0, "king safety"),
        ConceptDelta("Pawns", -0.18, -0.1, -0.3, "pawn structure"),
    ]
    chart = visuals.comparison_chart(
        "Where the two lines end, from your side (pawns)", [c.label for c in concepts],
        [("Refutation minus best line", [c.value for c in concepts])], value_format="signed_float2",
    )
    return Explanation(
        epd=" ".join(fen.split()[:4]), fen=fen, played=played, best=best, best_line=bl, refutation=ref,
        motifs=motifs or [], concepts=concepts,
        facts=["You give up the bishop pair.", "Your king stays in the centre after move 12."],
        text=text or f"After {played} the engine's reply wins material or space; {best} keeps the balance. "
        "Count the attackers on the squares your move leaves behind.",
        sources=[Source("Stockfish 16, depth 20", license="GPL-3.0", retrieved="2026-09-26")],
        insight_id=insight_id, kind=kind, drop=21.0, time_class=time_class, color=color,  # type: ignore[arg-type]
        game_url=GAME.format(game), games=[GAME.format(game + k) for k in range(1, 3)], repeats=repeats,
        opening=opening, diagram=diagram, chart=chart, drill_themes=["fork", "hangingPiece"],
    )


def sample_coaching(extra: int = 12) -> Coaching:
    sicilian_opening = OpeningFacts(
        eco="B45", name="Sicilian Defense: Taimanov Variation",
        masters=[MoveStat("a6", "a7a6", 2210, 0.52, 0.46), MoveStat("d6", "d7d6", 1011, 0.24, 0.47),
                 MoveStat("Nf6", "g8f6", 812, 0.19, 0.44), MoveStat("e5", "e6e5", 12, 0.003, 0.29)],
        peers=[MoveStat("Nf6", "g8f6", 50210, 0.31, 0.49), MoveStat("a6", "a7a6", 30115, 0.19, 0.5),
               MoveStat("e5", "e6e5", 22101, 0.14, 0.41)],
        peer_groups=[1200, 1400, 1600], played_rank_masters=4, played_rank_peers=3,
        master_game="https://lichess.org/Tq3ZrV2m",
        wiki_text="1...c5 is the Sicilian defence, a counter-attacking, asymmetric opening. Black controls the d4 "
        "square with a flank pawn.",
        wiki_url="https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._e4/1...c5",
        cloud_lines=[_line(SICILIAN, ["a7a6", "f1e2", "d8c7"], -30)],
        sources=[Source("Lichess opening explorer (masters)", "https://explorer.lichess.org/masters", "2026-09-26"),
                 Source("Wikibooks", "https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._e4/1...c5", "2026-09-26",
                        "CC BY-SA 4.0")],
    )
    exps = [
        _explanation(SICILIAN, "5...e5", "5...a6", ["e6e5", "d4b5", "a7a6", "b5d6", "f8d6", "d1d6"],
                     ["a7a6", "f1e2", "g8f6", "e1g1"], cp_ref=-152, cp_best=-53, color="black", time_class="blitz",
                     kind="repeated", repeats=10, insight_id="positions.repeated.sicilian-5e5",
                     motifs=[Motif("fork", "refutation", 3, ["d6", "e8", "c8"], "opponent"),
                             Motif("advancedPawn", "best", 1, [], "you")],
                     text="5...e5 leaves d6 with no pawn guard, so the knight lands there with check and takes your "
                     "dark-squared bishop. At the end of that line you stand 0.31 pawns worse on piece activity than "
                     "after 5...a6. Before a pawn move, look at the squares it stops guarding.",
                     opening=sicilian_opening, game=11),
        _explanation(QGA, "4.e4", "4.Nf3", ["e2e4", "d8d4"], ["g1f3", "g8f6", "e2e4"], cp_ref=-67, cp_best=53,
                     color="white", time_class="rapid", kind="repeated", repeats=6,
                     insight_id="positions.repeated.qga-4e4",
                     motifs=[Motif("hangingPiece", "refutation", 1, ["d4"], "opponent")],
                     text="d4 is attacked twice (knight on c6, queen on d8) and defended once, so 4...Qxd4 wins it. "
                     "Count attackers and defenders before a pawn push.", game=21),
        _explanation(SICILIAN_2, "2...Nc6", "2...cxd4", ["b8c6", "d4d5", "c6b8"], ["c5d4", "c2c3", "g8f6"],
                     cp_ref=-160, cp_best=-14, color="black", time_class="bullet", kind="choice", repeats=17,
                     text="A knight on c6 is a target for d4-d5. Take on d4 first; ...Nc6 later comes with tempo "
                     "against a queen on d4.", game=31, with_diagram=False),
    ]
    rook = Explanation(
        epd=" ".join(ROOK_END.split()[:4]), fen=ROOK_END, played="60.Kc4", best="60.Ra6+", best_line=None,
        refutation=None, text="With a rook against a bare king this is a win; drive the king to the edge first.",
        # the shapes coach/sources/tablebase.py and coach/maia.py write: a move's "played_category" is Lichess's,
        # for the side to move after it (the opponent); "played_result" is yours. Maia's "rating" is a Lichess one.
        tablebase={"category": "win", "result": "win", "dtz": 21, "dtm": 21, "best": "Ra6+", "best_uci": "a1a6",
                   "best_result": "win", "played": "Kc4", "played_result": "win", "played_category": "loss"},
        kind="error", drop=9.0, time_class="blitz", color="white",
        maia={"rating": 1400, "chesscom_rating": 949, "model": "blitz", "p_best": 0.31, "p_played": 0.22},
        diagram=visuals.position_diagram("Rook ending", ROOK_END, played="Kc4", best="Ra6+", time_class="blitz"),
        sources=[Source("Lichess tablebase", "https://tablebase.lichess.org", "2026-09-26")],
    )
    exps.append(rook)
    for k in range(extra):  # more explanations: the list folds after the first ten
        fen = [SICILIAN, QGA, SICILIAN_2][k % 3]
        exps.append(_explanation(
            fen, *(("5...e5", "5...a6", ["e6e5", "d4b5"], ["a7a6", "f1e2"]) if k % 3 == 0 else
                   ("4.e4", "4.Nf3", ["e2e4", "d8d4"], ["g1f3", "g8f6"]) if k % 3 == 1 else
                   ("2...Nc6", "2...cxd4", ["b8c6", "d4d5"], ["c5d4", "c2c3"])),
            cp_ref=-100 - k, cp_best=-20, color="black" if k % 3 != 1 else "white",
            time_class=["bullet", "blitz", "rapid"][k % 3], game=100 + k))

    puzzles = [drill_puzzle(*p) for p in PUZZLES]
    motif_rows = [["Fork", 3.1, 2.4, 2.9, 2.2], ["Hanging piece", 5.2, 4.9, 6.0, 5.1], ["Back-rank mate", 0.8, 0.5, 1.1, 0.4]]
    motif_table = Table("Tactical patterns in the mistakes (per 100 moves)", ["Pattern", "You missed", "They missed",
                        "You allowed", "They allowed"], motif_rows, ["text", "float1", "float1", "float1", "float1"])
    return Coaching(
        explanations=exps,
        notes=["Opening explorer skipped: no Lichess token (LICHESS_TOKEN).", "Maia not installed: pip install maia2."],
        settings={"depth": 20, "max_positions": 150},
        motif_profile=motif_table,
        motif_chart=visuals.comparison_chart(
            "Patterns you miss vs your opponents (per 100 moves)", [r[0] for r in motif_rows],
            [("You missed", [r[1] for r in motif_rows]), ("They missed", [r[2] for r in motif_rows])],
            value_format="float1", table=motif_table),
        drills=[
            Drill("fork", "30 fork puzzles", "https://lichess.org/training/fork", "bigmuffeater-drill-fork.pgn",
                  (1200, 1600), puzzles[:2], "You missed 23 forks in 300 games (your opponents: 15)."),
            Drill("backRankMate", "30 back-rank mate puzzles", "https://lichess.org/training/backRankMate",
                  "bigmuffeater-drill-backRankMate.pgn", (1200, 1600), puzzles[2:4], "You allowed 9 back-rank mates."),
            Drill("hangingPiece", "30 hanging-piece puzzles", "https://lichess.org/training/hangingPiece", "",
                  (1200, 1600), puzzles[4:], "440 of your 1,121 mistakes left material hanging."),
        ],
        review_due=[ReviewItem("own:g11:9", "own", "5...e5 in the Open Sicilian (find the better move)", SICILIAN,
                               "https://lichess.org/analysis/" + SICILIAN.replace(" ", "_"), "2026-09-26", 1),
                    ReviewItem("lichess:01243", "drill", "Fork puzzle 01243", puzzles[0].fen, puzzles[0].url,
                               "2026-09-26", 2)],
        review=[ReviewItem(f"own:g{k}:9", "own", f"Your mistake in game {k}", [SICILIAN, QGA, SICILIAN_2][k % 3],
                           GAME.format(k), f"2026-09-{27 + k % 3}", 0) for k in range(10)],
        theory_exit=Table("Where you leave named opening theory", ["Line", "Games", "Average exit move", "First inaccuracy"],
                          [["Sicilian Defense: Taimanov", 21, 5.2, 7.4], ["Queen's Gambit Accepted", 7, 3.9, 4.1]],
                          ["text", "int", "float1", "float1"]),
        endgames=Table("Endgames checked with the tablebase", ["Ending", "Games", "Won positions not won", "Drawn positions lost"],
                       [["Rook", 12, 3, 1], ["Pawn", 9, 2, 2]], ["text", "int", "int", "int"]),
        weekly_plan=[PlanEntry("Fork puzzles (bigmuffeater-drill-fork.pgn)", 60, "2026-10-10", "10 minutes on six days"),
                     PlanEntry("Review your Sicilian 5...e5 games", 30, "2026-10-10")],
        progress=[ProgressItem("Clock", "clock_by_15", 0.5, 0.38, "clock used by move 15: 50% -> 38%", True),
                  ProgressItem("Blunders", "blunders", 2.1, 2.3, "blunders per 100 moves: 2.1 -> 2.3", False)],
        llm={"model": "claude-opus-5-5", "accepted": 12, "rejected": 2},
    )


def _finding(key: str, kind: str, category: str, title: str, detail: str, severity: float, confidence: float,
             formats: dict[str, int], chart: Chart | None = None, diagram: Diagram | None = None, games=()) -> Insight:
    return Insight(key, kind, category, title, detail, severity, confidence, evidence={"n": sum(formats.values())},
                   study=["Replay three of these games and find the moment it turned."],
                   example_games=list(games), formats=formats, chart=chart, diagram=diagram)


def sample_report(*, views: bool = True, coaching: bool = True, time_class: str = "", scale: float = 1.0) -> Report:
    """A report for BigMuffEater (bullet 652, blitz 949, rapid 1,130 games) with format views and coaching."""
    formats = {time_class: FORMATS[time_class]} if time_class else dict(FORMATS)
    fmt_labels = [visuals.FORMAT_NAMES[t] for t in formats]
    score_chart = visuals.comparison_chart(
        "Score vs expected by format", fmt_labels,
        [("You", [0.47, 0.51, 0.55][: len(fmt_labels)]), ("Expected", [0.5, 0.5, 0.52][: len(fmt_labels)])],
        reference=0.5, kind="bar")
    caro = _finding(
        "openings.weakness.black.sicilian-defense", "weakness", "openings",
        "The Open Sicilian with 5...e5 is costing you points as Black",
        "63 games as Black: −16 per 100 games vs your rating.", 0.7 * scale, 0.82,
        {k: v // 10 for k, v in formats.items()}, chart=score_chart, games=[GAME.format(11), GAME.format(12)])
    clock = _finding(
        "time.weakness.flagging", "weakness", "time", "You lose on time more than players at your level",
        "31% of your losses are on time, against 18% for your opponents.", 0.45, 0.74,
        {"blitz": formats.get("blitz", 0), "bullet": formats.get("bullet", 0)},
        chart=visuals.comparison_chart("Losses on time", ["You", "Your opponents"], [("Share", [0.31, 0.18])]),
        games=[GAME.format(41)])
    habit = _finding(
        "positions.repeated.sicilian-5e5", "weakness", "positions", "You keep playing 5...e5 in the Open Sicilian",
        "10 of 21 games; Stockfish prefers 5...a6.", 0.5, 0.9, {"blitz": 14, "rapid": 7},
        diagram=visuals.position_diagram("5...e5 (better 5...a6)", SICILIAN, orientation="black", played="e5",
                                         best="a6", time_class="blitz"))
    strength = _finding(
        "color.strength.white", "strength", "color", "You score well with White", "+6 per 100 games vs your rating.",
        0.4, 0.8, formats, chart=visuals.comparison_chart(
            "Score as White by format", fmt_labels, [("Difference", [0.04, 0.07, 0.05][: len(fmt_labels)])],
            value_format="signed_pct"))
    note = _finding("results.observation.formats", "observation", "results", "Most of your games are rapid",
                    "41% of your games are rapid.", 0.1, 0.9, formats)

    results = ModuleResult("results", "Results & rating", "You score about what your rating predicts.",
                           kpis=[Kpi("Games", sum(formats.values()), "int"), Kpi("Score", 0.51, "pct")],
                           charts=[score_chart], insights=[strength, note])
    openings = ModuleResult("openings", "Openings", "As Black the Open Sicilian loses points.", insights=[caro])
    time_mgmt = ModuleResult("time_mgmt", "Clock & time management", "You flag more than your opponents.",
                             insights=[clock])
    mistakes = ModuleResult(
        "mistakes", "Positions you keep getting wrong", "The same wrong move in the same position, game after game.",
        insights=[habit],
        tables=[Table("Your costliest mistakes (puzzles)", ["Opening", "Your move", "Better", "Winning chances lost",
                      "Position", "Game"],
                      [["Sicilian Defense", "5...e5", "5...a6", 0.21, "https://lichess.org/analysis/" + SICILIAN.replace(" ", "_"), GAME.format(11)],
                       ["Queen's Gambit Accepted", "4.e4", "4.Nf3", 0.18, "https://lichess.org/analysis/" + QGA.replace(" ", "_"), GAME.format(21)]],
                      ["text", "text", "text", "pct", "url", "url"], key_columns=[1, 2, 4])],
        diagrams=[
            visuals.position_diagram("Sicilian Defense: 5...e5", SICILIAN, orientation="black", played="e5", best="a6",
                                     caption="After 1.e4 c5 2.Nf3 Nc6 3.Nc3 e6 4.d4 cxd4 5.Nxd4. You played 5...e5 "
                                     "in 10 of 21 games.", link=GAME.format(11), time_class="blitz"),
            visuals.position_diagram("Queen's Gambit Accepted: 4.e4", QGA, played="e4", best="Nf3",
                                     others=[("Qxd4", "threat")], marks=[Mark("d4", "target"), Mark("c6", "attacker")],
                                     caption="After 1.d4 d5 2.c4 dxc4 3.Nc3 Nc6. You played 4.e4 in 6 of 7 games.",
                                     link=GAME.format(21), time_class="rapid"),
        ],
    )
    mistakes.diagrams[1].arrows.append(Arrow("d8", "d4", "threat"))
    mistakes.diagrams[1].strips = [visuals.line_strip("What 4.e4 allows", QGA, ["e2e4", "d8d4", "d1d4", "c6d4"])]
    engine = ModuleResult("engine_stats", "Engine review", "Accuracy and blunders in 300 analysed games.",
                          kpis=[Kpi("Accuracy", 71.2, "float1")])
    plan = [StudyItem("Fix the Open Sicilian (Black)", caro.detail, ["Learn 5...a6 and why 5...e5 fails."],
                      games=[GAME.format(11)], category="Opening repertoire", insight_ids=[caro.id, habit.id],
                      findings=[caro.title, habit.title], target="−5 per 100 games or better")]
    tc_name = visuals.FORMAT_NAMES.get(time_class, "")
    report = Report(
        username="BigMuffEater",
        generated_at=datetime(2026, 9, 26, 18, 0, tzinfo=timezone.utc),
        filters=f"rated {tc_name.lower() or 'bullet+blitz+rapid'}, standard chess",
        n_games=sum(formats.values()),
        date_from=datetime(2021, 3, 1, tzinfo=timezone.utc),
        date_to=datetime(2026, 9, 25, tzinfo=timezone.utc),
        modules=[results, openings, time_mgmt, engine, mistakes],
        strengths=[strength],
        weaknesses=[caro, habit, clock],
        study_plan=plan,
        engine_note="Stockfish 16 at depth 12 on 300 games: 100 bullet, 100 blitz, 100 rapid (most recent in each)."
        if not time_class else f"Stockfish 16 at depth 12 on {ENGINE_FORMATS[time_class]} {time_class} games.",
        summary_lines=["Bullet 652 · Blitz 949 · Rapid 1,130.", "Start with the Open Sicilian as Black."],
        game_labels={GAME.format(11): "Loss · 3+0 · vs 1012 · 12 Aug"},
        time_class=time_class,
        formats=formats,
        engine_formats={time_class: ENGINE_FORMATS[time_class]} if time_class else dict(ENGINE_FORMATS),
        coaching=sample_coaching() if coaching and not time_class else None,
    )
    if views and not time_class:
        report.format_reports = {tc: sample_report(views=False, time_class=tc, scale=0.9) for tc in ("bullet", "blitz", "rapid")}
    return report
