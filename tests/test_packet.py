"""The coaching packet (C4): what the LLM coach gets, built from the report or its JSON export.

``coaching_report()`` is shared with the other C4 tests (test_verify, test_llm, test_ask, test_progress).
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from chess_insights.coach import packet as packet_mod
from chess_insights.coach.packet import build_packet, claim_index, numbered_moves, position_index, user_ratings
from chess_insights.insights import build_study_plan
from chess_insights.models import (
    Coaching,
    ConceptDelta,
    Explanation,
    Insight,
    Line,
    ModuleResult,
    Motif,
    MoveStat,
    OpeningFacts,
    Report,
    Source,
)
from chess_insights.report.json_export import to_dict

# The three habit positions from the coaching plan (positions before the move you played).
SICILIAN_5E5 = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 0 5"
QGA_4E4 = "r1bqkbnr/ppp1pppp/2n5/8/2pP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"
SICILIAN_2NC6 = "rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
REPEATED_ID = "mistakes.weakness.sicilian-5-e5"
OPENING_ID = "openings.weakness.black.sicilian-defense"
CLOCK_ID = "time.weakness.slow-opening-blitz"
RAPID_ID = "results.strength.time-control-rapid"
OBSERVATION_ID = "engine.observation.missed-mates"


def epd(fen: str) -> str:
    return " ".join(fen.split()[:4])


def habit_explanations() -> list[Explanation]:
    """Explanations for the three habit positions, with the lines Stockfish 16 gives for them."""
    sicilian = Explanation(
        epd=epd(SICILIAN_5E5),
        fen=SICILIAN_5E5,
        played="5...e5",
        best="5...a6",
        best_line=Line(SICILIAN_5E5, ["a7a6", "f1e2", "g8f6"], cp_end=-53, depth=20),
        refutation=Line(SICILIAN_5E5, ["e6e5", "d4b5", "a7a6", "b5d6", "f8d6", "d1d6"], cp_end=-152, depth=20),
        motifs=[Motif("fork", "refutation", ply=3, squares=["d6", "e8", "b7"], side="opponent")],
        concepts=[
            ConceptDelta("Mobility", -0.31, label="piece activity"),
            ConceptDelta("King safety", -0.19, label="king safety"),
            ConceptDelta("Pawns", -0.18, label="pawn structure"),
        ],
        text="After 5...e5 the knight jumps to d6 with check (6.Ndb5 and 7.Nd6+) and takes your dark-squared bishop.",
        insight_id=REPEATED_ID,
        kind="repeated",
        drop=21.0,
        time_class="blitz",
        color="black",
        game_url="https://www.chess.com/game/live/1001",
        games=["https://www.chess.com/game/live/1001", "https://www.chess.com/game/live/1002"],
        repeats=10,
        drill_themes=["fork"],
    )
    qga = Explanation(
        epd=epd(QGA_4E4),
        fen=QGA_4E4,
        played="4.e4",
        best="4.Nf3",
        best_line=Line(QGA_4E4, ["g1f3", "g8f6"], cp_end=53, depth=20),
        refutation=Line(QGA_4E4, ["e2e4", "d8d4"], cp_end=-67, depth=20),
        motifs=[Motif("hangingPiece", "refutation", ply=1, squares=["d4"], side="opponent")],
        text="4.e4 leaves d4 attacked twice and defended once, and 4...Qxd4 takes it.",
        kind="error",
        drop=15.0,
        time_class="rapid",
        color="white",
        game_url="https://www.chess.com/game/live/2001",
        opening=OpeningFacts(
            eco="D20",
            name="Queen's Gambit Accepted",
            masters=[
                MoveStat("Nf3", "g1f3", games=120, share=0.45, score=0.58),
                MoveStat("d5", "d4d5", games=80, share=0.3),
            ],
            sources=[Source("Lichess opening explorer (masters)", url="https://explorer.lichess.org/masters?play=d2d4",
                            retrieved="2026-09-26", license="CC0")],
        ),
    )
    sicilian_2 = Explanation(
        epd=epd(SICILIAN_2NC6),
        fen=SICILIAN_2NC6,
        played="2...Nc6",
        best="2...cxd4",
        best_line=Line(SICILIAN_2NC6, ["c5d4"], cp_end=-14),
        refutation=Line(SICILIAN_2NC6, ["b8c6", "d4d5", "c6b8"], cp_end=-160),
        text="3.d5 chases the knight back to b8.",
        insight_id=OBSERVATION_ID,
        kind="error",
        drop=18.0,
        time_class="bullet",
        color="black",
        game_url="https://www.chess.com/game/live/3001",
    )
    return [sicilian, qga, sicilian_2]


def coaching_report() -> Report:
    """A small but complete report: ratings per format, claims, a study plan with baselines, three explanations."""
    strengths = [
        Insight(RAPID_ID, "strength", "results", "You do best in rapid",
                "+6 per 100 games vs your rating in 1,132 rapid games.", 0.3, 0.9,
                evidence={"n": 1132, "delta": 0.06, "time_class": "rapid"}, formats={"rapid": 1132}),
    ]
    weaknesses = [
        Insight(OPENING_ID, "weakness", "openings", "The Sicilian Defense is costing you points as Black",
                "−16 per 100 games vs your rating over 63 games.", 0.5, 0.8,
                evidence={"family": "Sicilian Defense", "color": "black", "n": 63, "delta": -0.16},
                formats={"bullet": 23, "blitz": 40}),
        Insight(CLOCK_ID, "weakness", "time", "You spend too much clock on the opening in blitz",
                "You use 50% of your clock on your first 15 moves; your opponents use 35%.", 0.4, 0.9,
                evidence={"your_share": 0.5, "opponent_share": 0.35, "time_class": "blitz", "n": 400},
                formats={"blitz": 400}),
        Insight(REPEATED_ID, "weakness", "positions", "You keep playing 5...e5 in the Open Sicilian",
                "You played 5...e5 in 10 of 21 games that reached this position.", 0.6, 0.9,
                evidence={"errors": 10, "reached": 21, "move": "e5", "best": "a6", "fen": SICILIAN_5E5},
                formats={"blitz": 21}),
    ]
    results = ModuleResult(
        key="results",
        title="Results",
        summary="",
        stats={"by_time_control": {
            "Bullet": {"rating": {"current": 650, "n": 652}},
            "Blitz": {"rating": {"current": 949, "n": 949}},
            "Rapid": {"rating": {"current": 1130, "n": 1132}},
            "Blitz (chess960)": {"rating": {"current": 1200, "n": 30}},
        }},
    )
    return Report(
        username="BigMuffEater",
        generated_at=datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
        filters="rated games, standard chess",
        n_games=2733,
        date_from=None,
        date_to=None,
        modules=[results],
        strengths=strengths,
        weaknesses=weaknesses,
        study_plan=build_study_plan(weaknesses),
        engine_note="Stockfish 16 at depth 12 on 300 games: 100 bullet, 100 blitz, 100 rapid.",
        summary_lines=["Your ratings: rapid 1130, blitz 949, bullet 650."],
        formats={"bullet": 652, "blitz": 949, "rapid": 1132},
        engine_formats={"bullet": 100, "blitz": 100, "rapid": 100},
        coaching=Coaching(explanations=habit_explanations(), notes=[], settings={"depth": 20}),
    )


def test_packet_schema():
    report = coaching_report()
    packet = build_packet(report)
    assert set(packet) == {
        "packet_version", "player", "report", "claims", "format_claims", "study_plan", "drills", "progress",
        "positions",
    }
    json.dumps(packet, allow_nan=False)  # plain JSON

    assert packet["player"]["ratings"] == {"bullet": 650, "blitz": 949, "rapid": 1130}
    assert packet["player"]["engine_games_per_format"] == {"bullet": 100, "blitz": 100, "rapid": 100}
    assert packet["report"]["date"] == "2026-09-26"

    # every claim, with its numbers and the formats behind it
    claims = {c["id"]: c for c in packet["claims"]}
    assert set(claims) == {RAPID_ID, OPENING_ID, CLOCK_ID, REPEATED_ID}
    for c in packet["claims"]:
        assert {"id", "kind", "title", "detail", "evidence", "formats"} <= set(c)
        assert c["kind"] in ("strength", "weakness")
    assert claims[OPENING_ID]["formats"] == {"bullet": 23, "blitz": 40}
    assert claims[CLOCK_ID]["evidence"]["your_share"] == 0.5

    # the study plan with its targets and baselines
    plan = {p["title"]: p for p in packet["study_plan"]}
    clock = plan["You spend too much clock on the opening in blitz"]
    assert clock["id"].startswith("plan-")
    assert "50%" in clock["target"]
    assert clock["baseline"]["metric"] == "share of the clock used on the first 15 moves (blitz)"
    assert clock["baseline"]["value"] == 0.5

    # positions: FEN, moves as numbered SAN, evaluations in pawns from your side, motifs, concepts, sources
    positions = position_index(packet)
    assert list(positions) == [epd(SICILIAN_5E5), epd(QGA_4E4), epd(SICILIAN_2NC6)]
    p = positions[epd(SICILIAN_5E5)]
    assert p["fen"] == SICILIAN_5E5 and p["time_class"] == "blitz" and p["you_play"] == "black"
    assert p["played"] == "5...e5" and p["best"] == "5...a6"
    assert p["refutation"]["moves"] == ["5...e5", "6.Ndb5", "6...a6", "7.Nd6+", "7...Bxd6", "8.Qxd6"]
    assert p["refutation"]["eval_pawns"] == -1.52
    assert p["best_line"]["moves"] == ["5...a6", "6.Be2", "6...Nf6"] and p["best_line"]["eval_pawns"] == -0.53
    assert p["pawns_lost"] == 0.99
    assert p["win_chance_lost"] == 21.0 and p["repeats"] == 10
    assert p["motifs"] == [
        {"theme": "fork", "line": "refutation", "side": "opponent", "squares": ["d6", "e8", "b7"],
         "drill": "https://lichess.org/training/fork"}
    ]
    assert [c["pawns"] for c in p["concepts"]] == [-0.31, -0.19, -0.18]
    assert p["insight_id"] == REPEATED_ID and p["is_claim"] is True
    assert p["drills"] == ["https://lichess.org/training/fork"]
    assert p["template_text"].startswith("After 5...e5")

    q = positions[epd(QGA_4E4)]
    assert q["time_class"] == "rapid" and q["refutation"]["moves"] == ["4.e4", "4...Qxd4"]
    assert q["opening"]["name"] == "Queen's Gambit Accepted"
    assert [m["move"] for m in q["opening"]["masters"]] == ["4.Nf3", "4.d5"]
    assert q["opening"]["sources"][0]["url"].startswith("https://explorer.lichess.org/")
    assert "is_claim" in q and q["is_claim"] is False  # an error that no claim is about

    # an observation is not a claim
    assert positions[epd(SICILIAN_2NC6)]["is_claim"] is False
    assert OBSERVATION_ID not in claim_index(packet)


def test_packet_from_json_matches_packet_from_report():
    report = coaching_report()
    data = json.loads(json.dumps(to_dict(report)))
    assert build_packet(data) == build_packet(report)


def test_packet_holds_at_most_20_positions_one_per_position():
    report = coaching_report()
    first = report.coaching.explanations[0]
    many = []
    for i in range(30):
        e = copy.deepcopy(first)
        e.epd = f"{first.epd} #{i}"  # distinct keys
        many.append(e)
    dup = copy.deepcopy(first)
    dup.played = "5...d6"
    report.coaching.explanations = [first, dup] + many
    packet = build_packet(report)
    assert len(packet["positions"]) == packet_mod.MAX_POSITIONS == 20
    assert [p["played"] for p in packet["positions"]].count("5...d6") == 0  # the second text of the same EPD waits
    assert len(build_packet(report, max_positions=5)["positions"]) == 5


def test_packet_without_coaching_or_engine():
    report = coaching_report()
    report.coaching = None
    packet = build_packet(report)
    assert packet["positions"] == [] and len(packet["claims"]) == 4
    empty = build_packet({})
    assert empty["claims"] == [] and empty["positions"] == [] and empty["player"]["ratings"] == {}


def test_format_claims_are_listed_per_format():
    report = coaching_report()
    blitz = copy.deepcopy(report)
    blitz.time_class = "blitz"
    blitz.format_reports = {}
    blitz.strengths = []
    blitz.weaknesses = [w for w in blitz.weaknesses if w.id == CLOCK_ID]
    report.format_reports = {"blitz": blitz}
    packet = build_packet(report)
    assert [c["id"] for c in packet["format_claims"]["blitz"]] == [CLOCK_ID]
    assert packet["format_claims"]["blitz"][0]["view"] == "blitz"
    assert CLOCK_ID in claim_index(packet)


def test_numbered_moves_replay_and_stop_at_an_illegal_move():
    assert numbered_moves(SICILIAN_5E5, ["e6e5", "d4b5"]) == ["5...e5", "6.Ndb5"]
    assert numbered_moves(SICILIAN_5E5, None, ["e5", "Ndb5", "a6"]) == ["5...e5", "6.Ndb5", "6...a6"]
    assert numbered_moves(SICILIAN_5E5, ["e6e5", "d4d8", "a7a6"]) == ["5...e5"]  # d4d8 is not a move
    assert numbered_moves("not a fen", ["e2e4"]) == []


def test_user_ratings_skip_variant_pools():
    assert user_ratings(coaching_report()) == {"bullet": 650, "blitz": 949, "rapid": 1130}


def test_long_quoted_texts_are_cut_at_a_sentence():
    report = coaching_report()
    qga = report.coaching.explanations[1]
    qga.opening.wiki_text = "The Queen's Gambit Accepted gives up the centre for a pawn. " * 60
    wiki = position_index(build_packet(report))[epd(QGA_4E4)]["opening"]["wiki_text"]
    assert len(wiki) <= packet_mod.MAX_TEXT and wiki.endswith(".")


def test_a_line_found_from_another_game_is_numbered_as_this_position():
    report = coaching_report()
    exp = report.coaching.explanations[0]
    other_game = SICILIAN_5E5.replace(" 0 5", " 0 12")  # the same position, other move counters
    exp.refutation.fen = other_game
    p = position_index(build_packet(report))[epd(SICILIAN_5E5)]
    assert p["refutation"]["moves"][:2] == ["5...e5", "6.Ndb5"] and "fen" not in p["refutation"]
