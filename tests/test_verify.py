"""The fact check for LLM texts (C4): moves replayed along the packet's lines, numbers and claims from the packet."""

from __future__ import annotations

from datetime import date

import pytest

from chess_insights.coach.packet import build_packet, position_index
from chess_insights.coach.verify import (
    check_moves,
    named_moves,
    named_numbers,
    packet_numbers,
    verify_answer,
    verify_explanation,
    verify_plan,
)
from test_packet import (
    CLOCK_ID,
    OBSERVATION_ID,
    OPENING_ID,
    QGA_4E4,
    RAPID_ID,
    REPEATED_ID,
    SICILIAN_2NC6,
    SICILIAN_5E5,
    coaching_report,
    epd,
)


@pytest.fixture(scope="module")
def packet():
    return build_packet(coaching_report())


def check(packet, fen, text, claim_ids=()):
    return verify_explanation(text, position_index(packet)[epd(fen)], packet, list(claim_ids))


# --------------------------------------------------------------------------- moves
def test_move_notations_are_recognised():
    moves = named_moves(
        "Nf3, then 5...e5, 5... e5 and 5…e5; O-O and 0-0-0; exd5; e8=Q+ and Qxd4#; the d6 square; e4."
    )
    assert [(m.number, m.white, m.san) for m in moves] == [
        (None, None, "Nf3"),
        (5, False, "e5"),
        (5, False, "e5"),
        (5, False, "e5"),
        (None, None, "O-O"),
        (None, None, "O-O-O"),
        (None, None, "exd5"),
        (None, None, "e8Q"),
        (None, None, "Qxd4"),
    ]  # bare squares ("d6", "e4") are squares, not moves
    assert [(m.number, m.white) for m in named_moves("6.Ndb5 and 6. Ndb5 and 6. ...a6")] == [
        (6, True), (6, True), (6, False)
    ]
    # "...e5" is Black's move without a number
    assert [(m.number, m.white, m.san) for m in named_moves("the reply ...e5, or …Qxd4")] == [
        (None, False, "e5"), (None, False, "Qxd4")
    ]


@pytest.mark.parametrize(
    "fen, line, text",
    [
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", ["2.exd5", "2...Qxd5"],
         "2.exd5 Qxd5 is the line."),
        ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", ["4.O-O"], "Castle with O-O first."),
        ("k7/4P3/8/8/8/8/8/4K3 w - - 0 1", ["1.e8=Q+"], "Promote with e8=Q+ at once."),
        (SICILIAN_5E5, ["5...e5", "6.Ndb5"], "After 5... e5 comes Ndb5."),
    ],
)
def test_moves_in_the_lines_pass(fen, line, text):
    assert check_moves(text, [{"fen": fen, "best_line": {"moves": line}}]) == []


def test_illegal_move_is_rejected(packet):
    result = check(packet, SICILIAN_5E5, "After 5...e5 the answer 6.Qxh7 wins at once.")
    assert not result.ok
    assert any("6.Qxh7" in p and "not a legal move" in p for p in result.problems)


def test_legal_move_outside_the_lines_is_rejected(packet):
    result = check(packet, SICILIAN_5E5, "After 5...e5 White plays 6.Nf5 and wins.")
    assert not result.ok and any("6.Nf5" in p and "not in the position's lines" in p for p in result.problems)


def test_wrong_move_number_is_rejected(packet):
    assert check(packet, SICILIAN_5E5, "The knight lands with 7.Nd6+ after 6.Ndb5.").ok
    assert not check(packet, SICILIAN_5E5, "The knight lands with 8.Nd6+ after 6.Ndb5.").ok
    assert not check(packet, SICILIAN_5E5, "The knight lands with 7...Nd6+.").ok  # White's move given as Black's


def test_black_moves_without_numbers_are_checked(packet):
    assert check(packet, SICILIAN_5E5, "After ...e5 the knight jumps to b5.").ok
    assert not check(packet, SICILIAN_5E5, "After ...Qb6 the knight is pinned.").ok
    assert not check(packet, SICILIAN_5E5, "White answers ...Ndb5.").ok  # Ndb5 is White's move


def test_moves_from_another_position_are_rejected(packet):
    # 4...Qxd4 is in the QGA lines, not in the Sicilian's
    assert check(packet, QGA_4E4, "4.e4 drops the d4 pawn to 4...Qxd4.").ok
    assert not check(packet, SICILIAN_5E5, "Like 4...Qxd4, this drops a pawn.").ok


def test_opening_database_moves_count_as_given_lines(packet):
    assert check(packet, QGA_4E4, "Masters play 4.Nf3 and 4.d5 here, never 4.e4.").ok


# --------------------------------------------------------------------------- numbers
def test_numbers_are_found_but_not_move_numbers_squares_or_small_words():
    nums = named_numbers("After 5...e5 (you played it twice, in 10 of 21 games) the d6 square costs 1.52 pawns, "
                         "31% of the time, −0.3 on mobility, 1,132 games and ten more.")
    assert [(n.value, n.percent) for n in nums] == [
        (10.0, False), (21.0, False), (1.52, False), (31.0, True), (0.3, False), (1132.0, False), (10.0, False)
    ]


def test_board_geometry_is_not_a_number(packet):
    assert check(packet, SICILIAN_5E5, "After 5...e5 the knight reaches your 6th rank.").ok
    assert [n.value for n in named_numbers("rank 7 and the 2nd rank, but 12 games")] == [12.0]


def test_invented_evaluation_is_rejected(packet):
    result = check(packet, SICILIAN_5E5, "After 5...e5 you end up 2.7 pawns worse than after 5...a6.")
    assert not result.ok and any("2.7" in p for p in result.problems)


def test_rounded_numbers_and_percentages_pass(packet):
    text = "After 5...e5 you stand 1.5 pawns worse, against 0.5 after 5...a6, and you lose 21% of your winning chances."
    assert check(packet, SICILIAN_5E5, text).ok
    # 0.1 is the tolerance: 1.52 may become 1.6 but not 1.4
    assert check(packet, SICILIAN_5E5, "After 5...e5 you stand 1.6 pawns worse.").ok
    assert not check(packet, SICILIAN_5E5, "After 5...e5 you stand 1.4 pawns worse.").ok


def test_numbers_from_the_linked_claim_pass_but_not_from_elsewhere(packet):
    # 10 of 21 games comes from the repeated-mistake claim this position belongs to
    assert check(packet, SICILIAN_5E5, "You played 5...e5 in 10 of 21 games.").ok
    # 63 games belongs to a different claim that the text doesn't cite
    assert not check(packet, SICILIAN_5E5, "You played 5...e5 in 63 games.").ok
    assert check(packet, SICILIAN_5E5, "The Sicilian Defense has cost you over 63 games.", [OPENING_ID]).ok


def test_percentages_match_shares_in_the_packet(packet):
    positions = position_index(packet)
    q = dict(positions[epd(QGA_4E4)])
    assert packet_numbers(q)  # shares 0.45 and 0.3 of the masters' moves
    assert check(packet, QGA_4E4, "Masters choose 4.Nf3 in 45% of their games.").ok
    assert not check(packet, QGA_4E4, "Masters choose 4.Nf3 in 62% of their games.").ok


def test_links_and_dates_must_be_in_the_packet(packet):
    assert check(packet, SICILIAN_5E5, "Practise forks at https://lichess.org/training/fork.").ok
    assert not check(packet, SICILIAN_5E5, "Read https://example.com/sicilian first.").ok
    assert not check(packet, SICILIAN_5E5, "You first played 5...e5 on 2025-01-03.").ok


# --------------------------------------------------------------------------- claims
def test_invented_claim_is_rejected(packet):
    result = check(packet, SICILIAN_5E5, "This is typical of your weakness in endgames.")
    assert not result.ok and any("weakness" in p for p in result.problems)


def test_cited_ids_must_be_claims(packet):
    assert not check(packet, SICILIAN_5E5, "After 5...e5 comes 6.Ndb5.", ["openings.weakness.white.london-system"]).ok
    assert not check(packet, SICILIAN_5E5, "After 5...e5 comes 6.Ndb5.", [OBSERVATION_ID]).ok  # an observation
    assert not check(packet, SICILIAN_5E5, f"See {OBSERVATION_ID} for more.").ok
    assert check(packet, SICILIAN_5E5, "After 5...e5 comes 6.Ndb5.", [OPENING_ID]).ok


def test_weakness_wording_needs_a_claim_on_that_subject(packet):
    # the position's own claim (the repeated mistake) and a cited claim both back their own subject
    assert check(packet, SICILIAN_5E5, "Playing 5...e5 in the Open Sicilian is your weakness here.").ok
    text = "It is part of why the Sicilian is costing you points as Black."
    assert check(packet, SICILIAN_5E5, text, [OPENING_ID]).ok
    # ... but not another subject
    assert not check(packet, SICILIAN_5E5, "It shows your weakness on the clock in bullet.", [OPENING_ID]).ok
    # a position with no claim behind it cannot be called a weakness
    assert not check(packet, SICILIAN_2NC6, "2...Nc6 is your weakness in the Sicilian.").ok
    # "weakness" of a square is chess, not a claim about you
    assert check(packet, SICILIAN_5E5, "5...e5 leaves a weakness on d6 that 7.Nd6+ uses.").ok
    assert check(packet, SICILIAN_5E5, "7.Nd6+ lands on your d6 weakness.").ok
    assert check(packet, SICILIAN_5E5, "It hits your dark-square weakness.").ok


def test_strength_wording_needs_a_strength_claim(packet):
    assert check(packet, QGA_4E4, "Rapid is your strength, so slow down: 4.Nf3 keeps d4 covered.", [RAPID_ID]).ok
    assert not check(packet, QGA_4E4, "Openings are your strength.", [OPENING_ID]).ok
    assert not check(packet, QGA_4E4, "You are good at blitz openings.", [CLOCK_ID]).ok


# --------------------------------------------------------------------------- formats
def test_the_format_named_must_be_the_games_or_the_claims(packet):
    assert check(packet, SICILIAN_5E5, "In this blitz game 5...e5 lost the bishop to 7.Nd6+.").ok
    result = check(packet, SICILIAN_5E5, "In this bullet game 5...e5 lost the bishop to 7.Nd6+.")
    assert not result.ok and any("says bullet" in p for p in result.problems)
    # the cited opening claim covers bullet and blitz games
    assert check(packet, SICILIAN_5E5, "The Sicilian Defense costs you in bullet too.", [OPENING_ID]).ok
    assert check(packet, QGA_4E4, "Rapid is your strength: take the time to play 4.Nf3.", [RAPID_ID]).ok


# --------------------------------------------------------------------------- style
def test_style_rules(packet):
    four = "5...e5 leaves d6 weak. 6.Ndb5 hits it. 7.Nd6+ takes the bishop. Play 5...a6 instead."
    assert any("4 sentences" in p for p in check(packet, SICILIAN_5E5, four).problems)
    three = "5...e5 leaves d6 weak. After 6.Ndb5 a6 7.Nd6+ you are 1.52 pawns down. Play 5...a6 instead."
    assert check(packet, SICILIAN_5E5, three).ok  # move dots and decimals do not end sentences
    assert not check(packet, SICILIAN_5E5, "After 5...e5 you lose 99 centipawns.").ok
    assert not check(packet, SICILIAN_5E5, "After **5...e5** comes 6.Ndb5.").ok
    assert not check(packet, SICILIAN_5E5, "   ").ok


def test_template_texts_pass_the_verifier(packet):
    for p in packet["positions"]:
        result = verify_explanation(p["template_text"], p, packet)
        assert result.ok, (p["epd"], result.problems)


# --------------------------------------------------------------------------- answers and the weekly plan
def test_answer_checks_moves_in_the_positions_it_names(packet):
    good = "In the Queen's Gambit Accepted, 4.e4 lets 4...Qxd4 take the d4 pawn; 4.Nf3 keeps it."
    assert verify_answer(good, packet, epds=[epd(QGA_4E4)]).ok
    assert verify_answer(good, packet).ok  # no position named: any position's lines
    assert not verify_answer(good, packet, epds=[epd(SICILIAN_5E5)]).ok
    assert not verify_answer("Nothing here.", packet, epds=["8/8/8/8/8/8/8/8 w - -"]).ok
    assert verify_answer("Your Sicilian Defense as Black scores −16 per 100 games over 63 games.", packet).ok
    assert 83.0 not in packet_numbers(packet) and 0.83 not in packet_numbers(packet)
    assert not verify_answer("You lose 83% of your Sicilian games.", packet).ok


def test_plan_keeps_study_items_only_and_fits_the_budget(packet):
    items = [p["title"] for p in packet["study_plan"]]
    today = date(2026, 9, 26)
    entries = [
        {"item": items[0], "minutes_per_week": 60, "recheck_date": "2026-10-24"},
        {"item": "Learn the Najdorf main line", "minutes_per_week": 60, "recheck_date": "2026-10-24"},
        {"item": packet["study_plan"][1]["id"], "minutes_per_week": 45, "recheck_date": "next month"},
        {"item": items[0].upper(), "minutes_per_week": 30, "recheck_date": "2026-10-24"},  # a repeat
    ]
    plan, dropped = verify_plan(entries, packet, practice_minutes=20, today=today)
    assert [p.item for p in plan] == [items[0], items[1]]
    assert plan[0].recheck_date == "2026-10-24"
    assert plan[1].recheck_date == "2026-10-24"  # an unreadable date becomes four weeks from today
    assert plan[0].note == packet["study_plan"][0]["target"]
    assert len(dropped) == 2 and "Najdorf" in dropped[0]

    # 105 minutes a week do not fit 10 minutes a day: scaled to 70 in steps of 5
    plan, _ = verify_plan(entries[:1] + entries[2:3], packet, practice_minutes=10, today=today)
    assert sum(p.minutes_per_week for p in plan) <= 70
    assert all(p.minutes_per_week % 5 == 0 and p.minutes_per_week >= 5 for p in plan)


def test_verify_handles_an_empty_packet():
    empty = build_packet({})
    assert not verify_answer("After 1.e4 you win.", empty).ok
    assert verify_answer("Your report has no games yet.", empty).ok
    entry = {"item": "x", "minutes_per_week": 10, "recheck_date": ""}
    assert verify_plan([entry], empty, 20, date(2026, 1, 1))[0] == []


def test_repeated_claim_id_is_known(packet):
    # the fixture's ids line up with the report's claims
    assert {REPEATED_ID, OPENING_ID, CLOCK_ID, RAPID_ID} == {c["id"] for c in packet["claims"]}
