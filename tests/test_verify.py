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


# --------------------------------------------------------------------------- review fixes
def test_moves_written_in_a_row_must_follow_each_other_in_one_line(packet):
    # each move is in some line, but 6.Ndb5 does not follow 5...a6 (the best line goes 5...a6 6.Be2)
    result = check(packet, SICILIAN_5E5, "After 5...a6 6.Ndb5 you win a piece.")
    assert not result.ok and any("do not follow each other" in p for p in result.problems)
    assert check(packet, SICILIAN_5E5, "Stockfish's answer is 6.Ndb5 a6 7.Nd6+ Bxd6 8.Qxd6.").ok
    # the same line numbered one move off
    assert not check(packet, SICILIAN_5E5, "Stockfish's answer is 6.Ndb5 a6 8.Nd6+.").ok
    # moves in separate phrases are checked one by one
    assert check(packet, SICILIAN_5E5, "Instead of 5...e5, play 5...a6; after 6.Be2 you are fine.").ok


def test_a_bare_square_after_a_white_move_is_blacks_reply(packet):
    assert check(packet, SICILIAN_5E5, "After 6.Ndb5 a6 7.Nd6+ you lose the bishop.").ok
    result = check(packet, SICILIAN_5E5, "After 6.Ndb5 h5 7.Nd6+ you lose the bishop.")
    assert not result.ok and any("h5" in p for p in result.problems)
    assert check(packet, SICILIAN_5E5, "After 6.Ndb5, the d6 square is the problem.").ok  # a square in words


def test_moves_are_compared_as_moves_not_as_spelling(packet):
    assert check(packet, SICILIAN_5E5, "After 5...e5 comes 6.Nd4b5.").ok  # over-specified, but the same move
    result = check(packet, SICILIAN_5E5, "After e6e5 comes d4b5.")
    assert not result.ok and any("computer notation" in p for p in result.problems)


def test_signed_evaluations_keep_their_sign(packet):
    # the packet's evaluations are from your side: 5...e5 leaves you at -1.52
    assert check(packet, SICILIAN_5E5, "After 5...e5 Stockfish says −1.52 for you.").ok
    result = check(packet, SICILIAN_5E5, "After 5...e5 Stockfish says +1.52 for you.")
    assert not result.ok and any("+1.52" in p for p in result.problems)
    assert check(packet, SICILIAN_5E5, "After 5...e5 it is +1.5 for White.").ok  # the side is named
    assert check(packet, SICILIAN_5E5, "After 5...e5 you are 1.5 pawns down.").ok  # a size, no sign
    # "−16 per 100 games" is the Sicilian claim's number; "+16" is not
    assert check(packet, SICILIAN_5E5, "The Sicilian Defense scores −16 per 100 games for you.", [OPENING_ID]).ok
    assert not check(packet, SICILIAN_5E5, "The Sicilian Defense scores +16 per 100 games for you.", [OPENING_ID]).ok


def test_material_won_or_lost_must_be_captured_in_a_line(packet):
    assert check(packet, SICILIAN_5E5, "After 5...e5 the knight check on d6 takes your dark-squared bishop.").ok
    result = check(packet, SICILIAN_5E5, "After 5...e5 6.Ndb5 wins your queen.")
    assert not result.ok and any("wins your queen" in p for p in result.problems)
    assert not check(packet, SICILIAN_5E5, "After 5...e5 White wins your knight.").ok  # only White's knight is taken
    assert check(packet, SICILIAN_5E5, "After 5...e5 you lose the bishop pair.").ok  # a pair is not a capture
    assert check(packet, QGA_4E4, "4.e4 drops a pawn to 4...Qxd4.").ok


def test_links_must_match_a_packet_link_whole(packet):
    assert check(packet, SICILIAN_5E5, "Replay https://www.chess.com/game/live/1001 first.").ok
    assert not check(packet, SICILIAN_5E5, "Replay https://www.chess.com/game/live/100 first.").ok  # a prefix
    assert not check(packet, SICILIAN_5E5, "See https://lichess.org for more.").ok


def test_move_n_must_be_a_move_of_the_lines(packet):
    assert check(packet, SICILIAN_5E5, "On move 5 you played 5...e5.").ok
    assert check(packet, SICILIAN_5E5, "By move 8 the bishop is gone.").ok  # 8.Qxd6 is in the refutation
    assert not check(packet, SICILIAN_5E5, "On move 12 you played 5...e5.").ok


def test_formats_are_checked_per_sentence_and_rapid_is_not_always_a_format(packet):
    assert check(packet, SICILIAN_5E5, "5...e5 slows your rapid development.").ok  # not the rapid format
    assert not check(packet, SICILIAN_5E5, "In rapid you would find 5...a6.").ok  # a blitz game
    # a format may come with your own number for it
    assert check(packet, SICILIAN_5E5, "At your 1130 rapid rating you would find 5...a6.").ok
    assert not check(packet, SICILIAN_5E5, "At your 1200 rapid rating you would find 5...a6.").ok


def test_habit_wording_needs_a_claim(packet):
    assert check(packet, SICILIAN_5E5, "Your habit of 5...e5 in the Open Sicilian costs a pawn's worth.").ok
    result = check(packet, SICILIAN_2NC6, "Your habit of 2...Nc6 lets 3.d5 chase the knight.")
    assert not result.ok and any("habit" in p or "weakness" in p for p in result.problems)


def test_a_claims_category_does_not_make_it_about_everything(packet):
    # the rapid strength is in the results category; it does not make all your results a strength
    assert not check(packet, QGA_4E4, "Your results are your strength.", [RAPID_ID]).ok
    assert check(packet, QGA_4E4, "Rapid is your strength: 4.Nf3 keeps d4 covered.", [RAPID_ID]).ok


def test_answers_name_only_formats_the_question_or_the_facts_are_about(packet):
    text = "You keep playing 5...e5 in the Open Sicilian: 10 of 21 games."
    assert verify_answer(text, packet, claim_ids=[REPEATED_ID]).ok
    in_bullet = "In bullet you keep playing 5...e5 in the Open Sicilian: 10 of 21 games."
    assert not verify_answer(in_bullet, packet, claim_ids=[REPEATED_ID]).ok  # that claim is about blitz games
    assert verify_answer(in_bullet.replace("bullet", "blitz"), packet, claim_ids=[REPEATED_ID]).ok
    assert verify_answer("The engine looked at 100 bullet games.", packet, question="What about bullet?").ok
    assert verify_answer("The engine looked at 100 bullet games.", packet).ok  # your own number for bullet


def test_mate_lines_give_the_mate_not_a_pawn_count():
    from test_packet import coaching_report as make

    report = make()
    exp = report.coaching.explanations[0]
    exp.refutation.cp_end, exp.refutation.mate_end = -1000, -3
    packet = build_packet(report)
    p = position_index(packet)[epd(SICILIAN_5E5)]
    assert p["refutation"]["mate_in"] == -3 and "eval_pawns" not in p["refutation"]
    assert "pawns_lost" not in p  # a mate is not "10 pawns"
    assert -10.0 not in packet_numbers(p["refutation"])  # nothing to write "10 pawns down" from


def test_no_practice_time_means_no_weekly_plan(packet):
    items = [p["title"] for p in packet["study_plan"]]
    entry = {"item": items[0], "minutes_per_week": 30, "recheck_date": "2026-10-24"}
    plan, dropped = verify_plan([entry], packet, practice_minutes=0, today=date(2026, 9, 26))
    assert plan == [] and len(dropped) == 1


def test_an_evaluation_in_words_must_point_the_engines_way(packet):
    # 5...e5 leaves you at -1.52, 5...a6 at -0.53: 0.99 pawns between them
    assert not check(packet, SICILIAN_5E5, "After 5...e5 you stand 1.5 pawns better.").ok
    assert check(packet, SICILIAN_5E5, "After 5...e5 you stand 1.5 pawns worse.").ok
    assert check(packet, SICILIAN_5E5, "After 5...e5 you are a pawn worse off than after 5...a6.").ok
    assert check(packet, SICILIAN_5E5, "With 5...a6 you would be 1 pawn better than after 5...e5.").ok
    assert check(packet, SICILIAN_5E5, "You end up 0.31 pawns worse on piece activity.").ok  # a concept difference
    assert not check(packet, SICILIAN_5E5, "You end up 0.31 pawns better on piece activity.").ok
    assert check(packet, SICILIAN_5E5, "After 5...e5 White is 1.5 pawns better.").ok  # a side named: not checked


def test_saying_which_side_you_play_does_not_excuse_a_flipped_sign(packet):
    assert not check(packet, SICILIAN_5E5, "As Black, after 5...e5 you are +1.52.").ok
    assert not check(packet, SICILIAN_5E5, "As Black, after 5...e5 you stand 1.5 pawns better.").ok
    assert check(packet, SICILIAN_5E5, "After 5...e5 White's advantage is +1.5.").ok


def _real_position(fen, best, best_eval, refutation, ref_eval, color, text, time_class="blitz"):
    """A position with lines as Stockfish 16 gave them (depth 12) and coach-core's template wording."""
    from chess_insights.coach.packet import _position

    def line(moves, value):
        cp, mate = value if isinstance(value, tuple) else (value, None)
        return {"fen": fen, "moves_uci": moves, "cp_end": cp, "mate_end": mate}

    exp = {"fen": fen, "epd": epd(fen), "played": "", "best_line": line(best, best_eval),
           "refutation": line(refutation, ref_eval), "color": color, "time_class": time_class, "text": text}
    p = _position(exp, set())
    p["played"] = p["refutation"]["moves"][0]
    p["best"] = p["best_line"]["moves"][0]
    return p, {"player": {}, "claims": [], "positions": [p]}


def test_real_engine_lines_and_template_wording_pass():
    # a back-rank mate: 24.Qxf4? Qd1+ 25.Rxd1 Rxd1#
    fen = "3r2k1/1b3p1p/p3p1pn/1p2P1N1/1P1qNn2/P4Q2/B4PPP/2R3K1 w - - 1 24"
    text = ("After 24.Qxf4, Black mates in 2: 24...Qd1+ 25.Rxd1 Rxd1#. Stockfish rates the line mate in 2 for Black, "
            "against −0.95 after 24.h3. Next time, look at every check your opponent has after your move.")
    p, pk = _real_position(fen, ["h2h3", "d4e5", "c1c5", "e5b8"], -95, ["f3f4", "d4d1", "c1d1", "d8d1"],
                           (-1000, -2), "white", text)
    assert verify_explanation(text, p, pk).ok
    llm_text = ("24.Qxf4 walks into a back-rank mate: 24...Qd1+ 25.Rxd1 Rxd1#. 24.h3 first gives your king air and "
                "keeps you at about −1.0. Before taking material, check every check your opponent has.")
    assert verify_explanation(llm_text, p, pk).ok
    assert not verify_explanation("After 24.Qxf4 Black mates in 3.", p, pk).ok
    assert not verify_explanation("24.Qxf4 loses your queen to 24...Qd1+.", p, pk).ok  # nothing takes it

    # a hanging bishop: 20...Kg7? 21.Qxd3, where 20...Qf7+ mates in 5
    fen = "r4rk1/ppp4p/3p4/3P2pK/5qN1/3b3P/PPP3P1/R2Q3R b - - 1 20"
    text = ("After 20...Kg7, White has an undefended piece to take: 21.Qxd3 Qf7+ 22.Kxg5 Qe7+. Stockfish rates the "
            "line −4.00 for you, against mate in 5 for you after 20...Qf7+. Next time, count the attackers and "
            "defenders of each piece before you move.")
    p, pk = _real_position(fen, ["f4f7", "h5g5", "f7e7", "g4f6", "e7f6"], (1000, 5),
                           ["g8g7", "d1d3", "f4f7", "h5g5", "f7e7", "g5h5"], -400, "black", text)
    assert verify_explanation(text, p, pk).ok
    assert verify_explanation("After 20...Kg7, 21.Qxd3 takes your bishop; 20...Qf7+ mates in 5.", p, pk).ok
    assert not verify_explanation("After 20...Kg7, 21.Qxd3 takes your knight.", p, pk).ok
    assert not verify_explanation("After 20...Kg7 you are +4.0 for you.", p, pk).ok
