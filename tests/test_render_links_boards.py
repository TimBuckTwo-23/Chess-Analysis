"""Game labels on every game link, one board per position in a section (the costliest puzzles and your choices
as their own groups), and the AI coach's note: once, and never its rejected texts."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone

from test_report import check_html

from chess_insights import visuals
from chess_insights.models import Coaching, Diagram, Explanation, Insight, ModuleResult, Report, Table
from chess_insights.report import render_html, render_markdown, to_json
from chess_insights.report.html import module_board_groups, same_board

SICILIAN = "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5"
QGA = "r1bqkbnr/ppp1pppp/2n5/8/2p5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 4"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
GAME = "https://www.chess.com/game/live/{}"
LABEL = "Loss · 5+0 · vs 1512 · 12 Aug"
REJECTED = "5.Qxh7 wins the queen and the game is over."  # a text the fact check threw out


def board(title: str, fen: str = SICILIAN, played: str = "e5", best: str = "a6", **kw) -> Diagram:
    return visuals.position_diagram(title, fen, orientation="black" if " b " in fen else "white", played=played,
                                    best=best, **kw)


def finding(diagram: Diagram, iid: str = "positions.repeated.sicilian-5e5") -> Insight:
    return Insight(iid, "observation", "positions", "You keep playing 5...e5 in the Open Sicilian",
                   "10 of 21 games.", 0.3, 0.5, diagram=diagram)


def report_of(*modules: ModuleResult, coaching: Coaching | None = None, **kw) -> Report:
    return Report(username="Tester", generated_at=datetime(2026, 9, 25, 14, 3, tzinfo=timezone.utc), filters="",
                  n_games=100, date_from=None, date_to=None, modules=list(modules), strengths=[], weaknesses=[],
                  study_plan=[], coaching=coaching, **kw)


def section(html: str, key: str) -> str:
    return re.search(rf'<section class="section module" id="{key}".*?</section>', html, re.S).group(0)


def mistakes_module() -> ModuleResult:
    """As analysis/mistakes.py builds it: the finding reuses its repeated-mistake board (the same object), and one
    board per costliest puzzle follows, in the puzzles table's order."""
    repeated = board("Sicilian Defense: 5...e5", link=GAME.format(1))
    puzzles = [board("Puzzle 1: Queen's Gambit Accepted", QGA, "e4", "Nf3", link=GAME.format(2)),
               board("Puzzle 2: Sicilian Defense", SICILIAN, "e5", "a6", link=GAME.format(3))]
    table = Table("Your costliest mistakes (puzzles)", ["Opening", "Game"],
                  [["Queen's Gambit Accepted", GAME.format(2)], ["Sicilian Defense", GAME.format(3)]], ["text", "url"])
    return ModuleResult("mistakes", "Positions you keep getting wrong", "The same wrong move, game after game.",
                        insights=[finding(repeated)], tables=[table], diagrams=[repeated, *puzzles])


# --------------------------------------------------------------------------- one board per position
def test_same_board_is_the_same_object_or_the_same_position_and_arrows():
    a = board("A")
    assert same_board(a, a) and same_board(a, copy.deepcopy(a))
    assert same_board(a, board("Another title, same picture", caption="other words"))
    assert not same_board(a, board("A", best="Nf6"))  # another better move: another picture
    assert not same_board(a, board("A", QGA, "e4", "Nf3"))


def test_a_section_does_not_draw_its_findings_board_twice():
    module = mistakes_module()
    groups = module_board_groups(module)
    assert [(g.heading, [d.title for d in g.diagrams]) for g in groups] == [
        ("Your costliest mistakes as puzzles", ["Puzzle 1: Queen's Gambit Accepted", "Puzzle 2: Sicilian Defense"])]
    # a copy of the finding's board (same position, same arrows) counts as the same board; puzzle 2, in the
    # same position, stays: it is the table's row 2
    module.diagrams[0] = copy.deepcopy(module.diagrams[0])
    assert [(g.heading, len(g.diagrams)) for g in module_board_groups(module)] == [
        ("Your costliest mistakes as puzzles", 2)]
    html = section(render_html(report_of(module)), "mistakes")
    assert html.count('aria-label="Sicilian Defense: 5...e5') == 1  # in the finding's card only
    assert html.count('<figure class="board-fig board-fig--card">') == 1
    md = render_markdown(report_of(module))
    assert md.count("**Sicilian Defense: 5...e5**") == 1  # under the finding only


def test_the_costliest_puzzles_are_their_own_group_outside_the_fold_in_table_order():
    html = render_html(report_of(mistakes_module()))
    check_html(html)
    sec = section(html, "mistakes")
    fold = sec.index('<details class="more">')
    group = re.search(r'<div class="board-group"><h3 class="sub-head">Your costliest mistakes as puzzles '
                      r'<span class="count">2</span></h3>.*?</div></div>', sec, re.S)
    assert group and group.end() <= fold  # visible without opening "Show 1 table"
    assert sec.index(">Findings <") < group.start()
    titles = re.findall(r'<span class="block-title">([^<]+)</span>', group.group(0))
    assert titles == ["Puzzle 1: Queen&#x27;s Gambit Accepted", "Puzzle 2: Sicilian Defense"]
    # the Markdown puts them right after the puzzles table
    md = render_markdown(report_of(mistakes_module()))
    table_at = md.index("#### Your costliest mistakes (puzzles)")
    group_at = md.index("_Your costliest mistakes as puzzles:_")
    assert table_at < group_at < md.index("**Puzzle 1: Queen's Gambit Accepted**") < md.index("**Puzzle 2: Sicilian")
    assert md.index("**Puzzle 2: Sicilian") < md.index("### Findings")


def test_choice_boards_are_grouped_under_your_choices_at_key_moves():
    choices = [board("Your choice after 1.e4 c5 (White)", START, "e4", "d4"),
               board("Your choice at move 1 (White)", START, "d4", "e4")]
    other = board("After 1.e4 c5: 2.Nf3 is what masters choose", QGA, "e4", "Nf3")
    table = Table("Your choices at key moves", ["Position", "Your move"], [["1.e4 c5", "2.Nf3"]])
    module = ModuleResult("openings", "Openings", "s", tables=[table], diagrams=[choices[0], other, choices[1]])
    groups = module_board_groups(module)
    assert [(g.heading, len(g.diagrams)) for g in groups] == [("", 1), ("Your choices at key moves", 2)]
    sec = section(render_html(report_of(module)), "openings")
    assert ">Your choices at key moves <span class=\"count\">2</span></h3>" in sec
    assert sec.index("board-group") < sec.index('<details class="more">')
    md = render_markdown(report_of(module))
    assert md.index("**After 1.e4 c5: 2.Nf3") < md.index("#### Your choices at key moves") < md.index(
        "_Your choices at key moves:_") < md.index("**Your choice after 1.e4 c5 (White)**")


def test_a_group_whose_table_is_missing_gets_a_heading_in_the_markdown():
    module = mistakes_module()
    module.tables = []
    md = render_markdown(report_of(module))
    assert md.index("#### Your costliest mistakes as puzzles") < md.index("**Puzzle 1:")


# --------------------------------------------------------------------------- game labels on every game link
def test_every_game_link_shows_its_label():
    module = mistakes_module()
    labels = {GAME.format(k): f"{LABEL} ({k})" for k in (1, 2, 3)}
    html = render_html(report_of(module, game_labels=labels))
    sec = section(html, "mistakes")
    for k in (2, 3):  # a board's game link and a table's game cell
        assert sec.count(f">{LABEL} ({k})<span aria-hidden=\"true\"> ↗</span></a>") == 2
    assert "Open the game" not in sec
    md = render_markdown(report_of(module, game_labels=labels))
    assert f"[{LABEL} (2) ↗]({GAME.format(2)})" in md  # the table cell
    assert f"[{LABEL} (2)]({GAME.format(2)})" in md  # the board
    # a game the report has no label for keeps the plain words
    plain = render_html(report_of(mistakes_module()))
    assert "Open the game" in plain and 'game<span aria-hidden="true"> ↗</span>' in plain
    assert "[game ↗](" in render_markdown(report_of(mistakes_module()))


# --------------------------------------------------------------------------- the AI coach's note
def llm_coaching() -> Coaching:
    exp = Explanation(epd=" ".join(SICILIAN.split()[:4]), fen=SICILIAN, played="5...e5", best="5...a6",
                      best_line=None, refutation=None, text="Template text.", time_class="blitz")
    note = ("AI coach (claude-opus-5-5): 0 of the 1 explanation it was given reworded; 1 failed the fact check and "
            "keeps the built-in text.")
    return Coaching(explanations=[exp], notes=["Opening explorer skipped (offline).", note],
                    llm={"model": "claude-opus-5-5", "sent": 1, "accepted": 0, "rejected": 1, "ignored": 0,
                         "plan_dropped": 0, "rejections": [{"epd": exp.epd, "text": REJECTED,
                                                            "problems": ["5.Qxh7 is not in the lines"]}]})


def test_the_ai_coach_note_shows_once_and_never_a_rejected_text():
    coaching = llm_coaching()
    view = report_of(ModuleResult("mistakes", "Positions", "s"), coaching=copy.deepcopy(coaching),
                     time_class="blitz", formats={"blitz": 80})
    rep = report_of(ModuleResult("mistakes", "Positions", "s"), coaching=coaching,
                    formats={"blitz": 80, "rapid": 70}, format_reports={"blitz": view})
    html, md = render_html(rep), render_markdown(rep)
    for text in (html, md):
        assert text.count("AI coach") == 1  # the main view's notes only; the blitz view does not repeat it
        assert "Opening explorer skipped (offline)." in text
        assert REJECTED not in text and "5.Qxh7" not in text and "not in the lines" not in text
    assert REJECTED in to_json(rep) and json.loads(to_json(rep))["coaching"]["llm"]["rejections"]


def test_the_ai_coach_counts_are_shown_when_no_note_says_them():
    coaching = llm_coaching()
    coaching.notes = []  # a report whose coaching kept the counts only
    html = render_html(report_of(ModuleResult("mistakes", "Positions", "s"), coaching=coaching))
    assert html.count("AI coach") == 1
    assert ("AI coach (claude-opus-5-5): 0 of the 1 explanation it was given reworded; 1 failed the fact check and "
            "keeps the built-in text.") in html
    assert REJECTED not in html


def test_the_markdown_shows_the_endgame_boards_with_their_game_label():
    rook = board("Rook ending: 60.Kc4", "8/8/8/3k4/8/2K5/r7/1R6 w - - 0 60", "Kc4", "Rb5", link=GAME.format(9))
    rook.fen = "8/8/8/3k4/8/2K5/r7/1R6 w - - 0 60"
    coaching = Coaching(endgames=Table("Endings", ["Ending", "Game"], [["Rook", GAME.format(9)]], ["text", "url"]),
                        endgame_diagrams=[rook])
    md = render_markdown(report_of(ModuleResult("engine_stats", "Engine review", "s"), coaching=coaching,
                                   game_labels={GAME.format(9): LABEL}))
    part = md[md.index("### Endgames checked with the tablebase"):]
    assert part.index("| Rook |") < part.index("**Rook ending: 60.Kc4**")
    assert f"[{LABEL}]({GAME.format(9)})" in part and f"[{LABEL} ↗]({GAME.format(9)})" in part
    html = render_html(report_of(ModuleResult("engine_stats", "Engine review", "s"), coaching=coaching,
                                 game_labels={GAME.format(9): LABEL}))
    assert html.count(f">{LABEL}<span aria-hidden=\"true\"> ↗</span></a>") == 2  # the board and the table row
