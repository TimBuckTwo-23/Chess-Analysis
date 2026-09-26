"""The report on a phone: the page's weight, one board per position, what a view says about its format, deeper
checks that clear a move, observations that were never tested, and the smaller rendering rules behind them."""

from __future__ import annotations

import copy
import re

from render_sample import QGA, SICILIAN, sample_report
from test_report import check_html

from chess_insights import visuals
from chess_insights.models import Arrow, Chart, Coaching, Diagram, Insight, ModuleResult, Series, Table
from chess_insights.report import render_html, render_markdown
from chess_insights.report import html as H


def _view(html: str, key: str) -> str:
    m = re.search(rf'<section class="fmt-view fmt-view--{key}".*?(?=<section class="fmt-view |</main>)', html, re.S)
    assert m, key
    return m.group(0)


def _section(html: str, sid: str) -> str:
    return re.search(rf'<section class="section[^"]*" id="{sid}".*?</section>', html, re.S).group(0)


def _card(html: str, anchor: str) -> str:
    return re.search(rf'<article class="card[^"]*" id="{anchor}">.*?</article>', html, re.S).group(0)


# --------------------------------------------------------------------------- the deeper check clears a move
def _cleared(verdict: str, *, best_is_played: bool) -> "tuple":
    rep = sample_report(views=False)
    exp = rep.coaching.explanations[0]  # 5...e5, the finding positions.repeated.sicilian-5e5 links to it
    exp.verdict = verdict
    if best_is_played:
        exp.best = exp.played
    return rep, exp


def test_a_move_the_deeper_check_finds_fine_is_not_called_a_mistake():
    rep, exp = _cleared("fine", best_is_played=True)
    html = render_html(rep)
    check_html(html)
    card = _card(html, "why-1")
    assert ">Deeper check: not a mistake at depth 20 · Blitz<" in card  # the lines' depth
    assert "winning chances lost" not in card and "Your mistake" not in card and "Repeated mistake" not in card
    assert "you played it in 10 games" in card
    assert "Better:" not in card and "better 5...e5" not in card  # no "better" line naming your own move
    assert "What 5...e5 allows" in card
    # the finding and the section's board link to it by what it says
    finding = re.search(r'<article class="card insight[^>]*id="f-positions-repeated-sicilian-5e5".*?</article>', html,
                        re.S).group(0)
    assert ">What a deeper check says</a>" in finding and "Why this goes wrong" not in finding
    assert "quick check&#x27;s choice a6" in finding  # its green arrow is only the quicker analysis's choice
    md = render_markdown(rep)
    block = md.split("### 1. ")[1].split("### 2. ")[0]
    assert block.startswith("5...e5\n")
    assert "_Deeper check: not a mistake at depth 20 · Blitz · you played it in 10 games_" in block
    assert "winning chances lost" not in block and "Better" not in block
    assert "what a deeper check says: explanation 1 below" in md and "quick check's choice a6 (green)" in md


def test_a_close_move_names_the_engine_choice_without_calling_it_better():
    rep, exp = _cleared("close", best_is_played=False)
    html = render_html(rep)
    card = _card(html, "why-1")
    assert ">Deeper check: not a mistake at depth 20 · Blitz<" in card and "winning chances lost" not in card
    assert '<span class="why-alt">first choice 5...a6</span>' in card and "why-best" not in card
    assert "<dt>Stockfish&#x27;s first choice: 5...a6</dt>" in card and "Better:" not in card
    md = render_markdown(rep)
    assert "### 1. 5...e5, Stockfish's first choice 5...a6" in md and "- Stockfish's first choice: 5...a6: `" in md
    # an error keeps its kicker, its cost and its better line
    rep.coaching.explanations[0].verdict = "error"
    card = _card(render_html(rep), "why-1")
    assert ">Repeated mistake · Blitz<" in card and "21% winning chances lost" in card and "<dt>Better: 5...a6</dt>" in card


def test_kicker_helpers_read_any_verdict_safely():
    rep, exp = _cleared("fine", best_is_played=True)
    exp.best_line = exp.refutation = None
    assert H.explanation_kicker(exp) == ["Deeper check: not a mistake", "Blitz"]  # no depth known
    assert H.explanation_kicker(exp, "blitz") == ["Deeper check: not a mistake"]
    for odd in ("", None, "FINE?", 3):
        exp.verdict = odd
        assert H.explanation_verdict(exp) == "error" and H.explanation_kicker(exp, "blitz") == ["Repeated mistake"]
    assert H.best_line_label(exp) == ""  # best == played


# --------------------------------------------------------------------------- Lichess from your side
def test_analyse_on_lichess_opens_from_your_side():
    assert H.lichess_analysis_url(SICILIAN, "black").endswith("_b_KQkq_-_0_5?color=black")
    assert H.lichess_analysis_url(QGA, "white") == "https://lichess.org/analysis/" + QGA.replace(" ", "_")
    rep = sample_report(views=False)
    html, md = render_html(rep), render_markdown(rep)
    black = "https://lichess.org/analysis/" + SICILIAN.replace(" ", "_") + "?color=black"
    assert f'href="{black}"' in html and f"({black})" in md
    assert "https://lichess.org/analysis/" + QGA.replace(" ", "_") + '"' in html  # White's boards need nothing
    # the table's position links still find their explanation
    from chess_insights.report import boards

    assert boards.fen_from_analysis_url(black) == SICILIAN


# --------------------------------------------------------------------------- observations are not tested
def test_observations_say_not_tested_instead_of_a_confidence():
    rep = sample_report(views=False)
    html = render_html(rep)
    note = re.search(r'<article class="card insight[^>]*id="f-results-observation-formats".*?</article>', html, re.S).group(0)
    assert '<span class="pill pill--obs">Observation · not tested</span>' in note and "confidence" not in note
    claim = re.search(r'<article class="card insight[^>]*id="f-color-strength-white".*?</article>', html, re.S).group(0)
    assert "High confidence" in claim and "not tested" not in claim
    md = render_markdown(rep)
    assert "**Observation · Overall results: Most of your games are rapid** — 41% of your games are rapid. " \
           "_(Observation · not tested · Bullet 652" in md
    assert "**Strength · Colour balance: You score well with White** — +6 per 100 games vs your rating. " \
           "_(High confidence · Bullet 652" in md


# --------------------------------------------------------------------------- the engine review's module key
def test_coaching_follows_the_engine_review_under_its_real_key():
    for key in ("engine", "engine_stats"):
        rep = sample_report(views=False)
        rep.modules = [m for m in rep.modules if m.key != "mistakes"]
        engine = next(m for m in rep.modules if m.key == "engine_stats")
        engine.key = key
        html, md = render_html(rep), render_markdown(rep)
        order = re.findall(r'<section class="section[^"]*" id="([\w-]+)"', html)
        assert order[order.index(key) + 1] == "why", order
        assert md.index("## Engine review") < md.index("## Why these moves go wrong") < md.index("## Practice")
    assert H._coaching_at([ModuleResult("results", "R", ""), ModuleResult("engine", "E", ""),
                           ModuleResult("habits", "H", "")]) == 2


# --------------------------------------------------------------------------- one board per position per view
def test_a_board_drawn_higher_up_is_linked_not_drawn_again():
    rep = sample_report(views=False)
    engine = next(m for m in rep.modules if m.key == "engine_stats")
    twin = copy.deepcopy(next(m for m in rep.modules if m.key == "mistakes").diagrams[1])  # 4.e4, arrows and all
    engine.diagrams = [twin]
    html = render_html(rep)
    check_html(html)
    mistakes = _section(html, "mistakes")
    assert _section(html, "engine_stats").count('<svg class="cb"') == 1
    ref = re.search(r'<figure class="board-fig board-fig--ref" id="mistakes-b\d+">.*?</figure>', mistakes, re.S).group(0)
    assert "Queen&#x27;s Gambit Accepted: 4.e4" in ref and "You played 4.e4 in 6 of 7 games" in ref  # caption kept
    assert re.search(r'<a class="ref-link" href="#engine_stats-b1">Board shown above, in Engine review</a>', ref)
    assert '<svg class="cb' not in ref and "What 4.e4 allows" not in mistakes  # nor its line of play again
    # another picture of the same position (other arrows) is drawn
    engine.diagrams[0].arrows = [Arrow("e2", "e4", "played")]
    html = render_html(rep)
    assert "board-fig--ref" not in _section(html, "mistakes") and "What 4.e4 allows" in html


def test_each_view_draws_its_own_boards():
    rep = sample_report()
    html = render_html(rep)
    check_html(html)
    for key in ("all", "blitz"):  # a board is never replaced by a link into another view
        view = _view(html, key)
        for m in re.finditer(r'class="ref-link" href="#([^"]+)"', view):
            assert f'id="{m.group(1)}"' in view


# --------------------------------------------------------------------------- board groups on a phone
def _module_with_boards(n: int, heading: str = "") -> ModuleResult:
    boards_ = []
    for k in range(n):
        d = visuals.position_diagram(f"Puzzle {k + 1}: Sicilian" if heading else f"Position {k + 1}", SICILIAN,
                                     orientation="black", played="e5", best="a6", time_class="blitz")
        d.arrows.append(Arrow(f"{'abcdefgh'[k]}2", f"{'abcdefgh'[k]}3", "neutral"))  # each its own picture
        d.strips = [visuals.line_strip("What happened in the game", SICILIAN, ["e6e5", "d4b5"])]
        boards_.append(d)
    return ModuleResult("mistakes", "Positions you keep getting wrong", "s", diagrams=boards_)


def test_a_group_shows_four_boards_and_folds_the_rest_without_their_lines():
    rep = sample_report(views=False, coaching=False)
    rep.modules = [_module_with_boards(7)]
    html = render_html(rep)
    check_html(html)
    section = _section(html, "mistakes")
    fold = re.search(r'<details class="more boards-more"><summary>Show 3 more positions</summary>.*?</details>', section, re.S)
    assert fold and fold.group(0).count('<svg class="cb"') == 3 and "cb--mini" not in fold.group(0)
    assert section.count('<svg class="cb"') == 7 and section.count("cb cb--mini") == 4 * 2  # lines for the open four
    # the puzzles' "what happened in the game" lines fold under their titles
    rep.modules = [_module_with_boards(5, heading="puzzles")]
    rep.modules[0].tables = [Table("Your costliest mistakes (puzzles)", ["a"], [["x"]])]
    section = _section(render_html(rep), "mistakes")
    assert section.count('<details class="strip strip--fold"><summary class="strip-title">What happened in the game'
                         '</summary>') == 4
    assert "Show 1 more position<" in section


def test_boards_in_a_multi_format_finding_say_their_format():
    rep = sample_report(views=False)
    html = render_html(rep)
    habit = re.search(r'<article class="card insight[^>]*id="f-positions-repeated-sicilian-5e5".*?</article>', html,
                      re.S).group(0)
    assert '<span class="fmt-chip">Blitz game</span>' in habit  # the card's chip says Blitz 14 · Rapid 7
    rep.weaknesses[1].formats = {"blitz": 14}
    habit = re.search(r'<article class="card insight[^>]*id="f-positions-repeated-sicilian-5e5".*?</article>',
                      render_html(rep), re.S).group(0)
    assert "Blitz game" not in habit and "Blitz only" in habit  # the chip says it already


# --------------------------------------------------------------------------- format views say their format
def test_views_name_their_format_in_every_section_head():
    html = render_html(sample_report())
    blitz = _view(html, "blitz")
    heads = re.findall(r"<h2 [^>]*>(.*?)</h2>", blitz, re.S)
    assert heads and all("Blitz only" in h for h in heads), heads
    assert "Blitz only" not in "".join(re.findall(r"<h2 [^>]*>(.*?)</h2>", _view(html, "all"), re.S))
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    assert ".fmt-bar{position:sticky;top:0" in style and ".fmt [id]{scroll-margin-top:" in style


def test_a_line_under_the_tabs_says_which_formats_have_no_view():
    rep = sample_report()
    assert '<p class="fmt-missing">' not in render_html(rep)  # every format has its view
    rep.formats = {**rep.formats, "daily": 8}
    html = render_html(rep)
    line = re.search(r'<p class="fmt-missing">(.*?)</p>', html).group(1)
    assert line == "Daily (8 games) has too few games for a view of its own (a view needs 60): its games count in All."
    assert html.index('<div class="fmt-bar">') < html.index('<p class="fmt-missing">') < html.index('"fmt-view fmt-view--all"')
    rep.formats = {**rep.formats, "bullet": 39}
    del rep.format_reports["bullet"]
    assert ("Bullet (39 games) and daily (8 games) have too few games for views of their own (a view needs 60): "
            "their games count in All.") in render_html(rep)
    # no views at all: said above the sections
    rep.format_reports = {}
    rep.formats = {"blitz": 40, "rapid": 12}
    html = render_html(rep)
    assert "Blitz (40 games) and rapid (12 games) have too few games" in html and '<div class="fmt-bar">' not in html


# --------------------------------------------------------------------------- the weight of the page
def test_the_page_writes_at_most_twelve_explanations_and_says_where_the_rest_are():
    rep = sample_report()
    extra = [copy.deepcopy(e) for e in rep.coaching.explanations[4:]] * 12
    rep.coaching.explanations += extra
    n = len(rep.coaching.explanations)
    html = render_html(rep)
    check_html(html)
    why = _section(_view(html, "all"), "why")
    assert why.count('<article class="card why"') == H.WHY_HTML_MAX == 12
    assert f"{n - 12} more explanations are left out of this page to keep it light on a phone. All {n} are in the " \
           "Markdown version of this report" in why
    assert len(re.findall(r'<details class="more why-more"><summary>Show 7 more', why)) == 1  # five open
    for key in ("bullet", "blitz", "rapid"):  # a view repeats three at most
        assert _view(html, key).count('<article class="card why"') <= H.WHY_VIEW_MAX == 3
    md = render_markdown(rep)
    assert f"### {n}. " in md  # the Markdown keeps every one
    # the explanations findings link to are written first, whatever their place in the list
    rep.coaching.explanations = extra + rep.coaching.explanations[:4]
    html = _view(render_html(rep), "all")
    k = len(extra) + 1  # 5...e5, which the habit finding links to
    assert f'id="why-{k}"' in html and f'href="#why-{k}">Why this goes wrong</a>' in html


def test_a_realistic_report_stays_light(tmp_path):
    rep = sample_report()
    base = rep.coaching.explanations
    rep.coaching.explanations = [copy.deepcopy(base[i % len(base)]) for i in range(150)]  # the workflow's --coach-max
    size = len(render_html(rep).encode())
    rep.coaching.explanations = base[:12]
    assert size < len(render_html(rep).encode()) * 1.1  # 138 more explanations add (almost) nothing to the page


# --------------------------------------------------------------------------- tables and charts shown once
def _with_coaching_tables(rep):
    chart, table = rep.coaching.motif_chart, rep.coaching.motif_profile
    engine = next(m for m in rep.modules if m.key == "engine_stats")
    engine.charts.append(chart)  # as coach.profile.annotate adds them
    engine.tables.append(table)
    openings = next(m for m in rep.modules if m.key == "openings")
    theory = rep.coaching.theory_exit
    openings.charts.append(Chart("hbar", theory.title, ["a", "b"], [Series("x", [1.0, 2.0])], table=theory))
    return rep


def test_the_motif_profile_and_theory_exit_are_shown_once():
    rep = _with_coaching_tables(sample_report(views=False))
    html = render_html(rep)
    check_html(html)
    engine, why = _section(html, "engine_stats"), _section(html, "why")
    assert "Tactical patterns in the mistakes" not in engine and "Patterns you miss vs your opponents" not in engine
    assert 'The tactical patterns in your mistakes are under <a href="#why-patterns">Why these moves go wrong</a>' in engine
    assert html.count(">Patterns you miss vs your opponents (per 100 moves)<") == 1
    assert html.count('aria-label="Tactical patterns in the mistakes (per 100 moves)"') == 1
    patterns = re.search(r'<div class="coach-block" id="why-patterns">.*?</div></div>', why, re.S).group(0)
    numbers = re.search(r'<details class="chart-data"><summary>Show the numbers</summary>.*?</details>', patterns, re.S)
    assert numbers and "Tactical patterns in the mistakes" in numbers.group(0)  # the table sits under the chart
    assert html.count('<th scope="col" class="num">You missed</th>') == 1
    # where you leave theory: charted in Openings, a line in Why
    assert html.count("First inaccuracy") == 1 and 'Charted in <a href="#openings">Openings</a>' in why
    md = render_markdown(rep)
    assert md.count("| Pattern | You missed |") == 1 and md.count("First inaccuracy") == 1
    assert "_The tactical patterns in your mistakes are under “Why these moves go wrong”._" in md
    assert "_Charted in Openings, with the numbers._" in md


def test_numeric_tables_are_marked_for_a_narrow_first_column():
    rep = sample_report(views=False, coaching=False)
    html = render_html(rep)
    assert '<table class="t-num">' in html  # Score vs expected by format: a label and numbers
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    assert ".ci .t-num td:first-child:not(.num){min-width:8ch}" in style
    assert ".frames{list-style:none;display:grid;grid-template-columns:repeat(4,minmax(0,110px))" in style


# --------------------------------------------------------------------------- review boards you can solve
def test_review_boards_are_big_enough_to_solve():
    html = render_html(sample_report(views=False))
    practice = _section(html, "practice")
    due = re.search(r"Due for review</h3>.*?</ul>", practice, re.S).group(0)
    assert due.count('<li class="review-item review-item--solve">') == 2 and "zoom" not in due
    upcoming = re.search(r"Coming up for review</h3>.*", practice, re.S).group(0)
    assert upcoming.count('<label class="zoom" title="Tap to enlarge"><input type="checkbox" class="zoom-box"') == 10
    assert 'class="zoom-box" aria-label="Enlarge the board" data-k' not in upcoming  # never remembered
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    assert ".review-item--solve .cb,.review-item:has(.zoom-box:checked) .cb{max-width:240px}" in style


# --------------------------------------------------------------------------- choice boards keep green for the engine
def test_a_choice_board_key_names_the_best_scoring_move():
    d = Diagram("Your choice after 1.e4 e5 2.Nf3 Nc6 (White)", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                arrows=[Arrow("f1", "b5", "played"), Arrow("f1", "c4", "line")])
    key = H._arrow_key_html(d)
    assert "your move Bb5" in key and "best for you so far Bc4" in key
    other = Diagram("Main line", d.fen, arrows=[Arrow("f1", "c4", "line")])
    assert ">Bc4<" in H._arrow_key_html(other)  # a plain line of play keeps its plain key
    note = next(g for g in H.BOARD_GROUPS if g[1] == "Your choices at key moves")[3]
    assert "blue marks the move that has scored best for you so far" in note


def test_insight_with_a_diagram_but_no_formats_renders():
    rep = sample_report(views=False, coaching=False)
    ins = Insight("x.observation.y", "observation", "results", "t", "d", 0.1, 0.5,
                  diagram=visuals.position_diagram("P", SICILIAN, orientation="black", played="e5", time_class="blitz"))
    rep.modules[0].insights.append(ins)
    check_html(render_html(rep))
