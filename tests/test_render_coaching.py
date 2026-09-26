"""Per-finding visuals, format views and the coaching sections in the HTML, Markdown and JSON reports."""

from __future__ import annotations

import copy
import json
import re

import pytest
from render_sample import QGA, ROOK_END, SICILIAN, sample_coaching, sample_report
from test_report import check_html, empty_report

from chess_insights.models import (
    Arrow,
    Chart,
    Coaching,
    Diagram,
    Drill,
    DrillPuzzle,
    Explanation,
    Insight,
    Frame,
    ReviewItem,
    Strip,
)
from chess_insights.report import render_html, render_markdown, to_json
from chess_insights.report.html import format_chip_text

SPRITE = 'class="ci-sprite"'


@pytest.fixture(scope="module")
def report():
    return sample_report()


@pytest.fixture(scope="module")
def page(report):
    return render_html(report)


def _view(html: str, key: str) -> str:
    """The HTML of one format view."""
    m = re.search(rf'<section class="fmt-view fmt-view--{key}".*?(?=<section class="fmt-view |</main>)', html, re.S)
    assert m, key
    return m.group(0)


# --------------------------------------------------------------------------- the page as a whole
def test_standalone_and_fragment_share_one_sprite(report, page):
    frag = render_html(report, standalone=False)
    for html in (page, frag):
        checker = check_html(html)
        assert html.count(SPRITE) == 1  # one piece sprite for every board, in every view
        assert len(checker.ids) == len(set(checker.ids)), [i for i in checker.ids if checker.ids.count(i) > 1][:5]
        internal = {h[1:] for t, h in checker.hrefs if h.startswith("#")}
        assert internal <= set(checker.ids)  # every in-page link has its target ...
        uses = set(re.findall(r'<use href="#([^"]+)"', html))
        assert uses and uses <= set(checker.ids)  # ... and every board piece its sprite entry
    lowered = frag.lower()
    for tag in ("!doctype", "html", "head", "body"):
        assert not re.search(rf"<{tag}[\s>]", lowered) and f"</{tag}>" not in lowered, tag
    # the sprite sits outside the views, so a hidden view never hides it
    assert page.index(SPRITE) < page.index('class="fmt-bar"')
    assert page.count('<svg class="cb') == frag.count('<svg class="cb') > 30


def test_format_switcher_is_radio_tabs_without_script(page):
    radios = re.findall(r'<input class="fmt-radio" type="radio" name="ci-fmt" id="fmt-(\w+)"( checked)?>', page)
    assert [k for k, _ in radios] == ["all", "bullet", "blitz", "rapid"]
    assert [bool(c) for _, c in radios] == [True, False, False, False]
    assert re.findall(r'<label for="fmt-(\w+)">', page) == ["all", "bullet", "blitz", "rapid"]
    assert "All<span class=\"fmt-long\"> formats</span>" in page
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    for k in ("all", "bullet", "blitz", "rapid"):
        assert f"#fmt-{k}:checked~.fmt-view--{k}{{display:block}}" in style
        assert f'#fmt-{k}:focus-visible~.fmt-bar label[for="fmt-{k}"]' in style
    assert ".fmt-view{display:none}" in style
    assert page.count("<main") == 1  # views are sections inside one <main>


def test_each_view_says_what_it_covers_and_links_inside_itself(report, page):
    assert "All formats: 2,731 games" in _view(page, "all")
    for tc, n in (("bullet", "652"), ("blitz", "949"), ("rapid", "1,130")):
        view = _view(page, tc)
        assert f"{tc.title()} only: {n} games" in view
        assert f"on {report.format_reports[tc].engine_formats[tc]} {tc} games" in view  # its own engine note
        hrefs = re.findall(r'<a [^>]*href="#([^"]+)"', view)  # links (a board's <use> points into the sprite)
        assert hrefs and all(h.startswith(f"v-{tc}-") for h in hrefs), [h for h in hrefs if not h.startswith("v-")][:3]
        ids = re.findall(r' id="([^"]+)"', view)
        assert ids and all(i.startswith(f"v-{tc}-") for i in ids)
    main = _view(page, "all")
    assert 'href="#weaknesses"' in main and 'id="study"' in main  # the main view keeps the plain ids
    assert "Engine-analysed games:" in main and "Bullet 100" in main


def test_a_report_without_views_keeps_the_old_layout():
    rep = sample_report(views=False)
    html = render_html(rep)
    check_html(html)
    assert '<input class="fmt-radio"' not in html and '<main class="sections">' in html
    assert html.count(SPRITE) == 1
    assert "Bullet 652" in html  # the masthead lists the formats and the engine sample
    assert "Stockfish 16 at depth 12 on 300 games" in html.split("</header>")[0]


# --------------------------------------------------------------------------- findings
def test_every_finding_card_shows_its_chart_board_and_formats(page):
    main = _view(page, "all")
    card = re.search(r'<article class="card insight[^>]*id="f-openings-weakness-black-sicilian-defense".*?</article>', main, re.S).group(0)
    assert 'class="ci-chart ci-chart--card"' in card and 'viewBox="0 0 300 ' in card  # compact, readable chart
    assert re.search(r"Bullet 65.*Blitz 94.*Rapid 113", re.sub(r"<[^>]+>", "", card))
    habit = re.search(r'<article class="card insight[^>]*id="f-positions-repeated-sicilian-5e5".*?</article>', main, re.S).group(0)
    assert '<figure class="board-fig board-fig--card">' in habit and 'class="a a-played"' in habit
    assert 'href="#why-1"' in habit and "Why this goes wrong" in habit
    # in a one-format view the chip would only repeat the view's heading
    assert "fmt-chip" not in re.search(r'<article class="card insight.*?</article>', _view(page, "blitz"), re.S).group(0)


def test_format_chip_text():
    assert format_chip_text({"blitz": 88, "bullet": 210}) == "Bullet 210 · Blitz 88"
    assert format_chip_text({"blitz": 88}) == "Blitz only"
    assert format_chip_text({"rapid": 3, "daily": 1, "bullet": 0}) == "Rapid 3 · Daily 1"
    assert format_chip_text({}) == format_chip_text(None) == format_chip_text({"blitz": "x"}) == ""


def test_glance_names_the_formats_of_a_partial_finding(page):
    glance = re.search(r'<section class="sw-col" id="weaknesses".*?</section>', page, re.S).group(0)
    assert "blitz and rapid" in glance and "bullet and blitz" in glance
    assert "bullet, blitz and rapid" not in glance  # a finding about every format needs no label


def test_module_boards_have_arrows_strips_and_why_links(page):
    main = _view(page, "all")
    section = re.search(r'<section class="section module" id="mistakes".*?</section>', main, re.S).group(0)
    # the Sicilian board is the finding's own (same position and arrows): drawn once, in the finding's card
    assert len(re.findall(r'<figure class="board-fig"[ >]', section)) == 1
    assert section.count('<figure class="board-fig board-fig--card">') == 1
    assert 'class="a a-threat"' in section and 'class="rg rg-target"' in section
    assert '<div class="pos pos--lines">' in section and section.count('class="cb cb--mini"') == 4
    assert ">What 4.e4 allows<" in section and ">4...Qxd4<" in section
    assert section.count('class="why-link"') >= 3  # both boards and the costliest-puzzle rows
    assert re.search(r'board<span aria-hidden="true"> ↗</span></a> <a class="why-link" href="#why-1">why</a>', section)
    assert "your move e5" in section and "better a6" in section  # the colour key in words


# --------------------------------------------------------------------------- coaching
def test_coaching_section_cards(page):
    main = _view(page, "all")
    nav = re.search(r'<nav class="chips".*?</nav>', main, re.S).group(0)
    assert re.findall(r'href="#([\w-]+)"', nav)[-3:] == ["mistakes", "why", "practice"]  # after the positions
    assert '<a href="#why">Why<span class="count">12</span></a>' in nav  # the cards on the page
    why = re.search(r'<section class="section module coach" id="why".*?</section>', main, re.S).group(0)
    assert ">Why these moves go wrong<" in why
    first = re.search(r'<article class="card why" id="why-1">.*?</article>', why, re.S).group(0)
    assert first.count('class="cb cb--mini"') == 9  # the refutation (5 frames) and the better line (4)
    assert "5...e5 6.Ndb5 a6 7.Nd6+ Bxd6 8.Qxd6 (−1.52 for you)" in first
    assert 'href="https://lichess.org/training/fork"' in first and ">advanced pawn<" in first
    assert 'viewBox="0 0 300 ' in first  # the concept chart
    assert "the masters&#x27; choice number 4" in first and "Lichess players rated 1200–1800" in first
    assert "Wikibooks, CC BY-SA 4.0" in first and "GPL-3.0" in first
    assert 'href="#f-positions-repeated-sicilian-5e5"' in first  # back to the finding
    # an explanation without a diagram gets one built from its position and lines
    third = re.search(r'<article class="card why" id="why-3">.*?</article>', why, re.S).group(0)
    assert 'class="a a-played"' in third and third.count('class="cb cb--mini"') == 6
    tb = re.search(r'<article class="card why" id="why-4">.*?</article>', why, re.S).group(0)
    assert ("With perfect play this position is a win for you; the tablebase move is Ra6+; after your move it is "
            "a win for you.") in tb  # "played_category": "loss" is the opponent's
    assert "Lichess players rated about 1400 (about 949 on chess.com) find 60.Ra6+ 31% of the time" in tb
    # the first five are open (a card is a phone screen or two), the rest fold away; at most WHY_HTML_MAX (12) are
    # written, the ones a finding links to first, and a line says where the others are
    more = re.search(r'<details class="more why-more"><summary>Show 7 more explanations</summary>.*?</details>', why, re.S)
    assert more and 'id="why-6"' in more.group(0) and 'id="why-5"' not in more.group(0)
    assert why.count('<article class="card why"') == 12
    assert ("4 more explanations are left out of this page to keep it light on a phone. All 16 are in the Markdown "
            "version of this report") in why
    assert "Patterns in the mistakes" in why and "Where you leave opening theory" in why
    assert "Endgames checked with the tablebase" in why and "no Lichess token" in why
    assert "12 explanations reworded" in why


def test_practice_section(page):
    main = _view(page, "all")
    practice = re.search(r'<section class="section module coach" id="practice".*?</section>', main, re.S).group(0)
    assert "Improved</span><span>clock used by move 15: 50% -&gt; 38%" in practice and "Not yet" in practice
    assert practice.count('<input type="checkbox" data-k=') == 12  # 2 due + 10 scheduled, all tickable
    assert "Show 2 more" in practice
    drill = re.search(r'<article class="card drill">.*?</article>', practice, re.S).group(0)
    assert "30 fork puzzles" in drill and "bigmuffeater-drill-fork.pgn" in drill
    assert 'href="https://lichess.org/training/fork"' in drill and 'href="https://lichess.org/training/01243"' in drill
    assert drill.count('class="a a-best"') == 2  # the first solution move of each puzzle, green
    assert ">Your week<" in practice and "fork.pgn): 10 minutes on six days<" in practice
    assert ">10 Oct" in practice  # (with the year when it is not this one)
    # checkboxes of the review list are remembered like the study plan's
    assert ".review input[type=checkbox][data-k]" in page


def test_format_view_shows_that_formats_explanations(report, page):
    from chess_insights.report.html import WHY_VIEW_MAX

    blitz = _view(page, "blitz")
    exps = [e for e in report.coaching.explanations if e.time_class == "blitz"]
    assert blitz.count('<article class="card why"') == min(len(exps), WHY_VIEW_MAX) > 0
    assert 'id="v-blitz-why-1"' in blitz and "Practice" not in re.search(r'<nav.*?</nav>', blitz, re.S).group(0)


def test_format_view_repeats_at_most_three_explanations(report):
    from chess_insights.report.html import WHY_VIEW_MAX, view_coaching

    rep = copy.deepcopy(report)
    base = rep.coaching.explanations[0]  # a blitz position, and a finding of every view links to it
    rep.coaching.explanations = [copy.deepcopy(base) for _ in range(13)]
    blitz = view_coaching(rep.format_reports["blitz"], rep, "blitz")
    assert len(blitz.explanations) == WHY_VIEW_MAX == 3
    assert blitz.notes == ["10 more explanations of blitz positions are left out of this view: see the All formats "
                           "view, and the Markdown report for every one."]
    # another format's explanation comes only when one of the view's findings links to it
    lone = copy.deepcopy(base)
    lone.insight_id, lone.fen, lone.epd = None, ROOK_END, " ".join(ROOK_END.split()[:4])
    rep.coaching.explanations = [lone]
    assert view_coaching(rep.format_reports["rapid"], rep, "rapid") is None  # no rapid game, no rapid finding
    rep.coaching.explanations = [copy.deepcopy(base)]
    assert len(view_coaching(rep.format_reports["rapid"], rep, "rapid").explanations) == 1  # a rapid finding's
    # the pipeline's own per-format coaching wins (as pipeline.view_coaching builds it: every explanation of the
    # format, the notes and the AI coach's counts), but is capped the same way, and the counts (for the whole
    # report) stay in the All formats view
    own = Coaching(explanations=[copy.deepcopy(base) for _ in range(12)], notes=["own"],
                   llm={"model": "m", "accepted": 12, "rejected": 2})
    rep.format_reports["rapid"].coaching = own
    rapid = view_coaching(rep.format_reports["rapid"], rep, "rapid")
    assert rapid.explanations == own.explanations[:3] and rapid.llm == {}
    assert rapid.notes == ["own", "9 more explanations of rapid positions are left out of this view: see the All "
                                  "formats view, and the Markdown report for every one."]
    assert len(own.explanations) == 12 and own.llm  # the report's own data is left alone
    html = _view(render_html(rep), "rapid")
    assert html.count('<article class="card why"') == 3 and "explanations reworded" not in html


def test_absent_coaching_renders_nothing():
    for coaching in (None, Coaching()):
        rep = sample_report(views=False, coaching=False)
        rep.coaching = coaching
        html = render_html(rep)
        check_html(html)
        assert "Why these moves go wrong" not in html and 'id="practice"' not in html
        md = render_markdown(rep)
        assert "## Why these moves go wrong" not in md and "## Practice" not in md
    rep.coaching = Coaching(notes=["Coaching needs the engine analysis (--engine): nothing to explain yet."])
    html = render_html(rep)
    assert "nothing to explain yet" in html and "Puzzle packs" not in html and "Due for review" not in html


def test_broken_positions_never_crash():
    rep = empty_report()
    rep.modules[0].insights = [
        Insight("x.1", "weakness", "results", "t", "d", 0.5, 0.9, formats={"blitz": 3},
                diagram=Diagram("Broken", "not a fen", arrows=[Arrow("e2", "e4", "played")],
                                strips=[Strip("s", [Frame("also bad", "1.e4")])]),
                chart=Chart("bar", "Empty", [], []))
    ]
    rep.coaching = Coaching(
        explanations=[Explanation("bad", "bad fen", "1.e4", None, None, None, text="Still explained.")],
        review_due=[ReviewItem("own:1", "own", "Bad", "zzz", "javascript:alert(1)", "2026-09-26")],
        drills=[Drill("fork", "Forks", "javascript:alert(1)", puzzles=[DrillPuzzle("p", "nope", ["e2e4"], ["e4"], 1500)])],
    )
    html = render_html(rep)
    check_html(html)
    assert "Still explained." in html and ">Bad<" in html and "Forks" in html
    assert 'href="javascript' not in html
    assert '<svg class="cb' not in html and SPRITE not in html  # no board, so no sprite either
    assert "Blitz only" in html
    md = render_markdown(rep)
    assert "Still explained." in md
    json.loads(to_json(rep))


def test_annotations_win_over_a_legacy_svg():
    rep = empty_report()
    legacy = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="8" height="8"/></svg>'
    rep.modules[0].diagrams = [Diagram("Old", SICILIAN, svg=legacy), Diagram("New", SICILIAN, svg=legacy,
                                                                             arrows=[Arrow("e6", "e5", "played")])]
    html = render_html(rep)
    boards_html = re.findall(r'<div class="board-svg"[^>]*>(.*?)</div>', html, re.S)
    assert boards_html[0].startswith('<svg xmlns="http://www.w3.org/2000/svg"')  # legacy, sanitised
    assert boards_html[1].startswith('<svg class="cb"') and 'class="a a-played"' in boards_html[1]


# --------------------------------------------------------------------------- Markdown and JSON
def test_markdown_positions_views_and_coaching(report):
    md = render_markdown(report)
    assert "<svg" not in md
    for heading in ("# Bullet only: 652 games", "# Blitz only: 949 games", "# Rapid only: 1,130 games",
                    "## Why these moves go wrong", "## Practice", "### Puzzle packs", "### Due for review"):
        assert f"\n{heading}\n" in md, heading
    assert md.index("## Positions you keep getting wrong") < md.index("## Why these moves go wrong") < md.index("# Bullet only")
    assert "Formats: Bullet 652 · Blitz 949 · Rapid 1,130" in md
    assert f"`{SICILIAN}`" in md and "https://lichess.org/analysis/" + QGA.replace(" ", "_") in md
    assert "Arrows: Your move e4 (red), better Nf3 (green), threat Qxd4 (orange)" in md
    assert "What 4.e4 allows: `4.e4 Qxd4 5.Qxd4 Nxd4`" in md
    assert "`5...e5 6.Ndb5 a6 7.Nd6+ Bxd6 8.Qxd6 (−1.52 for you)`" in md
    assert "What 5...e5 allowed: [fork](https://lichess.org/training/fork)" in md
    assert "_(High confidence · Blitz 14 · Rapid 7)_" in md
    assert "Chart: Score as White by format: Bullet +4%, Blitz +7%, Rapid +5%" in md
    assert "why: explanation 1 below" in md
    assert "- [ ] Fork puzzle 01243 — due 2026-09-26" in md
    assert "([Wikibooks, CC BY-SA 4.0](https://en.wikibooks.org/wiki/Chess_Opening_Theory/1._e4/1...c5))" in md


def test_json_carries_everything_as_data(report):
    text = to_json(report)

    def no_constants(name):
        raise AssertionError(name)

    data = json.loads(text, parse_constant=no_constants)
    assert "<svg" not in text and '"svg"' not in text  # boards are FEN + annotations, never markup
    assert set(data["format_reports"]) == {"bullet", "blitz", "rapid"}
    assert data["format_reports"]["blitz"]["time_class"] == "blitz"
    exp = data["coaching"]["explanations"][0]
    assert exp["refutation"]["moves_san"][:2] == ["e5", "Ndb5"] and exp["opening"]["masters"][0]["san"] == "a6"
    assert exp["diagram"]["arrows"][0] == {"start": "e6", "end": "e5", "kind": "played"}
    assert exp["diagram"]["strips"][0]["frames"][1]["move"] == "6.Ndb5"
    assert data["coaching"]["drills"][0]["rating_range"] == [1200, 1600]  # tuples become lists
    assert data["weaknesses"][0]["formats"] == {"bullet": 65, "blitz": 94, "rapid": 113}
    assert data["weaknesses"][0]["chart"]["series"][0]["name"] == "You"


def test_nan_in_coaching_is_null():
    rep = empty_report()
    rep.coaching = copy.deepcopy(sample_coaching(extra=0))
    rep.coaching.explanations[0].drop = float("nan")
    rep.coaching.settings["x"] = float("inf")
    data = json.loads(to_json(rep))
    assert data["coaching"]["explanations"][0]["drop"] is None and data["coaching"]["settings"]["x"] is None
    html = render_html(rep)
    assert "nan" not in re.sub(r"<[^>]+>", "", html).lower().split()


# --------------------------------------------------------------------------- wording of engine and database facts
def test_tablebase_facts_are_from_your_side():
    from chess_insights.report.html import tablebase_text

    # the shape coach/sources/tablebase.py writes; a move's "played_category" is the opponent's (Lichess API)
    kept = {"category": "win", "result": "win", "best": "Ra6+", "played": "Kc4", "played_category": "loss"}
    assert tablebase_text(kept) == ("with perfect play this position is a win for you; the tablebase move is Ra6+; "
                                    "after your move it is a win for you")
    assert tablebase_text({"category": "win", "played_category": "draw"}).endswith("after your move it is a draw")
    assert tablebase_text({"category": "draw", "played_category": "win"}).endswith("it is a loss for you")
    assert "a loss for you on the board, but the 50-move rule saves a draw" in tablebase_text(
        {"category": "draw", "played_category": "cursed-win"})
    # "played_result" (your side already) wins over the raw category
    assert tablebase_text({"played_result": "draw", "played_category": "win"}) == "after your move it is a draw"
    for junk in (None, "win", {}, {"category": "unknown"}, {"category": 3, "played_category": None}):
        assert tablebase_text(junk) == ""


def test_maia_names_the_lichess_rating():
    from chess_insights.report.html import maia_text

    exp = Explanation("e", SICILIAN, "5...e5", "5...a6", None, None)
    text = maia_text({"rating": 1390, "chesscom_rating": 949, "p_best": 0.31, "p_played": 0.2}, exp)
    assert text == ("In the Maia-2 model of human play, Lichess players rated about 1390 (about 949 on chess.com) "
                    "find 5...a6 31% of the time and play 5...e5 20% of the time.")
    assert maia_text({"p_best": 0.31}, exp).startswith("In the Maia-2 model of human play, players at your level")
    assert maia_text({"rating": 1390}, exp) == "" and maia_text(None, exp) == ""


def test_line_verdicts_checkmate_and_cut_lines():
    from chess_insights.models import Line
    from chess_insights.report.html import line_text, pawns_text

    assert pawns_text(-152) == "−1.52 for you" and pawns_text(None, 3) == "mate in 3 for you"
    assert pawns_text(-1000, -2) == "mate in 2 against you"
    # mate 0: the line ends in checkmate; the engine's mate-sized centipawns say who gave it
    assert pawns_text(1000, 0) == "checkmate for you" and pawns_text(-1000, 0) == "checkmate against you"
    fools = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert line_text(Line(fools, ["f2f3", "e7e5", "g2g4", "d8h4"], cp_end=-1000, mate_end=0)) == (
        "1.f3 e5 2.g4 Qh4# (checkmate against you)")
    # a line cut short (or stopped at an illegal move) says so before the verdict of its end
    ref = Line(SICILIAN, ["e6e5", "d4b5", "a7a6", "b5d6", "f8d6", "d1d6"], cp_end=-152)
    assert line_text(ref, limit=3) == "5...e5 6.Ndb5 a6 … (−1.52 for you)"
    assert line_text(Line(SICILIAN, ["e6e5", "e2e4"], cp_end=-152)) == "5...e5 … (−1.52 for you)"


def test_why_links_name_a_choice_point_as_one():
    rep = sample_report(views=False)
    for exp in rep.coaching.explanations:
        exp.kind = "choice"
    main = render_html(rep)
    card = re.search(r'<article class="card insight[^>]*id="f-positions-repeated-sicilian-5e5".*?</article>', main, re.S)
    assert "The engine&#x27;s lines for this position" in card.group(0) and "Why this goes wrong" not in card.group(0)
    assert "Why it goes wrong" not in main and ">The engine&#x27;s lines for this position</a>" in main


# --------------------------------------------------------------------------- views and odd coaching data
def test_format_views_one_per_id_and_never_a_copy_of_the_main_report():
    from chess_insights.report.html import format_views

    rep = sample_report(views=False, coaching=False)
    blitz = sample_report(views=False, coaching=False, time_class="blitz")
    blitz.time_class = ""  # a view that forgot its format gets it from its key
    rep.format_reports = {"Blitz": blitz, "blitz": blitz, "all": blitz, "bullet": "not a report"}
    views = format_views(rep)
    assert [tc for tc, _ in views] == ["blitz"] and views[0][1].time_class == "blitz" and blitz.time_class == ""
    html = render_html(rep)
    check_html(html)
    assert html.count('<input class="fmt-radio"') == 2 and "Blitz only: 949 games" in html
    # the only format of a one-format report: no views (the view would repeat the report)
    one = sample_report(views=False, coaching=False, time_class="blitz")
    one.time_class = ""
    one.format_reports = {"blitz": sample_report(views=False, coaching=False, time_class="blitz")}
    assert format_views(one) == [] and '<input class="fmt-radio"' not in render_html(one)


def test_a_link_into_another_view_opens_that_view(page):
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    assert "fmt-view--" in script and "r.checked = true" in script and "scrollIntoView" in script


def test_malformed_coaching_data_never_crashes():
    from chess_insights.models import Line, MoveStat, OpeningFacts, PlanEntry, ProgressItem, Table

    bad = Explanation(epd=None, fen="garbage", played=None, best=None, best_line=None, refutation=None,
                      motifs=[None], concepts=[None], facts=[None, ""], text=None, sources=[None], kind=None,
                      drop=float("nan"), time_class=None, game_url=None, games=None, repeats=None, opening="x",
                      tablebase="str", maia=None, diagram="x", chart="y", drill_themes=None)
    odd = Explanation("", SICILIAN, "5...e5", "5...a6", Line(SICILIAN, ["zz"]), Line("bad", ["e6e5"], cp_end=-100),
                      tablebase={"category": None, "played_category": 5}, maia={"rating": "x", "p_best": float("nan")},
                      opening=OpeningFacts(masters=[None, MoveStat("")], played_rank_masters=float("nan")),
                      diagram=Diagram("t", SICILIAN, arrows=[None, Arrow(None, None)],
                                      strips=[Strip("s", [None, Frame("bad"), Frame(SICILIAN, None)]), None]))
    rep = empty_report()
    rep.coaching = Coaching(
        explanations=[bad, odd, None],
        drills=[Drill("fork", None, None, None, None, [DrillPuzzle("1", "bad", [], [], None), None,
                                                      DrillPuzzle("2", SICILIAN, None, [], "x")]), None],
        review=[ReviewItem("a", "own", None, "bad", None, "garbage"), None],
        weekly_plan=[PlanEntry("x", None), PlanEntry(None, 3)],
        progress=[ProgressItem("a", "m", None, float("nan"), "")],
        motif_profile=Table("t", ["a"], [[None]]), motif_chart=Chart("pie", "c", [], []), llm={"accepted": "x"},
    )
    html = render_html(rep)
    check_html(html)
    assert html.count(SPRITE) == 1 and '<svg class="cb' in html  # the readable positions still get boards
    assert "5...e5" in render_markdown(rep)
    json.loads(to_json(rep))


def test_endgame_boards_are_drawn_when_the_coaching_carries_them():
    from chess_insights import visuals

    rep = sample_report(views=False)
    rep.coaching.endgame_diagrams = [visuals.position_diagram(  # not in every version of the contract
        "Rook ending: Kc4", "8/8/4k3/8/8/3K4/8/R7 w - - 0 60", played="Kc4", best="Ra6+", time_class="blitz")]
    html = render_html(rep)
    check_html(html)
    block = re.search(r'Endgames checked with the tablebase</h3>.*?</div></div>', html, re.S).group(0)
    # explanation 4 draws the same position with the same arrows higher up: one board, and a link to it here
    assert "Rook ending: Kc4" in block and 'href="#why-4">Board shown above, with 60.Kc4</a>' in block
    assert 'class="a a-best"' not in block
    rep.coaching.endgame_diagrams[0].arrows = rep.coaching.endgame_diagrams[0].arrows[:1]  # another picture
    block = re.search(r'Endgames checked with the tablebase</h3>.*?</div></div>', render_html(rep), re.S).group(0)
    assert "Rook ending: Kc4" in block and 'class="a a-played"' in block and "Board shown above" not in block


def test_concept_note_is_rendered_with_its_credit():
    from chess_insights.report.html import _concept_note_html

    note = {"label": "king safety", "text": "Keep the king <safe>.", "credit": "Capablanca, Chess Fundamentals",
            "url": "https://www.gutenberg.org/ebooks/33870"}
    html = _concept_note_html(note)
    assert "About king safety:" in html and "&lt;safe&gt;" in html
    assert 'href="https://www.gutenberg.org/ebooks/33870"' in html
    assert _concept_note_html({}) == "" and _concept_note_html(None) == ""
    assert "javascript" not in _concept_note_html(dict(note, url="javascript:alert(1)"))
