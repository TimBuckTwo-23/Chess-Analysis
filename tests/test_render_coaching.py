"""Per-finding visuals, format views and the coaching sections in the HTML, Markdown and JSON reports."""

from __future__ import annotations

import copy
import json
import re

import pytest
from render_sample import QGA, SICILIAN, sample_coaching, sample_report
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
    assert section.count('<figure class="board-fig">') == 2
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
    assert '<a href="#why">Why<span class="count">16</span></a>' in nav
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
    assert "a win for you" in tb and "Ra6+" in tb and "players rated about 1400 find 60.Ra6+ 31%" in tb
    # the first ten are open, the rest fold away
    more = re.search(r'<details class="more why-more"><summary>Show 6 more explanations</summary>.*?</details>', why, re.S)
    assert more and 'id="why-11"' in more.group(0) and 'id="why-10"' not in more.group(0)
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
    assert ">Your week<" in practice and ">10 minutes on six days<" in practice
    # checkboxes of the review list are remembered like the study plan's
    assert ".review input[type=checkbox][data-k]" in page


def test_format_view_shows_that_formats_explanations(report, page):
    blitz = _view(page, "blitz")
    exps = [e for e in report.coaching.explanations if e.time_class == "blitz"]
    assert blitz.count('<article class="card why"') == len(exps) > 0
    assert 'id="v-blitz-why-1"' in blitz and "Practice" not in re.search(r'<nav.*?</nav>', blitz, re.S).group(0)


def test_format_view_repeats_at_most_ten_explanations(report):
    from chess_insights.report.html import view_coaching

    rep = copy.deepcopy(report)
    rep.coaching.explanations = [copy.deepcopy(rep.coaching.explanations[0]) for _ in range(13)]
    blitz = view_coaching(rep.format_reports["blitz"], rep, "blitz")
    assert len(blitz.explanations) == 10 and "3 more explanations of blitz positions" in blitz.notes[0]
    assert view_coaching(rep.format_reports["rapid"], rep, "rapid") is None  # none of rapid games
    own = Coaching(notes=["own"])
    rep.format_reports["rapid"].coaching = own
    assert view_coaching(rep.format_reports["rapid"], rep, "rapid") is own  # the pipeline's own wins


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
