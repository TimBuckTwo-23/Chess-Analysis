"""Report renderers: HTML (standalone + Artifact fragment), Markdown, JSON and write_report."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from chess_insights.models import Chart, Insight, Kpi, ModuleResult, Report, Series, StudyItem, Table
from chess_insights.report import render_html, render_markdown, to_dict, to_json, write_report
from chess_insights.report.html import (
    MINUS,
    MISSING,
    chart_table,
    format_value,
    glyph_for,
    infer_format,
    module_anchors,
    nice_ticks,
    safe_url,
)

XSS = "<script>alert(1)</script>"
JS_URL = "javascript:alert(1)"
GAME = "https://www.chess.com/game/live/{}"


# --------------------------------------------------------------------------- sample report
def _ins(kind, category, title, detail, severity, confidence, *, games=(), study=(), key=None) -> Insight:
    return Insight(
        id=key or f"{category}.{kind}.{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}",
        kind=kind,
        category=category,
        title=title,
        detail=detail,
        severity=severity,
        confidence=confidence,
        evidence={"n": 24, "score": 0.41, "expected": 0.52},
        study=list(study),
        example_games=list(games),
    )


def build_sample_report(hostile: bool = False) -> Report:
    """A realistic multi-module report exercising every format, chart kind and edge case."""
    bad = f" {XSS} & \"quotes\" 'single'" if hostile else ""
    months = [f"2024-{m:02d}" for m in range(1, 13)]

    caro = _ins(
        "weakness",
        "openings",
        "The Caro-Kann is costing you points as Black" + bad,
        "24 games as Black: you scored 38% where your ratings predicted 52% (−14 points per 100 games).",
        0.7,
        0.82,
        games=[GAME.format(105000000002), GAME.format(105000000017), JS_URL if hostile else GAME.format(105000000021)],
        study=[
            "Replay your 9 Caro-Kann losses and mark the move where you left known theory.",
            "Learn one main line against the Advance Variation (3.e5 Bf5).",
        ],
    )
    time_trouble = _ins(
        "weakness",
        "time",
        "You lose too many blitz games on time",
        "31% of your blitz losses were on time, against 18% for players at your level; you reach move 30 with 41s left on average.",
        0.45,
        0.74,
        games=[GAME.format(105000000031), GAME.format(105000000044)],
        study=["Aim to reach move 15 with at least 70% of your clock."],
    )
    tilt = _ins(
        "weakness",
        "habits",
        "Your results drop after two losses in a row",
        "After two straight losses you score 39% against an expected 50% (58 games).",
        0.35,
        0.45,
        games=[GAME.format(105000000051)],
    )
    italian = _ins(
        "strength",
        "openings",
        "The Italian Game is working well for you as White",
        "41 games: 63% scored where 51% was expected.",
        0.6,
        0.88,
        games=[GAME.format(105000000001)],
        study=["Keep it: add one sideline (the Evans Gambit) to stay unpredictable."],
    )
    endgames = _ins(
        "strength",
        "conversion",
        "You convert winning positions reliably",
        "You won 87% of games where you were +3 or better at move 30.",
        0.3,
        0.6,
    )
    observation = _ins(
        "observation",
        "results",
        "Most of your games are blitz",
        "78% of your games are blitz; rapid results are based on only 12 games.",
        0.1,
        0.9,
    )

    results = ModuleResult(
        key="results",
        title="Results & rating",
        summary="You scored 53% against opponents who were on average slightly weaker, 4 points below what your rating predicts.",
        kpis=[
            Kpi("Games", 1234, "int"),
            Kpi("Score", 0.534, "pct", hint="wins + half the draws"),
            Kpi("Score vs expected", -0.043, "signed_pct", hint="per game, Elo expected score"),
            Kpi("Current rating", 1512, "rating"),
            Kpi("Rating change", 87, "signed_int", hint="since January"),
            Kpi("Draw rate", 0.053, "pct"),
            Kpi("Average game length", 38.46, "float1", hint="moves"),
            Kpi("Performance vs expected", 1.0432, "float2"),
            Kpi("Best win", GAME.format(105000000009), "url"),
            Kpi("Most played opponent", "knightrider77" + bad, "text"),
            Kpi("Peak rating date", None, "text"),
        ],
        charts=[
            Chart(
                kind="line",
                title="Rating by month" + bad,
                labels=months,
                series=[
                    Series("Blitz", [1421, 1438, 1450, None, 1472, 1466, 1490, 1484, 1501, 1498, 1507, 1512]),
                    Series("Rapid", [1502, None, 1515, 1522, 1519, None, None, 1540, 1548, 1551, 1549, 1560]),
                ],
                value_format="rating",
            ),
            Chart(
                kind="stacked_bar",
                title="Result mix by time control",
                labels=["Bullet", "Blitz", "Rapid", "Daily"],
                series=[
                    Series("Wins", [0.44, 0.49, 0.55, 0.61]),
                    Series("Draws", [0.03, 0.05, 0.09, 0.14]),
                    Series("Losses", [0.53, 0.46, 0.36, 0.25]),
                ],
                value_format="pct",
            ),
            Chart(
                kind="bar",
                title="Score vs expected by opponent strength",
                labels=["≤ −200", "−200 to −100", "−100 to 0", "0 to +100", "+100 to +200", "≥ +200"],
                series=[Series("Score − expected", [0.021, 0.034, -0.012, -0.061, -0.094, None])],
                value_format="signed_pct",
                note="Opponent rating minus yours. The last bucket has no games yet.",
            ),
            Chart(
                kind="bar",
                title="Score by colour",
                labels=["White", "Black"],
                series=[Series("Score", [0.57, 0.49]), Series("Expected", [0.52, 0.51])],
                value_format="pct",
                reference=0.5,
            ),
        ],
        tables=[
            Table(
                title="By time class",
                columns=["Time class", "Games", "Score", "Expected", "Difference", "Avg opponent", "Avg clock left"],
                rows=[
                    ["Bullet", 88, 0.47, 0.5, -0.03, 1455.2, 12.4],
                    ["Blitz", 962, 0.531, 0.52, 0.011, 1498.0, 41.0],
                    ["Rapid", 172, 0.58, 0.55, 0.03, 1520.7, 312.0],
                    ["Daily", 12, 0.625, 0.5, 0.125, None, None],
                ],
                formats=["text", "int", "pct", "pct", "signed_pct", "rating", "seconds"],
                note="Daily games have no clock, so clock columns are empty.",
            )
        ],
        insights=[observation],
    )

    openings = ModuleResult(
        key="openings",
        title="Openings",
        summary="Your White repertoire is in good shape; as Black the Caro-Kann and Sicilian Najdorf lines lose points.",
        kpis=[Kpi("Openings played", 37, "int"), Kpi("Most played", "Italian Game", "text")],
        charts=[
            Chart(
                kind="hbar",
                title="Score vs expected by opening (Black)",
                labels=[
                    "Caro-Kann Defense: Advance Variation, Short Variation" + bad,
                    "Sicilian Defense: Najdorf Variation, English Attack",
                    "French Defense",
                    "Scandinavian Defense: Mieses-Kotroc Variation",
                    "King's Indian Defense",
                    "Queen's Gambit Declined",
                ],
                series=[Series("Score − expected", [-0.14, -0.08, 0.02, 0.05, None, 0.0])],
                value_format="signed_pct",
            ),
            Chart(
                kind="hbar",
                title="Score and expected score (White)",
                labels=["Italian Game", "Ruy Lopez", "Scotch Game", "Vienna Game"],
                series=[Series("Score", [0.63, 0.51, 0.55, 0.42]), Series("Expected", [0.51, 0.52, 0.5, 0.5])],
                value_format="pct",
                reference=0.5,
            ),
        ],
        tables=[
            Table(
                title="Opening lines | with pipes" + bad,
                columns=["ECO", "Opening", "Colour", "Games", "Score", "Difference", "Example"],
                rows=[
                    ["B12", "Caro-Kann Defense: Advance Variation" + bad, "Black", 24, 0.38, -0.14, GAME.format(105000000002)],
                    ["C50", "Italian Game: Giuoco Pianissimo", "White", 41, 0.63, 0.12, GAME.format(105000000001)],
                    ["B90", "Sicilian | Najdorf", "Black", 15, 0.43, -0.08, JS_URL if hostile else None],
                ],
                formats=["text", "text", "text", "int", "pct", "signed_pct", "url"],
            )
        ],
        insights=[caro, italian],
    )

    long_labels = ["After a win", "After a draw", "After a loss", "After two losses in a row", "First game of the day", "Late-night games (after 23:00)"]
    minutes = [datetime(2024, 3, 1) + timedelta(days=d) for d in range(40)]
    time_mgmt = ModuleResult(
        key="time_mgmt",
        title="Clock & time management",
        summary="You spend the most time in the middlegame and often reach the endgame short of time in blitz.",
        kpis=[
            Kpi("Average time left at move 30", 95, "seconds"),
            Kpi("Longest think", 4000, "seconds", hint="daily games excluded"),
            Kpi("Median move time", 7.25, "seconds"),
            Kpi("Games lost on time", 0.31, "pct"),
        ],
        charts=[
            Chart(
                kind="bar",
                title="Average time per move by phase",
                labels=["Opening", "Middlegame", "Endgame"],
                series=[Series("Seconds", [3.2, 11.8, 6.4])],
                value_format="seconds",
            ),
            Chart(
                kind="line",
                title="Clock left at move 30, last 40 days",
                labels=[f"{d:%d %b}" for d in minutes],
                series=[Series("Clock", [60 + 50 * math.sin(i / 4) + i for i in range(40)])],
                value_format="seconds",
            ),
            Chart(
                kind="bar",
                title="Score in different situations",
                labels=long_labels,
                series=[Series("Score", [0.55, 0.52, 0.47, 0.39, 0.5, 0.41])],
                value_format="pct",
                reference=0.5,
            ),
        ],
        tables=[
            Table(
                title="Slowest moves",
                columns=["Game", "Move", "Time spent", "Clock after", "Played"],
                rows=[
                    [GAME.format(105000000061), "14... Nxe4", 95.0, 4000, date(2024, 5, 2)],
                    [GAME.format(105000000062), "23. Rxd7+", 7.25, 62.5, date(2024, 5, 9)],
                    [GAME.format(105000000063), "9. O-O", float("nan"), None, None],
                ],
                formats=None,
            )
        ],
        insights=[time_trouble],
    )

    habits = ModuleResult(
        key="habits",
        title="Habits & tilt",
        summary="Long sessions and back-to-back losses are where your results slip.",
        charts=[
            Chart(
                kind="bar",
                title="Games by hour of day",
                labels=[f"{h:02d}" for h in range(24)],
                series=[Series("Games", [4, 2, 1, 0, 0, 0, 1, 3, 6, 9, 12, 15, 22, 25, 18, 20, 31, 44, 58, 71, 66, 49, 30, 12])],
                value_format="int",
            ),
            Chart(
                kind="stacked_bar",
                title="Results by session length",
                labels=["1 game", "2–3 games", "4–6 games", "7+ games"],
                series=[Series("Wins", [51, 120, 88, 41]), Series("Draws", [4, 9, 7, 2]), Series("Losses", [40, 101, 92, 60])],
                value_format="int",
            ),
        ],
        insights=[tilt],
    )

    endings = ModuleResult(
        key="endings",
        title="How your games end",
        summary="Not enough data: fewer than 20 decisive games in this filter.",
        charts=[Chart(kind="bar", title="Endings", labels=["Checkmate", "Resignation"], series=[Series("Games", [None, None])])],
        tables=[Table(title="Terminations", columns=["How", "Games"], rows=[])],
    )
    engine = ModuleResult(
        key="engine_stats",
        title="Engine review",
        summary="This section could not be computed (RuntimeError: Stockfish crashed).",
        stats={"error": "RuntimeError: Stockfish crashed", "nan": float("nan"), "inf": float("inf")},
    )

    plan = [
        StudyItem(
            title=caro.title,
            why=caro.detail,
            actions=caro.study + ["Study 3-5 annotated master games in the Advance Variation."],
            games=caro.example_games,
            category="Opening repertoire",
            priority=caro.priority,
        ),
        StudyItem(
            title=time_trouble.title,
            why=time_trouble.detail,
            actions=time_trouble.study + ["Play some 3+2 games to practise finishing with increment."],
            games=time_trouble.example_games,
            category="Clock management",
            priority=time_trouble.priority,
        ),
        StudyItem(
            title=tilt.title,
            why=tilt.detail,
            actions=["After two losses in a row, take at least a 15-minute break."],
            games=[],
            category="Playing habits",
            priority=tilt.priority,
        ),
    ]
    return Report(
        username=("TesterBob" + bad) if hostile else "TesterBob",
        generated_at=datetime(2026, 9, 25, 14, 3, tzinfo=timezone.utc),
        filters="rated + casual, blitz+rapid, standard chess, since 2024-01-01",
        n_games=1234,
        date_from=datetime(2024, 1, 3, 18, 2, tzinfo=timezone.utc),
        date_to=datetime(2024, 12, 30, 21, 40, tzinfo=timezone.utc),
        modules=[results, openings, time_mgmt, habits, endings, engine],
        strengths=[italian, endgames],
        weaknesses=[caro, time_trouble, tilt],
        study_plan=plan,
        engine_note="Stockfish 16 at depth 12 on the 150 most recent games.",
        headline="Biggest opportunity: The Caro-Kann is costing you points as Black. Biggest strength: The Italian Game is working well for you as White.",
    )


def empty_report() -> Report:
    return Report(
        username="",
        generated_at=datetime(2026, 9, 25, 14, 3),
        filters="",
        n_games=0,
        date_from=None,
        date_to=None,
        modules=[ModuleResult(key="results", title="Results & rating", summary="Not enough games to analyse.")],
        strengths=[],
        weaknesses=[],
        study_plan=[],
    )


# --------------------------------------------------------------------------- HTML checker
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr", "param"}


class TagChecker(HTMLParser):
    """Records balance errors plus every href/src/id so tests can inspect them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.hrefs: list[tuple[str, str]] = []  # (tag, href)
        self.srcs: list[str] = []
        self.ids: list[str] = []
        self.tags: list[str] = []

    def _attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            if name == "href":
                self.hrefs.append((tag, value or ""))
            elif name == "src":
                self.srcs.append(value or "")
            elif name == "id":
                self.ids.append(value or "")

    def handle_starttag(self, tag, attrs):
        self._attrs(tag, attrs)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self._attrs(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-4:]}")
        else:
            self.stack.pop()


def check_html(text: str) -> TagChecker:
    checker = TagChecker()
    checker.feed(text)
    checker.close()
    assert not checker.errors, checker.errors[:5]
    assert not checker.stack, f"unclosed tags: {checker.stack}"
    return checker


@pytest.fixture(scope="module")
def report() -> Report:
    return build_sample_report()


@pytest.fixture(scope="module")
def hostile() -> Report:
    return build_sample_report(hostile=True)


# --------------------------------------------------------------------------- value formatting
@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (1234, "int", "1,234"),
        (1234.5, "int", "1,235"),
        (-7, "int", f"{MINUS}7"),
        (0.534, "pct", "53%"),
        (0.535, "pct", "54%"),
        (0.053, "pct", "5.3%"),
        (0.0, "pct", "0%"),
        (1.5, "pct", "150%"),
        (-0.25, "pct", f"{MINUS}25%"),
        (0.04, "signed_pct", "+4%"),
        (-0.07, "signed_pct", f"{MINUS}7%"),
        (0.004, "signed_pct", "+0.4%"),
        (0.0, "signed_pct", "0%"),
        (12, "signed_int", "+12"),
        (-1234, "signed_int", f"{MINUS}1,234"),
        (0, "signed_int", "0"),
        (1512, "rating", "1512"),
        (1498.6, "rating", "1499"),
        (38.46, "float1", "38.5"),
        (1234.5, "float1", "1,234.5"),
        (1.005, "float2", "1.01"),
        (-0.5, "float2", f"{MINUS}0.50"),
        (95, "seconds", "1:35"),
        (4000, "seconds", "1:06:40"),
        (7.25, "seconds", "7.3s"),
        (59.96, "seconds", "1:00"),
        (0, "seconds", "0.0s"),
        (-5, "seconds", f"{MINUS}5.0s"),
        ("Caro-Kann", "text", "Caro-Kann"),
        (True, "text", "yes"),
        (datetime(2024, 5, 2), "text", "2024-05-02"),
        (datetime(2024, 5, 2, 18, 30), "text", "2024-05-02 18:30"),
        ("https://www.chess.com/game/live/1", "url", "https://www.chess.com/game/live/1"),
        (None, "pct", MISSING),
        (float("nan"), "int", MISSING),
        (float("inf"), "float1", MISSING),
        (pd.NA, "int", MISSING),
        (pd.NaT, "text", MISSING),
        ("", "text", MISSING),
        ("n/a", "int", "n/a"),  # text in a numeric column is shown as is
        ("1,500", "rating", "1500"),
        (1234, None, "1,234"),
        (0.5, None, "0.50"),
        (7.0, "bogus", "7"),
    ],
)
def test_format_value(value, fmt, expected):
    assert format_value(value, fmt) == expected


def test_infer_format():
    assert infer_format(3) == "int"
    assert infer_format(3.0) == "int"
    assert infer_format(3.25) == "float2"
    assert infer_format(True) == "text"
    assert infer_format("https://x.org/a") == "url"
    assert infer_format("hello") == "text"


def test_safe_url():
    assert safe_url("https://www.chess.com/game/live/1") == "https://www.chess.com/game/live/1"
    assert safe_url(" http://example.org/a?b=1&c=2 ") == "http://example.org/a?b=1&c=2"
    for bad in (JS_URL, "JavaScript:alert(1)", " javascript:alert(1)", "data:text/html,x", "//evil.org", "https://a b", None, 5, "ftp://x"):
        assert safe_url(bad) is None, bad


def test_nice_ticks_are_round():
    assert nice_ticks(0, 0.63, "pct") == [0, 0.2, 0.4, 0.6, 0.8]
    assert nice_ticks(0, 1, "pct") == [0, 0.25, 0.5, 0.75, 1.0]
    assert nice_ticks(-0.094, 0.034, "signed_pct") == [-0.1, -0.05, 0, 0.05]
    assert nice_ticks(1421, 1560, "rating") == [1400, 1450, 1500, 1550, 1600]
    assert nice_ticks(0, 3, "int") == [0, 1, 2, 3]
    assert nice_ticks(0, 400, "seconds") == [0, 120, 240, 360, 480]
    assert nice_ticks(0, 0, "int") == [0, 1, 2, 3, 4]
    assert nice_ticks(5, 5) == [4.5, 4.75, 5.0, 5.25, 5.5]


def test_glyphs_follow_priority():
    weak = lambda p: Insight("x", "weakness", "results", "t", "d", severity=p, confidence=1.0)  # noqa: E731
    strong = lambda p: Insight("x", "strength", "results", "t", "d", severity=p, confidence=1.0)  # noqa: E731
    assert [glyph_for(weak(p)) for p in (0.9, 0.3, 0.1)] == ["??", "?", "?!"]
    assert [glyph_for(strong(p)) for p in (0.6, 0.2)] == ["!!", "!"]
    assert glyph_for(Insight("x", "observation", "results", "t", "d", 1.0, 1.0)) == "="


# --------------------------------------------------------------------------- HTML
def test_html_standalone_document(report):
    html = render_html(report)
    assert html.startswith("<!doctype html>")
    assert '<html lang="en">' in html
    assert '<meta charset="utf-8">' in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' in html
    assert "<title>TesterBob · Chess Insights</title>" in html
    checker = check_html(html)
    assert {"head", "body", "style", "main", "svg"} <= set(checker.tags)


def test_html_fragment_for_artifacts(report):
    frag = render_html(report, standalone=False)
    assert frag.startswith("<title>")
    assert re.match(r"<title>[^<]*</title>\s*<style>", frag)
    lowered = frag.lower()
    for tag in ("!doctype", "html", "head", "body"):
        assert not re.search(rf"</?{tag}[\s>]", lowered), tag
    check_html(frag)
    # same content as the standalone page
    assert frag.count("<svg") == render_html(report).count("<svg")


def test_html_theme_tokens_and_layout(report):
    html = render_html(report)
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    assert style.startswith(":root{--paper:#F6F6F3;")
    assert '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){' in style
    assert ':root[data-theme="dark"]{' in style
    assert style.count(";color-scheme:dark") == 2  # both dark blocks
    assert "body{margin:0;background:var(--paper)" in style
    assert "padding-block:" in style and "padding-left:max(clamp(16px" in style
    assert "overflow-x:auto" in style and "font-variant-numeric:tabular-nums" in style
    assert "url(" not in style  # no external assets from CSS
    for i in range(1, 9):
        assert f"--s{i}:" in style


def test_html_escapes_hostile_data(hostile):
    for html in (render_html(hostile), render_html(hostile, standalone=False)):
        assert "<script>alert" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        checker = check_html(html)
        for tag, href in checker.hrefs:
            assert "javascript:" not in href.lower()
            if tag == "link":
                assert href.startswith("https://fonts.googleapis.com/")
            else:
                assert href.startswith(("https://", "http://", "#")), href
        assert not checker.srcs  # no external scripts or images
        external = {href for tag, href in checker.hrefs if tag == "a" and href.startswith("http")}
        assert external and all("chess.com" in h for h in external)
        # the unsafe URL is still visible as text, just not a link
        assert "javascript:alert(1)" in html


def test_html_sections_nav_and_cards(report):
    html = render_html(report)
    checker = check_html(html)
    anchors = [h[1:] for t, h in checker.hrefs if t == "a" and h.startswith("#")]
    assert anchors[:3] == ["study", "strengths", "weaknesses"]
    assert {"results", "openings", "time_mgmt", "habits", "endings", "engine_stats"} <= set(anchors)
    assert set(anchors) <= set(checker.ids)
    assert len(checker.ids) == len(set(checker.ids))  # ids are unique
    # study plan: ordered list, action checklist, review links
    assert '<ol class="plan">' in html and html.count('class="plan-item"') == 3
    assert "Review these games" in html and ">Game 1<" in html
    # glyph badges + confidence pills
    assert 'aria-label="Serious weakness"' in html and ">??</span>" in html
    assert ">!!</span>" in html and ">=</span>" in html
    assert "High confidence" in html and "Medium confidence" in html
    # KPI formats
    for text in ("1,234", "53%", f"{MINUS}4%", "1512", "+87", "5.3%", "38.5", "1.04", "1:35", "1:06:40", "7.3s", MISSING):
        assert f">{text}<" in html, text
    # generated time + method notes
    assert "25 Sep 2026, 14:03 UTC" in html and "Elo expected score" in html
    assert "3 Jan 2024 – 30 Dec 2024" in html


def test_html_charts(report):
    html = render_html(report)
    svgs = re.findall(r"<svg .*?</svg>", html, re.S)
    charted = sum(1 for m in report.modules for c in m.charts if any(v is not None for s in c.series for v in s.values))
    assert len(svgs) == charted
    for svg in svgs:
        assert re.search(r'viewBox="0 0 400 [\d.]+"', svg)
        assert "<title>" in svg  # every chart carries hover tooltips
    assert html.count('<ul class="legend"') == 6  # multi-series charts + the single-series chart with a reference
    assert "Reference 50%" in html
    assert 'class="zero"' in html  # signed charts emphasise the zero line
    rating_line = next(s for s in svgs if "Rating by month" in s)
    assert 'class="ln k1"' in rating_line and 'class="ln k3"' in rating_line  # Blitz and Rapid keep their own colours
    # a month or two without games (None) does not break a monthly line
    assert "ln--gap" not in rating_line and not re.search(r'class="ln k\d" d="M[^"]* M', rating_line)
    situations = next(s for s in svgs if "Score in different situations" in s)
    assert "rotate(" not in situations and "Late-night games (after 23:00)" in situations.replace("</tspan>", " ")  # long labels: bars turn horizontal
    black = next(s for s in svgs if "(Black)" in s)
    assert black.count("<tspan") > 6  # long opening names wrap onto a second line
    assert "Caro-Kann Defense: Advance Variation, Short Variation\n" in black  # full name in tooltip
    assert "No data to chart yet." in html  # all-None chart
    assert html.count('class="chart-data"') == charted  # every chart has its data table twin
    assert "No rows yet." in html


def test_chart_table_matches_series():
    chart = Chart(kind="line", title="t", labels=["a", "b", "c"], series=[Series("x", [1, None]), Series("", [3, 4, 5, 6])])
    table = chart_table(chart)
    assert table.columns == ["", "x", "Series 2"]
    # the value beyond the labels is kept, labelled by its position
    assert table.rows == [["a", 1.0, 3.0], ["b", None, 4.0], ["c", None, 5.0], ["4", None, 6.0]]


@pytest.mark.parametrize(
    "chart",
    [
        Chart(kind="bar", title="empty", labels=[], series=[]),
        Chart(kind="bar", title="no labels", labels=[], series=[Series("x", [1, 2, 3])]),
        Chart(kind="hbar", title="zeros", labels=["a", "b"], series=[Series("x", [0, 0])]),
        Chart(kind="bar", title="negative only", labels=["a", "b"], series=[Series("x", [-3, -1])], value_format="signed_int"),
        Chart(kind="line", title="single point", labels=["only"], series=[Series("x", [0.5])], value_format="pct", reference=0.5),
        Chart(kind="line", title="isolated", labels=list("abcde"), series=[Series("x", [None, 2, None, 4, None])]),
        Chart(kind="stacked_bar", title="mixed signs", labels=["a", "b"], series=[Series("up", [2, 3]), Series("down", [-1, None])]),
        Chart(kind="stacked_bar", title="ten", labels=["a", "b"], series=[Series(f"s{i}", [i, i + 1]) for i in range(10)]),
        Chart(kind="line", title="ten lines", labels=["a", "b"], series=[Series(f"s{i}", [i, i + 1]) for i in range(10)]),
        Chart(kind="bar", title="hidden only", labels=["a"], series=[Series(f"s{i}", [None]) for i in range(8)] + [Series("s9", [1])]),
        Chart(kind="pie", title="unknown kind", labels=["a"], series=[Series("x", [1])]),  # type: ignore[arg-type]
        Chart(kind="bar", title="text values", labels=["a", "b"], series=[Series("x", ["3", "oops"])], value_format="text"),  # type: ignore[list-item]
        Chart(kind="hbar", title="big", labels=["x" * 200], series=[Series("x", [1e12])], value_format="int"),
        Chart(kind="bar", title="many", labels=[f"Category number {i}" for i in range(60)], series=[Series("x", list(range(60)))]),
        Chart(kind="line", title="nan", labels=["a", "b"], series=[Series("x", [float("nan"), float("inf")])]),
    ],
    ids=lambda c: c.title,
)
def test_chart_edge_cases_render(chart):
    rep = empty_report()
    rep.modules[0].charts = [chart]
    html = render_html(rep)
    check_html(html)
    assert "NaN" not in html and "Infinity" not in html and ">inf<" not in html
    render_markdown(rep)


def test_folded_and_truncated_series():
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind="stacked_bar", title="ten", labels=["a"], series=[Series(f"s{i}", [1]) for i in range(10)]),
        Chart(kind="line", title="ten lines", labels=["a", "b"], series=[Series(f"s{i}", [i, i]) for i in range(10)]),
    ]
    html = render_html(rep)
    assert ">Other</li>" in html
    assert "Showing the first 8 of 10 series" in html
    assert "f9" not in re.sub(r"#[0-9a-fA-F]{6}", "", html.split("</style>", 1)[1])  # never a 9th colour


def test_empty_report_renders_everywhere(tmp_path):
    rep = empty_report()
    html = render_html(rep)
    check_html(html)
    assert "<title>Chess Insights</title>" in html
    assert "0 games" in html and "Nothing to study yet" in html and "No clear strengths yet" in html
    md = render_markdown(rep)
    assert md.startswith("# Chess Insights\n")
    data = json.loads(to_json(rep))
    assert data["date_from"] is None and data["generated_at"] == "2026-09-25T14:03:00"
    assert len(write_report(rep, tmp_path / "empty")) == 3


def test_module_anchors_are_unique_and_safe():
    mods = [ModuleResult(key=k, title=k, summary="") for k in ("results", "results", "study", "9 lives", "", "a/b<c>")]
    assert module_anchors(mods) == ["results", "results-2", "study-section", "m-9-lives", "module-5", "a-b-c"]


# --------------------------------------------------------------------------- Markdown
def test_markdown_structure(report):
    md = render_markdown(report)
    assert md.startswith("# TesterBob · Chess Insights\n")
    assert "**1,234 games** · 3 Jan 2024 – 30 Dec 2024 · rated + casual" in md
    for heading in ("## Study plan", "## Strengths", "## Weaknesses", "## Results & rating", "## Openings", "### Findings"):
        assert f"\n{heading}\n" in md, heading
    assert "\n1. **The Caro-Kann is costing you points as Black** — _Opening repertoire_\n" in md
    assert "\n   - [ ] Replay your 9 Caro-Kann losses" in md
    assert "Review these games: [Game 1](https://www.chess.com/game/live/105000000002)" in md
    assert "- `??` **The Caro-Kann is costing you points as Black**" in md
    assert "- `=` **Observation · Overall results: Most of your games are blitz**" in md
    assert "| Time class | Games | Score | Expected | Difference | Avg opponent | Avg clock left |" in md
    assert "|---|---:|---:|---:|---:|---:|---:|" in md
    assert "| Daily | 12 | 63% | 50% | +13% | — | — |" in md
    assert "Sicilian \\| Najdorf" in md  # pipes inside cells are escaped
    assert "[game ↗](https://www.chess.com/game/live/105000000001)" in md
    assert "#### Rating by month" in md and "| 2024-04 | — | 1522 |" in md
    assert "_Reference line: 50%._" in md
    assert "- Average time left at move 30: **1:35**" in md
    assert "_Generated 25 Sep 2026, 14:03 UTC by chess-insights._" in md


def test_markdown_escapes_hostile_data(hostile):
    md = render_markdown(hostile)
    assert "<script>" not in md and "\\<script\\>alert(1)\\</script\\>" in md
    assert "](javascript:" not in md.lower()
    assert "javascript:alert(1)" in md  # shown as text, not linked
    table = md.split("| ECO |", 1)[1].split("\n\n", 1)[0].splitlines()
    unescaped_pipes = {len(re.findall(r"(?<!\\)\|", line)) for line in table[1:]}
    assert unescaped_pipes == {8}  # 7 columns in every row: hostile text cannot add cells


# --------------------------------------------------------------------------- JSON
def test_json_round_trip(report):
    def no_constants(name):
        raise AssertionError(f"non-standard JSON constant {name}")

    text = to_json(report)
    data = json.loads(text, parse_constant=no_constants)
    assert datetime.fromisoformat(data["generated_at"]) == report.generated_at
    assert data["date_from"] == "2024-01-03T18:02:00+00:00"
    assert data["username"] == "TesterBob" and data["n_games"] == 1234
    weakness = data["weaknesses"][0]
    assert weakness["priority"] == pytest.approx(0.7 * 0.82)
    assert weakness["example_games"][0].startswith("https://")
    engine = next(m for m in data["modules"] if m["key"] == "engine_stats")
    assert engine["stats"] == {"error": "RuntimeError: Stockfish crashed", "nan": None, "inf": None}
    time_table = next(m for m in data["modules"] if m["key"] == "time_mgmt")["tables"][0]
    assert time_table["rows"][0][4] == "2024-05-02" and time_table["rows"][2][2] is None
    assert data["study_plan"][0]["actions"][0].startswith("Replay")
    assert to_dict(report) == data


def test_json_handles_numpy_and_pandas_values():
    import numpy as np  # pandas dependency; only used to build test inputs

    rep = empty_report()
    rep.modules[0].stats = {
        "np_int": np.int64(3),
        "np_float": np.float64(0.25),
        "np_nan": np.float64("nan"),
        "array": np.array([1.0, np.nan]),
        "ts": pd.Timestamp("2024-06-01T18:00:00Z"),
        "nat": pd.NaT,
        "na": pd.NA,
        "series": pd.Series([1, 2], index=["a", "b"]),
        "frame": pd.DataFrame({"x": [1.5, None]}),
        (1, "tuple"): {3, 1, 2},
        date(2024, 1, 1): timedelta(minutes=2),
        "path": Path("/tmp/x"),
    }
    stats = json.loads(to_json(rep))["modules"][0]["stats"]
    assert stats["np_int"] == 3 and stats["np_float"] == 0.25 and stats["np_nan"] is None
    assert stats["array"] == [1.0, None]
    assert stats["ts"] == "2024-06-01T18:00:00+00:00" and stats["nat"] is None and stats["na"] is None
    assert stats["series"] == {"a": 1, "b": 2}
    assert stats["frame"] == [{"x": 1.5}, {"x": None}]
    assert stats["1|tuple"] == [1, 2, 3] and stats["2024-01-01"] == 120.0 and stats["path"] == "/tmp/x"


# --------------------------------------------------------------------------- write_report
def test_write_report_writes_all_formats(report, tmp_path):
    paths = write_report(report, tmp_path / "nested" / "dir" / "testerbob")
    assert [p.name for p in paths] == ["testerbob.html", "testerbob.md", "testerbob.json"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert paths[0].read_text(encoding="utf-8").startswith("<!doctype html>")
    assert paths[1].read_text(encoding="utf-8").startswith("# TesterBob")
    assert json.loads(paths[2].read_text(encoding="utf-8"))["username"] == "TesterBob"


def test_write_report_strips_suffix_and_filters_formats(report, tmp_path):
    paths = write_report(report, str(tmp_path / "report.html"), formats=["JSON", "markdown", "json"])
    assert [p.name for p in paths] == ["report.json", "report.md"]
    assert not (tmp_path / "report.html").exists()
    dotted = write_report(report, tmp_path / "john.doe", formats=("html",))
    assert dotted == [tmp_path / "john.doe.html"]
    with pytest.raises(ValueError, match="unknown report format"):
        write_report(report, tmp_path / "x", formats=("pdf",))
    assert not (tmp_path / "x.html").exists()


# --------------------------------------------------------------------------- review fixes (regressions)
def _board_svg(fen: str = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3") -> str:
    import chess
    import chess.svg

    board = chess.Board(fen)
    arrows = [chess.svg.Arrow(chess.A7, chess.A6, color="red"), chess.svg.Arrow(chess.G8, chess.F6, color="green")]
    return chess.svg.board(board, orientation=chess.BLACK, arrows=arrows, size=280, check=chess.E1)


def _diagram_report(*diagrams) -> Report:
    from chess_insights.models import Diagram

    rep = empty_report()
    rep.modules[0].diagrams = [d if isinstance(d, Diagram) else Diagram(*d) for d in diagrams]
    return rep


BOARD_TAGS = {"svg", "defs", "g", "path", "rect", "circle", "use", "line", "polygon", "radialgradient", "stop", "text", "title"}


def _board_parts(html: str) -> list[str]:
    return re.findall(r'<div class="board-svg"[^>]*>(.*?)</div>', html, re.S)


@pytest.mark.parametrize(
    "svg",
    [
        '<svg onload="alert(1)"><rect width="8" height="8"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject><iframe srcdoc="&lt;script&gt;alert(1)&lt;/script&gt;"></iframe></foreignObject></svg>',
        '<svg><image href="https://evil.example/pixel.png" width="1" height="1"/></svg>',
        '<svg><a href="jav&#x61;script:alert(1)"><rect width="8" height="8"/></a></svg>',
        '<svg><rect width="8" height="8" style="fill:url(https://evil.example/a.svg#x)"/></svg>',
        '<svg><desc><![CDATA[x><img src=x onerror=alert(1)>]]></desc></svg>',
        '<svg><set attributeName="onmouseover" to="alert(1)"/></svg>',
        '<svg><use href="https://evil.example/sprite.svg#a"/></svg>',
        '<svg><g><rect width="8" height="8" onclick="alert(1)"/></g></svg>',
        '<svg/><img src=x onerror=alert(1)>',
    ],
)
def test_diagram_svg_is_sanitised_by_allowlist(svg):
    fen = "8/8/8/4k3/8/8/4K3/8 w - - 0 1"
    html = render_html(_diagram_report(("Hostile", fen, svg, "caption", GAME.format(1))))
    checker = check_html(html)
    lowered = html.lower()
    for needle in ("onload", "onerror", "onclick", "onmouseover", "foreignobject", "iframe", "evil.example", "<img", "<set", "<a href=\"jav"):
        assert needle not in lowered, needle
    assert not checker.srcs
    (board,) = _board_parts(html)
    tags = set(re.findall(r"<([a-zA-Z]+)", board))
    assert {t.lower() for t in tags} <= BOARD_TAGS, tags
    # a rejected SVG falls back to a board drawn from the FEN by python-chess
    assert "<use" in board and "white-king" in board and "black-king" in board


def test_diagram_svg_from_python_chess_is_kept_with_unique_ids():
    html = render_html(_diagram_report(("First", "x", _board_svg(), "c1", ""), ("Second", "x", _board_svg(), "c2", "")))
    checker = check_html(html)
    boards = _board_parts(html)
    assert len(boards) == 2
    for board in boards:
        assert board.count('class="arrow"') == 4  # both arrows (shaft + head) survive
        assert "radialGradient" in board  # check highlight
    assert len(checker.ids) == len(set(checker.ids)), [i for i in checker.ids if checker.ids.count(i) > 1][:5]
    refs = set(re.findall(r'href="#([^"]+)"', html)) | set(re.findall(r"url\(#([^)]+)\)", html))
    assert refs and refs <= set(checker.ids)


def test_diagram_without_usable_svg_or_fen_shows_text_only():
    html = render_html(_diagram_report(("Broken", "not a fen", "<svg onload=alert(1)></svg>", "caption", "")))
    check_html(html)
    assert "onload" not in html and ">Broken<" in html


RESULT_MIX = [
    "Won by checkmate",
    "Won by resignation",
    "Won on time",
    "Won: abandoned / other",
    "Drawn",
    "Lost: abandoned / other",
    "Lost on time",
    "Lost by resignation",
    "Lost by checkmate",
]


def _legend_items(html: str) -> list[tuple[str, str]]:
    legend = re.search(r'<ul class="legend"[^>]*>(.*?)</ul>', html, re.S).group(1)
    return re.findall(r'<span class="key (?:key--line )?(sw[\w-]+)" aria-hidden="true"></span>([^<]*)</li>', legend)


def test_result_mix_chart_uses_semantic_colours_and_never_folds_results():
    chart = Chart(
        kind="stacked_bar",
        title="How your games end",
        labels=["Blitz", "Rapid"],
        series=[Series(name, [0.1, 0.12]) for name in RESULT_MIX],
        value_format="pct",
    )
    rep = empty_report()
    rep.modules[0].charts = [chart]
    html = render_html(rep)
    items = _legend_items(html)
    assert [name for _, name in items] == RESULT_MIX  # nothing folded into "Other"
    swatch = dict((name, cls) for cls, name in items)
    assert swatch["Drawn"] == "sw-d"
    wins = {swatch[n] for n in RESULT_MIX[:4]}
    losses = {swatch[n] for n in RESULT_MIX[5:]}
    assert wins == {"sw-w1", "sw-w2", "sw-w3", "sw-w4"} and losses == {"sw-l1", "sw-l2", "sw-l3", "sw-l4"}
    assert swatch["Won by checkmate"] == "sw-w1" and swatch["Lost by checkmate"] == "sw-l1"  # most decisive = strongest
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    for token in ("--s-w1:", "--s-w4:", "--s-d:", "--s-l1:", "--s-l4:"):
        assert style.count(token) == 3, token  # light + both dark blocks
    svg = re.search(r"<svg .*?</svg>", html, re.S).group(0)
    assert 'class="bar f-w1"' in svg and 'class="bar f-l1"' in svg and 'class="bar f-d"' in svg


def test_result_series_are_put_in_win_draw_loss_order():
    chart = Chart(
        kind="stacked_bar",
        title="Results by session length",
        labels=["1 game", "2+ games"],
        series=[Series("Losses", [3, 4]), Series("Wins", [5, 6]), Series("Draws", [1, 0])],
        value_format="int",
    )
    rep = empty_report()
    rep.modules[0].charts = [chart]
    html = render_html(rep)
    assert _legend_items(html) == [("sw-w1", "Wins"), ("sw-d", "Draws"), ("sw-l1", "Losses")]
    assert chart_table(chart).columns == ["", "Wins", "Draws", "Losses"]
    assert "| 1 game | 5 | 1 | 3 |" in render_markdown(rep)


def test_series_colour_follows_identity_not_position():
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind="line", title="a", labels=["1", "2"], series=[Series("Bullet", [1, 2]), Series("Blitz", [1, 2]), Series("Rapid", [2, 3])]),
        Chart(kind="line", title="b", labels=["1", "2"], series=[Series("Blitz", [1, 2]), Series("Rapid", [2, 3])]),
    ]
    html = render_html(rep)
    legends = re.findall(r'<ul class="legend"[^>]*>(.*?)</ul>', html, re.S)
    colours = [dict((n, c) for c, n in re.findall(r'class="key key--line (sw[\w-]+)"[^>]*></span>([^<]*)<', lg)) for lg in legends]
    assert colours[0]["Blitz"] == colours[1]["Blitz"] and colours[0]["Rapid"] == colours[1]["Rapid"]
    assert len(set(colours[1].values())) == 2


@pytest.mark.parametrize("kind", ["bar", "hbar", "line"])
def test_zero_reference_is_the_zero_baseline_not_a_second_line(kind):
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind=kind, title="Score vs expected", labels=["a", "b", "c"], series=[Series("Score − expected", [0.05, -0.04, 0.02])], value_format="signed_pct", reference=0.0)
    ]
    html = render_html(rep)
    svg = re.search(r"<svg .*?</svg>", html, re.S).group(0)
    assert 'class="ref"' not in svg and "Reference 0%" not in html
    assert '<ul class="legend"' not in html  # a single series with no other reference needs no legend
    assert svg.count('class="zero"') == 1
    assert "Reference line" not in render_markdown(rep)


def _x_label_boxes(svg: str) -> list[tuple[float, float]]:
    import html as _h

    boxes = []
    for m in re.finditer(r'<text class="t-x" x="([\d.]+)" y="[\d.]+"[^>]*text-anchor="middle">([^<]*)</text>', svg):
        width = len(_h.unescape(m.group(2))) * 13 * 0.58
        boxes.append((float(m.group(1)) - width / 2, float(m.group(1)) + width / 2))
    return boxes


@pytest.mark.parametrize("n", [6, 11, 16, 25, 39])
def test_line_chart_month_labels_never_overlap(n):
    labels = [f"{2020 + i // 12}-{i % 12 + 1:02d}" for i in range(n)]
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind="line", title="Rating by month", labels=labels, series=[Series("Blitz", [1500 + i for i in range(n)]), Series("Rapid", [1600 - i for i in range(n)])], value_format="rating")
    ]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    boxes = _x_label_boxes(svg)
    assert len(boxes) >= 2 and boxes[-1][1] <= 400 and boxes[0][0] >= 0
    assert all(a[1] + 2 <= b[0] for a, b in zip(boxes, boxes[1:])), boxes
    assert re.search(rf">{labels[-1]}</text>", svg)  # the most recent month is always labelled


@pytest.mark.parametrize("n", [8, 12, 13, 16])
def test_bar_value_labels_never_collide(n):
    rep = empty_report()
    rep.modules[0].charts = [Chart(kind="bar", title="t", labels=[str(i) for i in range(n)], series=[Series("x", [-0.123] * n)], value_format="signed_pct")]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    caps = [float(x) for x in re.findall(r'<text class="t-val" x="([\d.]+)"', svg)]
    width = len(f"{MINUS}12%") * 12 * 0.58
    assert all(b - a >= width + 2 for a, b in zip(caps, caps[1:]))


@pytest.mark.parametrize(
    ("value", "fmt", "expected"),
    [
        (0.0004, "pct", "<0.1%"),  # a real but tiny rate is not "0%"
        (0.00049, "pct", "<0.1%"),
        (0.0005, "pct", "0.1%"),
        (0.0996, "pct", "10%"),  # rounds up to 10%: no stray decimal
        (-0.0996, "pct", f"{MINUS}10%"),
        (0.0996, "signed_pct", "+10%"),
        (-0.0, "pct", "0%"),
        (-0.0, "float1", "0.0"),
        (-0.04, "float1", "0.0"),
        (-0.4, "signed_int", "0"),
        (90061, "seconds", "25:01:01"),
    ],
)
def test_format_value_edge_cases(value, fmt, expected):
    assert format_value(value, fmt) == expected


@pytest.mark.parametrize("fmt", ["int", "float1", "float2", "signed_int", "rating", "pct", "seconds"])
def test_huge_numbers_stay_short(fmt):
    for x in (1e300, -1e300, 1.5e18):
        text = format_value(x, fmt)
        assert len(text) <= 16 and "e" in text.lower(), (fmt, text)
    assert format_value(123456789012, "int") == "123,456,789,012"  # ordinary big numbers keep grouping


def test_huge_values_do_not_break_the_chart():
    rep = empty_report()
    rep.modules[0].charts = [Chart(kind="bar", title="huge", labels=["a", "b"], series=[Series("x", [1e300, 2e300])])]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    xs = [float(x) for x in re.findall(r'<path class="bar[^"]*" d="M([\d.]+),', svg)]
    assert len(xs) == 2 and all(0 < x < 400 for x in xs)


def test_markdown_code_spans_links_and_block_markers():
    from chess_insights.models import Diagram
    from chess_insights.report.markdown import md_link, md_text

    rep = empty_report()
    rep.modules[0].diagrams = [Diagram("Pos", "8/8/8/8/8/8/8/8 w - - 0 1 `**x**` <b>", "", "c", "")]
    md = render_markdown(rep)
    line = next(l for l in md.splitlines() if "Pos" in l)
    spans = re.findall(r"(`+)(.+?)\1", line)
    assert spans and all("\\" not in body for _, body in spans)  # no backslash escapes inside code spans
    assert "<b>" in spans[-1][1] and line.count("`") % 2 == 0
    # a backslash in a URL cannot escape the closing parenthesis of the link
    link = md_link("https://example.org/a\\", "x")
    assert link.endswith(")") and not link.endswith("\\)")
    # text that would become a heading closer, a thematic break or maths is escaped
    assert md_text("---") != "---" and md_text("***").count("\\") == 3
    assert md_text("Game #1 #").endswith("\\#")
    assert md_text("$x$") == "\\$x\\$"
    # a leading move number is escaped at the dot: a backslash before a digit would be shown literally
    assert md_text("1. e4 e5 2. Nf3") == "1\\. e4 e5 2. Nf3" and md_text("12) x") == "12\\) x"
    assert md_text("+ 5 points").startswith("\\+") and md_text("- x").startswith("\\-") and md_text("===") != "==="
    assert md_text("+5 points") == "+5 points" and md_text("-3 = bad") == "-3 = bad"  # not block markers


def test_json_never_crashes_on_signalling_nan_decimal():
    from decimal import Decimal

    rep = empty_report()
    rep.modules[0].stats = {"snan": Decimal("sNaN"), "nan": Decimal("NaN"), "x": Decimal("1.5")}
    stats = json.loads(to_json(rep))["modules"][0]["stats"]
    assert stats == {"snan": None, "nan": None, "x": 1.5}


# Advance widths in em of DejaVu Sans, the widest common fallback for system-ui (Linux Chromium,
# measured); the IBM Plex / San Francisco / Roboto faces are narrower, so fitting these fits them too.
DEJAVU_EM = {"−20%": 3.058, "+10%": 3.058, "100%": 2.858, "+5.0%": 3.378, "much lower (< −150)": 10.844, "2020-05": 4.177, "Won by checkmate": 9.62}


@pytest.mark.parametrize("text", list(DEJAVU_EM))
def test_label_width_estimate_is_not_below_real_fallback_font(text):
    from chess_insights.report.html import _tw

    assert _tw(text, 100) >= DEJAVU_EM[text] * 100 - 1


def test_chart_labels_stay_inside_the_svg():
    """Tick labels are right-aligned in the left margin, value labels sit right of hbars: both need real widths."""
    from chess_insights.report.html import _tw

    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind="bar", title="signed", labels=["Bullet", "Blitz", "Rapid"], series=[Series("Δ", [-0.18, 0.04, 0.61])], value_format="signed_pct"),
        Chart(kind="hbar", title="openings", labels=["Italian Game", "Sicilian Defense"], series=[Series("Δ", [0.1, -0.05])], value_format="signed_pct"),
    ]
    svgs = re.findall(r"<svg .*?</svg>", render_html(rep), re.S)
    for svg in svgs:
        for x, anchor, txt in re.findall(r'<text class="t-(?:tick|val)" x="([\d.-]+)"[^>]*?(?:text-anchor="(\w+)")?[^>]*>([^<]*)</text>', svg):
            import html as _h

            w, x = _tw(_h.unescape(txt), 12), float(x)
            left, right = {"end": (x - w, x), "middle": (x - w / 2, x + w / 2)}.get(anchor, (x, x + w))
            assert left >= -0.5 and right <= 400.5, (txt, left, right)


def _css_tokens(style: str, block: str) -> dict[str, str]:
    body = re.search(block, style).group(1)
    return dict(re.findall(r"--([\w-]+):(#[0-9A-Fa-f]{6})", body))


def _contrast(a: str, b: str) -> float:
    def lum(h: str) -> float:
        r, g, b_ = (int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b_)

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_colours_meet_wcag_aa_in_both_themes():
    style = re.search(r"<style>(.*?)</style>", render_html(empty_report()), re.S).group(1)
    light = _css_tokens(style, r"^:root\{(.*?)\}")
    dark = _css_tokens(style, r':root\[data-theme="dark"\]\{(.*?)\}')
    for theme in (light, dark):
        for fg in ("ink", "ink-2", "muted", "accent"):
            for bg in ("paper", "surface"):
                assert _contrast(theme[fg], theme[bg]) >= 4.5, (fg, bg, theme[fg], theme[bg])
        for fg, bg in (("good", "good-wash"), ("bad", "bad-wash"), ("neutral", "neutral-wash"), ("tip-ink", "tip-bg")):
            assert _contrast(theme[fg], theme[bg]) >= 4.5, (fg, bg)


def test_column_labels_wrap_upright_before_rotating():
    rep = empty_report()
    labels = ["much lower (< −150)", "lower (−150 to −50)", "similar (±50)", "higher (+50 to +150)", "much higher (> +150)"]
    rep.modules[0].charts = [Chart(kind="bar", title="t", labels=labels, series=[Series("Δ", [0.02, -0.03, 0.01, 0.05, None])], value_format="signed_pct")]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    assert "rotate(" not in svg
    import html as _h

    shown = " ".join(_h.unescape(t) for t in re.findall(r'<tspan class="t-x"[^>]*>([^<]*)</tspan>', svg))
    assert all(word in shown.split() for label in labels for word in label.split())  # nothing ellipsized


def test_long_labels_turn_columns_into_horizontal_bars():
    labels = ["Within 15 min of a loss", "Within 15 min of a win", "Within 15 min of a draw", "15–30 min after the previous game", "After a break (> 30 min) or first game"]
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(kind="bar", title="after", labels=labels, series=[Series("Score − expected", [-0.04, 0.02, 0.0, 0.01, -0.01])], value_format="signed_pct"),
        Chart(kind="stacked_bar", title="stacked", labels=labels, series=[Series("Wins", [1, 2, 3, 4, 5]), Series("Losses", [1, 2, 3, 4, 5])]),
    ]
    html = render_html(rep)
    svgs = re.findall(r"<svg .*?</svg>", html, re.S)
    import html as _h

    shown = _h.unescape(re.sub(r"</tspan><tspan[^>]*>", " ", svgs[0]))
    assert "rotate(" not in svgs[0] and all(label in shown for label in labels)
    assert 'data-kind="bar"' in html
    assert "rotate(-40" in svgs[1]  # no horizontal stacked bars: long labels still rotate there


def test_month_axis_is_drawn_to_scale_and_bridges_gaps():
    """Monthly labels skip inactive months: x must follow the calendar, and a gap must look like one."""
    rep = empty_report()
    labels = ["2017-12", "2018-02", "2021-07", "2021-08", "2021-09"]
    rep.modules[0].charts = [Chart(kind="line", title="Rating by month", labels=labels, series=[Series("Blitz", [760, 780, 1260, 1200, 1310])], value_format="rating")]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    xs = sorted({float(x) for x in re.findall(r'<line class="xh" x1="([\d.]+)"', svg)})
    months = [0, 2, 43, 44, 45]
    step = (xs[-1] - xs[0]) / 45
    assert all(abs(x - (xs[0] + m * step)) < 0.2 for x, m in zip(xs, months)), xs
    solid = re.search(r'<path class="ln k1" d="([^"]*)"', svg).group(1)
    assert solid.count("M") == 2  # 2017-12..2018-02 | 2021-07..09: no solid line across the 3-year break
    assert re.search(r'<path class="ln ln--gap k1" d="M[^"]+"', svg)  # a dashed bridge shows the gap


def test_category_line_bridges_missing_values_with_a_dashed_connector():
    rep = empty_report()
    rep.modules[0].charts = [Chart(kind="line", title="t", labels=list("abcde"), series=[Series("x", [1, 2, None, 4, 5])])]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    assert re.search(r'<path class="ln k1" d="M[^"]* M', svg)
    gap = re.search(r'<path class="ln ln--gap k1" d="([^"]*)"', svg).group(1)
    assert gap.count("M") == 1 and gap.count("L") == 1


def test_wide_tables_show_they_scroll_and_keep_row_labels_readable():
    style = re.search(r"<style>(.*?)</style>", render_html(empty_report()), re.S).group(1)
    wrap = re.search(r"(?:^|\n)\.table-wrap\{([^}]*)\}", style).group(1)
    assert "overflow-x:auto" in wrap and wrap.count(" local") == 2 and wrap.count(" scroll") == 2  # scroll shadows
    assert ".ci-chart .table-wrap{--wrap-bg:var(--surface)}" in style  # covers match the chart card
    assert re.search(r"\.ci td:first-child:not\(\.num\)\{min-width:1[2-4]ch\}", style)
    assert style.count("--scroll-shadow:") == 3


def test_line_chart_labels_the_first_and_last_point():
    labels = ["2017-12", "2018-02", "2021-07", "2022-09", "2023-06", "2023-08", "2023-09", "2023-10"]
    rep = empty_report()
    rep.modules[0].charts = [Chart(kind="line", title="r", labels=labels, series=[Series("Blitz", [700 + 50 * i for i in range(8)]), Series("Daily", [1300 - 40 * i for i in range(8)])], value_format="rating")]
    svg = re.search(r"<svg .*?</svg>", render_html(rep), re.S).group(0)
    shown = re.findall(r'<text class="t-x"[^>]*>([^<]*)</text>', svg)
    assert shown[0] == "2017-12" and shown[-1] == "2023-10"
    boxes = _x_label_boxes(svg)
    assert all(a[1] + 2 <= b[0] for a, b in zip(boxes, boxes[1:]))


# --------------------------------------------------------------------------- second review round (regressions)
# OKLab distance x100 under normal vision and under protanopia / deuteranopia (Machado, Oliveira &
# Fernandes 2009, severity 1): the measure and floors the chart palette was validated with.
_MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
}


def _oklab(hex_colour: str, cvd: str | None = None) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rgb = [lin(int(hex_colour[i : i + 2], 16) / 255) for i in (1, 3, 5)]
    if cvd:
        rgb = [min(1.0, max(0.0, sum(m * c for m, c in zip(row, rgb)))) for row in _MACHADO[cvd]]
    r, g, b = rgb
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _too_alike(a: str, b: str) -> bool:
    """Below the palette floors: normal-vision dE < 15, or colour-blind dE < 6."""
    normal = 100 * math.dist(_oklab(a), _oklab(b))
    cvd = min(100 * math.dist(_oklab(a, k), _oklab(b, k)) for k in _MACHADO)
    return normal < 15 or cvd < 6


@pytest.fixture(scope="module")
def themes() -> list[dict[str, str]]:
    style = re.search(r"<style>(.*?)</style>", render_html(empty_report()), re.S).group(1)
    return [_css_tokens(style, r"^:root\{(.*?)\}"), _css_tokens(style, r':root\[data-theme="dark"\]\{(.*?)\}')]


def _slot_colour(theme: dict[str, str], slot: str) -> str:
    return theme[f"s{slot}" if slot.isdigit() else f"s-{slot}"]


def _clash(themes, a: str, b: str) -> bool:
    return any(_too_alike(_slot_colour(t, a), _slot_colour(t, b)) for t in themes)


def test_time_class_colours_clear_the_palette_floors(themes):
    import itertools

    from chess_insights.report.html import _CLASHING_SLOTS, series_slots

    measured = {frozenset(p) for p in itertools.combinations("1234", 2) if _clash(themes, *p)}
    assert measured == set(_CLASHING_SLOTS)  # the table the renderer uses matches the palette
    classes = ["Bullet", "Blitz", "Rapid", "Daily"]
    for k in (2, 3):
        for names in itertools.permutations(classes, k):
            slots = series_slots(list(names))
            assert not any(_clash(themes, a, b) for a, b in itertools.combinations(slots, 2)), (names, slots)
    # brengall99's real rating chart: Blitz and Daily were orange and yellow; blitz keeps one colour everywhere
    assert series_slots(["Blitz", "Daily"]) == ["1", "4"] and series_slots(["Blitz", "Rapid", "Daily"]) == ["1", "3", "4"]
    assert series_slots(["Bullet", "Blitz", "Rapid", "Daily"]) == ["2", "1", "3", "4"]  # all four: slots 1-4


def test_result_colours_never_clash_across_roles(themes):
    import itertools

    from chess_insights.report.html import _RESULT_CLASH, series_slots

    steps = [f"w{k}" for k in range(1, 5)] + ["d"] + [f"l{k}" for k in range(1, 5)]
    measured = {frozenset((a, b)) for a, b in itertools.combinations(steps, 2) if a[0] != b[0] and _clash(themes, a, b)}
    assert measured == set(_RESULT_CLASH)
    wins = ["Won by checkmate", "Won by resignation", "Won on time", "Won: abandoned / other"]
    losses = ["Lost: abandoned / other", "Lost on time", "Lost by resignation", "Lost by checkmate"]
    unavoidable = {(3, False, 4), (4, False, 4)}  # w3/w4 clash with l4, and 4 losses must end on l4
    for n_win, draw, n_loss in itertools.product(range(5), (True, False), range(5)):
        names = wins[:n_win] + (["Drawn"] if draw else []) + losses[4 - n_loss :]
        if len({n_win > 0, draw, n_loss > 0} - {False}) < 2 and not (draw and (n_win or n_loss)):
            continue
        slots = series_slots(names)
        assert len(slots) == len(names)
        if n_win:
            assert slots[0] == "w1"  # the most decisive result is always the strongest colour
        if n_loss:
            assert slots[-1] == "l1"
        touching = [(a, b) for a, b in zip(slots, slots[1:]) if a[0] != b[0]]
        clashes = [(a, b) for a, b in touching if _clash(themes, a, b)]
        assert not clashes or (n_win, draw, n_loss) in unavoidable, (names, slots, clashes)
    # the verifier's case: two win types, the draw and two loss types
    assert series_slots(wins[:1] + wins[2:3] + ["Drawn"] + losses[2:]) == ["w1", "w4", "d", "l4", "l1"]


BRENGALL_MONTHS = ["2017-12", "2018-02", "2021-07", "2022-09", "2023-06", "2023-08", "2023-09", "2023-10"]


def test_monthly_lines_run_on_across_short_breaks_and_bridges_stay_visible():
    """brengall99's real rating chart: 8 months with games over six years."""
    rep = empty_report()
    rep.modules[0].charts = [
        Chart(
            kind="line",
            title="Rating by month",
            labels=BRENGALL_MONTHS,
            series=[
                Series("Blitz", [762.0, 782.0, 1260.0, 1195.0, 1316.0, 1321.0, 1337.0, 1372.0]),
                Series("Daily", [1297.0, 946.0, 1016.0, None, 978.0, 985.0, 987.0, 972.0]),
            ],
            value_format="rating",
        )
    ]
    html = render_html(rep)
    svg = re.search(r"<svg .*?</svg>", html, re.S).group(0)
    blitz = re.search(r'<path class="ln k1" d="([^"]*)"', svg).group(1).split("M")[1:]
    daily = re.search(r'<path class="ln k4" d="([^"]*)"', svg).group(1).split("M")[1:]
    # 2017-12..2018-02 and 2023-06..2023-10 are solid; only breaks of 3+ months without games are bridged
    assert [seg.count("L") + 1 for seg in blitz] == [2, 1, 1, 4]
    assert [seg.count("L") + 1 for seg in daily] == [2, 1, 4]
    assert re.search(r'<path class="ln ln--gap k1" d="([^"]*)"', svg).group(1).count("M") == 3
    assert re.search(r'<path class="ln ln--gap k4" d="([^"]*)"', svg).group(1).count("M") == 2
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    gap = re.search(r"\.ci-chart \.ln--gap\{([^}]*)\}", style).group(1)
    assert "opacity" not in gap  # full series colour: no faint bridges
    dash, space = (float(v) for v in re.search(r"stroke-dasharray:([\d.]+) ([\d.]+)", gap).groups())
    assert dash >= 4 and dash / (dash + space) >= 0.5


def test_numpy_and_python_durations_are_seconds_everywhere():
    import numpy as np
    from chess_insights.report.html import is_missing

    rep = empty_report()
    rep.modules[0].stats = {
        "td": np.timedelta64(5, "s"),
        "ms": np.timedelta64(1500, "ms"),
        "nat": np.timedelta64("NaT"),
        "months": np.timedelta64(2, "M"),
        "dt": np.datetime64("2024-01-01T10:00"),
        "dt_ns": np.datetime64("2024-01-01T00:00:00.000000001"),
        "dt_nat": np.datetime64("NaT"),
        "median": np.median(np.array([3, 5, 9], dtype="timedelta64[s]")),
    }
    stats = json.loads(to_json(rep))["modules"][0]["stats"]
    assert stats == {
        "td": 5.0,
        "ms": 1.5,
        "nat": None,
        "months": "2 months",
        "dt": "2024-01-01T10:00:00",
        "dt_ns": "2024-01-01T00:00:00",
        "dt_nat": None,
        "median": 5.0,
    }
    assert format_value(np.timedelta64(90, "s")) == "1:30" and format_value(np.timedelta64(5, "s"), "seconds") == "5.0s"
    assert format_value(timedelta(seconds=75)) == "1:15" and format_value(timedelta(seconds=75), "text") == "1:15"
    assert is_missing(np.timedelta64("NaT")) and format_value(np.timedelta64("NaT"), "int") == MISSING
    assert format_value(np.datetime64("2024-05-02T18:30")) == "2024-05-02 18:30"
    rep.modules[0].kpis = [Kpi("Think time", np.timedelta64(95, "s")), Kpi("None", np.timedelta64("NaT"), "seconds")]
    html = render_html(rep)
    assert ">1:35<" in html and "seconds<" not in html and "NaT" not in html
    assert "- Think time: **1:35**" in render_markdown(rep)


def test_board_transform_check_runs_in_linear_time():
    import time

    from chess_insights.report.html import _SVG_TRANSFORM, sanitize_board_svg

    for text in ("translate(10, 20)", "rotate(45 22.5 22.5) scale(0.5)", "matrix(1 0 0 1 0 0),translate(-1e1)", " scale(2) "):
        assert _SVG_TRANSFORM.match(text), text
    for text in ("scale(1) url(x)", "translate(1)x", "scale(1)" * 9, "expression(alert(1))"):
        assert not _SVG_TRANSFORM.match(text), text
    start = time.perf_counter()
    for k in (18, 30, 200):
        assert not _SVG_TRANSFORM.match("scale(1)  " * k + "!")
        svg = f'<svg xmlns="http://www.w3.org/2000/svg"><g transform="{"scale(1)  " * k}!"><rect width="1" height="1"/></g></svg>'
        assert sanitize_board_svg(svg) == ""
    assert time.perf_counter() - start < 0.5  # was 78 s for k=18


def test_write_report_survives_bad_text_and_never_truncates(report, tmp_path, monkeypatch):
    import copy

    from chess_insights import report as report_pkg

    good = write_report(report, tmp_path / "bob")
    before = {p: p.read_bytes() for p in good}
    # a renderer error leaves every earlier file exactly as it was, and no temporary files behind
    def broken(_report):
        raise TypeError("boom")

    monkeypatch.setitem(report_pkg.FORMATS, "json", ("json", broken))
    with pytest.raises(TypeError, match="boom"):
        write_report(report, tmp_path / "bob")
    assert {p: p.read_bytes() for p in good} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bob.html", "bob.json", "bob.md"]
    monkeypatch.undo()
    # a lone surrogate (broken input) is written as U+FFFD instead of crashing mid-write
    bad = copy.deepcopy(report)
    bad.modules[0].summary = json.loads('"bad \\ud800 surrogate"')
    paths = write_report(bad, tmp_path / "bob")
    for p in paths:
        text = p.read_bytes().decode("utf-8")
        assert "bad � surrogate" in text or "bad \\ufffd surrogate" in text
    assert json.loads(paths[2].read_text(encoding="utf-8"))["modules"][0]["summary"] == "bad � surrogate"


def test_write_report_into_a_folder(report, tmp_path):
    folder = tmp_path / "Desktop"
    folder.mkdir()
    assert write_report(report, folder, ["md"]) == [folder / "testerbob.md"]
    assert write_report(report, str(folder) + "/", ["html"]) == [folder / "testerbob.html"]
    new = tmp_path / "new-folder"
    assert write_report(empty_report(), f"{new}/", ["json"]) == [new / "chess-insights.json"]  # trailing separator: a folder
    assert not (tmp_path / "Desktop.md").exists()
    assert write_report(report, tmp_path / "plain", ["md"]) == [tmp_path / "plain.md"]  # a stem is still a stem


def _tick_boxes(svg: str) -> list[tuple[float, float]]:
    import html as _h

    from chess_insights.report.html import _tw

    boxes = []
    for x, txt in re.findall(r'<text class="t-tick" x="([\d.-]+)" y="[\d.]+" text-anchor="middle">([^<]*)</text>', svg):
        w = _tw(_h.unescape(txt), 12)
        boxes.append((float(x) - w / 2, float(x) + w / 2))
    return boxes


@pytest.mark.parametrize("scale", [1, 25, 1000, 25_000, 100_000, 3_000_000])
@pytest.mark.parametrize("fmt", [None, "int", "float2", "signed_int", "seconds", "pct"])
def test_horizontal_bar_ticks_never_overlap(scale, fmt):
    rep = empty_report()
    labels = [f"Row {i}" for i in range(12)]
    series = [Series(f"s{j}", [scale * ((i * 7 + j * 3) % 11) / 10 for i in range(12)]) for j in range(3)]
    rep.modules[0].charts = [
        Chart(kind="hbar", title="hbar", labels=labels, series=series, value_format=fmt),
        Chart(kind="bar", title="grouped", labels=[f"A rather long category name {i}" for i in range(6)], series=series, value_format=fmt),
    ]
    svgs = re.findall(r"<svg .*?</svg>", render_html(rep), re.S)
    assert len(svgs) == 2 and "rotate(" not in svgs[1]  # long labels: the grouped bars turned horizontal
    for svg in svgs:
        boxes = _tick_boxes(svg)
        assert len(boxes) >= 2
        assert all(a[1] + 4 <= b[0] for a, b in zip(boxes, boxes[1:])), boxes
        assert boxes[0][0] >= -0.5 and boxes[-1][1] <= 400.5


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1e300, "1e+300"), (-1e300, f"{MINUS}1e+300"), (1.5e18, "1.5e+18"), (-1e-9, "0"), (-0.00001, "0"), (-0.0, "0"),
     (-1.5, f"{MINUS}1.5"), (0.12345678, "0.1235"), (-7, f"{MINUS}7"), (123456789012345678, "123456789012345678")],
)
def test_text_format_numbers_are_short_and_never_negative_zero(value, expected):
    assert format_value(value, "text") == expected


def test_huge_text_values_stay_short_in_kpis_and_chart_labels():
    rep = empty_report()
    rep.modules[0].kpis = [Kpi("Huge", 1e300)]
    rep.modules[0].charts = [Chart(kind="bar", title="c", labels=[1e300, -1e-9], series=[Series("x", [1, 2])])]
    html = render_html(rep)
    assert "0" * 30 not in html and ">1e+300<" in html and ">-0<" not in html
    assert chart_table(rep.modules[0].charts[0]).rows[1][0] == "0"


def test_decimal_values_format_like_floats():
    from decimal import Decimal

    assert format_value(Decimal("0.5"), "pct") == "50%" and format_value(Decimal("-2.25"), "float1") == f"{MINUS}2.3"
    for bad in (Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")):
        assert format_value(bad, "pct") == MISSING and format_value(bad) == MISSING and format_value(bad, "text") == MISSING
    assert infer_format(Decimal("2")) == "int" and infer_format(Decimal("0.25")) == "float2"
    assert format_value(Decimal("0.25")) == "0.25" and format_value(Decimal("0.50"), "text") == "0.5"
    rep = empty_report()
    rep.modules[0].tables = [Table("t", ["a", "b"], [["x", Decimal("0.5")], ["y", Decimal("NaN")]], ["text", "pct"])]
    md = render_markdown(rep)
    assert "| x | 50% |" in md and f"| y | {MISSING} |" in md
    assert json.loads(to_json(rep))["modules"][0]["tables"][0]["rows"] == [["x", 0.5], ["y", None]]


def test_markdown_bare_urls_become_clean_links():
    from chess_insights.report.markdown import md_text

    def links(text: str) -> list[tuple[str, str]]:
        return re.findall(r"\[((?:\\.|[^\]\\])*)\]\(([^)\s]*)\)", text)

    out = md_text("See https://example.com/a_b_(c). Or www.example.com/_x_, then stop")
    (label1, href1), (label2, href2) = links(out)
    unescape = lambda s: re.sub(r"\\(.)", r"\1", s)  # noqa: E731
    assert unescape(label1) == "https://example.com/a_b_(c)" and href1 == "https://example.com/a_b_%28c%29"
    assert unescape(label2) == "www.example.com/_x" and href2 == "http://www.example.com/_x"
    assert "\\" not in href1 + href2  # no backslash ends up in a link target
    assert out.endswith("\\_, then stop") and out.startswith("See [")
    # never linked: other schemes, addresses glued to other text (no "![...](...)" image), unsafe URLs
    for text in ("javascript:alert(1)", "Wow!https://evil.example/x.png", "[x]https://e.com", "http://"):
        assert not links(md_text(text)), text
    # a pipe inside an address cannot add a table cell
    rep = empty_report()
    rep.modules[0].tables = [Table("t", ["a", "b"], [["https://e.com/a|b", "x"]], ["text", "text"])]
    row = next(l for l in render_markdown(rep).splitlines() if "e.com" in l)
    assert len(re.findall(r"(?<!\\)\|", row)) == 3 and "%7C" in row


def test_cut_labels_keep_their_distinguishing_end():
    import html as _h

    rep = empty_report()
    hbar_labels = [f"Sicilian Defense: Najdorf Variation, English Attack {i} ({'White' if i % 2 else 'Black'})" for i in range(40)]
    col_labels = [f"Queen's Gambit Declined: Exchange Variation, line {i}" for i in range(40)]
    rep.modules[0].charts = [
        Chart(kind="hbar", title="h", labels=hbar_labels, series=[Series("x", list(range(40)))]),
        Chart(kind="bar", title="c", labels=col_labels, series=[Series("x", list(range(40)))]),
    ]
    hsvg, csvg = re.findall(r"<svg .*?</svg>", render_html(rep), re.S)
    rows = [" ".join(_h.unescape(t) for t in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", g)) for g in re.findall(r'<text class="t-y"[^>]*>(.*?)</text>', hsvg)]
    assert len(rows) == 40 and len(set(rows)) == 40, rows[:3]
    assert all(r.endswith("(White)" if i % 2 else "(Black)") and "…" in r for i, r in enumerate(rows))
    shown = [_h.unescape(t) for t in re.findall(r'<text class="t-x"[^>]*transform="rotate[^>]*>([^<]*)</text>', csvg)]
    assert shown and len(set(shown)) == len(shown) and all(re.search(r"line \d+$", s) for s in shown), shown[:3]


def test_module_ids_never_repeat_page_ids():
    keys = ["results", "results-h", "study-h", "weaknesses-h", "Results", "strengths"]
    rep = empty_report()
    rep.modules = [ModuleResult(key=k, title=k, summary="s") for k in keys]
    checker = check_html(render_html(rep))
    assert len(checker.ids) == len(set(i.lower() for i in checker.ids)), sorted(checker.ids)
    html = render_html(rep)
    for target in re.findall(r'(?:href="#|aria-labelledby=")([^"]+)"', html):
        assert target in checker.ids, target


def test_result_mix_table_is_transposed_to_fit_a_phone():
    chart = Chart(
        kind="stacked_bar", title="How your games end", labels=["Bullet", "Blitz", "Daily"],
        series=[Series(name, [0.1, 0.12, 0.2]) for name in RESULT_MIX], value_format="pct",
    )
    table = chart_table(chart)
    assert table.columns == ["", "Bullet", "Blitz", "Daily"]
    assert [row[0] for row in table.rows] == RESULT_MIX and table.rows[0][1:] == [0.1, 0.12, 0.2]
    rep = empty_report()
    rep.modules[0].charts = [chart]
    md = render_markdown(rep)
    assert "|  | Bullet | Blitz | Daily |" in md and "| Won by checkmate | 10% | 12% | 20% |" in md
    # few series: one column per series, as before
    few = chart_table(Chart(kind="bar", title="t", labels=["a"], series=[Series(f"s{i}", [i]) for i in range(4)]))
    assert few.columns == ["", "s0", "s1", "s2", "s3"]


def test_values_beyond_the_labels_are_never_dropped():
    rep = empty_report()
    rep.modules[0].charts = [Chart(kind="bar", title="t", labels=["only one"], series=[Series("x", [1, 2, 3])])]
    html = render_html(rep)
    svg = re.search(r"<svg .*?</svg>", html, re.S).group(0)
    assert len(re.findall(r'<path class="bar ', svg)) == 3
    md = render_markdown(rep)
    assert "| only one | 1 |" in md and "| 2 | 2 |" in md and "| 3 | 3 |" in md


def _role(name: str) -> str:
    return name.split()[0][:3].lower()  # "won" / "dra" / "los"


def test_stacked_result_colours_check_the_segments_that_really_touch(themes):
    import itertools
    import random

    def render_stack(values: dict[str, list[float]], n: int) -> tuple[dict[str, str], list[list[str]]]:
        chart = Chart(kind="stacked_bar", title="How your games end", labels=[f"TC{i}" for i in range(n)],
                      series=[Series(k, v) for k, v in values.items()], value_format="pct")
        rep = empty_report()
        rep.modules[0].charts = [chart]
        legend = {name: cls.removeprefix("sw-") for cls, name in _legend_items(render_html(rep))}
        stacks = [[k for k in RESULT_MIX if values[k][i]] for i in range(n)]  # zero segments are not drawn
        return legend, stacks

    def clashes(slot: dict[str, str], stacks) -> list[tuple[str, str]]:
        return [(a, b) for st in stacks for a, b in zip(st, st[1:]) if _role(a) != _role(b) and _clash(themes, slot[a], slot[b])]

    # Caruana's real chart: no "Won on time" or "abandoned" games, so "Won by resignation" sits on "Drawn"
    real = dict(zip(RESULT_MIX, ([0, 0.11], [0.25, 0.56], [0, 0], [0, 0], [0.13, 0.33], [0, 0], [0.13, 0], [0.5, 0], [0, 0])))
    legend, stacks = render_stack(real, 2)
    assert list(legend) == [k for k in RESULT_MIX if any(real[k])]  # nothing to draw: not in the key
    assert not clashes(legend, stacks)
    assert legend["Won by checkmate"] == "w1" and legend["Won by resignation"] == "w4"

    rng = random.Random(7)
    for _ in range(150):
        values = {k: [rng.choice([0, 0, 0.1]) for _ in range(3)] for k in RESULT_MIX}
        if len({_role(k) for k in RESULT_MIX if any(values[k])}) < 2:
            continue
        legend, stacks = render_stack(values, 3)
        found = clashes(legend, stacks)
        if not found:
            continue
        # only acceptable when no increasing choice of steps per arm avoids every clash
        drawn = [k for k in RESULT_MIX if any(values[k])]
        wins = [k for k in drawn if _role(k) == "won"]
        losses = [k for k in drawn if _role(k) == "los"]
        for w, lo in itertools.product(itertools.combinations(range(1, 5), len(wins)), itertools.combinations(range(1, 5), len(losses))):
            slot = {"Drawn": "d", **{k: f"w{s}" for k, s in zip(wins, w)}, **{k: f"l{s}" for k, s in zip(losses, reversed(lo))}}
            assert clashes(slot, stacks), (values, legend, slot)
