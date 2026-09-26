"""Study-plan targets against the previous report (C4)."""

from __future__ import annotations

import pytest

from chess_insights.coach import CoachConfig, progress
from chess_insights.report.json_export import to_dict
from test_packet import coaching_report

CLOCK = "share of the clock used on the first 15 moves (blitz)"


def previous_report(changes: dict[str, float]) -> dict:
    """The fixture report's JSON with some baselines changed, as the last report on disk would be."""
    data = to_dict(coaching_report())
    data["generated_at"] = "2026-08-29T12:00:00+00:00"
    for item in data["study_plan"]:
        metric = (item.get("baseline") or {}).get("metric")
        if metric in changes:
            item["baseline"]["value"] = changes[metric]
    return data


def test_progress_lines_compare_the_same_metric():
    report = coaching_report()
    clock_item = next(i for i in report.study_plan if (i.baseline or {}).get("metric") == CLOCK)
    clock_item.baseline["value"] = 0.38
    progress.annotate(report, CoachConfig(previous=previous_report({CLOCK: 0.5})))
    lines = {p.metric: p for p in report.coaching.progress}
    clock = lines[CLOCK]
    assert clock.text == f"{CLOCK}: 50% → 38%"
    assert (clock.before, clock.now, clock.improved) == (0.5, 0.38, True)
    assert clock.title == clock_item.title
    # unchanged numbers are neither better nor worse
    assert all(p.improved is None for m, p in lines.items() if m != CLOCK)
    assert report.coaching.settings["progress_since"] == "2026-08-29"


@pytest.mark.parametrize(
    "metric, before, now, text, improved",
    [
        ("share of the clock used on the first 15 moves (blitz)", 0.38, 0.5, "38% → 50%", False),
        ("score vs rating in the Sicilian Defense (black), per game", -0.16, -0.04,
         "−16 per 100 games → −4 per 100 games", True),
        ("blunders per 100 moves", 3.2, 2.5, "3.2 → 2.5", True),
        ("share of winning positions converted", 0.65, 0.7, "65% → 70%", True),
        ("engine eval after move 10 in the Caro-Kann Defense (centipawns)", -80, -30, "−0.8 → −0.3", True),
        ("games with 5...e5 in this position", 10, 12, "10 → 12", False),
        ("rated games started 23:00-02:00", 40, 25, "40 → 25", True),
        ("something new", 1.5, 2.0, "1.50 → 2.00", None),
    ],
)
def test_direction_and_units(metric, before, now, text, improved):
    line = progress.compare(metric, before, now)
    assert line.text.endswith(text)
    assert line.improved is improved


def test_centipawn_metrics_are_shown_in_pawns():
    line = progress.compare("engine eval after move 10 in the Caro-Kann Defense (centipawns)", -80, -30)
    assert line.text.startswith("engine eval after move 10 in the Caro-Kann Defense (pawns):")


def test_no_previous_report_no_progress():
    report = coaching_report()
    progress.annotate(report, CoachConfig())
    assert report.coaching.progress == []
    assert "progress_since" not in report.coaching.settings


def test_metrics_missing_from_the_previous_report_are_left_out():
    report = coaching_report()
    previous = previous_report({})
    previous["study_plan"] = [{"title": "Old item", "baseline": {"metric": "something else", "value": 1}}, "junk"]
    progress.annotate(report, CoachConfig(previous=previous))
    assert report.coaching.progress == []


def test_no_coaching_block_is_fine():
    report = coaching_report()
    report.coaching = None
    progress.annotate(report, CoachConfig(previous=previous_report({})))
    assert report.coaching is None
