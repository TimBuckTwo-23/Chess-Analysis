"""Study-plan targets against the previous report (C4): no verdict without a test."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chess_insights import pipeline
from chess_insights.coach import CoachConfig, progress
from chess_insights.models import Coaching, Report, StudyItem
from chess_insights.report.json_export import to_dict
from chess_insights.stats import MeanTest
from factories import make_game
from test_packet import coaching_report

CLOCK = "share of the clock used on the first 15 moves (blitz)"
LATE = "rated games started 23:00–03:00"
T0 = datetime(2026, 8, 1, 19, tzinfo=timezone.utc)


def previous_report(changes: dict[str, float]) -> dict:
    """The fixture report's JSON with some baselines changed, as the last report on disk would be."""
    data = to_dict(coaching_report())
    data["generated_at"] = "2026-08-29T12:00:00+00:00"
    for item in data["study_plan"]:
        metric = (item.get("baseline") or {}).get("metric")
        if metric in changes:
            item["baseline"]["value"] = changes[metric]
    return data


def test_progress_lines_show_both_numbers_without_a_verdict():
    report = coaching_report()
    clock_item = next(i for i in report.study_plan if (i.baseline or {}).get("metric") == CLOCK)
    clock_item.baseline["value"] = 0.38
    progress.annotate(report, CoachConfig(previous=previous_report({CLOCK: 0.5})))
    lines = {p.metric: p for p in report.coaching.progress}
    clock = lines[CLOCK]
    assert clock.text == f"{CLOCK}: 50% → 38%"  # the previous report has no dates: no count of new games
    # over all the games, most of them in both reports: no badge, however large the change
    assert (clock.before, clock.now, clock.improved) == (0.5, 0.38, None)
    assert clock.title == clock_item.title
    assert all(p.improved is None for p in lines.values())
    assert report.coaching.settings["progress_since"] == "2026-08-29"


@pytest.mark.parametrize(
    "metric, before, now, text, better",
    [
        ("share of the clock used on the first 15 moves (blitz)", 0.38, 0.5, "38% → 50%", -1),
        ("score vs rating in the Sicilian Defense (black), per game", -0.16, -0.04,
         "score vs rating in the Sicilian Defense (black): −16 → −4 per 100 games", +1),
        ("losses on time minus wins on time per game (blitz)", 0.05, 0.02,
         "losses on time minus wins on time (blitz): +5 → +2 per 100 games", -1),
        ("blunders per 100 moves", 3.2, 2.5, "3.2 → 2.5", -1),
        ("share of winning positions converted", 0.65, 0.7, "65% → 70%", +1),
        ("engine eval after move 10 in the Caro-Kann Defense (centipawns)", -80, -30, "−0.8 → −0.3", +1),
        ("games with 5...e5 in this position", 10, 12, "10 → 12", -1),
        ("rated games started 23:00-02:00", 40, 25, "40 → 25", -1),
        ("something new", 1.5, 2.0, "1.50 → 2.00", None),
    ],
)
def test_direction_and_units(metric, before, now, text, better):
    line = progress.compare(metric, before, now)
    assert line.text.endswith(text)
    assert line.improved is None  # side by side: never a verdict
    assert progress.direction(metric) == better


def test_centipawn_metrics_are_shown_in_pawns():
    line = progress.compare("engine eval after move 10 in the Caro-Kann Defense (centipawns)", -80, -30)
    assert line.text.startswith("engine eval after move 10 in the Caro-Kann Defense (pawns):")


def test_count_metrics_are_the_ones_target_for_counts():
    assert progress.is_count("rated games started 23:00–03:00")
    assert progress.is_count("quick losses in the French Defense")
    assert progress.is_count("games with 5...e5 in this position")
    assert not progress.is_count(CLOCK) and not progress.is_count("score vs rating right after a loss, per game")


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


# --------------------------------------------------------------------------- the games since the previous report
def played(late: list[bool], start: datetime, per_session: int = 1):
    """One game per entry, late-night (00:30) or evening (19:00), ``per_session`` games a day in one session."""
    games = []
    for i, is_late in enumerate(late):
        day, k = divmod(i, per_session)
        hour = 24 if is_late else 19  # 00:xx the next morning
        begin = start.replace(hour=0) + timedelta(days=day, hours=hour, minutes=30 + 10 * k)
        games.append(make_game(start_time=begin, end_time=begin + timedelta(minutes=8)))
    return games


def two_reports(before_late: int, n_before: int, new_late: int, n_new: int, per_session: int = 1,
                metrics: tuple[str, ...] = (LATE,)):
    """The previous report's JSON (``n_before`` games, ``before_late`` of them late at night) and this report
    (those games plus ``n_new`` newer ones, ``new_late`` late), each with a study-plan item per metric, and the
    games."""
    earlier = played([i < before_late for i in range(n_before)], T0)
    later = played([i < new_late for i in range(n_new)], earlier[-1].end_time + timedelta(days=1), per_session)
    games = earlier + later

    def report(gs, late_count):
        plan = [StudyItem(title=f"Item {m}", why="", actions=[], baseline={"metric": m, "value": late_count})
                for m in metrics]
        return Report(username="BigMuffEater", generated_at=datetime(2026, 9, 26, tzinfo=timezone.utc), filters="",
                      n_games=len(gs), date_from=gs[0].end_time, date_to=gs[-1].end_time, modules=[], strengths=[],
                      weaknesses=[], study_plan=plan, coaching=Coaching())

    previous = to_dict(report(earlier, before_late))
    return previous, report(games, before_late + new_late), games


def test_a_count_metric_compares_the_new_games_with_the_earlier_ones():
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=40)
    progress.annotate(report, CoachConfig(previous=previous), games)
    [line] = report.coaching.progress
    last = datetime.fromisoformat(previous["date_to"])  # the previous report's last game
    assert line.text == (f"{LATE}: 40% of your 100 games before {last.day} {last:%b} → 0% of your 40 games since")
    assert line.improved is True and (line.before, line.now) == (40, 40)  # the all-time counts stay as they were
    assert report.coaching.settings["progress_new_games"] == 40


def test_a_count_metric_that_got_worse_says_not_yet():
    previous, report, games = two_reports(before_late=10, n_before=100, new_late=30, n_new=40)
    progress.annotate(report, CoachConfig(previous=previous), games)
    [line] = report.coaching.progress
    assert line.improved is False
    assert "10% of your 100 games before" in line.text and "75% of your 40 games" in line.text


def test_games_in_one_session_count_as_one_unit():
    """The same counts from four sessions of ten games: the new games' time of day is four draws, not forty."""
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=40, per_session=10)
    progress.annotate(report, CoachConfig(previous=previous), games)
    [line] = report.coaching.progress
    assert line.improved is None and line.text.endswith("→ 0% of your 40 games since (no clear change yet)")
    assert progress.session_sizes(games[100:]) == [10, 10, 10, 10] and len(progress.session_sizes(games[:100])) == 100
    assert progress.effective_units(games[100:]) == 4 and progress.effective_units(games[:100]) == 100


def test_a_few_long_sessions_are_worth_fewer_games_than_many_short_ones():
    """Kish's effective sample size: ten sessions of one game and one of ten are worth 400 / 110 = 3.6 games."""
    games = played([False] * 10, T0) + played([False] * 10, T0 + timedelta(days=30), per_session=10)
    assert sorted(progress.session_sizes(games)) == [1] * 10 + [10]
    assert progress.effective_units(games) == pytest.approx(400 / 110)
    daily = [make_game(time_class="daily", base_seconds=86400, end_time=T0 + timedelta(days=i)) for i in range(3)]
    assert progress.session_sizes(daily) == [1, 1, 1] and progress.effective_units([]) == 0


def test_too_few_new_games_are_not_compared():
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=progress.MIN_NEW_GAMES - 1)
    progress.annotate(report, CoachConfig(previous=previous), games)
    [line] = report.coaching.progress
    assert line.improved is None
    assert line.text.startswith(f"{LATE}: 0 of your 19 games since ") and line.text.endswith(
        ", 40% of the 100 before (too few new games to compare yet)")


def test_without_the_games_or_with_other_games_nothing_is_judged():
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=40)
    progress.annotate(report, CoachConfig(previous=previous))  # no games: the periods can't be told apart
    [line] = report.coaching.progress
    last = datetime.fromisoformat(previous["date_to"])
    assert line.improved is None and line.text == f"{LATE}: 40 → 40 in all your games (40 new games since " \
                                                  f"{last.day} {last:%b})"
    # the previous report covered other games (another filter): the counts' difference is not the new games' count
    previous["n_games"] = 90
    progress.annotate(report, CoachConfig(previous=previous), games)
    assert report.coaching.progress[0].improved is None
    # another player's report
    previous["n_games"], previous["username"] = 100, "someone_else"
    progress.annotate(report, CoachConfig(previous=previous), games)
    assert report.coaching.progress[0].improved is None


def test_other_metrics_say_how_many_games_are_new_and_judge_nothing():
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=40, metrics=(CLOCK,))
    previous["study_plan"][0]["baseline"]["value"] = 0.5
    report.study_plan[0].baseline["value"] = 0.1
    progress.annotate(report, CoachConfig(previous=previous), games)
    [line] = report.coaching.progress
    last = datetime.fromisoformat(previous["date_to"])
    assert line.improved is None
    assert line.text == f"{CLOCK}: 50% → 10% (all your games; 40 new games since {last.day} {last:%b})"


def test_tested_lines_are_adjusted_together(monkeypatch):
    """Two tested lines: p = 0.04 alone would pass, but not after Benjamini-Hochberg with a p = 0.9 beside it."""
    metrics = (LATE, "quick losses in the French Defense")
    p_values = iter([0.04, 0.9])
    monkeypatch.setattr(progress, "two_proportion_test",
                        lambda *a: MeanTest(n=1, mean=0.0, se=1.0, z=0.0, p_value=next(p_values)))
    previous, report, games = two_reports(before_late=40, n_before=100, new_late=0, n_new=40, metrics=metrics)
    progress.annotate(report, CoachConfig(previous=previous), games)
    assert [p.improved for p in report.coaching.progress] == [None, None]
    assert all(p.text.endswith("(no clear change yet)") for p in report.coaching.progress)
    p_values = iter([0.02, 0.03])
    monkeypatch.setattr(progress, "two_proportion_test",
                        lambda *a: MeanTest(n=1, mean=0.0, se=1.0, z=0.0, p_value=next(p_values)))
    progress.annotate(report, CoachConfig(previous=previous), games)
    assert [p.improved for p in report.coaching.progress] == [True, True]


def test_the_pipeline_hands_progress_the_games(monkeypatch):
    from chess_insights import coach

    seen = {}

    def finish(report, cfg, games=None):
        seen["games"] = games

    monkeypatch.setattr(coach, "finish_coaching", finish)
    report = coaching_report()
    games = [make_game()]
    pipeline.finish_coaching(report, CoachConfig(), games)
    assert seen["games"] is games
    monkeypatch.setattr(coach, "finish_coaching", lambda report, cfg: seen.update(old=True))  # an older signature
    pipeline.finish_coaching(report, CoachConfig(), games)
    assert seen["old"] is True
