"""engine.select_engine_games: which games Stockfish analyses (C0). Pure: no Stockfish needed."""

from datetime import datetime, timedelta, timezone

import pytest

from chess_insights.engine import BALANCED_TIME_CLASSES, ENGINE_SAMPLES, MIN_PLIES, select_engine_games
from factories import make_game

MOVES = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e5"]
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def games_of(time_class, n, start=0, step=1, **kw):
    """``n`` games of one format; game k ended ``start + k * step`` hours before T0 (k = 0 is the newest)."""
    return [
        make_game(game_id=f"{time_class}-{k}", time_class=time_class, moves_san=MOVES,
                  end_time=T0 - timedelta(hours=start + k * step), **kw)
        for k in range(n)
    ]


def counts(games):
    out = {}
    for g in games:
        out[g.time_class] = out.get(g.time_class, 0) + 1
    return out


def is_newest_first(games):
    return all((a.end_time, a.game_id) >= (b.end_time, b.game_id) for a, b in zip(games, games[1:]))


# --------------------------------------------------------------------------- recent (the old behaviour)
def test_recent_takes_the_most_recent_analysable_games():
    bullet = games_of("bullet", 20, start=0)  # the newest 20
    blitz = games_of("blitz", 5, start=100)
    short = make_game(game_id="short", moves_san=MOVES[: MIN_PLIES - 2], end_time=T0 + timedelta(hours=1))
    variant = make_game(game_id="variant", moves_san=MOVES, rules="crazyhouse", end_time=T0 + timedelta(hours=2))
    chosen = select_engine_games(blitz + [short, variant] + bullet, 10)
    assert [g.game_id for g in chosen] == [f"bullet-{k}" for k in range(10)]
    assert select_engine_games(blitz + bullet, 10, sample="recent") == chosen


def test_recent_with_a_time_class_filter():
    bullet, blitz, rapid = games_of("bullet", 20), games_of("blitz", 8, start=50), games_of("rapid", 8, start=60)
    chosen = select_engine_games(bullet + blitz + rapid, 12, time_classes=["blitz", "RAPID"])
    assert counts(chosen) == {"blitz": 8, "rapid": 4}
    assert is_newest_first(chosen)


# --------------------------------------------------------------------------- balanced
def test_balanced_splits_evenly_most_recent_in_each_format():
    bullet, blitz, rapid = games_of("bullet", 50), games_of("blitz", 50, start=200), games_of("rapid", 50, start=400)
    chosen = select_engine_games(bullet + blitz + rapid, 30, sample="balanced")
    assert counts(chosen) == {"bullet": 10, "blitz": 10, "rapid": 10}
    assert {g.game_id for g in chosen} == {f"{tc}-{k}" for tc in ("bullet", "blitz", "rapid") for k in range(10)}
    assert is_newest_first(chosen)


def test_an_odd_game_goes_to_the_slower_format():
    pool = games_of("bullet", 20) + games_of("blitz", 20, start=50) + games_of("rapid", 20, start=90)
    assert counts(select_engine_games(pool, 10, sample="balanced")) == {"rapid": 4, "blitz": 3, "bullet": 3}
    assert counts(select_engine_games(pool, 11, sample="balanced")) == {"rapid": 4, "blitz": 4, "bullet": 3}


def test_a_format_that_runs_short_leaves_its_share_to_the_others():
    pool = games_of("bullet", 100) + games_of("blitz", 100, start=200) + games_of("rapid", 5, start=400)
    chosen = select_engine_games(pool, 60, sample="balanced")
    assert counts(chosen) == {"rapid": 5, "blitz": 28, "bullet": 27}
    assert len(chosen) == 60


def test_balanced_with_a_time_class_filter_and_daily_only_when_listed():
    pool = (games_of("bullet", 30) + games_of("blitz", 30, start=50) + games_of("rapid", 30, start=90)
            + games_of("daily", 30, start=130, time_control="1/86400", base_seconds=86400))
    assert "daily" not in BALANCED_TIME_CLASSES
    assert "daily" not in counts(select_engine_games(pool, 40, sample="balanced"))
    assert counts(select_engine_games(pool, 40, time_classes=["blitz", "rapid"], sample="balanced")) == {
        "blitz": 20, "rapid": 20,
    }
    assert counts(select_engine_games(pool, 40, time_classes=["daily", "rapid"], sample="balanced")) == {
        "daily": 20, "rapid": 20,
    }


@pytest.mark.parametrize("sample", ENGINE_SAMPLES)
def test_max_games_larger_than_the_pool_takes_every_game(sample):
    pool = games_of("bullet", 7) + games_of("blitz", 3, start=20)
    for limit in (100, None):
        chosen = select_engine_games(pool, limit, sample=sample)
        assert counts(chosen) == {"bullet": 7, "blitz": 3} and is_newest_first(chosen)
    assert select_engine_games(pool, 0, sample=sample) == []
    assert select_engine_games(pool, -1, sample=sample) == []
    assert select_engine_games([], 10, sample=sample) == []


@pytest.mark.parametrize("sample", ENGINE_SAMPLES)
def test_ties_on_end_time_do_not_depend_on_the_input_order(sample):
    same = T0 - timedelta(hours=1)
    tied = [make_game(game_id=f"tied-{c}", time_class="blitz", moves_san=MOVES, end_time=same) for c in "cab"]
    newer = make_game(game_id="newer", time_class="blitz", moves_san=MOVES, end_time=T0)
    first = select_engine_games([newer] + tied, 2, sample=sample)
    again = select_engine_games(list(reversed(tied)) + [newer], 2, sample=sample)
    assert [g.game_id for g in first] == [g.game_id for g in again] == ["newer", "tied-c"]


def test_a_repeated_game_keeps_its_last_copy():
    old = make_game(game_id="dup", time_class="blitz", moves_san=MOVES, end_time=T0, opponent="first")
    new = make_game(game_id="dup", time_class="blitz", moves_san=MOVES, end_time=T0, opponent="second")
    assert [g.opponent for g in select_engine_games([old, new], 5)] == ["second"]


@pytest.mark.parametrize("sample", ENGINE_SAMPLES)
def test_a_game_that_is_turned_down_is_replaced_by_the_next_of_its_format(sample):
    pool = games_of("bullet", 10) + games_of("rapid", 10, start=50)
    asked = []

    def usable(game):
        asked.append(game.game_id)
        return game.game_id not in ("rapid-0", "bullet-1")

    chosen = select_engine_games(pool, 6, sample=sample, usable=usable)
    ids = {g.game_id for g in chosen}
    assert "rapid-0" not in ids and "bullet-1" not in ids and len(chosen) == 6
    if sample == "balanced":
        assert counts(chosen) == {"bullet": 3, "rapid": 3} and "rapid-3" in ids
    assert len(asked) == 6 + 2 - (sample == "recent")  # no game is asked about that isn't taken or turned down


def test_an_unknown_sample_is_an_error():
    with pytest.raises(ValueError, match="balanced"):
        select_engine_games([], 10, sample="random")
