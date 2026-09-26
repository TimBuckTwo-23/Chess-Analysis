"""coach/rating_map.py: lichess_equivalent, the one chess.com -> Lichess conversion of the coaching layer."""

from __future__ import annotations

from chess_insights.coach import rating_map


def test_lichess_equivalent_is_to_lichess_for_a_format_with_rows():
    for tc in ("blitz", "rapid", "bullet"):
        for rating in (400, 949, 1234, 2800):
            assert rating_map.lichess_equivalent(rating, tc) == rating_map.to_lichess(rating, tc)
    assert rating_map.lichess_equivalent(949, "blitz") == 1390 and isinstance(
        rating_map.lichess_equivalent(949.0, "blitz"), int)


def test_lichess_equivalent_applies_the_overrides_like_the_other_helpers():
    overrides = {"blitz": [[900, 1500], [1000, 1600]]}
    assert rating_map.lichess_equivalent(950, "blitz", overrides) == rating_map.to_lichess(950, "blitz", overrides) == 1550
    assert rating_map.lichess_equivalent(950, "blitz", {"blitz": {"900": 1500, "1000": 1600}}) == 1550  # a dict too
    assert rating_map.lichess_equivalent(950, "rapid", overrides) == rating_map.to_lichess(950, "rapid")
    # unusable rows fall back on the built-in table instead of failing
    assert rating_map.lichess_equivalent(950, "blitz", {"blitz": [[900, "x"]]}) == rating_map.to_lichess(950, "blitz")
    assert rating_map.source_of(950, "blitz", overrides) == "your rating map"
    assert rating_map.source_of(950, "blitz", {"blitz": [[900, "x"]]}) == rating_map.SOURCE_NAME


def test_formats_without_rows_borrow_a_table_and_say_so():
    assert rating_map.lichess_equivalent(1100, "daily") == rating_map.to_lichess(1100, "rapid")
    assert rating_map.lichess_equivalent(1100, "chess960") == rating_map.to_lichess(1100, "blitz")
    assert rating_map.lichess_equivalent(1100, "daily", {"daily": [[1000, 1200], [1200, 1400]]}) == 1300
    assert rating_map.source_of(1100, "daily") == f"{rating_map.SOURCE_NAME}, rapid rows, rough estimate"
    assert rating_map.source_of(949, "blitz") == rating_map.SOURCE_NAME  # between the two checked rows
    assert rating_map.source_of(1500, "blitz").endswith("rough estimate")
