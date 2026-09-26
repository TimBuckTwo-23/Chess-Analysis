"""Drill packs, the review schedule and the study plan's drill actions (coach/drills.py, insights.practice_actions).

The puzzles come from tests/fixtures/lichess_puzzles_drill_subset.csv: 2,000 real Lichess puzzles (CC0) with their
themes, popularity and opening tags.
"""

import csv
from datetime import date

import chess
import chess.pgn
import pytest

from chess_insights import insights
from chess_insights.analysis.engine_stats import TACTIC_TEXT
from chess_insights.coach import CoachConfig, build_coaching, critical, drills
from chess_insights.context import AnalysisContext
from chess_insights.models import Coaching, Drill, DrillPuzzle, Explanation, Insight, ModuleResult, ReviewItem
from factories import make_game, make_game_eval, make_ply_eval

TODAY = date(2026, 9, 26)
RATING = (1200, 1600)
COUNTS = {
    "fork": {"you_missed": 23, "opp_missed": 15, "you_allowed": 4, "opp_allowed": 3},
    "pin": {"you_missed": 10, "opp_missed": 12, "you_allowed": 9, "opp_allowed": 2},
    "skewer": {"you_missed": 10, "opp_missed": 4, "you_allowed": 1, "opp_allowed": 1},
    "backRankMate": {"you_missed": 0, "opp_missed": 2, "you_allowed": 5, "opp_allowed": 1},
    "hangingPiece": {"you_missed": 2, "opp_missed": 2, "you_allowed": 0, "opp_allowed": 0},
}
LINES = [  # (colour, opening, family, games)
    ("white", "Italian Game: Giuoco Pianissimo", "Italian Game", 10),
    ("white", "Italian Game: Two Knights Defense", "Italian Game", 4),
    ("white", "London System", "London System", 3),
    ("white", "Scotch Game", "Scotch Game", 1),
    ("black", "Sicilian Defense: Open", "Sicilian Defense", 8),
    ("black", "French Defense: Advance Variation", "French Defense", 5),
    ("black", "Caro-Kann Defense", "Caro-Kann Defense", 2),
    ("black", "Scandinavian Defense", "Scandinavian Defense", 1),
]


@pytest.fixture(scope="module")
def subset(fixtures_dir):
    return fixtures_dir / "lichess_puzzles_drill_subset.csv"


def _evals_for(g, errors=(), tags=None):
    plies = []
    for i in range(g.plies):
        mine = g.is_my_ply(i)
        bad = mine and i in errors
        plies.append(make_ply_eval(
            ply=i, mover="white" if i % 2 == 0 else "black", is_user=mine, san=g.moves_san[i],
            best_san="h3" if bad else g.moves_san[i], win_before=60.0 if bad else 50.0,
            win_after=60.0 - errors.get(i, 0) if bad else 50.0, judgement="blunder" if bad else None,
            tags=list(tags or []) if bad else [],
        ))
    return make_game_eval(g.game_id, plies)


def make_ctx(tags=None):
    games, evals = [], {}
    k = 0
    for colour, opening, family, n in LINES:
        for _ in range(n):
            g = make_game(color=colour, opening=opening, opening_family=family,
                          time_class=["blitz", "rapid", "bullet"][k % 3])
            first = 0 if colour == "white" else 1
            evals[g.game_id] = _evals_for(g, {first + 4: 10 + k, first + 8: 5 + k % 3}, tags)
            games.append(g)
            k += 1
    return AnalysisContext(username="tester", games=games, evals=evals)


def coaching_with_counts():
    return Coaching(settings={"motif_profile": {"games": 300, "counts": COUNTS}})


def cfg(subset, tmp_path, **kw):
    return CoachConfig(puzzle_db=subset, out_stem=tmp_path / "me", drill_rating=RATING, **kw)


def read_pgn(path):
    games = []
    with open(path, encoding="utf-8") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            games.append(game)
    return games


def fixture_rows(subset):
    with open(subset, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- packs
def test_three_most_missed_motifs_get_packs_of_the_most_popular_puzzles(subset, tmp_path):
    ctx, coaching = make_ctx(), coaching_with_counts()
    drills.annotate(ctx, coaching, [], cfg(subset, tmp_path), TODAY)
    motif_packs = [d for d in coaching.drills if d.theme != "openings"]
    assert [d.theme for d in motif_packs] == ["fork", "pin", "skewer"]  # most missed; pin before skewer on allowed
    fork = motif_packs[0]
    assert fork.title == "30 fork puzzles" and fork.link == "https://lichess.org/training/fork"
    assert fork.reason == "you missed 23 forks in 300 games; your opponents 15"
    assert fork.rating_range == RATING and fork.file == "me-drill-fork.pgn"
    # the most popular first, then the most played, then the id: deterministic
    rows = [r for r in fixture_rows(subset) if "fork" in r["Themes"].split() and 1200 <= int(r["Rating"]) <= 1600]
    rows.sort(key=lambda r: (-int(r["Popularity"]), -int(r["NbPlays"]), r["PuzzleId"]))
    assert [p.puzzle_id for p in fork.puzzles] == [r["PuzzleId"] for r in rows[:30]]
    ids = [p.puzzle_id for d in coaching.drills for p in d.puzzles]
    assert len(ids) == len(set(ids))  # no puzzle in two packs
    for d in motif_packs:
        assert len(d.puzzles) == 30
        assert all(d.theme in p.themes and RATING[0] <= p.rating <= RATING[1] for p in d.puzzles)
    again = coaching_with_counts()
    drills.annotate(ctx, again, [], cfg(subset, tmp_path), TODAY)
    assert [[p.puzzle_id for p in d.puzzles] for d in again.drills] == [[p.puzzle_id for p in d.puzzles]
                                                                         for d in coaching.drills]


def test_every_pack_is_written_as_pgn_that_reads_back_with_legal_moves(subset, tmp_path):
    coaching = coaching_with_counts()
    drills.annotate(make_ctx(), coaching, [], cfg(subset, tmp_path), TODAY)
    assert {d.file for d in coaching.drills} == {"me-drill-fork.pgn", "me-drill-pin.pgn", "me-drill-skewer.pgn",
                                                 "me-drill-openings.pgn"}
    for drill in coaching.drills:
        games = read_pgn(tmp_path / drill.file)
        assert len(games) == len(drill.puzzles) == 30
        for game, puzzle in zip(games, drill.puzzles):
            assert not game.errors
            assert game.headers["SetUp"] == "1" and game.headers["FEN"] == puzzle.fen
            assert game.headers["Result"] == "*" and game.headers["Site"] == puzzle.url
            assert puzzle.url in game.comment and puzzle.url == f"https://lichess.org/training/{puzzle.puzzle_id}"
            board = game.board()
            for move in game.mainline_moves():
                assert move in board.legal_moves
                board.push(move)
            assert [m.uci() for m in game.mainline_moves()] == puzzle.solution_uci


def test_a_pack_starts_after_the_opponents_first_move(subset, tmp_path):
    coaching = coaching_with_counts()
    drills.annotate(make_ctx(), coaching, [], cfg(subset, tmp_path), TODAY)
    by_id = {r["PuzzleId"]: r for r in fixture_rows(subset)}
    for p in coaching.drills[0].puzzles[:10]:
        r = by_id[p.puzzle_id]
        board = chess.Board(r["FEN"])
        first, *solution = r["Moves"].split()
        board.push_uci(first)
        assert p.fen == board.fen() and p.solution_uci == solution


def test_openings_pack_matches_the_families_of_your_most_played_lines(subset, tmp_path):
    ctx = make_ctx()
    assert drills.top_families(ctx.games) == [
        ("white", "Italian Game", 14), ("black", "Sicilian Defense", 8), ("black", "French Defense", 5),
        ("white", "London System", 3), ("black", "Caro-Kann Defense", 2),
    ]
    coaching = coaching_with_counts()
    drills.annotate(ctx, coaching, [], cfg(subset, tmp_path), TODAY)
    [pack] = [d for d in coaching.drills if d.theme == "openings"]
    assert pack.title == "30 puzzles from your openings" and pack.link == "https://lichess.org/training/openings"
    assert "Italian Game, Sicilian Defense, French Defense" in pack.reason
    families = [drills.family_keys(f) for _, f, _ in drills.top_families(ctx.games)]
    assert all(any(drills.tags_match(k, p.opening_tags) for k in families) for p in pack.puzzles)
    # taken in turn from each family, and for the side you play it with when there are enough such puzzles
    first = pack.puzzles[:5]
    assert [drills.tags_match(families[i], first[i].opening_tags) for i in range(5)] == [True] * 5
    italian = [p for p in pack.puzzles if drills.tags_match(families[0], p.opening_tags)]
    assert italian and all(chess.Board(p.fen).turn == chess.WHITE for p in italian)


@pytest.mark.parametrize(
    "family, tags, expected",
    [
        ("Queen's Gambit", ["Queens_Gambit_Declined"], True),
        ("Queen's Gambit", ["Queens_Pawn_Game"], False),
        ("Ruy Lopez Opening", ["Ruy_Lopez", "Ruy_Lopez_Berlin_Defense"], True),
        ("London System", ["Queens_Pawn_Game", "Queens_Pawn_Game_London_System"], True),
        ("Evans Gambit", ["Italian_Game", "Italian_Game_Evans_Gambit"], True),
        ("Indian Game", ["Indian_Defense"], True),
        ("Indian Game", ["Kings_Indian_Defense"], False),
        ("King's Indian Attack", ["Kings_Indian_Defense"], False),
        ("Petrov's Defense", ["Russian_Game"], True),
        ("Alekhine's Defense", ["Alekhine_Defense"], True),
        ("Grünfeld Defense", ["Grunfeld_Defense"], True),
        ("Sicilian Defense", [], False),
    ],
)
def test_chess_com_families_match_lichess_opening_tags(family, tags, expected):
    assert drills.tags_match(drills.family_keys(family), tags) is expected


def test_without_the_puzzle_database_there_is_one_note_and_no_packs(tmp_path):
    for db in (None, tmp_path / "missing.csv"):
        coaching = coaching_with_counts()
        drills.annotate(make_ctx(), coaching, [], CoachConfig(puzzle_db=db, out_stem=tmp_path / "me"), TODAY)
        assert coaching.drills == [] and not list(tmp_path.glob("*.pgn"))
        assert coaching.notes == ["Run chess-insights puzzles-db once to get drill packs."]
        assert coaching.review and all(i.kind == "own" for i in coaching.review)  # your own positions still come back


def test_packs_without_an_output_stem_are_not_written(subset, tmp_path):
    coaching = coaching_with_counts()
    drills.annotate(make_ctx(), coaching, [], CoachConfig(puzzle_db=subset, drill_rating=RATING, drill_size=5), TODAY)
    assert coaching.drills and all(d.file == "" and len(d.puzzles) == 5 for d in coaching.drills)


def test_themes_fall_back_to_the_explanations_then_the_engine_tags(subset, tmp_path):
    explanations = [
        Explanation(epd="x", fen=chess.STARTING_FEN, played="1.e4", best=None, best_line=None, refutation=None,
                    drill_themes=themes)
        for themes in (["pin"], ["pin", "skewer"], ["skewer"], ["pin"])
    ]
    ctx = make_ctx(tags=["hung_material"])
    picked = drills.pick_themes(ctx, Coaching(explanations=explanations))
    assert [t for t, _ in picked] == ["pin", "skewer", "hangingPiece"]
    assert picked[0][1] == "3 positions explained in this report turn on pins"
    assert picked[2][1].startswith("you left material hanging 68 times in 34 games; your opponents 0")
    coaching = Coaching()
    drills.annotate(ctx, coaching, [], cfg(subset, tmp_path), TODAY)
    assert [d.theme for d in coaching.drills][:1] == ["hangingPiece"]
    assert drills.pick_themes(make_ctx(), Coaching()) == []  # nothing to go on: only the openings pack


# --------------------------------------------------------------------------- review schedule
def item(item_id, due="", step=0, kind="own"):
    return ReviewItem(item_id=item_id, kind=kind, title=item_id, fen=chess.STARTING_FEN, url="u", due=due, step=step)


def as_json(items):
    return [vars(i) for i in items]


def test_new_items_are_due_tomorrow():
    review, due = drills.schedule([item("own:a:1"), item("lichess:X", kind="drill"), item("own:a:1")], None, TODAY)
    assert due == []
    assert [(i.item_id, i.due, i.step) for i in review] == [("own:a:1", "2026-09-27", 0),
                                                            ("lichess:X", "2026-09-27", 0)]


def test_due_items_move_to_their_next_step_and_leave_after_the_last():
    previous = {"coaching": {
        "review": as_json([
            item("own:a:1", "2026-09-25", 0),  # due: next step, 3 days from today
            item("own:b:2", "2026-09-26", 2),  # due today: step 3, 21 days
            item("lichess:C", "2026-09-20", 3, "drill"),  # due after the last step: done
            item("lichess:D", "2026-10-02", 1, "drill"),  # not due yet: unchanged
        ]) + [{"item_id": "broken"}, "not a dict"],
        "review_due": as_json([item("own:e:5", "2026-09-19", 3)]),  # finished in the last report
    }}
    new = [item("own:a:1"), item("own:e:5"), item("own:f:6"), item("lichess:C", kind="drill")]
    review, due = drills.schedule(new, previous, TODAY)
    assert [i.item_id for i in due] == ["own:a:1", "own:b:2", "lichess:C"]
    assert due[0].step == 0 and due[0].due == "2026-09-25"  # as they were
    got = {i.item_id: (i.due, i.step) for i in review}
    assert got == {
        "own:a:1": ("2026-09-29", 1),
        "own:b:2": ("2026-10-17", 3),
        "lichess:D": ("2026-10-02", 1),
        "own:f:6": ("2026-09-27", 0),
    }
    assert [i.item_id for i in review] == ["own:f:6", "own:a:1", "lichess:D", "own:b:2"]  # by due date
    assert drills.schedule(new, previous, TODAY) == (review, due)  # deterministic


def test_the_report_schedules_your_costliest_mistakes_and_the_first_puzzles_of_each_pack(subset, tmp_path):
    ctx = make_ctx()
    coaching = coaching_with_counts()
    drills.annotate(ctx, coaching, [], cfg(subset, tmp_path), TODAY)
    own = [i for i in coaching.review if i.kind == "own"]
    lichess = [i for i in coaching.review if i.kind == "drill"]
    assert len(own) == 10 and len(lichess) == 5 * len(coaching.drills)
    assert own[0].item_id.startswith("own:") and own[0].item_id.count(":") >= 2
    assert own[0].url.startswith("https://lichess.org/analysis/") and "find a better move" in own[0].title.lower()
    assert all(i.due == "2026-09-27" and i.step == 0 for i in coaching.review)
    assert lichess[0].item_id == f"lichess:{coaching.drills[0].puzzles[0].puzzle_id}"
    # a week later, everything is due once and comes back three days on
    later = coaching_with_counts()
    previous = {"coaching": {"review": as_json(coaching.review), "review_due": []}}
    drills.annotate(ctx, later, [], cfg(subset, tmp_path, previous=previous), date(2026, 10, 3))
    assert len(later.review_due) == len(coaching.review)
    assert {(i.due, i.step) for i in later.review} == {("2026-10-06", 1)}


# --------------------------------------------------------------------------- the whole coaching step
def test_build_coaching_brings_packs_and_a_schedule(subset, tmp_path, monkeypatch):
    monkeypatch.setattr(critical, "select_critical", lambda ctx, modules, n: [])
    ctx = make_ctx(tags=["hung_material"])
    coaching = build_coaching(ctx, [], cfg(subset, tmp_path, offline=True, profile=False, today=TODAY))
    assert [d.theme for d in coaching.drills] == ["hangingPiece", "openings"]
    assert coaching.review and not any("skipped" in n for n in coaching.notes)


# --------------------------------------------------------------------------- the study plan
def _drill(theme, n=30, file="", reason=""):
    puzzle = DrillPuzzle(puzzle_id="p", fen=chess.STARTING_FEN, solution_uci=[], solution_san=[], rating=1300)
    return Drill(theme=theme, title=f"{n} {theme} puzzles", link=f"https://lichess.org/training/{theme}", file=file,
                 puzzles=[puzzle] * n, reason=reason)


def _weakness(id_, category, study, **evidence):
    return Insight(id=id_, kind="weakness", category=category, title=id_, detail="", severity=0.8, confidence=0.9,
                   study=list(study), evidence=evidence)


def test_practice_actions_without_drills_is_unchanged():
    stats = {"puzzles_exported": 40, "puzzle_file": "p.pgn"}
    mistakes = ModuleResult(key="mistakes", title="", summary="", stats=stats)
    actions = insights.practice_actions([mistakes])
    assert actions == [actions[0]] and "40 positions" in actions[0] and actions.drills == {}
    assert insights.practice_actions([]) == [] and insights.practice_actions([], Coaching()).drills == {}


def test_practice_actions_carry_one_action_per_pack():
    coaching = Coaching(drills=[
        _drill("fork", file="me-drill-fork.pgn", reason="you missed 23 forks in 300 games; your opponents 15"),
        _drill("pin"), _drill("openings"), _drill("skewer", n=0),
    ])
    practice = insights.practice_actions([], coaching)
    assert list(practice) == [] and list(practice.drills) == ["fork", "pin"]
    assert practice.drills["fork"] == ("30 fork puzzles in me-drill-fork.pgn, or lichess.org/training/fork "
                                       "(you missed 23 forks in 300 games; your opponents 15).")
    assert practice.drills["pin"] == "30 pin puzzles at lichess.org/training/pin."


def test_tactics_and_blunders_items_get_one_drill_action_in_place_of_generic_puzzles():
    hung = TACTIC_TEXT["hung_material"]
    weaknesses = [
        _weakness("engine.weakness.hung-material", "tactics", hung["study"]),
        _weakness("engine.weakness.blunder-rate", "blunders", ["Before every move, do a blunder check.",
                                                                "Replay the linked games at each blunder."]),
        _weakness("tactics.weakness.motif-missed.fork", "tactics",
                  ["Drill forks with themed puzzles at your level: https://lichess.org/training/fork.",
                   "Replay the linked games at the moment of the miss."], theme="fork"),
    ]
    coaching = Coaching(drills=[_drill("fork", file="me-drill-fork.pgn"), _drill("hangingPiece"), _drill("openings")])
    mistakes = ModuleResult(key="mistakes", title="", summary="", stats={"puzzles_exported": 12})
    practice = insights.practice_actions([mistakes], coaching)
    plan = insights.build_study_plan(weaknesses, practice=practice)
    by_category = {item.category.split(" ·")[0]: item for item in plan}
    tactics, blunders, positions = by_category["Tactics"], by_category["Blunders"], by_category["Your own positions"]
    assert tactics.insight_ids[0] == "engine.weakness.hung-material"
    drill_lines = [a for a in tactics.actions if "puzzles" in a and "lichess.org/training/" in a]
    assert drill_lines == ["30 hangingPiece puzzles at lichess.org/training/hangingPiece."]  # the finding's own pattern
    assert not any(insights.is_generic_puzzle_action(a) for a in tactics.actions if a not in drill_lines)
    assert len(tactics.actions) == 3
    assert [a for a in blunders.actions if "lichess.org/training/" in a] == [
        "30 fork puzzles in me-drill-fork.pgn, or lichess.org/training/fork."]  # the next pack not yet taken
    assert len(blunders.actions) <= 3 and not any("20 minutes of puzzles" in a for a in blunders.actions)
    assert positions.actions == list(practice)  # your own puzzles, as before
    # without packs the plan is exactly what it was
    plain = insights.build_study_plan(weaknesses, practice=list(practice))
    assert insights.build_study_plan(weaknesses, practice=insights.practice_actions([mistakes])) == plain
    assert any("hanging piece" in a for a in next(i for i in plain if i.category.startswith("Tactics")).actions)
