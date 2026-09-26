"""The Engine review's ModuleResult.key is engine_stats.KEY ("engine"), not its file name ("engine_stats").

fill.module_games decides which games a finding without formats rests on: for the engine sections, only the games
Stockfish analysed. With the wrong key an engine finding got the formats of every game instead.
"""

from chess_insights import fill
from chess_insights.analysis import engine_stats, mistakes
from chess_insights.context import AnalysisContext
from chess_insights.models import Insight, ModuleResult
from factories import make_game, make_game_eval


def ctx_with_one_analysed_bullet_game():
    games = [make_game(time_class="bullet")] + [make_game(time_class="blitz") for _ in range(3)]
    analysed = games[0].game_id
    return AnalysisContext(username="tester", games=games, evals={analysed: make_game_eval(analysed, [])})


def test_the_engine_sections_rest_on_the_analysed_games_only():
    ctx = ctx_with_one_analysed_bullet_game()
    assert engine_stats.KEY == "engine" and mistakes.KEY == "mistakes"
    for key in (engine_stats.KEY, "engine_stats", mistakes.KEY):
        assert key in fill.ENGINE_MODULES
        assert [g.time_class for g in fill.module_games(key, ctx)] == ["bullet"]
    assert len(fill.module_games("habits", ctx)) == 4


def test_an_engine_finding_without_formats_gets_the_analysed_games_formats():
    ctx = ctx_with_one_analysed_bullet_game()
    finding = Insight(id=f"{engine_stats.KEY}.weakness.x", kind="weakness", category="tactics", title="x",
                      detail="", severity=0.5, confidence=0.9)
    fill.fill_formats([ModuleResult(key=engine_stats.KEY, title="Engine review", summary="", insights=[finding])],
                      ctx)
    assert finding.formats == {"bullet": 1}
