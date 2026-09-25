# Architecture

```
chess.com PubAPI ──api.py──> raw game JSON (disk cache, one file per month)
                                   │
                              parse.py            PGN files (manual export) ──parse.py──┐
                                   ▼                                                  │
                              list[Game]  <───────────────────────────────────────────┘
                                   │  dataset.filter_games (time class, rules, rated, dates)
                                   ▼
                  ┌──────── AnalysisContext (games, df, evals, options) ────────┐
                  │                                                              │
      optional engine.py (Stockfish) ── evals: dict[game_id, GameEval] ──────────┤
                  │                                                              │
   analysis/results  openings  time_mgmt  endings  habits  engine_stats   (each: analyze(ctx) -> ModuleResult)
                  │                                                              │
                  └──────────── insights.py: rank + dedupe + study plan ─────────┘
                                   ▼
                                Report ──report/html.py | markdown.py | json_export.py──> files
```

All shared types live in `src/chess_insights/models.py`. Read it first.

## Contracts

* **Game** — one game normalised to the analysed player's side (`color`, `outcome`,
  `my_rating`, `clocks` …). Produced only by `parse.py`.
* **GameEval / PlyEval** — per-ply Stockfish verdicts for one game. Produced only by `engine.py`.
* **AnalysisContext** (`context.py`) — `games` (filtered, sorted by end time), `df`
  (`dataset.to_frame`, one row per game), `evals`, `options`.
* **Analysis module** — `analysis/<name>.py` exposing

  ```python
  def analyze(ctx: AnalysisContext) -> ModuleResult
  ```

  It must never raise on empty/small input (return a ModuleResult with a summary
  saying there is not enough data). It builds its own `Insight`s, because it
  knows its domain best.
* **Insight** — `severity` (effect size, 0..1) × `confidence` (statistical
  confidence, 0..1) = `priority`. Titles are plain language; `detail` carries the
  numbers; `study` holds concrete actions; `example_games` holds chess.com URLs.
* **Report** — top strengths / weaknesses / study plan plus every module's
  tables and charts. Renderers only ever see a `Report`.

## Statistical rules (apply everywhere)

1. Compare scores with the **Elo expected score**, not 50 %. Beating weaker
   players is not a strength.
2. Never flag anything below the module's minimum sample size (defaults: 8
   games for an opening family as one colour, 20 for a split like
   "after a loss").
3. Use `stats.mean_test` / `stats.two_proportion_test` for significance.
   Convert to `Insight.confidence` with `stats.combined_confidence`.
4. Shrink small-sample rates (`stats.shrink`) before sorting "best/worst" lists.
5. Standard chess only by default. Variants and abandoned 0-move games distort
   everything.
6. Don't mix time classes when a metric depends on the clock (time trouble, move
   times). Report them per time class.

## Module ownership

| Module | Responsibility |
|---|---|
| `api.py` | HTTP client for the chess.com PubAPI: User-Agent, serial requests, retries with backoff on 429/5xx, typed errors |
| `fetch.py` | Download all monthly archives into a disk cache (past months are immutable, the current month is refreshed) and load them offline |
| `parse.py` | chess.com JSON / PGN → `Game`: result mapping, termination, time control, opening name from `ECOUrl`, SAN moves + `[%clk]` clocks |
| `dataset.py` | Filtering and the pandas frame |
| `stats.py` | Shared statistics helpers |
| `synth.py` | Synthetic chess.com-format archives for a player with planted strengths and weaknesses (demo and recovery tests) |
| `analysis/results.py` | Score vs expected, rating trends, colour, opponent-strength buckets, time classes |
| `analysis/openings.py` | Repertoire performance by opening family/line and colour |
| `analysis/time_mgmt.py` | Clock usage, time trouble, flagging, fast/slow moves |
| `analysis/endings.py` | How games end, game-length performance |
| `analysis/habits.py` | Sessions, tilt after losses, fatigue, time of day |
| `engine.py` | Stockfish analysis (parallel workers, disk cache), Lichess-style win%, accuracy, judgements, phases, tags |
| `analysis/engine_stats.py` | Accuracy, blunder rates, phase weaknesses, conversion, missed tactics, opening outcome |
| `insights.py` | Rank, dedupe and diversify insights; build the study plan |
| `pipeline.py` | Run everything, isolate module failures, build the `Report` |
| `report/*` | HTML (self-contained, mobile-first, dark mode), Markdown and JSON output |
| `cli.py` | `chess-insights fetch / report / demo` |
