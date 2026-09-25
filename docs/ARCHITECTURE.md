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
3. **One claim rule.** A strength or weakness needs `stats.significance(...)`
   to pass: enough games, a BH-adjusted p-value at or below the tier's alpha,
   and a confidence derived from the adjusted p. Tiers: `stats.ALPHA` (0.05)
   for pre-specified hypotheses (colour, opening families, after a loss, late
   night); `stats.STRICT_ALPHA` (0.01) for everything else. Build tests with
   `stats.summarize` / `mean_test(..., min_sd=...)` / `two_proportion_test` /
   `one_sided`. Anything that fails the rule is at most an observation.
   `tests/test_null_calibration.py` enforces the result: on data with no real
   effects, a report may contain on average no more than 0.3 false claims.
4. Shrink small-sample rates (`stats.shrink`) before sorting "best/worst" lists.
5. Standard chess only by default. Variants and abandoned 0-move games distort
   everything.
6. Don't mix time classes when a metric depends on the clock (time trouble, move
   times). Report them per time class.
7. When the same test runs over many groups (every opening family, every
   time-of-day bucket), adjust with `stats.bh_adjust` before calling anything
   significant.
8. Opponent-strength effects must be significant, in the same direction,
   against both the plain Elo expectation and `stats.attenuated_expected`.
   Rating noise makes *everyone* look like they underperform against weaker
   players and overperform against stronger ones, and the true size of that
   effect is uncertain (`stats.ATTENUATION_RANGE`). Colour comparisons use
   `stats.colour_expected` (White's first-move edge).
9. Exclude games with fewer than 4 plies (aborted starts, instant abandons) from
   skill metrics. They still count in the results totals.
10. Word findings as associations ("you score worse after 11 pm"), not causes.

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
