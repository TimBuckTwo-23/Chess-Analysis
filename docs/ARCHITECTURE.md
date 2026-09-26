# Architecture

```
chess.com PubAPI ──api.py──> raw game JSON (disk cache, one file per month)
                                   │                 saved JSON archives (--json) ──┐
                              parse.py            PGN files (--pgn) ──parse.py──────┤
                                   ▼                                                │
                              list[Game]  <─────────────────────────────────────────┘
                                   │  dataset.filter_games (time class, rules, rated, dates)
                                   ▼
                  ┌──────── AnalysisContext (games, df, evals, options) ────────┐
                  │                                                              │
      optional engine.py (Stockfish): select_engine_games (recent | balanced)    │
                  └──── evals: dict[game_id, GameEval] ──────────────────────────┤
                                                                                 │
   analysis/results openings time_mgmt endings habits engine_stats mistakes   (each: analyze(ctx) -> ModuleResult;
                  │                                                            pictures from visuals.py)
   optional coach.build_coaching (--coach): critical → deep → motifs/concepts → explain, sources, drills
                  │                                                              │
   fill.py: formats + a fallback chart/board for every finding a module left bare  │
                  │                                                              │
                  └───── insights.py: rank + dedupe + grouped study plan + headline ─────┘
                                   ▼
                                Report ── coach.finish_coaching (progress, Maia, LLM) ──┐
                                   │                                                    │
                                   ├─ format_reports: the same pipeline per format (60+ games each)
                                   ▼
                                Report ──report/html.py | markdown.py | json_export.py──> files
```

All shared types live in `src/chess_insights/models.py`. Read it first.

## Contracts

* **Game** — one game normalised to the analysed player's side (`color`, `outcome`,
  `my_rating`, `clocks` …). Produced only by `parse.py`.
* **GameEval / PlyEval** — per-ply Stockfish verdicts for one game. Produced only by `engine.py`.
* **AnalysisContext** (`context.py`) — `games` (filtered, sorted by end time), `evals`,
  `options` (e.g. `tz`, `puzzle_file`, `demo`), and `df` (`dataset.to_frame`, one row per
  game, built on first use; no module reads it at the moment, so pandas could become an
  optional dependency).
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
  `formats` is the number of games behind it per time class ("blitz only"); `chart`
  (the numbers behind it, split by format where the data allows) and `diagram` (the
  position it is about) are its picture. Modules set them where they know best;
  `fill.py` fills the gaps (below).
* **Chart / Diagram** — plain data, drawn by the renderers. A `Diagram` is a FEN plus
  arrows (the move played red, the better move green, threats orange), marked squares and
  optional `strips` (a line of play as a row of small boards). `visuals.py` builds both.
* **Report** — every strength and weakness (ranked), the study plan (one `StudyItem`
  per cause, with the insight ids it covers, a target and a baseline), the headline
  lines, labels for the linked games, and every module's tables and charts. A `Chart`
  can carry its full `table` (shown under "Show the numbers" instead of listing the same
  numbers twice); a `Table` can name its `key_columns` (what a phone shows first).
  `formats` / `engine_formats` count the games (and engine-analysed games) per time class;
  `format_reports` holds one full Report per format with enough games (`time_class` set, no
  views of its own); `coaching` holds the coaching layer's output (or None).
  Renderers only ever see a `Report`.
* **Coaching** — explanations (`Explanation`: the position, both engine lines, motifs,
  concept differences, text, board), drills, the review schedule, progress and notes on
  what was skipped. It never creates or changes an Insight (the one exception, motif claims,
  goes through `stats.significance` in `coach/profile.py`). Settings: `coach.CoachConfig`.

## Per-format views

`pipeline.run_analysis` runs the modules once on every game, then, when the games cover two or
more formats, once more for each format with at least `MIN_FORMAT_GAMES` games, on that
format's games and engine evals only. Each view is a full `Report` (`time_class` set, its own
claims, study plan and engine note, no nested views). The coaching runs once, on all games;
each view gets a copy with only its format's explanations. `options["format_views"] = False`
skips the views.

## Statistical rules (apply everywhere)

1. Compare scores with the **Elo expected score**, not 50 %. Beating weaker
   players is not a strength.
2. Never flag anything below the module's minimum sample size (defaults: 8
   games for an opening family as one colour, 25 for a split like
   "after a loss").
3. **One claim rule.** A strength or weakness needs `stats.significance(...)`
   to pass: enough games, a BH-adjusted p-value at or below the tier's alpha,
   and a confidence derived from the adjusted p. Tiers: `stats.ALPHA` (0.05)
   for pre-specified hypotheses (colour, opening families, after a loss, late
   night); `stats.STRICT_ALPHA` (0.01) for everything else. Build tests with
   `stats.summarize` / `mean_test(..., min_sd=...)` / `two_proportion_test` /
   `one_sided`. Anything that fails the rule is at most an observation.
   `tests/test_null_calibration.py` enforces the result: on data with no real
   effects, a report (its lists and every section) may contain on average no more
   than 0.3 false claims.
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
   effect is uncertain (`stats.ATTENUATION_RANGE`). Colour claims must hold for
   every White first-move edge in `stats.WHITE_EDGE_RANGE`.
9. **Compare subsets with the player's other games**, not only with the rating's
   expectation. A lagging rating (an improving player) lifts every game, so an
   opening is compared with the player's other openings of the same colour, a
   game-length band with games of other lengths, and so on.
10. Exclude games with fewer than 4 plies (aborted starts, instant abandons) from
   skill metrics. They still count in the results totals.
11. Word findings as associations ("you score worse after 11 pm"), not causes, and
   in the report's units: "+9 per 100 games vs your rating", pawns rather than
   centipawns.
12. Say which formats a finding rests on (`Insight.formats`: games per time class), and give
   it a picture of the numbers behind it. Neither may change what is claimed.

## Module ownership

| Module | Responsibility |
|---|---|
| `api.py` | HTTP client for the chess.com PubAPI: User-Agent, serial requests, retries with backoff on 429/5xx, typed errors |
| `fetch.py` | Download all monthly archives into a disk cache (past months are immutable, the current month is refreshed) and load them offline |
| `parse.py` | chess.com JSON / PGN → `Game`: result mapping, termination, time control, opening name from `ECOUrl`, SAN moves + `[%clk]` clocks |
| `models.py` | The shared data contracts (`Game`, `GameEval`, `Insight`, `ModuleResult`, `StudyItem`, `Report` ...) |
| `context.py` | `AnalysisContext`, the one object every analysis module receives |
| `dataset.py` | Filtering and the pandas frame |
| `stats.py` | Shared statistics helpers, the claim rule, and number wording (`pct`, `per100_games`, `vs_rating`) |
| `synth.py` | Synthetic chess.com-format archives for a player with planted strengths and weaknesses (demo and recovery tests) |
| `analysis/results.py` | Score vs expected, rating trends, colour, opponent-strength buckets, time classes |
| `analysis/openings.py` | Repertoire performance by opening family/line and colour; your choices at key moves; repertoire breadth counted on your own moves |
| `analysis/time_mgmt.py` | Time trouble, flagging, opening pace, clock balance at moves 20 and 30, think time by move number |
| `analysis/endings.py` | How games end, game-length performance |
| `analysis/habits.py` | Sessions, tilt after losses, fatigue, time of day |
| `engine.py` | Which games Stockfish analyses (`select_engine_games`: most recent, or balanced across formats), Stockfish analysis (parallel workers, disk cache), Lichess-style win%, accuracy, judgements, phases, tags |
| `analysis/engine_stats.py` | Accuracy, blunder rates, phase weaknesses, conversion, missed tactics, opening outcome |
| `analysis/mistakes.py` | Repeated mistakes keyed by position (EPD), the costliest-puzzles table, and the `--puzzles` PGN export |
| `insights.py` | Rank and dedupe insights; group weaknesses by cause into the study plan (easiest first, targets); the headline lines |
| `visuals.py` | Shared picture builders: format counts and wording ("blitz only"), comparison charts, split-by-format charts, boards with move arrows, strips of small boards for a line |
| `fill.py` | Fallbacks run for every report: `fill_formats` (the format mix of the games a module analysed, narrowed to the format a finding names) and `fill_visuals` (a chart from the finding's evidence, or the board for a position; nothing when no pattern fits) |
| `pipeline.py` | Run everything, isolate module and coaching failures, build the `Report` and its per-format views (`MIN_FORMAT_GAMES` = 60) |
| `coach/` | The coaching layer (`--coach`): `critical` (positions to explain), `deep` (deeper Stockfish lines, own cache), `motifs` (tactics along the lines, Lichess theme names), `concepts` (Stockfish 16 eval terms at the line ends, board facts), `sources/` (Lichess explorer, cloud eval, tablebase, chess-openings, Wikibooks; cached, optional), `openings_info`, `endgames`, `explain` (templates, boards), `profile` (motif profile; the motif claims), `drills` and `puzzles_db` (puzzle packs, review schedule), `puzzles` (the richer `--puzzles` PGN), `progress`, `maia`, `llm` and `ask` (the optional LLM coach and its verifier) |
| `report/*` | HTML (one file, mobile-first, dark mode; self-contained apart from web fonts, with offline fallbacks), Markdown and JSON output |
| `cli.py` | `chess-insights fetch / report / demo / puzzles-db / ask` |
| `__main__.py` | `python -m chess_insights` |
