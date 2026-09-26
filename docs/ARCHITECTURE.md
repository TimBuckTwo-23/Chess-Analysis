# Architecture

```
chess.com PubAPI ──api.py──> raw game JSON (disk cache, one file per month)
                                   │                 saved JSON archives (--json) ──┐
                              parse.py            PGN files (--pgn) ──parse.py──────┤
                                   ▼                                                │
                              list[Game]  <─────────────────────────────────────────┘
                                   │  dataset.filter_games (time class, rules, rated, dates)
                                   ▼
                  AnalysisContext (games, evals, options)
                                   │  optional engine.py (Stockfish): select_engine_games (recent | balanced) → evals
                                   ▼
   pipeline.run_modules: analysis/results openings time_mgmt endings habits structure engine_stats mistakes
                                   │  (each: analyze(ctx) -> ModuleResult; its pictures from visuals.py)
                                   ▼
   optional coach.build_coaching (--coach), BEFORE ranking:
       critical → deep → explain (motifs, concepts, concept_notes) → deep.profile_lines → puzzles.fill_puzzle_lines
       → openings_info (sources/: explorer, cloud eval, openings_db, wikibooks) → endgames (sources/tablebase)
       → profile (motif profile; motif claims through the claim rule) → drills (puzzles_db, review schedule)
                                   │  may add tables, charts and boards to sections; claims only via profile
                                   ▼
   pipeline.apply_deep_verdicts: a repeated mistake whose move the deeper search clears
                                   │  (Explanation.verdict "close" / "fine") becomes an observation
                                   ▼
   pipeline.build_report: fill.py (formats + a fallback chart or board) → insights.py (rank, dedupe,
                                   │  grouped study plan, headline) → Report
                                   ▼
   optional coach.finish_coaching, AFTER ranking: progress (with the games), maia, llm (packet →
                                   │  Claude → verify)
                                   │
                                   ├─ pipeline.build_format_views: on each format's games (60+ games):
                                   │  run_modules → apply_deep_verdicts → view_only_observations (a claim
                                   │  the main report doesn't make becomes an observation) → build_report;
                                   │  no coaching run of its own, a copy of the main coaching with the
                                   │  view's explanations → Report.format_reports
                                   ▼
             report/html.py (boards.py draws every board) | markdown.py | json_export.py ──> files
```

All shared types live in `src/chess_insights/models.py`. Read it first.

## Contracts

* **Game** — one game normalised to the analysed player's side (`color`, `outcome`,
  `my_rating`, `clocks` …). Produced only by `parse.py`.
* **GameEval / PlyEval** — per-ply Stockfish verdicts for one game. Produced only by `engine.py`.
* **AnalysisContext** (`context.py`) — `games` (filtered, sorted by end time), `evals`,
  `options` (`tz`, `engine_sample`, `engine_time_classes`, `puzzle_file`, `demo`,
  `format_views`, and module thresholds such as `structure.min_games`), and `df`
  (`dataset.to_frame`, one row per game, built on first use; no module reads it at the moment,
  so pandas could become an optional dependency).
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
  arrows (the move played red; green only for the engine's move; blue for a line of play or, on
  the openings' choice boards, the move that has scored best for you; threats orange), marked
  squares and optional `strips` (a line of play as a row of small boards). `visuals.py` builds
  both; `report/boards.py` draws a board as inline SVG with one shared piece sprite per page
  (python-chess's piece drawings: see the licence note in PLAN.md).
* **Report** — every strength and weakness (ranked), the study plan (one `StudyItem`
  per cause, with the insight ids it covers, a target and a baseline), the headline
  lines, labels for the linked games, and every module's tables and charts. A `Chart`
  can carry its full `table` (shown under "Show the numbers" instead of listing the same
  numbers twice); a `Table` can name its `key_columns` (what a phone shows first).
  `formats` / `engine_formats` count the games (and engine-analysed games) per time class;
  `format_reports` holds one full Report per format with enough games (`time_class` set, no
  views of its own); `coaching` holds the coaching layer's output (or None).
  Renderers only ever see a `Report`.
* **Coaching types** — `CriticalPosition` (a position to explain: `kind` "repeated", "error"
  or "choice", the move played, its cost, the games it came up in), `Line` (moves from a FEN
  with the engine's verdict at the end, from the side to move), `Motif` (a Lichess theme
  name, the line it is on, the ply, the squares, who carries it out), `ConceptDelta` (a
  Stockfish 16 evaluation term or a board fact, refutation end minus best-line end, in pawns),
  `Source` (where an external fact came from, with its licence), `OpeningFacts`,
  `Explanation` (both lines, motifs, concepts, board facts, text, `text_source` "template"
  or "llm", sources, board and chart, and `verdict`: "error" when the deeper search confirms the
  mistake, "close" when it finds the move under an inaccuracy and under a pawn behind its first
  choice or a choice point's move is under the openings section's bar, "fine" when the move is
  its own first choice), `Drill` / `DrillPuzzle`, `ReviewItem`,
  `PlanEntry`, `ProgressItem` (`improved` True / False only when progress's test passed, else
  None).
* **Coaching** — explanations, notes on what was skipped, settings, the motif profile table
  and chart, drills, the review schedule (`review`, and `review_due` from earlier reports),
  the theory-exit and tablebase tables and boards, the weekly plan, progress, the LLM's
  counts (`llm`), and, for the `--puzzles` export, `puzzle_lines` / `puzzle_themes` for the
  errors that export can use and the explained ones. Settings: `coach.CoachConfig`.

## The coaching layer and the claim rule

`pipeline.run_analysis` runs, in this order:

1. `run_modules`: every analysis module on every game.
2. `build_coaching` (with a `CoachConfig`, i.e. `--coach` with `--engine`): the coaching steps
   that only need the games, the evals and the modules' findings. It runs **before ranking**
   because its only possible claims, the motif claims, must be ranked, grouped into the study
   plan and deduplicated like every other finding, because the study plan's practice
   actions point at its puzzle packs, and because its deeper verdicts can change a finding's
   kind (next step). Without evals only the engine-free steps run (opening facts, theory exit,
   the openings puzzle pack).
3. `apply_deep_verdicts`: a strength or weakness that an explanation explains (the same insight
   id whatever its kind, and for a finding about one move, the same position and move;
   `pipeline.explains`) whose `Explanation.verdict` is "close" or "fine" becomes an observation:
   its detail starts with the deeper search's result, "… is better" leaves the title, the habit
   sentence and the "why X beats Y" study step go, its board loses the green arrow, and the
   section's summary names the moves cleared. Never the reverse.
4. `build_report`: `fill.fill` gives every finding its formats and, where its module gave
   none, a picture; `insights.py` ranks, merges and builds the study plan and headline.
5. `finish_coaching`, **after ranking**: the steps that need the finished report. Progress
   compares the study plan's baselines with the previous report's, and gets the report's games
   to tell the games since that report from the earlier ones (docs/METHODOLOGY.md, section 6);
   Maia-2 reorders the explanations; the LLM coach gets a packet of the report's claims, study
   plan and up to 20 explanations (`coach/packet.py`), and its texts replace a template only
   when `coach/verify.py` accepts them (nothing is sent under `--offline`).
6. `build_format_views` (below), with the ids of the main report's claims
   (`pipeline.claim_ids`).

`apply_deep_verdicts` and `view_only_observations` each run inside a guard (`pipeline._safely`):
if one fails, the findings stay as the modules made them.

Rules the coaching layer keeps:

* **It never creates, changes or removes an Insight**, with two exceptions: the motif claims
  ("you miss forks more often than your opponents"), which `coach/profile.py` builds with
  `stats.significance` at `stats.STRICT_ALPHA`, Benjamini–Hochberg adjusted across every
  pattern and both kinds, and appends to the Engine review section's insights; and the deeper
  verdicts, which `pipeline.apply_deep_verdicts` (not the coaching itself) uses to turn a
  finding into an observation, never the reverse. After `build_coaching`, every section's
  insights are validated again (`pipeline._clean_insights`).
* It may add tables, charts and boards to sections: the motif profile table and chart to
  the Engine review (the renderers show them once, under "Why these moves go wrong", with a
  pointer from the Engine review); the "What stronger players play here" table, facts on the
  openings section's own boards and the theory-exit chart to Openings.
* **Every step is isolated.** Each step runs inside `coach._step` and each source inside
  its own guard; a failure becomes one line in `Coaching.notes`, and the report builds with
  `--offline`, without keys, without Stockfish 16 and without the puzzle database.
  `pipeline.build_coaching` / `finish_coaching` wrap the whole layer the same way.
* **External facts carry their source** (`Source`, shown with the fact) and go through
  `coach/sources/http.py`: one request at a time in the whole process, at least a second
  between two requests to one host, a disk cache per source with its own lifetime, and a
  per-report request cap in the caller.
* Nothing the LLM writes reaches the report unchecked, and a rejected text keeps its
  template (docs/METHODOLOGY.md, section 6).

## Per-format views

`pipeline.run_analysis` runs the modules once on every game, then, when the games cover two or
more formats, once more for each format with at least `MIN_FORMAT_GAMES` (60) games, on that
format's games and engine evals only. Each view is a full `Report` (`time_class` set, its own
sections, study plan and engine note, no nested views). Before a view is ranked, its findings go
through the same `apply_deep_verdicts` as the main report's and then
`pipeline.view_only_observations`: a strength or weakness whose id is not a claim of the main
report becomes an observation in the view, its card, chart, board and title kept, with the first
sentence "Seen in your <format> games only, and not strong enough across all your games to call
it a finding: it may be chance." (and its section's summary says how many). So the page claims
nothing the main report doesn't (docs/METHODOLOGY.md, section 2); `tests/test_null_calibration.py`
checks it on the null world.

The coaching runs once, on all games; each view gets a copy (`pipeline.view_coaching`) with the
explanations of its format, of positions reached in its games, and of its own findings (first,
carrying the view's insight id), and the notes, while drills, the motif profile and the review
schedule stay with the main report (they are practice for every format) and the view's study
plan still points at the main report's puzzle packs; its own-puzzles action says how many of
the one puzzle file's puzzles are from its format (`pipeline.puzzle_file_formats`). The views
re-run the modules, so they carry no motif claims.

The HTML page shows the views behind sticky format tabs (radio inputs and CSS, no script); in a
view every section head says "<Format> only", and a line under the tabs names the formats
without a view and why ("Bullet (39 games) and daily (8 games) have too few games for views of
their own"). The page writes at most 12 explanation cards in the main view (`WHY_HTML_MAX`, the
ones findings link to first, 5 open: `WHY_OPEN`) and 3 in each format view (`WHY_VIEW_MAX`);
a line says how many are left out, and the Markdown report and the JSON keep every one. The
Markdown report lists the views one after another; the JSON export has them under
`format_reports`. `options["format_views"] = False` skips the views.
(`visuals.MIN_FORMAT_GAMES`, 10, is the much smaller bar for a format's own bar in a chart
split by format.)

## Statistical rules (apply everywhere)

1. Compare scores with the **Elo expected score**, not 50 %. Beating weaker
   players is not a strength.
2. Never flag anything below the module's minimum sample size (defaults: 8
   games for an opening family as one colour, 25 for a split like
   "after a loss", 30 games for an opening habit or a motif).
3. **One claim rule.** A strength or weakness needs `stats.significance(...)`
   to pass: enough games, a BH-adjusted p-value at or below the tier's alpha,
   and a confidence derived from the adjusted p. Tiers: `stats.ALPHA` (0.05)
   for pre-specified hypotheses (colour, opening families, after a loss, late
   night); `stats.STRICT_ALPHA` (0.01) for everything else. Build tests with
   `stats.summarize` / `mean_test(..., min_sd=...)` / `two_proportion_test` /
   `one_sided`. Anything that fails the rule is at most an observation.
   `tests/test_null_calibration.py` enforces the result: on data with no real
   effects, a report (its lists and every section) may contain on average no more
   than 0.3 false claims, and so may the whole page: the format views add no claim
   the main report doesn't make.
4. Shrink small-sample rates (`stats.shrink`) before sorting "best/worst" lists.
5. Standard chess only by default. Variants and abandoned 0-move games distort
   everything.
6. Don't mix time classes when a metric depends on the clock (time trouble, move
   times). Report them per time class.
7. When the same test runs over many groups (every opening family, every
   time-of-day bucket, the four opening habits, every motif), adjust with `stats.bh_adjust`
   before calling anything significant.
8. Opponent-strength effects must be significant, in the same direction,
   against both the plain Elo expectation and `stats.attenuated_expected`.
   Rating noise makes *everyone* look like they underperform against weaker
   players and overperform against stronger ones, and the true size of that
   effect is uncertain (`stats.ATTENUATION_RANGE`). Colour claims must hold for
   every White first-move edge in `stats.WHITE_EDGE_RANGE`.
9. **Compare subsets with the player's other games**, not only with the rating's
   expectation. A lagging rating (an improving player) lifts every game, so an
   opening is compared with the player's other openings of the same colour, a
   game-length band with games of other lengths, and so on. Habits that both players
   have in every game (clock use, blunders, castling, patterns missed) are compared with
   the opponents in the same games.
10. Exclude games with fewer than 4 plies (aborted starts, instant abandons) from
   skill metrics. They still count in the results totals.
11. Word findings as associations ("you score worse after 11 pm"), not causes, and
   in the report's units: "+9 per 100 games vs your rating", pawns rather than
   centipawns.
12. Say which formats a finding rests on (`Insight.formats`: games per time class), and give
   it a picture of the numbers behind it. Neither may change what is claimed.
13. The coaching layer explains and routes to practice. It makes no claims except the motif
   claims, which go through rule 3 like everything else; its deeper verdicts can only turn a
   claim into an observation.
14. A checked premise or a narrower view can take a claim away, never add one: the deeper
   verdicts, the fold rule of `analysis/structure.py` and the format views'
   `view_only_observations` all turn would-be claims into observations that say why.

## Module ownership

| Module | Responsibility |
|---|---|
| `api.py` | HTTP client for the chess.com PubAPI: User-Agent, serial requests, retries with backoff on 429/5xx, typed errors |
| `fetch.py` | Download all monthly archives into a disk cache (past months are immutable, the current month is refreshed) and load them offline |
| `parse.py` | chess.com JSON / PGN → `Game`: result mapping, termination, time control, opening name from `ECOUrl`, SAN moves + `[%clk]` clocks |
| `models.py` | The shared data contracts (`Game`, `GameEval`, `Insight`, `ModuleResult`, `StudyItem`, `Report`, the coaching types ...) |
| `context.py` | `AnalysisContext`, the one object every analysis module receives |
| `dataset.py` | Filtering and the pandas frame |
| `stats.py` | Shared statistics helpers, the claim rule, and number wording (`pct`, `per100_games`, `vs_rating`) |
| `synth.py` | Synthetic chess.com-format archives for a player with planted strengths and weaknesses (demo and recovery tests) |
| `analysis/results.py` | Score vs expected, rating trends, colour, opponent-strength buckets, time classes |
| `analysis/openings.py` | Repertoire performance by opening family/line and colour; your choices at key moves (`choice_positions`: the choice points of your main lines as positions, with each move's games, score and engine cost, which the coaching explains); repertoire breadth counted on your own moves |
| `analysis/time_mgmt.py` | Time trouble, flagging, opening pace, clock balance at moves 20 and 30, think time by move number |
| `analysis/endings.py` | How games end, game-length performance |
| `analysis/habits.py` | Sessions, tilt after losses, fatigue, time of day |
| `analysis/structure.py` | Development & king safety: castled by move 10, queen out early, knights and bishops out by move 10, pawn moves in the first 10, each paired with the opponent in the same game (colour-balanced test, one BH family at the strict alpha); the fold rule (`fold_check`: a would-be finding must hold without the one or two colour × opening-family groups whose gap stands out, else it is a "Mostly your <family> games" observation); a chart by format and a board from one game per finding |
| `engine.py` | Which games Stockfish analyses (`select_engine_games`: most recent, or balanced across formats), Stockfish analysis (parallel workers, disk cache), Lichess-style win%, accuracy, judgements, phases, tags |
| `analysis/engine_stats.py` | Accuracy, blunder rates, phase weaknesses, conversion, missed tactics, opening outcome |
| `analysis/mistakes.py` | Repeated mistakes keyed by position (EPD), up to 15 listed, each with its board; the costliest-puzzles table (Lichess links open from your side); and the `--puzzles` PGN export |
| `insights.py` | Rank and dedupe insights; group weaknesses by cause into the study plan (easiest first, targets, practice actions including the coaching's puzzle packs); the headline lines |
| `visuals.py` | Shared picture builders: format counts and wording ("blitz only"), comparison charts, split-by-format charts, boards with move arrows, strips of small boards for a line |
| `fill.py` | Fallbacks run for every report and view: `fill_formats` (the games behind a finding per format: the format its evidence names, the games that reached its position, the games of its opening and colour when the evidence's own counts confirm them, else every game the module analysed) and `fill_visuals` (a chart from the finding's evidence, or the board for a position; nothing when no pattern fits) |
| `pipeline.py` | Run everything, isolate module and coaching failures, apply the deeper verdicts (`apply_deep_verdicts`), build the `Report` and its per-format views (`MIN_FORMAT_GAMES` = 60), keeping the views to the main report's claims (`view_only_observations`) |
| `coach/__init__.py` | `build_coaching` (before ranking) and `finish_coaching` (after, with the report's games for progress), each step isolated; the rule that the layer makes no claims but the motif claims |
| `coach/config.py` | `CoachConfig` and the defaults (depth 20, 150 positions, the drill window at your level with 1200–1600 as the fallback, 20 practice minutes a day, review steps 1/3/7/21 days, the Claude model) |
| `coach/critical.py` | The positions to explain, one per position and move, at most `--coach-max`: repeated mistakes, then errors losing 8+ win-% points (costliest first), then your usual move at each choice point (which keeps a fifth of the slots left after the repeated mistakes when there are more errors than slots); standard chess only |
| `coach/deep.py` | Deeper Stockfish searches: MultiPV best line and the refutation of your move (time-capped, own cache per engine and depth); short lines for every error of both sides for the motif profile and the puzzle export |
| `coach/motifs.py` | Tactical pattern detectors on python-chess, named with Lichess's theme names; the precision gate (`GATED_THEMES`); along engine lines, only patterns the line forces, and only new ones (a pin that stood before your move is left out; `carried_by` reads the other line) |
| `coach/concepts.py` | Stockfish 16's classical evaluation terms and python-chess board facts, compared about three moves into both lines at the first settled position (`see`: no capture left that wins material); Material and Imbalance only when both ends have the same material, Winnable never |
| `coach/explain.py` | Two or three template sentences (a forced mate first; what changes hands, counted from settled positions with recaptures as recaptures; the point of the line), the board with both lines as strips, a chart of the concept differences, the concept note (only for a concept your move made worse), and `Explanation.verdict` |
| `coach/openings_info.py` | Opening names, explorer moves (masters and players above you), cloud-eval lines and Wikibooks extracts for opening positions; the "What stronger players play here" table; where you leave named theory |
| `coach/endgames.py` | Tablebase-checked endings with 7 pieces or fewer: wins not won and draws lost, by ending type, with boards |
| `coach/profile.py` | The motif profile (patterns missed and allowed per 100 moves, you vs your opponents; observations) and the motif claims ("missed" per chance with a clustered log-ratio contrast against your other tactical chances, "allowed" per move and as a share of your own errors) |
| `coach/drills.py` | Puzzle packs for your most-missed patterns and your openings, at your level (`drill_window`: your Lichess-equivalent rating ±200 inside 800–2200, or `--drill-rating`), written atomically as PGN; the review schedule |
| `coach/puzzles_db.py` | `chess-insights puzzles-db`: stream the Lichess puzzle database and keep the filtered subset |
| `coach/puzzles.py` | The richer `--puzzles` PGN: the engine's line as solution, a `Themes` header from the solution printed only, the explanation as a comment |
| `coach/progress.py` | Study-plan targets against the previous report's numbers: count metrics tested on the games since it against the earlier ones (two-proportion test on Kish effective sizes of sessions, 20+ new games, BH at 0.05), the rest side by side without a badge |
| `coach/rating_map.py` | chess.com ratings on the Lichess scale (a small editable table) and the explorer's rating groups |
| `coach/maia.py` | Optional Maia-2 probabilities of the best and the played move at your rating |
| `coach/packet.py` | The facts packet the LLM may use: claims, study plan, up to 20 explained positions |
| `coach/llm.py` | The optional LLM coach: rewrites explanations and drafts a weekly plan through the verifier; sends nothing under `--offline` |
| `coach/verify.py` | The fact check for LLM text: moves replayed along the packet's lines, numbers, captures, claims, formats, style |
| `coach/ask.py` | `chess-insights ask`: an answer from the last report's JSON through the same verifier, else (and always with `--offline`) the report's own words |
| `coach/sources/http.py` | The request layer every public source uses: one request at a time, 1 s per host, disk cache with per-source lifetimes, 429/401/timeout handling, one note per failure |
| `coach/sources/lichess_explorer.py` | Lichess opening explorer, masters and rated Lichess games (needs a token; cached 30 days) |
| `coach/sources/cloud_eval.py` | Lichess cloud evaluations, up to three lines (no key; cached 30 days) |
| `coach/sources/tablebase.py` | Lichess tablebase, 7 pieces or fewer (no key; cached for good) |
| `coach/sources/wikibooks.py` | Two credited sentences from Wikibooks Chess Opening Theory (no key; cached 90 days) |
| `coach/sources/openings_db.py` | Opening names by position from the bundled `data/openings.tsv` (lichess-org/chess-openings, CC0), and the script that rebuilds it |
| `coach/sources/concept_notes.py` | The bundled notes on positional concepts (`data/concepts.json`) and their public-domain book citations |
| `data/` | `openings.tsv`, `concepts.json` and `NOTICE.md` (where the bundled data comes from and its licences) |
| `report/html.py` | One HTML file, mobile-first, dark mode, self-contained apart from web fonts; the sticky format tabs, one board per position per view (a repeat becomes "Board shown above"), "Observation · not tested" labels, "Why these moves go wrong" (at most 12 cards, 3 per view) and "Practice" |
| `report/boards.py` | Boards as inline SVG from a FEN plus arrows and marks, one piece sprite per page; the same boards in words for Markdown |
| `report/markdown.py`, `report/json_export.py` | Markdown and JSON output (the JSON holds everything, the views and the coaching included) |
| `cli.py` | `chess-insights fetch / report / demo / puzzles-db / ask` |
| `__main__.py` | `python -m chess_insights` |
| `scripts/motif_precision.py` | Precision and recall of the motif detectors against Lichess's puzzle themes |
| `scripts/record_fixtures.py` | Records the public reference data the tests use (run by the `fixtures` workflow); sends `LICHESS_TOKEN` to explorer.lichess.org only, never on a redirect, and never records a request header |
