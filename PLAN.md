# Plan

## Goal

Answer three questions from *all* of a player's chess.com games, with evidence:

1. **What am I good at?** Keep doing it and build the repertoire around it.
2. **Where do I lose points?** Rank the leaks by how many rating points they cost.
3. **What should I study next?** Concrete, game-linked actions, most valuable first.

The value comes from **scale and honesty**: hundreds or thousands of games, every claim
checked against a benchmark, and nothing said that the data cannot support.

## Phase 0 — Proof of concept (done)

| Area | What exists |
|---|---|
| Data | chess.com PubAPI client: descriptive User-Agent, serial requests, back-off, conditional requests, workarounds for the API's known quirks. Month-by-month cache. PGN/JSON import. Parsing checked against 1,285 real recorded games (a research set kept outside the repository: `tests/test_parse_real.py` skips without it) and the 32 real games in the test fixtures. The development sandbox gets 403 from api.chess.com; the downloader has since run on GitHub's machines (Phase 1b). |
| Parsing | Results, terminations, clocks, openings from `ECOUrl`, pre-game rating reconstruction, variants and set-up positions, 2009-era records. |
| Analyses | Results vs expectation, colour, opponent strength, openings (within colour), clock and time trouble, how games end, sessions/tilt/time of day. |
| Engine | Stockfish at Lichess parity: win %, accuracy, judgements, phases. Blunders by phase and under time pressure, conversion, resilience, missed tactics, opening outcome. Parallel, cached, Ctrl+C-safe. Designed for Windows (spawn-safe process start, paths, console encodings) but not yet run on Windows: the CI workflow has a Windows job for when the folder is split out. |
| Personal training | Positions you repeatedly get wrong (board diagrams; a follow-up error in the same games folded into the first), and PGN puzzles from your own mistakes, the ten costliest linked to a Lichess board. |
| Trust | One claim rule (BH-adjusted significance, two alpha tiers). Null-world calibration: about 0.2 false claims per report, counting every section (< 0.3 in every setting tested). Power tests with planted effects. Lichess parity tests. |
| Output | Phone-first HTML (light/dark), Markdown, JSON. Three headline lines (ratings, where to start, a strength); every strength and weakness as one linked line; a study plan with one item per cause, easiest first, at most three tickable actions, a target with today's number (stored in the JSON) and labelled example games; sections with charts and tables folded away. Plain units: "per 100 games vs your rating", pawns. |
| Demo | Synthetic player with 8 planted traits, played out by Stockfish. `chess-insights demo --engine --engine-games 400` finds 6 of the 8 (the Italian strength and rapid-over-blitz are too small to prove at 400 games); without `--engine` it finds 4 and reports the two engine traits as not tested. It also lists the claims that match no planted trait, and the report is marked as synthetic, with no links to real games. |

About 790 automated tests at the end of Phase 0, including 76 end-to-end CLI tests.

## Phase 1b — Pictures, formats and the coaching layer (September 2026, built)

What the user asked for after the first real report (26 Sep: a 3,354-game history, run on
GitHub; 295 of the 300 engine games were bullet, and the findings were text and chess notation
only), and the coaching-layer plan (C0–C4). All of it is built and merged on the `integration`
branch; about 1,400 automated tests now (97 of them end-to-end CLI tests).

| Area | What was built | Tested how |
|---|---|---|
| Engine sample (C0) | `--engine-sample balanced` (the same number of bullet, blitz and rapid games, most recent in each; the GitHub default) and `--engine-time-class`; the engine note says how many games of each format were analysed | Unit tests of the selection (quotas, top-up, filters, ties, replacement of games that don't replay); the full engine suite |
| Formats on every finding | `Insight.formats` on every finding ("blitz only"), `Report.formats` / `engine_formats`, and a view per format with 60+ games (its own sections, study plan and engine note), behind format tabs in the HTML; since Phase 1c a view keeps only the main report's claims | Synthetic multi-format sets; null-world runs of the views (docs/METHODOLOGY.md §2; Phase 1c below) |
| A picture for every finding | Module charts and boards (arrows for the move played and the better move, strips for lines), inline SVG boards with one piece sprite per page (`report/boards.py`), and `fill.py` giving any finding left bare a chart from its evidence or its board | Every finding of the synthetic demo report and of its format views has formats and a picture |
| Development & king safety | `analysis/structure.py`: castling by move 10, early queen moves, development, early pawn moves, paired with the opponent in the same game, colour-balanced, one BH family at the strict alpha; the fold rule for repertoire gaps (Phase 1c) | Null world: 0.234 false claims per report at 64 × 600 games, none from this section; planted habits found in 31, 32 and 32 of 32 worlds at 600 games, with the fold rule in place |
| Critical positions and explanations (C1) | `--coach`: repeated mistakes, errors and choice points re-searched at depth 20 (time-capped, cached), both lines, motifs (forcing only, precision-gated), Stockfish 16 concept terms and board facts, template explanations with boards and strips, concept notes citing public-domain classics; the richer `--puzzles` PGN | Stockfish 16 locally; hand-built positions; the three habit positions of the coaching plan |
| Motifs and drills (C2) | Motif detectors with Lichess theme names and a precision gate (core themes 0.96–1.00 on 1,573 Lichess puzzles); the motif profile and the motif claims (two tests each, BH, strict alpha, ratio 1.3; "missed" judged per chance since Phase 1c); puzzle packs from the Lichess puzzle database (`chess-insights puzzles-db`), at your level since Phase 1c; the review schedule (1, 3, 7, 21 days) | The puzzle sample; simulated motif counts, including coupled worlds (Phase 1c); a drill-subset sample |
| Public sources (C3) | Lichess opening explorer (token), cloud eval, tablebase endings, bundled chess-openings names, Wikibooks extracts with credit, where you leave opening theory; polite HTTP layer with caches | Answers recorded from the real services on 26 Sep by the `fixtures` workflow; a guard fails any test that reaches the network |
| Progress, Maia and the LLM coach (C4) | Progress against the previous report's targets (badges only from a test since Phase 1c); optional Maia-2 probabilities; `--coach-llm` (explanations and a weekly plan through the verifier) and `chess-insights ask` | The LLM through a mocked client and adversarial texts for the verifier; Maia through a stub predictor |
| GitHub run | Balanced engine sample and coaching by default, the previous report from the `reports` branch, the puzzle database cached by month, the two optional secrets passed through the environment only, drill packs saved with the report (the cache, time limits and publishing were reworked in Phase 1c) | A text-level check of the workflow file |

The coaching explains and routes to practice. The only new claims it can make (patterns you
miss or allow more often than your opponents) go through the claim rule; they are calibrated on
simulated counts, since the null world has no engine lines.

## Phase 1c — Review fixes (26 September 2026, built)

A cross-cutting review of the integrated branch (statistical honesty, chess correctness, the
report on a phone, security, an end-to-end run of the demo) found places where the page said
more than its tests support, or said a chess fact wrongly. All of it is fixed on `integration`
(`git log 95a286e..a75490f`); about 1,540 automated tests now (106 in `tests/test_cli.py`).

| Area | What changed | Checked how |
|---|---|---|
| Format views | A view's strength or weakness that the main report does not claim is an observation in that view ("Seen in your blitz games only, and not strong enough across all your games to call it a finding: it may be chance"), card, chart and board kept; so the page claims nothing the main report doesn't (`pipeline.view_only_observations`). A view's explanations include its findings' own (a repeated position spans formats); its puzzle action counts its format's puzzles in the one file | `tests/test_null_calibration.py` asserts it on the null world. Distinct false claims per page: 0.20 at 400 games (60 reports) and 0.10 at 1,200 (40 reports), the main report's own (0.50 and 0.58 before; counting every view's claims, 0.62 at both before and 0.32 and 0.15 now). A view's repeats of main-report claims at 400 games: bullet 0.000, blitz 0.067, rapid 0.050 per view |
| Deeper verdicts | `Explanation.verdict` ("error", "close", "fine"); `pipeline.apply_deep_verdicts` turns a repeated-mistake finding whose move the deeper search clears into an observation (its wording, the section summary and the board without the green arrow to match), in the main report and every view, before ranking; the explanation card says "Deeper check: not a mistake at depth N" | `tests/test_format_views.py`, `tests/test_report_phone.py` |
| Progress | Count targets compare the share among the games since the previous report with the share before: a two-proportion test on Kish effective sizes of sessions, at least 20 new games, Benjamini–Hochberg across the tested lines at 0.05; "Improved" / "Not yet" only when it passes. Other metrics are shown side by side, without a badge | `tests/test_progress.py` (the mechanics). The false-badge rate and power on realistic schedules are not recorded with the tests yet |
| Motif claims | "Missed" judged per chance (of the patterns your opponents' errors left you) plus a clustered log-ratio contrast against your other tactical chances; "allowed" per move and as a share of your errors, leaving out missed chances. The profile chart and table say they are observations | Simulated counts (`tests/test_profile.py`): 0 to 0.033 false claims per report in same-rate, coupled and every-chance worlds (the old tests: 0.9 to 26 in the coupled ones). Missing forks twice as often per chance: found in 30/40, 39/40, 40/40 worlds at 300/600/1,000 analysed games; 1.5 times as often: 14/26/35 of 40 |
| Development & king safety | The fold rule: a would-be finding must still pass without the one or two (colour, opening family) groups whose gap stands out (re-tested with BH); otherwise it is a "Mostly your <family> games" observation pointing at the opening's main lines | `line_habits` null variant (the opening line sets both sides' habits): no claim from this section in 80 reports of 600 games (36 of 80 without the rule). Null calibration unchanged at 0.234 (64 × 600); planted habits 31/32, 32/32, 32/32 |
| Chess wording | Concepts and material read at settled comparison points (a static exchange count); what changes hands, counted from before a capture that can still be taken back (a recapture is not a win); pins that stood before the move and patterns the better line allows too not blamed on the move; a forced mate first; the concept note only for a concept the move made worse (never Material, Imbalance or Winnable); Winnable never shown; puzzle `Themes` only from the solution printed | Recorded Stockfish 16 lines of the reviewed positions (`tests/test_explain_review.py`), `tests/test_concepts.py`, `tests/test_motifs.py` |
| Report | Sticky format tabs; section heads say the view's format; a line names the formats without a view; at most 12 explanation cards in the HTML (5 open; 3 per view; all in the Markdown and JSON); one board per position per view; a board for every repeated-mistake row; blue for the move that scored best for you, green only for the engine's move; "Observation · not tested" labels; Lichess links open from your side | `tests/test_report_phone.py`. The demo player's page with 150 explanations (`--coach-max 150`): 1.4 MB, against 4.4 MB before |
| Workflow and CLI | Report workflow: cache restore and save split, saved even when the analysis fails, the analysis step limited to 80 minutes, a warning for runs that may not fit, a lowercase cache key with the old key as a second restore key, the `llm` extra installed only when `ANTHROPIC_API_KEY` is set, the publish retried after another run's. Fixtures workflow: the recorder sends `LICHESS_TOKEN` to the opening explorer only. CLI: `--offline` sends nothing to Claude and `ask --offline` quotes the report; the drill window follows your level and `--drill-rating` is validated; the puzzle PGN and drill packs are written atomically; the engine note says "at least 10 plies" | `tests/test_report_workflow.py` (text-level), `tests/test_record_fixtures.py`, `tests/test_cli.py`, `tests/test_drills.py`, `tests/test_llm.py`, `tests/test_ask.py`. None of the workflow changes has run on GitHub yet |

### Not yet run for real

* **chess.com from this machine.** The development sandbox gets 403 from api.chess.com. The
  downloader has run on GitHub's machines (two reports on 26 Sep), but not from a home
  connection, and nothing has run on Windows yet.
* **Lichess and Wikibooks from this machine.** The sandbox cannot reach them. The code's live
  path has not run anywhere: the tests replay answers the `fixtures` workflow recorded on
  GitHub. The opening explorer answered 401 without a token there, so the explorer with a token
  has never been exercised (the fixtures workflow now passes the `LICHESS_TOKEN` secret to the
  recorder, but has not run with it yet), and `chess-insights puzzles-db` has never downloaded
  the real database (the fixtures workflow fetched it with its own code for the test samples).
* **The Claude API.** `--coach-llm` and `ask` have only met a mocked client; no real key has
  been used, so neither the model's texts nor the verifier's rejection rate on them are known.
  That `--offline` sends nothing is tested with the same mock.
* **Maia-2.** The `maia2` package has never been installed or run; only the stub is tested.
* **The GitHub workflow's new steps**: the balanced default, the coaching, the monthly puzzle
  database cache, fetching the previous report, the two secrets and saving the drill packs, and
  from Phase 1c: the cache restored and saved in separate steps (saved even after a failed or
  timed-out analysis), the 80-minute limit on the analysis step, the long-run warning, the
  lowercase cache key and its fallback to the old key, the `llm` extra installed only when the
  `ANTHROPIC_API_KEY` secret is set, and the retried publish. The tests only read the workflow
  file.
* **A full `--coach` run on a real history**: how long the deeper searches, the pattern profile
  and the first round of Lichess and Wikibooks requests take on GitHub's four cores (the
  workflow file's comment estimates about 15 minutes for the defaults; no run has timed it), and
  how the explanations read on real games.
* **Progress on real report pairs.** The count test has only run on constructed schedules in
  the tests; its false-badge rate and power on realistic schedules are not recorded with them.
* **The automatic drill window on real ratings.** It rests on the rating map's rough table
  (two checked rows).

## Phase 1 — Validate on your real games (next)

* **Run the new workflow.** Push `integration` to `main` and run the report for the players in
  `players.txt`; check each new step in the run log, the time it takes, the drill packs on the
  `reports` branch, and the coaching's notes. Then add `LICHESS_TOKEN` and `ANTHROPIC_API_KEY`
  (the workflow now installs the `llm` extra when the key is set) and check what they add, and
  run the `fixtures` workflow once with the token to record the explorer's real answers. Check
  on a deliberately long run that the cache is saved when the analysis stops at 80 minutes and
  that the next run carries on.
* **Decide the licence.** `pyproject.toml` says MIT, while python-chess (a required dependency
  whose piece drawings every HTML report embeds) is GPL-3.0-or-later: the owner's call, before
  the package or reports are published (Known limitations).
* **First live run from home.** Run `chess-insights report <you> --engine --coach --tz <your
  time zone> -v` and keep the console output if anything goes wrong.
* Read the report critically against what you know about your chess. Note anything
  surprising, wrong or missing, and read the explanations of positions you remember: is the
  named pattern the one you missed, and does the text say what a coach would?
* Spot-check a few games against Lichess's own computer analysis, which uses the same
  formulas (chess.com's Game Review accuracy is on a different scale and is not a check).
* **Decide the scope.** For example, the last 12–24 months of rated blitz and rapid.
  More games give more power: many effects need 600–1,200 games to be found reliably
  (docs/METHODOLOGY.md §2).
* Decide how many games to analyse with the engine. At depth 12 it takes about 2.5 s
  of engine time per game on the measured cloud machine (28 positions per second per
  engine), less on a recent desktop core; on GitHub, 1,000 games at depth 15 took 32
  minutes. The default is the 150 most recent (300 balanced on GitHub), and
  `--engine-games all` is realistic overnight.

## Phase 2 — Deeper insights

Roughly in order of value for effort (the research catalogue's top-15 list, plus what
the first reviews of the demo report asked for). Phases 1b and 1c built parts of several items;
they stay listed until they have been checked on real reports:

1. **Spaced repetition for your puzzles.** Built: your costliest mistakes and the drill
   packs' first puzzles come back 1, 3, 7 and 21 days after each report, carried over through
   the previous report's JSON. Still to do: record solved / failed (the page's tick boxes stay
   in the browser) and re-test the missed ones sooner.
2. **Tactical motif labels** for missed tactics and blunders (fork, pin, skewer,
   discovered attack, back rank, hanging piece). Built: detectors along the engine lines, the
   motif profile, the motif claims and themed puzzle packs.
3. **Punish rate.** How often you exploit your opponent's mistakes, compared with
   how often they exploit yours (Lichess "awareness").
4. **Endgame types.** Built: tablebase-checked endings with 7 pieces or fewer (wins not won,
   draws lost) by ending type, from the Lichess tablebase. Still to do: error rates in every
   rook, king-and-pawn, minor-piece and queen ending, and local Syzygy tablebases.
5. **Opening depth.** Built: where each game leaves named theory (the bundled Lichess
   chess-openings data, offline) and your first inaccuracy per line.
6. **Structural features.** Built: castling by move 10, early queen moves, development and
   early pawn moves, against the opponent in the same games, with the fold rule for gaps that
   come from the repertoire. Still to do: results by castling side, opposite-side castling and
   queen trades.
7. **Clock behaviour.** Instant moves in non-forced positions, losing with lots of
   time left, and flagging from winning positions (engine-confirmed).
8. **Critical moments.** Built: the costliest positions are explained with both lines. Still to
   do: the largest evaluation swings in each game, and how much clock you spent on them.
9. **Example games that open the critical position.** Explained positions are drawn as boards
   linked to their game; findings still carry bare game links (labelled "Loss · 5+0 · vs 1512 ·
   12 Aug"). Carrying the move number would let every engine finding open the position itself
   and say "flagged at move 38".
10. **Thrown half-points and blown wins, engine-free.** From the final position's
   material (or the engine eval when there is one): losses on time and draws from winning
   positions, mated while ahead, resigning in positions that were not lost yet (catalogue
   END-2, END-3, END-4, TIM-2b), and the full list of blown wins.
11. **Result once an endgame is reached** (catalogue LEN-2), and a quick re-queue after a
   loss (starting the next game within a minute or two, HAB-2) as a sharper tilt signal.
12. **Opening position tree.** Drill down from the family to the moves where your games
   diverge. Built: the choice points of your main lines as boards, with what masters and
   stronger Lichess players play there (with a token). Still to do: the move-10 evaluation per
   branch.
13. **Rapid vs blitz as a finding.** "Your chess is fine, your blitz clock isn't" needs a
   test that separates the clock from the chess (per-format engine accuracy against the
   same opponents); today each format's view shows its own score against its rating, a
   pattern seen in one format only is an observation there, and no test compares the views.
14. **Drills at your level.** Built in Phase 1c: without `--drill-rating`, the packs are rated
   around your Lichess-equivalent rating (±200, inside 800–2200). Still to do: check the
   rating map's rows against more data, and follow your solved / failed puzzles.

## Phase 3 — Coaching layer

* **LLM coach, progress tracking, a weekly schedule.** Built in Phase 1b (C4: `--coach-llm`,
  `chess-insights ask`, progress against the previous report's targets, the weekly plan), not
  yet run against the real API. Still to do once it has run on real reports: tune the plan's
  minutes to what you actually practise, and track solved/failed puzzles between reports.
* **Per-format reports in one go.** Done in Phase 1b: every format with 60+ games gets its
  own view inside the report.

## Phase 4 — App

* Scheduled refresh (cron or a GitHub Action) and a small web dashboard, reusing the
  FastAPI + React stack from the trading bot dashboard.
* Multiple accounts: compare with friends, or follow a student.

## Known limitations

| Limitation | Notes |
|---|---|
| Statistical power | Honest tests need data. At 400 games, opening and colour effects are often not significant yet. The report stays quiet rather than guess. |
| Lagging ratings | Improving players outperform their rating everywhere. Subsets are compared with your other games, so this doesn't create false findings, but "better at rapid than blitz" style effects are hard to see. |
| chess.com accuracy | CAPS2 is proprietary. Engine accuracy follows Lichess's formulas instead, and chess.com's number is shown separately where it exists. |
| Engine time | Minutes for hundreds of games at depth 12. Cached, so each game is analysed only once per engine version and depth. The coaching's searches are capped per position and cached too. |
| Stockfish 16 | The concept terms need Stockfish 16, the last version that prints its classical evaluation; with another version the explanations use board facts only. |
| Pattern names | The detectors follow Lichess's own tagger, so the precision gate shows faithful definitions, not agreement with a human coach; along engine lines precision is lower (0.89–1.00 at 8 plies on the core themes), and about half of Lichess's discovered attacks go unnamed. |
| Rating map | chess.com-to-Lichess ratings come from a small table with two checked rows; the explorer's "players above you" and Maia's "at your rating" are approximate. |
| LLM verifier | Cannot check evaluations in words without a number, whose move an unnumbered piece move is, or chess ideas said in words only. |
| Online sources | Lichess and Wikibooks answer one request at a time, at least a second apart; the first coached report of a player spends a few minutes on them, later ones use the cache. |
| Install size | pandas is a dependency although no analysis module reads the per-game frame any more (about 25 MB on Windows); it could become optional. Maia-2 brings PyTorch. |
| Licence | `pyproject.toml` declares MIT, but python-chess, a required dependency whose piece drawings every HTML report embeds, is GPL-3.0-or-later. Open decision for the owner before the package or its reports are published: relicense, redraw the pieces, or accept and document the combination. |
| Format-only effects | A view keeps only the main report's claims, so a real effect that shows in one format alone is at most an observation in its view, however strong it is there: one error budget per page costs that power. |
| Deeper verdicts | A repeated mistake is re-checked only when the coaching explains it (repeated mistakes come first, up to `--coach-max`); without `--coach` the quicker game analysis decides. |
| Progress | Only count targets get a verdict, and only from 20 new games whose sessions are enough to test; its error rates on realistic schedules are not recorded yet. |
| Associations, not causes | "You score worse late at night" could be tiredness or a different player pool at that hour. |
