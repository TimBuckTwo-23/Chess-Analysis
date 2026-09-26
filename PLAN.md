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
| Formats on every finding | `Insight.formats` on every finding ("blitz only"), `Report.formats` / `engine_formats`, and a view per format with 60+ games (its own claims, sections, study plan and engine note), behind a switcher in the HTML | Synthetic multi-format sets; null-world runs of the views (docs/METHODOLOGY.md §2: about 0.1–0.3 false claims per view, 0.5 across all views, re-measured after integration) |
| A picture for every finding | Module charts and boards (arrows for the move played and the better move, strips for lines), inline SVG boards with one piece sprite per page (`report/boards.py`), and `fill.py` giving any finding left bare a chart from its evidence or its board | Every finding of the synthetic demo report and of its format views has formats and a picture |
| Development & king safety | `analysis/structure.py`: castling by move 10, early queen moves, development, early pawn moves, paired with the opponent in the same game, colour-balanced, one BH family at the strict alpha | Null world: 0.234 false claims per report at 64 × 600 games, none from this section; planted habits found in 31, 32 and 32 of 32 worlds at 600 games |
| Critical positions and explanations (C1) | `--coach`: repeated mistakes, errors and choice points re-searched at depth 20 (time-capped, cached), both lines, motifs (forcing only, precision-gated), Stockfish 16 concept terms and board facts, template explanations with boards and strips, concept notes citing public-domain classics; the richer `--puzzles` PGN | Stockfish 16 locally; hand-built positions; the three habit positions of the coaching plan |
| Motifs and drills (C2) | Motif detectors with Lichess theme names and a precision gate (core themes 0.96–1.00 on 1,573 Lichess puzzles); the motif profile and the motif claims (two tests, BH, strict alpha, ratio 1.3); puzzle packs from the Lichess puzzle database (`chess-insights puzzles-db`); the review schedule (1, 3, 7, 21 days) | The puzzle sample; simulated motif counts (no false claims in 200 null sets, 17 of 20 planted effects found); a drill-subset sample |
| Public sources (C3) | Lichess opening explorer (token), cloud eval, tablebase endings, bundled chess-openings names, Wikibooks extracts with credit, where you leave opening theory; polite HTTP layer with caches | Answers recorded from the real services on 26 Sep by the `fixtures` workflow; a guard fails any test that reaches the network |
| Progress, Maia and the LLM coach (C4) | Progress against the previous report's targets; optional Maia-2 probabilities; `--coach-llm` (explanations and a weekly plan through the verifier) and `chess-insights ask` | The LLM through a mocked client and adversarial texts for the verifier; Maia through a stub predictor |
| GitHub run | Balanced engine sample and coaching by default, the previous report from the `reports` branch, the puzzle database cached by month, the two optional secrets passed through the environment only, drill packs saved with the report | A text-level check of the workflow file |

The coaching explains and routes to practice. The only new claims it can make (patterns you
miss or allow more often than your opponents) go through the claim rule; they are calibrated on
simulated counts, since the null world has no engine lines.

### Not yet run for real

* **chess.com from this machine.** The development sandbox gets 403 from api.chess.com. The
  downloader has run on GitHub's machines (two reports on 26 Sep), but not from a home
  connection, and nothing has run on Windows yet.
* **Lichess and Wikibooks from this machine.** The sandbox cannot reach them. The code's live
  path has not run anywhere: the tests replay answers the `fixtures` workflow recorded on
  GitHub. The opening explorer answered 401 without a token there, so the explorer with a token
  has never been exercised, and `chess-insights puzzles-db` has never downloaded the real
  database (the fixtures workflow fetched it with its own code for the test samples).
* **The Claude API.** `--coach-llm` and `ask` have only met a mocked client; no real key has
  been used, so neither the model's texts nor the verifier's rejection rate on them are known.
* **Maia-2.** The `maia2` package has never been installed or run; only the stub is tested.
* **The GitHub workflow's new steps**: the balanced default, the coaching, the monthly puzzle
  database cache, fetching the previous report, the two secrets and saving the drill packs.
  Known gap: the workflow installs the package without the `llm` extra, so an
  `ANTHROPIC_API_KEY` secret has no effect yet (the report notes that the AI coach was
  skipped).
* **A full `--coach` run on a real history**: how long the deeper searches, the pattern profile
  and the first round of Lichess and Wikibooks requests take on GitHub's four cores, and how the
  explanations read on real games.

## Phase 1 — Validate on your real games (next)

* **Run the new workflow.** Push `integration` to `main` and run the report for the players in
  `players.txt`; check each new step in the run log, the time it takes, the drill packs on the
  `reports` branch, and the coaching's notes. Install the `llm` extra in the workflow when the
  API key is set, then add `LICHESS_TOKEN` and `ANTHROPIC_API_KEY` and check what they add.
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
the first reviews of the demo report asked for). Phase 1b built parts of several items; they
stay listed until they have been checked on real reports:

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
   early pawn moves, against the opponent in the same games. Still to do: results by castling
   side, opposite-side castling and queen trades.
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
   same opponents); today each format's view shows its own score against its rating, and no
   test compares the views.
14. **Drills at your level.** The drill packs use a fixed rating window (1200–1600 unless
   `--drill-rating` is given); derive it from your rating through the rating map.

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
| Licence | `pyproject.toml` declares MIT, but python-chess, a required dependency whose piece drawings every HTML report embeds, is GPL-3.0-or-later. This needs a decision before the package is published. |
| Associations, not causes | "You score worse late at night" could be tiredness or a different player pool at that hour. |
