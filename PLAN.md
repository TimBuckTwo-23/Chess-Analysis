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
| Data | chess.com PubAPI client: descriptive User-Agent, serial requests, back-off, conditional requests, workarounds for the API's known quirks. Month-by-month cache. PGN/JSON import. Parsing checked against 1,285 real recorded games (a research set kept outside the repository: `tests/test_parse_real.py` skips without it) and the 32 real games in the test fixtures. Not yet run live against chess.com: the development sandbox gets 403 from api.chess.com. |
| Parsing | Results, terminations, clocks, openings from `ECOUrl`, pre-game rating reconstruction, variants and set-up positions, 2009-era records. |
| Analyses | Results vs expectation, colour, opponent strength, openings (within colour), clock and time trouble, how games end, sessions/tilt/time of day. |
| Engine | Stockfish at Lichess parity: win %, accuracy, judgements, phases. Blunders by phase and under time pressure, conversion, resilience, missed tactics, opening outcome. Parallel, cached, Ctrl+C-safe. Designed for Windows (spawn-safe process start, paths, console encodings) but not yet run on Windows: the CI workflow has a Windows job for when the folder is split out. |
| Personal training | Positions you repeatedly get wrong (board diagrams; a follow-up error in the same games folded into the first), and PGN puzzles from your own mistakes, the ten costliest linked to a Lichess board. |
| Trust | One claim rule (BH-adjusted significance, two alpha tiers). Null-world calibration: about 0.2 false claims per report, counting every section (< 0.3 in every setting tested). Power tests with planted effects. Lichess parity tests. |
| Output | Phone-first HTML (light/dark), Markdown, JSON. Three headline lines (ratings, where to start, a strength); every strength and weakness as one linked line; a study plan with one item per cause, easiest first, at most three tickable actions, a target with today's number (stored in the JSON) and labelled example games; sections with charts and tables folded away. Plain units: "per 100 games vs your rating", pawns. |
| Demo | Synthetic player with 8 planted traits, played out by Stockfish. `chess-insights demo --engine --engine-games 400` finds 6 of the 8 (the Italian strength and rapid-over-blitz are too small to prove at 400 games); without `--engine` it finds 4 and reports the two engine traits as not tested. It also lists the claims that match no planted trait, and the report is marked as synthetic, with no links to real games. |

About 790 automated tests, including 76 end-to-end CLI tests.

## Phase 1 — Validate on your real games (next)

* **First live run.** The downloader has only seen recorded chess.com responses so far
  (see Phase 0). Run `chess-insights report <you> --engine --tz <your time zone> -v` from
  home and keep the console output if anything goes wrong.
* Read the report critically against what you know about your chess. Note anything
  surprising, wrong or missing.
* Spot-check a few games against Lichess's own computer analysis, which uses the same
  formulas (chess.com's Game Review accuracy is on a different scale and is not a check).
* **Decide the scope.** For example, the last 12–24 months of rated blitz and rapid.
  More games give more power: many effects need 600–1,200 games to be found reliably
  (docs/METHODOLOGY.md §2).
* Decide how many games to analyse with the engine. At depth 12 it takes about 2.5 s
  of engine time per game on the measured cloud machine (28 positions per second per
  engine), less on a recent desktop core; the default is the 150 most recent, and
  `--engine-games all` is realistic overnight.

## Phase 2 — Deeper insights

Roughly in order of value for effort (the research catalogue's top-15 list, plus what
the first reviews of the demo report asked for):

1. **Spaced repetition for your puzzles.** Schedule the exported positions, track
   solved/failed, and re-test missed ones.
2. **Tactical motif labels** for missed tactics and blunders (fork, pin, skewer,
   discovered attack, back rank, hanging piece), from python-chess attack maps
   along the engine line. "Study tactics" then becomes "study forks".
3. **Punish rate.** How often you exploit your opponent's mistakes, compared with
   how often they exploit yours (Lichess "awareness").
4. **Endgame types.** Error rates in rook, king-and-pawn, minor-piece and queen
   endings, plus local Syzygy tablebases to count won endings that weren't converted.
5. **Opening depth.** The move where you leave known theory in each line, from the
   Lichess chess-openings database (open data, usable offline), and your first
   inaccuracy per line.
6. **Structural features.** Results by castling side, opposite-side castling and
   queen trades.
7. **Clock behaviour.** Instant moves in non-forced positions, losing with lots of
   time left, and flagging from winning positions (engine-confirmed).
8. **Critical moments.** The largest evaluation swings in each game, and how much
   clock you spent on them.
9. **Example games that open the critical position.** Findings carry bare game links
   today (labelled "Loss · 5+0 · vs 1512 · 12 Aug"); carrying the move number would let
   engine findings open the position itself (as a Lichess board, like the puzzles) and
   say "flagged at move 38".
10. **Thrown half-points and blown wins, engine-free.** From the final position's
   material (or the engine eval when there is one): losses on time and draws from winning
   positions, mated while ahead, resigning in positions that were not lost yet (catalogue
   END-2, END-3, END-4, TIM-2b), and the full list of blown wins.
11. **Result once an endgame is reached** (catalogue LEN-2), and a quick re-queue after a
   loss (starting the next game within a minute or two, HAB-2) as a sharper tilt signal.
12. **Opening position tree.** Drill down from the family to the moves where your games
   diverge (the "Your choices at key moves" table is the first step), with the move-10
   evaluation per branch.
13. **Rapid vs blitz as a finding.** "Your chess is fine, your blitz clock isn't" needs a
   test that separates the clock from the chess (per-format engine accuracy against the
   same opponents); today the report only shows each format's score against its rating.

## Phase 3 — Coaching layer

* **LLM coach.** The Claude API reads the JSON report and writes a weekly plan in
  plain language. You can ask it follow-up questions grounded in your own games
  ("why do I lose with the Caro-Kann?").
* **Progress tracking.** Each study-plan item already stores its metric and today's
  value in the JSON (`study_plan[].baseline`). Next: read the previous report's JSON and
  print "clock used by move 15: 50% → 38%" next to each target.
* **A weekly schedule.** Minutes per item per week and when to re-check, as the research
  catalogue's study-plan design describes (§5); the plan has the targets but no timetable.
* **Per-format reports in one go.** Openings, habits and engine findings are pooled over
  all time controls; `--time-class blitz --out reports/me-blitz` gives one format today.

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
| Engine time | Minutes for hundreds of games at depth 12. Cached, so each game is analysed only once per engine version and depth. |
| Install size | pandas is a dependency although no analysis module reads the per-game frame any more (about 25 MB on Windows); it could become optional. |
| Associations, not causes | "You score worse late at night" could be tiredness or a different player pool at that hour. |
