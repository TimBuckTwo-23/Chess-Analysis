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
| Data | chess.com PubAPI client: descriptive User-Agent, serial requests, back-off, conditional requests, workarounds for the API's known quirks. Month-by-month cache. PGN/JSON import. Checked against ~1,300 real recorded games. |
| Parsing | Results, terminations, clocks, openings from `ECOUrl`, pre-game rating reconstruction, variants and set-up positions, 2009-era records. |
| Analyses | Results vs expectation, colour, opponent strength, openings (within colour), clock and time trouble, how games end, sessions/tilt/time of day. |
| Engine | Stockfish at Lichess parity: win %, accuracy, judgements, phases. Blunders by phase and under time pressure, conversion, resilience, missed tactics, opening outcome. Parallel, cached, Ctrl+C-safe, Windows-safe. |
| Personal training | Positions you repeatedly get wrong (board diagrams), and PGN puzzles from your own mistakes. |
| Trust | One claim rule (BH-adjusted significance, two alpha tiers). Null-world calibration: < 0.3 false claims per report. Power tests with planted effects. Lichess parity tests. |
| Output | Mobile-first HTML (light/dark), Markdown, JSON; ranked strengths, weaknesses and a study plan. |
| Demo | Synthetic player with 8 planted traits, played out by Stockfish; `chess-insights demo` shows which traits are recovered. |

About 770 automated tests, including 74 end-to-end CLI tests.

## Phase 1 — Validate on your real games (next)

* Run `chess-insights report <you> --engine --tz <your time zone>` and read the
  report critically against what you know about your chess. Note anything
  surprising, wrong or missing.
* Compare the engine accuracy with chess.com's Game Review accuracy on games that have
  both, to confirm the depth setting.
* **Decide the scope.** For example, the last 12–24 months of rated blitz and rapid.
  More games give more power: many effects need 600–1,200 games to be found reliably
  (docs/METHODOLOGY.md §2).
* Decide how many games to analyse with the engine. At depth 12 it takes about 1 s
  of CPU per game on a modern desktop; the default is the 150 most recent, and all
  of them is realistic overnight.

## Phase 2 — Deeper insights (ranked by the research catalogue)

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

## Phase 3 — Coaching layer

* **LLM coach.** The Claude API reads the JSON report and writes a weekly plan in
  plain language. You can ask it follow-up questions grounded in your own games
  ("why do I lose with the Caro-Kann?").
* **Progress tracking.** Re-run monthly and compare with the last report
  ("Caro-Kann: −0.15 → −0.03 points per game since you started studying it").

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
| Associations, not causes | "You score worse late at night" could be tiredness or a different player pool at that hour. |
