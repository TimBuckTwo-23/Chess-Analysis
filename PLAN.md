# Plan

## Goal

Answer three questions from *all* of a player's chess.com games, with evidence:

1. **What am I good at?** Keep doing it and build the repertoire around it.
2. **Where do I lose points?** Rank the leaks by how many rating points they cost.
3. **What should I study next?** Concrete, game-linked actions, most valuable first.

The value comes from **scale and honesty**: hundreds or thousands of games, every claim
checked against a benchmark (your Elo expectation, or your opponents in the same
games), and no claims from tiny samples.

## Phase 0 — Proof of concept (this repository)

| Area | Status |
|---|---|
| chess.com PubAPI client: descriptive User-Agent, serial requests, retries with back-off, conditional requests (ETag), workarounds for known API quirks | done |
| Local month-by-month cache: re-runs only fetch the current month | done |
| Parser for chess.com JSON and plain PGN: results, terminations, clocks, openings from `ECOUrl`, pre-game rating reconstruction. Verified on ~1,200 real recorded games | done |
| Analyses: results vs expectation, openings, clock, how games end, habits and tilt | done |
| Optional Stockfish review: Lichess-style accuracy, blunder rates by phase, time pressure, conversion, missed tactics, opening outcome | done |
| Ranked strengths, weaknesses and study plan; HTML, Markdown and JSON report | done |
| Synthetic player with planted traits: a demo, plus a check that the analysis recovers them | done |

## Phase 1 — Validate on your real games (next)

* Run `chess-insights report <you> --engine`. Sanity-check the report against what
  you already know about your chess.
* Compare the engine accuracy with chess.com's Game Review accuracy on games that have
  both. This calibrates the engine depth.
* Tune the thresholds (minimum games, effect sizes) if the report is too cautious or
  too eager for your volume of games.
* Decide on the default scope, e.g. last 12 months of rated blitz and rapid.

## Phase 2 — Deeper engine insights

* **Personal puzzle deck.** Export every position where you blundered (FEN + best
  move) as a PGN study or CSV for spaced repetition. Training on your own mistakes is
  the highest-leverage study there is.
* **Tactical motif labels** for missed tactics (fork, pin, skewer, discovered attack,
  back-rank, hanging piece), derived from python-chess attack maps along the engine's
  principal variation.
* **Endgame types**: error rates in rook, king-and-pawn, minor-piece and queen endings,
  so "study endgames" becomes "study rook endgames".
* **Opening depth**: the move where you usually leave known theory and the first
  inaccuracy per opening line, using a masters opening explorer.
* **Critical moments**: the biggest evaluation swings, and how much clock you spent on
  them.

## Phase 3 — Coaching layer

* An LLM coach (the Claude API) that reads the JSON report and writes a weekly plan in
  plain language. You can ask it follow-ups ("why do I lose with the Caro-Kann?"),
  grounded in your own games.
* **Progress tracking.** Re-run monthly and compare against the previous report
  ("Caro-Kann: −0.15 → −0.03 points per game since you studied it").

## Phase 4 — App

* A scheduled refresh (cron or a GitHub Action) and a small web dashboard, reusing the
  FastAPI + React stack from the trading bot dashboard.
* Multiple accounts: compare with friends, or track a student.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| chess.com blocks or rate-limits the client | Serial requests, descriptive User-Agent with contact, caching; PGN import as a fallback |
| Small samples produce false "weaknesses" | Minimum sample sizes, significance tests, shrinkage, and confidence shown on every insight |
| Opponent strength confounds win rates | Everything is compared with the Elo expected score |
| chess.com accuracy (CAPS2) is proprietary | Use it where present; compute Lichess-style accuracy locally with Stockfish |
| Engine analysis is slow on thousands of games | Most recent N games, parallel workers, per-game cache |
