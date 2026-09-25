# Methodology

How chess-insights turns a pile of games into claims about your play, and why you
can trust them or not.

## 1. The benchmark: expected score, not win rate

A 55% win rate means nothing on its own. It depends on who you played. Every
result-based metric compares your score with the **Elo expected score** for each
game:

```
E = 1 / (1 + 10 ^ ((opponent_rating − your_rating) / 400))
```

The metric is the average of `score − E` over a group of games (win = 1,
draw = ½, loss = 0). "−0.10 points per game" means you score 10 percentage points
below what your ratings predict in that group. Near 50% that is about 70 Elo.

**Pre-game ratings.** chess.com reports each player's rating *after* the game, which
would slightly reward the winner in hindsight. Your pre-game rating is your
post-game rating from the previous rated game in the same pool (rules + time
control). Your opponent's is estimated by assuming a symmetric rating change.

**Rating noise.** Ratings are noisy estimates of strength, so actual results
regress toward 50%. *Everyone* appears to underperform against lower-rated players
and to overperform against higher-rated ones. The opponent-strength section
therefore judges each bucket against an attenuated expectation,
`0.5 + 0.75 × (E − 0.5)`, rather than raw `E`.

## 2. Separating signal from noise

A few hundred games carry much less information than it seems. The standard error
of an average `score − E` is roughly:

| games in the group | ± (1 standard error) | ≈ Elo |
|---|---|---|
| 20 | 0.105 | 73 |
| 50 | 0.066 | 46 |
| 100 | 0.047 | 33 |
| 400 | 0.023 | 16 |

So a 20-game opening needs a ~150 Elo effect before anything can be said. The tool:

* **Enforces minimum sample sizes** before an insight can become a strength or
  weakness.
* **Tests significance.** It uses a z-test of the mean `score − E`, and
  two-proportion tests for rates. When a test is repeated over many groups (every
  opening, every hour of the day), p-values are Benjamini–Hochberg adjusted.
* **Shrinks small groups toward the overall average** before ranking, so a 3–0
  opening doesn't top the "strengths" list.
* **Uses self-relative benchmarks** where possible. Clock usage, blunder rates and
  conversion are compared with *your opponents in the same games*. They are
  rating-matched peers facing the same positions.

Every insight carries a **severity** (how big the effect is) and a **confidence**
(how sure we are). The report ranks by their product and shows the confidence
label next to each claim.

These are associations, not causes. "You score worse after 11 pm" might be
tiredness, or it might be the different player pool at that hour. The tool
controls for opponent rating, not for everything.

## 3. Engine analysis (optional)

With `--engine`, Stockfish evaluates every position of your most recent games at a
fixed depth, and the tool applies **Lichess's published formulas** (ported from the
open-source lila/scalachess code) so that numbers are comparable to what you see on
Lichess:

* **Win %** from centipawns: `50 + 50 · (2 / (1 + e^(−0.00368208 · cp)) − 1)`,
  with cp clamped to ±1000 and mates counted as ±1000.
* **Move accuracy** from the win-% drop Δ:
  `103.1668 · e^(−0.04354 · Δ) − 3.1669 + 1`, bounded 0–100.
* **Game accuracy**: the average of a volatility-weighted mean and a harmonic mean
  of move accuracies.
* **Inaccuracy / mistake / blunder**: the move loses ≥ 5 / 10 / 15 percentage
  points of win chance. There are also mate-specific rules (a mate allowed, a mate
  lost).
* **Game phases**: Lichess's *Divider*. The middlegame starts when ≤ 10 major/minor
  pieces remain, the back rank empties, or the pieces become mixed. The endgame
  starts when ≤ 6 remain.

chess.com's own "accuracy" (CAPS2) is proprietary and uses a different scale. It
appears only for games you ran Game Review on. The report shows it separately and
never mixes it with engine accuracy.

**Repeated mistakes** are found by keying the position before each of your
engine-flagged errors by its EPD (the board, side to move, castling and en-passant
rights). Transpositions are therefore counted together, and positions where you
went wrong in two or more games are listed with the engine's preferred move.

## 4. What is excluded

* **Variants** (Chess960, bughouse, odds chess…) are excluded by default. Pass
  `--rules all` to include them.
* **Games that start from a set-up position** (thematic tournaments) are excluded
  from opening statistics.
* **Daily games** are left out of clock and session statistics. Their clocks
  measure days, and archived daily clocks store time spent rather than time
  remaining.
* **Games with fewer than 4 plies** are left out of skill metrics.
