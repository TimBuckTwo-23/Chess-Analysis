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
and to overperform against higher-rated ones. How strongly is uncertain: the
attenuated expectation `0.5 + a × (E − 0.5)` fits real results for some `a`
between about 0.75 (a pessimistic guess) and 1.0 (plain Elo); a realistic 75-Elo
noise on the rating difference gives about 0.94. The opponent-strength section
shows the bucket numbers against `a = 0.75`, but only makes a claim that holds for
*every* `a` in that range (see section 2).

**Colour.** At equal ratings White scores about 52% and Black about 48%. Groups
that are all one colour (an opening as Black) are judged against that
colour-adjusted expectation, `E ± 0.02`; otherwise every White opening would drift
toward "strength" and every Black one toward "weakness".

## 2. Separating signal from noise

A few hundred games carry much less information than it seems. The standard error
of an average `score − E` is roughly:

| games in the group | ± (1 standard error) | ≈ Elo |
|---|---|---|
| 20 | 0.105 | 73 |
| 50 | 0.066 | 46 |
| 100 | 0.047 | 33 |
| 400 | 0.023 | 16 |

So a 20-game opening needs a ~150 Elo effect before anything can be said.

### The claim rule

Something becomes a **strength or weakness** only if all three hold; otherwise it
is at most an *observation* (shown in its section, never in the top lists or the
study plan):

1. **Enough games** (module minimums, e.g. 8 rated games in an opening family as one
   colour, 20–25 in a split such as "right after a loss").
2. **A significant test after multiple-testing adjustment.** z-tests of the mean
   `score − E`, difference tests between two groups, paired tests against your
   opponents in the same games, two-proportion tests for rates. When a test is
   repeated over a family of groups (every opening, every time class, every block
   of the day) the p-values are Benjamini–Hochberg adjusted, and the adjusted p must
   be at or below the family's α (next section).
3. **An effect big enough to matter** (e.g. 0.06 points per game, about 40 Elo).

This rule lives in one place, `stats.significance`, and every module uses it.
An insight's **confidence** is `1 − adjusted p`, scaled down below three times the
minimum sample, so every claim has a confidence of at least 0.55 and "high
confidence" (≥ 0.7) needs a clear result on a decent sample. Small groups are also
**shrunk toward 0** (`stats.shrink`) before ranking or charting, so a 3–0 opening
doesn't top the "strengths" list, and small streaky samples never get a standard
error below that of a player scoring exactly as expected.

### The error budget: two tiers

A report runs about fifteen families of tests. If a family's claims can go either
way, pure noise makes it claim something with probability about α; if it only ever
claims one direction ("you score *worse* after a loss") about α/2, or α with a
one-sided test. The expected number of false claims per report is roughly the sum,
so the families are split into two tiers:

| α | Claims | Why |
|---|---|---|
| **0.05** | White vs Black; opening families; scoring worse straight after a loss; scoring worse late at night (23:00–03:00) | The claims players most need, and the hardest to detect: effects of 5–15 points per 100 games that only part of the games carry. The last two are fixed in advance with their direction, so they use a one-sided test. |
| **0.01** | Other times of day, game length, time controls, opponent strength, session length, how losses end, time trouble, losing on time, slow openings, clock handling, quick losses in one opening | Exploratory scans (many ways to get lucky), claims with a less direct reading (being mated more than you mate also depends on resignation habits), and clock or quick-loss habits, whose paired or mirror tests have power to spare when the habit is real. |

The budget adds up to about 4 × 0.05 + 11 × ≤ 0.01 ≈ 0.25, and measures at about
0.2 false claims per report (below). A report that says wrong things is worse than
one that says little, so the budget is kept well under the 0.3 target; the price is
paid in power at 400 games (below).

### Benchmarks chosen to avoid false claims

* **Self-relative comparisons** wherever possible. Clock usage, blunder rates and
  conversion are compared with *your opponents in the same games* (rating-matched
  peers on the same clock and position). "After a loss", times of day and session
  length compare those games with all your other games, so a rating that lags
  behind your strength doesn't create a finding.
* **Openings** are tested against the colour-adjusted expectation, and the family
  tests are *weighted* Benjamini–Hochberg adjusted with weights = games played: a
  leak in the opening you play in 40% of your games costs far more points than one
  in a sideline, so it gets most of the error budget (the family-wise error rate is
  unchanged).
* **Quick losses in one opening** compare how often your losses in it are over
  within 25 moves with how often your *wins* in it are (your opponents' quick
  collapses in the same openings). A sharp opening produces short games for both
  sides and is not a weakness.
* **Opponent strength** must be significant, in the same direction, against both
  the attenuated (`a = 0.75`) and the plain Elo (`a = 1.0`) expectation. With
  realistic rating noise, `a = 0.75` alone manufactures "you reliably beat
  lower-rated players" and plain Elo alone "you drop points against lower-rated
  players".
* **Time controls.** Each rating pool is judged against its own rating, so each
  pool's score-vs-expected is near zero by construction; picking the best and the
  worst of several noisy numbers finds a "gap" in pure noise. Each time control is
  instead compared with all your other time controls together, BH-adjusted over the
  time controls.
* **Late night** is one pre-specified window, 23:00–03:00 in your time zone (a
  4-hour block starting at midnight would cut it in two). If an overlapping 4-hour
  block is also flagged, the report keeps the one finding with the stronger evidence.
* **The rating trend** is always an observation: a rating is a random walk around
  your strength, and a 90-day swing of 100+ points happens by chance.

### Calibration: how often the report is wrong, and what it finds

`tests/null_world.py` generates a *null world*: games with realistic rating noise
(75 Elo on the true strength difference) and White's first-move edge, but where
openings, terminations, clocks, schedules and game lengths have nothing to do with
results. Every strength or weakness reported on it is false. The same generator can
*plant* realistic effects, each shifting only the games it concerns.

False claims in the top lists, 64 null worlds per size:

| | 400 games | 600 games |
|---|---|---|
| Before this calibration | 4.95 per report (max 10) | 4.44 (max 9) |
| Now | **0.20** (max 2) | **0.23** (max 2) |

Share of 32 planted worlds in which the report's top weaknesses name the planted
effect (right kind and category). *Plain*: the effect sits on top of an otherwise
accurate rating. *Priced in*: the rating has absorbed the effect, so every game is
lifted by `share × effect` and the affected games fall short of expectation by less
(the realistic case for a long-standing habit; the gap to the other games is the
same).

| Planted effect | 400 plain | 600 plain | 400 priced in | 600 priced in |
|---|---|---|---|---|
| Caro-Kann as Black −0.15 (40% of Black games) | 53% | 88% | 47% | 72% |
| Right after a loss −0.12 | 84% | 97% | 72% | 91% |
| Late night (23:00–03:00, ~29% of games) −0.10 | 72% | 88% | 59% | 84% |
| Time trouble in 35% more games, −0.10 in them | 100% | 100% | 94% | 97% |
| 25% of mate/resignation losses become losses on time | 66% | 88% | | |
| 40% of Italian Game losses (its main White opening) over within 22 moves, score unchanged | 53% | 78% | | |
| Black −0.08 | 31% | 53% | 50% | 62% |

Before this calibration the old thresholds "found" most of these too, but only
because they claimed almost anything: the same reports carried 3–6 other,
mostly false, strengths and weaknesses. Late-night play, split across two 4-hour
blocks, was found in 19% / 31% of worlds.

What this means for you:

* **With 600+ games** the report finds opening, tilt, late-night and clock effects
  of these sizes most of the time. **With 400 games** it finds the stronger ones
  (tilt, late night, time trouble) and misses about half of the others; it says
  less rather than guessing.
* **A colour imbalance of 8 points per 100 games needs about 1,000 games** (found in
  62–71% of worlds at 1,000): the gap between two halves of your games is simply
  that noisy.
* The null world is idealised: each game's true strength equals the current rating.
  Real ratings lag (after a losing run you are briefly underrated), which makes
  "after a loss" *harder* to find, never easier, and the error budget assumes
  families of tests are roughly independent. With several real effects at once,
  their knock-on effects (a weak Caro-Kann makes "Black" look weak too) are real
  associations, not false claims, but they can crowd the top lists.

Run `CALIBRATION_RUNS=64 CALIBRATION_GAMES=600 pytest tests/test_null_calibration.py`
to re-measure the false-claim rate after changing a threshold, and
`tests/test_power.py` checks that the planted effects above are still found.

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
