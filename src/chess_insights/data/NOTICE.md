# Bundled data and where it comes from

## openings.tsv

Every named opening position from [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings)
(files `a.tsv` to `e.tsv`: ECO code, name and moves), merged into one file. Two columns were added with
python-chess: `uci` (the moves in UCI notation) and `epd` (the position after them).

The upstream data is dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): it may be copied, changed and redistributed
without asking. Retrieved on 26 September 2026 (3,815 positions). Rebuild it with
`python -m chess_insights.coach.sources.openings_db a.tsv b.tsv c.tsv d.tsv e.tsv -o openings.tsv`.

## concepts.json

Short notes on positional ideas (king safety, development, pawn structure ...), written for this project in
its own words. Each note points to the chapter or section of a public-domain book where the idea is explained:

- José Raúl Capablanca, *Chess Fundamentals* (1921), Project Gutenberg eBook
  [#33870](https://www.gutenberg.org/ebooks/33870)
- Edward Lasker, *Chess Strategy* (1915, translated by J. du Mont), Project Gutenberg eBook
  [#5614](https://www.gutenberg.org/ebooks/5614)

Both books are in the public domain in the USA. No text from them is bundled; only chapter and section titles
are cited.

## Not bundled

Answers from the Lichess opening explorer, cloud eval and tablebase APIs and short extracts from
[Wikibooks Chess Opening Theory](https://en.wikibooks.org/wiki/Chess_Opening_Theory) are downloaded when a
report is built and kept only in the local cache (`<cache>/sources/`). Wikibooks text is
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): the report quotes at most two sentences and
credits "Wikibooks, CC BY-SA 4.0" with a link to the page.
