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

## Not bundled: downloaded when needed

- **The Lichess puzzle database** ([database.lichess.org](https://database.lichess.org),
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)): `chess-insights puzzles-db` downloads it and
  keeps a filtered subset in the cache folder; the drill packs written next to a report are taken from it. The
  tests use a 1,573-puzzle sample of it (`tests/fixtures/lichess_puzzles_sample.csv`).
- **Answers from the Lichess opening explorer, cloud eval and tablebase APIs** are downloaded when a report is
  built, kept in the local cache (`<cache>/sources/`) and shown in the report with their source.
- **Short extracts from [Wikibooks Chess Opening Theory](https://en.wikibooks.org/wiki/Chess_Opening_Theory)**
  are downloaded the same way. Wikibooks text is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/):
  the report quotes at most two sentences and credits "Wikibooks, CC BY-SA 4.0" with a link to the page. The
  extract is part of the report files (HTML, Markdown and JSON), so keep the credit when you share a report.

## Programs the report uses but does not include

- **Stockfish** ([stockfishchess.org](https://stockfishchess.org), GPL-3.0) is a separate program that you
  install (the GitHub workflow installs Ubuntu's package) and that chess-insights runs over UCI. Stockfish 16 is
  the version the explanations' evaluation terms need.
- **Maia-2** (optional, `pip install maia2`, MIT) downloads its model weights on first use.
- **python-chess** (a dependency, GPL-3.0-or-later) draws the chess pieces that every HTML report embeds.
