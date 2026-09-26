"""Clients for public chess data (C3): each one request at a time, disk-cached, silent when unavailable.

==================  =====================================================  ============================
module              what it gives                                          access
==================  =====================================================  ============================
http                the shared request layer: lock, 1 s per host, cache,   (used by the clients below)
                    429 / 401 / timeout handling, one note per failure
lichess_explorer    masters' and your rating groups' moves in a position   Lichess token; cached 30 days
cloud_eval          up to three engine lines from the Lichess cloud        no key; cached 30 days
tablebase           exact results with 7 pieces or fewer                   no key; cached for good
wikibooks           two credited sentences from Chess Opening Theory       no key; cached 90 days
openings_db         opening names by position (chess-openings, CC0)        bundled, no network
concept_notes       notes on positional concepts citing public-domain      bundled, no network
                    books (Capablanca, Lasker on Project Gutenberg)
==================  =====================================================  ============================

Every client returns None (and leaves one line in the coaching notes) when its source is unavailable, so the
report builds offline, without a token and without a network. Owner: sources.
"""

from .http import Fetched, Fetcher

__all__ = ["Fetched", "Fetcher"]
