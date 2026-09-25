from datetime import datetime, timezone

import pytest
import requests

from chess_insights import api, fetch


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Maps URL -> list of responses (consumed in order; last one repeats)."""

    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        queue = self.routes.get(url)
        if queue is None:
            return FakeResponse(404)
        resp = queue[0] if len(queue) == 1 else queue.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


BASE = "https://api.chess.com/pub/player/tester"


def make_client(routes, **kw):
    sleeps = []
    client = api.ChessComClient(session=FakeSession(routes), sleep=sleeps.append, min_interval=0, **kw)
    return client, sleeps


def test_normalize_username_lowercases_and_rejects_paths():
    assert api.normalize_username(" TesTer ") == "tester"
    for bad in ("../etc", "a/b", "", "x" * 60, "name?x=1"):
        with pytest.raises(ValueError):
            api.normalize_username(bad)


def test_parse_archive_url():
    assert api.parse_archive_url(f"{BASE}/games/2024/03") == (2024, 3)
    with pytest.raises(ValueError):
        api.parse_archive_url(f"{BASE}/games/archives")


def test_user_agent_includes_contact():
    client, _ = make_client({}, contact="me@example.com")
    ua = client.session.headers["User-Agent"]
    assert ua.startswith("chess-insights/") and "me@example.com" in ua


def test_archives_sorted_oldest_first():
    client, _ = make_client(
        {f"{BASE}/games/archives": [FakeResponse(200, {"archives": [f"{BASE}/games/2024/02", f"{BASE}/games/2023/11"]})]}
    )
    assert client.archives("Tester") == [f"{BASE}/games/2023/11", f"{BASE}/games/2024/02"]


def test_retries_on_429_honouring_retry_after_then_succeeds():
    client, sleeps = make_client(
        {
            f"{BASE}": [
                FakeResponse(429, headers={"Retry-After": "7"}),
                FakeResponse(503),
                FakeResponse(200, {"username": "tester"}),
            ]
        },
        backoff_base=0.5,
    )
    assert client.profile("tester") == {"username": "tester"}
    assert sleeps == [7.0, 1.0]  # Retry-After, then backoff 0.5 * 2**1


def test_gives_up_after_max_retries():
    client, sleeps = make_client({f"{BASE}": [FakeResponse(429)]}, max_retries=2, backoff_base=0.1)
    with pytest.raises(api.RateLimited):
        client.profile("tester")
    assert len(sleeps) == 3


def test_connection_errors_are_retried():
    client, _ = make_client({f"{BASE}": [requests.ConnectionError("reset"), FakeResponse(200, {"ok": 1})]})
    assert client.profile("tester") == {"ok": 1}


def test_unknown_player_raises_player_not_found():
    client, _ = make_client({})
    with pytest.raises(api.PlayerNotFound):
        client.archives("nobody")


def test_other_http_errors_raise():
    client, _ = make_client({f"{BASE}": [FakeResponse(403, {"message": "forbidden"})]})
    with pytest.raises(api.ChessComError):
        client.profile("tester")


def test_month_conditional_request_and_404_as_empty():
    client, _ = make_client(
        {
            f"{BASE}/games/2024/03": [FakeResponse(304, headers={"ETag": '"abc"'})],
        }
    )
    res = client.month("tester", 2024, 3, etag='"abc"')
    assert res.not_modified and res.games == []
    assert client.session.calls[-1][1] == {"If-None-Match": '"abc"'}
    empty = client.month("tester", 2024, 4)
    assert empty.games == [] and not empty.not_modified


def _routes_for_sync(month_payloads):
    routes = {
        f"{BASE}": [FakeResponse(200, {"username": "tester"})],
        f"{BASE}/stats": [FakeResponse(200, {"chess_blitz": {}})],
        f"{BASE}/games/archives": [
            FakeResponse(200, {"archives": [f"{BASE}/games/{y}/{m:02d}" for (y, m) in month_payloads]})
        ],
    }
    for (y, m), games in month_payloads.items():
        routes[f"{BASE}/games/{y}/{m:02d}"] = [FakeResponse(200, {"games": games}, headers={"ETag": f'"{y}{m}"'})]
    return routes


def test_sync_downloads_then_skips_complete_months(tmp_path):
    payloads = {
        (2024, 1): [{"uuid": "a"}, {"uuid": "b"}],
        (2024, 2): [{"uuid": "c"}],
    }
    now = lambda: datetime(2024, 2, 20, tzinfo=timezone.utc)  # noqa: E731  (Feb is the current month)
    client, _ = make_client(_routes_for_sync(payloads))
    s1 = fetch.sync("Tester", client, tmp_path, now=now)
    assert (s1.months_listed, s1.months_downloaded, s1.games_total) == (2, 2, 3)
    store = fetch.GameStore(tmp_path, "tester")
    assert store.load_month((2024, 1))["complete"] is True
    assert store.load_month((2024, 2))["complete"] is False
    assert store.load_snapshot("profile") == {"username": "tester"}

    # second run: January is complete -> not requested; February is refreshed with its ETag
    client2, _ = make_client(_routes_for_sync(payloads))
    s2 = fetch.sync("tester", client2, tmp_path, now=now, snapshots=False)
    urls = [u for u, _ in client2.session.calls]
    assert f"{BASE}/games/2024/01" not in urls
    assert client2.session.calls[-1] == (f"{BASE}/games/2024/02", {"If-None-Match": '"20242"'})
    assert (s2.months_cached, s2.months_downloaded) == (1, 1)


def test_sync_respects_since_until(tmp_path):
    payloads = {(2023, 12): [{"uuid": "x"}], (2024, 1): [{"uuid": "a"}], (2024, 2): [{"uuid": "c"}]}
    client, _ = make_client(_routes_for_sync(payloads))
    s = fetch.sync("tester", client, tmp_path, since=(2024, 1), until=(2024, 1), snapshots=False)
    assert s.months_listed == 1
    assert fetch.GameStore(tmp_path, "tester").months() == [(2024, 1)]


def test_load_games_dedupes_and_filters(tmp_path):
    store = fetch.GameStore(tmp_path, "tester")
    t = datetime(2024, 5, 1, tzinfo=timezone.utc)
    store.save_month((2024, 3), [{"uuid": "a"}, {"uuid": "b"}], etag=None, fetched_at=t, complete=True)
    store.save_month((2024, 4), [{"uuid": "b"}, {"url": "u1"}], etag=None, fetched_at=t, complete=True)
    assert [g.get("uuid") or g.get("url") for g in fetch.load_games("tester", tmp_path)] == ["a", "b", "u1"]
    assert len(fetch.load_games("tester", tmp_path, since=(2024, 4))) == 2


def test_corrupt_month_file_is_treated_as_missing(tmp_path):
    store = fetch.GameStore(tmp_path, "tester")
    store.games_dir.mkdir(parents=True)
    store.month_path((2024, 1)).write_text("{not json")
    assert store.load_month((2024, 1)) is None


def test_parse_month():
    assert fetch.parse_month("2024-03") == (2024, 3)
    assert fetch.parse_month("2024/11") == (2024, 11)
    with pytest.raises(ValueError):
        fetch.parse_month("2024-13")
