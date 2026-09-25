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


def test_gives_up_after_max_retries_without_a_pointless_last_sleep():
    client, sleeps = make_client({f"{BASE}": [FakeResponse(429)]}, max_retries=2, backoff_base=0.1)
    with pytest.raises(api.RateLimited):
        client.profile("tester")
    assert len(client.session.calls) == 3
    assert len(sleeps) == 2  # between attempts only, not after the last one


def test_connection_errors_are_retried():
    client, _ = make_client({f"{BASE}": [requests.ConnectionError("reset"), FakeResponse(200, {"ok": 1})]})
    assert client.profile("tester") == {"ok": 1}


def test_unknown_player_raises_player_not_found():
    client, _ = make_client({})
    with pytest.raises(api.PlayerNotFound):
        client.archives("nobody")


def test_403_explains_user_agent_and_network():
    client, _ = make_client({f"{BASE}": [FakeResponse(403)]})
    with pytest.raises(api.ChessComError, match="User-Agent"):
        client.profile("tester")


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
    assert empty.games == [] and not empty.not_modified and empty.missing


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


def test_sync_does_not_cache_a_404_month(tmp_path):
    routes = _routes_for_sync({(2024, 1): [{"uuid": "a"}]})
    routes[f"{BASE}/games/2024/01"] = [FakeResponse(404, {"message": "An internal error has occurred"})]
    client, _ = make_client(routes)
    s = fetch.sync("tester", client, tmp_path, snapshots=False)
    assert s.months_missing == 1 and fetch.GameStore(tmp_path, "tester").months() == []


def test_sync_falls_back_to_join_date_when_archive_list_404s(tmp_path):
    joined = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp())
    routes = {
        f"{BASE}": [FakeResponse(200, {"username": "tester", "joined": joined})],
        f"{BASE}/games/2024/01": [FakeResponse(200, {"games": [{"uuid": "a"}]})],
        f"{BASE}/games/2024/02": [FakeResponse(200, {"games": []})],
        f"{BASE}/games/2024/03": [FakeResponse(200, {"games": [{"uuid": "b"}]})],
    }
    client, _ = make_client(routes)
    now = lambda: datetime(2024, 3, 5, tzinfo=timezone.utc)  # noqa: E731
    s = fetch.sync("tester", client, tmp_path, snapshots=False, now=now)
    assert (s.months_listed, s.games_total) == (3, 2)


def test_refresh_all_revalidates_complete_months(tmp_path):
    payloads = {(2024, 1): [{"uuid": "a"}]}
    now = lambda: datetime(2024, 6, 1, tzinfo=timezone.utc)  # noqa: E731
    client, _ = make_client(_routes_for_sync(payloads))
    fetch.sync("tester", client, tmp_path, now=now, snapshots=False)
    routes = _routes_for_sync(payloads)
    routes[f"{BASE}/games/2024/01"] = [FakeResponse(200, {"games": [{"uuid": "a", "accuracies": {"white": 90}}]})]
    client2, _ = make_client(routes)
    s = fetch.sync("tester", client2, tmp_path, now=now, snapshots=False, refresh_all=True)
    assert s.months_downloaded == 1
    assert fetch.load_games("tester", tmp_path)[0]["accuracies"] == {"white": 90}


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


# --------------------------------------------------------------------------- HTTP details
def test_retry_after_as_http_date():
    from email.utils import format_datetime
    from datetime import timedelta

    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    client, sleeps = make_client({f"{BASE}": [FakeResponse(429, headers={"Retry-After": when}), FakeResponse(200, {})]})
    client.profile("tester")
    assert 20 <= sleeps[0] <= 31


@pytest.mark.parametrize("value", ["inf", "1e12", "86400", "-5", "nan", "soon"])
def test_retry_after_is_sane_and_capped(value):
    client, sleeps = make_client({f"{BASE}": [FakeResponse(429, headers={"Retry-After": value}), FakeResponse(200, {})]})
    client.profile("tester")
    assert 0 <= sleeps[0] <= client.max_retry_after


def test_invalid_json_body_is_retried():
    client, _ = make_client({f"{BASE}": [FakeResponse(200, None), FakeResponse(200, {"username": "tester"})]})
    assert client.profile("tester") == {"username": "tester"}


def test_tls_errors_are_not_retried_and_mention_proxies():
    client, sleeps = make_client({f"{BASE}": [requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED")]})
    with pytest.raises(api.ChessComError, match="(?i)certificate.*proxy|proxy.*certificate"):
        client.profile("tester")
    assert len(client.session.calls) == 1 and sleeps == []


def test_offline_network_error_message_is_actionable():
    client, _ = make_client({f"{BASE}": [requests.ConnectionError("Name or service not known")]}, max_retries=1)
    with pytest.raises(api.ChessComError, match="api.chess.com"):
        client.profile("tester")


@pytest.mark.parametrize(
    "text", ["Tester", " tester ", "@Tester", "https://www.chess.com/member/Tester", "chess.com/member/tester/",
             "https://www.chess.com/stats/live/blitz/Tester"],
)
def test_username_accepts_handles_and_profile_urls(text):
    assert api.normalize_username(text) == "tester"


def test_default_session_keeps_gzip_and_sets_user_agent():
    client = api.ChessComClient()
    assert "gzip" in client.session.headers.get("Accept-Encoding", "")
    assert client.session.headers["User-Agent"].startswith("chess-insights/")


def test_real_http_stack_redirect_gzip_and_weak_etag():
    """Against a local server: requests follows chess.com's 301 to the lowercase URL, un-gzips the body,
    and a weak ETag round-trips to a 304."""
    import gzip
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output clean
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("User-Agent"), self.headers.get("If-None-Match")))
            if self.path != self.path.lower():
                self.send_response(301)
                self.send_header("Location", self.path.lower())
                self.end_headers()
                return
            if self.headers.get("If-None-Match") == 'W/"v1"':
                self.send_response(304)
                self.send_header("ETag", 'W/"v1"')
                self.end_headers()
                return
            body = gzip.compress(b'{"games": [{"uuid": "a"}]}')
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("ETag", 'W/"v1"')
            self.send_header("Last-Modified", "Friday, 25-Oct-2024 02:32:13 GMT+0000")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session = requests.Session()
        session.trust_env = False  # never route 127.0.0.1 through a proxy
        base = f"http://127.0.0.1:{server.server_port}/pub"
        client = api.ChessComClient(session=session, base_url=base, min_interval=0, contact="me@example.com")
        data, resp = client.request("player/Tester/games/2024/03")
        assert data == {"games": [{"uuid": "a"}]} and resp.headers["etag"] == 'W/"v1"'
        assert [p for p, _, _ in seen] == ["/pub/player/Tester/games/2024/03", "/pub/player/tester/games/2024/03"]
        assert all("me@example.com" in ua for _, ua, _ in seen)  # the UA survives the redirect
        month = client.month("tester", 2024, 3, etag=resp.headers["etag"])
        assert month.not_modified and seen[-1][2] == 'W/"v1"'
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- sync edge cases
def test_304_without_etag_keeps_the_cached_etag(tmp_path):
    payloads = {(2024, 2): [{"uuid": "c"}]}
    now = lambda: datetime(2024, 2, 20, tzinfo=timezone.utc)  # noqa: E731
    client, _ = make_client(_routes_for_sync(payloads))
    fetch.sync("tester", client, tmp_path, now=now, snapshots=False)
    routes = _routes_for_sync(payloads)
    routes[f"{BASE}/games/2024/02"] = [FakeResponse(304)]  # no ETag header on the 304
    client2, _ = make_client(routes)
    s = fetch.sync("tester", client2, tmp_path, now=now, snapshots=False)
    assert s.months_not_modified == 1
    store = fetch.GameStore(tmp_path, "tester")
    assert store.load_month((2024, 2))["etag"] == '"20242"'
    assert len(store.load_month((2024, 2))["games"]) == 1


def test_if_modified_since_is_used_when_there_is_no_etag(tmp_path):
    lm = "Friday, 25-Oct-2024 02:32:13 GMT+0000"
    routes = _routes_for_sync({(2024, 10): [{"uuid": "a"}]})
    routes[f"{BASE}/games/2024/10"] = [FakeResponse(200, {"games": [{"uuid": "a"}]}, headers={"Last-Modified": lm})]
    now = lambda: datetime(2024, 10, 25, tzinfo=timezone.utc)  # noqa: E731
    client, _ = make_client(routes)
    fetch.sync("tester", client, tmp_path, now=now, snapshots=False)
    client2, _ = make_client(routes)
    fetch.sync("tester", client2, tmp_path, now=now, snapshots=False)
    assert client2.session.calls[-1] == (f"{BASE}/games/2024/10", {"If-Modified-Since": lm})


def test_a_410_month_is_not_requested_again(tmp_path):
    routes = _routes_for_sync({(2024, 1): []})
    routes[f"{BASE}/games/2024/01"] = [FakeResponse(410)]
    client, _ = make_client(routes)
    s = fetch.sync("tester", client, tmp_path, snapshots=False)
    assert s.months_missing == 0
    client2, _ = make_client(routes)
    fetch.sync("tester", client2, tmp_path, snapshots=False)
    assert f"{BASE}/games/2024/01" not in [u for u, _ in client2.session.calls]


def test_closed_account_is_reported(tmp_path):
    routes = _routes_for_sync({(2024, 1): [{"uuid": "a"}]})
    routes[f"{BASE}"] = [FakeResponse(200, {"username": "tester", "status": "closed:fair_play_violations"})]
    client, _ = make_client(routes)
    said = []
    s = fetch.sync("tester", client, tmp_path, snapshots=False, progress=said.append)
    assert s.games_total == 1 and s.account_status == "closed:fair_play_violations"
    assert any("closed" in m for m in said)


def test_closed_account_without_archives_does_not_scan_every_month(tmp_path):
    joined = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp())
    routes = {f"{BASE}": [FakeResponse(200, {"username": "tester", "status": "closed", "joined": joined})]}
    client, _ = make_client(routes)
    with pytest.raises(api.PlayerNotFound, match="closed"):
        fetch.sync("tester", client, tmp_path, snapshots=False)
    assert len(client.session.calls) <= 3


def test_load_games_dedupes_across_url_forms(tmp_path):
    store = fetch.GameStore(tmp_path, "tester")
    t = datetime(2024, 5, 1, tzinfo=timezone.utc)
    store.save_month((2020, 3), [{"url": "https://www.chess.com/live/game/42"}], etag=None, fetched_at=t, complete=True)
    store.save_month((2020, 4), [{"uuid": "u", "url": "https://www.chess.com/game/live/42"},
                                 {"url": "https://www.chess.com/game/daily/42"}], etag=None, fetched_at=t, complete=True)
    assert len(fetch.load_games("tester", tmp_path)) == 2


def test_corrupt_snapshot_is_treated_as_missing(tmp_path):
    store = fetch.GameStore(tmp_path, "tester")
    store.root.mkdir(parents=True)
    (store.root / "profile.json").write_text("{oops", encoding="utf-8")
    assert store.load_snapshot("profile") is None


def test_atomic_write_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    real_replace = fetch.os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_replace(src, dst)

    monkeypatch.setattr(fetch.os, "replace", flaky_replace)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    fetch._atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert len(calls) == 2 and fetch.json.loads((tmp_path / "x.json").read_text(encoding="utf-8")) == {"a": 1}
    assert not list(tmp_path.glob(".tmp-*"))


# --------------------------------------------------------------------------- reading user files
@pytest.mark.parametrize(
    "raw",
    [
        "Réti Opening – “quoted” …".encode("utf-8"),
        "\ufeffRéti Opening – “quoted” …".encode("utf-8"),  # Notepad
        "Réti Opening – “quoted” …".encode("utf-16"),  # Windows PowerShell 5 `Out-File` / `>`
        "Réti Opening – “quoted” …".encode("cp1252"),  # legacy Windows tools
    ],
    ids=["utf-8", "utf-8-bom", "utf-16", "cp1252"],
)
def test_read_text_handles_windows_encodings(tmp_path, raw):
    path = tmp_path / "f.txt"
    path.write_bytes(raw)
    assert fetch.read_text(path) == "Réti Opening – “quoted” …"


def test_load_json_files_accepts_every_shape(tmp_path):
    game = {"uuid": "g1", "white": {"username": "a"}, "black": {"username": "b"}}
    shapes = {
        "archive.json": {"games": [game]},
        "bom.json": "\ufeff" + fetch.json.dumps({"games": [dict(game, uuid="g2")]}),
        "single.json": dict(game, uuid="g3"),
        "list.json": [dict(game, uuid="g4")],
        "list_of_archives.json": [{"games": [dict(game, uuid="g5")]}, {"games": [dict(game, uuid="g6")]}],
        "by_url.json": {"https://api.chess.com/pub/player/a/games/2024/01": {"games": [dict(game, uuid="g7")]}},
    }
    paths = []
    for name, data in shapes.items():
        p = tmp_path / name
        p.write_text(data if isinstance(data, str) else fetch.json.dumps(data), encoding="utf-8")
        paths.append(p)
    assert [g["uuid"] for g in fetch.load_json_files(paths)] == [f"g{i}" for i in range(1, 8)]


@pytest.mark.parametrize("name", ["con", "AUX", "nul", "com1", "lpt9", "prn"])
def test_windows_device_names_are_not_used_as_folder_names(tmp_path, name):
    """chess.com usernames like 'con' or 'aux' are valid, but Windows can't create such folders."""
    store = fetch.GameStore(tmp_path, name)
    assert store.username == name.lower()
    assert store.root.name == f"_{name.lower()}"
    assert fetch.GameStore(tmp_path, "conrad").root.name == "conrad"


# --------------------------------------------------------------------------- verifier regressions
def _local_server(handler_body):
    """(base_url, seen user agents, shutdown) of a local HTTP server answering every GET with ``handler_body``."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.headers.get("User-Agent"))
            body = handler_body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def shutdown():
        server.shutdown()
        server.server_close()

    return f"http://127.0.0.1:{server.server_port}/pub", seen, shutdown


@pytest.mark.parametrize(
    "contact, kept",
    [
        ("Łukasz Nowak <lukasz@example.pl>", "<lukasz@example.pl>"),
        ("张伟 zhang@example.cn", "zhang@example.cn"),
        ("José <jose@example.es>", "Jose <jose@example.es>"),
        ("me@example.com\n", "me@example.com"),
        ("me@example.com\r\nX-Injected: 1", "me@example.com X-Injected: 1"),
        ("\tme@example.com\u200b ", "me@example.com"),
    ],
)
def test_any_contact_is_sent_over_a_real_http_stack(contact, kept, monkeypatch):
    """--contact / CHESS_INSIGHTS_CONTACT with non-Latin-1 text or a newline used to crash
    (UnicodeEncodeError) or fail every request as a 'network error' after ~31 s of retries."""
    base, seen, shutdown = _local_server(b'{"username": "tester"}')
    try:
        for how in ("argument", "environment"):
            session = requests.Session()
            session.trust_env = False
            sleeps = []
            if how == "environment":
                monkeypatch.setenv("CHESS_INSIGHTS_CONTACT", contact)
            client = api.ChessComClient(
                contact=contact if how == "argument" else None, session=session, base_url=base,
                min_interval=0, sleep=sleeps.append,
            )
            assert client.profile("tester") == {"username": "tester"}
            assert sleeps == []  # first attempt succeeded
        assert len(seen) == 2
        for ua in seen:
            assert ua.isascii() and ua.isprintable() and kept in ua and "  " not in ua
    finally:
        shutdown()


def test_header_safe_keeps_non_ascii_recognisable():
    assert api.header_safe("Łukasz") == "%C5%81ukasz"
    assert api.header_safe("Réti") == "Reti"
    assert api.header_safe("a\x00b\x7fc") == "a b c"


def test_no_connection_gives_up_after_one_retry_and_says_so():
    """No internet / blocked host / packets dropped: 2 attempts, not 6 (~3.5 minutes of silence before)."""
    notes = []
    client, sleeps = make_client({f"{BASE}": [requests.exceptions.ConnectTimeout("connect timed out")]}, notify=notes.append)
    with pytest.raises(api.ChessComError, match="after 2 attempts.*internet connection"):
        client.profile("tester")
    assert len(client.session.calls) == 2 and sleeps == [1.0]
    assert len(notes) == 1 and "retrying in 1s" in notes[0] and "ConnectTimeout" in notes[0]


def test_refused_connection_and_dns_failure_are_connect_failures():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens on this port now
    session = requests.Session()
    session.trust_env = False
    sleeps = []
    client = api.ChessComClient(session=session, base_url=f"http://127.0.0.1:{port}/pub", min_interval=0, sleep=sleeps.append)
    with pytest.raises(api.ChessComError, match="after 2 attempts"):
        client.profile("tester")
    assert sleeps == [1.0]


def test_default_timeouts_are_split_into_connect_and_read():
    timeouts = []

    class Session(requests.Session):
        def get(self, url, headers=None, timeout=None):
            timeouts.append(timeout)
            return FakeResponse(200, {})

    api.ChessComClient(session=Session(), min_interval=0).profile("tester")
    connect, read = timeouts[0]
    assert connect <= 10 and read >= 30


def test_a_connection_broken_mid_answer_still_gets_every_retry():
    client, sleeps = make_client({f"{BASE}": [requests.exceptions.ChunkedEncodingError("cut off")]}, max_retries=3)
    with pytest.raises(api.ChessComError):
        client.profile("tester")
    assert len(client.session.calls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_persistent_429_stays_within_the_wait_budget_and_announces_each_wait():
    """A 429 with Retry-After: 3600 used to mean 5 silent waits of 120 s (10 minutes)."""
    notes = []
    client, sleeps = make_client({f"{BASE}": [FakeResponse(429, headers={"Retry-After": "3600"})]}, notify=notes.append)
    with pytest.raises(api.RateLimited):
        client.profile("tester")
    assert sum(sleeps) <= client.max_total_wait <= 180
    assert all(s <= client.max_retry_after <= 60 for s in sleeps)
    assert len(notes) == len(sleeps) and all("HTTP 429" in n and "retrying in 60s" in n for n in notes)


def test_a_bad_request_is_not_retried():
    client, sleeps = make_client({f"{BASE}": [requests.exceptions.InvalidHeader("bad header value")]})
    with pytest.raises(api.ChessComError, match="InvalidHeader"):
        client.profile("tester")
    assert len(client.session.calls) == 1 and sleeps == []


def test_load_json_files_unwraps_wrappers_inside_a_games_list(tmp_path, fixtures_dir):
    """{"games": [{"case": ..., "game": {...}}]} (the shape of the edge-case fixture) was read as 0 games."""
    game = {"uuid": "g1", "white": {"username": "a"}, "black": {"username": "b"}}
    path = tmp_path / "wrapped.json"
    path.write_text(fetch.json.dumps({"games": [{"case": "x", "game": game}, dict(game, uuid="g2")]}), encoding="utf-8")
    assert [g["uuid"] for g in fetch.load_json_files([path])] == ["g1", "g2"]
    edge = fetch.load_json_files([fixtures_dir / "real_chesscom_edge_cases.json"])
    raw = fetch.json.loads((fixtures_dir / "real_chesscom_edge_cases.json").read_text(encoding="utf-8"))
    assert len(edge) == len(raw["games"]) > 0 and all("white" in g for g in edge)


@pytest.mark.parametrize(
    "raw, expected",
    [('D:\\chess"', "D:\\chess"), ('"reports/me"', "reports/me"), ("plain/dir", "plain/dir")],
)
def test_expand_path_drops_the_stray_quote_cmd_exe_leaves(raw, expected):
    """cmd.exe turns --cache-dir "D:\\chess\\" into D:\\chess" (WinError 123 on every write)."""
    assert str(fetch.expand_path(raw)) == str(fetch.Path(expected))
