"""scripts/record_fixtures.py and .github/workflows/fixtures.yml: the Lichess token reaches the opening explorer only.

The explorer answers 401 without a token, so the recorded explorer fixtures were all error pages. The recorder now
sends LICHESS_TOKEN (the repository secret, through the environment) as a Bearer header to explorer.lichess.org
only, never after a redirect, and never writes a request header into a recorded file. No network here.
"""

import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_WORKFLOW = (ROOT / ".github" / "workflows" / "fixtures.yml").read_text(encoding="utf-8")
TOKEN = "lip_fixture_secret_123"


@pytest.fixture(scope="module")
def recorder():
    spec = importlib.util.spec_from_file_location("record_fixtures", ROOT / "scripts" / "record_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(recorder, monkeypatch):
    """Every request urlopen was given (no network); each answer is a small JSON body."""
    requests = []

    def urlopen(req, timeout=None):
        requests.append(req)
        return FakeResponse(b'{"white": 10, "draws": 5, "black": 7, "moves": []}')

    monkeypatch.setattr(recorder.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(recorder.time, "sleep", lambda s: None)
    return requests


@pytest.mark.parametrize("url, gets_token", [
    ("https://explorer.lichess.org/masters?fen=x&topGames=2", True),
    ("https://explorer.lichess.org/lichess?fen=x&speeds=blitz,rapid", True),
    ("https://lichess.org/api/cloud-eval?fen=x&multiPv=3", False),
    ("https://tablebase.lichess.org/standard?fen=x", False),
    ("https://en.wikibooks.org/w/api.php?titles=explorer.lichess.org", False),
    ("https://explorer.lichess.org.example.com/masters", False),
])
def test_the_token_goes_to_the_opening_explorer_only(recorder, monkeypatch, url, gets_token):
    monkeypatch.setenv("LICHESS_TOKEN", TOKEN)
    assert recorder.auth_header(url) == ({"Authorization": f"Bearer {TOKEN}"} if gets_token else {})
    for unset in ("", "   "):
        monkeypatch.setenv("LICHESS_TOKEN", unset)
        assert recorder.auth_header(url) == {}
    monkeypatch.delenv("LICHESS_TOKEN")
    assert recorder.auth_header(url) == {}


def test_the_token_is_not_sent_on_after_a_redirect(recorder, monkeypatch, sent):
    monkeypatch.setenv("LICHESS_TOKEN", TOKEN)
    recorder.get("https://explorer.lichess.org/masters?fen=x")
    recorder.get("https://lichess.org/api/cloud-eval?fen=x")
    explorer, cloud = sent
    assert explorer.unredirected_hdrs == {"Authorization": f"Bearer {TOKEN}"}  # urllib drops these on a redirect
    assert "Authorization" not in explorer.headers and not cloud.has_header("Authorization")


def test_recorded_files_never_hold_a_request_header(recorder, monkeypatch, sent, tmp_path):
    monkeypatch.setenv("LICHESS_TOKEN", TOKEN)
    assert recorder.record(tmp_path, "explorer_masters_x", "https://explorer.lichess.org/masters?fen=x") == {
        "status": 200}
    text = (tmp_path / "explorer_masters_x.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    assert set(doc) == {"url", "status", "content_type", "recorded", "body"}
    assert doc["status"] == 200 and doc["body"]["white"] == 10
    assert TOKEN not in text and "Bearer" not in text and "Authorization" not in text


def test_the_workflow_passes_the_secret_through_the_environment():
    record = FIXTURES_WORKFLOW[FIXTURES_WORKFLOW.index("- name: Record"):FIXTURES_WORKFLOW.index("- name: Save")]
    assert "LICHESS_TOKEN: ${{ secrets.LICHESS_TOKEN }}" in record
    assert "run: python scripts/record_fixtures.py fixtures-out" in record
    assert FIXTURES_WORKFLOW.count("secrets.LICHESS_TOKEN") == 1
    for run in re.findall(r"run: (.*)", FIXTURES_WORKFLOW):
        assert "${{" not in run and "LICHESS_TOKEN" not in run
