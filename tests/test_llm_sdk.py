"""The LLM coach against the real Anthropic SDK, over a mocked HTTP transport (no network, no API key).

The fakes in test_llm.py only check what llm.py sends; these check that the SDK itself accepts it (no TypeError
on ``output_config`` or the streaming helper), that the model id reaches the request, and that the SDK's parsed
reply, streamed or not, becomes accepted and rejected texts. Skipped when the ``llm`` extra is not installed.
"""

from __future__ import annotations

import json

import pytest

anthropic = pytest.importorskip("anthropic")
try:  # anthropic 1.x is built on httpx2 (and rejects an httpx client); 0.x on httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover - an 0.x SDK
    httpx = pytest.importorskip("httpx")

from chess_insights.coach import CoachConfig, llm  # noqa: E402
from test_llm import GOOD_SICILIAN, llm_reply  # noqa: E402
from test_packet import QGA_4E4, SICILIAN_2NC6, SICILIAN_5E5, coaching_report, epd  # noqa: E402

MODEL = "claude-test-model"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _message(model: str, content: list, stop_reason) -> dict:
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1200, "output_tokens": 0}}


def _sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class FakeAPI:
    """An HTTP handler that answers POST /v1/messages like the Messages API: a server-sent event stream when the
    request asks for one (a thinking block, then the reply text in pieces), else one JSON message."""

    def __init__(self, reply: dict | str, stop_reason: str = "end_turn", status: int = 200):
        self.text = reply if isinstance(reply, str) else json.dumps(reply)
        self.stop_reason = stop_reason
        self.status = status
        self.requests: list[tuple[httpx.Request, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        if self.status != 200:
            return httpx.Response(self.status, json={"type": "error", "error": {
                "type": "authentication_error", "message": "invalid x-api-key"}})
        if not body.get("stream"):
            content = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": self.text}]
            return httpx.Response(200, json=_message(body["model"], content, self.stop_reason))
        half = len(self.text) // 2
        events = [
            {"type": "message_start", "message": _message(body["model"], [], None)},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "ping"},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": self.text[:half]}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": self.text[half:]}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
             "usage": {"output_tokens": 900}},
            {"type": "message_stop"},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(events))


def sdk_client(api: FakeAPI, **kwargs):
    return anthropic.Anthropic(api_key=kwargs.pop("api_key", "sk-test"), max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(api)), **kwargs)


def check_sent(api: FakeAPI, *, stream: bool) -> dict:
    (request, body), = api.requests
    assert request.method == "POST" and request.url.path == "/v1/messages"
    assert body["model"] == MODEL and body["max_tokens"] == llm.MAX_TOKENS
    assert body["output_config"] == {"effort": llm.EFFORT,
                                     "format": {"type": "json_schema", "schema": llm.OUTPUT_SCHEMA}}
    assert bool(body.get("stream")) is stream
    assert body["system"] == llm.SYSTEM_PROMPT and body["messages"][0]["role"] == "user"
    assert "thinking" not in body and "temperature" not in body  # adaptive by default; sampling is not sent
    return body


def check_verified(report) -> None:
    by_epd = {e.epd: e for e in report.coaching.explanations}
    assert by_epd[epd(SICILIAN_5E5)].text == GOOD_SICILIAN and by_epd[epd(SICILIAN_5E5)].text_source == "llm"
    assert all(by_epd[epd(fen)].text_source == "template" for fen in (QGA_4E4, SICILIAN_2NC6))
    info = report.coaching.llm
    assert (info["model"], info["accepted"], info["rejected"], info["ignored"]) == (MODEL, 1, 2, 1)
    assert {r["epd"] for r in info["rejections"]} == {epd(QGA_4E4), epd(SICILIAN_2NC6)}


def test_the_sdk_accepts_the_streamed_request_and_the_reply_is_verified():
    report = coaching_report()
    api = FakeAPI(llm_reply(report))
    llm.annotate(report, CoachConfig(llm=True, llm_model=MODEL), client=sdk_client(api))
    check_sent(api, stream=True)  # messages.stream: the streaming helper takes output_config
    check_verified(report)


def test_the_sdk_accepts_the_plain_request_too():
    """A client without the streaming helper gets messages.create: the SDK takes output_config there as well.
    (The SDK refuses a plain request this long under its default timeout; llm.make_client sets one.)"""
    report = coaching_report()
    api = FakeAPI(llm_reply(report))
    real = sdk_client(api, timeout=llm.TIMEOUT_S)

    class CreateOnly:
        messages = type("Messages", (), {"create": staticmethod(real.messages.create)})()

    llm.annotate(report, CoachConfig(llm=True, llm_model=MODEL), client=CreateOnly())
    check_sent(api, stream=False)
    check_verified(report)


def test_the_client_llm_builds_is_a_real_sdk_client(monkeypatch):
    """No client passed: llm.make_client's settings (key, timeout, retries) go through the real constructor."""
    report = coaching_report()
    api = FakeAPI(llm_reply(report))
    real, built = anthropic.Anthropic, []

    def with_transport(**kwargs):
        client = real(http_client=httpx.Client(transport=httpx.MockTransport(api)), **kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(anthropic, "Anthropic", with_transport)
    llm.annotate(report, CoachConfig(llm=True, llm_model=MODEL, anthropic_api_key="sk-from-settings"))
    (client,) = built
    assert client.timeout == llm.TIMEOUT_S and client.max_retries == llm.MAX_RETRIES
    (request, _), = api.requests
    assert request.headers["x-api-key"] == "sk-from-settings"
    check_sent(api, stream=True)
    check_verified(report)


def test_sdk_errors_and_refusals_keep_the_templates(monkeypatch):
    real = anthropic.Anthropic
    for api, words in ((FakeAPI("", status=401), "the API key was rejected (401)"),
                       (FakeAPI("", stop_reason="refusal"), "the model declined the request")):
        monkeypatch.setattr(anthropic, "Anthropic", lambda api=api, **kw: real(
            http_client=httpx.Client(transport=httpx.MockTransport(api)), **kw))
        report = coaching_report()
        templates = [e.text for e in report.coaching.explanations]
        llm.annotate(report, CoachConfig(llm=True, llm_model=MODEL, anthropic_api_key="sk-bad"))
        assert [e.text for e in report.coaching.explanations] == templates
        assert report.coaching.llm == {}
        (note,) = [n for n in report.coaching.notes if n.startswith("AI coach")]
        assert words in note and "sk-bad" not in note
