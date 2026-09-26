"""`chess-insights ask`: answers from the last report's JSON, through the verifier (C4). No network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chess_insights.coach import CoachConfig, ask, llm
from chess_insights.report.json_export import to_dict
from test_llm import FakeClient, reply
from test_packet import OPENING_ID, QGA_4E4, coaching_report, epd


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def report_json():
    return json.loads(json.dumps(to_dict(coaching_report())))  # as read from the report file


def test_verified_answer_is_returned(report_json):
    text = "Your Sicilian Defense as Black scores −16 per 100 games against your rating over 63 games."
    client = FakeClient(reply({"answer": text, "claim_ids": [OPENING_ID], "epds": []}))
    assert ask.answer("Why do I lose with the Sicilian?", report_json, CoachConfig(), client=client) == text
    (call,) = client.messages.calls
    assert call["output_config"]["format"]["schema"] == ask.OUTPUT_SCHEMA
    assert "Why do I lose with the Sicilian?" in call["messages"][0]["content"]


def test_answer_with_moves_from_a_named_position(report_json):
    text = "In your rapid game, 4.e4 lets 4...Qxd4 take the d4 pawn; 4.Nf3 keeps it."
    client = FakeClient(reply({"answer": text, "claim_ids": [], "epds": [epd(QGA_4E4)]}))
    question = "What goes wrong in the Queen's Gambit Accepted?"
    assert ask.answer(question, report_json, CoachConfig(), client=client) == text


def test_unverified_answer_falls_back_to_the_report(report_json):
    invented = "Play 4.Qa4 instead: masters score 83% with it."
    client = FakeClient(reply({"answer": invented, "claim_ids": [], "epds": [epd(QGA_4E4)]}))
    out = ask.answer("Why do I lose with the Sicilian?", report_json, CoachConfig(), client=client)
    assert invented not in out
    assert out.startswith("I could not check an answer")
    assert "The Sicilian Defense is costing you points as Black" in out  # the report's own claim, quoted
    assert "(bullet and blitz)" in out  # with the formats behind it


def test_invented_weakness_is_not_shown(report_json):
    client = FakeClient(reply({"answer": "Your weakness is the endgame.", "claim_ids": [], "epds": []}))
    out = ask.answer("What is my weakness in endgames?", report_json, CoachConfig(), client=client)
    assert "Your weakness is the endgame." not in out and out.startswith("I could not check")


def test_no_sdk_or_no_key_says_what_is_missing(monkeypatch, report_json):
    monkeypatch.setattr(llm, "load_sdk", lambda: None)
    out = ask.answer("Why do I lose with the Sicilian?", report_json, CoachConfig())
    assert "chess-insights[llm]" in out and "ANTHROPIC_API_KEY" in out
    assert "Sicilian Defense is costing you points" in out

    monkeypatch.setattr(llm, "load_sdk", lambda: SimpleNamespace())
    out = ask.answer("How is my clock in blitz?", report_json, CoachConfig())
    assert "ANTHROPIC_API_KEY" in out and "chess-insights[llm]" not in out
    assert "too much clock on the opening in blitz" in out


def test_offline_quotes_the_report_without_asking_claude(monkeypatch, report_json):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_sdk", lambda: pytest.fail("--offline must not load the SDK"))
    client = FakeClient(reply({"answer": "Anything.", "claim_ids": [], "epds": []}))
    out = ask.answer("Why do I lose with the Sicilian?", report_json, CoachConfig(offline=True), client=client)
    assert client.messages.calls == []
    assert out.startswith(ask.OFFLINE_REASON + " Here is what your report says that may help:")
    assert "Sicilian Defense is costing you points" in out and "Anything." not in out
    out = ask.answer("Should I buy a wooden board?", report_json, CoachConfig(offline=True))
    assert out.startswith(ask.OFFLINE_REASON) and "Your ratings: rapid 1130, blitz 949, bullet 650." in out


def test_api_error_falls_back(report_json):
    client = FakeClient(RuntimeError("connection reset"))
    out = ask.answer("Why do I lose with the Sicilian?", report_json, CoachConfig(), client=client)
    assert out.startswith("No answer this time") and "Sicilian" in out


def test_nothing_relevant_quotes_the_summary(report_json):
    client = FakeClient(RuntimeError("down"))
    out = ask.answer("Should I buy a wooden board?", report_json, CoachConfig(), client=client)
    assert "Your ratings: rapid 1130, blitz 949, bullet 650." in out


def test_empty_question_and_empty_report():
    assert "Ask a question" in ask.answer("   ", {}, CoachConfig())
    client = FakeClient(RuntimeError("down"))
    assert "nothing more to go on" in ask.answer("Why?", {}, CoachConfig(), client=client)


def test_a_damaged_report_gets_a_message_not_a_traceback():
    out = ask.answer("Why?", {"study_plan": [{"title": "x", "baseline": 5}]}, CoachConfig())
    assert "could not be read" in out
    assert "could not be read" not in ask.answer("Why?", [], CoachConfig(), client=FakeClient(RuntimeError("down")))


def test_an_answer_naming_a_format_nothing_is_about_falls_back(report_json):
    text = "In bullet you keep playing 5...e5 in the Open Sicilian."
    client = FakeClient(reply({"answer": text, "claim_ids": ["mistakes.weakness.sicilian-5-e5"], "epds": []}))
    out = ask.answer("Why do I keep losing in the Open Sicilian?", report_json, CoachConfig(), client=client)
    assert text not in out and out.startswith("I could not check")
