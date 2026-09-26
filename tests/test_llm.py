"""The LLM coach and Maia-2 (C4), with a mocked API client and a fake predictor: no network, no anthropic package."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from chess_insights.coach import CoachConfig, finish_coaching, llm, maia
from chess_insights.coach.config import DEFAULT_LLM_MODEL
from chess_insights.coach.packet import build_packet
from chess_insights.report.json_export import to_dict
from test_packet import QGA_4E4, REPEATED_ID, SICILIAN_2NC6, SICILIAN_5E5, coaching_report, epd

GOOD_SICILIAN = (
    "After 5...e5 the knight goes 6.Ndb5 and 7.Nd6+, checking your king and taking the dark-squared bishop. "
    "That line ends 1.5 pawns down for you, against 0.5 after 5...a6. "
    "Before a pawn push, check which squares it stops guarding."
)
ILLEGAL_QGA = "4.e4 walks into 4...Qxd4, and then 5.Qxh7 is hopeless."
INVENTED_EVAL = "After 2...Nc6, 3.d5 Nb8 leaves you 3.4 pawns worse."


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)


def reply(data, stop_reason="end_turn"):
    """A Messages API response: a (hidden) thinking block, then the JSON text."""
    text = data if isinstance(data, str) else json.dumps(data)
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


class FakeStream:
    """``messages.stream(...)``: the request is made on entering, like the SDK's stream manager."""

    def __init__(self, out):
        self.out = out

    def __enter__(self):
        if isinstance(self.out, BaseException):
            raise self.out
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.out


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.responses.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        out = self.responses.pop(0)
        if isinstance(out, TypeError):  # an unknown keyword fails at the call, before any request
            raise out
        return FakeStream(out)


class FakeClient:
    def __init__(self, *responses, **kwargs):
        self.kwargs = kwargs
        self.messages = FakeMessages(responses)


def plan_titles(report):
    return [item.title for item in report.study_plan]


def llm_reply(report):
    titles = plan_titles(report)
    return {
        "explanations": [
            {"epd": epd(SICILIAN_5E5), "text": GOOD_SICILIAN, "claim_ids": [REPEATED_ID]},
            {"epd": epd(QGA_4E4), "text": ILLEGAL_QGA, "claim_ids": []},
            {"epd": epd(SICILIAN_2NC6), "text": INVENTED_EVAL, "claim_ids": []},
            {"epd": "8/8/8/8/8/8/8/8 w - -", "text": "A position that is not in the report.", "claim_ids": []},
        ],
        "weekly_plan": [
            {"item": titles[0], "minutes_per_week": 60, "recheck_date": "2026-10-24"},
            {"item": "Memorise the Najdorf", "minutes_per_week": 30, "recheck_date": "2026-10-24"},
        ],
    }


def comparable(report):
    data = to_dict(report)
    data.pop("generated_at", None)
    return data


# --------------------------------------------------------------------------- LLM off
def test_report_is_identical_with_the_llm_off_or_without_a_key():
    base = comparable(coaching_report())
    off = finish_coaching(coaching_report(), CoachConfig(llm=False))
    no_key = finish_coaching(coaching_report(), CoachConfig(llm=True))  # no ANTHROPIC_API_KEY anywhere
    assert comparable(off) == base
    assert comparable(no_key) == base


def test_missing_sdk_leaves_templates_and_a_note(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_sdk", lambda: None)
    report = coaching_report()
    llm.annotate(report, CoachConfig(llm=True))
    assert all(e.text_source == "template" for e in report.coaching.explanations)
    assert any("chess-insights[llm]" in n for n in report.coaching.notes)
    assert report.coaching.llm == {} and report.coaching.weekly_plan == []


def test_offline_sends_nothing_whatever_the_credentials(monkeypatch):
    """--offline: no request to Claude even with a key, the SDK and a client at hand; one note says so."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "load_sdk", lambda: pytest.fail("--offline must not load the SDK"))
    base = comparable(coaching_report())
    for client in (None, FakeClient(reply(llm_reply(coaching_report())))):
        report = coaching_report()
        llm.annotate(report, CoachConfig(llm=True, offline=True, anthropic_api_key="sk-test"), client=client)
        assert client is None or client.messages.calls == []
        assert report.coaching.notes[-1] == llm.OFFLINE_NOTE and llm.OFFLINE_NOTE.startswith("AI coach skipped")
        report.coaching.notes.pop()
        assert comparable(report) == base  # templates, plan and counts untouched
    report = finish_coaching(coaching_report(), CoachConfig(llm=True, offline=True))
    assert report.coaching.notes.count(llm.OFFLINE_NOTE) == 1 and "sk-test" not in json.dumps(to_dict(report))


# --------------------------------------------------------------------------- the request
def test_request_uses_structured_output_and_the_configured_model():
    report = coaching_report()
    client = FakeClient(reply({"explanations": [], "weekly_plan": []}))
    llm.annotate(report, CoachConfig(llm=True, practice_minutes=20), client=client)
    (call,) = client.messages.calls
    assert call["model"] == DEFAULT_LLM_MODEL == "claude-opus-5-5"
    assert call["max_tokens"] == llm.MAX_TOKENS
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": llm.OUTPUT_SCHEMA}
    assert call["output_config"]["effort"] == llm.EFFORT
    assert "thinking" not in call and "tool_choice" not in call  # adaptive thinking; no forced tool use
    assert "centipawns" in call["system"] and "claim_ids" in call["system"]
    (message,) = call["messages"]
    assert message["role"] == "user"
    assert "20 minutes a day" in message["content"] and "140 minutes a week" in message["content"]
    sent = json.loads(re.search(r"<packet>\n(.*)\n</packet>", message["content"], re.S).group(1))
    assert sent == json.loads(json.dumps(build_packet(report)))


def test_client_is_built_with_the_key_timeout_and_retries(monkeypatch):
    built = {}

    class FakeAnthropic(FakeClient):
        def __init__(self, **kwargs):
            built.update(kwargs)
            super().__init__(reply({"explanations": [], "weekly_plan": []}), **kwargs)

    monkeypatch.setattr(llm, "load_sdk", lambda: SimpleNamespace(Anthropic=FakeAnthropic))
    report = coaching_report()
    llm.annotate(report, CoachConfig(llm=True, anthropic_api_key="sk-from-settings", llm_model="claude-test"))
    assert built == {"api_key": "sk-from-settings", "timeout": llm.TIMEOUT_S, "max_retries": llm.MAX_RETRIES}
    assert report.coaching.llm["model"] == "claude-test"


def test_a_client_without_streaming_gets_a_plain_request():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return reply({"explanations": [], "weekly_plan": []})

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    report = coaching_report()
    llm.annotate(report, CoachConfig(llm=True), client=client)
    assert len(calls) == 1 and report.coaching.llm["accepted"] == 0


def test_older_sdk_without_output_config_gets_it_as_a_body_field():
    ok = reply({"explanations": [], "weekly_plan": []})
    client = FakeClient(TypeError("create() got an unexpected keyword argument 'output_config'"), ok)
    llm.annotate(coaching_report(), CoachConfig(llm=True), client=client)
    first, second = client.messages.calls
    assert "output_config" not in second and second["extra_body"]["output_config"]["format"]["type"] == "json_schema"


# --------------------------------------------------------------------------- the verifier on the reply
def test_verified_texts_replace_templates_and_the_rest_keep_them():
    report = coaching_report()
    templates = {e.epd: e.text for e in report.coaching.explanations}
    claims_before = (to_dict(report)["strengths"], to_dict(report)["weaknesses"], to_dict(report)["study_plan"])
    llm.annotate(report, CoachConfig(llm=True, practice_minutes=20), client=FakeClient(reply(llm_reply(report))))
    by_epd = {e.epd: e for e in report.coaching.explanations}

    assert by_epd[epd(SICILIAN_5E5)].text == GOOD_SICILIAN
    assert by_epd[epd(SICILIAN_5E5)].text_source == "llm"
    for fen in (QGA_4E4, SICILIAN_2NC6):  # an illegal move and an invented evaluation
        assert by_epd[epd(fen)].text == templates[epd(fen)]
        assert by_epd[epd(fen)].text_source == "template"

    llm_info = report.coaching.llm
    assert (llm_info["model"], llm_info["sent"], llm_info["accepted"], llm_info["rejected"], llm_info["ignored"]) == (
        "claude-opus-5-5", 3, 1, 2, 1)  # the text for a position it was not sent is ignored, not checked
    problems = {r["epd"]: " ".join(r["problems"]) for r in llm_info["rejections"]}
    assert "5.Qxh7" in problems[epd(QGA_4E4)] and "3.4" in problems[epd(SICILIAN_2NC6)]
    (note,) = [n for n in report.coaching.notes if n.startswith("AI coach")]
    assert "1 of the 3 explanations it was given reworded" in note
    assert "2 failed the fact check and keep the built-in text" in note

    # the weekly plan keeps the study-plan item only
    assert [p.item for p in report.coaching.weekly_plan] == [plan_titles(report)[0]]
    assert report.coaching.weekly_plan[0].minutes_per_week == 60
    assert llm_info["plan_dropped"] == 1

    # the LLM never touches a claim or the study plan
    assert (to_dict(report)["strengths"], to_dict(report)["weaknesses"], to_dict(report)["study_plan"]) == claims_before


def test_a_second_text_for_the_same_position_is_ignored():
    report = coaching_report()
    data = {"explanations": [{"epd": epd(SICILIAN_5E5), "text": GOOD_SICILIAN, "claim_ids": []}] * 2, "weekly_plan": []}
    llm.annotate(report, CoachConfig(llm=True), client=FakeClient(reply(data)))
    counts = report.coaching.llm
    assert (counts["accepted"], counts["rejected"], counts["ignored"]) == (1, 0, 1)


# --------------------------------------------------------------------------- failures keep the templates
class FakeAPIStatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class FakeRateLimitError(FakeAPIStatusError):
    pass


class FakeAuthenticationError(FakeAPIStatusError):
    pass


class FakeAPIConnectionError(Exception):
    pass


class FakeAPITimeoutError(FakeAPIConnectionError):
    pass


FAKE_SDK_ERRORS = dict(
    APIStatusError=FakeAPIStatusError,
    RateLimitError=FakeRateLimitError,
    AuthenticationError=FakeAuthenticationError,
    APIConnectionError=FakeAPIConnectionError,
    APITimeoutError=FakeAPITimeoutError,
)


@pytest.mark.parametrize(
    "error, words",
    [
        (FakeRateLimitError("rate limited", 429), "rate-limiting this key after retrying (429)"),
        (FakeAuthenticationError("invalid x-api-key", 401), "API key was rejected (401)"),
        (FakeAPIStatusError("overloaded", 529), "API returned an error (529)"),
        (FakeAPITimeoutError("timed out"), "did not answer within 300 s"),
        (FakeAPIConnectionError("connection reset"), "could not be reached"),
    ],
)
def test_api_errors_leave_the_templates(monkeypatch, error, words):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    sdk = SimpleNamespace(Anthropic=lambda **kw: FakeClient(error, **kw), **FAKE_SDK_ERRORS)
    monkeypatch.setattr(llm, "load_sdk", lambda: sdk)
    report = coaching_report()
    before = comparable(coaching_report())
    llm.annotate(report, CoachConfig(llm=True))
    after = comparable(report)
    note = after["coaching"]["notes"].pop()
    assert words in note and "keep their built-in text" in note
    assert after == before  # nothing else changed


@pytest.mark.parametrize(
    "response, words",
    [
        (reply({}, stop_reason="refusal"), "declined"),
        (reply('{"explanations": [', stop_reason="max_tokens"), "cut off"),
        (reply("not json"), "not valid JSON"),
        (RuntimeError("socket closed"), "RuntimeError: socket closed"),
    ],
)
def test_unusable_replies_leave_the_templates(response, words):
    report = coaching_report()
    llm.annotate(report, CoachConfig(llm=True), client=FakeClient(response))
    assert all(e.text_source == "template" for e in report.coaching.explanations)
    assert words in report.coaching.notes[-1]
    assert report.coaching.llm == {} and report.coaching.weekly_plan == []


def test_no_coaching_no_call():
    report = coaching_report()
    report.coaching = None
    client = FakeClient()
    llm.annotate(report, CoachConfig(llm=True), client=client)
    assert client.messages.calls == []


# --------------------------------------------------------------------------- Maia-2
PROBS = {"a7a6": 0.40, "e6e5": 0.30, "g1f3": 0.10, "e2e4": 0.60, "c5d4": 0.80, "b8c6": 0.10}


def fake_predictor(calls):
    def predict(fen, rating, model_type):
        calls.append((epd(fen), rating, model_type))
        return dict(PROBS)

    return predict


def test_maia_adds_probabilities_and_ranks_by_drop_times_p_best():
    report = coaching_report()
    cfg = CoachConfig(maia=True)
    calls = []
    maia.annotate(report, cfg, predictor=fake_predictor(calls))
    exps = report.coaching.explanations
    # drop x P(best): 2...Nc6 18 x 0.8 = 14.4, 5...e5 21 x 0.4 = 8.4, 4.e4 15 x 0.1 = 1.5
    assert [e.epd for e in exps] == [epd(SICILIAN_2NC6), epd(SICILIAN_5E5), epd(QGA_4E4)]
    models = {e: (rating, model) for e, rating, model in calls}
    assert models[epd(SICILIAN_2NC6)][1] == "blitz"  # bullet uses the blitz model
    assert models[epd(SICILIAN_5E5)][1] == "blitz"
    assert models[epd(QGA_4E4)][1] == "rapid"
    sic = next(e for e in exps if e.epd == epd(SICILIAN_5E5))
    assert sic.maia == {
        "rating": maia.lichess_rating(949, "blitz", cfg)[0],
        "chesscom_rating": 949,
        "model": "blitz",
        "p_best": 0.4,
        "p_played": 0.3,
    }
    assert report.coaching.settings["maia"]["positions"] == 3


def test_maia_rates_you_with_the_rating_map_and_its_overrides():
    """One conversion for the whole layer: rating_map.lichess_equivalent with CoachConfig.rating_map."""
    from chess_insights.coach import rating_map

    assert maia.lichess_rating(949, "blitz", CoachConfig()) == (1390, rating_map.SOURCE_NAME)  # the plan's "about 1,390"
    assert maia.lichess_rating(900, "blitz", CoachConfig())[0] == 1360
    cfg = CoachConfig(rating_map={"rapid": [[1000, 1400], [1200, 1550]]})
    assert maia.lichess_rating(1100, "rapid", cfg) == (1475, "your rating map")
    assert maia.lichess_rating(1100, "blitz", cfg)[0] == rating_map.to_lichess(1100, "blitz")  # other formats as built in
    # daily games (Maia's rapid model) convert with the rapid rows, the overrides' when they give some
    assert maia.lichess_rating(1100, "daily", cfg) == (1475, "your rating map, rapid rows")
    assert maia.lichess_rating(1100, "daily", CoachConfig())[0] == rating_map.to_lichess(1100, "rapid")
    assert not hasattr(maia, "FALLBACK_POINTS") and not hasattr(maia, "_interpolate")


def test_maia_passes_your_rating_map_to_the_predictor():
    report = coaching_report()
    calls = []
    cfg = CoachConfig(maia=True, rating_map={"blitz": [[900, 1500], [1000, 1600]]})
    maia.annotate(report, cfg, predictor=fake_predictor(calls))
    sic = next(e for e in report.coaching.explanations if e.epd == epd(SICILIAN_5E5))
    assert sic.maia["rating"] == 1550 and (epd(SICILIAN_5E5), 1550, "blitz") in calls
    assert report.coaching.settings["maia"]["ratings"]["blitz"]["source"] == "your rating map"


def test_maia_reads_moves_from_the_side_to_moves_view():
    report = coaching_report()
    mirrored = {maia._mirror(k): v for k, v in PROBS.items()}
    maia.annotate(report, CoachConfig(maia=True), predictor=lambda fen, r, m: dict(mirrored))
    sic = next(e for e in report.coaching.explanations if e.epd == epd(SICILIAN_5E5))
    assert (sic.maia["p_best"], sic.maia["p_played"]) == (0.4, 0.3)


def test_maia_without_the_package_or_offline_leaves_a_note(monkeypatch):
    monkeypatch.setattr(maia, "maia2_predictor", lambda: None)
    report = coaching_report()
    order = [e.epd for e in report.coaching.explanations]
    maia.annotate(report, CoachConfig(maia=True))
    assert any("maia2" in n for n in report.coaching.notes)
    assert [e.epd for e in report.coaching.explanations] == order
    offline = coaching_report()
    maia.annotate(offline, CoachConfig(maia=True, offline=True), predictor=None)
    assert any("--offline" in n for n in offline.coaching.notes)


def test_maia_failures_stop_early_and_leave_the_order():
    report = coaching_report()
    order = [e.epd for e in report.coaching.explanations]
    calls = []

    def broken(fen, rating, model_type):
        calls.append(fen)
        raise RuntimeError("weights not found")

    maia.annotate(report, CoachConfig(maia=True), predictor=broken)
    assert [e.epd for e in report.coaching.explanations] == order
    assert all(e.maia == {} for e in report.coaching.explanations)
    assert any("weights not found" in n for n in report.coaching.notes)


def test_finish_coaching_runs_maia_then_the_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(maia, "maia2_predictor", lambda: fake_predictor(calls))
    seen = {}

    def fake_llm(report, cfg):
        seen["order"] = [e.epd for e in report.coaching.explanations]

    monkeypatch.setattr(llm, "annotate", fake_llm)
    finish_coaching(coaching_report(), CoachConfig(maia=True, llm=True))
    assert seen["order"][0] == epd(SICILIAN_2NC6)  # the LLM sees Maia's order
    assert len(calls) == 3


# --------------------------------------------------------------------------- review fixes
def test_the_note_says_how_many_explanations_were_not_sent():
    import copy

    report = coaching_report()
    extra = []
    for i in range(25):  # more positions than the coach is given
        e = copy.deepcopy(report.coaching.explanations[0])
        e.epd = f"{e.epd} #{i}"
        extra.append(e)
    report.coaching.explanations += extra
    llm.annotate(report, CoachConfig(llm=True), client=FakeClient(reply({"explanations": [], "weekly_plan": []})))
    (note,) = [n for n in report.coaching.notes if n.startswith("AI coach")]
    assert "0 of the 20 explanations it was given reworded" in note
    assert "the other 8 keep the built-in text (it is given at most 20 positions)" in note
    assert report.coaching.llm["sent"] == 20


def test_unexpected_errors_are_short_and_never_show_the_key():
    key = "sk-ant-secret-123"
    error = RuntimeError(f"request failed with header x-api-key: {key} " + "x" * 500)
    report = coaching_report()
    llm.annotate(report, CoachConfig(llm=True, anthropic_api_key=key), client=FakeClient(error))
    note = report.coaching.notes[-1]
    assert key not in note and "[key]" in note and len(note) < 320


def test_rejected_texts_are_kept_short_for_the_record():
    report = coaching_report()
    long_text = "After 5...e5 you lose 7 pawns. " * 40
    data = {"explanations": [{"epd": epd(SICILIAN_5E5), "text": long_text, "claim_ids": []}], "weekly_plan": []}
    llm.annotate(report, CoachConfig(llm=True), client=FakeClient(reply(data)))
    (rejection,) = report.coaching.llm["rejections"]
    assert len(rejection["text"]) <= llm.MAX_REJECTED_CHARS


def test_maia_keys_on_the_real_board_are_left_as_they_are():
    black_to_move = SICILIAN_5E5
    real = {"a7a6": 0.4, "e6e5": 0.3, "g8f6": 0.2}
    assert maia.board_view(real, black_to_move) == real
    flipped = {maia._mirror(k): v for k, v in real.items()}
    assert maia.board_view(flipped, black_to_move) == real
