"""Unit tests for aud.intelligence: the provider-agnostic seam, provider
selection, validation of untrusted model output, and the real
`AnthropicBackend` response-parsing boundary.

No hand-written fake provider anywhere in this file. Two different
techniques replace the previous `FakeBackend`:

- Tests of `advisor._parse_model_json`/`_validate_and_build_plan` call
  those pure functions directly with literal strings -- this is testing
  aud's OWN parsing/validation logic with plain Python values, not
  simulating any third party's behaviour, so no replay or fake is needed.
- Tests of `advise()`'s wiring (does it forward model/max_tokens, does the
  user prompt carry reference measurements) and of the real
  `AnthropicBackend.complete()` use `tests/replay.py`'s recorded-response
  replays -- see RECORDING.md. The `content: ["thinking", "text"]` shape
  that once broke this exact parser is replayed from a REAL captured
  response, not reconstructed by hand.

Honest exception, named rather than hidden: `OpenAIBackend`, `GoogleBackend`
and `AzureOpenAIBackend` have never been called live (only Anthropic has --
see RECORDING.md's "anthropic/... was not produced by this run" and the
shared findings this project keeps: "Only Anthropic of four provider
backends has been called live"). There is no recording to replay for the
other three. The tests below construct request/response bodies from each
provider's PUBLICLY DOCUMENTED API shape (the URL already cited in each
backend class's own docstring) rather than a real captured call -- this is
a deliberately weaker guarantee than a replay, and is not the same thing
AGENTS.md SS3b's "never hand-write a mock" rule forbids: that rule is about
guessing a third-party LIBRARY's internal object shape (attributes, method
vs. property) without ever having run it, which a docs-shaped HTTP JSON
body is not. It IS still an assumption that could be wrong if the real API
drifts from its own docs, and that is named here rather than left silent.
What these tests DO prove: aud's own parsing code, exercised against a
well-shaped and a malformed body of the documented shape, raises a named
`AudError` for the malformed one rather than an opaque KeyError -- the
exact class of defect the Anthropic `content[0]["text"]` bug was. What they
do NOT prove: that the real provider actually returns bodies shaped this
way today.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from aud.intelligence import advisor
from aud.intelligence.interface import (
    DEFAULT_MODELS,
    PROVIDER_ENV_VARS,
    AnthropicBackend,
    AzureOpenAIBackend,
    GoogleBackend,
    OpenAIBackend,
    resolve_backend,
)
from aud.schemas import AudError
from tests import replay

# ---------------------------------------------------------------------------
# _parse_model_json / _validate_and_build_plan: aud's OWN pure logic,
# exercised directly with literal inputs -- no backend, no third party.
# ---------------------------------------------------------------------------


def test_non_json_output_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json("not json at all")
    assert excinfo.value.code == "bad_model_output"


def test_top_level_shape_must_be_exactly_stages() -> None:
    bad = {"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "x"}], "notes": "extra"}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_empty_stage_list_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan({"stages": []})
    assert excinfo.value.code == "bad_model_plan"


def test_non_list_stages_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan({"stages": "eq"})
    assert excinfo.value.code == "bad_model_plan"


def test_stage_entry_missing_a_required_key_is_rejected() -> None:
    bad = {"stages": [{"stage": "limit", "params": {}}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unknown_stage_name_is_rejected() -> None:
    bad = {"stages": [{"stage": "cut", "params": {}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_stage_name_not_in_allowed_advise_vocabulary_is_rejected_even_if_it_is_a_real_plan_stage() -> None:
    """'eq_match' is a real render stage but not one advise may choose --
    it needs a stored curve, which advise never holds."""
    bad = {"stages": [{"stage": "eq_match", "params": {"curve": []}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_duplicate_stage_is_rejected() -> None:
    bad = {
        "stages": [
            {"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "x"},
            {"stage": "loudness", "params": {"target_lufs": -16.0}, "reason": "y"},
        ]
    }
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_params_not_an_object_is_rejected() -> None:
    bad = {"stages": [{"stage": "limit", "params": [1, 2, 3], "reason": "x"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_blank_reason_is_rejected() -> None:
    bad = {"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "   "}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_out_of_range_param_is_rejected() -> None:
    """ceiling_dbtp must be <= 0.0 -- this reuses aud.lib.limit's own validator."""
    bad = {"stages": [{"stage": "limit", "params": {"ceiling_dbtp": 3.0}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unexpected_param_name_is_rejected() -> None:
    """compress requires 'bands'; an unknown kwarg is a TypeError from the builder, caught and re-tagged."""
    bad = {"stages": [{"stage": "compress", "params": {"frobnicate": 1.0}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_well_formed_proposal_is_accepted_and_carries_reasoning() -> None:
    good = json.dumps(
        {
            "stages": [
                {
                    "stage": "eq",
                    "params": {"hpf": 60.0, "lpf": None, "peaks": [[3200.0, -2.0, 1.2]]},
                    "reason": "crest factor 14 dB and a quiet noise floor allow a gentle hpf at 60 Hz.",
                },
                {
                    "stage": "limit",
                    "params": {"ceiling_dbtp": -1.0},
                    "reason": "true peak is -6.2 dBTP today; -1.0 dBTP ceiling is the standard R128 max.",
                },
            ]
        }
    )
    plan, reasoning = advisor._validate_and_build_plan(advisor._parse_model_json(good))
    assert [s.stage for s in plan.stages] == ["eq", "limit"]
    assert len(reasoning) == 2
    assert all(r["reason"] for r in reasoning)


# ---------------------------------------------------------------------------
# advise(): wiring, using a REAL recorded response replayed through a
# recording-backed IntelligenceBackend (tests/replay.py). Never a
# hand-authored plan -- these are the model's actual real answers.
# ---------------------------------------------------------------------------


def test_well_formed_real_response_is_accepted_and_carries_reasoning() -> None:
    """Replays anthropic/advise-clean-sonnet-thinking.json -- a REAL
    response (eq, loudness, limit), captured from a 'thinking'-capable
    model, after AnthropicBackend already stripped the thinking block.
    """
    backend = replay.ReplayAdviceBackend("advise-clean-sonnet-thinking")
    plan, reasoning = advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    assert [s.stage for s in plan.stages] == ["eq", "loudness", "limit"]
    assert len(reasoning) == 3
    assert reasoning[0]["stage"] == "eq"


def test_wrapped_code_fence_is_really_present_in_the_haiku_recording_and_is_stripped() -> None:
    """anthropic/advise-clean-haiku.json's real response is genuinely
    wrapped in ```json fences (Haiku's own formatting choice, not
    something this test constructs) -- proving the fence-stripping code
    against real fenced output, not a hand-wrapped string.
    """
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    raw_text = replay.recorded_text_block(backend._recording)
    assert raw_text.strip().startswith("```")  # confirms the fixture really is fenced

    plan, _reasoning = advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    assert [s.stage for s in plan.stages] == ["loudness", "limit"]


def test_backend_is_called_with_the_requested_model_and_max_tokens() -> None:
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model="fake-model-1",
        max_tokens=999,
    )
    assert backend.calls[0]["model"] == "fake-model-1"
    assert backend.calls[0]["max_tokens"] == 999
    assert "aud" in backend.calls[0]["system"]


def test_reference_measurements_reach_the_user_prompt() -> None:
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements={"integrated_lufs": -10.0},
        backend=backend,
        model="fake",
    )
    assert "reference" in backend.calls[0]["user"].lower()
    assert "-10.0" in backend.calls[0]["user"]


# ---------------------------------------------------------------------------
# The real AnthropicBackend, transport replayed -- proves the actual
# content-block parsing that broke on a 'thinking' response, against a
# REAL captured response of each shape.
# ---------------------------------------------------------------------------


def test_anthropic_backend_parses_a_plain_text_content_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replays anthropic/advise-clean-haiku.json -- content: ["text"]."""
    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-haiku")
    backend = AnthropicBackend(api_key="test-key")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)


def test_anthropic_backend_skips_a_leading_thinking_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replays anthropic/advise-clean-sonnet-thinking.json -- content:
    ["thinking", "text"], the exact shape that raised KeyError on
    `content[0]["text"]` before this was fixed. Found by calling a real
    reasoning-capable model; no hand-written fake could have produced it.
    """
    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-sonnet-thinking")
    assert [b["type"] for b in recording["response"]["content"]] == ["thinking", "text"]

    backend = AnthropicBackend(api_key="test-key")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)


def test_anthropic_backend_replay_fails_loudly_for_a_missing_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(replay.RecordingNotFoundError):
        replay.install_anthropic_urlopen_replay(monkeypatch, "does-not-exist")


# ---------------------------------------------------------------------------
# OpenAI / Google / Azure OpenAI response-shape parsing -- docs-shaped, not
# recorded live (see the module docstring's honest caveat above). Each pair
# proves the same two things the Anthropic replay tests prove for the real
# recorded shape: a well-formed body parses to the expected text, and a
# malformed one raises AudError(code="provider_request_failed") -- never an
# opaque KeyError/IndexError -- which is exactly the class of defect the
# real content[0]["text"] bug on Anthropic was.
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _install_docs_shaped_response(monkeypatch: pytest.MonkeyPatch, body: dict) -> None:
    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del request, timeout
        return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def test_openai_backend_parses_the_documented_chat_completions_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://platform.openai.com/docs/api-reference/chat -- documented
    shape, not a recorded live call (see module docstring)."""
    _install_docs_shaped_response(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hello from openai"}}]}
    )
    backend = OpenAIBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gpt-4o-mini") == "hello from openai"


def test_openai_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docs_shaped_response(monkeypatch, {"choices": []})  # documented field, unexpectedly empty
    backend = OpenAIBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gpt-4o-mini")
    assert excinfo.value.code == "provider_request_failed"


def test_google_backend_parses_the_documented_generate_content_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://ai.google.dev/api/generate-content -- documented shape, not
    a recorded live call (see module docstring)."""
    _install_docs_shaped_response(
        monkeypatch, {"candidates": [{"content": {"parts": [{"text": "hello from google"}]}}]}
    )
    backend = GoogleBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gemini-2.0-flash") == "hello from google"


def test_google_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docs_shaped_response(monkeypatch, {"candidates": [{"content": {"parts": []}}]})
    backend = GoogleBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gemini-2.0-flash")
    assert excinfo.value.code == "provider_request_failed"


def test_azure_openai_backend_parses_the_documented_chat_completions_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """https://learn.microsoft.com/azure/ai-services/openai/reference --
    documented shape, not a recorded live call (see module docstring)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-resource.openai.azure.com")
    _install_docs_shaped_response(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hello from azure"}}]}
    )
    backend = AzureOpenAIBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gpt-4o-mini") == "hello from azure"


def test_azure_openai_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-resource.openai.azure.com")
    _install_docs_shaped_response(monkeypatch, {"choices": [{"message": {}}]})  # missing "content"
    backend = AzureOpenAIBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gpt-4o-mini")
    assert excinfo.value.code == "provider_request_failed"


# --- Provider selection -------------------------------------------------------


def test_provider_env_vars_match_the_manifest_order() -> None:
    assert PROVIDER_ENV_VARS == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )


def test_resolve_backend_raises_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AudError) as excinfo:
        resolve_backend(None)
    assert excinfo.value.code == "provider_credential_missing"
    for var in PROVIDER_ENV_VARS:
        assert var in excinfo.value.message or var in excinfo.value.remedy


def test_resolve_backend_picks_first_configured_provider_in_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    _backend, provider, model = resolve_backend(None)
    assert provider == "openai"
    assert model == DEFAULT_MODELS["openai"]


def test_resolve_backend_honours_explicit_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    _backend, provider, model = resolve_backend("claude-custom")
    assert provider == "anthropic"
    assert model == "claude-custom"


def test_resolve_backend_honours_aud_model_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("AUD_MODEL", "claude-from-env")
    _backend, provider, model = resolve_backend(None)
    assert provider == "anthropic"
    assert model == "claude-from-env"


def test_gemini_api_key_also_selects_the_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    _backend, provider, _model = resolve_backend(None)
    assert provider == "google"


# ---------------------------------------------------------------------------
# Model-tier-dependent gate/expand behaviour, pinned against REAL recorded
# advise responses (never a hand-authored plan -- see tests/replay.py and
# RECORDING.md's anthropic/ note).
#
# Measured on identical prompts and identical, real measurements
# (noise_floor_dbfs -30.51 vs integrated_lufs -13.06 -- a ~17.4 dB gap,
# well under the ~40 dB "clean" separation the system prompt names):
# claude-sonnet-5 reached for 'expand' and cited the two numbers; claude-
# haiku-4-5 did not reach for either 'gate' or 'expand' on the same
# material. On genuinely clean material (noise_floor_dbfs -64.54 vs
# integrated_lufs -11.76, a ~52.8 dB gap), neither model proposes one.
# Before this test, that was anecdote from one interactive session. These
# assertions run the real recorded text through aud's OWN validator and
# plan builder (advisor._validate_and_build_plan, via advisor.advise()),
# so a prompt change that broke this reasoning would fail a real test
# rather than requiring someone to notice the story stopped being true.
# ---------------------------------------------------------------------------


def _advise_from_recording(recording_name: str) -> tuple[list[str], list[dict[str, str]]]:
    backend = replay.ReplayAdviceBackend(recording_name)
    plan, reasoning = advisor.advise(
        {"integrated_lufs": -13.06, "true_peak_dbtp": -6.95, "noise_floor_dbfs": -30.51},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    return [s.stage for s in plan.stages], reasoning


def test_hissy_material_sonnet_thinking_response_yields_a_quiet_end_stage() -> None:
    """Replays anthropic/advise-hissy-sonnet-thinking.json -- the real
    response that reached for 'expand' and cited noise_floor_dbfs vs
    integrated_lufs. Pins that this recorded response, run through aud's
    real validator, actually produces a chain containing a quiet-end
    repair stage -- not just that the recorded text mentions one.
    """
    stage_names, reasoning = _advise_from_recording("advise-hissy-sonnet-thinking")
    assert "expand" in stage_names
    assert "gate" not in stage_names  # the model chose the gentler device, not the harder one
    expand_reason = next(r["reason"] for r in reasoning if r["stage"] == "expand")
    assert "17.4" in expand_reason or "noise_floor" in expand_reason.lower()


def test_hissy_material_haiku_response_does_not_reach_for_a_quiet_end_stage() -> None:
    """Replays anthropic/advise-hissy-haiku.json -- the real response from
    a smaller/faster model on the SAME material and prompt that DID NOT
    propose gate/expand. This is the pinned half of the model-tier finding:
    identical measurements, different model, different chain.
    """
    stage_names, _reasoning = _advise_from_recording("advise-hissy-haiku")
    assert "gate" not in stage_names
    assert "expand" not in stage_names


@pytest.mark.parametrize("recording_name", ["advise-clean-haiku", "advise-clean-sonnet-thinking"])
def test_clean_material_responses_never_reach_for_a_quiet_end_stage(recording_name: str) -> None:
    """Replays both clean-material recordings (haiku and sonnet-thinking):
    on a ~52.8 dB noise-floor/loudness separation, well above the system
    prompt's ~40 dB clean-recording guideline, neither real model proposed
    gate or expand. This is the control for the two tests above -- proves
    the quiet-end stage tracks the ACTUAL noise floor, not just the model.
    """
    stage_names, _reasoning = _advise_from_recording(recording_name)
    assert "gate" not in stage_names
    assert "expand" not in stage_names


def test_azure_openai_requires_an_endpoint_too(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    with pytest.raises(AudError) as excinfo:
        resolve_backend(None)
    assert excinfo.value.code == "provider_config_incomplete"
