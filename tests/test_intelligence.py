"""Unit tests for aud.intelligence: the provider-agnostic seam, provider
selection, and validation of untrusted model output.

No network call anywhere in this file -- every test injects a FakeBackend
implementing the IntelligenceBackend protocol directly.
"""

from __future__ import annotations

import json

import pytest

from aud.intelligence.advisor import advise
from aud.intelligence.interface import DEFAULT_MODELS, PROVIDER_ENV_VARS, resolve_backend
from aud.schemas import AudError


class FakeBackend:
    """A canned IntelligenceBackend -- returns whatever text it was built with."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict] = []

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        self.calls.append({"system": system, "user": user, "model": model, "max_tokens": max_tokens})
        return self.response_text


_MEASUREMENTS = {
    "sample_rate": 44100,
    "channels": 2,
    "duration_seconds": 12.0,
    "integrated_lufs": -22.5,
    "true_peak_dbtp": -6.2,
    "crest_factor_db": 14.0,
    "sibilance_ratio": 0.02,
    "octave_band_energy_db": {"125_hz": -30.0, "1000_hz": -20.0},
    "noise_floor_dbfs": -55.0,
}

_GOOD_RESPONSE = json.dumps(
    {
        "stages": [
            {
                "stage": "eq",
                "params": {"hpf": 60.0, "lpf": None, "peaks": [[3200.0, -2.0, 1.2]]},
                "reason": "crest factor 14 dB and a quiet noise floor (-55 dBFS) allow a gentle hpf at 60 Hz.",
            },
            {
                "stage": "compress",
                "params": {"bands": [120.0, 900.0, 5500.0], "ratio": 2.0},
                "reason": "moderate crest factor suggests light multiband control is enough.",
            },
            {
                "stage": "loudness",
                "params": {"target_lufs": -14.0},
                "reason": "measured integrated loudness is -22.5 LUFS, well below the -14 target.",
            },
            {
                "stage": "limit",
                "params": {"ceiling_dbtp": -1.0},
                "reason": "true peak is -6.2 dBTP today; -1.0 dBTP ceiling is the standard R128 max.",
            },
        ]
    }
)


def _advise(response_text: str, *, model: str = "fake-model"):
    return advise(
        _MEASUREMENTS,
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=FakeBackend(response_text),
        model=model,
    )


# --- Well-formed proposals ---------------------------------------------------


def test_well_formed_plan_is_accepted_and_carries_reasoning() -> None:
    plan, reasoning = _advise(_GOOD_RESPONSE, model="fake-model-1")
    stage_names = [s.stage for s in plan.stages]
    assert stage_names == ["eq", "compress", "loudness", "limit"]
    assert len(reasoning) == 4
    assert all(r["reason"] for r in reasoning)
    assert reasoning[0]["stage"] == "eq"


def test_backend_is_called_with_the_requested_model_and_max_tokens() -> None:
    backend = FakeBackend(_GOOD_RESPONSE)
    advise(
        _MEASUREMENTS,
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


def test_wrapped_code_fence_is_stripped_before_parsing() -> None:
    fenced = "```json\n" + _GOOD_RESPONSE + "\n```"
    plan, _ = _advise(fenced)
    assert [s.stage for s in plan.stages] == ["eq", "compress", "loudness", "limit"]


def test_reference_measurements_reach_the_user_prompt() -> None:
    backend = FakeBackend(_GOOD_RESPONSE)
    advise(
        _MEASUREMENTS,
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements={"integrated_lufs": -10.0},
        backend=backend,
        model="fake",
    )
    assert "reference" in backend.calls[0]["user"].lower()
    assert "-10.0" in backend.calls[0]["user"]


# --- Untrusted output: every rejection path ---------------------------------


def test_non_json_output_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        _advise("not json at all")
    assert excinfo.value.code == "bad_model_output"


def test_top_level_shape_must_be_exactly_stages() -> None:
    bad = json.dumps(
        {"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "x"}], "notes": "extra"}
    )
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_empty_stage_list_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        _advise(json.dumps({"stages": []}))
    assert excinfo.value.code == "bad_model_plan"


def test_non_list_stages_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        _advise(json.dumps({"stages": "eq"}))
    assert excinfo.value.code == "bad_model_plan"


def test_stage_entry_missing_a_required_key_is_rejected() -> None:
    bad = json.dumps({"stages": [{"stage": "limit", "params": {}}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unknown_stage_name_is_rejected() -> None:
    bad = json.dumps({"stages": [{"stage": "cut", "params": {}, "reason": "because"}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_stage_name_not_in_allowed_advise_vocabulary_is_rejected_even_if_it_is_a_real_plan_stage() -> None:
    """'eq_match' is a real render stage but not one advise may choose --
    it needs a stored curve, which advise never holds."""
    bad = json.dumps({"stages": [{"stage": "eq_match", "params": {"curve": []}, "reason": "because"}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_duplicate_stage_is_rejected() -> None:
    bad = json.dumps(
        {
            "stages": [
                {"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "x"},
                {"stage": "loudness", "params": {"target_lufs": -16.0}, "reason": "y"},
            ]
        }
    )
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_params_not_an_object_is_rejected() -> None:
    bad = json.dumps({"stages": [{"stage": "limit", "params": [1, 2, 3], "reason": "x"}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_blank_reason_is_rejected() -> None:
    bad = json.dumps({"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "   "}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_out_of_range_param_is_rejected() -> None:
    """ceiling_dbtp must be <= 0.0 -- this reuses aud.lib.limit's own validator."""
    bad = json.dumps({"stages": [{"stage": "limit", "params": {"ceiling_dbtp": 3.0}, "reason": "because"}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unexpected_param_name_is_rejected() -> None:
    """compress requires 'bands'; an unknown kwarg is a TypeError from the builder, caught and re-tagged."""
    bad = json.dumps({"stages": [{"stage": "compress", "params": {"frobnicate": 1.0}, "reason": "because"}]})
    with pytest.raises(AudError) as excinfo:
        _advise(bad)
    assert excinfo.value.code == "bad_model_plan"


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


def test_azure_openai_requires_an_endpoint_too(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    with pytest.raises(AudError) as excinfo:
        resolve_backend(None)
    assert excinfo.value.code == "provider_config_incomplete"
