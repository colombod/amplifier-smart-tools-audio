"""Integration tests for `advise`/`master` -- the smart tier.

Every test injects a FakeBackend (no network, no credential, no cost)
except the ones proving the opposite: that with no credential configured,
advise/master refuse honestly while every deterministic verb keeps working.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aud import lib
from aud.schemas import AudError

_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


class FakeBackend:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        return self.response_text


_SIMPLE_CHAIN = json.dumps(
    {
        "stages": [
            {"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "bring it up to -14 LUFS"},
            {"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "protect against inter-sample overs"},
        ]
    }
)


# --- lib-level: FakeBackend injection, real render/verify -------------------


def test_lib_advise_returns_a_plan_that_pipes_into_render(tiny_wav: Path, tmp_path: Path) -> None:
    outcome = lib.advise(str(tiny_wav), target_lufs=-14.0, backend=FakeBackend(_SIMPLE_CHAIN), model="fake-1")
    assert [s["stage"] for s in outcome["stages"]] == ["loudness", "limit"]
    assert outcome["measurements"]["sample_rate"] == 44100
    assert outcome["provider"] == "injected"
    assert outcome["model"] == "fake-1"

    out_path = tmp_path / "out.wav"
    render_result = lib.render(outcome["plan"], str(tiny_wav), str(out_path))
    assert out_path.exists()
    stage_names = [s["stage"] for s in render_result["report"]["stages"]]
    assert stage_names == ["loudness", "limit"]


def test_lib_master_renders_and_verifies_end_to_end(tiny_wav: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "mastered.wav"
    result = lib.master(
        str(tiny_wav),
        str(out_path),
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        backend=FakeBackend(_SIMPLE_CHAIN),
        model="fake-1",
    )
    assert out_path.exists()
    assert result["out_path"] == str(out_path)
    assert [s["stage"] for s in result["stages"]] == ["loudness", "limit"]
    assert result["render"]["stages"][-1]["stage"] == "limit"
    assert result["verify"]["lufs_ok"] is True
    assert result["verify"]["ceiling_ok"] is True
    assert isinstance(result["verify"]["measured"]["integrated_lufs"], float)
    assert isinstance(result["verify"]["measured"]["true_peak_dbtp"], float)
    # The plan travels as a plain dict in master's result -- JSON-serializable, matching the CLI's envelope.
    assert result["plan"]["stages"][0]["stage"] in ("loudness", "limit")
    json.dumps(result)  # must not raise: master's whole result is one JSON document


def test_lib_master_dry_run_does_not_render(tiny_wav: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "should-not-exist.wav"
    result = lib.master(
        str(tiny_wav),
        str(out_path),
        backend=FakeBackend(_SIMPLE_CHAIN),
        model="fake-1",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert "verify" not in result
    assert "render" not in result
    assert not out_path.exists()


def test_lib_advise_with_reference_measures_both_files(tiny_wav: Path, tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    reference = tmp_path / "reference.wav"
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(reference), np.stack([tone, tone], axis=1), sr, subtype="PCM_24")

    outcome = lib.advise(
        str(tiny_wav),
        reference_path=str(reference),
        backend=FakeBackend(_SIMPLE_CHAIN),
        model="fake-1",
    )
    assert outcome["reference_measurements"] is not None
    assert outcome["reference_measurements"]["sample_rate"] == 44100


def test_lib_advise_refuses_with_no_credentials(tiny_wav: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AudError) as excinfo:
        lib.advise(str(tiny_wav))
    assert excinfo.value.code == "provider_credential_missing"


def test_lib_master_refuses_with_no_credentials_and_touches_nothing(
    tiny_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    out_path = tmp_path / "untouched.wav"
    with pytest.raises(AudError) as excinfo:
        lib.master(str(tiny_wav), str(out_path))
    assert excinfo.value.code == "provider_credential_missing"
    assert not out_path.exists()


def test_lib_advise_bad_model_output_never_produces_a_partial_plan(tiny_wav: Path) -> None:
    with pytest.raises(AudError) as excinfo:
        lib.advise(str(tiny_wav), backend=FakeBackend("garbage, not json"), model="fake-1")
    assert excinfo.value.code == "bad_model_output"


# --- CLI-level: real subprocess, real refusal, real deterministic verbs -----


def _run(args: list[str], env: dict[str, str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_cli_advise_refuses_with_no_credentials_but_other_verbs_still_work(
    scrubbed_env: dict[str, str], tiny_wav: Path
) -> None:
    advise_proc = _run(["advise", str(tiny_wav)], env=scrubbed_env)
    assert advise_proc.returncode != 0
    payload = json.loads(advise_proc.stdout)
    assert payload["error"]["code"] == "provider_credential_missing"
    assert any(var in payload["error"]["message"] or var in payload["error"]["remedy"] for var in _PROVIDER_ENV_VARS)

    analyze_proc = _run(["analyze", str(tiny_wav)], env=scrubbed_env)
    assert analyze_proc.returncode == 0, analyze_proc.stderr

    detect_proc = _run(["detect", "silence", str(tiny_wav)], env=scrubbed_env)
    assert detect_proc.returncode == 0, detect_proc.stderr

    plan_proc = _run(["plan"], env=scrubbed_env)
    assert plan_proc.returncode == 0, plan_proc.stderr
    eq_proc = _run(["eq", "--hpf", "40"], env=scrubbed_env, input_text=plan_proc.stdout)
    assert eq_proc.returncode == 0, eq_proc.stderr
    limit_proc = _run(["limit", "--ceiling", "-1.0"], env=scrubbed_env, input_text=eq_proc.stdout)
    assert limit_proc.returncode == 0, limit_proc.stderr


def test_cli_master_refuses_with_no_credentials_and_writes_no_file(
    scrubbed_env: dict[str, str], tiny_wav: Path, tmp_path: Path
) -> None:
    out_path = tmp_path / "out.wav"
    proc = _run(["master", str(tiny_wav), str(out_path)], env=scrubbed_env)
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "provider_credential_missing"
    assert not out_path.exists()


def test_cli_advise_and_master_help_no_longer_say_not_yet_built() -> None:
    advise_help = _run(["advise", "--help"], env=None)
    assert advise_help.returncode == 0, advise_help.stderr
    assert "not yet built" not in advise_help.stdout.lower()
    assert "provider_credential_missing" in advise_help.stdout

    master_help = _run(["master", "--help"], env=None)
    assert master_help.returncode == 0, master_help.stderr
    assert "not yet built" not in master_help.stdout.lower()
