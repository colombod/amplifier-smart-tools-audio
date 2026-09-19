"""Integration tests for `advise`/`master` -- the smart tier.

Every test that needs a model's answer replays a REAL recorded Anthropic
response (`tests/replay.py`'s `ReplayAdviceBackend`, backed by
`tests/fixtures/recorded/anthropic/`) -- never a hand-authored plan. The
one deliberate exception is `_GarbageTextBackend`, used for exactly one
test (bad-model-output rejection): no real *successful* recording can ever
supply invalid JSON, because recordings only capture calls that worked, so
that one adversarial-input case is the one thing a replay structurally
cannot provide. It stands in for OUR OWN error-handling path, not for any
third party's shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aud import lib
from aud.schemas import AudError
from tests import replay

_SR = 44100

_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


class _GarbageTextBackend:
    """Returns literally invalid text -- no real recording can supply this
    (every recording captured a call that worked). Stands in for OUR OWN
    `bad_model_output` error path, not for any provider's real shape.
    """

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        del system, user, model, max_tokens
        return "garbage, not json"


# --- Regression guard: `lufs_ok`/`ceiling_ok` must not be vacuous ----------
#
# Both are booleans that read True whenever the measured value happens to
# land on the right side of the line -- including when the limiter never
# engaged at all. Measured directly (see the module docstring on replays):
# `tiny_wav` run through a bare loudness+limit chain never engages the
# limiter (max_gain_reduction_db == 0.0) -- the loudness stage's own gain
# already lands the peak under the ceiling by luck of this content, not by
# enforcement. Widening the hot spike to 200 samples (still the same
# multiplier as before) is what forces real engagement with the REAL
# recorded "expand, loudness, limit" chain below -- measured directly by
# running that real plan against several spike widths until one produced
# nonzero max_gain_reduction_db while still settling within lib.verify()'s
# tolerances; 200 samples is the smallest that does.
_HOT_SPIKE_SAMPLES = 200
_HOT_SPIKE_MULTIPLIER = 300.0
_CEILING_TOLERANCE_DB = 0.05  # same numerical tolerance limiter.py itself uses for ceiling_met
_LUFS_TOLERANCE_LU = 0.5  # same tolerance lib.verify() itself uses for lufs_ok


def _hot_wav(tmp_path: Path) -> Path:
    """Like conftest's `tiny_wav`, but with a hot enough transient that the
    limiter must engage once run through the real recorded hissy chain
    (expand, loudness, limit; see `replay.ReplayAdviceBackend`).
    """
    seconds = 2.0
    t = np.arange(int(_SR * seconds)) / _SR
    x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 3000 * t)
    rng = np.random.default_rng(0)
    x = x + 0.02 * rng.standard_normal(t.size)
    hot = slice(_SR // 4, _SR // 4 + _HOT_SPIKE_SAMPLES)
    x[hot] *= _HOT_SPIKE_MULTIPLIER
    path = tmp_path / "hot_in.wav"
    sf.write(str(path), np.stack([x, x], axis=1), _SR, subtype="PCM_24")
    return path


# --- lib-level: real recorded responses replayed, real render/verify -------


def test_lib_advise_returns_a_plan_that_pipes_into_render(tiny_wav: Path, tmp_path: Path) -> None:
    """Replays anthropic/advise-clean-haiku.json -- a real (loudness, limit) answer."""
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    outcome = lib.advise(str(tiny_wav), target_lufs=-16.0, backend=backend, model="fake-1")
    assert [s["stage"] for s in outcome["stages"]] == ["loudness", "limit"]
    assert outcome["measurements"]["sample_rate"] == 44100
    assert outcome["provider"] == "injected"
    assert outcome["model"] == "fake-1"

    out_path = tmp_path / "out.wav"
    render_result = lib.render(outcome["plan"], str(tiny_wav), str(out_path))
    assert out_path.exists()
    stage_names = [s["stage"] for s in render_result["report"]["stages"]]
    assert stage_names == ["loudness", "limit"]


def test_lib_master_renders_and_verifies_end_to_end(tmp_path: Path) -> None:
    """Replays anthropic/advise-hissy-sonnet-thinking.json -- a real
    (expand, loudness, limit) answer, chosen for a genuinely noisy/hissy
    measurement report, not hand-authored for this test.
    """
    hot_wav = _hot_wav(tmp_path)
    out_path = tmp_path / "mastered.wav"
    backend = replay.ReplayAdviceBackend("advise-hissy-sonnet-thinking")
    result = lib.master(
        str(hot_wav),
        str(out_path),
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        backend=backend,
        model="fake-1",
    )
    assert out_path.exists()
    assert result["out_path"] == str(out_path)
    assert [s["stage"] for s in result["stages"]] == ["expand", "loudness", "limit"]
    assert result["render"]["stages"][-1]["stage"] == "limit"

    limit_stage = result["render"]["stages"][-1]
    print(f"\n[advise_master] limit stage: {limit_stage}")
    # The canary: without this, `ceiling_ok` below proves nothing -- it would
    # be equally true of a limiter that never touched the signal.
    assert limit_stage["max_gain_reduction_db"] != 0.0, (
        "the hot fixture did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_ok/lufs_ok below are not evidence master() works; fix the fixture, not the assertion"
    )

    measured = result["verify"]["measured"]
    assert abs(measured["integrated_lufs"] - (-16.0)) <= _LUFS_TOLERANCE_LU, result["verify"]
    assert measured["true_peak_dbtp"] <= -1.0 + _CEILING_TOLERANCE_DB, result["verify"]
    # The booleans are still necessary, just no longer sufficient on their own.
    assert result["verify"]["lufs_ok"] is True
    assert result["verify"]["ceiling_ok"] is True
    assert isinstance(result["verify"]["measured"]["integrated_lufs"], float)
    assert isinstance(result["verify"]["measured"]["true_peak_dbtp"], float)
    # The plan travels as a plain dict in master's result -- JSON-serializable, matching the CLI's envelope.
    assert result["plan"]["stages"][0]["stage"] == "expand"
    json.dumps(result)  # must not raise: master's whole result is one JSON document


def test_lib_master_dry_run_does_not_render(tiny_wav: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "should-not-exist.wav"
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    result = lib.master(
        str(tiny_wav),
        str(out_path),
        backend=backend,
        model="fake-1",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert "verify" not in result
    assert "render" not in result
    assert not out_path.exists()


def test_lib_advise_with_reference_measures_both_files(tiny_wav: Path, tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(reference), np.stack([tone, tone], axis=1), sr, subtype="PCM_24")

    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    outcome = lib.advise(
        str(tiny_wav),
        reference_path=str(reference),
        backend=backend,
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
        lib.advise(str(tiny_wav), backend=_GarbageTextBackend(), model="fake-1")
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
