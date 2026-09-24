"""Real media through the library and a credential-free CLI, no mocked DSP.

Synthetic signals are generated here; no personal media or provider calls.
The transient-rich fixture reproduces the reported gain-then-limit shortfall,
not the exact measurements of the user's unavailable original recording.
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
from aud.dsp import engine, io
from aud.plan import new_plan, ordered, write_plan
from aud.schemas import AudError


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in lib._CONFIG_DEFAULTS:
        monkeypatch.delenv(f"AUD_{name.upper()}", raising=False)
    monkeypatch.setenv("AUD_CONFIG", str(tmp_path / "absent.toml"))


@pytest.fixture
def transient_wav(tmp_path):
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = 0.06 * np.sin(2 * np.pi * 220 * t)
    x += 0.7 * np.exp(-(((t % 0.3 - 0.1) / 0.003) ** 2)) * np.sin(2 * np.pi * 1800 * t)
    path = tmp_path / "transients.wav"
    sf.write(path, np.stack([x, x], axis=1), sr, subtype="PCM_24")
    return path


def _targeted_plan():
    return lib.limit(lib.loudness(new_plan(), target_lufs=-16), ceiling_dbtp=-1.5)


def test_final_encoded_shortfall_warns_without_changing_the_audio(transient_wav, tmp_path):
    plan = _targeted_plan()
    original_plan = write_plan(plan)
    result = lib.render(plan, str(transient_wav), str(tmp_path / "out.wav"))
    report = result["report"]
    verification = lib.verify(result["out_path"], target_lufs=-16)
    assert report["verification"] == verification
    assert verification["lufs_ok"] is False
    assert lib.verify(result["out_path"], ceiling_dbtp=-1.5)["ceiling_ok"] is True
    assert lib.analyze(str(transient_wav))["crest_factor_db"] > 18
    assert report["stages"][0]["measured_lufs_after"] == pytest.approx(-16, abs=0.01)
    assert verification["measured"]["integrated_lufs"] < -16.5
    assert len(report["warnings"]) == 1
    warning = report["warnings"][0]
    assert warning["code"] == "loudness_target_missed"
    assert warning["target_lufs"] == -16
    assert warning["measured_lufs"] == verification["measured"]["integrated_lufs"]
    assert warning["difference_lu"] == warning["measured_lufs"] + 16
    assert warning["tolerance_lu"] == 0.5
    assert "If peak limiting" in warning["remedy"]
    assert write_plan(plan) == original_plan

    # The warning must not sneak in gain or compression or another encode.
    x, sr = io.read_audio(transient_wav)
    expected, _ = engine.apply_plan(x, sr, ordered(plan))
    baseline = tmp_path / "baseline.wav"
    io.write_audio(baseline, expected, sr, "PCM_24")
    assert Path(result["out_path"]).read_bytes() == baseline.read_bytes()
    json.dumps(result, allow_nan=False)


def test_within_target_has_real_verification_and_no_warning(tiny_wav, tmp_path):
    plan = lib.loudness(new_plan(), target_lufs=-24)
    result = lib.render(plan, str(tiny_wav), str(tmp_path / "within.wav"))
    assert result["report"]["verification"]["lufs_ok"] is True
    assert result["report"]["warnings"] == []
    assert result["report"]["verification"] == lib.verify(result["out_path"], target_lufs=-24)


def test_no_target_does_not_infer_one_from_settings(tiny_wav, tmp_path, monkeypatch):
    monkeypatch.setenv("AUD_DEFAULT_TARGET_LUFS", "-40")
    result = lib.render(lib.limit(new_plan()), str(tiny_wav), str(tmp_path / "no-target.wav"))
    assert result["report"]["warnings"] == []
    assert "verification" not in result["report"]
    assert "lufs_ok" not in result["report"]


def test_last_loudness_target_wins_without_rewriting_an_old_plan(tiny_wav, tmp_path):
    plan = lib.loudness(lib.loudness(new_plan(), target_lufs=-24), target_lufs=-20)
    plan.created_with = "aud/0.11.1"
    serialized = write_plan(plan)
    result = lib.render(plan, str(tiny_wav), str(tmp_path / "last.wav"))
    assert result["report"]["verification"]["target_lufs"] == -20
    assert result["report"]["verification"]["lufs_ok"] is True
    assert result["report"]["warnings"] == []
    assert write_plan(plan) == serialized


def test_output_format_changes_are_measured_after_all_processing(tiny_wav, tmp_path, monkeypatch):
    # Identical stereo channels folded to mono lose ~3 LU despite the
    # intermediate loudness stage hitting its target. Not a limiter cause.
    monkeypatch.setenv("AUD_SAMPLE_RATE_POLICY", "32000")
    monkeypatch.setenv("AUD_OUTPUT_SUBTYPE", "PCM_16")
    plan = lib.downmix(lib.loudness(new_plan(), target_lufs=-24))
    result = lib.render(plan, str(tiny_wav), str(tmp_path / "mono.wav"))
    report = result["report"]
    assert sf.info(result["out_path"]).samplerate == 32000
    assert sf.info(result["out_path"]).subtype == "PCM_16"
    assert report["stages"][0]["measured_lufs_after"] == pytest.approx(-24, abs=0.01)
    assert report["verification"] == lib.verify(result["out_path"], target_lufs=-24)
    assert report["verification"]["lufs_ok"] is False
    assert report["warnings"][0]["measured_lufs"] < -26.5


def _cli(plan, source, destination, tmp_path):
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", "render", str(source), str(destination)],
        input=write_plan(plan),
        text=True,
        capture_output=True,
        env={"PATH": str(Path(sys.executable).parent), "AUD_CONFIG": str(tmp_path / "absent.toml")},
        timeout=30,
    )


def test_cli_exposes_the_library_warning_with_no_credentials(transient_wav, tmp_path):
    destination = tmp_path / "cli.wav"
    proc = _cli(_targeted_plan(), transient_wav, destination, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    result = json.loads(proc.stdout)["result"]
    assert result["report"]["warnings"][0]["code"] == "loudness_target_missed"
    assert result["report"]["verification"] == lib.verify(str(destination), target_lufs=-16)
    assert destination.is_file()


def test_silence_is_unmeasurable_not_a_success_or_nonfinite_json(tmp_path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros((48000, 2)), 48000, subtype="PCM_24")
    result = lib.render(_targeted_plan(), str(source), str(tmp_path / "silent-out.wav"))
    report = result["report"]
    assert report["verification"]["lufs_ok"] is False
    warning = report["warnings"][0]
    assert warning["code"] == "loudness_unmeasurable"
    assert warning["measured_lufs"] is None
    assert warning["difference_lu"] is None
    assert "compression" not in warning["remedy"]
    json.dumps(result, allow_nan=False)
    proc = _cli(_targeted_plan(), source, tmp_path / "silent-cli.wav", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "Infinity" not in proc.stdout
    assert "NaN" not in proc.stdout
    assert json.loads(proc.stdout)["result"]["report"]["warnings"] == report["warnings"]


def test_write_failure_never_returns_successful_verification(tiny_wav, tmp_path):
    out = tmp_path / "unsupported.invalid-extension"
    with pytest.raises(AudError) as exc:
        lib.render(_targeted_plan(), str(tiny_wav), str(out))
    assert exc.value.code == "audio_write_error"
    proc = _cli(_targeted_plan(), tiny_wav, out, tmp_path)
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert json.loads(proc.stderr)["error"]["code"] == "audio_write_error"


def test_input_failure_never_claims_measurement(tmp_path):
    source = tmp_path / "not-audio.wav"
    source.write_text("not audio", encoding="utf-8")
    proc = _cli(_targeted_plan(), source, tmp_path / "out.wav", tmp_path)
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert json.loads(proc.stderr)["error"]["code"] == "audio_decode_error"
    assert not (tmp_path / "out.wav").exists()


def test_verify_uses_existing_half_lu_tolerance_on_real_audio(tiny_wav):
    measured = lib.verify(str(tiny_wav))["measured"]["integrated_lufs"]
    assert lib.verify(str(tiny_wav), target_lufs=measured + 0.5)["lufs_ok"] is True
    assert lib.verify(str(tiny_wav), target_lufs=measured - 0.5)["lufs_ok"] is True
    assert lib.verify(str(tiny_wav), target_lufs=measured + 0.5001)["lufs_ok"] is False
    assert lib.verify(str(tiny_wav), target_lufs=measured - 0.5001)["lufs_ok"] is False
