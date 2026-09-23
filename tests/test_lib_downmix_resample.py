"""Tests for aud.lib.downmix/resample's plan-document shape and validation,
plus aud.lib.render actually writing files at the rate/channel-count asked
for -- measured from the WRITTEN FILE via soundfile, not computed from the
request.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aud import lib
from aud.plan import new_plan
from aud.schemas import AudError

SR = 44100


def _stereo_wav(tmp_path: Path, seconds: float = 1.0, sr: int = SR, name: str = "in.wav") -> Path:
    t = np.arange(int(seconds * sr)) / sr
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    path = tmp_path / name
    sf.write(str(path), np.stack([left, right], axis=1), sr, subtype="PCM_24")
    return path


# --- plan-builder shape and validation ---------------------------------------


def test_downmix_appends_a_stage_with_no_params() -> None:
    plan = lib.downmix(new_plan())
    stage = plan.stages[-1]
    assert stage.stage == "downmix"
    assert stage.params == {}


def test_resample_appends_a_stage_with_target_hz() -> None:
    plan = lib.resample(new_plan(), target_hz=48000)
    stage = plan.stages[-1]
    assert stage.stage == "resample"
    assert stage.params == {"target_hz": 48000}


@pytest.mark.parametrize("bad_value", [0, -48000, 44100.5, "48000", True])
def test_resample_rejects_a_non_positive_or_non_integer_target(bad_value) -> None:
    with pytest.raises(AudError) as excinfo:
        lib.resample(new_plan(), target_hz=bad_value)
    assert excinfo.value.code == "bad_param"
    assert excinfo.value.remedy


# --- render: rate is what was asked, measured from the WRITTEN file ---------


def test_render_with_explicit_resample_stage_writes_the_requested_rate(tmp_path: Path) -> None:
    in_path = _stereo_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan = lib.resample(new_plan(), target_hz=16000)
    result = lib.render(plan, str(in_path), str(out_path))

    written, written_sr = sf.read(result["out_path"])
    print(f"\n[render] requested 16000 Hz, written file reports sample_rate={written_sr}")
    assert written_sr == 16000
    # Duration within one sample period at the lower rate.
    input_duration = 1.0
    output_duration = written.shape[0] / written_sr
    assert abs(output_duration - input_duration) <= 1.0 / 16000

    resample_reports = [s for s in result["report"]["stages"] if s["stage"] == "resample"]
    assert len(resample_reports) == 1
    assert resample_reports[0]["target_hz"] == 16000
    assert result["report"]["sample_rate"] == 16000


def test_render_with_explicit_downmix_stage_writes_exactly_one_channel(tmp_path: Path) -> None:
    in_path = _stereo_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan = lib.downmix(new_plan())
    result = lib.render(plan, str(in_path), str(out_path))

    written, _written_sr = sf.read(result["out_path"], always_2d=True)
    print(f"\n[render] downmix output shape={written.shape}")
    assert written.shape[1] == 1
    assert float(np.max(np.abs(written))) <= 1.0

    downmix_reports = [s for s in result["report"]["stages"] if s["stage"] == "downmix"]
    assert len(downmix_reports) == 1
    assert downmix_reports[0]["input_channels"] == 2


def test_render_downmix_then_resample_together(tmp_path: Path) -> None:
    in_path = _stereo_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan = new_plan()
    plan = lib.downmix(plan)
    plan = lib.resample(plan, target_hz=8000)
    result = lib.render(plan, str(in_path), str(out_path))

    written, written_sr = sf.read(result["out_path"], always_2d=True)
    assert written_sr == 8000
    assert written.shape[1] == 1


def test_render_of_already_mono_input_with_downmix_stage_is_a_no_op(tmp_path: Path) -> None:
    """The documented no-op case: downmixing an already-mono file succeeds
    and reports input_channels == 1, rather than erroring.
    """
    t = np.arange(SR) / SR
    mono = 0.3 * np.sin(2 * np.pi * 250 * t)
    in_path = tmp_path / "mono_in.wav"
    sf.write(str(in_path), mono, SR, subtype="PCM_24")
    out_path = tmp_path / "out.wav"

    plan = lib.downmix(new_plan())
    result = lib.render(plan, str(in_path), str(out_path))

    written, _sr = sf.read(result["out_path"], always_2d=True)
    assert written.shape[1] == 1
    downmix_report = next(s for s in result["report"]["stages"] if s["stage"] == "downmix")
    assert downmix_report["input_channels"] == 1


def test_render_missing_input_file_is_file_not_found(tmp_path: Path) -> None:
    plan = lib.resample(new_plan(), target_hz=48000)
    with pytest.raises(AudError) as excinfo:
        lib.render(plan, str(tmp_path / "does_not_exist.wav"), str(tmp_path / "out.wav"))
    assert excinfo.value.code == "file_not_found"


# --- sample_rate_policy wiring ------------------------------------------------


def test_render_with_no_resample_stage_applies_the_integer_sample_rate_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUD_SAMPLE_RATE_POLICY", "24000")
    in_path = _stereo_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan = new_plan()  # no explicit resample stage
    result = lib.render(plan, str(in_path), str(out_path))

    _written, written_sr = sf.read(result["out_path"])
    print(f"\n[render] AUD_SAMPLE_RATE_POLICY=24000, written file reports sample_rate={written_sr}")
    assert written_sr == 24000
    assert "sample_rate_policy_applied" in result["report"]
    assert result["report"]["sample_rate_policy_applied"]["target_hz"] == 24000
    assert result["report"]["sample_rate"] == 24000


def test_render_with_preserve_policy_and_no_resample_stage_keeps_input_rate(tmp_path: Path) -> None:
    in_path = _stereo_wav(tmp_path, sr=SR)
    out_path = tmp_path / "out.wav"

    plan = new_plan()
    result = lib.render(plan, str(in_path), str(out_path))

    _written, written_sr = sf.read(result["out_path"])
    assert written_sr == SR
    assert "sample_rate_policy_applied" not in result["report"]


def test_explicit_resample_stage_takes_precedence_over_the_sample_rate_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented precedence rule: an explicit `resample` stage always
    wins over the config-level policy -- render never applies both.
    """
    monkeypatch.setenv("AUD_SAMPLE_RATE_POLICY", "24000")
    in_path = _stereo_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan = lib.resample(new_plan(), target_hz=32000)
    result = lib.render(plan, str(in_path), str(out_path))

    _written, written_sr = sf.read(result["out_path"])
    print(f"\n[render] explicit resample=32000 vs policy=24000 -> written {written_sr}")
    assert written_sr == 32000
    assert "sample_rate_policy_applied" not in result["report"]
    resample_reports = [s for s in result["report"]["stages"] if s["stage"] == "resample"]
    assert len(resample_reports) == 1
    assert resample_reports[0]["target_hz"] == 32000
