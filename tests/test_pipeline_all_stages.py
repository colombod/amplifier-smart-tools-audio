"""End-to-end regression test for DEFECT 2: every implemented stage chained
together and actually rendered, not just eyeballed.

This is the literal repro from the bug report: plan | eq | compress |
saturate | loudness | limit | render, then verify. It must not crash, and
verify must report the render actually near the requested loudness and
under the requested ceiling.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100

# --- Regression guard: `lufs_ok`/`ceiling_ok` must not be vacuous ----------
#
# Both are booleans that read True whenever the measured value happens to
# land on the right side of the line -- including when the limiter never
# engaged at all and the upstream chain (eq/compress/saturate/loudness) just
# happened to land under the ceiling on this host. See the agent report and
# tests/test_dsp_limiter.py's own regression-guard comment: two renders of
# this exact chain on two different hosts produced true peaks ~1 dB apart
# while both reported the render as "fine", because in both cases the
# limiter's own `max_gain_reduction_db` was 0.0 -- the ceiling was met by
# accident, not by enforcement. `conftest.py`'s shared `tiny_wav` fixture is
# not hot enough to force engagement through this whole chain (measured:
# max_gain_reduction_db == 0.0 with it), so this test builds its own,
# hotter, fixture rather than changing the shared one other tests rely on.

_HOT_SPIKE_SAMPLES = 50  # ~1.1 ms @ 44.1kHz
_HOT_SPIKE_MULTIPLIER = 300.0
# A short (~1ms), very hot spike rather than a broad loud passage: LUFS
# integrates energy over ~400ms gated blocks, so a spike this brief barely
# moves the integrated loudness (measured drift from an un-spiked render:
# well under 0.15 LU) while still handing the limiter, several stages later,
# a true peak well above the -1.0 dBTP ceiling. Measured on this exact chain
# (see agent report): a broad loud region wide enough to force engagement
# also drags LUFS past lib.verify's 0.5 LU tolerance or leaves the true peak
# within a few hundredths of a dB of the ceiling (where lib.verify's
# zero-tolerance `ceiling_ok` flips unpredictably); this brief-spike shape
# decouples the two, leaving comfortable, repeatable margin on both:
# max_gain_reduction_db ~= -3.3 dB (real engagement), integrated LUFS within
# ~0.11 LU of target (0.5 LU budget), true peak ~0.24 dB under the ceiling
# (0.05 dB numerical-tolerance budget) -- all bit-exact across repeated runs.
_CEILING_TOLERANCE_DB = 0.05  # same numerical tolerance limiter.py itself uses for ceiling_met
_LUFS_TOLERANCE_LU = 0.5  # same tolerance lib.verify() itself uses for lufs_ok


def _hot_wav(tmp_path: Path) -> Path:
    """Like conftest's `tiny_wav`, but with a hot enough transient that the
    limiter must engage by the time it reaches the end of this chain.
    """
    seconds = 2.0
    t = np.arange(int(SR * seconds)) / SR
    x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 3000 * t)
    rng = np.random.default_rng(0)
    x = x + 0.02 * rng.standard_normal(t.size)
    hot = slice(SR // 4, SR // 4 + _HOT_SPIKE_SAMPLES)
    x[hot] *= _HOT_SPIKE_MULTIPLIER
    path = tmp_path / "hot_in.wav"
    sf.write(str(path), np.stack([x, x], axis=1), SR, subtype="PCM_24")
    return path


def _run(args: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_all_implemented_stages_chain_and_render_without_crashing(tmp_path: Path) -> None:
    in_path = _hot_wav(tmp_path)
    out_path = tmp_path / "out.wav"

    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr

    eq_proc = _run(["eq", "--hpf", "40", "--peak", "3000,-3,1.2"], input_text=plan_proc.stdout)
    assert eq_proc.returncode == 0, eq_proc.stderr

    compress_proc = _run(["compress", "--bands", "120,900,5500", "--ratio", "3"], input_text=eq_proc.stdout)
    assert compress_proc.returncode == 0, compress_proc.stderr

    saturate_proc = _run(["saturate", "--drive", "1.5", "--mix", "0.25"], input_text=compress_proc.stdout)
    assert saturate_proc.returncode == 0, saturate_proc.stderr

    loudness_proc = _run(["loudness", "--target", "-14"], input_text=saturate_proc.stdout)
    assert loudness_proc.returncode == 0, loudness_proc.stderr

    limit_proc = _run(["limit", "--ceiling", "-1.0"], input_text=loudness_proc.stdout)
    assert limit_proc.returncode == 0, limit_proc.stderr

    render_proc = _run(["render", str(in_path), str(out_path)], input_text=limit_proc.stdout)
    assert render_proc.returncode == 0, render_proc.stderr
    render_result = json.loads(render_proc.stdout)["result"]
    assert render_result["out_path"] == str(out_path)
    stage_names = [s["stage"] for s in render_result["report"]["stages"]]
    assert stage_names == ["eq", "compress", "saturate", "loudness", "limit"]
    assert out_path.exists()

    limit_stage = next(s for s in render_result["report"]["stages"] if s["stage"] == "limit")
    print(f"\n[pipeline] limit stage: {limit_stage}")
    # The canary: without this, `ceiling_ok` below proves nothing -- it would
    # be equally true of a limiter that never touched the signal.
    assert limit_stage["max_gain_reduction_db"] != 0.0, (
        "the hot fixture did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_ok/lufs_ok below are not evidence the chain works; fix the fixture, not the assertion"
    )

    verify_proc = _run(["verify", str(out_path), "--target", "-14", "--ceiling", "-1.0"])
    assert verify_proc.returncode == 0, verify_proc.stderr
    verify_result = json.loads(verify_proc.stdout)["result"]

    print(f"\n[pipeline] verify: {verify_result}")
    measured = verify_result["measured"]
    assert abs(measured["integrated_lufs"] - (-14.0)) <= _LUFS_TOLERANCE_LU, verify_result
    assert measured["true_peak_dbtp"] <= -1.0 + _CEILING_TOLERANCE_DB, verify_result
    # The booleans are still necessary, just no longer sufficient on their own.
    assert verify_result["lufs_ok"] is True, verify_result
    assert verify_result["ceiling_ok"] is True, verify_result
