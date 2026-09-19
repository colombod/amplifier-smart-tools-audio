"""End-to-end test of the mastering plan dispatcher (engine.apply_plan)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from aud.dsp import engine

SR = 48000


@dataclass
class _Stage:
    """Minimal duck-typed stand-in for a real plan stage: `.stage` + `.params`.

    engine.apply_plan only relies on these two attributes (see its
    Protocol), so it does not need to import the plan/schema types owned
    elsewhere in this package.
    """

    stage: str
    params: dict[str, Any] = field(default_factory=dict)


def _noise(seconds: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    return rng.normal(0.0, 0.1, size=(n, 2))


# --- Regression guard: `ceiling_met` must not be vacuous, and the
# "sane, not garbage" bound must actually be a bound --------------------
#
# Plain `_noise()` (std 0.1) never comes close to a -1.0 dBTP ceiling, so
# `ceiling_met is True` reads True whether or not the limiter did anything
# -- see the agent report and test_dsp_limiter.py's own regression-guard
# comment. Worse, `input_true_peak_dbtp > output_true_peak_dbtp - 20` is
# satisfied by a silent no-op limiter just as easily as a working one (a
# 20 dB-wide band is not a bound on limiting behavior, only a guard
# against genuinely garbage numbers). A brief hot spike added to the noise
# forces real engagement through this exact eq/compress/loudness/limit
# chain, at which point the *measured* gap between input and output true
# peak becomes meaningful evidence rather than a tautology.
_HOT_SPIKE_SAMPLES = 50  # ~1ms @ 48kHz
_HOT_SPIKE_MULTIPLIER = 10.0
# Measured on this exact chain (see agent report): multipliers below ~4
# leave the limiter fully quiet (max_gain_reduction_db == 0.0); 5-20
# reliably engage it (multiple dB of real reduction) with the output
# settling at a stable ~-1.37 dBTP -- comfortably under the ceiling, not
# just within limiter.py's 0.05 dB numerical tolerance. 10.0 sits in the
# middle of that band.


def test_apply_plan_runs_eq_compress_loudness_limit_end_to_end_with_real_numbers():
    x = _noise()
    hot = slice(SR // 4, SR // 4 + _HOT_SPIKE_SAMPLES)
    x[hot] *= _HOT_SPIKE_MULTIPLIER
    stages = [
        _Stage(
            "eq",
            {
                "hpf_hz": 40.0,
                "lpf_hz": None,
                "peaks": [{"freq_hz": 3000.0, "gain_db": -2.0, "q": 1.4}],
                "shelves": [{"type": "high", "freq_hz": 12000.0, "gain_db": 2.0, "q": 0.7}],
            },
        ),
        _Stage(
            "compress",
            {
                "crossovers_hz": [300.0, 3000.0],
                "bands": [
                    {"threshold_db": -20.0, "ratio": 2.0},
                    {"threshold_db": -18.0, "ratio": 2.5},
                    {"threshold_db": -18.0, "ratio": 2.0},
                ],
            },
        ),
        _Stage("loudness", {"target_lufs": -16.0}),
        _Stage("limit", {"ceiling_dbtp": -1.0, "lookahead_ms": 5.0, "release_ms": 100.0}),
    ]

    y, report = engine.apply_plan(x, SR, stages)

    assert y.shape == x.shape
    stage_names = [s["stage"] for s in report["stages"]]
    assert stage_names == ["eq", "compress", "loudness", "limit"]

    eq_report = report["stages"][0]
    print(f"\n[engine] eq: {eq_report}")
    assert eq_report["hpf_hz"] == 40.0
    assert len(eq_report["peaks_applied"]) == 1
    assert len(eq_report["shelves_applied"]) == 1
    assert eq_report["shelves_applied"][0]["type"] == "high"
    assert isinstance(eq_report["sample_peak_dbfs_before"], float)
    assert isinstance(eq_report["sample_peak_dbfs_after"], float)

    compress_report = report["stages"][1]
    print(f"[engine] compress: {compress_report}")
    assert compress_report["crossovers_hz"] == [300.0, 3000.0]
    assert len(compress_report["bands"]) == 3
    for band in compress_report["bands"]:
        assert isinstance(band["max_gain_reduction_db"], float)
        assert isinstance(band["avg_gain_reduction_db"], float)

    loudness_report = report["stages"][2]
    print(f"[engine] loudness: {loudness_report}")
    assert loudness_report["target_lufs"] == -16.0
    assert abs(loudness_report["measured_lufs_after"] - (-16.0)) < 0.5

    limit_report = report["stages"][3]
    print(f"[engine] limit: {limit_report}")
    # The canary: without this, `ceiling_met` and the peak-gap check below
    # prove nothing -- both are equally satisfied by a limiter that never
    # touched the signal.
    assert limit_report["max_gain_reduction_db"] != 0.0, (
        "the hot fixture did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_met/the peak-gap check below are not evidence the limiter works; fix the fixture, not the assertion"
    )
    assert limit_report["ceiling_met"] is True
    assert limit_report["output_true_peak_dbtp"] <= -1.0 + 0.1
    # A real bound now that the limiter is known to be engaged: the input
    # peak must sit measurably (not just "not garbage") above the output.
    assert limit_report["input_true_peak_dbtp"] > limit_report["output_true_peak_dbtp"] + 1.0


def test_apply_plan_raises_a_clear_error_for_an_unimplemented_stage():
    # A fabricated name, not any real stage: every stage contracts/plan.v1.md
    # names (including deess/dereverb/eq_match/reverb/stretch/pitch) is
    # implemented in this build's registry (see engine.py's module
    # docstring), so this guards the dispatcher's honest-failure behavior
    # itself rather than pinning it to one stage's implementation status.
    x = _noise(seconds=0.1)
    stages = [_Stage("not_a_real_stage", {"amount": 6})]
    with pytest.raises(ValueError, match="not yet implemented"):
        engine.apply_plan(x, SR, stages)


def test_apply_plan_with_empty_stage_list_is_a_no_op():
    x = _noise(seconds=0.1)
    y, report = engine.apply_plan(x, SR, [])
    assert np.array_equal(y, x)
    assert report["stages"] == []
