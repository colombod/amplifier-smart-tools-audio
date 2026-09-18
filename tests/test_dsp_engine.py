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


def test_apply_plan_runs_eq_compress_loudness_limit_end_to_end_with_real_numbers():
    x = _noise()
    stages = [
        _Stage(
            "eq",
            {
                "hpf": 40.0,
                "lpf": None,
                "peaks": [[3000.0, -2.0, 1.4]],
                "low_shelf": None,
                "high_shelf": [12000.0, 2.0, 0.7],
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
    assert limit_report["ceiling_met"] is True
    assert limit_report["output_true_peak_dbtp"] <= -1.0 + 0.1
    assert limit_report["input_true_peak_dbtp"] > limit_report["output_true_peak_dbtp"] - 20  # sane, not garbage


def test_apply_plan_raises_a_clear_error_for_an_unimplemented_stage():
    x = _noise(seconds=0.1)
    stages = [_Stage("deess", {"amount": 6})]
    with pytest.raises(ValueError, match="deess"):
        engine.apply_plan(x, SR, stages)


def test_apply_plan_with_empty_stage_list_is_a_no_op():
    x = _noise(seconds=0.1)
    y, report = engine.apply_plan(x, SR, [])
    assert np.array_equal(y, x)
    assert report["stages"] == []
