"""Apply an ordered mastering plan to a signal in one pass.

`apply_plan` dispatches each stage in the (already ordered) plan to its DSP
function via a registry, and builds a report that is specific enough to be
useful after a render: gain applied, gain reduction per band, true peak
before and after, etc. -- not just "eq: done".

Only five stages are implemented in this module: eq, compress, saturate,
loudness, limit. Three further stages (eq-match, de-ess, de-reverb, reverb,
time-stretch/pitch-shift) are a separate, later milestone; `apply_plan`
does not special-case their names -- ANY stage not in the registry raises
`NotImplementedStageError` (a `ValueError` subclass), so the caller always
gets an honest "not implemented yet" rather than a silent no-op or a
confusing KeyError. aud.lib maps that exception to a `not_implemented`
AudError; it is not an internal aud bug.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from aud.dsp import dynamics, limiter, loudness, saturation
from aud.dsp import filters as _filters
from aud.schemas import NotImplementedStageError

__all__ = ["apply_plan"]

_EPS = 1e-12


class _Stage(Protocol):
    stage: str
    params: dict[str, Any]


def _peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * math.log10(max(peak, _EPS))


def _apply_eq(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    before = _peak_dbfs(x)
    y = x
    peaks_applied = []

    hpf = params.get("hpf")
    if hpf:
        y = _filters.apply_sos(y, _filters.highpass(sr, hpf))

    lpf = params.get("lpf")
    if lpf:
        y = _filters.apply_sos(y, _filters.lowpass(sr, lpf))

    for f, gain_db, q in params.get("peaks") or []:
        y = _filters.apply_sos(y, _filters.peaking(sr, f, gain_db, q))
        peaks_applied.append({"f": f, "gain_db": gain_db, "q": q})

    low_shelf = params.get("low_shelf")
    if low_shelf:
        f, gain_db, q = low_shelf
        y = _filters.apply_sos(y, _filters.low_shelf(sr, f, gain_db, q))

    high_shelf = params.get("high_shelf")
    if high_shelf:
        f, gain_db, q = high_shelf
        y = _filters.apply_sos(y, _filters.high_shelf(sr, f, gain_db, q))

    return y, {
        "hpf_hz": hpf,
        "lpf_hz": lpf,
        "peaks_applied": peaks_applied,
        "low_shelf_applied": low_shelf,
        "high_shelf_applied": high_shelf,
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


def _apply_compress(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    before = _peak_dbfs(x)
    # Field names match contracts/plan.v1.md's "compress" stage exactly:
    # crossover frequencies under crossovers_hz, per-band settings under
    # bands (one entry per len(crossovers_hz) + 1 band).
    crossovers_hz = list(params["crossovers_hz"])
    bands = [dynamics.BandParams(**band) for band in params["bands"]]

    y, stats = dynamics.multiband_compress(x, sr, crossovers_hz, bands)

    band_reports = []
    for i, (band_params, band_stats) in enumerate(zip(bands, stats["bands"], strict=True)):
        band_reports.append(
            {
                "index": i,
                "threshold_db": band_params.threshold_db,
                "ratio": band_params.ratio,
                "max_gain_reduction_db": band_stats["max_gain_reduction_db"],
                "avg_gain_reduction_db": band_stats["avg_gain_reduction_db"],
            }
        )

    return y, {
        "crossovers_hz": crossovers_hz,
        "bands": band_reports,
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


def _apply_saturate(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    del sr  # saturation is sample-rate-agnostic at this API boundary
    before = _peak_dbfs(x)
    drive = params.get("drive", 1.0)
    mode = params.get("mode", "soft")
    mix = params.get("mix", 1.0)

    y = saturation.saturate(x, drive=drive, mode=mode, mix=mix)

    return y, {
        "drive": drive,
        "mode": mode,
        "mix": mix,
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


def _apply_loudness(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    target_lufs = params["target_lufs"]
    before_lufs = loudness.integrated_lufs(x, sr)

    y, applied_gain_db = loudness.normalize(x, sr, target_lufs)

    return y, {
        "target_lufs": target_lufs,
        "measured_lufs_before": before_lufs,
        "measured_lufs_after": loudness.integrated_lufs(y, sr),
        "applied_gain_db": applied_gain_db,
    }


def _apply_limit(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    ceiling_dbtp = params.get("ceiling_dbtp", -1.0)
    lookahead_ms = params.get("lookahead_ms", 5.0)
    release_ms = params.get("release_ms", 100.0)

    y, stats = limiter.brickwall(
        x,
        sr,
        ceiling_dbtp=ceiling_dbtp,
        lookahead_ms=lookahead_ms,
        release_ms=release_ms,
    )
    return y, stats


_REGISTRY = {
    "eq": _apply_eq,
    "compress": _apply_compress,
    "saturate": _apply_saturate,
    "loudness": _apply_loudness,
    "limit": _apply_limit,
}


def apply_plan(x: np.ndarray, sr: int, stages: list[_Stage]) -> tuple[np.ndarray, dict]:
    """Apply an ordered list of mastering stages in one pass.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        stages: Stages already in canonical mastering order, each an
            object exposing `.stage` (str) and `.params` (dict).

    Returns:
        (y, report) where report = {"stages": [per-stage report dict, ...]}.
        Each per-stage dict always includes "stage" (the stage name) plus
        whatever specifics that stage's handler records.

    Raises:
        ValueError: A stage name is not in the implemented registry. The
            error names the stage explicitly rather than failing silently
            or leaving it as a no-op.
    """
    y = np.asarray(x, dtype=np.float64)
    report: dict[str, Any] = {"stages": []}

    for stage in stages:
        name = stage.stage
        handler = _REGISTRY.get(name)
        if handler is None:
            raise NotImplementedStageError(name, tuple(_REGISTRY))
        params = stage.params or {}
        y, stage_report = handler(y, sr, params)
        stage_report = {"stage": name, **stage_report}
        report["stages"].append(stage_report)

    return y, report
