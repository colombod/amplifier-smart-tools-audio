"""Apply an ordered mastering plan to a signal in one pass.

`apply_plan` dispatches each stage in the (already ordered) plan to its DSP
function via a registry, and builds a report that is specific enough to be
useful after a render: gain applied, gain reduction per band, true peak
before and after, etc. -- not just "eq: done".

Seven stages are implemented in this module: cut, strip_silence, eq,
compress, saturate, loudness, limit. Six further stages (stretch, pitch,
dereverb, deess, eq_match, reverb) are a separate, later milestone;
`apply_plan` does not special-case their names -- ANY stage not in the
registry raises `NotImplementedStageError` (a `ValueError` subclass), so
the caller always gets an honest "not implemented yet" rather than a
silent no-op or a confusing KeyError. aud.lib maps that exception to a
`not_implemented` AudError; it is not an internal aud bug.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from aud.dsp import dynamics, limiter, loudness, saturation
from aud.dsp import edit as _edit
from aud.dsp import filters as _filters
from aud.dsp import resolve as _resolve
from aud.schemas import NotImplementedStageError

__all__ = ["MissingDspModuleError", "apply_plan"]

_EPS = 1e-12


class MissingDspModuleError(ValueError):
    """A stage handler needed a `dsp` submodule that has not landed in this build yet.

    `strip_silence` detects at render time (`aud.dsp.detect.detect_silence`);
    `snap="transient"` needs onsets (`aud.dsp.detect.detect_transients`).
    Neither is built in every checkout at every point in time, so a handler
    that needs one imports it lazily and raises this -- never a bare
    `ImportError`, never a silent no-op -- if it is absent. `aud.lib.render`
    catches this and maps it to `AudError(code="not_implemented")`, the same
    boundary pattern `NotImplementedStageError` uses.
    """

    def __init__(self, stage: str, needs: str, exc: Exception) -> None:
        self.stage = stage
        self.needs = needs
        super().__init__(f"Stage '{stage}' needs '{needs}', which is not available in this build: {exc}")


def _onsets_for_snap(x: np.ndarray, sr: int, snap: str, stage: str) -> list[float] | None:
    """Onset positions for `snap="transient"`; None for every other snap mode.

    Lazy import, by the same discipline as `aud.lib`'s sample-touching
    functions (see aud/lib.py's module docstring): a missing `dsp.detect`
    only breaks the one snap mode that needs it, not every render.
    """
    if snap != "transient":
        return None
    try:
        from aud.dsp.detect import detect_transients
    except ImportError as exc:
        raise MissingDspModuleError(stage, "aud.dsp.detect.detect_transients", exc) from exc
    transients = detect_transients(x, sr)
    regions = getattr(transients, "regions", transients)  # tolerate a bare list or a RegionsDoc-shaped result
    onsets: list[float] = []
    for region in regions:
        start_s = region["start_s"] if isinstance(region, dict) else region.start_s
        onsets.append(float(start_s))
    return onsets


def _apply_cut(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    regions = params["regions"]
    snap = params.get("snap", "zero_crossing")
    onsets = _onsets_for_snap(x, sr, snap, "cut")

    edit_points, dropped = _resolve.resolve_points(
        x,
        sr,
        regions,
        pad_out_ms=params.get("pad_out_ms", 0.0),
        pad_in_ms=params.get("pad_in_ms", 0.0),
        snap=snap,
        snap_window_ms=params.get("snap_window_ms", 20.0),
        onsets=onsets,
    )
    return _edit.cut_regions(
        x,
        sr,
        edit_points,
        fade_in_ms=params.get("fade_in_ms", 0.0),
        fade_out_ms=params.get("fade_out_ms", 0.0),
        crossfade_ms=params.get("crossfade_ms", 10.0),
        crossfade_shape=params.get("crossfade_shape", "equal_power"),
        regions_dropped_by_padding=len(dropped),
    )


def _regions_with_keep(silence_regions: Any, keep_ms: float) -> list[dict[str, float]]:
    """Shrink each detected silence span so `keep_ms` of it survives, symmetrically.

    `strip_silence` carries a policy (contracts/plan.v1.md), not positions:
    `keep_ms` is "how much silence is left behind in place of each removed
    one", so the removal handed to `resolve_points` is the detected span
    shrunk by `keep_ms` before padding/snap ever see it. A span keep_ms
    cannot shrink to a positive length needs no removal at all.
    """
    half_keep_s = (keep_ms / 1000.0) / 2.0
    shrunk: list[dict[str, float]] = []
    for region in silence_regions:
        start_s = (region["start_s"] if isinstance(region, dict) else region.start_s) + half_keep_s
        end_s = (region["end_s"] if isinstance(region, dict) else region.end_s) - half_keep_s
        if end_s > start_s:
            shrunk.append({"start_s": float(start_s), "end_s": float(end_s)})
    return shrunk


def _apply_strip_silence(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    try:
        from aud.dsp.detect import detect_silence
    except ImportError as exc:
        raise MissingDspModuleError("strip_silence", "aud.dsp.detect.detect_silence", exc) from exc

    silence_doc = detect_silence(
        x,
        sr,
        threshold_above_floor_db=params.get("threshold_above_floor_db", 6.0),
        min_len_ms=params.get("min_len_ms", 400.0),
    )
    silence_regions = getattr(silence_doc, "regions", silence_doc)
    regions = _regions_with_keep(silence_regions, params.get("keep_ms", 150.0))

    snap = params.get("snap", "zero_crossing")
    onsets = _onsets_for_snap(x, sr, snap, "strip_silence")

    edit_points, dropped = _resolve.resolve_points(
        x,
        sr,
        regions,
        pad_out_ms=params.get("pad_out_ms", 80.0),
        pad_in_ms=params.get("pad_in_ms", 80.0),
        snap=snap,
        snap_window_ms=params.get("snap_window_ms", 20.0),
        onsets=onsets,
    )
    return _edit.cut_regions(
        x,
        sr,
        edit_points,
        fade_in_ms=params.get("fade_in_ms", 0.0),
        fade_out_ms=params.get("fade_out_ms", 0.0),
        crossfade_ms=params.get("crossfade_ms", 10.0),
        crossfade_shape=params.get("crossfade_shape", "equal_power"),
        regions_dropped_by_padding=len(dropped),
    )


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
    "cut": _apply_cut,
    "strip_silence": _apply_strip_silence,
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
