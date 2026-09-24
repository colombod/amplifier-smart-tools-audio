"""Apply an ordered mastering plan to a signal in one pass.

`apply_plan` dispatches each stage in the (already ordered) plan to its DSP
function via a registry, and builds a report that is specific enough to be
useful after a render: gain applied, gain reduction per band, true peak
before and after, etc. -- not just "eq: done".

Seventeen stages are implemented in this module: cut, strip_silence, gate,
expand, dereverb, deess, eq, eq_match, compress, saturate, reverb, stretch,
pitch, loudness, limit, downmix, resample -- every stage the canonical
order (contracts/plan.v1.md) names. `apply_plan` does not special-case
stage names -- ANY stage not in the registry raises
`NotImplementedStageError` (a `ValueError` subclass), so the caller always
gets an honest "not implemented yet" rather than a silent no-op or a
confusing KeyError. aud.lib maps that exception to a `not_implemented`
AudError; it is not an internal aud bug.

Sample-rate tracking: every stage handler has signature `(x, sr, params) ->
(y, report)` -- `sr` in, no changed `sr` out, because fifteen of the
seventeen stages never change the rate. `resample` is the one exception,
and its new rate has nowhere to go through that signature. Rather than
widen the executor contract for all seventeen stages (a change to every
existing handler, to serve the one stage that needs it), `apply_plan`'s own
loop special-cases the single stage name that changes `sr` -- see the loop
below -- and publishes the final rate in the returned report's top-level
`sample_rate` key, which `aud.lib.render` reads to decide what to hand
`_write_audio`. Channel count needs no equivalent tracking: it is a
property of the array `y` itself, and every stage already receives
whatever `y` the previous stage returned.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from aud.dsp import channels as _channels
from aud.dsp import deess as _deess
from aud.dsp import dereverb as _dereverb
from aud.dsp import dynamics, eqmatch, limiter, loudness, reverb, saturation, timepitch
from aud.dsp import edit as _edit
from aud.dsp import filters as _filters
from aud.dsp import gate as _gate
from aud.dsp import resample as _resample_module
from aud.dsp import resolve as _resolve
from aud.dsp.edit import CrossfadeExceedsGapError
from aud.dsp.eqmatch import CurveError
from aud.schemas import NotImplementedStageError

__all__ = ["MissingDspModuleError", "StageParamError", "apply_plan"]

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


class StageParamError(ValueError):
    """A stage handler rejected one of its own params as out of range or the wrong shape.

    A plan's `params` (`aud.plan.Stage.params`) is an unvalidated dict --
    format 1 never pinned per-stage param schemas at the `Plan` model level,
    so a hand-authored plan can carry a param a real caller (the CLI's own
    per-verb builders in `aud.lib`) would never construct. `apply_plan`'s
    dispatch loop is the render-time backstop: any bare `ValueError` a stage
    handler raises for its own params -- not one of the other named boundary
    exceptions this module and `aud.dsp.edit`/`aud.dsp.eqmatch` already carry
    forward unchanged -- is caught here and re-raised as this type, with the
    stage name attached. `aud.lib.render` maps it to `AudError(code=
    "bad_param")` instead of letting the CLI's catch-all report a caller's
    typo as an internal aud bug (see AGENTS.md #4).

    Subclasses ValueError, not AudError, for the same reason
    `MissingDspModuleError`/`NotImplementedStageError` do: this is still
    inside `dsp/`'s call chain (see AGENTS.md #8), so `AudError` construction
    stays a job for `aud.lib`, above the boundary. The original message is
    kept verbatim -- every `dsp/` validation raise already names the field,
    the value found, and the constraint (see e.g. `aud.dsp.filters` and
    `aud.dsp.gate`); only the stage name was missing, and that is what this
    type adds.
    """

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"'{stage}': {detail}")


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


# The plan contract (contracts/plan.v1.md's "deess" stage) exposes a single
# centre frequency, freq_hz -- not the two band edges dsp.deess.deess takes.
# A one-octave-wide band centred on freq_hz (edges at freq/sqrt(2) and
# freq*sqrt(2)) is a standard, defensible way to turn a centre frequency
# into a band: it is symmetric in log-frequency (perceptually the natural
# scale) and matches the width dsp/analysis.py already uses for its own
# octave-band energy report.
_DEESS_BAND_OCTAVE_RATIO = 2.0**0.5


def _apply_deess(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    amount_db = params.get("amount_db", 6.0)
    freq_hz = params.get("freq_hz", 6500.0)
    nyquist = sr / 2.0

    low = freq_hz / _DEESS_BAND_OCTAVE_RATIO
    high = min(freq_hz * _DEESS_BAND_OCTAVE_RATIO, 0.999 * nyquist)

    y, dsp_stats = _deess.deess(x, sr, amount_db=amount_db, band=(low, high))

    return y, {
        "amount_db": amount_db,
        "freq_hz": freq_hz,
        "band_low_hz": dsp_stats["band_low_hz"],
        "band_high_hz": dsp_stats["band_high_hz"],
        "max_gain_reduction_db": dsp_stats["max_gain_reduction_db"],
        "avg_gain_reduction_db": dsp_stats["avg_gain_reduction_db"],
        "frames_reduced_pct": dsp_stats["frames_reduced_pct"],
    }


def _apply_gate(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    y, stats = _gate.gate(
        x,
        sr,
        threshold_above_floor_db=params.get("threshold_above_floor_db", 12.0),
        threshold_db=params.get("threshold_db"),
        range_db=params.get("range_db", 20.0),
        attack_ms=params.get("attack_ms", 2.0),
        hold_ms=params.get("hold_ms", 50.0),
        release_ms=params.get("release_ms", 150.0),
        crossovers_hz=params.get("crossovers_hz") or None,
        lookahead_ms=params.get("lookahead_ms", 3.0),
        sidechain_hpf_hz=params.get("sidechain_hpf_hz", 80.0),
    )
    return y, stats


def _apply_expand(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    y, stats = _gate.expand(
        x,
        sr,
        threshold_above_floor_db=params.get("threshold_above_floor_db", 6.0),
        threshold_db=params.get("threshold_db"),
        ratio=params.get("ratio", 2.0),
        knee_db=params.get("knee_db", 6.0),
        attack_ms=params.get("attack_ms", 5.0),
        hold_ms=params.get("hold_ms", 50.0),
        release_ms=params.get("release_ms", 150.0),
        crossovers_hz=params.get("crossovers_hz") or None,
        lookahead_ms=params.get("lookahead_ms", 3.0),
        sidechain_hpf_hz=params.get("sidechain_hpf_hz", 80.0),
    )
    return y, stats


def _apply_dereverb(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    amount_db = params.get("amount_db", 6.0)
    y, dsp_stats = _dereverb.dereverb(x, sr, amount_db=amount_db)

    return y, {
        "amount_db": amount_db,
        "tau_ms": dsp_stats["tau_ms"],
        "guard_ms": dsp_stats["guard_ms"],
        "max_gain_reduction_db": dsp_stats["max_gain_reduction_db"],
        "avg_gain_reduction_db": dsp_stats["avg_gain_reduction_db"],
        "bins_reduced_pct": dsp_stats["bins_reduced_pct"],
    }


def _apply_eq(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    # Field names match contracts/plan.v1.md's "eq" stage exactly: hpf_hz/
    # lpf_hz for the two corners, peaks/shelves as arrays of objects
    # ({"freq_hz", "gain_db", "q"} and {"type", "freq_hz", "gain_db", "q"}
    # respectively) -- not the raw triples/singular low_shelf/high_shelf
    # keys this handler used to read, which no hand-written,
    # contract-conformant plan could ever produce.
    before = _peak_dbfs(x)
    y = x
    peaks_applied = []

    hpf_hz = params.get("hpf_hz")
    if hpf_hz:
        y = _filters.apply_sos(y, _filters.highpass(sr, hpf_hz))

    lpf_hz = params.get("lpf_hz")
    if lpf_hz:
        y = _filters.apply_sos(y, _filters.lowpass(sr, lpf_hz))

    for peak in params.get("peaks") or []:
        f, gain_db, q = peak["freq_hz"], peak["gain_db"], peak["q"]
        y = _filters.apply_sos(y, _filters.peaking(sr, f, gain_db, q))
        peaks_applied.append({"freq_hz": f, "gain_db": gain_db, "q": q})

    shelves_applied = []
    for shelf in params.get("shelves") or []:
        shelf_type, f, gain_db, q = shelf["type"], shelf["freq_hz"], shelf["gain_db"], shelf["q"]
        if shelf_type == "low":
            y = _filters.apply_sos(y, _filters.low_shelf(sr, f, gain_db, q))
        else:
            y = _filters.apply_sos(y, _filters.high_shelf(sr, f, gain_db, q))
        shelves_applied.append({"type": shelf_type, "freq_hz": f, "gain_db": gain_db, "q": q})

    return y, {
        "hpf_hz": hpf_hz,
        "lpf_hz": lpf_hz,
        "peaks_applied": peaks_applied,
        "shelves_applied": shelves_applied,
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


def _apply_eq_match(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    # Field names match contracts/plan.v1.md's "eq_match" stage exactly:
    # `curve` is an array of [freq_hz, gain_db] pairs (measurements, never
    # a file path -- see dsp/eqmatch.py's module docstring), `amount` is
    # the 0.0-1.0 dial, `max_gain_db` is the single, symmetric clamp the
    # contract documents. `dsp.eqmatch.apply_curve`'s own API is more
    # general (separate max_boost_db/max_cut_db) for testability; the plan
    # contract exposes only the symmetric case, so both are set from it.
    before = _peak_dbfs(x)
    curve = params["curve"]
    amount = params.get("amount", 1.0)
    max_gain_db = params.get("max_gain_db", 12.0)

    y = eqmatch.apply_curve(
        x,
        sr,
        {"curve": curve},
        strength=amount,
        max_boost_db=max_gain_db,
        max_cut_db=max_gain_db,
    )

    return y, {
        "amount": amount,
        "max_gain_db": max_gain_db,
        "curve_points": len(curve),
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


# Per-field defaults for a "compress" band object, matching
# contracts/plan.v1.md's documented band defaults exactly. `dynamics.
# BandParams`'s own dataclass defaults (threshold_db=-18.0, attack_ms=10.0,
# release_ms=120.0, ...) are tuned for that module's own tests, not this
# contract -- so a hand-written plan band that omits a field must not
# silently pick up dynamics.py's defaults instead of the ones
# contracts/plan.v1.md promises. Merged under any band dict before
# `BandParams` is constructed; a band dict's own explicit values win.
_COMPRESS_BAND_DEFAULTS: dict[str, float] = {
    "threshold_db": -24.0,
    "ratio": 2.0,
    "attack_ms": 20.0,
    "release_ms": 180.0,
    "knee_db": 6.0,
    "makeup_db": 0.0,
}


def _apply_compress(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    before = _peak_dbfs(x)
    # Field names match contracts/plan.v1.md's "compress" stage exactly:
    # crossover frequencies under crossovers_hz, per-band settings under
    # bands (one entry per len(crossovers_hz) + 1 band).
    crossovers_hz = list(params["crossovers_hz"])
    bands = [dynamics.BandParams(**{**_COMPRESS_BAND_DEFAULTS, **band}) for band in params["bands"]]

    if crossovers_hz:
        y, stats = dynamics.multiband_compress(x, sr, crossovers_hz, bands)
    else:
        # contracts/plan.v1.md: "N crossovers produce N + 1 bands; [] is
        # single-band" -- a hand-written plan may legitimately render with
        # zero crossovers. crossover.split() itself refuses an empty list
        # (it always needs at least one frequency to split at), so the
        # genuinely single-band case is compressed directly rather than
        # routed through the splitter/recombiner.
        band_y, band_stats = dynamics.compress(x, sr, bands[0])
        y, stats = band_y, {"bands": [band_stats]}

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


# The plan contract (contracts/plan.v1.md's "reverb" stage) exposes only
# mix/decay_s/predelay_ms -- no room_size or damping knob. room_size is
# derived from decay_s (see dsp/reverb.py's module docstring for why
# room_size, not gain, is the FDN's decay-time control); damping is a
# fixed, mastering-appropriate constant, not a caller-facing dial.
_REVERB_MAX_DECAY_S = 4.0
_REVERB_DAMPING = 0.35


def _apply_reverb(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    before = _peak_dbfs(x)
    mix = params.get("mix", 0.15)
    decay_s = params.get("decay_s", 1.2)
    predelay_ms = params.get("predelay_ms", 0.0)

    room_size = float(np.clip(decay_s / _REVERB_MAX_DECAY_S, 0.02, 0.98))
    y, dsp_stats = reverb.reverb(
        x,
        sr,
        room_size=room_size,
        damping=_REVERB_DAMPING,
        wet=mix,
        dry=1.0 - mix,
        pre_delay_ms=predelay_ms,
    )

    return y, {
        "mix": mix,
        "decay_s": decay_s,
        "predelay_ms": predelay_ms,
        "room_size_derived": room_size,
        "estimated_decay_s": dsp_stats["estimated_decay_s"],
        "sample_peak_dbfs_before": before,
        "sample_peak_dbfs_after": _peak_dbfs(y),
    }


def _apply_stretch(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    ratio = params.get("ratio", 1.0)
    y, dsp_stats = timepitch.time_stretch(x, sr, ratio, quality="auto")
    return y, {
        "ratio": ratio,
        "engine": dsp_stats["engine"],
        "input_samples": dsp_stats["input_samples"],
        "output_samples": dsp_stats["output_samples"],
    }


def _apply_pitch(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    semitones = params.get("semitones", 0.0)
    y, dsp_stats = timepitch.pitch_shift(x, sr, semitones, quality="auto")
    return y, {
        "semitones": semitones,
        "frequency_ratio": dsp_stats["frequency_ratio"],
        "engine": dsp_stats["engine"],
    }


def _apply_loudness(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    target_lufs = params["target_lufs"]
    before_lufs = loudness.integrated_lufs(x, sr)

    y, applied_gain_db = loudness.normalize(x, sr, target_lufs)
    after_lufs = loudness.integrated_lufs(y, sr)

    return y, {
        "target_lufs": target_lufs,
        # Silence is unmeasurable, not JSON's non-standard -Infinity.
        "measured_lufs_before": before_lufs if math.isfinite(before_lufs) else None,
        "measured_lufs_after": after_lufs if math.isfinite(after_lufs) else None,
        "applied_gain_db": applied_gain_db,
    }


def _apply_limit(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    # Defaults match contracts/plan.v1.md's "limit" stage exactly for any
    # field a hand-written plan omits: ceiling_dbtp -1.0, lookahead_ms 5.0,
    # release_ms 50.0, oversample 4. release_ms previously defaulted to
    # 100.0 here (dsp.limiter.brickwall's own default, tuned independently
    # of this contract) and oversample was never read at all -- a plan
    # naming a non-default oversample was silently ignored.
    ceiling_dbtp = params.get("ceiling_dbtp", -1.0)
    lookahead_ms = params.get("lookahead_ms", 5.0)
    release_ms = params.get("release_ms", 50.0)
    oversample = params.get("oversample", 4)

    y, stats = limiter.brickwall(
        x,
        sr,
        ceiling_dbtp=ceiling_dbtp,
        lookahead_ms=lookahead_ms,
        release_ms=release_ms,
        oversample=oversample,
    )
    return y, stats


def _apply_downmix(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    return _channels.downmix(x, sr, params)


def _apply_resample(x: np.ndarray, sr: int, params: dict[str, Any]) -> tuple[np.ndarray, dict]:
    target_hz = params["target_hz"]
    return _resample_module.resample(x, sr, target_hz)


_REGISTRY = {
    "cut": _apply_cut,
    "strip_silence": _apply_strip_silence,
    "gate": _apply_gate,
    "expand": _apply_expand,
    "dereverb": _apply_dereverb,
    "deess": _apply_deess,
    "eq": _apply_eq,
    "eq_match": _apply_eq_match,
    "compress": _apply_compress,
    "saturate": _apply_saturate,
    "reverb": _apply_reverb,
    "stretch": _apply_stretch,
    "pitch": _apply_pitch,
    "loudness": _apply_loudness,
    "limit": _apply_limit,
    "downmix": _apply_downmix,
    "resample": _apply_resample,
}

# The one stage whose executor changes the sample rate for every stage
# after it -- see this module's docstring ("Sample-rate tracking") for why
# this is a targeted loop special-case rather than a widened executor
# contract.
_SAMPLE_RATE_CHANGING_STAGE = "resample"


def apply_plan(x: np.ndarray, sr: int, stages: list[_Stage]) -> tuple[np.ndarray, dict]:
    """Apply an ordered list of mastering stages in one pass.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        stages: Stages already in canonical mastering order, each an
            object exposing `.stage` (str) and `.params` (dict).

    Returns:
        (y, report) where report = {"stages": [...], "sample_rate": int}.
        Each per-stage dict in "stages" always includes "stage" (the stage
        name) plus whatever specifics that stage's handler records.
        "sample_rate" is `sr` unless a `resample` stage ran, in which case
        it is that stage's `target_hz` -- see this module's docstring
        ("Sample-rate tracking"). `aud.lib.render` reads this key to know
        what rate to write the rendered file at.

    Raises:
        ValueError: A stage name is not in the implemented registry. The
            error names the stage explicitly rather than failing silently
            or leaving it as a no-op.
        StageParamError: A stage handler rejected one of its own params
            (out of range, wrong shape) -- e.g. a hand-authored plan with
            an invalid field a real caller could never construct. Carries
            the stage name; `aud.lib.render` maps it to `AudError(code=
            "bad_param")`.
    """
    y = np.asarray(x, dtype=np.float64)
    current_sr = sr
    report: dict[str, Any] = {"stages": []}

    for stage in stages:
        name = stage.stage
        handler = _REGISTRY.get(name)
        if handler is None:
            raise NotImplementedStageError(name, tuple(_REGISTRY))
        params = stage.params or {}
        try:
            y, stage_report = handler(y, current_sr, params)
        except (MissingDspModuleError, CrossfadeExceedsGapError, CurveError):
            # Already-named boundary exceptions `aud.lib.render` recognises
            # on their own terms (not_implemented / crossfade_exceeds_gap /
            # eq_match's own bad_param mapping) -- pass through unchanged
            # rather than flattening them into a generic StageParamError.
            raise
        except ValueError as exc:
            raise StageParamError(name, str(exc)) from exc
        if name == _SAMPLE_RATE_CHANGING_STAGE:
            current_sr = stage_report["target_hz"]
        stage_report = {"stage": name, **stage_report}
        report["stages"].append(stage_report)

    report["sample_rate"] = current_sr
    return y, report
