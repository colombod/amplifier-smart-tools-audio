"""Honest, numeric-only measurement of what is actually in a file.

Every value is a number (or null) -- no prose verdicts, no "sounds good".
Judgement about what those numbers mean belongs to a later, model-backed
verb; this module only measures.

`integrated_lufs` and `loudness_range_lu` are a DOCUMENTED partial outcome:
both are BS.1770-gated measurements that pyloudnorm cannot compute for
audio shorter than its gating block (`_BS1770_MIN_SECONDS`, 0.4s) -- every
other field in `analyze`'s result still measures normally. When that
precondition is not met, both are `null` and `loudness_unavailable_reason`
names why, rather than the whole call failing or (the previous defect) a
bare `except Exception` around only the range measurement silently
swallowing any bug alongside the one genuinely expected condition.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from aud.dsp import limiter as _limiter
from aud.dsp import loudness as _loudness

__all__ = ["analyze"]

_EPS = 1e-12

# Standard preferred octave-band center frequencies, 31.5 Hz .. 16 kHz.
_OCTAVE_CENTERS = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)

_SIBILANCE_LOW_HZ = 5000.0
_SIBILANCE_HIGH_HZ = 9000.0

_CLIP_THRESHOLD = 0.999

# pyloudnorm.Meter's default `block_size` (seconds): the minimum audio
# length its BS.1770 gating needs. Below this, both `integrated_lufs` and
# `loudness_range` raise `ValueError("Audio must have length greater than
# the block size.")` -- a real, easily reproducible condition (any file
# shorter than 400ms), not a hypothetical.
_BS1770_MIN_SECONDS = 0.4


def _finite_or_none(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, _EPS))


def _octave_band_energy_db(x: np.ndarray, sr: int) -> dict[str, float | None]:
    nyquist = sr / 2.0
    bands: dict[str, float | None] = {}
    for center in _OCTAVE_CENTERS:
        low = center / math.sqrt(2)
        high = center * math.sqrt(2)
        low = max(low, 1.0)
        high = min(high, 0.999 * nyquist)
        key = str(center)
        if low >= high:
            bands[key] = None
            continue
        sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        band_signal = signal.sosfilt(sos, x, axis=0)
        rms = float(np.sqrt(np.mean(band_signal**2)))
        bands[key] = _db(rms)
    return bands


def _sibilance_ratio(x: np.ndarray, sr: int) -> float | None:
    nyquist = sr / 2.0
    low = _SIBILANCE_LOW_HZ
    high = min(_SIBILANCE_HIGH_HZ, 0.999 * nyquist)
    if low >= high:
        return None
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    band_signal = signal.sosfilt(sos, x, axis=0)
    band_energy = float(np.sum(band_signal**2))
    total_energy = float(np.sum(x**2))
    if total_energy <= _EPS:
        return 0.0
    return band_energy / total_energy


def _noise_floor_dbfs(x: np.ndarray, sr: int) -> float:
    """Crude noise-floor estimate: a low percentile of short-term RMS."""
    frame_len = max(1, int(0.05 * sr))
    n_frames = x.shape[0] // frame_len
    if n_frames < 1:
        rms = float(np.sqrt(np.mean(x**2)))
        return _db(rms)
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len, -1)
    frame_rms = np.sqrt(np.mean(frames**2, axis=(1, 2)))
    floor = float(np.percentile(frame_rms, 10))
    return _db(floor)


def _ambience_decay_estimate_ms(x: np.ndarray, sr: int) -> float | None:
    """Crude decay-rate heuristic -- NOT a true RT60 measurement.

    Averages the dB/frame slope over frames where short-term energy is
    decreasing, extrapolates to a "time to decay 60 dB" figure the same
    way a rough RT60 estimate would, and clips it to a sane range. This is
    a coarse signal-level heuristic, not a room-acoustics measurement.
    """
    frame_len = max(1, int(0.02 * sr))
    n_frames = x.shape[0] // frame_len
    if n_frames < 3:
        return None
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len, -1)
    frame_rms = np.sqrt(np.mean(frames**2, axis=(1, 2)))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, _EPS))
    diffs = np.diff(frame_db)
    decaying = diffs[diffs < 0]
    if decaying.size == 0:
        return None
    avg_decay_db_per_frame = float(np.mean(decaying))
    frame_duration_s = frame_len / sr
    decay_db_per_sec = avg_decay_db_per_frame / frame_duration_s
    if decay_db_per_sec >= 0:
        return None
    estimate_ms = (60.0 / abs(decay_db_per_sec)) * 1000.0
    return float(min(estimate_ms, 5000.0))


def analyze(x: np.ndarray, sr: int) -> dict:
    """Measure a signal. Every value is a number or null; JSON-serializable.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.

    Returns:
        A dict of measurements (see module docstring for the honesty contract).
    """
    x = np.asarray(x, dtype=np.float64)
    n_samples, n_channels = x.shape[0], x.shape[1] if x.ndim == 2 else 1
    if x.ndim == 1:
        x = x[:, None]

    sample_peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    crest_factor_db = _db(sample_peak) - _db(rms) if rms > _EPS else None

    # BS.1770 gating needs at least `_BS1770_MIN_SECONDS` of audio (see the
    # module docstring and the constant's own comment); checking that
    # PRECONDITION up front -- rather than calling into pyloudnorm and
    # catching whatever it raises -- means the only exception this module
    # ever expects from loudness measurement is the one it has already
    # named, and anything else pyloudnorm might raise still surfaces as a
    # genuine crash instead of a silently swallowed null.
    loudness_measurable = sr > 0 and n_samples >= sr * _BS1770_MIN_SECONDS
    loudness_unavailable_reason: str | None = None
    if loudness_measurable:
        integrated_lufs = _loudness.integrated_lufs(x, sr)
        lra = _loudness.loudness_range(x, sr)
    else:
        integrated_lufs = None
        lra = None
        loudness_unavailable_reason = (
            f"sample_rate must be a positive integer, got {sr!r}"
            if sr <= 0
            else (
                f"audio is {n_samples / sr:.3f}s ({n_samples} samples at {sr} Hz); BS.1770 loudness "
                f"gating needs at least {_BS1770_MIN_SECONDS}s"
            )
        )

    dc_offset = [float(np.mean(x[:, ch])) for ch in range(n_channels)]

    stereo_correlation: float | None = None
    mid_side_ratio: float | None = None
    if n_channels >= 2:
        left, right = x[:, 0], x[:, 1]
        if np.std(left) > _EPS and np.std(right) > _EPS:
            stereo_correlation = float(np.corrcoef(left, right)[0, 1])
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid**2)))
        side_rms = float(np.sqrt(np.mean(side**2)))
        mid_side_ratio = side_rms / mid_rms if mid_rms > _EPS else 0.0

    clipped_sample_count = int(np.sum(np.abs(x) >= _CLIP_THRESHOLD))

    result = {
        "sample_rate": int(sr),
        "channels": int(n_channels),
        "duration_seconds": float(n_samples / sr) if sr else 0.0,
        "integrated_lufs": _finite_or_none(integrated_lufs),
        "loudness_range_lu": _finite_or_none(lra),
        "sample_peak_dbfs": _db(sample_peak),
        "true_peak_dbtp": _limiter.true_peak_dbtp(x, sr),
        "crest_factor_db": crest_factor_db,
        "dc_offset": dc_offset,
        "rms_dbfs": _db(rms) if rms > _EPS else None,
        "octave_band_energy_db": _octave_band_energy_db(x, sr),
        "stereo_correlation": stereo_correlation,
        "mid_side_ratio": mid_side_ratio,
        "noise_floor_dbfs": _noise_floor_dbfs(x, sr),
        "clipped_sample_count": clipped_sample_count,
        "sibilance_ratio": _sibilance_ratio(x, sr),
        "ambience_decay_estimate_ms": _ambience_decay_estimate_ms(x, sr),
    }
    if loudness_unavailable_reason is not None:
        result["loudness_unavailable_reason"] = loudness_unavailable_reason
    return result
