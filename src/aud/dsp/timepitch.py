"""Time-stretching and pitch-shifting: retime or re-pitch, never both at once.

`time_stretch` changes duration while preserving pitch. `pitch_shift`
changes pitch while preserving duration. Combining both effects is two
plan stages (`stretch` then `pitch`, or the reverse), never a hidden third
mode in this module.

Base tier -- always available, zero extra dependencies -- is a phase
vocoder: STFT analysis, the standard "true instantaneous frequency"
phase-advance formula (unwrap the per-bin phase difference against the
expected advance, integrate at the synthesis hop), reconstructed with
IDENTITY PHASE LOCKING (Puckette/Laroche-Dolson) across bins within a
frame. A plain per-bin-independent phase vocoder smears transients ("loses
its punch", the well-known "phasiness" artifact) because each bin's phase
drifts independently after enough frames; locking each frame's non-peak
bins to their nearest spectral peak -- preserving THIS frame's own
inter-bin phase relationship while only the peak's phase is time-advanced
by the tracking formula -- measurably reduces that smearing. It does not
eliminate it; no STFT phase vocoder does.

Optional quality tier: if `python_stretch` (Signalsmith Stretch, MIT,
declared as this package's `stretch` extra in pyproject.toml) is
importable, it is used instead. Imported LAZILY, inside the two public
functions, never at module level -- the base tier must keep working with
zero extra install (AGENTS.md #3). This module's `_stretch_signalsmith`/
`_pitch_signalsmith` are coded directly against `python_stretch`'s
documented API (https://pypi.org/project/python-stretch/:
`ps.Signalsmith.Stretch()`, `.preset(channels, sample_rate)`,
`.timeFactor = ...`, `.setTransposeSemitones(...)`, `.process(audio)` with
audio shaped (channels, samples)) but has NOT been exercised against a
real install in this repo's test suite -- installing extras here is
DTU-only. `quality="phase_vocoder"` forces the path that IS tested.

Pitch-shift is built on time-stretch plus resampling, not as a separate
algorithm: stretch by the shift ratio (changes duration, preserves pitch),
then resample back to the original duration (changes nothing but the
playback rate, which is exactly a pitch shift). This reuses the one
correctness-critical piece of DSP here instead of maintaining two.
"""

from __future__ import annotations

import importlib.util
import math
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from scipy import signal

__all__ = ["pitch_shift", "time_stretch"]

_QUALITY_MODES = ("auto", "phase_vocoder", "signalsmith")
Quality = Literal["auto", "phase_vocoder", "signalsmith"]


def _signalsmith_available() -> bool:
    return importlib.util.find_spec("python_stretch") is not None


def _select_engine(quality: str) -> str:
    if quality not in _QUALITY_MODES:
        raise ValueError(f"quality must be one of {_QUALITY_MODES}, got {quality!r}")
    if quality == "phase_vocoder":
        return "phase_vocoder"
    if quality == "signalsmith":
        if not _signalsmith_available():
            raise RuntimeError(
                "quality='signalsmith' requested but python_stretch is not importable. "
                "Install this package's 'stretch' extra, or use quality='auto'/'phase_vocoder'."
            )
        return "signalsmith"
    # auto
    return "signalsmith" if _signalsmith_available() else "phase_vocoder"


# ---------------------------------------------------------------------------
# Phase vocoder (base tier)
# ---------------------------------------------------------------------------


def _identity_phase_lock(mag: np.ndarray, phase: np.ndarray, phase_acc: np.ndarray) -> np.ndarray:
    """Lock each bin's accumulated phase to its nearest spectral peak.

    Peaks are simple local maxima of |X[k]|. Every bin between two
    consecutive peaks (split at the midpoint) is re-derived as
    `phase_acc[peak] + (phase[bin] - phase[peak])`: the peak's own
    phase carries the time-advance computed by the true-frequency
    formula, and the bin keeps THIS frame's own phase relationship to
    that peak, rather than drifting from its own independently
    integrated (and much noisier, for a non-peak bin) phase history.
    """
    n = mag.shape[0]
    if n < 3:
        return phase_acc
    peak_mask = np.zeros(n, dtype=bool)
    peak_mask[1:-1] = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:])
    peak_mask[0] = mag[0] >= mag[1]
    peak_mask[-1] = mag[-1] >= mag[-2]
    peaks = np.nonzero(peak_mask)[0]
    if peaks.size == 0:
        return phase_acc

    locked = phase_acc.copy()
    bounds = [(int(peaks[j]) + int(peaks[j + 1])) // 2 for j in range(len(peaks) - 1)]
    starts = [0, *(b + 1 for b in bounds)]
    ends = [*bounds, n - 1]
    for p, s, e in zip(peaks, starts, ends, strict=True):
        region = slice(s, e + 1)
        locked[region] = phase_acc[p] + (phase[region] - phase[p])
    return locked


def _phase_vocoder_stretch_mono(x: np.ndarray, factor: float, n_fft: int = 1024) -> np.ndarray:
    """Stretch one channel by `factor`. Returns exactly round(len(x) * factor) samples."""
    n_in = x.shape[0]
    target_len = max(1, round(n_in * factor))
    if n_in == 0:
        return np.zeros(target_len)

    # Shrink the frame for very short inputs (test fixtures, tail ends of a
    # render) rather than crashing on frame > signal.
    while n_fft > 64 and n_fft > n_in:
        n_fft //= 2
    n_fft = max(n_fft, 64)
    hop_analysis = max(1, n_fft // 4)
    hop_synthesis = max(1, round(hop_analysis * factor))

    window = signal.windows.hann(n_fft, sym=False)
    pad = n_fft // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(n_fft)])
    n_frames = 1 + max(0, (len(xp) - n_fft) // hop_analysis)

    n_bins = n_fft // 2 + 1
    omega = 2.0 * np.pi * np.arange(n_bins) / n_fft

    out_len = pad + (n_frames - 1) * hop_synthesis + n_fft + pad
    y = np.zeros(out_len)
    win_sum = np.zeros(out_len)

    prev_phase = np.zeros(n_bins)
    phase_acc = np.zeros(n_bins)

    for i in range(n_frames):
        start = i * hop_analysis
        frame = xp[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        if i == 0:
            phase_acc = phase.copy()
        else:
            delta = phase - prev_phase - omega * hop_analysis
            delta_wrapped = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
            true_freq = omega + delta_wrapped / hop_analysis
            phase_acc = phase_acc + true_freq * hop_synthesis
            phase_acc = _identity_phase_lock(mag, phase, phase_acc)
        prev_phase = phase

        out_spec = mag * np.exp(1j * phase_acc)
        out_frame = np.fft.irfft(out_spec, n=n_fft) * window

        out_start = i * hop_synthesis
        y[out_start : out_start + n_fft] += out_frame
        win_sum[out_start : out_start + n_fft] += window * window

    win_sum[win_sum < 1e-8] = 1.0
    y = y / win_sum

    y = y[pad:]
    if y.shape[0] < target_len:
        y = np.concatenate([y, np.zeros(target_len - y.shape[0])])
    return y[:target_len]


def _resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (n_samples, n_channels) x so its length is exactly target_len."""
    n_in = x.shape[0]
    if n_in == target_len:
        return x
    if n_in == 0 or target_len <= 0:
        return np.zeros((max(target_len, 0), x.shape[1]))
    frac = Fraction(target_len, n_in).limit_denominator(2000)
    y = signal.resample_poly(x, up=frac.numerator, down=frac.denominator, axis=0)
    if y.shape[0] > target_len:
        y = y[:target_len]
    elif y.shape[0] < target_len:
        y = np.concatenate([y, np.zeros((target_len - y.shape[0], y.shape[1]))], axis=0)
    return y


# ---------------------------------------------------------------------------
# Signalsmith Stretch (optional quality tier) -- UNTESTED against a real
# install in this repo; see module docstring.
# ---------------------------------------------------------------------------


def _stretch_signalsmith(x2: np.ndarray, sr: int, factor: float) -> np.ndarray:
    import python_stretch as ps  # lazy: optional 'stretch' extra, see module docstring

    n_channels = x2.shape[1]
    audio = np.ascontiguousarray(x2.T, dtype=np.float32)  # (channels, samples), per documented API
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(n_channels, sr)
    stretch.timeFactor = float(factor)
    processed = stretch.process(audio)
    return np.asarray(processed, dtype=np.float64).T


def _pitch_signalsmith(x2: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import python_stretch as ps  # lazy: optional 'stretch' extra, see module docstring

    n_channels = x2.shape[1]
    audio = np.ascontiguousarray(x2.T, dtype=np.float32)
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(n_channels, sr)
    stretch.setTransposeSemitones(float(semitones))
    processed = stretch.process(audio)
    return np.asarray(processed, dtype=np.float64).T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def time_stretch(x: np.ndarray, sr: int, factor: float, *, quality: Quality = "auto") -> tuple[np.ndarray, dict]:
    """Retime a signal by `factor`, preserving pitch.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        factor: Output duration / input duration. > 1 lengthens.
        quality: "auto" (use Signalsmith Stretch if importable, else the
            built-in phase vocoder), "phase_vocoder" (force the built-in),
            or "signalsmith" (force the optional tier; raises if the
            'stretch' extra is not installed).

    Returns:
        (y, stats). stats["engine"] names which implementation ran
        ("phase_vocoder" or "signalsmith"); stats also reports the exact
        input/output sample counts and the measured length ratio.

    Raises:
        ValueError: factor is not a finite number > 0, or quality is not
            one of the three modes above.
        RuntimeError: quality="signalsmith" but python_stretch is absent.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"factor must be a finite number > 0, got {factor!r}")

    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n_in = x2.shape[0]

    engine = _select_engine(quality)
    if engine == "signalsmith":
        y2 = _stretch_signalsmith(x2, sr, factor)
    else:
        channels = [_phase_vocoder_stretch_mono(x2[:, c], factor) for c in range(x2.shape[1])]
        y2 = np.stack(channels, axis=1)

    stats: dict[str, Any] = {
        "engine": engine,
        "requested_factor": float(factor),
        "input_samples": int(n_in),
        "output_samples": int(y2.shape[0]),
        "measured_factor": float(y2.shape[0] / n_in) if n_in else 0.0,
    }
    y = y2[:, 0] if squeeze else y2
    return y, stats


def pitch_shift(x: np.ndarray, sr: int, semitones: float, *, quality: Quality = "auto") -> tuple[np.ndarray, dict]:
    """Re-pitch a signal by `semitones`, preserving duration.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        semitones: Positive is up. No range restriction at this layer --
            contracts/plan.v1.md's [-24, 24] is a plan-level constraint
            enforced in aud.lib, not a structural limit of the math here.
        quality: Same three modes as `time_stretch`.

    Returns:
        (y, stats). stats["engine"] names which implementation ran;
        stats["frequency_ratio"] is 2**(semitones/12).

    Raises:
        ValueError: semitones is not finite, or quality is not one of the
            three modes above.
        RuntimeError: quality="signalsmith" but python_stretch is absent.
    """
    if not math.isfinite(semitones):
        raise ValueError(f"semitones must be finite, got {semitones!r}")

    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n_in = x2.shape[0]
    ratio = 2.0 ** (semitones / 12.0)

    engine = _select_engine(quality)
    if engine == "signalsmith":
        y2 = _pitch_signalsmith(x2, sr, semitones)
    else:
        channels = [_phase_vocoder_stretch_mono(x2[:, c], ratio) for c in range(x2.shape[1])]
        stretched2 = np.stack(channels, axis=1)
        y2 = _resample_to_length(stretched2, n_in)

    stats: dict[str, Any] = {
        "engine": engine,
        "semitones": float(semitones),
        "frequency_ratio": float(ratio),
        "input_samples": int(n_in),
        "output_samples": int(y2.shape[0]),
    }
    y = y2[:, 0] if squeeze else y2
    return y, stats
