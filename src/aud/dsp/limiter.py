"""True-peak aware, lookahead brickwall limiter.

Architecture note (this is the design decision, not an afterthought): this
module limits the FULL-BAND signal, once, after multiband compression has
already recombined its bands into one signal. It deliberately does not
offer a "limit each band separately" mode. Independently limited bands can
each individually meet a ceiling and still sum above it once recombined --
peaks in adjacent bands are not generally time-aligned with each other, so
per-band ceilings do not compose into a whole-signal ceiling. This mirrors
how the commercial mastering chains this tool is modeled on are built:
FabFilter Pro-MB (multiband dynamics) feeds into Pro-L2 (one full-band
true-peak limiter); iZotope's Dynamics module feeds into its Maximizer.
Multiband dynamics shapes tone/energy per band; one final full-band limiter
is what actually guarantees the ceiling. The arithmetic (sum of N
independently-bounded signals is bounded only by N times the bound, not by
the bound itself) is on that side of the argument.

Sample peak understates true peak: a reconstructed (band-limited, i.e. real
D/A) waveform can overshoot between two adjacent samples that are each
individually below full scale -- the intersample peak. `true_peak_dbtp`
oversamples (polyphase FIR resampling, i.e. `scipy.signal.resample_poly`)
to approximate what the reconstructed analog waveform's peak actually is,
which is the entire reason this function exists as distinct from a plain
`max(abs(x))`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

__all__ = ["brickwall", "true_peak_dbtp"]

_EPS = 1e-12


def true_peak_dbtp(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Estimate the true (intersample) peak level, in dBTP.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz (unused by the resampling math itself, kept
            for API symmetry/future use, e.g. filter design tied to rate).
        oversample: Oversampling factor for the polyphase reconstruction.

    Returns:
        The peak absolute sample value of the oversampled reconstruction,
        in dBTP (0 dBTP == full scale).
    """
    x = np.asarray(x, dtype=np.float64)
    if oversample <= 1:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
    else:
        x_os = signal.resample_poly(x, up=oversample, down=1, axis=0)
        peak = float(np.max(np.abs(x_os))) if x_os.size else 0.0
    return float(20.0 * np.log10(max(peak, _EPS)))


def _lookahead_min(gain: np.ndarray, window: int) -> np.ndarray:
    """gain_la[n] = min(gain[n : n + window]) (forward-looking, causal-with-lookahead)."""
    if window <= 1:
        return gain
    # origin shifts the (odd- or even-sized) window forward so it covers
    # [n, n+window-1] instead of being centered on n. Edge samples near the
    # tail reuse the nearest available value (mode="nearest") since there
    # is no real "future" data beyond the end of the buffer -- a small,
    # documented approximation affecting at most the last `window` samples.
    origin = -(window // 2)
    return ndimage.minimum_filter1d(gain, size=window, mode="nearest", origin=origin)


def brickwall(
    x: np.ndarray,
    sr: int,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 100.0,
    oversample: int = 4,
) -> tuple[np.ndarray, dict]:
    """Lookahead, true-peak-aware brickwall limiter.

    Design: work in an oversampled domain throughout so the ceiling is
    enforced against the *true* peak, not the sample peak. A forward
    (lookahead) minimum filter over the target per-sample gain gives early
    warning of an upcoming peak (the "attack" is effectively instantaneous
    because the gain has already started dropping before the peak
    arrives); a one-pole filter then smooths the *release* back toward
    unity gain. The whole gain-applied signal is delayed by the lookahead
    window so the anticipatory gain reduction lines up with the peak it
    was computed for, then decimated back to the original rate.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        ceiling_dbtp: True-peak ceiling, in dBTP (e.g. -1.0).
        lookahead_ms: Lookahead window, in milliseconds.
        release_ms: Release time constant, in milliseconds.
        oversample: Oversampling factor used for true-peak detection and
            gain computation.

    Returns:
        (y, stats) where stats = {
            "input_true_peak_dbtp": float,
            "output_true_peak_dbtp": float,
            "max_gain_reduction_db": float,
            "ceiling_dbtp": float,
            "ceiling_met": bool,
        }
    """
    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[0]
    input_tp = true_peak_dbtp(x, sr, oversample=oversample)

    if oversample <= 1:
        x_os = x.copy()
        sr_os = sr
    else:
        x_os = signal.resample_poly(x, up=oversample, down=1, axis=0)
        sr_os = sr * oversample

    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)
    level = np.max(np.abs(x_os), axis=1) if x_os.ndim == 2 else np.abs(x_os)
    instantaneous_gain = np.minimum(1.0, ceiling_lin / np.maximum(level, _EPS))

    window = max(1, round(lookahead_ms / 1000.0 * sr_os))
    gain_la = _lookahead_min(instantaneous_gain, window)

    release_coeff = np.exp(-1.0 / (sr_os * max(release_ms, 1e-3) / 1000.0))
    smoothed = np.empty_like(gain_la)
    prev = 1.0
    for i in range(gain_la.shape[0]):
        target = gain_la[i]
        # attack: already anticipated by the lookahead min-filter, apply immediately.
        # release: smooth back toward unity gain with the one-pole coefficient.
        prev = target if target < prev else release_coeff * prev + (1 - release_coeff) * target
        smoothed[i] = prev

    gain_applied = x_os * smoothed[:, None] if x_os.ndim == 2 else x_os * smoothed

    delay = window - 1
    if delay > 0:
        delayed = np.zeros_like(gain_applied)
        delayed[delay:] = gain_applied[:-delay]
        gain_applied = delayed

    y = gain_applied if oversample <= 1 else signal.resample_poly(gain_applied, up=1, down=oversample, axis=0)

    # resample_poly's output length can differ by a sample or two from the
    # exact input length; trim/pad to guarantee the contract "same shape as x".
    if y.shape[0] > n_in:
        y = y[:n_in]
    elif y.shape[0] < n_in:
        pad = np.zeros((n_in - y.shape[0], y.shape[1]) if y.ndim == 2 else (n_in - y.shape[0],))
        y = np.concatenate([y, pad], axis=0)

    output_tp = true_peak_dbtp(y, sr, oversample=oversample)
    max_gain_reduction_db = 20.0 * np.log10(max(float(np.min(smoothed)), _EPS))
    # Small numerical tolerance: a limiter that lands within 0.05 dB of the
    # ceiling due to floating-point/oversampling-filter ripple has met it in
    # every practical sense; anything beyond that is reported honestly as not met.
    ceiling_met = bool(output_tp <= ceiling_dbtp + 0.05)

    stats = {
        "input_true_peak_dbtp": input_tp,
        "output_true_peak_dbtp": output_tp,
        "max_gain_reduction_db": float(max_gain_reduction_db),
        "ceiling_dbtp": float(ceiling_dbtp),
        "ceiling_met": ceiling_met,
    }
    return y, stats
