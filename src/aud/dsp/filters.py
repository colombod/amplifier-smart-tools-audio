"""RBJ audio-EQ-cookbook biquads, and plain Butterworth HPF/LPF, as scipy sos.

Reference for peaking/shelf coefficients: Robert Bristow-Johnson's
"Audio EQ Cookbook" (the de-facto standard formulas used across the audio
DSP world -- see e.g. https://www.w3.org/documents/audio-eq-cookbook/).

Design choice: the cookbook derives shelving-filter Q from a "shelf slope"
parameter S, not from Q directly. This module's public contract takes a
single `q` for peaking, low_shelf and high_shelf alike (a uniform knob
across filter types), so shelving alpha is computed the same way as
peaking alpha (``alpha = sin(w0) / (2*Q)``) rather than introducing a
second, shelf-only slope parameter. This is a deliberate simplification:
it means "q" for a shelf does not correspond 1:1 to the cookbook's "S",
but it keeps one consistent knob across all three filter types.

highpass/lowpass take an explicit `order` (not fixed at 2 like the cookbook
biquads), so those are built directly from `scipy.signal.butter(..., output="sos")`
rather than from cookbook formulas -- an exact, standard Butterworth
response at any even or odd order, with no hand-derived higher-order
coefficients to get wrong.

All filters are applied causally (`scipy.signal.sosfilt`), never
zero-phase (`sosfiltfilt`): this is a mastering processor with real
attack/release timing elsewhere in the chain, and a zero-phase filter
would smear transients backward in time, which is not physically
realizable and not what a listener (or a downstream compressor) hears.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

__all__ = [
    "apply_sos",
    "high_shelf",
    "highpass",
    "low_shelf",
    "lowpass",
    "peaking",
]


def _validate_freq(sr: int, f0: float) -> None:
    nyquist = sr / 2.0
    if not (0 < f0 < nyquist):
        raise ValueError(f"f0={f0} Hz must be between 0 and the Nyquist frequency ({nyquist} Hz) for sr={sr}")


def _normalize_sos_row(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> np.ndarray:
    """Build a single-section sos row, normalized so a0 == 1."""
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def peaking(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ peaking (bell) EQ filter.

    Args:
        sr: Sample rate in Hz.
        f0: Center frequency in Hz.
        gain_db: Boost (positive) or cut (negative) at f0, in dB.
        q: Quality factor (higher = narrower bell).

    Returns:
        scipy sos array, shape (1, 6).
    """
    _validate_freq(sr, f0)
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")

    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * q)

    b0 = 1 + alpha * a
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a
    a0 = 1 + alpha / a
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a
    return _normalize_sos_row(b0, b1, b2, a0, a1, a2)


def low_shelf(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ low-shelf filter (boost/cut everything below f0).

    Args:
        sr: Sample rate in Hz.
        f0: Shelf midpoint frequency in Hz.
        gain_db: Shelf gain in dB.
        q: Quality factor controlling the transition shape (see module
            docstring for how this maps to the cookbook's "S" slope).

    Returns:
        scipy sos array, shape (1, 6).
    """
    _validate_freq(sr, f0)
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")

    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * q)
    sqrt_a = math.sqrt(a)

    b0 = a * ((a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = 2 * a * ((a - 1) - (a + 1) * cos_w0)
    b2 = a * ((a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = -2 * ((a - 1) + (a + 1) * cos_w0)
    a2 = (a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    return _normalize_sos_row(b0, b1, b2, a0, a1, a2)


def high_shelf(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ high-shelf filter (boost/cut everything above f0).

    Args:
        sr: Sample rate in Hz.
        f0: Shelf midpoint frequency in Hz.
        gain_db: Shelf gain in dB.
        q: Quality factor controlling the transition shape (see module
            docstring for how this maps to the cookbook's "S" slope).

    Returns:
        scipy sos array, shape (1, 6).
    """
    _validate_freq(sr, f0)
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")

    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * q)
    sqrt_a = math.sqrt(a)

    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * sqrt_a * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * sqrt_a * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * sqrt_a * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * sqrt_a * alpha
    return _normalize_sos_row(b0, b1, b2, a0, a1, a2)


def highpass(sr: int, f0: float, order: int = 2) -> np.ndarray:
    """Butterworth highpass filter.

    Args:
        sr: Sample rate in Hz.
        f0: -3 dB corner frequency in Hz.
        order: Filter order (2 => 12 dB/octave, etc.).

    Returns:
        scipy sos array, shape (ceil(order/2), 6).
    """
    _validate_freq(sr, f0)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    return signal.butter(order, f0, btype="highpass", fs=sr, output="sos")


def lowpass(sr: int, f0: float, order: int = 2) -> np.ndarray:
    """Butterworth lowpass filter.

    Args:
        sr: Sample rate in Hz.
        f0: -3 dB corner frequency in Hz.
        order: Filter order (2 => 12 dB/octave, etc.).

    Returns:
        scipy sos array, shape (ceil(order/2), 6).
    """
    _validate_freq(sr, f0)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    return signal.butter(order, f0, btype="lowpass", fs=sr, output="sos")


def apply_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply a cascade of second-order sections, causally, per channel.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sos: scipy sos array as returned by the filter builders above.

    Returns:
        Filtered array, same shape and dtype as x (cast to float64).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return signal.sosfilt(sos, x, axis=0)
    return signal.sosfilt(sos, x, axis=0)
