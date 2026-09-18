"""Oversampled waveshaping saturation: soft / tape / tube.

Every mode oversamples before the nonlinearity and decimates after. A
waveshaper is a nonlinear operation, so it generates harmonics above the
input's own bandwidth; run at the original sample rate, those harmonics
that land above Nyquist fold back (alias) into the audible range as
inharmonic, un-musical artifacts. That aliasing is the literal difference
between a "warm" saturator and a "cheap/harsh" one -- the algorithm is
identical, only the aliasing differs.

Modes:
    soft -- symmetric tanh soft-clip (odd-harmonic only; no DC shift).
    tape -- gently asymmetric soft-clip: positive and negative half-cycles
            see slightly different effective drive, which generates a
            mild 2nd-harmonic (even-order) component on top of tanh's
            odd-order series -- the "gentle 2nd-harmonic term" called for.
    tube -- the same asymmetric construction as tape, with a larger
            asymmetry factor, so the 2nd-harmonic content is more
            pronounced (per spec: "more 2nd harmonic" than tape).

Dry/wet mix includes a gain compensation stage: tanh-family saturation
reduces RMS level as drive increases (it is, among other things, a soft
compressor), so without compensation turning up "mix" alone would make
the signal quieter -- conflating the *mix* knob with a *level* change.
Compensation matches the wet signal's RMS back to the dry signal's RMS
(clamped to a sane maximum boost) before blending, so mix controls
character, not loudness.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["saturate"]

_EPS = 1e-12
_MAX_COMPENSATION_DB = 12.0


def _shape_soft(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(drive * x)


def _shape_asymmetric(x: np.ndarray, drive: float, asymmetry: float) -> np.ndarray:
    """tanh with a different effective drive on each half-cycle.

    asymmetry=0 reduces to plain tanh(drive*x). A small positive asymmetry
    makes the negative half-cycle clip slightly harder than the positive
    half, which is what generates the even-harmonic (2nd harmonic) content
    described in the module docstring for "tape" and "tube".
    """
    pos = x >= 0
    drive_neg = drive * (1.0 + asymmetry)
    y = np.where(
        pos,
        np.tanh(drive * x),
        np.tanh(drive_neg * x) / (1.0 + asymmetry),
    )
    return y


def saturate(
    x: np.ndarray, drive: float = 1.0, mode: str = "soft", mix: float = 1.0, oversample: int = 4
) -> np.ndarray:
    """Apply oversampled waveshaping saturation.

    Args:
        x: Array of shape (n_samples, n_channels).
        drive: Pre-gain into the nonlinearity (higher = more saturation).
        mode: "soft", "tape", or "tube".
        mix: Dry/wet blend, 0.0 (bypass) to 1.0 (fully wet).
        oversample: Oversampling factor applied before the nonlinearity.

    Returns:
        Array, same shape as x.

    Raises:
        ValueError: Unknown mode.
    """
    if mode not in ("soft", "tape", "tube"):
        raise ValueError(f"mode must be 'soft', 'tape', or 'tube', got {mode!r}")

    x = np.asarray(x, dtype=np.float64)
    n_in = x.shape[0]

    x_os = signal.resample_poly(x, up=oversample, down=1, axis=0) if oversample > 1 else x

    if mode == "soft":
        wet_os = _shape_soft(x_os, drive)
    elif mode == "tape":
        wet_os = _shape_asymmetric(x_os, drive, asymmetry=0.15)
    else:  # "tube"
        wet_os = _shape_asymmetric(x_os, drive, asymmetry=0.4)

    wet = signal.resample_poly(wet_os, up=1, down=oversample, axis=0) if oversample > 1 else wet_os

    if wet.shape[0] > n_in:
        wet = wet[:n_in]
    elif wet.shape[0] < n_in:
        pad = np.zeros((n_in - wet.shape[0], wet.shape[1]) if wet.ndim == 2 else (n_in - wet.shape[0],))
        wet = np.concatenate([wet, pad], axis=0)

    dry_rms = float(np.sqrt(np.mean(x * x)))
    wet_rms = float(np.sqrt(np.mean(wet * wet)))
    max_gain = 10.0 ** (_MAX_COMPENSATION_DB / 20.0)
    compensation = min(dry_rms / max(wet_rms, _EPS), max_gain) if dry_rms > _EPS else 1.0
    wet_compensated = wet * compensation

    return (1.0 - mix) * x + mix * wet_compensated
