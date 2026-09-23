"""Sample-rate conversion for the `resample` plan stage.

Uses `scipy.signal.resample_poly`'s polyphase resampler, which applies its
own anti-aliasing low-pass filter as part of the resampling. This module
never hand-rolls decimation and never strips that filter -- doing either
would reintroduce exactly the aliasing `resample_poly` exists to prevent.

The up/down ratio is reduced to small integers via `Fraction.limit_denominator`
before it reaches `resample_poly`, the same idiom already used across this
codebase for rate conversion: `aud.dsp.speech._resample_to_whisper_rate`
(resampling to faster-whisper's fixed 16 kHz), `aud.dsp.timepitch` (the
`stretch`/`pitch` stages' resampling step), `aud.dsp.reverb` (matching an
impulse response's native rate to the render rate), `aud.dsp.limiter` and
`aud.dsp.saturation` (oversampling for true-peak measurement and anti-alias
around a nonlinearity). A raw `Fraction(target, source)` for an arbitrary
pair of real-world sample rates can have a denominator in the millions;
`limit_denominator` keeps `resample_poly`'s per-channel FIR design cheap.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal

__all__ = ["resample"]

# Matches aud.dsp.speech._resample_to_whisper_rate's own bound: small enough
# that resample_poly's FIR design stays cheap, large enough that the
# fraction reduction is still an accurate approximation of the true ratio
# for any pair of real-world audio sample rates (8 kHz-192 kHz and beyond).
_MAX_DENOMINATOR = 2000


def resample(x: np.ndarray, sr: int, target_hz: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample `x` from `sr` to `target_hz`, preserving its (n, channels) shape.

    Args:
        x: Array of shape (n_samples, n_channels), or (n_samples,) for a
            mono signal.
        sr: `x`'s current sample rate, in Hz.
        target_hz: The sample rate to resample to, in Hz.

    Returns:
        `(y, report)`. `y` has the same number of channels as `x` (2-D in,
        2-D out; 1-D in, 1-D out), resampled to `target_hz`. `report`:
        `source_hz`, `target_hz`, `input_samples`, `output_samples`.

    A no-op when `target_hz == sr`: returns `x` itself, not a copy of it
    processed through the resampler for nothing -- matches
    `aud.dsp.speech._resample_to_whisper_rate`'s own no-op rule.

    Every channel is resampled independently but with the identical
    up/down ratio (and therefore the identical anti-alias filter), so
    channels stay time- and phase-aligned relative to each other after
    the call.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    input_samples = x2.shape[0]

    if int(target_hz) == int(sr):
        output = x2
    else:
        frac = Fraction(int(target_hz), int(sr)).limit_denominator(_MAX_DENOMINATOR)
        channels = [
            signal.resample_poly(x2[:, c], up=frac.numerator, down=frac.denominator) for c in range(x2.shape[1])
        ]
        output = np.stack(channels, axis=1)

    y = output[:, 0] if squeeze else output
    return y, {
        "source_hz": int(sr),
        "target_hz": int(target_hz),
        "input_samples": int(input_samples),
        "output_samples": int(output.shape[0]),
    }
