"""Controlled ambience: a small feedback delay network (FDN), plus convolution
against a recorded impulse response (IR).

This is a MASTERING tool's reverb, not a cathedral simulator: the manifest's
own words for this stage are "controlled ambience" (see
contracts/plan.v1.md's `reverb` stage and docs/ARCHITECTURE.md's character
stage). Defaults are chosen accordingly -- a touch of space on a dry voice,
not a plate or a hall -- and the two public functions here reflect that in
their default `wet`/`dry` split.

Algorithm choice: a 4-line feedback delay network (Schroeder/Jot-style),
not a plain comb/allpass cascade. Four delay lines at mutually-irrational-
ish lengths (`_BASE_DELAYS_MS`, a Freeverb-style spacing chosen to avoid
coincident resonances) are cross-fed through a normalized 4x4 Hadamard
matrix (`_HADAMARD4`) -- an orthogonal mix, so the feedback network neither
amplifies nor loses energy on its own; only the per-line feedback gain and
the damping filter remove energy, which is what makes the decay tail
controllable and analyzable (see `estimated_decay_s` below).

Decay-time design (this is the one design decision worth being explicit
about): `room_size` scales the delay-line LENGTHS; the per-line feedback
gain (`_FEEDBACK_GAIN`) is a FIXED constant, not derived from a requested
decay time. For a fixed gain, the standard reverberation-time relationship
RT60 = -3 * delay_seconds / log10(gain) means a longer delay line decays
more slowly at the SAME gain -- so increasing `room_size` (which lengthens
every delay line) monotonically lengthens the measured decay tail. This is
deliberate: it is the knob the module's own tests exercise directly. A
caller wanting a specific decay time in seconds (as `contracts/plan.v1.md`'s
`decay_s` field asks for) gets there via `aud.dsp.engine`'s mapping from
`decay_s` onto `room_size`, not by this module accepting `decay_s` itself --
this module's contract is the FDN's own structural parameters.

`damping` is a one-pole low-pass filter INSIDE the feedback path of every
line (not on the dry signal, and not only at the output tap): each pass
through a delay line loses more high-frequency energy than low, so a highly
damped tail is measurably darker (lower spectral centroid) than an
undamped one, in addition to (usually) decaying a little faster overall.

Buffer-length convention: like every other `dsp/` module in this codebase
(saturation, limiter, compress -- see docs/ARCHITECTURE.md #4, the
single-render architecture), `reverb()` and `convolve_ir()` return arrays
the SAME LENGTH as their input. The tail is truncated at the buffer
boundary rather than growing it. A caller who wants the ringing tail
preserved in full supplies trailing silence in the input before render,
the same accommodation any single-pass convolution effect needs in this
architecture.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from aud.dsp.channels import fold_to_mono

__all__ = ["convolve_ir", "reverb"]

_EPS = 1e-12

# Freeverb-style base delay lengths, deliberately not simple ratios of one
# another -- this is what keeps the tail sounding like diffuse reverberation
# rather than a small set of ringing resonances.
_BASE_DELAYS_MS: tuple[float, float, float, float] = (29.7, 37.1, 41.1, 43.7)

# Normalized 4x4 Hadamard matrix: H/2 is orthogonal (H/2 @ (H/2).T == I), so
# mixing the four line outputs through it neither adds nor removes energy --
# only _FEEDBACK_GAIN and the damping filter do that, which is what makes the
# decay time a legible function of those two parameters.
_HADAMARD4 = (
    np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    * 0.5
)

# Fixed per-line feedback attenuation. room_size scales delay LENGTH, not
# this gain -- see module docstring for why that is what makes decay time
# track room_size monotonically.
_FEEDBACK_GAIN = 0.80


def _peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * math.log10(max(peak, _EPS))


def _fdn_process(xin: np.ndarray, delay_samples: list[int], damping: float, gain: float) -> np.ndarray:
    """Run one channel through the 4-line FDN. Returns the wet tap, same length as xin.

    Read-then-overwrite-same-slot circular buffers: `buffers[k][ptr[k]]` holds
    the sample written `delay_samples[k]` steps ago (its current output);
    immediately after reading it, that slot is overwritten with the new
    value entering the line (input + mixed feedback), which is exactly a
    fixed-length delay line.
    """
    n = len(xin)
    n_lines = len(delay_samples)
    buffers = [np.zeros(d) for d in delay_samples]
    ptr = [0] * n_lines
    filt_state = np.zeros(n_lines)
    damped = np.zeros(n_lines)
    wet = np.zeros(n)

    for i in range(n):
        for k in range(n_lines):
            damped[k] = buffers[k][ptr[k]]
        # One-pole damping filter in the feedback path: more damping keeps
        # more of the PREVIOUS (already-damped) state, i.e. removes more
        # high-frequency energy on every pass through a line.
        filt_state[:] = (1.0 - damping) * damped + damping * filt_state
        damped[:] = filt_state
        fb = gain * (_HADAMARD4 @ damped)
        wet[i] = float(damped.sum()) / n_lines
        xi = xin[i]
        for k in range(n_lines):
            buffers[k][ptr[k]] = xi + fb[k]
            ptr[k] = ptr[k] + 1
            if ptr[k] >= delay_samples[k]:
                ptr[k] = 0
    return wet


def reverb(
    x: np.ndarray,
    sr: int,
    *,
    room_size: float = 0.5,
    damping: float = 0.35,
    wet: float = 0.15,
    dry: float = 0.85,
    pre_delay_ms: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Feedback-delay-network reverb: controlled ambience, not a cathedral.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        room_size: 0.0-1.0. Scales the FDN's delay-line lengths; the primary
            decay-time control (see module docstring for why it, not the
            feedback gain, is what's varied).
        damping: 0.0-1.0. One-pole low-pass coefficient in the feedback
            path; higher darkens (and somewhat shortens) the tail.
        wet: Reverb signal level in the output mix.
        dry: Original signal level in the output mix.
        pre_delay_ms: Delay before the reverb tail begins, relative to the
            dry signal. >= 0.

        `wet`/`dry` are a plain linear mix, not renormalized -- a caller
        supplying wet + dry > 1 gets a hotter output; the documented
        "never clips a sane input" guarantee holds for wet + dry <= 1.

    Returns:
        (y, stats). y has the same shape as x. stats includes the resolved
        delay-line lengths, the estimated decay time implied by them and
        the (fixed) feedback gain, and peak levels.

    Raises:
        ValueError: pre_delay_ms < 0, or room_size/damping outside [0, 1].
    """
    if not math.isfinite(pre_delay_ms) or pre_delay_ms < 0.0:
        raise ValueError(f"pre_delay_ms must be a finite number >= 0, got {pre_delay_ms!r}")
    if not (0.0 <= room_size <= 1.0):
        raise ValueError(f"room_size must be between 0.0 and 1.0, got {room_size!r}")
    if not (0.0 <= damping <= 1.0):
        raise ValueError(f"damping must be between 0.0 and 1.0, got {damping!r}")

    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n_in, n_ch = x2.shape

    pre_delay_samples = round(pre_delay_ms / 1000.0 * sr)
    scale = 0.3 + 1.4 * room_size
    delay_samples = [max(1, round(ms / 1000.0 * sr * scale)) for ms in _BASE_DELAYS_MS]

    wet_channels = []
    for ch in range(n_ch):
        xin = x2[:, ch]
        xin_padded = np.concatenate([np.zeros(pre_delay_samples), xin]) if pre_delay_samples else xin
        wet_full = _fdn_process(xin_padded, delay_samples, damping, _FEEDBACK_GAIN)
        wet_channels.append(wet_full[pre_delay_samples : pre_delay_samples + n_in])
    wet_signal = np.stack(wet_channels, axis=1)

    y2 = dry * x2 + wet * wet_signal
    y = y2[:, 0] if squeeze else y2

    mean_delay_s = float(np.mean(delay_samples)) / sr
    estimated_decay_s = -3.0 * mean_delay_s / math.log10(_FEEDBACK_GAIN)

    stats = {
        "room_size": float(room_size),
        "damping": float(damping),
        "wet": float(wet),
        "dry": float(dry),
        "pre_delay_ms": float(pre_delay_ms),
        "delay_samples": [int(d) for d in delay_samples],
        "feedback_gain": _FEEDBACK_GAIN,
        "estimated_decay_s": float(estimated_decay_s),
        "output_peak_dbfs": _peak_dbfs(y),
    }
    return y, stats


def _load_ir(ir_path_or_array: Any, target_sr: int) -> tuple[np.ndarray, int, bool]:
    """Load an IR as (n_samples, n_channels) float64 at target_sr.

    A path is read through `aud.dsp.io` (a sibling dsp module, not core/io
    -- see docs/ARCHITECTURE.md #6); an array is used directly. If the IR's
    native sample rate differs from `target_sr` it is resampled -- never
    silently applied at the wrong rate, which would detune every reflection
    in the tail.
    """
    if isinstance(ir_path_or_array, (str, Path)):
        from aud.dsp import io as _io

        ir, ir_sr = _io.read_audio(ir_path_or_array)
    else:
        ir = np.asarray(ir_path_or_array, dtype=np.float64)
        if ir.ndim == 1:
            ir = ir[:, None]
        ir_sr = target_sr

    resampled = ir_sr != target_sr
    if resampled:
        from fractions import Fraction

        frac = Fraction(target_sr, ir_sr).limit_denominator(2000)
        ir = signal.resample_poly(ir, up=frac.numerator, down=frac.denominator, axis=0)
    return np.asarray(ir, dtype=np.float64), ir_sr, resampled


def convolve_ir(
    x: np.ndarray,
    sr: int,
    ir_path_or_array: Any,
    *,
    wet: float = 0.15,
    dry: float = 0.85,
) -> tuple[np.ndarray, dict]:
    """Convolution reverb: apply a recorded impulse response via FFT convolution.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        ir_path_or_array: Path to an IR audio file, or an ndarray of shape
            (n_ir_samples,) or (n_ir_samples, n_ir_channels).
        wet: Convolved signal level in the output mix.
        dry: Original signal level in the output mix.

    Returns:
        (y, stats). y has the same shape as x (the convolution tail is
        truncated at the buffer boundary -- see module docstring). stats
        reports the IR's length and native sample rate, whether it needed
        resampling to match `sr`, and peak levels.

    IR channel handling: if the IR has the same channel count as x, each
    channel is convolved with its matching IR channel (a genuine stereo
    IR). Otherwise the IR is downmixed to mono and that one impulse
    response is applied identically to every channel of x.

    The IR is energy-normalized (divided by its own RMS) before
    convolution, so the wet level is governed by `wet`, not by whatever
    gain the IR file happened to be recorded at.
    """
    ir, ir_sr, resampled = _load_ir(ir_path_or_array, sr)
    ir_rms = float(np.sqrt(np.mean(ir * ir)))
    ir = ir / max(ir_rms, _EPS)

    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x
    n_in, n_ch = x2.shape

    if ir.shape[1] == n_ch:
        ir_channels = [ir[:, c] for c in range(n_ch)]
    else:
        # Analysis-only mono fold -- see aud.dsp.channels.fold_to_mono's docstring.
        ir_mono = fold_to_mono(ir)
        ir_channels = [ir_mono for _ in range(n_ch)]

    wet_channels = []
    for ch in range(n_ch):
        wet_full = signal.fftconvolve(x2[:, ch], ir_channels[ch], mode="full")
        wet_channels.append(wet_full[:n_in])
    wet_signal = np.stack(wet_channels, axis=1)

    y2 = dry * x2 + wet * wet_signal
    y = y2[:, 0] if squeeze else y2

    stats = {
        "ir_length_samples": int(ir.shape[0]),
        "ir_sample_rate": int(ir_sr),
        "resampled": resampled,
        "wet": float(wet),
        "dry": float(dry),
        "output_peak_dbfs": _peak_dbfs(y),
    }
    return y, stats
