"""Reduce a room's reverberant tail via decaying-envelope spectral subtraction.

WHY this technique. Single-channel (blind) dereverberation has one hard,
well-known limit: with only one microphone signal and no model of the room
or the source, there is no way to perfectly separate "direct sound" from
"the same sound arriving late, having bounced off the walls" -- the two are
the same signal, delayed and attenuated. Every practical single-channel
method is therefore a *statistical* approximation, not a reconstruction.
This module implements one of the simplest defensible members of that
family: assume the reverberant tail's power decays exponentially with a
fixed, documented time constant (`tau_ms`), track that decaying envelope
per frequency bin, and subtract an amount of it -- capped by `amount_db` --
from each STFT frame's magnitude, with a floor (so a bin is never fully
zeroed) and smoothing across both time and frequency (so the result is not
musical-noise artifacts, which is what unsmoothed spectral subtraction
produces).

Two design choices earn their own explanation:

1. **The tail estimator is a delayed, decaying peak-hold, not a running
   average or an unbounded accumulator.** An early version of this module
   used a leaky-integrator accumulator (`tail = decay * (tail + power)`);
   it was measurably wrong -- because that formulation's steady state is
   *larger* than the average input power for any signal, it suppressed
   perfectly dry, continuous material almost as hard as it suppressed
   genuine reverb. A decaying peak-hold does not have that failure mode:
   it can never exceed the loudest recent frame, and it decays back down
   between transients exactly like a real reverberant tail does.
2. **The estimator looks `guard_ms` into the past, not one frame back.**
   Without that guard delay, a sustained sound's own most recent energy
   gets compared against itself and treated as "reverberant tail," self-
   suppressing legitimate ongoing direct sound. `guard_ms` approximates the
   direct/early-reflection cutoff real room-acoustics models use (typically
   ~50-100 ms): only energy older than that is assumed to be late
   reverberation rather than the sound currently still happening.

Honesty about what this can and cannot do (see also docs/VISION.md's
general stance on overclaiming): `tau_ms` and `guard_ms` are FIXED
assumptions about "a typical room," not measured from the input, because
this module does not attempt blind T60/RT60 estimation. A room whose
actual decay is much longer than `tau_ms` will have its tail only partly
tracked (heavier settings trade more suppression for more risk of
artifacts and more damage to sustained, non-reverberant material -- there is
no setting that removes a heavy reverb cleanly). A room much shorter than
`tau_ms` will see the *tail* over-tracked but the floor/cap still bounds
the damage. This is a moderate-improvement tool for a moderately live room,
not a heavy-reverb remover, and its own report says only what it measured
applying gain, never a claim about how "dry" the result now sounds.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

__all__ = ["dereverb"]

_EPS = 1e-12

# Fixed, documented assumptions (see module docstring) -- not exposed via
# the plan contract (contracts/plan.v1.md's dereverb stage has one
# parameter, amount_db), so they are internal defaults rather than
# per-call-site knobs a plan document could drift on.
_TAU_MS = 500.0
_GUARD_MS = 80.0
_NPERSEG = 2048
_FREQ_SMOOTH_BINS = 3
_TIME_SMOOTH_FRAMES = 3


def _dereverb_channel(
    x: np.ndarray,
    sr: int,
    amount_db: float,
    tau_ms: float,
    guard_ms: float,
    nperseg: int,
    freq_smooth_bins: int,
    time_smooth_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one channel. Returns (y, gain_power) for stats aggregation."""
    nperseg = min(nperseg, max(64, x.shape[0]))
    noverlap = nperseg - max(1, nperseg // 4)
    hop = nperseg - noverlap
    hop_s = hop / sr

    _freqs, _times, zxx = signal.stft(
        x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True
    )
    power = np.abs(zxx) ** 2
    n_frames = power.shape[1]

    # Power (amplitude-squared) decay factor per hop, from the assumed
    # amplitude time constant tau_ms: |h(t)| ~ exp(-t/tau) => |h(t)|^2 ~
    # exp(-2t/tau).
    decay = float(np.exp(-2.0 * hop_s / (tau_ms / 1000.0)))
    guard_frames = max(1, round((guard_ms / 1000.0) / hop_s))

    tail = np.zeros_like(power)
    for m in range(1, n_frames):
        source_index = m - 1 - guard_frames
        source_power = power[:, source_index] if source_index >= 0 else 0.0
        tail[:, m] = decay * np.maximum(tail[:, m - 1], source_power)

    # Gain floor in the power domain: amount_db is the MAXIMUM reduction
    # (contracts/plan.v1.md), so a bin whose estimated tail dominates its
    # current power is cut to (at most) -amount_db, never further.
    gain_floor_power = 10.0 ** (-amount_db / 10.0)
    ratio = np.clip(tail / np.maximum(power, _EPS), 0.0, 1.0)
    gain_power = 1.0 - ratio * (1.0 - gain_floor_power)

    # Smoothing across frequency and time: unsmoothed per-bin, per-frame
    # spectral subtraction gain produces "musical noise" -- isolated bins
    # popping in and out of suppression from frame to frame. A small
    # moving-average window in both axes removes that without materially
    # blunting the suppression itself.
    if freq_smooth_bins > 1:
        gain_power = ndimage.uniform_filter1d(gain_power, size=freq_smooth_bins, axis=0, mode="nearest")
    if time_smooth_frames > 1:
        gain_power = ndimage.uniform_filter1d(gain_power, size=time_smooth_frames, axis=1, mode="nearest")
    gain_power = np.clip(gain_power, gain_floor_power, 1.0)

    gain_amplitude = np.sqrt(gain_power)
    zxx_out = zxx * gain_amplitude
    _, y = signal.istft(zxx_out, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=True)

    n_in = x.shape[0]
    if y.shape[0] > n_in:
        y = y[:n_in]
    elif y.shape[0] < n_in:
        y = np.concatenate([y, np.zeros(n_in - y.shape[0])])
    return y, gain_power


def dereverb(
    x: np.ndarray,
    sr: int,
    *,
    amount_db: float = 6.0,
    tau_ms: float = _TAU_MS,
    guard_ms: float = _GUARD_MS,
    nperseg: int = _NPERSEG,
    freq_smooth_bins: int = _FREQ_SMOOTH_BINS,
    time_smooth_frames: int = _TIME_SMOOTH_FRAMES,
) -> tuple[np.ndarray, dict]:
    """Reduce estimated late-reverberation energy, per channel, by up to `amount_db`.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        amount_db: Maximum reduction applied to the estimated reverberant
            component, in dB (contracts/plan.v1.md's `dereverb.amount_db`).
            `0.0` is a no-op (gain stays at 1.0 everywhere); larger values
            trade more suppression for more risk of audibly damaging
            sustained, non-reverberant material -- see module docstring.
        tau_ms: Assumed reverberant-tail amplitude decay time constant.
            Fixed, not measured from the signal (see module docstring).
        guard_ms: Assumed direct/early-reflection cutoff: only frames older
            than this are treated as candidate "late reverberation".
        nperseg: STFT frame length in samples.
        freq_smooth_bins: Moving-average window (bins) smoothing the gain
            mask across frequency, to suppress musical noise.
        time_smooth_frames: Moving-average window (frames) smoothing the
            gain mask across time, to suppress musical noise.

    Returns:
        (y, stats) where stats = {
            "amount_db": float, "tau_ms": float, "guard_ms": float,
            "max_gain_reduction_db": float, "avg_gain_reduction_db": float,
            "bins_reduced_pct": float,  # % of time-frequency bins touched
        }

    Raises:
        ValueError: `amount_db` is negative, or `tau_ms`/`guard_ms` are not
            positive.
    """
    if amount_db < 0.0:
        raise ValueError(f"amount_db must be >= 0, got {amount_db}")
    if tau_ms <= 0.0:
        raise ValueError(f"tau_ms must be > 0, got {tau_ms}")
    if guard_ms <= 0.0:
        raise ValueError(f"guard_ms must be > 0, got {guard_ms}")

    x = np.asarray(x, dtype=np.float64)
    mono_input = x.ndim == 1
    if mono_input:
        x = x[:, None]

    n_channels = x.shape[1]
    processed = []
    gain_power_all = []
    for ch in range(n_channels):
        y_ch, gain_power = _dereverb_channel(
            x[:, ch], sr, amount_db, tau_ms, guard_ms, nperseg, freq_smooth_bins, time_smooth_frames
        )
        processed.append(y_ch)
        gain_power_all.append(gain_power)

    y = np.stack(processed, axis=1)
    if mono_input:
        y = y[:, 0]

    gain_power_stack = np.concatenate([g.ravel() for g in gain_power_all])
    gain_db = 10.0 * np.log10(np.maximum(gain_power_stack, _EPS))
    reduced_mask = gain_db < -0.05

    stats = {
        "amount_db": float(amount_db),
        "tau_ms": float(tau_ms),
        "guard_ms": float(guard_ms),
        "max_gain_reduction_db": float(np.min(gain_db)) if gain_db.size else 0.0,
        "avg_gain_reduction_db": float(np.mean(gain_db)) if gain_db.size else 0.0,
        "bins_reduced_pct": float(100.0 * np.mean(reduced_mask)) if reduced_mask.size else 0.0,
    }
    return y, stats
