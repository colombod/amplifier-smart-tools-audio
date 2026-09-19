"""Dynamic de-esser: reduce sibilance only where it is actually loud.

This is NOT a static notch/shelf. A fixed EQ cut at the sibilant band would
dull every "s" -- and every non-sibilant sound sharing that band (cymbals,
"f", breath noise) -- for the whole file, all the time. A de-esser instead
watches the energy inside a narrow band and only pulls it down when that
band is *currently* louder than the rest of the programme; the moment the
sibilant burst ends, the gain returns to unity. The defining, testable
difference: a quiet passage with no sibilance at all must come out
essentially bit-identical (up to the crossover network's own reconstruction
floor -- see below), while a hot "s" gets pulled down toward `amount_db`.

Topology, in order:

1. Split the signal into three bands at `band`'s two edges using the same
   Linkwitz-Riley crossover this tool already uses for multiband
   compression (`aud.dsp.crossover`) -- reused rather than re-derived, per
   this module's design brief. Only the middle (sibilant) band is touched;
   the low and high bands pass straight through.
2. Track a continuous, mono (channel-summed), short-time RMS-ish envelope
   of the sibilant band's level, in dB.
3. Compute a soft-knee gain reduction that ramps from 0 dB at
   `threshold_db` to a hard ceiling of `-amount_db` over `ramp_db` of
   headroom, and stays capped at `-amount_db` beyond that -- so the caller's
   `amount_db` is a true, honoured maximum, never exceeded.
4. Apply a forward-looking (lookahead) minimum so the gain has already
   started dropping *before* a fast sibilant transient's peak arrives
   (identical in spirit to `aud.dsp.limiter`'s lookahead), then smooth the
   *release* back toward 0 dB with a one-pole filter. There is no separate
   attack smoothing: the lookahead already delivers an effectively
   instantaneous attack, exactly as the limiter's docstring explains for
   the same construction.
5. Recombine the three bands.

A note on the "silence in, silence out" property and the 0.1 dB tolerance
used by this module's tests: `aud.dsp.crossover`'s own docstring is explicit
that a 3-band (two-crossover) recursive split/recombine is a *magnitude*-flat
network, not a sample-for-sample identity one -- phase is not preserved
tap-for-tap. That means even a perfectly untouched band (gain exactly 1.0
throughout) will not reconstruct as bit-identical to the input in the time
domain; it reconstructs as level-identical. Tests here therefore compare
level in dB (RMS), the same methodology `tests/test_dsp_crossover.py` uses
for exactly this reason, not raw sample subtraction.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from aud.dsp import crossover

__all__ = ["deess"]

_EPS = 1e-12

#: Default sibilant band. Centred a little above 6 kHz, which is where
#: sibilance ("s", "sh", "t", "f") concentrates in most spoken-word
#: recordings; a full octave wide (edges at center/sqrt(2) and
#: center*sqrt(2)) so the band comfortably contains the energy without
#: also grabbing most of the vocal fundamental/formant range below it.
_DEFAULT_BAND: tuple[float, float] = (4500.0, 9000.0)


def _band_level_db(band: np.ndarray, sr: int, smoothing_ms: float = 6.0) -> np.ndarray:
    """Continuous, mono, short-time RMS-ish level of `band`, in dB, per sample.

    Channels are summed (mono-linked detection) for the same reason
    `aud.dsp.dynamics`'s compressor stereo-links by default: an
    independent per-channel decision would duck one channel's sibilance
    and not the other's, shifting the stereo image on every "s".
    """
    power = band * band
    if power.ndim == 2:
        power = np.mean(power, axis=1) if power.shape[1] > 1 else power[:, 0]
    coeff = np.exp(-1.0 / (sr * smoothing_ms / 1000.0))
    smoothed_power = signal.lfilter([1.0 - coeff], [1.0, -coeff], power)
    return 20.0 * np.log10(np.sqrt(np.maximum(smoothed_power, _EPS)))


def _lookahead_min(x: np.ndarray, window: int) -> np.ndarray:
    """gain_la[n] = min(x[n : n + window]) -- see aud.dsp.limiter._lookahead_min.

    Duplicated rather than imported: it is eight lines, `limiter`'s copy is
    private, and the two callers apply it to different signals (true-peak
    linear gain there, gain-reduction dB here) for different reasons, so
    sharing it would buy a common helper at the cost of coupling two
    otherwise-independent dsp modules to each other's internals.
    """
    if window <= 1:
        return x
    origin = -(window // 2)
    return ndimage.minimum_filter1d(x, size=window, mode="nearest", origin=origin)


def _release_smooth_db(gr_db: np.ndarray, sr: int, release_ms: float) -> np.ndarray:
    """One-pole *release-only* smoothing: react instantly to a deeper cut

    (the lookahead already anticipated it), ease back toward 0 dB no faster
    than `release_ms`. Same construction as `aud.dsp.limiter.brickwall`'s
    gain smoothing, for the same reason: the lookahead minimum has already
    done the "attack", so only the recovery needs a time constant.
    """
    release_coeff = np.exp(-1.0 / (sr * max(release_ms, 1e-3) / 1000.0))
    smoothed = np.empty_like(gr_db)
    prev = 0.0
    for n in range(gr_db.shape[0]):
        target = gr_db[n]
        prev = target if target < prev else release_coeff * prev + (1 - release_coeff) * target
        smoothed[n] = prev
    return smoothed


def deess(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -35.0,
    amount_db: float = 6.0,
    band: tuple[float, float] = _DEFAULT_BAND,
    lookahead_ms: float = 3.0,
    release_ms: float = 60.0,
    ramp_db: float = 6.0,
) -> tuple[np.ndarray, dict]:
    """Dynamically reduce sibilance in `band`, by up to `amount_db`.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        threshold_db: Sibilant-band level (dB, mono-summed RMS-ish) above
            which gain reduction starts. Below this, gain is exactly 0 dB.
        amount_db: Maximum gain reduction applied in the sibilant band, in
            dB. Never exceeded, regardless of how far above threshold the
            band gets -- this is a ceiling, not a ratio.
        band: (low_hz, high_hz) edges of the sibilant band. Must satisfy
            0 < low_hz < high_hz < Nyquist.
        lookahead_ms: How far ahead the gain-reduction detector looks so
            gain has already started dropping before a fast sibilant
            transient's peak arrives.
        release_ms: Time constant for gain recovering back toward 0 dB
            after the sibilant band quiets down.
        ramp_db: How many dB above `threshold_db` it takes to reach the
            full `amount_db` reduction (a soft ramp, not an instant step).

    Returns:
        (y, stats) where stats = {
            "band_low_hz": float, "band_high_hz": float,
            "threshold_db": float, "amount_db": float,
            "max_gain_reduction_db": float, "avg_gain_reduction_db": float,
            "frames_reduced_pct": float,  # % of samples with any reduction
        }

    Raises:
        ValueError: `band` is not `0 < low < high < Nyquist`, or
            `amount_db` is negative.
    """
    x = np.asarray(x, dtype=np.float64)
    mono_input = x.ndim == 1
    if mono_input:
        x = x[:, None]

    low, high = band
    nyquist = sr / 2.0
    if not (0.0 < low < high):
        raise ValueError(f"band=({low}, {high}) must satisfy 0 < low < high")
    if high >= nyquist:
        raise ValueError(f"band high edge {high} Hz must be below Nyquist ({nyquist} Hz) for sr={sr}")
    if amount_db < 0.0:
        raise ValueError(f"amount_db must be >= 0, got {amount_db}")

    below, sibilant, above = crossover.split(x, sr, [low, high])

    level_db = _band_level_db(sibilant, sr)
    excess_db = level_db - threshold_db
    fraction = np.clip(excess_db / max(ramp_db, _EPS), 0.0, 1.0)
    target_gr_db = -fraction * amount_db

    window = max(1, round(lookahead_ms / 1000.0 * sr))
    gr_lookahead_db = _lookahead_min(target_gr_db, window)
    gr_smoothed_db = _release_smooth_db(gr_lookahead_db, sr, release_ms)
    # Belt-and-suspenders: the release filter eases TOWARD 0, never away from
    # it, so it cannot overshoot past -amount_db -- but clamp explicitly so
    # amount_db is a documented, mechanically-enforced ceiling, not just an
    # emergent property of this particular smoothing construction.
    gr_smoothed_db = np.maximum(gr_smoothed_db, -amount_db)

    gain_lin = 10.0 ** (gr_smoothed_db / 20.0)
    sibilant_processed = sibilant * gain_lin[:, None]

    y = crossover.recombine([below, sibilant_processed, above])
    if mono_input:
        y = y[:, 0]

    reduced_mask = gr_smoothed_db < -0.05
    stats = {
        "band_low_hz": float(low),
        "band_high_hz": float(high),
        "threshold_db": float(threshold_db),
        "amount_db": float(amount_db),
        "max_gain_reduction_db": float(np.min(gr_smoothed_db)) if gr_smoothed_db.size else 0.0,
        "avg_gain_reduction_db": float(np.mean(gr_smoothed_db)) if gr_smoothed_db.size else 0.0,
        "frames_reduced_pct": float(100.0 * np.mean(reduced_mask)) if reduced_mask.size else 0.0,
    }
    return y, stats
