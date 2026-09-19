"""EQ-match: measure a reference's spectral shape, then correct toward it.

Two functions, two different callers:

`spectrum_profile(x, sr, ...)` measures a signal's own averaged, smoothed
spectral curve -- a small set of `[freq_hz, gain_db]` points, JSON-safe, and
enough on its own to describe "what this recording sounds like tonally"
without keeping the recording around. This is what `aud curve extract`
writes to disk, and what `aud eq-match --reference` measures at plan-build
time (see contracts/plan.v1.md's `eq_match` stage and its note: "curve is
measurements, not a file pointer" -- a stored plan must replay identically
with no access to the reference file, the same reasoning `strip_silence`
follows for its policy rather than positions).

`apply_curve(x, sr, curve, ...)` is the correction: it measures `x`'s own
current profile, diffs it against the stored `curve`, clamps the diff per
band, scales it by `strength`, and designs a **linear-phase** FIR from the
result via `scipy.signal.firwin2`. Linear phase, not a biquad cascade or a
zero-phase (`filtfilt`) design, and deliberately so: `dsp/filters.py`'s
biquads are for a mastering chain with real-time attack/release elsewhere
downstream, where a fixed, small phase response is expected and desired.
An EQ-match is a one-shot tone correction against a finished programme --
there is nothing downstream whose timing it needs to track -- so the only
thing that matters is not smearing the signal's transients asymmetrically
in time. A minimum-phase or biquad design would do exactly that (more delay
at some frequencies than others); an FIR built by `firwin2` is symmetric by
construction, so every frequency is delayed by the same constant amount
(`(numtaps - 1) / 2` samples) rather than by a frequency-dependent amount.
A constant delay is inaudible; smeared relative timing is what "phase
smearing" means and is exactly what this module exists to avoid.

Three things keep this from being a naive "make the spectra equal" filter:

1. `strength` is a dial, not a switch. `0.0` returns the input completely
   unchanged (no filtering happens at all); `1.0` applies the full measured
   correction; values between are a fraction of it, and the spectral
   distance to the reference decreases monotonically as `strength` rises.
2. `max_boost_db` / `max_cut_db` clamp the correction actually computed, in
   each band, in each direction, independently. Without a clamp, a band
   where either signal sits at its own measurement noise floor produces an
   enormous, meaningless correction -- boosting a silent band of the target
   up toward a reference's real content turns that target's noise floor
   into audible hiss, and the mirror case (a silent band in the reference)
   asks for an equally meaningless, unbounded cut. The clamp bounds both.
3. The curve is measurements (frequency/gain pairs), never a path to the
   reference file, so it survives being written to JSON, read back on a
   different machine, or replayed as part of a stored plan a year later.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
from scipy import signal

__all__ = ["CurveError", "apply_curve", "spectrum_profile"]


class CurveError(ValueError):
    """A `curve` argument is malformed, or incompatible with this signal's sample rate.

    Subclasses `ValueError` rather than `AudError` -- `dsp/` modules take
    arrays and parameters and return arrays; they do not raise user-facing
    errors (AGENTS.md #8). A dedicated subclass (rather than a bare
    `ValueError`) lets `aud.lib` catch curve-shape/compatibility problems
    specifically and report them as `bad_param`, without masking an
    unrelated `ValueError` from elsewhere in the chain as the wrong thing.
    """


_EPS = 1e-12
_MIN_BAND_HZ = 20.0
_NYQUIST_GUARD = 0.999  # keep the top band/design point strictly inside Nyquist
_DEFAULT_FFT_SIZE = 8192
_DEFAULT_FRACTION_OCTAVE = 3
_DEFAULT_FIR_TAPS = 2049


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else np.mean(x, axis=1)


def _band_edges(sr: int, fraction_octave: int) -> list[float]:
    """Fractional-octave band edges from ~20 Hz up to just under Nyquist."""
    nyquist = sr / 2.0
    top = nyquist * _NYQUIST_GUARD
    step = 2.0 ** (1.0 / fraction_octave)
    edges = [_MIN_BAND_HZ]
    f = _MIN_BAND_HZ * step
    while f < top:
        edges.append(f)
        f *= step
    edges.append(top)
    return edges


def spectrum_profile(
    x: np.ndarray,
    sr: int,
    *,
    fft_size: int = _DEFAULT_FFT_SIZE,
    fraction_octave: int = _DEFAULT_FRACTION_OCTAVE,
) -> dict[str, Any]:
    """Measure a signal's averaged, fractional-octave-smoothed spectral curve.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,). Channels
            are averaged to mono before measurement -- tonal balance is a
            property of the programme, not of one channel.
        sr: Sample rate in Hz.
        fft_size: Welch segment length used for the underlying power
            spectral density estimate. Longer resolves low frequencies more
            finely; shorter averages more segments for a smoother estimate.
        fraction_octave: Bands per octave the raw PSD is smoothed into
            (3 = third-octave, the audio-engineering default).

    Returns:
        A JSON-serializable, self-contained dict:
            {"sample_rate": int, "fft_size": int, "fraction_octave": int,
             "curve": [[freq_hz, gain_db], ...]}
        `curve` always has at least 2 points (contracts/plan.v1.md's
        `eq_match.curve` requires it), ascending by frequency, and is
        replayable by `apply_curve` with no access to `x` ever again.
    """
    mono = _mono(x)
    nperseg = min(fft_size, mono.shape[0]) if mono.shape[0] else 1
    nperseg = max(nperseg, 1)
    freqs, psd = signal.welch(mono, fs=sr, nperseg=nperseg, window="hann", detrend=False)
    psd = np.maximum(psd, _EPS)

    edges = _band_edges(sr, fraction_octave)
    curve: list[list[float]] = []
    for lo, hi in pairwise(edges):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band_power_db = 10.0 * math.log10(float(np.mean(psd[mask])))
        center_hz = float(math.sqrt(lo * hi))
        curve.append([center_hz, band_power_db])

    if len(curve) < 2:
        # A degenerate/very short input still owes the contract's "at least
        # 2 points" -- fall back to a flat curve spanning the whole range
        # rather than raising, since this is a measurement, not validation.
        overall_db = 10.0 * math.log10(float(np.mean(psd)))
        nyquist = sr / 2.0
        curve = [[_MIN_BAND_HZ, overall_db], [nyquist * _NYQUIST_GUARD, overall_db]]

    return {
        "sample_rate": int(sr),
        "fft_size": int(fft_size),
        "fraction_octave": int(fraction_octave),
        "curve": curve,
    }


def _extract_pairs(curve: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pull validated, ascending (freqs, gains_db) arrays out of a curve.

    Accepts either the rich dict `spectrum_profile` returns (a "curve" key
    holding the pairs, plus measurement metadata) or a bare array of pairs
    -- the shape contracts/plan.v1.md's `eq_match.curve` stage field
    actually stores. Raises `CurveError` (a `ValueError` subclass -- `dsp/`
    modules do not raise `AudError`, AGENTS.md #8) so `aud.lib` can turn a
    bad curve into a user-facing `bad_param` without masking an unrelated
    `ValueError` from elsewhere in the chain.
    """
    pairs = curve.get("curve") if isinstance(curve, dict) else curve
    if not isinstance(pairs, (list, tuple)) or len(pairs) < 2:
        raise CurveError("curve must contain at least 2 [freq_hz, gain_db] pairs")

    freqs: list[float] = []
    gains: list[float] = []
    previous_freq = -math.inf
    for point in pairs:
        if len(point) != 2:
            raise CurveError(f"curve point {point!r} is not a [freq_hz, gain_db] pair")
        freq_hz, gain_db = float(point[0]), float(point[1])
        if not (math.isfinite(freq_hz) and math.isfinite(gain_db)):
            raise CurveError(f"curve point {point!r} must be finite")
        if freq_hz <= 0:
            raise CurveError(f"curve frequency must be positive, got {freq_hz}")
        if freq_hz <= previous_freq:
            raise CurveError(f"curve frequencies must be strictly ascending, got {freq_hz} after {previous_freq}")
        previous_freq = freq_hz
        freqs.append(freq_hz)
        gains.append(gain_db)
    return np.array(freqs, dtype=np.float64), np.array(gains, dtype=np.float64)


def _interp_db(freqs_query: np.ndarray, curve_freqs: np.ndarray, curve_db: np.ndarray) -> np.ndarray:
    """Log-frequency interpolation, held flat beyond the curve's own edges.

    Tonal features are perceived and measured log-spaced (an octave is a
    ratio, not a fixed Hz span), so interpolating in log-frequency matches
    how the fractional-octave curve itself was built. A curve that does not
    span the target's full range (e.g. a band-limited reference) is
    deliberately NOT extrapolated as a slope -- it is held flat at its
    nearest known point, so a silent region of the reference is measured as
    "flat at whatever its edge measured", not invented.
    """
    log_curve_freqs = np.log(curve_freqs)
    log_query = np.log(np.maximum(freqs_query, _EPS))
    return np.interp(log_query, log_curve_freqs, curve_db, left=curve_db[0], right=curve_db[-1])


def apply_curve(
    x: np.ndarray,
    sr: int,
    curve: Any,
    *,
    strength: float = 1.0,
    max_boost_db: float = 12.0,
    max_cut_db: float = 12.0,
    numtaps: int = _DEFAULT_FIR_TAPS,
) -> np.ndarray:
    """Correct `x` toward a stored spectral curve, by a bounded, dialable amount.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz of `x`.
        curve: A `spectrum_profile`-shaped dict, or a bare array of
            `[freq_hz, gain_db]` pairs (contracts/plan.v1.md's `eq_match`
            stage stores the latter directly). Curve frequencies are
            absolute Hz, so a curve measured at one sample rate applies
            cleanly to audio at another, as long as every curve frequency
            is below the new signal's own Nyquist -- see `Raises` below.
        strength: `0.0` returns `x` completely unchanged (no measurement,
            no filtering); `1.0` applies the full clamped correction.
            Monotonic in between: the spectral distance to `curve` shrinks
            as `strength` rises.
        max_boost_db: Clamp on the correction in the boost direction, per
            band, before scaling by `strength`. Never `None` -- see the
            module docstring for why an unclamped correction is unsafe.
        max_cut_db: Clamp on the correction in the cut direction, per band,
            before scaling by `strength`.
        numtaps: FIR length. Forced to the next odd number if given even --
            `scipy.signal.firwin2` requires an odd tap count to allow a
            nonzero gain at the Nyquist frequency, which this filter's
            gain curve generally has.

    Returns:
        A new array, same shape and dtype (float64) as `x`.

    Raises:
        CurveError: `curve` has fewer than 2 points, a non-ascending or
            non-finite point, or its lowest frequency is at or above this
            signal's own Nyquist frequency (nothing of it could apply).
            `strength`/`max_boost_db`/`max_cut_db` outside their domains
            raise the same type, for one exception type a caller can catch.
    """
    x = np.asarray(x, dtype=np.float64)
    if not math.isfinite(strength) or strength < 0.0:
        raise CurveError(f"strength must be a finite number >= 0, got {strength!r}")
    if strength == 0.0:
        return x.copy()
    if not math.isfinite(max_boost_db) or max_boost_db < 0.0:
        raise CurveError(f"max_boost_db must be a finite number >= 0, got {max_boost_db!r}")
    if not math.isfinite(max_cut_db) or max_cut_db < 0.0:
        raise CurveError(f"max_cut_db must be a finite number >= 0, got {max_cut_db!r}")

    curve_freqs, curve_db = _extract_pairs(curve)
    nyquist = sr / 2.0
    if curve_freqs[0] >= nyquist:
        raise CurveError(
            f"curve's lowest frequency ({curve_freqs[0]} Hz) is at or above this signal's Nyquist "
            f"frequency ({nyquist} Hz) at sample rate {sr}; none of it can be applied"
        )

    own_freqs, own_db = _extract_pairs(spectrum_profile(x, sr)["curve"])

    target_db = _interp_db(own_freqs, curve_freqs, curve_db)
    diff_db = np.clip(target_db - own_db, -max_cut_db, max_boost_db) * strength

    # firwin2 requires freq[0] == 0 and freq[-1] == fs/2 exactly; anchor the
    # gain curve at those two design points, holding it flat from the
    # nearest measured band (the same "no invented slope" rule as _interp_db).
    design_freqs = np.concatenate(([0.0], own_freqs, [nyquist]))
    design_gain_db = np.concatenate(([diff_db[0]], diff_db, [diff_db[-1]]))
    design_freqs, unique_idx = np.unique(design_freqs, return_index=True)
    design_gain_db = design_gain_db[unique_idx]
    gain_linear = 10.0 ** (design_gain_db / 20.0)

    taps = numtaps if numtaps % 2 == 1 else numtaps + 1
    fir = signal.firwin2(taps, design_freqs, gain_linear, fs=sr)

    if x.ndim == 1:
        return signal.lfilter(fir, [1.0], x)
    return signal.lfilter(fir, [1.0], x, axis=0)
