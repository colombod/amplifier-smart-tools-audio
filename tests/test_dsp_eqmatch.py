"""Behavioural tests for aud.dsp.eqmatch -- synthesised signals, real numbers.

Every test measures something concrete (a spectral distance, a clamp bound,
an exact sample match) and prints the actual numbers with `-s`, in the same
spirit as the repo's other recolor/measurement-style tests: a claim like
"the correction closes the gap" is only worth something with the before/after
numbers attached.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aud.dsp import filters
from aud.dsp.eqmatch import CurveError, apply_curve, spectrum_profile

SR = 44100
DURATION_S = 2.0
N = int(SR * DURATION_S)


def _white_noise(seed: int, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).reshape(-1, 1) * 0.1


def _band_limit_fft(x: np.ndarray, sr: int, cutoff_hz: float, atten_db: float | None = None) -> np.ndarray:
    """Clean band-limit via FFT bin scaling, with no Butterworth stopband ripple.

    `atten_db=None` zeros the bins above `cutoff_hz` outright -- true
    digital silence, used where the test only needs "no real content up
    there" (e.g. matching a band-limited reference). `atten_db` set to a
    finite value instead scales those bins down by that many dB rather
    than to exact zero -- a residual noise floor, not true silence, which
    is what "boosting a near-silent band into hiss" actually means: there
    is nothing to boost out of literal zero (0 x any gain is still 0).
    """
    mono = x[:, 0]
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sr)
    mask = freqs > cutoff_hz
    spectrum[mask] *= 0.0 if atten_db is None else 10.0 ** (-atten_db / 20.0)
    return np.fft.irfft(spectrum, n=mono.shape[0]).reshape(-1, 1)


def _curve_arrays(curve_doc: dict) -> tuple[np.ndarray, np.ndarray]:
    pairs = curve_doc["curve"]
    freqs = np.array([p[0] for p in pairs], dtype=np.float64)
    gains = np.array([p[1] for p in pairs], dtype=np.float64)
    return freqs, gains


def _distance(curve_a: dict, curve_b: dict) -> float:
    """RMS dB distance between two curves, on curve_a's own frequency grid.

    curve_b is interpolated (log-frequency, flat-extrapolated) onto
    curve_a's band centers -- the same interpolation `apply_curve` itself
    uses -- so the metric reflects what the algorithm actually compares.
    """
    freqs_a, db_a = _curve_arrays(curve_a)
    freqs_b, db_b = _curve_arrays(curve_b)
    log_b = np.log(freqs_b)
    log_a = np.log(np.maximum(freqs_a, 1e-12))
    interp_b = np.interp(log_a, log_b, db_b, left=db_b[0], right=db_b[-1])
    return float(np.sqrt(np.mean((db_a - interp_b) ** 2)))


# ---------------------------------------------------------------------------
# The central test: a dull source matched toward a bright reference.
# ---------------------------------------------------------------------------


def test_eq_match_closes_the_spectral_distance_to_the_reference() -> None:
    noise = _white_noise(seed=1)
    # Dull: the same noise, high-shelf CUT above 2 kHz. Bright: the same noise,
    # high-shelf BOOSTED above 2 kHz. A shelf (not a steep lowpass) keeps real
    # content in every band on both sides, so the ~24 dB gap it creates is one
    # a bounded correction can actually close -- a steep lowpass would instead
    # drive the source to its numerical noise floor there, which is the
    # clamping tests' scenario, not this one.
    source = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=-12.0, q=0.7))
    reference = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=12.0, q=0.7))

    reference_curve = spectrum_profile(reference, SR)
    source_curve_before = spectrum_profile(source, SR)

    matched = apply_curve(source, SR, reference_curve, strength=1.0, max_boost_db=30.0, max_cut_db=30.0)
    matched_curve = spectrum_profile(matched, SR)

    distance_before = _distance(reference_curve, source_curve_before)
    distance_after = _distance(reference_curve, matched_curve)
    percent_closed = (distance_before - distance_after) / distance_before * 100.0

    print(
        f"\n[eq_match central test] distance before={distance_before:.3f} dB(rms), "
        f"after={distance_after:.3f} dB(rms), {percent_closed:.1f}% closed"
    )

    assert distance_after < distance_before
    # The correction must close the large majority of the gap, not just nudge it.
    assert percent_closed > 60.0, f"only {percent_closed:.1f}% of the spectral gap closed"


def test_eq_match_strength_is_monotonic() -> None:
    noise = _white_noise(seed=2)
    source = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=-12.0, q=0.7))
    reference = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=12.0, q=0.7))
    reference_curve = spectrum_profile(reference, SR)

    distances = {}
    for strength in (0.0, 0.5, 1.0):
        matched = apply_curve(source, SR, reference_curve, strength=strength, max_boost_db=30.0, max_cut_db=30.0)
        distances[strength] = _distance(reference_curve, spectrum_profile(matched, SR))

    print(
        f"\n[eq_match monotonicity] distance(0.0)={distances[0.0]:.3f} dB, "
        f"distance(0.5)={distances[0.5]:.3f} dB, distance(1.0)={distances[1.0]:.3f} dB"
    )

    assert distances[0.0] > distances[0.5] > distances[1.0]


def test_eq_match_strength_zero_returns_input_unchanged() -> None:
    noise = _white_noise(seed=3)
    source = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=-12.0, q=0.7))
    reference = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=12.0, q=0.7))
    reference_curve = spectrum_profile(reference, SR)

    matched = apply_curve(source, SR, reference_curve, strength=0.0)

    assert np.array_equal(matched, source)


# ---------------------------------------------------------------------------
# Clamping: an empty band must not produce an unbounded correction, in
# either direction.
# ---------------------------------------------------------------------------


def test_eq_match_clamps_boost_when_target_band_is_empty() -> None:
    """The target has only a residual noise floor above 5 kHz; the reference is full-band.

    Without a clamp, matching would try to boost the target's near-silent
    top end all the way up to the reference's real level there -- turning
    that residual noise floor into audible hiss (see the module
    docstring's failure mode #2). `max_boost_db` must cap it. The target's
    empty band is attenuated (not zeroed outright): boosting exact digital
    silence is still silence (0 x any gain is 0), so a *residual* floor is
    the scenario the clamp actually has to guard.
    """
    noise = _white_noise(seed=4)
    target = _band_limit_fft(noise, SR, cutoff_hz=5000.0, atten_db=40.0)
    reference = _white_noise(seed=5)  # full-band, normal level throughout

    reference_curve = spectrum_profile(reference, SR)
    target_curve_before = spectrum_profile(target, SR)

    max_boost_db = 12.0
    matched = apply_curve(target, SR, reference_curve, strength=1.0, max_boost_db=max_boost_db, max_cut_db=30.0)
    matched_curve = spectrum_profile(matched, SR)

    freqs_before, db_before = _curve_arrays(target_curve_before)
    freqs_after, db_after = _curve_arrays(matched_curve)
    freqs_ref, db_ref = _curve_arrays(reference_curve)
    high_mask = freqs_before > 6000.0  # comfortably inside the emptied band

    before_level = float(np.mean(db_before[high_mask]))
    after_level = float(np.mean(db_after[np.array([f > 6000.0 for f in freqs_after])]))
    ref_level = float(np.mean(db_ref[np.array([f > 6000.0 for f in freqs_ref])]))
    applied_boost = after_level - before_level

    print(
        f"\n[eq_match boost clamp] empty-band target before={before_level:.2f} dB, "
        f"reference={ref_level:.2f} dB, target after={after_level:.2f} dB, "
        f"applied boost={applied_boost:.2f} dB (limit={max_boost_db} dB)"
    )

    # The clamp must hold: the correction actually applied must not exceed
    # max_boost_db (with a small tolerance for band-grid interpolation).
    assert applied_boost <= max_boost_db + 1.0
    # It must be a REAL clamp engaging, not a no-op: the uncorrected gap is
    # far larger than what was allowed through.
    assert applied_boost > max_boost_db - 3.0
    # ...and it must still fall well short of the reference: an unclamped
    # correction would have closed almost all the way to it.
    assert after_level < ref_level - 15.0


def test_eq_match_clamps_cut_when_reference_band_is_empty() -> None:
    """The reference has nothing above 5 kHz; the target is full-band.

    Matching would ask for a huge cut in the target's top end to reach the
    reference's near-silence there. `max_cut_db` must cap it.
    """
    noise = _white_noise(seed=6)
    target = _white_noise(seed=7)  # full-band, normal level throughout
    reference = _band_limit_fft(noise, SR, cutoff_hz=5000.0)

    reference_curve = spectrum_profile(reference, SR)
    target_curve_before = spectrum_profile(target, SR)

    max_cut_db = 12.0
    matched = apply_curve(target, SR, reference_curve, strength=1.0, max_boost_db=30.0, max_cut_db=max_cut_db)
    matched_curve = spectrum_profile(matched, SR)

    freqs_before, db_before = _curve_arrays(target_curve_before)
    freqs_after, db_after = _curve_arrays(matched_curve)
    freqs_ref, db_ref = _curve_arrays(reference_curve)
    high_mask_before = freqs_before > 6000.0

    before_level = float(np.mean(db_before[high_mask_before]))
    after_level = float(np.mean(db_after[np.array([f > 6000.0 for f in freqs_after])]))
    ref_level = float(np.mean(db_ref[np.array([f > 6000.0 for f in freqs_ref])]))
    applied_cut = before_level - after_level

    print(
        f"\n[eq_match cut clamp] full-band target before={before_level:.2f} dB, "
        f"empty-band reference={ref_level:.2f} dB, target after={after_level:.2f} dB, "
        f"applied cut={applied_cut:.2f} dB (limit={max_cut_db} dB)"
    )

    assert applied_cut <= max_cut_db + 1.0
    # A real clamp: without it the target would have been cut almost down
    # to the reference's near-silent level.
    assert after_level > ref_level + 20.0


# ---------------------------------------------------------------------------
# The curve is measurements: round-trips through JSON, and travels across
# a sample-rate change cleanly (or refuses loudly -- this build works).
# ---------------------------------------------------------------------------


def test_curve_round_trips_through_json_and_applies_identically() -> None:
    noise = _white_noise(seed=8)
    reference = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=12.0, q=0.7))
    source = filters.apply_sos(noise, filters.high_shelf(SR, 2000.0, gain_db=-12.0, q=0.7))

    curve = spectrum_profile(reference, SR)
    restored = json.loads(json.dumps(curve))
    assert restored == curve

    direct = apply_curve(source, SR, curve, strength=1.0)
    via_json = apply_curve(source, SR, restored, strength=1.0)
    assert np.array_equal(direct, via_json)


def test_curve_measured_at_one_sample_rate_applies_correctly_at_another() -> None:
    """A curve measured at 44100 Hz, applied to a 48000 Hz signal.

    44100 Hz's Nyquist (22050 Hz) is below 48000 Hz's (24000 Hz), so every
    curve frequency is comfortably applicable -- this build makes that case
    work correctly (the alternative sanctioned by spec would be a clear
    refusal; a mismatch that instead silently mis-measured would be worse
    than either).
    """
    sr_reference = 44100
    n_reference = int(sr_reference * DURATION_S)
    noise_44100 = _white_noise(seed=9, n=n_reference)
    reference = filters.apply_sos(noise_44100, filters.high_shelf(sr_reference, 2000.0, gain_db=18.0, q=0.7))
    curve_44100 = spectrum_profile(reference, sr_reference)
    assert curve_44100["sample_rate"] == sr_reference
    assert max(f for f, _ in curve_44100["curve"]) < 48000 / 2.0

    sr_target = 48000
    n_target = int(sr_target * DURATION_S)
    noise_48000 = _white_noise(seed=9, n=n_target)
    target = filters.apply_sos(noise_48000, filters.high_shelf(sr_target, 2000.0, gain_db=-12.0, q=0.7))
    target_curve_before = spectrum_profile(target, sr_target)

    matched = apply_curve(target, sr_target, curve_44100, strength=1.0, max_boost_db=30.0, max_cut_db=30.0)
    assert matched.shape == target.shape
    matched_curve = spectrum_profile(matched, sr_target)

    distance_before = _distance(curve_44100, target_curve_before)
    distance_after = _distance(curve_44100, matched_curve)
    percent_closed = (distance_before - distance_after) / distance_before * 100.0

    print(
        f"\n[eq_match cross-sample-rate] 44100Hz curve -> 48000Hz target: "
        f"distance before={distance_before:.3f} dB, after={distance_after:.3f} dB, "
        f"{percent_closed:.1f}% closed"
    )

    assert distance_after < distance_before
    assert percent_closed > 50.0


def test_curve_whose_lowest_frequency_exceeds_target_nyquist_is_refused() -> None:
    """A curve that cannot possibly apply must fail loudly, not silently mis-measure.

    `spectrum_profile`'s own curves always start near 20 Hz (comfortably
    below almost any real Nyquist), so this uses a hand-authored curve --
    exactly the shape contracts/plan.v1.md's `eq_match.curve` stores
    directly -- whose lowest frequency is deliberately above the target's
    Nyquist, the concrete case the guard exists for.
    """
    curve = [[8000.0, 3.0], [9000.0, -2.0], [10000.0, 1.0]]

    low_sr = 8000  # Nyquist 4000 Hz -- below the curve's lowest point (8000 Hz)
    n_low = int(low_sr * DURATION_S)
    target = _white_noise(seed=11, n=n_low)

    with pytest.raises(CurveError):
        apply_curve(target, low_sr, curve, strength=1.0)


# ---------------------------------------------------------------------------
# Small unit-level guards.
# ---------------------------------------------------------------------------


def test_spectrum_profile_curve_has_at_least_two_ascending_points() -> None:
    noise = _white_noise(seed=12)
    curve = spectrum_profile(noise, SR)
    pairs = curve["curve"]
    assert len(pairs) >= 2
    freqs = [p[0] for p in pairs]
    assert freqs == sorted(freqs)
    assert len(set(freqs)) == len(freqs)
    for freq_hz, gain_db in pairs:
        assert math.isfinite(freq_hz)
        assert math.isfinite(gain_db)


def test_apply_curve_accepts_a_bare_pair_list_not_just_a_profile_dict() -> None:
    """contracts/plan.v1.md stores `eq_match.curve` as a bare array of
    pairs; `apply_curve` must accept that shape directly, not just the
    richer dict `spectrum_profile` returns."""
    noise = _white_noise(seed=13)
    source = filters.apply_sos(noise, filters.lowpass(SR, 2000.0, order=4))
    bare_curve = [[100.0, 0.0], [1000.0, 6.0], [10000.0, 12.0]]
    matched = apply_curve(source, SR, bare_curve, strength=1.0)
    assert matched.shape == source.shape
