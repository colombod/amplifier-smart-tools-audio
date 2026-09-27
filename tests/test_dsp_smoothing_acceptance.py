"""Acceptance tests for issue #18 ([aud] Step 8: two-axis smoothing).

Every measurement here is taken on the RENDERED signal (analyse -> gain
surface -> resynthesise), never on the internally-computed gain curve alone
-- the same rule `tests/test_dsp_collision_acceptance.py` follows, restated
in issue #9's own acceptance text. Real, physically bandlimited noise
(`scipy.signal.sosfiltfilt`, via `collision_test_support.band_limited_noise`)
is used wherever the scenario calls for "speech" or "music", not hand-built
band-energy arrays, for the same reason `test_dsp_collision_acceptance.py`
gives: seeing genuine spectral leakage, calibration, and STFT-framing
effects a real render would produce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aud.dsp import collision, stft
from aud.dsp.bands import band_edges
from aud.dsp.smoothing import (
    TransitionFasterThanHopError,
    duck_gain_surface,
    smooth_frequency_axis_erb_db,
    smooth_time_axis_db,
    upsample_bands_to_bins,
)
from tests import collision_test_support as ch
from tests import musical_noise as mn
from tests import smoothing_test_support as h


def _reversal_count(level_db: np.ndarray, eps_db: float = 0.2) -> int:
    """Number of direction reversals in `level_db`'s first difference,
    ignoring sub-`eps_db` wiggle (floating-point/measurement noise) --
    the same concept `aud.dsp.gate`'s own `open_count` diagnostic measures
    for chatter, generalised to a signed reversal count.
    """
    d = np.diff(level_db)
    d = d[np.abs(d) > eps_db]
    if d.size < 2:
        return 0
    signs = np.sign(d)
    return int(np.sum(signs[1:] != signs[:-1]))


# --- Acceptance 1: a deliberately frequency-discontinuous surface -> ---
# --- smoothing measurably improves the (disclosed, newly-introduced) ---
# --- musical-noise proxy metric -- see aud.dsp.smoothing's own module ---
# --- docstring, "Musical-noise measurement," for why Step 3's own ---
# --- proxy does not exist and this one is used instead. ---


def test_frequency_discontinuous_surface_smoothing_improves_kurtosis_ratio():
    """A per-bin gain surface with isolated, per-frame-RANDOM deep cuts
    (residual "un-subtracted" bins left at unity against an otherwise
    ducked floor -- the textbook musical-noise shape, Saruwatari et al.'s
    own description: "the amount of musical noise is highly correlated
    with the number of isolated power spectral components") is rendered
    both UNSMOOTHED and ERB-smoothed; the smoothed render's kurtosis ratio
    against the same observed signal must be measurably lower (closer to
    1, i.e. less musical noise), not merely different.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(3.0 * h.SR)) * 0.3

    b = band_edges(32, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins = h.N_FFT // 2 + 1
    spec = stft.analyze(x, h.SR, n_fft=h.N_FFT, hop=h.HOP)
    n_frames = spec.shape[1]

    rng2 = np.random.default_rng(1001)
    base_cut_db = 30.0
    outlier_prob = 0.02
    raw_bin_db = np.full((n_bins, n_frames), -base_cut_db)
    residual_mask = rng2.random((n_bins, n_frames)) < outlier_prob
    raw_bin_db[residual_mask] = 0.0  # isolated un-cut "leftover" bins, per-frame-random

    smoothed_bin_db = np.empty_like(raw_bin_db)
    for frame in range(n_frames):
        smoothed_bin_db[:, frame] = smooth_frequency_axis_erb_db(raw_bin_db[:, frame], b, h.N_FFT, h.SR)

    y_unsmoothed = h.apply_bin_gain_db_and_render(x, raw_bin_db)
    y_smoothed = h.apply_bin_gain_db_and_render(x, smoothed_bin_db)

    spec_obs = stft.analyze(x, h.SR, n_fft=h.N_FFT, hop=h.HOP)
    spec_uns = stft.analyze(y_unsmoothed, h.SR, n_fft=h.N_FFT, hop=h.HOP)
    spec_smo = stft.analyze(y_smoothed, h.SR, n_fft=h.N_FFT, hop=h.HOP)

    ratio_unsmoothed = mn.kurtosis_ratio(np.abs(spec_obs) ** 2, np.abs(spec_uns) ** 2)
    ratio_smoothed = mn.kurtosis_ratio(np.abs(spec_obs) ** 2, np.abs(spec_smo) ** 2)

    # Measured this session, 5 independent seeds: unsmoothed 22-30, smoothed
    # 1.04-1.06 -- a >20x reduction. Assert a conservative fraction of that.
    assert ratio_unsmoothed > 10.0, (
        f"test setup: expected the unsmoothed surface to read as heavily musical-noisy, got {ratio_unsmoothed}"
    )
    assert ratio_smoothed < 3.0, f"ERB smoothing did not bring the kurtosis ratio near 1; got {ratio_smoothed}"
    assert ratio_smoothed < ratio_unsmoothed / 5.0, (
        f"expected smoothing to improve (lower) the kurtosis ratio by at least 5x; "
        f"unsmoothed={ratio_unsmoothed:.3f}, smoothed={ratio_smoothed:.3f}"
    )


# --- Acceptance 2: speech-over-music -> no audible pumping/musical noise ---


def test_speech_over_music_duck_reduces_chatter_and_transitions_monotonically():
    """A music bed ducked by an intermittent speech-like key, via the real
    `collision_gains` (Step 6) LP output smoothed by `duck_gain_surface`
    (Step 8): the rendered bed's own level must show FEWER direction
    reversals ("chatter") than applying the same raw, un-smoothed
    `collision_gains` output directly -- and each individual engage/release
    transition in the smoothed render must move monotonically, not
    zig-zag, evidencing "a smooth residual" the way a zipper/pumping
    artifact would not.
    """
    rng = np.random.default_rng(31)
    dur = 6.0
    bands = ch.peaq_bands(32)

    target = ch.band_limited_noise(rng, dur, 80, 12000, sr=h.SR) * 0.6  # broadband "music"
    key = np.zeros(int(dur * h.SR))
    burst = ch.band_limited_noise(rng, dur, 300, 3000, sr=h.SR)  # speech-band noise
    period_s, on_s = 1.2, 0.5
    t = np.arange(len(key)) / h.SR
    on_mask = (t % period_s) < on_s
    key[on_mask] = burst[on_mask]

    target_power, target_spec, weights = ch.analyze_band_energy(target, bands)
    key_power, _, _ = ch.analyze_band_energy(key, bands)
    result = collision.collision_gains(target_power, key_power, bands, margin_db=3.0, g_min=0.0)

    # Raw (Step 6 only, no ballistics) render -- the pre-Step-8 baseline.
    proc_spec_raw = ch.apply_band_gain(target_spec, result.gain, weights)
    y_raw = ch.resynthesize(proc_spec_raw, len(target))

    # Step 8: smoothed render.
    bin_gain_amp, _stats = duck_gain_surface(
        result.gain, bands, h.N_FFT, h.HOP, h.SR, attack_ms=20.0, hold_ms=60.0, release_ms=150.0
    )
    y_smooth = stft.resynthesize(target_spec * bin_gain_amp, h.SR, len(target), n_fft=h.N_FFT, hop=h.HOP)

    lvl_orig = h.frame_level_db(target, sr=h.SR)
    lvl_raw = h.frame_level_db(y_raw, sr=h.SR)
    lvl_smooth = h.frame_level_db(y_smooth, sr=h.SR)
    edge = 15  # exclude STFT edge-padding frames, unrelated to the ducking algorithm
    interior = slice(edge, len(lvl_orig) - edge)

    reversals_raw = _reversal_count(lvl_raw[interior])
    reversals_smooth = _reversal_count(lvl_smooth[interior])
    assert reversals_smooth < reversals_raw, (
        f"expected smoothing to reduce chatter (direction reversals); raw={reversals_raw}, smoothed={reversals_smooth}"
    )
    assert reversals_smooth < reversals_raw * 0.8, "expected at least a 20% reduction in chatter"

    # Monotonic-transition check: find the single deepest engagement in the
    # smoothed render and verify its approach is monotonically decreasing
    # (no zig-zag) over the attack window immediately preceding its minimum.
    deepest_frame = int(np.argmin(lvl_smooth[interior])) + edge
    window_start = max(edge, deepest_frame - 8)
    approach = lvl_smooth[window_start : deepest_frame + 1]
    diffs = np.diff(approach)
    assert np.all(diffs <= 0.05), (
        f"the smoothed render's approach to its deepest duck was not monotonic (a zig-zag/zipper): {diffs}"
    )


# --- Acceptance 3: bed with a gap in the key -> returns to unity after release ---


def test_bed_returns_to_unity_after_release_following_key_gap():
    """A steady duck engagement followed by a gap (key drops silent) must
    let the rendered bed recover to within a stated tolerance of unity in
    the time the ballistics' own math predicts: `hold_ms + release_ms *
    ln(duck_db / tol_db)` (a one-pole's exact exponential-decay recovery
    time for a curve smoothed in the dB/log domain), not an arbitrary
    fixed number.
    """
    rng = np.random.default_rng(21)
    dur = 3.0
    x = rng.standard_normal(int(dur * h.SR)) * 0.2

    b = band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    spec = stft.analyze(x, h.SR, n_fft=h.N_FFT, hop=h.HOP)
    n_frames = spec.shape[1]
    n_bands = b["n_bands"]

    engage_start, engage_end = 20, n_frames // 2
    duck_db = -18.0
    hold_ms, release_ms = 50.0, 150.0
    band_gain_power = np.ones((n_bands, n_frames))
    band_gain_power[:, engage_start:engage_end] = 10 ** (duck_db / 10.0)

    bin_gain_amp, _stats = duck_gain_surface(
        band_gain_power, b, h.N_FFT, h.HOP, h.SR, attack_ms=20.0, hold_ms=hold_ms, release_ms=release_ms
    )
    y = stft.resynthesize(spec * bin_gain_amp, h.SR, len(x), n_fft=h.N_FFT, hop=h.HOP)

    orig_level = h.frame_level_db(x)
    lvl = h.frame_level_db(y)
    diff = lvl - orig_level

    tol_db = 1.0
    recovered_frame = None
    for f in range(engage_end, n_frames - 5):
        if np.all(np.abs(diff[f : f + 5]) < tol_db):
            recovered_frame = f
            break
    assert recovered_frame is not None, "bed never recovered to within tolerance of unity"

    hop_ms = h.HOP / h.SR * 1000.0
    measured_recovery_ms = (recovered_frame - engage_end) * hop_ms
    expected_recovery_ms = hold_ms + release_ms * math.log(abs(duck_db) / tol_db)
    tolerance_ms = 4.0 * hop_ms

    assert abs(measured_recovery_ms - expected_recovery_ms) <= tolerance_ms, (
        f"recovery took {measured_recovery_ms:.2f} ms; predicted {expected_recovery_ms:.2f} ms "
        f"+/- {tolerance_ms:.2f} ms"
    )


# --- Acceptance 4: a transition faster than the hop can resolve fails loud ---


def test_duck_transition_faster_than_hop_fails_with_named_error():
    b = band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    band_gain = np.ones((8, 20))
    hop_ms = h.HOP / h.SR * 1000.0
    with pytest.raises(TransitionFasterThanHopError) as excinfo:
        duck_gain_surface(band_gain, b, h.N_FFT, h.HOP, h.SR, attack_ms=hop_ms / 2.0, release_ms=150.0)
    assert excinfo.value.code == "duck_transition_faster_than_hop"


# --- Extra required test: transients are under-ducked without lookahead ---


def test_transient_duck_reaches_target_depth_with_lookahead_but_not_without():
    """A broadband transient duck request landing on exactly one STFT
    frame must be delivered close to its intended depth when the
    non-causal lookahead this module always applies (`duck_gain_surface`)
    is used, and measurably UNDER-delivered without it -- reproducing this
    repo's own measured -6.7/-9.0 dB (of an intended -20 dB) figure for the
    same underlying STFT-smear effect.
    """
    rng = np.random.default_rng(11)
    x = rng.standard_normal(int(2.0 * h.SR)) * 0.2

    b = band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    spec = stft.analyze(x, h.SR, n_fft=h.N_FFT, hop=h.HOP)
    n_frames = spec.shape[1]
    n_bands = b["n_bands"]
    transient_frame = n_frames // 2
    target_db = -20.0

    band_gain_power = np.ones((n_bands, n_frames))
    band_gain_power[:, transient_frame] = 10 ** (target_db / 10.0)

    frame_rate_hz = h.SR / h.HOP
    band_gain_db = 10.0 * np.log10(np.maximum(band_gain_power, 1e-30))

    # (a) no lookahead -- bypass duck_gain_surface's automatic lookahead.
    smoothed_db_no_la = np.empty_like(band_gain_db)
    for band in range(n_bands):
        smoothed_db_no_la[band], _ = smooth_time_axis_db(
            band_gain_db[band], frame_rate_hz, attack_ms=20.0, hold_ms=50.0, release_ms=150.0, lookahead_frames=0
        )
    bin_gain_db_no_la = upsample_bands_to_bins(smoothed_db_no_la, b, h.N_FFT, h.SR)
    y_no_la = stft.resynthesize(spec * 10.0 ** (bin_gain_db_no_la / 20.0), h.SR, len(x), n_fft=h.N_FFT, hop=h.HOP)

    # (b) with lookahead -- duck_gain_surface's own default behaviour.
    bin_gain_amp_la, stats = duck_gain_surface(
        band_gain_power, b, h.N_FFT, h.HOP, h.SR, attack_ms=20.0, hold_ms=50.0, release_ms=150.0
    )
    y_la = stft.resynthesize(spec * bin_gain_amp_la, h.SR, len(x), n_fft=h.N_FFT, hop=h.HOP)

    orig_level = h.frame_level_db(x)
    lvl_no_la = h.frame_level_db(y_no_la)
    lvl_la = h.frame_level_db(y_la)

    delivered_no_la = lvl_no_la[transient_frame] - orig_level[transient_frame]
    delivered_la = lvl_la[transient_frame] - orig_level[transient_frame]

    assert stats["lookahead_frames"] > 0, "duck_gain_surface must compute a non-zero automatic lookahead"
    assert delivered_no_la > -12.0, (
        f"test setup: without lookahead the transient should be clearly under-ducked (measured -6.7/-9.0 dB "
        f"regime); got {delivered_no_la:.2f} dB"
    )
    assert delivered_la < -14.0, f"with lookahead the transient should reach close to -20 dB; got {delivered_la:.2f} dB"
    assert delivered_la < delivered_no_la - 8.0, (
        f"lookahead must deliver measurably deeper attenuation at the transient's own frame; "
        f"no_lookahead={delivered_no_la:.2f} dB, with_lookahead={delivered_la:.2f} dB"
    )
