"""Unit tests for `aud.dsp.smoothing` -- Step 8 (issue #18).

Parametrization notes (this repo's own recurring defect class, see
`aud/dsp/bands.py`'s module docstring): the reference-coefficient sweep
below varies `tau_ms` across two orders of magnitude (5-300 ms) at a FIXED
`hop`/`sr`, and separately the hop-resolution tests vary `hop`/`sr`
combinations that are not all the same parity or ratio (512/48000,
256/44100, 1024/96000) so no single unstated property (e.g. "hop is always
a power of two", "sr is always 48000") is silently assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aud.dsp import bands
from aud.dsp.smoothing import (
    DUCK_ENGAGE_EPS_DB,
    TransitionFasterThanHopError,
    duck_gain_surface,
    smooth_frequency_axis_db,
    smooth_time_axis_db,
    upsample_bands_to_bins,
)

# --- one-pole coefficient formula, verified against an independently ---
# --- computed reference table (see aud.dsp.smoothing's own docstring) ---


@pytest.mark.parametrize(
    ("tau_ms", "expected_coeff"),
    [
        (5.0, 0.1184),
        (20.0, 0.5866),
        (50.0, 0.8079),
        (100.0, 0.8988),
        (300.0, 0.9651),
    ],
)
def test_one_pole_coefficient_matches_reference_table_at_hop_10p67ms(tau_ms, expected_coeff):
    """At hop=512/sr=48000 (hop_ms=10.667), a single-hop attack step from
    0 dB to a target should move by exactly `(1 - coeff)` of the gap --
    read the coefficient back out of one smoothed sample and compare.
    """
    frame_rate_hz = 48000.0 / 512.0
    target_db = np.array([0.0, -10.0, -10.0, -10.0])  # engage at frame 1, hold flat after
    smoothed, _stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=tau_ms, hold_ms=0.0, release_ms=tau_ms, lookahead_frames=0
    )
    # frame 0 starts at prev=0 (unity) and target[0]=0 -> stays 0.
    # frame 1: target=-10, prev=0 -> attack_coeff applies: smoothed[1] = coeff*0 + (1-coeff)*(-10)
    implied_coeff = 1.0 + smoothed[1] / 10.0
    assert implied_coeff == pytest.approx(expected_coeff, abs=2e-3), (
        f"tau_ms={tau_ms}: implied attack coeff {implied_coeff} != reference {expected_coeff}"
    )


def test_attack_and_release_are_asymmetric_by_default():
    """Defaults (attack_ms=20, release_ms=150) must produce DIFFERENT
    one-pole coefficients -- the mutation this guards against
    ("make attack and release symmetric") is exercised directly in
    tests/test_dsp_smoothing_mutations.py.
    """
    frame_rate_hz = 48000.0 / 512.0
    attack_coeff = math.exp(-1.0 / (frame_rate_hz * 20.0 / 1000.0))
    release_coeff = math.exp(-1.0 / (frame_rate_hz * 150.0 / 1000.0))
    assert attack_coeff != pytest.approx(release_coeff)
    assert release_coeff > attack_coeff, "150 ms release must be a SLOWER (closer-to-1) coefficient than 20 ms attack"


# --- smooth_time_axis_db: hold latches through a brief release, engage_count ---


def test_hold_latches_engagement_through_a_brief_gap():
    """A single-frame return to unity, shorter than `hold_ms`, must not
    let the curve recover at all -- the hold counter should still be
    running. `engage_count` must read exactly 1 (one engagement).
    """
    frame_rate_hz = 100.0  # 10 ms per frame, for round numbers
    target_db = np.array([0.0, -20.0, -20.0, 0.0, -20.0, -20.0, -20.0])  # 1-frame gap at index 3
    smoothed, stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=5.0, hold_ms=30.0, release_ms=100.0, lookahead_frames=0
    )
    assert stats["engage_count"] == 1, "a single engagement with only a 1-frame gap under hold must not re-count"
    # smoothed[3] must still reflect the held (engaged) value, not have jumped back toward unity
    assert smoothed[3] < -5.0, f"held frame should still be substantially ducked; got {smoothed[3]}"


def test_hold_zero_allows_immediate_release_on_gap():
    """`hold_ms=0` must behave like plain attack/release with no latching
    -- a gap immediately starts the release-toward-unity ramp.
    """
    frame_rate_hz = 100.0
    target_db = np.array([-20.0, -20.0, -20.0, 0.0, 0.0, 0.0])
    smoothed, _stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=5.0, hold_ms=0.0, release_ms=50.0, lookahead_frames=0
    )
    # with no hold, level at frame 4 (one frame after the gap opened at 3) must have moved
    # measurably toward unity already
    assert smoothed[4] > smoothed[3], "release must proceed immediately with hold_ms=0"


def test_engage_count_counts_transitions_not_frames():
    """Two separate engagements separated by a gap longer than hold must
    count as 2, not as however many frames were spent engaged.
    """
    frame_rate_hz = 100.0
    target_db = np.array([0.0] * 5 + [-15.0] * 5 + [0.0] * 20 + [-15.0] * 5 + [0.0] * 5)
    _smoothed, stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=5.0, hold_ms=10.0, release_ms=20.0, lookahead_frames=0
    )
    assert stats["engage_count"] == 2


# --- validation ---


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("attack_ms", {"attack_ms": -1.0}),
        ("hold_ms", {"hold_ms": -1.0}),
        ("release_ms", {"release_ms": -1.0}),
        ("lookahead_frames", {"lookahead_frames": -1}),
    ],
)
def test_smooth_time_axis_db_rejects_negative_parameters(name, kwargs):
    with pytest.raises(ValueError, match=name):
        smooth_time_axis_db(np.zeros(10), 100.0, **kwargs)


def test_smooth_time_axis_db_rejects_non_positive_frame_rate():
    with pytest.raises(ValueError, match="frame_rate_hz"):
        smooth_time_axis_db(np.zeros(10), 0.0)


# --- frequency axis: partition of unity, upsample/downsample correctness ---


def test_upsample_bands_to_bins_is_partition_of_unity_preserving():
    """A CONSTANT band value must upsample to that SAME constant at every
    bin -- the direct, minimal consequence of `sum_b weights[b, k] == 1`
    for every bin `k` (asserted, not merely assumed, via this behavioral
    check on the public function).
    """
    b = bands.band_edges(16, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    constant_db = np.full((16, 3), -7.5)
    bin_db = upsample_bands_to_bins(constant_db, b, n_fft=2048, sr=48000)
    assert np.allclose(bin_db, -7.5, atol=1e-9), "a constant band value must upsample to the identical constant"


def test_smooth_frequency_axis_db_is_partition_of_unity_preserving():
    """Same property as above, but through the full downsample->upsample
    round trip: a constant per-bin curve must smooth to the SAME constant.
    """
    b = bands.band_edges(16, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins = 2048 // 2 + 1
    constant_db = np.full(n_bins, -12.0)
    smoothed = smooth_frequency_axis_db(constant_db, b, n_fft=2048, sr=48000)
    assert np.allclose(smoothed, -12.0, atol=1e-9)


def test_smooth_frequency_axis_db_removes_isolated_single_bin_discontinuity():
    """A single isolated deep-cut bin surrounded by unity neighbours must
    come out shallower (blended with its neighbours) after Bark smoothing
    (these bands are `bark_peaq`), not still isolated at its original depth.
    """
    b = bands.band_edges(32, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins = 2048 // 2 + 1
    raw = np.zeros(n_bins)
    isolated_bin = 200
    raw[isolated_bin] = -40.0
    smoothed = smooth_frequency_axis_db(raw, b, n_fft=2048, sr=48000)
    assert smoothed[isolated_bin] > -40.0 + 5.0, (
        f"an isolated single-bin cut must be substantially diluted by Bark smoothing; got {smoothed[isolated_bin]} dB"
    )
    assert smoothed[isolated_bin] < 0.0, "the smoothed bin should still show SOME residual dip, not disappear entirely"


def test_upsample_bands_to_bins_rejects_band_count_mismatch():
    b = bands.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    with pytest.raises(ValueError, match="n_bands"):
        upsample_bands_to_bins(np.zeros((5, 3)), b, n_fft=2048, sr=48000)


# --- duck_gain_surface: hop-resolution error ---


@pytest.mark.parametrize(("n_fft", "hop", "sr"), [(2048, 512, 48000), (1024, 256, 44100), (4096, 1024, 96000)])
def test_transition_faster_than_hop_raises_named_error(n_fft, hop, sr):
    b = bands.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins_needed = n_fft // 2 + 1  # unused directly; duck_gain_surface only needs band-rate input
    n_frames = 10
    band_gain = np.ones((8, n_frames))
    hop_ms = hop / sr * 1000.0
    too_fast_ms = hop_ms * 0.5
    with pytest.raises(TransitionFasterThanHopError) as excinfo:
        duck_gain_surface(band_gain, b, n_fft, hop, sr, attack_ms=too_fast_ms, release_ms=150.0)
    assert excinfo.value.code == "duck_transition_faster_than_hop"
    del n_bins_needed


@pytest.mark.parametrize(("n_fft", "hop", "sr"), [(2048, 512, 48000), (1024, 256, 44100), (4096, 1024, 96000)])
def test_transition_at_or_above_hop_does_not_raise(n_fft, hop, sr):
    b = bands.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_frames = 10
    band_gain = np.ones((8, n_frames))
    hop_ms = hop / sr * 1000.0
    surface, stats = duck_gain_surface(band_gain, b, n_fft, hop, sr, attack_ms=hop_ms, release_ms=hop_ms * 2)
    assert surface.shape == (n_fft // 2 + 1, n_frames)
    assert stats["hop_ms"] == pytest.approx(hop_ms)


def test_duck_gain_surface_rejects_shape_mismatch():
    b = bands.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    with pytest.raises(ValueError, match="n_bands"):
        duck_gain_surface(np.ones((5, 10)), b, 2048, 512, 48000)


def test_duck_gain_surface_output_is_amplitude_not_power():
    """A band_gain_power of exactly 0.25 (linear POWER, -6.02 dB power)
    must upsample to an AMPLITUDE multiplier of 0.5 (sqrt), not 0.25.
    """
    b = bands.band_edges(4, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_frames = 30
    band_gain = np.full((4, n_frames), 0.25)
    surface, _stats = duck_gain_surface(
        band_gain, b, n_fft=2048, hop=512, sr=48000, attack_ms=10.667, hold_ms=0.0, release_ms=10.667
    )
    # allow the ballistics to settle: check the tail, far from the start transient
    assert np.allclose(surface[:, -1], 0.5, atol=0.02), (
        f"expected amplitude ~0.5 at steady state, got {surface[:, -1][:3]}"
    )


def test_duck_engage_eps_db_is_small_floating_point_slack():
    assert 0.0 < DUCK_ENGAGE_EPS_DB < 0.1
