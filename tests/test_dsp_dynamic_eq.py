"""Acceptance and mutation-proving tests for `aud.dsp.gate.dynamic_eq` --
Step 7 of the masking/ducking epic (issue #17): a per-band dynamic EQ
driven by an EXTERNAL key.

Every measurement here is taken from the RENDERED output of `dynamic_eq`
itself (real audio in, real audio out), never from an internal curve or
level array in isolation -- the same discipline
`tests/test_dsp_collision_acceptance.py` and `AGENTS.md` require. Each
test's docstring names the ONE design property it is built to catch, and
no expected value is ever computed by calling the function (or its
underlying curve helper) under test -- expected values are hand-derived
closed forms or independent measurements of the raw signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import gate as gate_mod

SR = 48000
_EPS = 1e-12


def _rms_db(x: np.ndarray) -> float:
    """Plain RMS level in dB -- an INDEPENDENT measurement, not a call into
    `aud.dsp.gate._detector_level_db` or any other code under test."""
    return 20.0 * np.log10(np.sqrt(np.mean(x * x)) + _EPS)


def _steady_tone(freq_hz: float, amplitude: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


# Measurement window: skip the first 200 ms so the detector's 5 ms one-pole
# level smoother (and, where relevant, the key highpass) has fully settled;
# a steady tone's RMS in this window is what "rendered level" means below.
_SETTLE_S = 0.2


def _steady_window(seconds: float, sr: int = SR) -> slice:
    return slice(int(_SETTLE_S * sr), int(seconds * sr))


# --- Validation ----------------------------------------------------------


def test_dynamic_eq_rejects_ratio_below_one_and_negative_knee_and_negative_depth():
    target = np.zeros(1000)
    key = np.zeros(1000)
    with pytest.raises(ValueError, match="ratio"):
        gate_mod.dynamic_eq(target, key, SR, ratio=0.5)
    with pytest.raises(ValueError, match="knee_db"):
        gate_mod.dynamic_eq(target, key, SR, knee_db=-1.0)
    with pytest.raises(ValueError, match="max_depth_db"):
        gate_mod.dynamic_eq(target, key, SR, max_depth_db=-1.0)


def test_dynamic_eq_rejects_key_with_a_different_sample_count():
    target = np.zeros(1000)
    key = np.zeros(500)
    with pytest.raises(ValueError, match="n_samples"):
        gate_mod.dynamic_eq(target, key, SR)


# --- Seam: detection runs on KEY, gain is applied to TARGET (issue #17's own seam) --


def test_silent_target_is_still_attenuated_when_the_key_is_loud():
    """THE seam issue #17 opens: `_process_bands` used to detect on the
    SAME signal it gained. If `dynamic_eq` still (accidentally) detected
    on `target` instead of `key`, a near-silent `target` would never cross
    threshold and would be left at unity gain no matter how loud `key` is.
    Here `target` is a real (not literally zero) but very quiet tone,
    `key` is loud and clearly above threshold: `target` must still be
    measurably cut."""
    seconds = 1.0
    target = _steady_tone(500.0, 0.001, seconds)  # ~ -66 dBFS peak, well below any sane threshold
    key = _steady_tone(1000.0, 0.5, seconds)  # ~ -9 dBFS RMS, well above threshold

    y, _stats = gate_mod.dynamic_eq(
        target, key, SR, threshold_db=-20.0, ratio=4.0, knee_db=0.0, max_depth_db=24.0, key_hpf_hz=None
    )
    win = _steady_window(seconds)
    before_db = _rms_db(target[win])
    after_db = _rms_db(y[win])
    print(f"\n[dynamic_eq] silent-target/loud-key: before={before_db:.2f} dBFS, after={after_db:.2f} dBFS")
    assert (before_db - after_db) > 3.0, "target must be attenuated when the KEY is loud, even if target is quiet"


def test_loud_target_is_left_alone_when_the_key_is_quiet():
    """The mirror of the test above: `target`'s OWN loudness must not
    drive the gain at all. A loud `target` with a quiet `key` (well below
    threshold) must come through essentially unchanged."""
    seconds = 1.0
    target = _steady_tone(500.0, 0.5, seconds)  # loud
    key = _steady_tone(1000.0, 0.0001, seconds)  # ~ -86 dBFS RMS, well below threshold

    y, _stats = gate_mod.dynamic_eq(
        target, key, SR, threshold_db=-20.0, ratio=4.0, knee_db=0.0, max_depth_db=24.0, key_hpf_hz=None
    )
    win = _steady_window(seconds)
    before_db = _rms_db(target[win])
    after_db = _rms_db(y[win])
    print(f"\n[dynamic_eq] loud-target/quiet-key: before={before_db:.2f} dBFS, after={after_db:.2f} dBFS")
    assert abs(before_db - after_db) < 0.5, (
        "target must be left alone when the KEY is quiet, regardless of its own level"
    )


# --- Acceptance 1: depth limit never exceeded, measured on rendered audio ----


def test_max_depth_db_limits_rendered_attenuation_full_band():
    """Acceptance criterion 1: given a requested maximum depth, rendered
    attenuation must never exceed it. `ratio`'s own curve is otherwise
    UNBOUNDED as the key's level rises (see
    `aud.dsp.dynamics.static_gain_reduction_db`'s docstring: the reduction
    grows linearly, without limit, as level moves further above
    threshold) -- a very loud key, with no depth limit, would demand far
    more than `max_depth_db`. This is a mutation-proving test: removing
    the `np.maximum(raw_db, -max_depth_db)` clip in `dynamic_eq`'s curve
    closure would make this fail immediately (the measured cut would blow
    straight past 10 dB)."""
    seconds = 1.0
    target = _steady_tone(500.0, 0.3, seconds)
    key = _steady_tone(1000.0, 0.9, seconds)  # very loud relative to a low threshold below

    max_depth_db = 10.0
    y, _stats = gate_mod.dynamic_eq(
        target,
        key,
        SR,
        threshold_db=-60.0,  # deliberately far below the key's own level
        ratio=20.0,  # a steep ratio that would demand a huge cut if unclipped
        knee_db=0.0,
        max_depth_db=max_depth_db,
        key_hpf_hz=None,
    )
    win = _steady_window(seconds)
    before_db = _rms_db(target[win])
    after_db = _rms_db(y[win])
    measured_cut_db = before_db - after_db
    print(f"\n[dynamic_eq] depth-limit: measured cut = {measured_cut_db:.3f} dB (limit {max_depth_db} dB)")
    assert measured_cut_db <= max_depth_db + 0.5, (
        f"rendered attenuation ({measured_cut_db:.2f} dB) must never exceed the requested max depth ({max_depth_db} dB)"
    )
    # Not just "under the limit" -- with this scenario's ratio/threshold, the
    # UNCLIPPED law would demand far more than max_depth_db, so a correctly
    # working clip should land close to the ceiling, not stop short of it by
    # some other unrelated amount (which would suggest the wrong quantity is
    # being clipped, e.g. clipping the curve to silence instead of the limit).
    assert measured_cut_db > max_depth_db - 1.0, (
        f"expected the clip to land near its ceiling ({max_depth_db} dB); measured only {measured_cut_db:.2f} dB"
    )


def test_max_depth_db_limits_rendered_attenuation_per_band_multiband():
    """Same property as above, but multiband: the depth limit must hold
    independently in EACH band, not just in aggregate."""
    seconds = 1.0
    target = _steady_tone(200.0, 0.3, seconds) + _steady_tone(4000.0, 0.3, seconds)
    key = _steady_tone(200.0, 0.9, seconds) + _steady_tone(4000.0, 0.9, seconds)

    max_depth_db = 8.0
    _y, stats = gate_mod.dynamic_eq(
        target,
        key,
        SR,
        threshold_db=-60.0,
        ratio=20.0,
        knee_db=0.0,
        max_depth_db=max_depth_db,
        crossovers_hz=[1000.0],
        key_hpf_hz=None,
    )
    for band in stats["bands"]:
        print(f"\n[dynamic_eq] band {band['index']}: max_attenuation_db={band['max_attenuation_db']:.2f}")
        assert band["max_attenuation_db"] >= -(max_depth_db + 0.5), (
            f"band {band['index']} exceeded the depth limit: {band['max_attenuation_db']:.2f} dB "
            f"vs limit -{max_depth_db} dB"
        )


# --- Acceptance 2: identical stereo channels stay identical (mono-fold proof) --


def test_identical_stereo_channels_remain_identical():
    """Acceptance criterion 2: identical stereo channels in must produce
    identical channels out -- proving detection folded to mono and the
    stereo image did not shift."""
    seconds = 1.0
    mono_target = _steady_tone(500.0, 0.3, seconds)
    mono_key = _steady_tone(1000.0, 0.4, seconds)
    target = np.stack([mono_target, mono_target], axis=1)
    key = np.stack([mono_key, mono_key], axis=1)

    y, _ = gate_mod.dynamic_eq(target, key, SR, threshold_db=-20.0, ratio=3.0, knee_db=3.0, max_depth_db=18.0)
    assert np.allclose(y[:, 0], y[:, 1], atol=1e-9), "identical stereo input must produce identical stereo output"


def test_asymmetric_stereo_key_detection_is_symmetric_under_channel_swap():
    """The real mutation-proof for "detection must fold to mono per band"
    (issue #17's stereo hazard). Because `dynamic_eq` (like `gate`/
    `expand`) applies ONE gain envelope to every channel of a band (see
    `_process_bands`'s `band * gain_lin[:, None]` broadcast), a target
    with identical input channels will ALWAYS come out with identical
    output channels regardless of how the key's channels are detected --
    that structural property alone does not prove the fold is correct.

    The property that DOES prove it: detection must not depend on which
    physical channel index happens to carry the louder content. Swap the
    key's loud/silent channels left-for-right and the MEASURED CUT must
    be the same either way. A detector that (incorrectly) reads only one
    hardcoded channel index would instead engage in one arrangement and
    stay silent in the other -- exactly the kind of channel-order
    dependence that, in an architecture where gain COULD legitimately
    differ per channel, is what produces an image shift; here it is
    caught directly, at the measurement it would corrupt."""
    seconds = 1.0
    mono_target = _steady_tone(500.0, 0.3, seconds)
    target = np.stack([mono_target, mono_target], axis=1)

    loud = _steady_tone(1000.0, 0.5, seconds)
    silent = np.zeros_like(loud)
    key_loud_left = np.stack([loud, silent], axis=1)
    key_loud_right = np.stack([silent, loud], axis=1)

    common = {"threshold_db": -20.0, "ratio": 4.0, "knee_db": 0.0, "max_depth_db": 24.0, "key_hpf_hz": None}
    y_loud_left, _ = gate_mod.dynamic_eq(target, key_loud_left, SR, **common)
    y_loud_right, _ = gate_mod.dynamic_eq(target, key_loud_right, SR, **common)

    win = _steady_window(seconds)
    cut_loud_left_db = _rms_db(target[win, 0]) - _rms_db(y_loud_left[win, 0])
    cut_loud_right_db = _rms_db(target[win, 0]) - _rms_db(y_loud_right[win, 0])
    print(
        f"\n[dynamic_eq] mono-fold symmetry: loud-left cut={cut_loud_left_db:.2f} dB, loud-right={cut_loud_right_db:.2f} dB"
    )

    assert cut_loud_left_db > 1.0, "the loud channel should pull the mono-folded detector above threshold"
    assert abs(cut_loud_left_db - cut_loud_right_db) < 0.2, (
        "detection must not depend on which channel index carries the loud content -- both arrangements "
        "must produce the same cut"
    )
    # And, structurally, both renders must still leave target's own channels identical.
    assert np.allclose(y_loud_left[:, 0], y_loud_left[:, 1], atol=1e-9)
    assert np.allclose(y_loud_right[:, 0], y_loud_right[:, 1], atol=1e-9)


# --- Acceptance 3: helper reuse -- static law matches a hand-derived closed form --


def test_rendered_gain_reduction_matches_the_hand_derived_static_law_hard_knee():
    """Acceptance criterion 4 (and 3, by construction): with `knee_db=0`,
    `aud.dsp.dynamics.static_gain_reduction_db`'s documented curve above
    threshold is the closed form

        reduction_db = (threshold_db - level_db) * (1 - 1/ratio)

    This expected value is HAND-DERIVED here from the module's own
    documented formula, not by calling `static_gain_reduction_db` (the
    function this feature reuses) -- the independent variable is the
    KEY's own measured RMS level, taken directly from the raw key array,
    not from any internal detector state."""
    seconds = 1.0
    threshold_db = -20.0
    ratio = 4.0

    key = _steady_tone(1000.0, 0.5, seconds)  # a level comfortably above threshold_db
    target = _steady_tone(300.0, 0.2, seconds)

    win = _steady_window(seconds)
    key_level_db = _rms_db(key[win])  # independent measurement of the key's own level
    expected_reduction_db = (threshold_db - key_level_db) * (1.0 - 1.0 / ratio)
    assert expected_reduction_db < -1.0, "test setup: expected a real, non-trivial reduction"

    y, _stats = gate_mod.dynamic_eq(
        target,
        key,
        SR,
        threshold_db=threshold_db,
        ratio=ratio,
        knee_db=0.0,
        max_depth_db=60.0,  # comfortably above the expected reduction -- must not clip here
        key_hpf_hz=None,
    )
    measured_reduction_db = _rms_db(y[win]) - _rms_db(target[win])
    print(
        f"\n[dynamic_eq] hard-knee law: key_level={key_level_db:.2f} dBFS, "
        f"expected={expected_reduction_db:.2f} dB, measured={measured_reduction_db:.2f} dB"
    )
    assert measured_reduction_db == pytest.approx(expected_reduction_db, abs=0.5), (
        f"rendered reduction ({measured_reduction_db:.2f} dB) should match the hand-derived static law "
        f"({expected_reduction_db:.2f} dB) within 0.5 dB"
    )


def test_ratio_one_is_a_no_op():
    """`ratio=1.0` means no reduction at all, regardless of how far above
    threshold the key sits -- a direct, cheap regression pin on the
    reused static law's own documented behaviour."""
    seconds = 0.5
    target = _steady_tone(300.0, 0.2, seconds)
    key = _steady_tone(1000.0, 0.9, seconds)
    y, stats = gate_mod.dynamic_eq(
        target, key, SR, threshold_db=-40.0, ratio=1.0, knee_db=0.0, max_depth_db=24.0, key_hpf_hz=None
    )
    assert stats["max_attenuation_db"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(y, target, atol=1e-9)


def test_knee_softens_the_transition_at_threshold_not_a_hard_step():
    """Mutation-proof for `knee_db` reuse specifically (distinct from
    `ratio`'s own test above): with a real knee, a key sitting EXACTLY at
    threshold must already show a small, non-zero reduction (the knee's
    quadratic connecting piece), whereas with `knee_db=0` (hard knee) the
    exact same key level shows ~0 dB reduction. If `dynamic_eq` silently
    ignored `knee_db` (e.g. always called the hard-knee path), these two
    renders would be indistinguishable."""
    seconds = 1.0
    threshold_db = -20.0
    ratio = 4.0
    knee_db = 6.0

    # An amplitude whose measured RMS lands almost exactly at threshold_db:
    # -20 dBFS RMS <=> amplitude = sqrt(2) * 10**(-20/20).
    amplitude = np.sqrt(2.0) * 10.0 ** (threshold_db / 20.0)
    key = _steady_tone(1000.0, amplitude, seconds)
    target = _steady_tone(300.0, 0.2, seconds)
    win = _steady_window(seconds)
    measured_key_level_db = _rms_db(key[win])
    print(f"\n[dynamic_eq] knee test: measured key level = {measured_key_level_db:.3f} dBFS (target -20.0)")

    y_hard, _ = gate_mod.dynamic_eq(
        target, key, SR, threshold_db=threshold_db, ratio=ratio, knee_db=0.0, max_depth_db=24.0, key_hpf_hz=None
    )
    y_soft, _ = gate_mod.dynamic_eq(
        target, key, SR, threshold_db=threshold_db, ratio=ratio, knee_db=knee_db, max_depth_db=24.0, key_hpf_hz=None
    )
    hard_cut_db = _rms_db(target[win]) - _rms_db(y_hard[win])
    soft_cut_db = _rms_db(target[win]) - _rms_db(y_soft[win])
    print(f"[dynamic_eq] at-threshold cut: hard knee={hard_cut_db:.3f} dB, soft knee={soft_cut_db:.3f} dB")

    assert abs(hard_cut_db) < 0.3, "a hard knee at exactly threshold should show ~0 dB reduction"
    assert soft_cut_db > 0.15, "a soft knee at exactly threshold should already show a real, non-zero reduction"


# --- Multiband: per-band independence -------------------------------------


def test_multiband_dynamic_eq_only_attenuates_the_band_where_the_key_is_loud():
    """Mirrors `gate`'s own multiband property test: the gain law is
    computed INDEPENDENTLY PER BAND (this feature's own binding ruling),
    so a key loud only in one band must attenuate only that band of the
    target, leaving the other band alone."""
    seconds = 1.0
    target = _steady_tone(150.0, 0.3, seconds) + _steady_tone(6000.0, 0.3, seconds)
    key = _steady_tone(150.0, 0.9, seconds) + _steady_tone(6000.0, 0.0005, seconds)

    _y, stats = gate_mod.dynamic_eq(
        target,
        key,
        SR,
        threshold_db=-30.0,
        ratio=6.0,
        knee_db=0.0,
        max_depth_db=24.0,
        crossovers_hz=[1000.0],
        key_hpf_hz=None,
    )
    low_band, high_band = stats["bands"]
    print(f"\n[dynamic_eq] low band (key loud):  {low_band}")
    print(f"[dynamic_eq] high band (key quiet): {high_band}")
    assert low_band["max_attenuation_db"] < -3.0, "the band where the key is loud must be attenuated"
    assert high_band["max_attenuation_db"] > -1.0, "the band where the key is quiet must be left alone"
