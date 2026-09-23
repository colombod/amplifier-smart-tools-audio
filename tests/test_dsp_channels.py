"""Behavioural tests for aud.dsp.channels: fold_to_mono and downmix.

Measurement, not booleans: every assertion here is a measured numeric
quantity (a peak level, a correlation, a channel count), and the overflow
and antiphase claims each carry a NEGATIVE CONTROL proving the test would
actually catch the defect it guards against.
"""

from __future__ import annotations

import numpy as np

from aud.dsp import channels

SR = 44100


def _tone(seconds: float, freq_hz: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# --- fold_to_mono: the consolidated analysis fold ---------------------------


def test_fold_to_mono_matches_the_old_per_module_one_liner():
    """Before consolidation, eqmatch/speech/reverb each computed exactly
    `x if x.ndim == 1 else np.mean(x, axis=1)`. The shared helper must
    still compute exactly that -- a behaviour change here would silently
    change three unrelated modules at once.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.3, size=(1000, 3))
    expected = np.mean(x, axis=1)
    result = channels.fold_to_mono(x)
    assert np.array_equal(result, expected)


def test_fold_to_mono_passes_1d_input_through_unchanged():
    x = np.array([0.1, -0.2, 0.3])
    assert channels.fold_to_mono(x) is x or np.array_equal(channels.fold_to_mono(x), x)


# --- downmix: fold rule never overflows -------------------------------------


def test_downmix_of_identical_channels_preserves_level_exactly():
    """Two identical full-scale-ish channels: mean-fold must reproduce the
    same signal exactly (not double it, not halve it), and produce exactly
    one channel.
    """
    x = _tone(0.5, 440.0, amplitude=0.9)
    stereo = np.stack([x, x], axis=1)
    mono, report = channels.downmix(stereo, SR)
    assert mono.shape == (stereo.shape[0], 1)
    assert report["input_channels"] == 2
    assert np.allclose(mono[:, 0], x, atol=1e-12)
    assert float(np.max(np.abs(mono))) <= 1.0


def test_downmix_two_full_scale_in_phase_channels_does_not_exceed_full_scale():
    """The overflow proof: two channels AT full scale, in phase (the
    worst case for a naive sum -- 1.0 + 1.0 = 2.0), must still fold to
    exactly 1.0 under the mean rule, never above.
    """
    n = 1000
    full_scale = np.ones(n)
    stereo = np.stack([full_scale, full_scale], axis=1)
    mono, report = channels.downmix(stereo, SR)
    peak = float(np.max(np.abs(mono)))
    print(f"\n[downmix] two full-scale in-phase channels -> peak {peak}")
    assert peak == 1.0
    assert report["peak_dbfs_after"] <= report["peak_dbfs_before"] + 1e-9


def test_negative_control_a_naive_sum_of_the_same_signal_would_overflow():
    """Sanity check on the overflow test's own sensitivity: prove that the
    REJECTED alternative (a plain sum, no divide) actually overflows on
    the exact signal above -- so the assertion above is evidence the mean
    rule matters, not a vacuous pass.
    """
    n = 1000
    full_scale = np.ones(n)
    stereo = np.stack([full_scale, full_scale], axis=1)
    naive_sum = np.sum(stereo, axis=1)
    assert float(np.max(np.abs(naive_sum))) == 2.0, "the naive-sum negative control itself must overflow to 2.0"


def test_downmix_many_identical_channels_still_bounded_by_full_scale():
    n = 500
    full_scale = np.ones(n)
    x = np.stack([full_scale] * 6, axis=1)
    mono, report = channels.downmix(x, SR)
    assert report["input_channels"] == 6
    assert mono.shape == (n, 1)
    assert float(np.max(np.abs(mono))) == 1.0


# --- downmix: antiphase is detected, never silently swallowed --------------


def test_downmix_antiphase_channels_is_detected_and_reported():
    x = _tone(0.5, 440.0, amplitude=0.5)
    stereo = np.stack([x, -x], axis=1)  # exact polarity inversion
    mono, report = channels.downmix(stereo, SR)

    print(f"\n[downmix] antiphase correlation={report['correlation']}")
    assert report["correlation"] is not None
    assert report["correlation"] < -0.99
    assert report["antiphase_detected"] is True
    # The actual audible consequence: near-total cancellation.
    assert float(np.max(np.abs(mono))) < 1e-9


def test_downmix_ordinary_uncorrelated_stereo_is_not_flagged_antiphase():
    """Negative control for the antiphase detector: two DIFFERENT,
    positively-behaved tones (not inverted copies of each other) must NOT
    trip the antiphase flag -- otherwise the detector would be trivially
    satisfied by any two-channel signal and prove nothing above.
    """
    left = _tone(0.5, 440.0, amplitude=0.5)
    right = _tone(0.5, 441.0, amplitude=0.5)  # near-identical freq, in phase
    stereo = np.stack([left, right], axis=1)
    _mono, report = channels.downmix(stereo, SR)

    print(f"\n[downmix] near-identical-tone correlation={report['correlation']}")
    assert report["antiphase_detected"] is False


def test_downmix_reports_none_correlation_for_digital_silence():
    silence = np.zeros((1000, 2))
    _mono, report = channels.downmix(silence, SR)
    assert report["correlation"] is None
    assert report["antiphase_detected"] is False


# --- downmix: mono passthrough, always 2-D ----------------------------------


def test_downmix_of_already_mono_input_is_a_reported_no_op():
    x = _tone(0.2, 300.0)
    mono_in = x.reshape(-1, 1)
    mono_out, report = channels.downmix(mono_in, SR)
    assert report["input_channels"] == 1
    assert mono_out.shape == mono_in.shape
    assert np.array_equal(mono_out, mono_in)


def test_downmix_output_is_always_2d_never_a_bare_squeeze():
    x = _tone(0.1, 500.0)
    stereo = np.stack([x, x], axis=1)
    mono, _report = channels.downmix(stereo, SR)
    assert mono.ndim == 2
    assert mono.shape[1] == 1
