"""Behavioural tests for loudness measurement and gain-only normalization."""

from __future__ import annotations

import numpy as np

from aud.dsp import loudness

SR = 48000


def _pink_ish_noise(seconds: float = 3.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    return rng.normal(0.0, 0.1, size=(n, 2))


def test_normalize_minus_30_lufs_signal_to_minus_14_lands_within_half_a_lu():
    base = _pink_ish_noise()

    # First calibrate a signal that measures at -30 LUFS using normalize()
    # itself (self-consistent: if normalize() is wrong, this setup step
    # would also be wrong, and the final assertion would catch it).
    at_minus_30, _ = loudness.normalize(base, SR, target_lufs=-30.0)
    measured_minus_30 = loudness.integrated_lufs(at_minus_30, SR)
    assert abs(measured_minus_30 - (-30.0)) < 0.5

    normalized, applied_gain_db = loudness.normalize(at_minus_30, SR, target_lufs=-14.0)
    measured_after = loudness.integrated_lufs(normalized, SR)

    print(f"\n[loudness] before={measured_minus_30:.3f} LUFS, target=-14.0, after={measured_after:.3f} LUFS")
    print(f"[loudness] applied_gain_db={applied_gain_db:.3f} dB")

    assert abs(measured_after - (-14.0)) < 0.5
    assert applied_gain_db > 0  # -30 -> -14 requires a gain increase


def test_normalize_gain_matches_the_simple_db_difference():
    base = _pink_ish_noise(seed=1)
    before = loudness.integrated_lufs(base, SR)
    target = -18.0

    y, applied_gain_db = loudness.normalize(base, SR, target_lufs=target)
    expected_gain_db = target - before

    assert abs(applied_gain_db - expected_gain_db) < 1e-9
    # And the linear gain actually applied to the samples matches applied_gain_db.
    ratio = np.abs(y[1000, 0] / base[1000, 0])
    expected_ratio = 10 ** (applied_gain_db / 20.0)
    assert abs(ratio - expected_ratio) < 1e-6


def test_normalize_of_silence_is_a_no_op_not_a_crash():
    silence = np.zeros((SR, 2))
    y, applied_gain_db = loudness.normalize(silence, SR, target_lufs=-14.0)
    assert applied_gain_db == 0.0
    assert np.array_equal(y, silence)


def test_loudness_range_returns_a_finite_number_for_varying_material():
    rng = np.random.default_rng(2)
    n = SR * 4
    envelope = np.concatenate([np.full(n // 2, 0.02), np.full(n - n // 2, 0.3)])
    x = (rng.normal(0.0, 1.0, size=(n, 2)) * envelope[:, None]).astype(np.float64)
    lra = loudness.loudness_range(x, SR)
    assert np.isfinite(lra)
    assert lra > 0.0
