"""Behavioural tests for the reverb-tail suppressor, against synthetic ground truth.

The only way to make a dereverberation claim falsifiable is to build a
signal whose reverberant amount is KNOWN: take a dry signal, convolve it
with a synthetic room impulse response (exponentially-decaying noise -- the
standard textbook model of a diffuse reverberant tail), and measure a
reverberation statistic (here: the decay rate of the short-time energy
envelope, in dB/s) before and after `dereverb`. A real improvement moves
that statistic measurably back toward the dry signal's own (much steeper)
decay rate.

This module is explicit (see its docstring) that this is a moderate,
honest improvement, not a heavy-reverb remover, and that it can also
mildly affect already-dry material (a documented, tested limit of blind
single-channel dereverberation). Both properties are asserted here with
numbers, not adjectives.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from aud.dsp import dereverb

SR = 48000
_EPS = 1e-12


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x * x)) + _EPS)


def _decay_rate_db_per_s(x: np.ndarray, sr: int) -> float | None:
    """Average dB/s slope of the short-time energy envelope, over frames
    where it is actually decreasing. More negative = faster decay.
    """
    frame_len = max(1, int(0.02 * sr))
    n_frames = x.shape[0] // frame_len
    if n_frames < 3:
        return None
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-9))
    diffs = np.diff(frame_db)
    decaying = diffs[diffs < 0]
    if decaying.size == 0:
        return None
    avg_decay_db_per_frame = float(np.mean(decaying))
    frame_duration_s = frame_len / sr
    return avg_decay_db_per_frame / frame_duration_s


def _make_bursty_dry_signal(
    seconds: float, sr: int, seed: int, burst_range: tuple[float, float], gap_range: tuple[float, float]
) -> np.ndarray:
    """Speech-like: band-limited noise bursts (onsets/offsets) separated by silence."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        burst_len = int(rng.uniform(*burst_range) * sr)
        gap_len = int(rng.uniform(*gap_range) * sr)
        end = min(n, pos + burst_len)
        env[pos:end] = 1.0
        pos = end + gap_len
    noise = rng.normal(0.0, 1.0, n)
    sos = signal.butter(4, [200.0, 4000.0], btype="bandpass", fs=sr, output="sos")
    sig = signal.sosfilt(sos, noise * env)
    return sig / np.max(np.abs(sig) + 1e-9) * 0.3


def _make_decaying_noise_ir(sr: int, decay_s: float, length_s: float, seed: int) -> np.ndarray:
    """A synthetic room impulse response: exponentially-decaying noise --
    the standard textbook model of a diffuse reverberant field."""
    rng = np.random.default_rng(seed)
    n = int(length_s * sr)
    t = np.arange(n) / sr
    ir = rng.normal(0.0, 1.0, n) * np.exp(-t / decay_s)
    ir[0] = 1.0  # dominant direct-sound impulse at t=0
    return ir / np.max(np.abs(ir))


def test_amount_zero_is_an_exact_no_op():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.1, (SR, 2))
    y, stats = dereverb.dereverb(x, SR, amount_db=0.0)
    assert np.max(np.abs(y - x)) < 1e-9
    assert stats["max_gain_reduction_db"] == pytest.approx(0.0, abs=1e-6)


def test_reverberant_material_measurably_decays_faster_after_dereverb():
    """The ground-truth test: dry -> convolve with a known decaying IR ->
    dereverb -> the output's decay rate must move toward the dry signal's,
    not just numerically differ."""
    dry = _make_bursty_dry_signal(4.0, SR, seed=2, burst_range=(0.1, 0.3), gap_range=(0.05, 0.15))
    ir = _make_decaying_noise_ir(SR, decay_s=0.4, length_s=1.0, seed=3)
    wet = signal.fftconvolve(dry, ir)[: len(dry)]
    wet = wet / np.max(np.abs(wet) + 1e-9) * 0.3

    y, stats = dereverb.dereverb(wet, SR, amount_db=10.0)

    dry_decay = _decay_rate_db_per_s(dry, SR)
    wet_decay = _decay_rate_db_per_s(wet, SR)
    out_decay = _decay_rate_db_per_s(y, SR)
    print(f"\n[dereverb] decay rate dB/s: dry={dry_decay:.2f} wet={wet_decay:.2f} dereverbed={out_decay:.2f}")

    assert dry_decay is not None
    assert wet_decay is not None
    assert out_decay is not None
    # Reverb makes the decay slower (less negative) than dry; dereverb must
    # make it measurably faster (more negative) again -- moved toward dry,
    # by a real, stated margin, not a rounding-level wobble.
    assert wet_decay > dry_decay + 50.0, "test construction sanity: wet must decay much slower than dry"
    assert out_decay < wet_decay - 3.0, (
        f"dereverbed decay rate ({out_decay:.2f} dB/s) did not improve by >= 3 dB/s over "
        f"the wet signal's ({wet_decay:.2f} dB/s)"
    )
    assert stats["amount_db"] == 10.0
    assert stats["max_gain_reduction_db"] < 0.0
    assert stats["bins_reduced_pct"] > 0.0


def test_already_dry_material_is_not_materially_damaged():
    """A signal with clear onsets/offsets and generous silence gaps (no
    reverberant smearing at all) must survive dereverb with only a modest,
    bounded overall level change -- stated tolerance: 3 dB RMS.

    This is NOT zero, and that is the honest point: a blind, single-channel
    detector cannot perfectly distinguish "this decayed because the source
    stopped" from "this decayed because it was reverberant tail" -- see
    dsp/dereverb.py's module docstring. 3 dB is the measured, tested bound
    for speech-like dry material at this module's default settings, not an
    aspirational zero.
    """
    dry_clean = _make_bursty_dry_signal(5.0, SR, seed=11, burst_range=(0.02, 0.05), gap_range=(0.4, 0.6))

    y, stats = dereverb.dereverb(dry_clean, SR, amount_db=10.0)

    before = _rms_db(dry_clean)
    after = _rms_db(y)
    delta = before - after
    print(f"\n[dereverb] already-dry material: before={before:.2f} dB after={after:.2f} dB delta={delta:.2f} dB")
    assert abs(delta) < 3.0, f"already-dry material's level moved by {delta:.2f} dB, expected < 3 dB"
    assert stats["bins_reduced_pct"] < 100.0


def test_rejects_invalid_parameters():
    x = np.zeros((SR, 1))
    with pytest.raises(ValueError, match="amount_db"):
        dereverb.dereverb(x, SR, amount_db=-1.0)
    with pytest.raises(ValueError, match="tau_ms"):
        dereverb.dereverb(x, SR, tau_ms=0.0)
    with pytest.raises(ValueError, match="guard_ms"):
        dereverb.dereverb(x, SR, guard_ms=-1.0)


def test_mono_input_round_trips_shape():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.1, SR)
    y, _stats = dereverb.dereverb(x, SR, amount_db=6.0)
    assert y.shape == x.shape
