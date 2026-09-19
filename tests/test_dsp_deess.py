"""Behavioural tests for the dynamic de-esser.

The defining property under test is DYNAMIC, not static: a fixed EQ cut
would dull every "s" (and everything else sharing that band) all the time.
A de-esser must only move when sibilance is actually present -- so these
tests assert both halves: sibilant bursts get pulled down by roughly the
requested amount, AND non-sibilant/quiet passages come out (very close to)
unchanged.

Methodology note: `aud.dsp.crossover`'s own docstring and test suite
(tests/test_dsp_crossover.py) are explicit that a multi-band LR4
split/recombine is a magnitude-flat network, not a sample-for-sample
identity one (phase is not preserved tap-for-tap). So "unchanged" here is
measured as RMS level in dB, the same methodology the crossover tests use,
not raw sample-domain subtraction -- see dsp/deess.py's module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from aud.dsp import deess

SR = 48000
_EPS = 1e-12


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x * x)) + _EPS)


def _band_rms_db(x: np.ndarray, sr: int, low: float, high: float) -> float:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfilt(sos, x, axis=0)
    return 20.0 * np.log10(np.sqrt(np.mean(band * band)) + _EPS)


def _tone_with_sibilant_bursts(seconds: float, sr: int, burst_windows: list[tuple[float, float]]) -> np.ndarray:
    """A steady 1 kHz tone (well below the default sibilant band) plus
    periodic band-limited-noise bursts inside the default band (4500-9000
    Hz), active only during `burst_windows` (each a (start_s, end_s) pair).
    """
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = (10 ** (-20.0 / 20.0)) * np.sin(2 * np.pi * 1000.0 * t)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, n)
    sos_bp = signal.butter(4, [5500.0, 7500.0], btype="bandpass", fs=sr, output="sos")
    band_noise = signal.sosfilt(sos_bp, noise)
    band_noise = band_noise / np.sqrt(np.mean(band_noise**2)) * (10 ** (-10.0 / 20.0))

    mask = np.zeros(n, dtype=bool)
    for start_s, end_s in burst_windows:
        mask[int(start_s * sr) : int(end_s * sr)] = True

    x = tone + band_noise * mask
    return np.stack([x, x], axis=1)


def test_sibilant_burst_is_attenuated_by_roughly_the_requested_amount():
    x = _tone_with_sibilant_bursts(3.0, SR, burst_windows=[(0.5, 1.0), (2.0, 2.5)])

    y, stats = deess.deess(x, SR, amount_db=10.0)

    # Steady middle of the first burst, well clear of the attack/release edges.
    seg = slice(int(0.6 * SR), int(0.9 * SR))
    before = _band_rms_db(x[seg], SR, 5500.0, 7500.0)
    after = _band_rms_db(y[seg], SR, 5500.0, 7500.0)
    reduction = before - after
    print(f"\n[deess] sibilant burst: before={before:.2f} dB after={after:.2f} dB reduction={reduction:.2f} dB")
    assert 8.0 < reduction < 11.5, f"expected ~10 dB reduction in the sibilant band, measured {reduction:.2f} dB"

    assert stats["amount_db"] == 10.0
    assert stats["max_gain_reduction_db"] <= -8.0
    assert stats["frames_reduced_pct"] > 0.0


def test_non_sibilant_passage_is_unchanged_within_a_tight_tolerance():
    x = _tone_with_sibilant_bursts(3.0, SR, burst_windows=[(0.5, 1.0), (2.0, 2.5)])
    y, _stats = deess.deess(x, SR, amount_db=10.0)

    # A stretch with no burst active: only the steady 1 kHz tone is present.
    seg = slice(int(1.5 * SR), int(1.9 * SR))
    before = _rms_db(x[seg])
    after = _rms_db(y[seg])
    delta = abs(before - after)
    print(f"\n[deess] non-sibilant passage: before={before:.4f} dB after={after:.4f} dB |delta|={delta:.4f} dB")
    assert delta < 0.1, f"non-sibilant passage moved by {delta:.4f} dB, expected < 0.1 dB"


def test_signal_with_no_sibilance_at_all_passes_through_essentially_untouched():
    seconds = 1.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    tone = (10 ** (-20.0 / 20.0)) * np.sin(2 * np.pi * 1000.0 * t)
    x = np.stack([tone, tone], axis=1)

    y, stats = deess.deess(x, SR, amount_db=10.0)

    before = _rms_db(x)
    after = _rms_db(y)
    delta = abs(before - after)
    print(f"\n[deess] no-sibilance signal: before={before:.4f} dB after={after:.4f} dB |delta|={delta:.4f} dB")
    assert delta < 0.1, f"no-sibilance signal moved by {delta:.4f} dB, expected < 0.1 dB"
    assert stats["max_gain_reduction_db"] == 0.0
    assert stats["avg_gain_reduction_db"] == 0.0
    assert stats["frames_reduced_pct"] == 0.0


def test_amount_db_is_a_true_ceiling_never_exceeded():
    """Even a very loud, sustained sibilant burst must not be reduced by
    more than `amount_db` -- it is documented as a maximum, not a ratio."""
    n = int(1.0 * SR)
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, n)
    sos_bp = signal.butter(4, [5500.0, 7500.0], btype="bandpass", fs=SR, output="sos")
    loud_sibilance = signal.sosfilt(sos_bp, noise)
    loud_sibilance = loud_sibilance / np.max(np.abs(loud_sibilance) + 1e-9) * 0.9
    x = np.stack([loud_sibilance, loud_sibilance], axis=1)

    y, stats = deess.deess(x, SR, amount_db=6.0)

    # The authoritative check: the per-sample gain applied inside deess()
    # itself never exceeds the documented ceiling.
    assert stats["max_gain_reduction_db"] >= -6.05  # small float slack

    # Cross-check via an independent external measurement filter. Its edges
    # (5500-7500 Hz) deliberately do NOT match deess()'s own processing band
    # (4500-9000 Hz) or the crossover network's transition regions, so this
    # is a looser sanity bound, not a second precise measurement of the cap.
    before = _band_rms_db(x, SR, 5500.0, 7500.0)
    after = _band_rms_db(y, SR, 5500.0, 7500.0)
    print(f"\n[deess] ceiling check: reduction={before - after:.2f} dB, cap=6 dB")
    assert (before - after) < 8.0


def test_rejects_invalid_band_and_negative_amount():
    x = np.zeros((SR, 2))
    with pytest.raises(ValueError, match="0 < low < high"):
        deess.deess(x, SR, band=(9000.0, 4500.0))
    with pytest.raises(ValueError, match="amount_db"):
        deess.deess(x, SR, amount_db=-1.0)
    with pytest.raises(ValueError, match="Nyquist"):
        deess.deess(x, SR, band=(4500.0, 30000.0))
