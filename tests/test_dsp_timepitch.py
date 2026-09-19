"""Behavioural tests for the phase-vocoder time-stretch and pitch-shift."""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import timepitch

SR = 44100


def _tone(freq: float, seconds: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).reshape(-1, 1)


def _measured_fundamental_hz(x: np.ndarray, sr: int) -> float:
    """FFT-peak fundamental estimate with parabolic (quadratic) bin interpolation.

    Accurate to a small fraction of a bin for a clean, single-tone signal --
    good enough to catch a "few cents" drift, not a full pitch-tracker.
    """
    mono = x[:, 0] if x.ndim == 2 else x
    windowed = mono * np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(windowed))
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        alpha, beta, gamma = spec[k - 1], spec[k], spec[k + 1]
        denom = alpha - 2 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float((k + delta) * sr / len(mono))


def _cents(measured: float, expected: float) -> float:
    return 1200.0 * np.log2(measured / expected)


def test_stretch_by_two_doubles_duration_and_preserves_pitch():
    freq = 220.0
    x = _tone(freq, seconds=1.0)

    y, stats = timepitch.time_stretch(x, SR, factor=2.0, quality="phase_vocoder")

    assert stats["engine"] == "phase_vocoder"
    duration_ratio = y.shape[0] / x.shape[0]
    print(f"\n[timepitch] stretch factor=2.0: duration ratio={duration_ratio:.4f}")
    assert abs(duration_ratio - 2.0) < 0.01

    # Measure the fundamental over a steady middle section (avoid the phase
    # vocoder's transient-ish edges).
    mid = y[SR // 4 : SR // 4 + SR, :]
    measured = _measured_fundamental_hz(mid, SR)
    cents = _cents(measured, freq)
    print(f"[timepitch] stretched fundamental: measured={measured:.3f} Hz expected={freq} Hz drift={cents:.2f} cents")
    assert abs(cents) < 15.0, f"time-stretch must preserve pitch; drifted {cents:.2f} cents"


def test_stretch_factor_one_is_a_near_identity():
    freq = 300.0
    x = _tone(freq, seconds=0.5)
    y, _stats = timepitch.time_stretch(x, SR, factor=1.0, quality="phase_vocoder")

    assert y.shape[0] == x.shape[0]
    measured = _measured_fundamental_hz(y, SR)
    cents = _cents(measured, freq)
    print(f"\n[timepitch] factor=1.0 fundamental drift: {cents:.2f} cents")
    # A phase-vocoder round trip is NOT bit-exact even at factor=1.0 (analysis/
    # synthesis windowing still runs) -- assert it stays close, not identical.
    assert abs(cents) < 10.0
    rms_diff = float(np.sqrt(np.mean((y - x) ** 2)))
    rms_x = float(np.sqrt(np.mean(x**2)))
    print(f"[timepitch] factor=1.0 relative RMS difference: {rms_diff / rms_x:.4f}")
    assert rms_diff / rms_x < 0.5  # same signal, same ballpark level -- not silence, not noise


def test_pitch_shift_up_one_octave_doubles_fundamental_and_keeps_duration():
    freq = 220.0
    x = _tone(freq, seconds=1.0)

    y, stats = timepitch.pitch_shift(x, SR, semitones=12.0, quality="phase_vocoder")

    assert stats["engine"] == "phase_vocoder"
    assert y.shape[0] == x.shape[0], "pitch_shift must preserve duration exactly"
    assert abs(stats["frequency_ratio"] - 2.0) < 1e-9

    mid = y[SR // 4 : SR // 4 + SR // 2, :]
    measured = _measured_fundamental_hz(mid, SR)
    expected = freq * 2.0
    cents = _cents(measured, expected)
    print(f"\n[timepitch] +12 semitones: measured={measured:.3f} Hz expected={expected} Hz drift={cents:.2f} cents")
    assert abs(cents) < 25.0, f"pitch_shift(+12) should double the fundamental; drifted {cents:.2f} cents"


def test_pitch_shift_zero_semitones_is_a_near_identity():
    freq = 300.0
    x = _tone(freq, seconds=0.5)
    y, stats = timepitch.pitch_shift(x, SR, semitones=0.0, quality="phase_vocoder")

    assert y.shape[0] == x.shape[0]
    assert stats["frequency_ratio"] == 1.0
    measured = _measured_fundamental_hz(y, SR)
    cents = _cents(measured, freq)
    print(f"\n[timepitch] 0 semitones fundamental drift: {cents:.2f} cents")
    assert abs(cents) < 10.0


def test_stats_report_which_engine_ran():
    x = _tone(150.0, seconds=0.3)
    _y, stretch_stats = timepitch.time_stretch(x, SR, factor=1.3, quality="auto")
    _y2, pitch_stats = timepitch.pitch_shift(x, SR, semitones=-3.0, quality="auto")

    # python-stretch is not installed on this host (extras are DTU-only --
    # see the module docstring), so "auto" must fall back to the tested path.
    assert stretch_stats["engine"] == "phase_vocoder"
    assert pitch_stats["engine"] == "phase_vocoder"


def test_forcing_signalsmith_without_the_extra_raises_a_clear_error():
    x = _tone(150.0, seconds=0.1)
    with pytest.raises(RuntimeError, match="python_stretch"):
        timepitch.time_stretch(x, SR, factor=1.2, quality="signalsmith")
