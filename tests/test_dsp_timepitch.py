"""Behavioural tests for the phase-vocoder time-stretch and pitch-shift."""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest

from aud.dsp import timepitch

SR = 44100

_SIGNALSMITH_INSTALLED = importlib.util.find_spec("python_stretch") is not None


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


# ---------------------------------------------------------------------------
# D4 regression: Signalsmith's `--factor` inversion, and the guard that
# would have caught it (contracts: measured output duration must match the
# requested factor, on BOTH engines, within a stated tolerance).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality", ["phase_vocoder", "signalsmith"])
@pytest.mark.parametrize("factor", [0.8, 1.2, 2.0])
def test_measured_duration_ratio_matches_requested_factor(quality, factor):
    """Real call, both engines -- the literal regression guard for D4.

    `signalsmith` is skipped-with-reason when python_stretch is not
    installed (installs here are DTU-only, see AGENTS.md); the assertion
    below is therefore UNVERIFIED against a real Signalsmith install on
    this host and is only exercised by the mocked test below instead.
    """
    if quality == "signalsmith" and not _SIGNALSMITH_INSTALLED:
        pytest.skip("python_stretch (the 'stretch' extra) is not installed on this host -- DTU-only install")

    x = _tone(220.0, seconds=1.0)
    y, stats = timepitch.time_stretch(x, SR, factor=factor, quality=quality)

    assert stats["engine"] == quality
    measured_ratio = y.shape[0] / x.shape[0]
    print(f"\n[timepitch] engine={quality} requested factor={factor} measured ratio={measured_ratio:.4f}")
    assert measured_ratio == pytest.approx(factor, rel=timepitch._STRETCH_RATIO_RELATIVE_TOLERANCE)
    assert stats["measured_factor"] == pytest.approx(measured_ratio, rel=1e-9)


class _FakeSignalsmithStretch:
    """Stands in for `python_stretch.Signalsmith.Stretch()`.

    Encodes the REAL, measured behaviour reported in D4: Signalsmith's
    `timeFactor` is the reciprocal of this module's `factor` convention, so
    `out_len == round(in_len / timeFactor)`. Exercises the actual
    `_stretch_signalsmith` code path (including its `1.0 / factor` fix) --
    this is not a stand-in for the assertion, it is a stand-in for the
    third-party library, so the fix itself is what gets tested.
    """

    def __init__(self) -> None:
        self.timeFactor = 1.0

    def preset(self, channels: int, sr: int) -> None:
        del sr
        self._channels = channels

    def process(self, audio: np.ndarray) -> np.ndarray:
        in_len = audio.shape[1]
        out_len = max(1, round(in_len / self.timeFactor))
        return np.zeros((audio.shape[0], out_len), dtype=np.float32)


def _install_fake_signalsmith(monkeypatch, stretch_cls: type) -> None:
    fake_signalsmith_ns = types.SimpleNamespace(Stretch=stretch_cls)
    fake_module = types.SimpleNamespace(Signalsmith=fake_signalsmith_ns)
    monkeypatch.setitem(sys.modules, "python_stretch", fake_module)
    monkeypatch.setattr(timepitch, "_signalsmith_available", lambda: True)


def test_signalsmith_inversion_fix_is_exercised_without_the_real_package(monkeypatch):
    """Mocked python_stretch, faithful to the measured D4 behaviour.

    Does not require the real optional dependency: it fakes only the
    third-party surface (`Signalsmith.Stretch`), and runs this module's own
    `_stretch_signalsmith`, which is where the `1.0 / factor` fix lives. If
    the inversion were reintroduced, this test fails (measured ratio would
    be ~1/factor, not factor) even on a host where python_stretch can never
    be installed.
    """
    _install_fake_signalsmith(monkeypatch, _FakeSignalsmithStretch)

    x = _tone(220.0, seconds=1.0)
    factor = 1.2
    y, stats = timepitch.time_stretch(x, SR, factor=factor, quality="signalsmith")

    assert stats["engine"] == "signalsmith"
    measured_ratio = y.shape[0] / x.shape[0]
    print(f"\n[timepitch] fake signalsmith: requested factor={factor} measured ratio={measured_ratio:.4f}")
    assert measured_ratio == pytest.approx(factor, rel=0.02)


class _BrokenSignalsmithStretch(_FakeSignalsmithStretch):
    """A hypothetical FUTURE regression: ignores `timeFactor` entirely."""

    def process(self, audio: np.ndarray) -> np.ndarray:
        return np.zeros_like(audio)


def test_ratio_guard_refuses_a_broken_engine_instead_of_returning_wrong_audio(monkeypatch):
    """The guard itself: an engine that silently ignores `factor` must be
    refused loudly, not shipped with a `measured_factor` nobody checked."""
    _install_fake_signalsmith(monkeypatch, _BrokenSignalsmithStretch)

    x = _tone(220.0, seconds=1.0)
    with pytest.raises(RuntimeError, match="measured duration ratio"):
        timepitch.time_stretch(x, SR, factor=1.2, quality="signalsmith")
