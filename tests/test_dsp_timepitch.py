"""Behavioural tests for the phase-vocoder time-stretch and pitch-shift.

The Signalsmith (python_stretch) quality tier is not installable in this
environment (extras are DTU-only). Every test below that exercises it
replays a REAL recorded python_stretch 0.3.1 run (tests/replay.py,
tests/fixtures/recorded/python_stretch/) instead of a hand-written fake --
see RECORDING.md for the measured timeFactor<->length relationship, the
float32 read-back quirk, and the methods-not-attributes quirk the replay
reproduces faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import timepitch
from tests import replay

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


# ---------------------------------------------------------------------------
# D4 regression: Signalsmith's `--factor` inversion, and the guard that
# would have caught it (contracts: measured output duration must match the
# requested factor, on BOTH engines, within a stated tolerance).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality", ["phase_vocoder", "signalsmith"])
@pytest.mark.parametrize("factor", [0.8, 1.2, 2.0])
def test_measured_duration_ratio_matches_requested_factor(quality, factor, monkeypatch):
    """Real call, both engines -- the literal regression guard for D4.

    `signalsmith` no longer skips when python_stretch is not installed:
    the real recording (tests/fixtures/recorded/python_stretch/) replays
    the measured timeFactor<->length relationship instead, so this runs
    (and would catch a reintroduced inversion) on any host, DTU or not.
    """
    if quality == "signalsmith":
        replay.install_python_stretch_replay(monkeypatch)

    x = _tone(220.0, seconds=1.0)
    y, stats = timepitch.time_stretch(x, SR, factor=factor, quality=quality)

    assert stats["engine"] == quality
    measured_ratio = y.shape[0] / x.shape[0]
    print(f"\n[timepitch] engine={quality} requested factor={factor} measured ratio={measured_ratio:.4f}")
    assert measured_ratio == pytest.approx(factor, rel=timepitch._STRETCH_RATIO_RELATIVE_TOLERANCE)
    assert stats["measured_factor"] == pytest.approx(measured_ratio, rel=1e-9)


def test_signalsmith_inversion_fix_is_exercised_via_the_real_recorded_relationship(monkeypatch):
    """Replays the REAL recorded python_stretch behaviour (RECORDING.md:
    `timeFactor` is the reciprocal of this module's `factor` convention --
    measured directly, six factors, `1.2 -> 0.8333` reproduced exactly),
    exercising the actual `_stretch_signalsmith` code path (including its
    `1.0 / factor` fix). If the inversion were reintroduced, this test
    fails (measured ratio would be ~1/factor, not factor) even on a host
    where python_stretch can never be installed.
    """
    replay.install_python_stretch_replay(monkeypatch)

    x = _tone(220.0, seconds=1.0)
    factor = 1.2
    y, stats = timepitch.time_stretch(x, SR, factor=factor, quality="signalsmith")

    assert stats["engine"] == "signalsmith"
    measured_ratio = y.shape[0] / x.shape[0]
    print(f"\n[timepitch] replayed signalsmith: requested factor={factor} measured ratio={measured_ratio:.4f}")
    assert measured_ratio == pytest.approx(factor, rel=0.02)


def test_ratio_guard_refuses_a_broken_engine_instead_of_returning_wrong_audio():
    """The guard itself: an engine that silently ignores `factor` must be
    refused loudly, not shipped with a `measured_factor` nobody checked.

    `_assert_ratio_holds` is `aud`'s OWN pure guard function -- calling it
    directly with a broken (engine, factor, n_in, n_out) combination tests
    our guard, not any third-party behaviour, so no replay or fake of any
    kind is needed here at all.
    """
    with pytest.raises(RuntimeError, match="measured duration ratio"):
        timepitch._assert_ratio_holds(engine="signalsmith", factor=1.2, n_in=22050, n_out=22050)


def test_signalsmith_replay_reproduces_the_recorded_byte_exact_audio_pair(monkeypatch):
    """The strongest evidence available: replay python_stretch against the
    EXACT recorded input array (not a freshly synthesised tone) and assert
    the output is the exact recorded output array, sample for sample --
    not just a matching length/ratio.
    """
    replay.install_python_stretch_replay(monkeypatch)
    input_arr, expected_output = replay.load_stretch_audio_pair(time_factor=2.0)

    y, stats = timepitch.time_stretch(input_arr, sr=22050, factor=0.5, quality="signalsmith")

    assert stats["engine"] == "signalsmith"
    assert y.shape[0] == expected_output.shape[0]
    np.testing.assert_allclose(y, expected_output.astype(np.float64), atol=1e-6)


def test_signalsmith_replay_reproduces_the_float32_time_factor_readback(monkeypatch):
    """RECORDING.md: `timeFactor` reads back through float32, e.g. setting
    0.8 reads back as 0.800000011920929, not the Python float exactly.
    A replay tidier than this (storing the double precisely) would hide a
    real precision detail a caller might depend on.
    """
    recording = replay.install_python_stretch_replay(monkeypatch)
    del recording
    import python_stretch as ps  # the installed replay module

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    stretch.timeFactor = 0.8
    assert stretch.timeFactor == pytest.approx(0.800000011920929, abs=1e-15)
    assert stretch.timeFactor != 0.8  # the exact float32 quirk, not a tolerant approximation


def test_signalsmith_replay_reproduces_the_methods_not_attributes_quirk(monkeypatch):
    """RECORDING.md: `inputLatency`/`outputLatency`/`blockSamples`/
    `intervalSamples` are METHODS on the real object, not attributes --
    `float(stretch.inputLatency)` raises TypeError on the real library
    (naming nanobind's bound-method type; Python's own `float()` builtin
    names whatever the real type is, so this replay's plain-Python method
    raises the same TypeError naming Python's own method type instead).
    The fact this raises TypeError at all -- rather than returning a float
    -- is the faithfully-reproduced quirk: a replay that made these plain
    floats would be tidier than reality, which is exactly the shape of
    defect this harness exists to prevent.
    """
    replay.install_python_stretch_replay(monkeypatch)
    import python_stretch as ps

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    with pytest.raises(TypeError, match="float\\(\\) argument must be a string or a real number"):
        float(stretch.inputLatency)
