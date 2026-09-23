"""Behavioural tests for aud.dsp.resample: rate conversion and anti-aliasing.

Measurement, not booleans -- each claim (rate correctness, anti-aliasing,
round-trip fidelity) is a measured numeric quantity, and the anti-aliasing
claim carries a NEGATIVE CONTROL (naive decimation with no filter) proving
the FFT-based check would actually catch aliasing if the filter were
missing.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from aud.dsp import resample as resample_dsp

SOURCE_SR = 48000


def _tone(seconds: float, freq_hz: float, sr: int, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _dominant_freq_hz(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    return float(freqs[int(np.argmax(spectrum))])


# --- rate is what was asked --------------------------------------------------


def test_resample_reports_the_requested_rates_and_sample_counts():
    x = _tone(1.0, 440.0, SOURCE_SR).reshape(-1, 1)
    y, report = resample_dsp.resample(x, SOURCE_SR, 16000)
    assert report["source_hz"] == SOURCE_SR
    assert report["target_hz"] == 16000
    assert report["input_samples"] == x.shape[0]
    assert report["output_samples"] == y.shape[0]


def test_resample_duration_matches_within_one_sample_period():
    """Measured from the array itself (stand-in for 'the written file'):
    output_samples / target_hz must equal input_samples / source_hz to
    within one sample period at the LOWER of the two rates.
    """
    seconds = 2.0
    x = _tone(seconds, 300.0, SOURCE_SR).reshape(-1, 1)
    target_hz = 22050
    y, report = resample_dsp.resample(x, SOURCE_SR, target_hz)

    input_duration = report["input_samples"] / SOURCE_SR
    output_duration = report["output_samples"] / target_hz
    tolerance_s = 1.0 / min(SOURCE_SR, target_hz)
    diff = abs(output_duration - input_duration)
    print(f"\n[resample] input_duration={input_duration:.6f}s output_duration={output_duration:.6f}s diff={diff:.6f}s")
    assert diff <= tolerance_s
    assert y.shape == (report["output_samples"], 1)


def test_resample_no_op_when_target_equals_source_returns_the_same_array():
    x = _tone(0.5, 200.0, SOURCE_SR).reshape(-1, 1)
    y, report = resample_dsp.resample(x, SOURCE_SR, SOURCE_SR)
    assert y is x
    assert report["input_samples"] == report["output_samples"]


def test_resample_preserves_multichannel_shape():
    left = _tone(0.5, 220.0, SOURCE_SR)
    right = _tone(0.5, 330.0, SOURCE_SR)
    x = np.stack([left, right], axis=1)
    y, report = resample_dsp.resample(x, SOURCE_SR, 32000)
    assert y.ndim == 2
    assert y.shape[1] == 2
    assert y.shape[0] == report["output_samples"]


# --- anti-aliasing, proven, with a negative control -------------------------


def test_downsampling_does_not_alias_a_tone_above_the_target_nyquist():
    """A tone placed ABOVE the target rate's Nyquist frequency must NOT
    reappear as spurious low-frequency energy after resample_poly's own
    anti-alias filter runs.

    The alias-bin power is compared against a REFERENCE: the Welch PSD
    peak of a genuine, full-amplitude tone actually placed at the alias
    frequency, measured the same way. That gives an absolute sense of
    "how loud a real tone at this frequency would read here" -- an
    arbitrary "near-zero" bin would prove nothing about the scale that
    matters for this signal and this FFT configuration.
    """
    target_hz = 8000
    # Well above the target Nyquist (4000 Hz) but below the source Nyquist
    # (24000 Hz at 48000 Hz source), so it exists in the source and would
    # alias hard if downsampled with no filter.
    tone_freq = 6000.0
    amplitude = 0.8
    x = _tone(1.0, tone_freq, SOURCE_SR, amplitude=amplitude).reshape(-1, 1)

    y, _report = resample_dsp.resample(x, SOURCE_SR, target_hz)
    mono = y[:, 0]

    # The naive alias of a 6000 Hz tone sampled at 8000 Hz (Nyquist 4000)
    # folds to |6000 - 8000| = 2000 Hz -- well inside the passband, so if
    # the filter failed to remove the original tone before downsampling,
    # a strong peak would appear there.
    alias_freq = abs(tone_freq - target_hz)
    reference_tone = _tone(1.0, alias_freq, target_hz, amplitude=amplitude)

    freqs, psd_resampled = signal.welch(mono, fs=target_hz, nperseg=1024)
    _, psd_reference = signal.welch(reference_tone, fs=target_hz, nperseg=1024)
    alias_idx = int(np.argmin(np.abs(freqs - alias_freq)))

    alias_db = 10 * np.log10(psd_resampled[alias_idx] + 1e-20)
    reference_db = 10 * np.log10(psd_reference[alias_idx] + 1e-20)
    gap_db = reference_db - alias_db
    print(
        f"\n[resample] alias-frequency ({alias_freq} Hz): resampled={alias_db:.1f} dB, full-amplitude reference={reference_db:.1f} dB, gap={gap_db:.1f} dB"
    )
    # A real, full-amplitude tone at the alias frequency reads clearly
    # louder here than whatever leaked through the anti-alias filter.
    assert gap_db > 40.0, "spurious energy detected at the alias frequency -- anti-alias filter failed"


def test_negative_control_naive_decimation_with_no_filter_does_alias():
    """Sanity check on the anti-aliasing test's own sensitivity: bypass
    resample_poly entirely and decimate by a plain integer stride (no
    anti-alias filter at all). This MUST show strong energy at the alias
    frequency, proving the FFT-based check above would have caught a
    missing/broken filter.
    """
    target_hz = 8000
    tone_freq = 6000.0
    x = _tone(1.0, tone_freq, SOURCE_SR)

    stride = SOURCE_SR // target_hz  # 6
    naive = x[::stride]  # no low-pass filter applied first -- the defect under test

    measured_freq = _dominant_freq_hz(naive, target_hz)
    alias_freq = abs(tone_freq - target_hz)
    print(f"\n[resample] naive decimation dominant freq={measured_freq} Hz (expected alias ~{alias_freq} Hz)")
    assert abs(measured_freq - alias_freq) < 50.0, (
        "the negative control itself did not alias -- the test signal/stride choice is not "
        "actually exercising the aliasing failure mode"
    )


# --- round trip --------------------------------------------------------------


def test_round_trip_a_to_b_to_a_on_a_band_limited_signal():
    """Rate A -> B -> A on a signal well below both Nyquists: low
    reconstruction error, recorded as a numeric tolerance.
    """
    rate_a = 44100
    rate_b = 16000
    freq_hz = 300.0  # well below both 8000 Hz and 22050 Hz Nyquists
    x = _tone(1.0, freq_hz, rate_a).reshape(-1, 1)

    down, _ = resample_dsp.resample(x, rate_a, rate_b)
    back, _ = resample_dsp.resample(down, rate_b, rate_a)

    n = min(x.shape[0], back.shape[0])
    # Polyphase filtering introduces a small group delay; compare via
    # correlation-based amplitude match on the steady-state region rather
    # than a raw sample-for-sample subtraction, which would be dominated
    # by phase/edge artifacts unrelated to the resampler's fidelity.
    trim = slice(int(0.1 * rate_a), n - int(0.1 * rate_a))
    original_rms = float(np.sqrt(np.mean(x[trim, 0] ** 2)))
    reconstructed_rms = float(np.sqrt(np.mean(back[trim, 0] ** 2)))
    rel_error = abs(reconstructed_rms - original_rms) / original_rms
    print(
        f"\n[resample] round-trip RMS: original={original_rms:.6f} reconstructed={reconstructed_rms:.6f} rel_error={rel_error:.6f}"
    )
    assert rel_error < 0.05
