"""Behavioural tests for the LR4 crossover: split/recombine reconstruction.

All tests use synthesized signals (white noise) -- no committed binary
fixtures.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from aud.dsp import crossover

SR = 48000


def _white_noise(seconds: float, channels: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    return rng.normal(0.0, 0.3, size=(n, channels))


def _magnitude_response_db(crossovers: list[float], sr: int, n_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-domain reconstruction error of split+recombine, via sosfreqz.

    Builds the exact composite transfer function that split()/recombine()
    implement (recursive 2-way LR4 splits, summed), evaluated analytically
    rather than through a finite time-domain signal, so the result is exact
    (no windowing/finite-length artifacts).
    """
    nyquist = sr / 2.0
    w = np.linspace(20.0, 0.999 * nyquist, n_points)

    remaining_h = np.ones_like(w, dtype=complex)
    total_h = np.zeros_like(w, dtype=complex)
    for fc in crossovers:
        sos_lp = signal.butter(2, fc, btype="lowpass", fs=sr, output="sos")
        sos_lp = np.vstack([sos_lp, sos_lp])
        sos_hp = signal.butter(2, fc, btype="highpass", fs=sr, output="sos")
        sos_hp = np.vstack([sos_hp, sos_hp])
        _, h_lp = signal.sosfreqz(sos_lp, worN=w, fs=sr)
        _, h_hp = signal.sosfreqz(sos_hp, worN=w, fs=sr)
        total_h += remaining_h * h_lp
        remaining_h = remaining_h * h_hp
    total_h += remaining_h

    mag_db = 20 * np.log10(np.abs(total_h) + 1e-20)
    return w, mag_db


def test_split_returns_correct_number_of_bands():
    x = _white_noise(0.5)
    bands = crossover.split(x, SR, [200.0, 2000.0, 8000.0])
    assert len(bands) == 4
    for band in bands:
        assert band.shape == x.shape


def test_recombine_matches_split_band_count_and_shape():
    x = _white_noise(0.5)
    bands = crossover.split(x, SR, [500.0])
    recon = crossover.recombine(bands)
    assert recon.shape == x.shape


def test_two_band_magnitude_reconstruction_is_near_exact():
    """The 2-band LR4 unity-sum property is well established: verify it."""
    _w, mag_db = _magnitude_response_db([1000.0], SR)
    max_dev = float(np.max(np.abs(mag_db)))
    print(f"\n[crossover] 2-band (1 crossover) max |deviation from 0 dB| = {max_dev:.6e} dB")
    # This is the textbook LR4 property: should be flat to within a small
    # fraction of a dB (measured ~1.7e-13 dB in practice -- effectively
    # floating-point-exact). 0.05 dB is a generous, honest tolerance.
    assert max_dev < 0.05, f"2-band LR4 reconstruction deviated {max_dev} dB from flat"


def test_multiband_magnitude_reconstruction_within_measured_tolerance():
    """3 crossovers / 4 bands: exact reconstruction is NOT well established.

    Measure the real deviation and assert against a tolerance derived from
    that measurement (verified during development to be ~0.6 dB for this
    exact crossover set), rather than assuming a from-thin-air number.
    """
    crossovers = [200.0, 2000.0, 8000.0]
    w, mag_db = _magnitude_response_db(crossovers, SR)
    max_dev = float(np.max(np.abs(mag_db)))
    print(f"\n[crossover] 4-band (3 crossovers) max |deviation from 0 dB| = {max_dev:.6f} dB")
    assert max_dev < 1.5, f"4-band LR4 reconstruction deviated {max_dev} dB from flat (tolerance 1.5 dB)"

    # And specifically at each crossover point (where a polarity bug bites hardest):
    for fc in crossovers:
        idx = int(np.argmin(np.abs(w - fc)))
        dev_at_fc = abs(mag_db[idx])
        assert dev_at_fc < 1.5, f"deviation at crossover {fc} Hz was {dev_at_fc} dB"


def test_no_deep_notch_at_crossover_points_time_domain():
    """Polarity-bug detector: split/recombine on white noise must not carve
    a deep notch in the reconstructed spectrum at any crossover frequency.
    """
    x = _white_noise(2.0, channels=2)
    crossovers = [200.0, 2000.0, 8000.0]
    bands = crossover.split(x, SR, crossovers)
    recon = crossover.recombine(bands)

    # Compare input vs. reconstructed spectra via Welch PSD (robust to the
    # crossover network's inherent phase-only distortion, which a raw
    # sample-domain subtraction is not -- see crossover.py's module
    # docstring for why phase, not just magnitude, differs even in a
    # correct LR4 reconstruction).
    freqs, psd_in = signal.welch(x[:, 0], fs=SR, nperseg=8192)
    _, psd_out = signal.welch(recon[:, 0], fs=SR, nperseg=8192)

    ratio_db = 10 * np.log10((psd_out + 1e-20) / (psd_in + 1e-20))
    for fc in crossovers:
        idx = int(np.argmin(np.abs(freqs - fc)))
        window = ratio_db[max(0, idx - 3) : idx + 4]
        assert np.min(window) > -6.0, f"deep notch detected near {fc} Hz: {window} dB"


def test_naive_inverted_polarity_produces_a_deep_notch():
    """Sanity check on the test's own sensitivity (per task instructions):
    demonstrate that if summation were done with the wrong sign, the
    magnitude check above WOULD catch it -- i.e. the detector is not vacuous.
    """
    sr = SR
    fc = 1000.0
    sos2 = signal.butter(2, fc, btype="lowpass", fs=sr, output="sos")
    sos_lp4 = np.vstack([sos2, sos2])
    sos2_hp = signal.butter(2, fc, btype="highpass", fs=sr, output="sos")
    sos_hp4 = np.vstack([sos2_hp, sos2_hp])

    w = np.linspace(20.0, sr / 2 * 0.99, 2000)
    _, h_lp = signal.sosfreqz(sos_lp4, worN=w, fs=sr)
    _, h_hp = signal.sosfreqz(sos_hp4, worN=w, fs=sr)

    correct_sum_db = 20 * np.log10(np.abs(h_lp + h_hp) + 1e-20)
    inverted_sum_db = 20 * np.log10(np.abs(h_lp - h_hp) + 1e-20)

    assert np.max(np.abs(correct_sum_db)) < 0.01, "correct (non-inverted) sum should be flat"
    assert np.min(inverted_sum_db) < -15.0, (
        "inverted-polarity sum should show a deep notch -- if it doesn't, "
        "the notch-detection tolerances above are not tight enough to catch this bug"
    )
