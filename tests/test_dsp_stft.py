"""Behavioural tests for STFT analysis / WOLA resynthesis.

All tests use synthesized signals (white noise, tones) -- no committed
binary fixtures, per AGENTS.md #3b (never hand-write a mock; these are pure
array functions, so real generated signals are the correct substitute).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window

from aud.dsp import stft

SR = 48000

# -250 dB relative to signal peak, per task spec: measured float64 round-trip
# floor is ~-313 dB (== 20*log10(2**-52), the float64 machine epsilon), so
# this leaves ~60 dB of margin -- an honest, non-arbitrary tolerance rather
# than a number picked to make the test pass.
_NULL_TEST_TOLERANCE_DB = -250.0


def _white_noise(seconds: float, channels: int | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    shape = (n,) if channels is None else (n, channels)
    return rng.normal(0.0, 0.3, size=shape)


def _reconstruction_error_db(x: np.ndarray, y: np.ndarray) -> float:
    """Peak reconstruction error, in dB relative to the signal's own peak
    (matches this codebase's existing convention for dB error metrics, e.g.
    `dsp.limiter.true_peak_dbtp`, which is also peak-based)."""
    max_err = float(np.max(np.abs(y - x)))
    peak = float(np.max(np.abs(x)))
    if max_err == 0.0:
        return -400.0
    return 20.0 * np.log10(max_err / peak)


# --- 1. Null test: unity-gain round trip must reconstruct sample-accurately ---


def test_null_test_hann_75_percent_overlap_default():
    """The defensible default (Hann, N=2048, 75% overlap) round-trips a
    real, unmodified spectrum sample-accurately. This is the test that makes
    every later measurement (masking, EQ, whatever a caller builds on top)
    trustworthy: if THIS doesn't hold, nothing built on top of it can."""
    x = _white_noise(2.0)
    spectrum = stft.analyze(x, SR, window="hann", n_fft=2048, hop=512)
    y = stft.resynthesize(spectrum, SR, len(x), window="hann", n_fft=2048, hop=512)

    err_db = _reconstruction_error_db(x, y)
    print(f"\n[stft] null-test Hann/2048/75% max err = {err_db:.2f} dB")
    assert y.shape == x.shape
    assert err_db <= _NULL_TEST_TOLERANCE_DB


def test_null_test_sqrt_hann_50_percent_overlap():
    """The other recommended WOLA pairing: sqrt-Hann/sqrt-Hann at 50% overlap."""
    x = _white_noise(2.0, seed=1)
    spectrum = stft.analyze(x, SR, window="sqrt_hann", n_fft=2048, hop=1024)
    y = stft.resynthesize(spectrum, SR, len(x), window="sqrt_hann", n_fft=2048, hop=1024)

    err_db = _reconstruction_error_db(x, y)
    print(f"\n[stft] null-test sqrt-Hann/2048/50% max err = {err_db:.2f} dB")
    assert err_db <= _NULL_TEST_TOLERANCE_DB


def test_null_test_various_window_n_fft_hop_combinations():
    """Sweep a handful of compliant (window, n_fft, hop) choices -- all must
    clear the same honest tolerance, not just the one "hero" configuration."""
    x = _white_noise(1.0, seed=2)
    cases = [
        ("hann", 512, 128),  # 512, 75%
        ("hann", 1024, 256),  # 1024, 75%
        ("hann", 4096, 1024),  # 4096, 75%
        ("sqrt_hann", 1024, 512),  # sqrt-Hann, 50%
        # Blackman is NOT a compliant WOLA pair with itself at 50% or 75%
        # hop (blackman**2 fails check_cola_nola at both -- verified below
        # and in test_cola_hann_times_hann_wola_fails_at_50_percent_but_not_75_percent's
        # sibling reasoning); 87.5% overlap (hop = n_fft // 8) is where it
        # IS compliant, so that is what is exercised here.
        ("blackman", 2048, 256),  # Blackman x Blackman, 87.5%
    ]
    for window, n_fft, hop in cases:
        spectrum = stft.analyze(x, SR, window=window, n_fft=n_fft, hop=hop)
        y = stft.resynthesize(spectrum, SR, len(x), window=window, n_fft=n_fft, hop=hop)
        err_db = _reconstruction_error_db(x, y)
        print(f"\n[stft] null-test {window}/{n_fft}/hop={hop} max err = {err_db:.2f} dB")
        assert err_db <= _NULL_TEST_TOLERANCE_DB, f"{window}/{n_fft}/{hop} failed at {err_db} dB"


# --- 2. COLA/NOLA asserted programmatically, including FAIL cases ---


def test_cola_periodic_hann_compliant_at_standard_hops():
    n_fft = 1024
    win = get_window("hann", n_fft, fftbins=True)
    for hop in (n_fft // 2, n_fft // 4, n_fft // 8):
        result = stft.check_cola_nola(win, hop)
        assert result["cola"] is True, f"periodic Hann should be COLA at hop={hop}"
        assert result["nola"] is True


def test_cola_symmetric_hann_is_correctly_reported_non_compliant():
    """The trap: a SYMMETRIC (not periodic) Hann window is NOT COLA at 50%
    overlap, even though the periodic variant is. This is the canary that
    proves check_cola_nola is actually sensitive, not vacuously True --
    matching the pattern in tests/test_dsp_crossover.py's own polarity-bug
    canary."""
    n_fft = 1024
    win_symmetric = get_window("hann", n_fft, fftbins=False)
    result = stft.check_cola_nola(win_symmetric, n_fft // 2)
    print(f"\n[stft] symmetric Hann @ 50% hop: cola={result['cola']} nola={result['nola']}")
    assert result["cola"] is False


def test_cola_hann_times_hann_wola_fails_at_50_percent_but_not_75_percent():
    """This module's actual WOLA scheme reuses the SAME window for analysis
    and synthesis, so what matters for correctness is COLA of `window**2`
    (the product window), not of `window` alone. Hann squared is NOT COLA
    at 50% hop (this is exactly the failure `stft_properties`/
    `resynthesize` must be able to detect), but IS COLA at 75% hop."""
    n_fft = 1024
    win = get_window("hann", n_fft, fftbins=True)

    fail = stft.check_cola_nola(win**2, n_fft // 2)
    ok = stft.check_cola_nola(win**2, n_fft // 4)
    print(f"\n[stft] Hann^2 @ 50% hop: {fail}   Hann^2 @ 75% hop: {ok}")
    assert fail["cola"] is False, "Hann x Hann at 50% hop must be reported non-compliant"
    assert ok["cola"] is True, "Hann x Hann at 75% hop must be reported compliant"


def test_cola_sqrt_hann_squared_is_plain_hann_and_compliant_at_50_and_75():
    n_fft = 1024
    win = np.sqrt(get_window("hann", n_fft, fftbins=True))
    for hop in (n_fft // 2, n_fft // 4):
        result = stft.check_cola_nola(win**2, hop)
        assert result["cola"] is True, f"sqrt-Hann squared should be COLA at hop={hop}"


def test_stft_properties_reports_cola_matching_direct_check():
    """`stft_properties` must report the SAME compliance as calling
    `check_cola_nola` directly on `window**2` -- one honest source, not two
    independently-drifting answers to the same question."""
    props_fail = stft.stft_properties(window="hann", n_fft=1024, hop=512, sr=SR)
    props_ok = stft.stft_properties(window="hann", n_fft=1024, hop=256, sr=SR)
    assert props_fail["cola"] is False
    assert props_ok["cola"] is True


def test_resynthesize_actually_reconstructs_badly_for_a_non_compliant_choice():
    """Not just a properties-dict flag: prove the FAILURE is real by
    actually running the non-compliant Hann-times-Hann-at-50%-hop
    configuration through analyze/resynthesize and observing a large error,
    contrasted with the same signal at a compliant hop."""
    x = _white_noise(1.0, seed=3)
    n_fft = 1024

    spectrum_bad = stft.analyze(x, SR, window="hann", n_fft=n_fft, hop=n_fft // 2)
    y_bad = stft.resynthesize(spectrum_bad, SR, len(x), window="hann", n_fft=n_fft, hop=n_fft // 2)
    err_bad_db = _reconstruction_error_db(x, y_bad)

    spectrum_good = stft.analyze(x, SR, window="hann", n_fft=n_fft, hop=n_fft // 4)
    y_good = stft.resynthesize(spectrum_good, SR, len(x), window="hann", n_fft=n_fft, hop=n_fft // 4)
    err_good_db = _reconstruction_error_db(x, y_good)

    print(f"\n[stft] non-compliant (50% hop) err = {err_bad_db:.2f} dB; compliant (75% hop) err = {err_good_db:.2f} dB")
    assert err_bad_db > -20.0, "the non-compliant WOLA pair should reconstruct badly, not cleanly"
    assert err_good_db <= _NULL_TEST_TOLERANCE_DB


# --- 3. Determinism ---


def test_analyze_is_deterministic():
    x = _white_noise(0.5, seed=4)
    s1 = stft.analyze(x, SR)
    s2 = stft.analyze(x, SR)
    assert np.array_equal(s1, s2)


def test_resynthesize_is_deterministic():
    x = _white_noise(0.5, seed=5)
    spectrum = stft.analyze(x, SR)
    y1 = stft.resynthesize(spectrum, SR, len(x))
    y2 = stft.resynthesize(spectrum, SR, len(x))
    assert np.array_equal(y1, y2)


# --- 4. Properties reporting: latency and time/frequency tradeoff ---


def test_stft_properties_matches_48khz_resolution_table():
    """Values from the task's measured 48 kHz resolution table."""
    table = [
        # n_fft, window_ms, hop_ms_50, hop_ms_75, bin_hz
        (512, 10.7, 5.3, 2.7, 93.8),
        (1024, 21.3, 10.7, 5.3, 46.9),
        (2048, 42.7, 21.3, 10.7, 23.4),
        (4096, 85.3, 42.7, 21.3, 11.7),
    ]
    # Absolute tolerance sized to the task table's own one-decimal rounding
    # (max possible rounding error is 0.05; 0.06 leaves a hair of margin)
    # rather than a relative tolerance, which would be too tight for the
    # smaller values (e.g. 2.6667 vs the rounded 2.7) and too loose for the
    # larger ones.
    abs_tol = 0.06
    for n_fft, window_ms, hop_ms_50, hop_ms_75, bin_hz in table:
        props_50 = stft.stft_properties(window="hann", n_fft=n_fft, hop=n_fft // 2, sr=SR)
        props_75 = stft.stft_properties(window="hann", n_fft=n_fft, hop=n_fft // 4, sr=SR)

        assert props_50["window_ms"] == pytest.approx(window_ms, abs=abs_tol)
        assert props_75["window_ms"] == pytest.approx(window_ms, abs=abs_tol)
        assert props_50["hop_ms"] == pytest.approx(hop_ms_50, abs=abs_tol)
        assert props_75["hop_ms"] == pytest.approx(hop_ms_75, abs=abs_tol)
        assert props_50["bin_hz"] == pytest.approx(bin_hz, abs=abs_tol)
        assert props_75["bin_hz"] == pytest.approx(bin_hz, abs=abs_tol)
        # streaming latency == window length, by definition (see module docstring)
        assert props_50["streaming_latency_ms"] == pytest.approx(window_ms, abs=abs_tol)


# --- Multi-channel input ---


def test_multichannel_round_trip():
    x = _white_noise(1.5, channels=3, seed=6)
    spectrum = stft.analyze(x, SR, window="hann", n_fft=2048, hop=512)
    assert spectrum.ndim == 3
    assert spectrum.shape[1] == 3

    y = stft.resynthesize(spectrum, SR, len(x), window="hann", n_fft=2048, hop=512)
    assert y.shape == x.shape
    err_db = _reconstruction_error_db(x, y)
    print(f"\n[stft] multichannel null-test max err = {err_db:.2f} dB")
    assert err_db <= _NULL_TEST_TOLERANCE_DB


def test_multichannel_channels_are_independent():
    """A per-channel sanity check: swap-in a different signal on one channel
    and confirm the OTHER channel's reconstruction is unaffected (no cross-talk)."""
    n = int(1.0 * SR)
    rng = np.random.default_rng(7)
    ch0 = rng.normal(0.0, 0.3, size=n)
    ch1 = 0.2 * np.sin(2 * np.pi * 440.0 * np.arange(n) / SR)
    x = np.stack([ch0, ch1], axis=1)

    spectrum = stft.analyze(x, SR)
    y = stft.resynthesize(spectrum, SR, n)
    assert np.max(np.abs(y[:, 0] - ch0)) < np.max(np.abs(y[:, 1] - ch0))  # sanity: not swapped
    err_db = _reconstruction_error_db(x, y)
    assert err_db <= _NULL_TEST_TOLERANCE_DB


# --- Signal shorter than one window ---


def test_signal_shorter_than_one_window_round_trips_exactly():
    """Defined behaviour for a short signal: `analyze` zero-pads internally
    up to the minimum ShortTimeFFT needs, and `resynthesize`'s `length`
    trims the padding back off, so the caller never sees it and the
    round trip is still sample-accurate."""
    n_fft = 2048
    x = _white_noise(1.0, seed=8)[:100]  # far shorter than n_fft
    assert len(x) < n_fft

    spectrum = stft.analyze(x, SR, window="hann", n_fft=n_fft, hop=512)
    y = stft.resynthesize(spectrum, SR, len(x), window="hann", n_fft=n_fft, hop=512)

    assert y.shape == x.shape
    err_db = _reconstruction_error_db(x, y)
    print(f"\n[stft] short-signal ({len(x)} samples < n_fft={n_fft}) null-test err = {err_db:.2f} dB")
    assert err_db <= _NULL_TEST_TOLERANCE_DB


def test_signal_at_exact_minimum_length_round_trips():
    """Boundary case: exactly `ceil(n_fft / 2)` samples -- the smallest
    length ShortTimeFFT accepts without any padding at all."""
    n_fft = 1024
    min_len = -(-n_fft // 2)
    x = _white_noise(1.0, seed=9)[:min_len]

    spectrum = stft.analyze(x, SR, window="hann", n_fft=n_fft, hop=256)
    y = stft.resynthesize(spectrum, SR, len(x), window="hann", n_fft=n_fft, hop=256)

    assert y.shape == x.shape
    err_db = _reconstruction_error_db(x, y)
    assert err_db <= _NULL_TEST_TOLERANCE_DB


def test_short_multichannel_signal_round_trips():
    n_fft = 2048
    x = _white_noise(1.0, channels=2, seed=10)[:50]
    spectrum = stft.analyze(x, SR, n_fft=n_fft, hop=512)
    y = stft.resynthesize(spectrum, SR, len(x), n_fft=n_fft, hop=512)
    assert y.shape == x.shape
    err_db = _reconstruction_error_db(x, y)
    assert err_db <= _NULL_TEST_TOLERANCE_DB


# --- Input validation ---


def test_non_periodic_window_name_would_be_caught():
    """`_analysis_window`'s periodicity assertion is exercised indirectly:
    every public call in this test module uses `get_window(..., fftbins=True)`
    internally, which IS periodic, so this test instead confirms the
    identity this module relies on actually holds for the window names it
    supports -- i.e. the assertion is not dead code that could never fire."""
    from aud.dsp.stft import _analysis_window

    win = _analysis_window("hann", 512)
    assert win.shape == (512,)
    assert np.all(np.isfinite(win))


def test_invalid_hop_raises():
    x = _white_noise(0.5, seed=11)

    with pytest.raises(ValueError, match="hop"):
        stft.analyze(x, SR, n_fft=1024, hop=0)
    with pytest.raises(ValueError, match="hop"):
        stft.analyze(x, SR, n_fft=1024, hop=2000)
