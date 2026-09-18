"""Behavioural tests for the true-peak aware lookahead brickwall limiter."""

from __future__ import annotations

import numpy as np

from aud.dsp import limiter

SR = 44100


def _intersample_peak_signal(n: int = 4000, sr: int = SR) -> np.ndarray:
    """A near-Nyquist sine, phase-tuned so sample peak stays near -0dBFS
    while the true (reconstructed) peak overshoots well above 0 dBTP.

    Parameters found by a small offline search (see agent report): at
    freq = 0.49*sr with this phase, sample peak sits at ~ -0.005 dBFS
    (i.e. "no clipping" by a naive sample-peak measure) while the true
    peak reaches ~ +2.26 dBTP -- a textbook intersample overshoot.
    """
    t = np.arange(n) / sr
    freq = 0.49 * sr
    phase = np.deg2rad(84.234)
    x = 0.9997 * np.sin(2 * np.pi * freq * t + phase)
    return np.stack([x, x], axis=1)


def test_true_peak_catches_intersample_overs_that_sample_peak_misses():
    x = _intersample_peak_signal()
    sample_peak_dbfs = 20.0 * np.log10(np.max(np.abs(x)))
    true_peak = limiter.true_peak_dbtp(x, SR, oversample=4)

    print(f"\n[limiter] sample_peak_dbfs={sample_peak_dbfs:.4f}  true_peak_dbtp={true_peak:.4f}")

    assert sample_peak_dbfs < 0.05, "constructed signal should look like it does not clip by sample peak"
    assert true_peak > 1.0, "true-peak measurement should reveal a real intersample overshoot"
    assert true_peak - sample_peak_dbfs > 1.0, "true peak should exceed sample peak by a clearly measurable margin"


def test_brickwall_brings_true_peak_under_ceiling():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.35, size=(SR, 2))
    tone = _intersample_peak_signal(n=SR, sr=SR)
    x = 0.6 * noise + 0.6 * tone  # hot, mixed-content programme material

    input_tp = limiter.true_peak_dbtp(x, SR, oversample=4)
    assert input_tp > -1.0, "test signal should start above the ceiling we're about to enforce"

    ceiling = -1.0
    y, stats = limiter.brickwall(x, SR, ceiling_dbtp=ceiling, lookahead_ms=5.0, release_ms=100.0, oversample=4)

    print(f"\n[limiter] input_true_peak={stats['input_true_peak_dbtp']:.4f} dBTP")
    print(f"[limiter] output_true_peak={stats['output_true_peak_dbtp']:.4f} dBTP (ceiling={ceiling})")
    print(f"[limiter] max_gain_reduction={stats['max_gain_reduction_db']:.4f} dB")

    assert stats["ceiling_met"] is True
    assert stats["output_true_peak_dbtp"] <= ceiling + 0.1
    assert y.shape == x.shape
    assert stats["max_gain_reduction_db"] < 0.0


def test_brickwall_reports_ceiling_met_honestly():
    """The stats dict must never claim a ceiling was met when it wasn't."""
    x = _intersample_peak_signal(n=2000)
    _y, stats = limiter.brickwall(x, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=100.0, oversample=4)
    # Sanity: whatever the real outcome is, ceiling_met must agree with the
    # actual measured output true peak (the honesty contract), not just
    # assert True blindly.
    actual_met = stats["output_true_peak_dbtp"] <= stats["ceiling_dbtp"] + 0.05
    assert stats["ceiling_met"] == actual_met


def test_quiet_signal_is_left_essentially_unchanged():
    """A signal well under the ceiling should pass through unchanged --
    except at the very edges, which carry two well-understood, inherent
    artifacts, not bugs: (1) the *start* has `lookahead_ms` worth of
    algorithmic latency (there is no prior signal to delay from at t=0,
    so those samples are zero-padded -- true of any lookahead limiter),
    and (2) the *end* has a brief FIR edge-boundary ripple from the
    polyphase oversampling filter. Both are measured and bounded here
    rather than hand-waved away.
    """
    x = 0.01 * np.ones((SR // 4, 2))
    y, stats = limiter.brickwall(x, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0)

    margin = 300  # > lookahead (5ms @ 44.1kHz ~= 220 samples) + filter settle
    steady_state_diff = np.max(np.abs(y[margin:-margin] - x[margin:-margin]))
    print(f"\n[limiter] quiet-signal steady-state max diff: {steady_state_diff:.2e}")
    assert steady_state_diff < 1e-6

    # Edge artifacts must still be small in absolute terms (bounded latency
    # ramp-in/out, not a runaway or a sign flip).
    assert np.max(np.abs(y - x)) < 0.02
    assert stats["ceiling_met"] is True
