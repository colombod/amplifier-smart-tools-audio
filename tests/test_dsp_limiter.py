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


# --- Regression guard: an assertion that cannot fail is not evidence -------
#
# Every test above proves the limiter behaves correctly *when it engages*.
# None of them proves it engages at all: `ceiling_met` is true whenever
# output_true_peak_dbtp happens to land under the ceiling, which is equally
# true of a signal that never came near the ceiling in the first place.
# Two renders of the identical commit -- one on a dev box, one in a clean
# container -- produced true peaks 0.97 dB apart while both reported
# ceiling_met: true, because the limiter never engaged on either host
# (max_gain_reduction_db was 0.0 both times): the upstream multiband
# compressor/saturator, not the limiter, decided the output level, and
# *that* differs host to host. `ceiling_ok`/`lufs_ok`-style booleans
# elsewhere in this suite share this shape (see agent report); this test
# forces engagement and pins the arithmetic so a chain whose limiter goes
# quiet again cannot pass silently.

_FORCED_CEILING_DBTP = -1.0
_FORCED_HEADROOM_OVER_CEILING_DB = 6.0  # deliberately "several dB" over the ceiling
_FORCED_FADE_MS = 50.0
# Fades the plateau's start/end so resample_poly's FIR does not ring on an
# abrupt digital step. Measured (see agent report): an un-faded, hard-edged
# plateau reads input_true_peak_dbtp ~1 dB hotter than its true amplitude and
# leaves ceiling_met False, purely from FIR boundary (Gibs-like) ringing at
# the array edges -- an artifact of testing a plateau signal, not of the
# limiter under test.
_GAIN_REDUCTION_TOLERANCE_DB = 0.05
# Measured drift from the exact -6.0 dB arithmetic on this host: 0.0055 dB
# (oversampling-filter ripple), bit-exact across 3 runs. Matches the
# module's own declared numerical tolerance for ceiling_met (see the
# ceiling_met comment in limiter.brickwall above).
_CEILING_TOLERANCE_DB = 0.05  # same tolerance limiter.py itself uses for ceiling_met


def _forced_plateau_signal(sr: int = SR, seconds: float = 1.0) -> tuple[np.ndarray, float]:
    """A constant-level plateau, `_FORCED_HEADROOM_OVER_CEILING_DB` dB over
    the ceiling, faded in/out so the limiter's own oversampling doesn't ring
    on the buffer's start/end discontinuity.

    Returns (signal, input_peak_dbtp) -- the latter is the *intended* peak
    level in dBTP, used to compute the exact expected gain reduction.
    """
    input_peak_dbtp = _FORCED_CEILING_DBTP + _FORCED_HEADROOM_OVER_CEILING_DB
    amplitude = 10.0 ** (input_peak_dbtp / 20.0)
    n = int(seconds * sr)
    fade_n = int(_FORCED_FADE_MS / 1000.0 * sr)
    x = np.full(n, amplitude)
    ramp = np.linspace(0.0, 1.0, fade_n)
    x[:fade_n] *= ramp
    x[-fade_n:] *= ramp[::-1]
    return np.stack([x, x], axis=1), input_peak_dbtp


def test_brickwall_actually_engages_and_matches_the_expected_gain_reduction():
    """Force the limiter to engage several dB over the ceiling and check the
    arithmetic, not just the boolean. A chain whose limiter never engages
    must never again be used as evidence that limiting works.
    """
    x, input_peak_dbtp = _forced_plateau_signal()
    y, stats = limiter.brickwall(
        x, SR, ceiling_dbtp=_FORCED_CEILING_DBTP, lookahead_ms=5.0, release_ms=100.0, oversample=4
    )

    print(f"\n[limiter] forced input_true_peak={stats['input_true_peak_dbtp']:.4f} dBTP")
    print(f"[limiter] forced output_true_peak={stats['output_true_peak_dbtp']:.4f} dBTP")
    print(f"[limiter] forced max_gain_reduction={stats['max_gain_reduction_db']:.4f} dB")

    # The canary: ceiling_met == True proves nothing if the limiter never
    # touched the signal. Fail loudly, with a diagnosable message, rather
    # than silently accepting a no-op chain as evidence of correctness.
    assert stats["max_gain_reduction_db"] != 0.0, (
        "the forced test signal did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_met alone is not evidence the limiter works; fix the fixture, not the assertion"
    )

    expected_reduction_db = _FORCED_CEILING_DBTP - input_peak_dbtp  # exact arithmetic: -6.0 dB
    assert abs(stats["max_gain_reduction_db"] - expected_reduction_db) < _GAIN_REDUCTION_TOLERANCE_DB, (
        f"measured gain reduction {stats['max_gain_reduction_db']:.4f} dB strayed from the expected "
        f"{expected_reduction_db:.4f} dB by more than {_GAIN_REDUCTION_TOLERANCE_DB} dB"
    )

    assert stats["output_true_peak_dbtp"] <= _FORCED_CEILING_DBTP + _CEILING_TOLERANCE_DB, (
        f"output true peak {stats['output_true_peak_dbtp']:.4f} dBTP is not meaningfully below "
        f"the ceiling {_FORCED_CEILING_DBTP} dBTP"
    )
    assert stats["ceiling_met"] is True
    assert y.shape == x.shape
