"""Behavioural tests for the feed-forward (multiband) compressor."""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import dynamics

SR = 48000


def _db(x: float) -> float:
    return 20.0 * np.log10(max(abs(x), 1e-12))


def test_quiet_signal_passes_through_unchanged():
    """Below threshold - knee/2: output must equal input within 0.01 dB."""
    n = SR // 2
    quiet_level = 10 ** (-40.0 / 20.0)  # -40 dBFS, well under -18-3=-21 dB
    x = np.full((n, 2), quiet_level)

    params = dynamics.BandParams()  # defaults: threshold=-18, ratio=2, knee=6
    y, stats = dynamics.compress(x, SR, params)

    assert stats["max_gain_reduction_db"] == 0.0
    assert stats["avg_gain_reduction_db"] == 0.0
    ratio_db = _db(np.max(np.abs(y - x))) - _db(quiet_level)
    # y should equal x exactly (gain==1 throughout); allow tiny float slack.
    assert np.max(np.abs(y - x)) < 1e-9, f"quiet signal changed by up to {ratio_db} dB-ish delta"


def test_gain_reduction_tracks_a_20db_step_with_measurable_attack_and_release():
    """A signal with a 20 dB step must show compression that ramps in/out
    with attack/release time constants measurable from the gain envelope.
    """
    seg = SR // 2  # 0.5 s per segment
    quiet_level = 10 ** (-40.0 / 20.0)  # -40 dBFS -> below knee, no reduction
    loud_level = 10 ** (-6.0 / 20.0)  # -6 dBFS -> well above knee, full ratio applies

    x = np.concatenate(
        [
            np.full((seg, 1), quiet_level),
            np.full((seg, 1), loud_level),
            np.full((seg, 1), quiet_level),
        ]
    )
    x = np.concatenate([x, x], axis=1)  # stereo

    attack_ms, release_ms = 20.0, 150.0
    params = dynamics.BandParams(
        threshold_db=-20.0, ratio=4.0, attack_ms=attack_ms, release_ms=release_ms, knee_db=6.0, makeup_db=0.0
    )
    _y, stats = dynamics.compress(x, SR, params, detector="peak", stereo_link=True)
    env_db = stats["gain_envelope_db"]

    # Expected steady-state reduction (well outside the knee):
    # y_db = T + (x_db - T)/ratio = -20 + (-6 - -20)/4 = -16.5 dBFS
    # gain_reduction = y_db - x_db = -16.5 - (-6) = -10.5 dB
    expected_steady_state_db = -10.5
    steady_state_measured = float(np.mean(env_db[seg + seg // 2 : 2 * seg]))
    print(
        f"\n[dynamics] steady-state gain reduction: measured={steady_state_measured:.3f} dB expected={expected_steady_state_db} dB"
    )
    assert abs(steady_state_measured - expected_steady_state_db) < 0.5

    # Attack: time to travel 63.2% of the way from 0 -> steady state, after the step at `seg`.
    target = steady_state_measured
    threshold_value = target * (1 - np.exp(-1.0))  # 63.2% of the way to `target` (target is negative)
    post_step = env_db[seg:]
    crossing = np.argmax(post_step <= threshold_value)  # first index where reduction has caught up
    measured_attack_ms = crossing / SR * 1000.0
    print(f"[dynamics] attack: measured~={measured_attack_ms:.2f} ms, configured={attack_ms} ms")
    assert 0.3 * attack_ms < measured_attack_ms < 3.0 * attack_ms

    # Release: time to recover 63.2% of the way back from steady-state to 0, after the step down at `2*seg`.
    post_release = env_db[2 * seg :]
    release_threshold_value = target * np.exp(-1.0)  # 63.2% of the way back to 0
    crossing_release = np.argmax(post_release >= release_threshold_value)
    measured_release_ms = crossing_release / SR * 1000.0
    print(f"[dynamics] release: measured~={measured_release_ms:.2f} ms, configured={release_ms} ms")
    assert 0.3 * release_ms < measured_release_ms < 3.0 * release_ms


def test_multiband_only_reduces_the_band_that_is_actually_loud():
    """3 crossovers / 4 bands: a tone loud in one band's range must only
    trigger meaningful gain reduction in that band.
    """
    seconds = 1.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    # 4000 Hz sits comfortably inside band index 2: (2000 Hz, 8000 Hz).
    tone_level_dbfs = -6.0
    tone = (10 ** (tone_level_dbfs / 20.0)) * np.sin(2 * np.pi * 4000.0 * t)
    x = np.stack([tone, tone], axis=1)

    crossovers = [200.0, 2000.0, 8000.0]
    bands = [dynamics.BandParams() for _ in range(4)]  # defaults: threshold=-18, ratio=2

    _y, stats = dynamics.multiband_compress(x, SR, crossovers, bands)

    reductions = [b["max_gain_reduction_db"] for b in stats["bands"]]
    print(f"\n[dynamics] per-band max gain reduction (dB): {reductions}")

    assert reductions[2] < -1.0, f"band 2 (2000-8000 Hz) should show real compression, got {reductions[2]} dB"
    for i in (0, 1, 3):
        assert reductions[i] > -3.0, f"band {i} should barely be touched by a 4kHz tone, got {reductions[i]} dB"


def test_multiband_compress_rejects_mismatched_band_and_crossover_counts():
    x = np.zeros((SR, 2))
    crossovers = [200.0, 2000.0]
    bands = [dynamics.BandParams()]  # needs 3, only gave 1
    with pytest.raises(ValueError, match=r"2|1"):
        dynamics.multiband_compress(x, SR, crossovers, bands)


# --- Bounded per-band arithmetic: makes host-to-host DSP drift visible -----
#
# test_multiband_only_reduces_the_band_that_is_actually_loud (above) only
# checks "some meaningful reduction happened in the loud band, and the quiet
# bands were barely touched" -- a bound wide enough that a materially
# different crossover or smoothing implementation could still pass. This
# test pins the *measured* per-band reduction to a tight, stated tolerance
# around a value calibrated by running this exact deterministic signal
# through this exact reference implementation (see agent report): a fast
# (4 kHz) carrier inside a slow (attack=10ms/release=120ms) envelope
# follower does not settle to the naive instantaneous-peak formula
# (threshold -18 + (level -6 - -18)/ratio 2 - level -6 == -6.0 dB) -- the
# one-pole smoothing only partially tracks a carrier many times faster than
# its own time constants, and that gap is itself part of what a drifting
# implementation could change without anything here noticing.

_MB_TONE_LEVEL_DBFS = -6.0
_MB_TONE_FREQ_HZ = 4000.0  # sits inside band 2: (2000 Hz, 8000 Hz)
_MB_CROSSOVERS_HZ = [200.0, 2000.0, 8000.0]

# Measured on this host, bit-exact across 3 runs (see agent report) -- this
# is the reference implementation's actual settled value, not the naive
# instantaneous-peak formula's -6.0 dB.
_MB_EXPECTED_MAX_REDUCTION_DB = -4.941244279194672
_MB_EXPECTED_AVG_REDUCTION_DB = -4.846086639403051
_MB_REDUCTION_TOLERANCE_DB = 0.3
# Wide enough to tolerate libm/BLAS-level float differences across hosts,
# tight enough that a real behavioural change (a different smoothing
# constant, a crossover slope change) trips it -- the same true-peak defect
# this test guards against moved the limiter's reading by ~1 dB host to host.
_MB_QUIET_BAND_TOLERANCE_DB = 0.1
# Bands 0/1/3 measure exactly 0.0 dB reduction here -- a 4 kHz tone is >20 dB
# down (LR4, 24 dB/octave) in each of them, safely below the knee -- so any
# nonzero reading is crossover leakage, not float noise.


def test_multiband_gain_reduction_matches_a_measured_tolerance_per_band():
    """Pin the loud band's reduction to a tight, stated tolerance (not just
    "less than -1 dB") and the quiet bands to near-zero, so a change in
    crossover or smoothing behaviour that shifts the numbers is caught --
    which `test_multiband_only_reduces_the_band_that_is_actually_loud`'s
    wide bound cannot do.
    """
    seconds = 1.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    tone = (10 ** (_MB_TONE_LEVEL_DBFS / 20.0)) * np.sin(2 * np.pi * _MB_TONE_FREQ_HZ * t)
    x = np.stack([tone, tone], axis=1)

    bands = [dynamics.BandParams() for _ in range(4)]  # defaults: threshold=-18, ratio=2, knee=6
    _y, stats = dynamics.multiband_compress(x, SR, _MB_CROSSOVERS_HZ, bands)

    max_reductions = [b["max_gain_reduction_db"] for b in stats["bands"]]
    avg_reductions = [b["avg_gain_reduction_db"] for b in stats["bands"]]
    print(f"\n[dynamics] per-band max reduction (dB): {max_reductions}")
    print(f"[dynamics] per-band avg reduction (dB): {avg_reductions}")

    assert abs(max_reductions[2] - _MB_EXPECTED_MAX_REDUCTION_DB) < _MB_REDUCTION_TOLERANCE_DB, (
        f"band 2 max reduction {max_reductions[2]:.4f} dB drifted from the measured baseline "
        f"{_MB_EXPECTED_MAX_REDUCTION_DB:.4f} dB by more than {_MB_REDUCTION_TOLERANCE_DB} dB"
    )
    assert abs(avg_reductions[2] - _MB_EXPECTED_AVG_REDUCTION_DB) < _MB_REDUCTION_TOLERANCE_DB, (
        f"band 2 avg reduction {avg_reductions[2]:.4f} dB drifted from the measured baseline "
        f"{_MB_EXPECTED_AVG_REDUCTION_DB:.4f} dB by more than {_MB_REDUCTION_TOLERANCE_DB} dB"
    )
    for i in (0, 1, 3):
        assert abs(max_reductions[i]) < _MB_QUIET_BAND_TOLERANCE_DB, (
            f"band {i} should show no threshold crossing at all for a 4 kHz tone, got {max_reductions[i]} dB"
        )
