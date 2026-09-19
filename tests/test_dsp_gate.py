"""Behavioural tests for aud.dsp.gate's `gate` and `expand`.

Five properties are asserted directly, one test each (see gate.py's module
docstring for the reasoning behind each):

1. Hold prevents chatter on material hovering at the threshold.
2. Threshold is relative to the measured noise floor, not an absolute level.
3. Lookahead lets a fast onset's attack survive.
4. Multiband: only the quiet band is touched.
5. Sidechain highpass: rumble cannot hold the gate open.

Plus basic validation and expander-curve correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import gate as gate_mod

SR = 48000
_EPS = 1e-12


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x * x)) + _EPS)


# --- Property 1: hold prevents chatter --------------------------------------


def test_hold_prevents_chatter_on_signal_hovering_at_threshold():
    """A carrier whose envelope oscillates across the threshold many times a
    second must open/close far fewer times with a real hold than with none.
    """
    seconds = 2.0
    n = int(seconds * SR)
    t = np.arange(n) / SR

    threshold_db = -20.0
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    # The detector measures RMS of a sine (a factor of sqrt(2), ~3 dB, below
    # its peak), so the envelope is centred on threshold_lin * sqrt(2) --
    # otherwise the RMS-based level would never actually cross threshold_db.
    # +/-50% modulation at 20 Hz crosses the threshold twice per cycle, ~40
    # times/sec, ~80 times over 2 seconds, with a period (50 ms) shorter
    # than the 100 ms hold used below -- so a real hold keeps the gate open
    # essentially continuously, while no hold chatters on every crossing.
    base_lin = threshold_lin * np.sqrt(2.0)
    envelope = base_lin * (1.0 + 0.5 * np.sin(2 * np.pi * 20.0 * t))
    carrier = np.sin(2 * np.pi * 1000.0 * t)
    x = envelope * carrier

    common = {
        "threshold_db": threshold_db,
        "range_db": 12.0,
        "attack_ms": 1.0,
        "release_ms": 20.0,
        "lookahead_ms": 0.0,  # isolate hold's effect from lookahead's own smoothing
        "sidechain_hpf_hz": None,
    }

    _, stats_with_hold = gate_mod.gate(x, SR, hold_ms=100.0, **common)
    _, stats_no_hold = gate_mod.gate(x, SR, hold_ms=0.0, **common)

    print(f"\n[gate] open_count with hold=100ms: {stats_with_hold['open_count']}")
    print(f"[gate] open_count with hold=0ms:   {stats_no_hold['open_count']}")
    assert stats_no_hold["open_count"] > stats_with_hold["open_count"] * 3
    assert stats_with_hold["open_count"] <= 3  # hold outlasts the modulation period: opens once and stays open


# --- Property 2: threshold is relative to the measured noise floor ---------


def test_threshold_above_floor_gives_the_same_result_at_a_different_absolute_level():
    seconds = 2.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = 0.3 * np.sin(2 * np.pi * 300.0 * t)
    quiet = slice(int(0.8 * SR), int(1.4 * SR))
    x[quiet] *= 0.05  # a clearly quieter passage inside an otherwise steady tone

    quieter_x = x * (10.0 ** (-20.0 / 20.0))  # the same file, 20 dB quieter overall

    common = {"threshold_above_floor_db": 10.0, "range_db": 15.0, "attack_ms": 2.0, "hold_ms": 20.0, "release_ms": 80.0}
    _, stats_a = gate_mod.gate(x, SR, **common)
    _, stats_b = gate_mod.gate(quieter_x, SR, **common)

    print(f"\n[gate] floor A={stats_a['noise_floor_dbfs']:.2f} dBFS, threshold={stats_a['threshold_db']:.2f} dBFS")
    print(f"[gate] floor B={stats_b['noise_floor_dbfs']:.2f} dBFS, threshold={stats_b['threshold_db']:.2f} dBFS")
    print(f"[gate] attenuated_pct A={stats_a['attenuated_pct']:.2f}%  B={stats_b['attenuated_pct']:.2f}%")

    assert abs(stats_a["noise_floor_dbfs"] - stats_b["noise_floor_dbfs"] - 20.0) < 1.0
    assert abs(stats_a["threshold_db"] - stats_b["threshold_db"] - 20.0) < 1.0
    # The floor-relative threshold makes the SAME material get gated either way.
    assert abs(stats_a["attenuated_pct"] - stats_b["attenuated_pct"]) < 3.0


def test_threshold_db_absolute_escape_hatch_overrides_the_floor():
    x = 0.3 * np.sin(2 * np.pi * 300.0 * np.arange(SR) / SR)
    _, stats = gate_mod.gate(x, SR, threshold_db=-6.0, threshold_above_floor_db=999.0)
    assert stats["threshold_db"] == -6.0
    assert stats["threshold_above_floor_db"] is None


# --- Property 3: lookahead lets a fast onset survive ------------------------


def test_lookahead_preserves_the_onset_that_no_lookahead_clips():
    silence_s = 0.4
    onset_s = 0.3
    n_silence = int(silence_s * SR)
    n_onset = int(onset_s * SR)
    t_onset = np.arange(n_onset) / SR
    onset = 0.8 * np.sin(2 * np.pi * 1000.0 * t_onset)
    x = np.concatenate([np.zeros(n_silence), onset])

    window = slice(n_silence, n_silence + int(0.005 * SR))  # first 5ms of the onset
    original_peak_db = 20.0 * np.log10(np.max(np.abs(x[window])) + _EPS)

    common = {
        # Close to the onset's own steady-state RMS level (~-4.9 dB for this
        # amplitude) rather than far below it: the detector must climb
        # nearly all the way to steady state before crossing, which is what
        # makes the no-lookahead case take several milliseconds to open.
        "threshold_db": -6.0,
        "range_db": 20.0,
        "attack_ms": 2.0,
        "hold_ms": 10.0,
        "release_ms": 20.0,
        "sidechain_hpf_hz": None,
    }
    y_with_lookahead, _ = gate_mod.gate(x, SR, lookahead_ms=15.0, **common)
    y_no_lookahead, _ = gate_mod.gate(x, SR, lookahead_ms=0.0, **common)

    peak_with_la = 20.0 * np.log10(np.max(np.abs(y_with_lookahead[window])) + _EPS)
    peak_no_la = 20.0 * np.log10(np.max(np.abs(y_no_lookahead[window])) + _EPS)

    print(f"\n[gate] onset peak: original={original_peak_db:.2f} dB")
    print(
        f"[gate] onset peak: with 15ms lookahead={peak_with_la:.2f} dB (loss {original_peak_db - peak_with_la:.2f} dB)"
    )
    print(f"[gate] onset peak: with 0ms lookahead={peak_no_la:.2f} dB (loss {original_peak_db - peak_no_la:.2f} dB)")

    assert (original_peak_db - peak_with_la) < 2.0, "lookahead should preserve the onset within ~2 dB"
    assert (original_peak_db - peak_no_la) > 5.0, "no lookahead should measurably clip the onset's first ms"


# --- Property 4: multiband only attenuates the quiet band ------------------


def test_multiband_gate_only_attenuates_the_band_at_the_noise_floor():
    seconds = 1.5
    n = int(seconds * SR)
    t = np.arange(n) / SR
    loud_low = 0.3 * np.sin(2 * np.pi * 150.0 * t)  # always well above threshold
    quiet_high = 0.001 * np.sin(2 * np.pi * 6000.0 * t)  # always well below threshold
    x = loud_low + quiet_high

    _, stats = gate_mod.gate(
        x,
        SR,
        threshold_db=-30.0,
        range_db=18.0,
        attack_ms=2.0,
        hold_ms=20.0,
        release_ms=50.0,
        lookahead_ms=2.0,
        sidechain_hpf_hz=None,
        crossovers_hz=[1000.0],
    )
    low_band, high_band = stats["bands"]
    print(f"\n[gate] low band (150Hz, loud):  {low_band}")
    print(f"[gate] high band (6kHz, quiet): {high_band}")

    assert low_band["attenuated_pct"] < 5.0
    assert high_band["attenuated_pct"] > 80.0
    assert high_band["max_attenuation_db"] < -15.0
    assert low_band["max_attenuation_db"] > -1.0


# --- Property 5: sidechain highpass keeps rumble from holding the gate open -


def test_sidechain_hpf_prevents_low_frequency_rumble_from_holding_the_gate_open():
    seconds = 2.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rumble = 0.1 * np.sin(2 * np.pi * 40.0 * t)  # constant, always present
    mid = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    burst_mask = np.zeros(n, dtype=bool)
    for start_s, end_s in [(0.0, 0.5), (1.0, 1.5)]:
        burst_mask[int(start_s * SR) : int(end_s * SR)] = True
    x = rumble + mid * burst_mask

    rumble_alone_db = _rms_db(rumble)
    print(f"\n[gate] rumble-alone level: {rumble_alone_db:.2f} dB")

    gap_window = slice(int(1.7 * SR), int(1.95 * SR))  # deep inside the second gap, past release settling
    gap_before_db = _rms_db(x[gap_window])

    common = {
        "threshold_db": -27.0,
        "range_db": 20.0,
        "attack_ms": 2.0,
        "hold_ms": 10.0,
        "release_ms": 30.0,
        "lookahead_ms": 0.0,
    }
    y_no_hpf, _ = gate_mod.gate(x, SR, sidechain_hpf_hz=None, **common)
    y_with_hpf, _ = gate_mod.gate(x, SR, sidechain_hpf_hz=80.0, **common)

    gap_no_hpf_db = _rms_db(y_no_hpf[gap_window])
    gap_with_hpf_db = _rms_db(y_with_hpf[gap_window])

    print(f"[gate] gap level before: {gap_before_db:.2f} dB")
    print(
        f"[gate] gap level, no sidechain HPF:   {gap_no_hpf_db:.2f} dB (delta {gap_before_db - gap_no_hpf_db:.2f} dB)"
    )
    print(
        f"[gate] gap level, with sidechain HPF: {gap_with_hpf_db:.2f} dB (delta {gap_before_db - gap_with_hpf_db:.2f} dB)"
    )

    # Without the sidechain filter, the rumble itself keeps the detector above
    # threshold, so the gap is left essentially untouched.
    assert abs(gap_before_db - gap_no_hpf_db) < 2.0
    # With it, the detector sees only the (silent) mid content in the gap, so
    # the gate closes and the whole band -- rumble included -- is ducked.
    assert (gap_before_db - gap_with_hpf_db) > 12.0


# --- Validation --------------------------------------------------------------


def test_gate_rejects_negative_range_and_negative_time_constants():
    x = np.zeros(SR)
    with pytest.raises(ValueError, match="range_db"):
        gate_mod.gate(x, SR, range_db=-1.0)
    with pytest.raises(ValueError, match="hold_ms"):
        gate_mod.gate(x, SR, hold_ms=-1.0)
    with pytest.raises(ValueError, match="attack_ms"):
        gate_mod.gate(x, SR, attack_ms=-1.0)
    with pytest.raises(ValueError, match="release_ms"):
        gate_mod.gate(x, SR, release_ms=-1.0)
    with pytest.raises(ValueError, match="lookahead_ms"):
        gate_mod.gate(x, SR, lookahead_ms=-1.0)


def test_expand_rejects_ratio_below_one_and_negative_knee():
    x = np.zeros(SR)
    with pytest.raises(ValueError, match="ratio"):
        gate_mod.expand(x, SR, ratio=0.5)
    with pytest.raises(ValueError, match="knee_db"):
        gate_mod.expand(x, SR, knee_db=-1.0)


def test_gate_rejects_non_ascending_crossovers_via_crossover_split():
    x = np.zeros(SR)
    with pytest.raises(ValueError, match="strictly increasing"):
        gate_mod.gate(x, SR, crossovers_hz=[2000.0, 500.0])


# --- Expander curve correctness ----------------------------------------------


def test_expander_attenuates_proportionally_not_as_a_hard_step():
    """Unlike `gate`, `expand` should show a *graded* reduction: material well
    below threshold reduced more than material just below it."""
    seconds = 1.0
    n = int(seconds * SR)
    t = np.arange(n) / SR
    # Two steady tones, one just below threshold, one well below it.
    just_below = 0.1 * np.sin(2 * np.pi * 500.0 * t)  # roughly -20 dBFS
    well_below = 0.01 * np.sin(2 * np.pi * 500.0 * t)  # roughly -40 dBFS

    common = {
        "threshold_db": -15.0,
        "ratio": 3.0,
        "knee_db": 3.0,
        "attack_ms": 5.0,
        "hold_ms": 10.0,
        "release_ms": 50.0,
        "lookahead_ms": 2.0,
    }
    _, stats_just_below = gate_mod.expand(just_below, SR, **common)
    _, stats_well_below = gate_mod.expand(well_below, SR, **common)

    print(f"\n[expand] just-below-threshold max reduction: {stats_just_below['max_attenuation_db']:.2f} dB")
    print(f"[expand] well-below-threshold max reduction:  {stats_well_below['max_attenuation_db']:.2f} dB")
    assert stats_well_below["max_attenuation_db"] < stats_just_below["max_attenuation_db"]


def test_expand_ratio_one_is_a_no_op():
    x = 0.02 * np.sin(2 * np.pi * 500.0 * np.arange(SR) / SR)
    y, stats = gate_mod.expand(x, SR, threshold_db=-6.0, ratio=1.0, knee_db=0.0)
    assert stats["max_attenuation_db"] == 0.0
    assert np.allclose(y, x)
