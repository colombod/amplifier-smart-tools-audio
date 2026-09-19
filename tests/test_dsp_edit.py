"""Behavioural tests for aud.dsp.edit: cutting, fades, and crossfades.

The click test and the equal-power-vs-linear dip test measure real numbers
against thresholds stated in the test itself, per
contracts/plan.v1.md#fades-and-crossfades -- the whole argument for
equal-power being the default is a measurable property, so it gets
measured here, not asserted by authority.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp.edit import CrossfadeExceedsGapError, crossfade, cut_regions, fade_in, fade_out
from aud.dsp.resolve import EditPoint, resolve_points

SR = 44100


def _tone(seconds: float, freq: float = 220.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def _max_discontinuity(x: np.ndarray) -> float:
    """Largest sample-to-sample jump in a mono signal -- the audible "click" measure."""
    return float(np.max(np.abs(np.diff(x))))


def _point(region_index: int, boundary: str, resolved_s: float, sr: int = SR) -> EditPoint:
    return EditPoint(
        region_index=region_index,
        boundary=boundary,
        nominal_s=resolved_s,
        padded_s=resolved_s,
        resolved_s=resolved_s,
        moved_ms=0.0,
        snap_requested="none",
        rule_applied="none",
        snap_failed=False,
        reason=None,
    )


# ---------------------------------------------------------------------------
# fade_in / fade_out
# ---------------------------------------------------------------------------


def test_fade_in_ramps_from_zero_to_original_amplitude():
    x = np.full(4410, 0.5)  # 100 ms of constant amplitude at 44.1kHz
    y = fade_in(x, SR, ms=10.0)
    assert y[0] == pytest.approx(0.0, abs=1e-9)
    assert y[-1] == pytest.approx(0.5, abs=1e-6)  # unaffected tail
    assert y[220] > y[0]  # monotonically rising through the ramp


def test_fade_out_ramps_to_zero_at_the_very_end():
    x = np.full(4410, 0.5)
    y = fade_out(x, SR, ms=10.0)
    assert y[-1] == pytest.approx(0.0, abs=1e-9)
    assert y[0] == pytest.approx(0.5, abs=1e-6)  # unaffected head


# ---------------------------------------------------------------------------
# crossfade
# ---------------------------------------------------------------------------


def test_crossfade_shorter_than_available_material_raises_valueerror():
    a = np.zeros(10)
    b = np.zeros(10)
    with pytest.raises(ValueError, match="crossfade"):
        crossfade(a, b, SR, ms=1.0)  # 1ms @ 44100 needs 44 samples; only 10 available


def _center_window(joined: np.ndarray, join_index: int, fade_len: int, half_width: int) -> np.ndarray:
    """A small band centered on the crossfade's midpoint (t=0.5, gain=gain=0.707 for equal_power).

    Averaging over the *whole* fade region dilutes the dip/bump with the
    ramp's edges (near t=0 and t=1, where one side dominates and there is
    almost no dip/bump at all); the theoretical ~3 dB difference between
    equal-power and linear is a property of the crossfade's midpoint.
    """
    center = join_index - fade_len // 2
    return joined[center - half_width : center + half_width]


def test_equal_power_crossfade_holds_rms_through_the_join_uncorrelated_signals():
    rng = np.random.default_rng(0)
    n = int(0.05 * SR)  # 50 ms of independent white noise on each side -- uncorrelated
    a = rng.normal(0.0, 0.3, size=n)
    b = rng.normal(0.0, 0.3, size=n)

    joined = crossfade(a, b, SR, ms=20.0, shape="equal_power")
    fade_len = round(20.0 / 1000.0 * SR)
    mid = _center_window(joined, len(a), fade_len, half_width=max(1, fade_len // 10))
    outside_rms = float(np.sqrt(np.mean(a[: n - fade_len] ** 2)))
    mid_rms = float(np.sqrt(np.mean(mid**2)))
    print(f"\n[edit] equal_power: outside_rms={outside_rms:.4f}, mid_rms={mid_rms:.4f}")
    # Equal power keeps perceived loudness flat: within ~1.5 dB of the
    # surrounding material, not the ~3 dB dip a linear crossfade produces.
    ratio_db = 20.0 * np.log10(mid_rms / outside_rms)
    assert abs(ratio_db) < 1.5


def test_linear_crossfade_of_uncorrelated_noise_measurably_dips():
    rng = np.random.default_rng(1)
    n = int(0.05 * SR)
    a = rng.normal(0.0, 0.3, size=n)
    b = rng.normal(0.0, 0.3, size=n)

    joined_linear = crossfade(a, b, SR, ms=20.0, shape="linear")
    joined_equal_power = crossfade(a, b, SR, ms=20.0, shape="equal_power")
    fade_len = round(20.0 / 1000.0 * SR)
    half_width = max(1, fade_len // 10)

    mid_linear = _center_window(joined_linear, len(a), fade_len, half_width)
    mid_equal_power = _center_window(joined_equal_power, len(a), fade_len, half_width)
    outside_rms = float(np.sqrt(np.mean(a[: n - fade_len] ** 2)))
    linear_rms = float(np.sqrt(np.mean(mid_linear**2)))
    equal_power_rms = float(np.sqrt(np.mean(mid_equal_power**2)))

    linear_dip_db = 20.0 * np.log10(linear_rms / outside_rms)
    equal_power_dip_db = 20.0 * np.log10(equal_power_rms / outside_rms)
    print(f"\n[edit] dip comparison: linear={linear_dip_db:.2f} dB, equal_power={equal_power_dip_db:.2f} dB")

    # This is the numeric evidence for the contract's documented default:
    # linear dips noticeably more than equal_power for uncorrelated material
    # (theory: linear ~ -3 dB at the midpoint, equal_power ~ 0 dB).
    assert linear_dip_db < equal_power_dip_db - 1.0
    assert linear_dip_db < -2.0  # a real, audible dip -- not a rounding artifact


# ---------------------------------------------------------------------------
# cut_regions -- the click test
# ---------------------------------------------------------------------------


def test_cut_with_zero_crossing_and_crossfade_avoids_a_click():
    freq = 220.0
    amplitude = 0.6
    x = _tone(2.0, freq=freq, sr=SR, amplitude=amplitude)
    period = 1.0 / freq

    # 0.5s and 0.8s are both exact multiples of this tone's period. Placing
    # the two nominal points at DIFFERENT fractions of a period (one at the
    # peak, one at the trough) guarantees the raw splice actually clicks --
    # placing both at the same phase (e.g. both peaks) would make the raw
    # splice a near-continuous, unrepresentative no-op.
    start_s = 0.5 + period / 4.0  # peak: sin(pi/2) == 1
    end_s = 0.8 + (3.0 * period) / 4.0  # trough: sin(3*pi/2) == -1

    def _nearest_zero_crossing(target_s: float) -> float:
        target_i = round(target_s * SR)
        window = round(period * SR)
        segment = x[max(0, target_i - window) : target_i + window]
        signs = np.sign(segment)
        signs[signs == 0] = 1.0
        changes = np.where(np.diff(signs) != 0)[0]
        base = max(0, target_i - window)
        return float((base + changes[np.argmin(np.abs(changes - window))]) / SR)

    resolved_start = _nearest_zero_crossing(start_s)
    resolved_end = _nearest_zero_crossing(end_s)
    points = [_point(0, "start", resolved_start), _point(0, "end", resolved_end)]

    crossfade_n = round(10.0 / 1000.0 * SR)
    y, report = cut_regions(
        x, SR, points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=10.0, crossfade_shape="equal_power"
    )
    # Measure the discontinuity IN THE JOIN AREA ONLY: a plain oscillating
    # tone already has a natural per-sample delta of amplitude*2*pi*freq/sr
    # (~0.0188 here) at every zero crossing throughout the whole file, which
    # would swamp/hide a join-specific click if measured globally.
    join_n = round(resolved_start * SR)
    click = _max_discontinuity(y[max(0, join_n - crossfade_n - 5) : join_n + crossfade_n + 5])

    # Same cut, but with the RAW nominal positions (no zero-crossing
    # alignment) and no crossfade at all -- the worst case this feature
    # exists to prevent.
    raw_points = [_point(0, "start", start_s), _point(0, "end", end_s)]
    y_raw, _ = cut_regions(
        x, SR, raw_points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=0.0, crossfade_shape="equal_power"
    )
    raw_join_n = round(start_s * SR)
    click_raw = _max_discontinuity(y_raw[max(0, raw_join_n - 5) : raw_join_n + 5])

    print(f"\n[edit] click test: resolved+crossfade={click:.6f}, raw hard splice at peak/trough={click_raw:.6f}")
    assert click < 0.05  # smoothed by zero-crossing alignment + crossfade
    assert click_raw > amplitude  # peak-to-trough hard splice: a real, dramatic click (~2x amplitude)
    assert click_raw > click * 5  # dramatically worse without resolution+crossfade
    assert report["joins_crossfaded"] == 1
    assert report["regions_removed"] == 1


def test_transient_snap_with_no_onset_still_avoids_a_click_at_the_join():
    """Regression test for the shipped defect: snap="transient" with no
    onset in the window used to report rule_applied="none" and take the
    raw (nominal/padded) position literally -- silently dropping the
    zero-crossing alignment every other rule gets for free. Zero crossing
    is the floor under every rule, not a peer of them
    (contracts/plan.v1.md#snap): a failed coarse search must still land on
    a zero crossing.

    Modelled on test_cut_with_zero_crossing_and_crossfade_avoids_a_click
    above: a peak-to-trough splice with no alignment is a real, measurable
    click; the fix must keep the join close to the signal's own natural
    sample-to-sample step instead.
    """
    freq = 220.0
    amplitude = 0.6
    x = _tone(2.0, freq=freq, sr=SR, amplitude=amplitude)
    period = 1.0 / freq

    # Same peak/trough construction as the click test above -- guarantees a
    # real, measurable click if the raw (unaligned) position is used.
    start_s = 0.5 + period / 4.0  # peak: sin(pi/2) == 1
    end_s = 0.8 + (3.0 * period) / 4.0  # trough: sin(3*pi/2) == -1

    points, dropped = resolve_points(
        x,
        SR,
        [{"start_s": start_s, "end_s": end_s}],
        pad_out_ms=0.0,
        pad_in_ms=0.0,
        snap="transient",
        snap_window_ms=20.0,
        onsets=[],  # no onsets anywhere -- the coarse (transient) search always misses
    )
    assert dropped == []
    start_point, end_point = points[0], points[1]

    # Never the caller-only sentinel, and the floor still ran at both
    # boundaries even though the coarse search missed at both.
    assert start_point.rule_applied == "zero_crossing_fallback"
    assert end_point.rule_applied == "zero_crossing_fallback"
    assert start_point.rule_applied != "none"
    assert end_point.rule_applied != "none"
    # Boundary-role asymmetry: only the END boundary counts as a genuine
    # failure (see test_dsp_resolve.py for the isolated behavioural tests).
    assert start_point.snap_failed is False
    assert end_point.snap_failed is True

    y_fixed, _ = cut_regions(
        x, SR, points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=0.0, crossfade_shape="equal_power"
    )
    join_n = round(start_point.resolved_s * SR)
    click_fixed = _max_discontinuity(y_fixed[max(0, join_n - 5) : join_n + 5])

    # What the pre-fix code produced: rule_applied="none", the raw padded
    # position taken literally, no zero-crossing alignment at all.
    raw_points = [_point(0, "start", start_point.padded_s), _point(0, "end", end_point.padded_s)]
    y_raw, _ = cut_regions(
        x, SR, raw_points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=0.0, crossfade_shape="equal_power"
    )
    raw_join_n = round(start_point.padded_s * SR)
    click_raw = _max_discontinuity(y_raw[max(0, raw_join_n - 5) : raw_join_n + 5])

    smooth_baseline = _max_discontinuity(x[:1000])  # the tone's own ordinary sample-to-sample step

    print(
        f"\n[edit] transient-fallback click test: fixed={click_fixed:.6f}, "
        f"raw(pre-fix)={click_raw:.6f}, smooth_baseline={smooth_baseline:.6f}"
    )
    assert click_fixed < smooth_baseline * 5  # aligned: close to the signal's own natural step
    assert click_raw > amplitude  # unaligned peak/trough splice: a real, dramatic click
    assert click_raw > click_fixed * 5  # dramatically worse without the fix


def test_cut_regions_total_duration_accounts_for_every_removed_second():
    x = _tone(3.0, freq=220.0, sr=SR, amplitude=0.4)
    points = [
        _point(0, "start", 0.5),
        _point(0, "end", 0.8),
        _point(1, "start", 1.5),
        _point(1, "end", 1.9),
    ]
    y, report = cut_regions(
        x, SR, points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=10.0, crossfade_shape="equal_power"
    )

    removed_seconds = (0.8 - 0.5) + (1.9 - 1.5)
    expected_len = x.shape[0] - round(removed_seconds * SR)
    crossfade_samples = round(10.0 / 1000.0 * SR)
    # Each crossfaded join additionally merges `crossfade_samples` of overlap
    # into one (that is the whole point of a crossfade), so the tolerance is
    # "within one crossfade length PER JOIN", not one crossfade length total.
    tolerance = crossfade_samples * report["joins_crossfaded"]
    print(f"\n[edit] duration: input={x.shape[0]}, output={y.shape[0]}, expected~={expected_len}, tol={tolerance}")
    assert abs(y.shape[0] - expected_len) <= tolerance
    assert report["regions_removed"] == 2
    assert report["seconds_removed"] == pytest.approx(removed_seconds, abs=1e-9)


def test_cut_regions_raises_crossfade_exceeds_gap_for_a_short_removal():
    x = _tone(1.0, freq=220.0, sr=SR)
    # A 5ms removal cannot hold a 20ms crossfade.
    points = [_point(0, "start", 0.5), _point(0, "end", 0.505)]
    with pytest.raises(CrossfadeExceedsGapError) as exc_info:
        cut_regions(x, SR, points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=20.0, crossfade_shape="equal_power")
    assert exc_info.value.region_index == 0
    assert exc_info.value.reason == "removal"


def test_cut_regions_raises_crossfade_exceeds_gap_for_a_short_kept_slice_between_removals():
    x = _tone(1.0, freq=220.0, sr=SR)
    # Two removals with only 5ms of kept material between them; a 20ms
    # crossfade on both sides of that sliver cannot fit.
    points = [
        _point(0, "start", 0.5),
        _point(0, "end", 0.6),
        _point(1, "start", 0.605),
        _point(1, "end", 0.7),
    ]
    with pytest.raises(CrossfadeExceedsGapError) as exc_info:
        cut_regions(x, SR, points, fade_in_ms=0.0, fade_out_ms=0.0, crossfade_ms=20.0, crossfade_shape="equal_power")
    assert exc_info.value.reason == "kept_slice"


def test_cut_regions_reports_dropped_count_and_snap_failures_passed_through():
    x = _tone(1.0, freq=220.0, sr=SR)
    failed_point = _point(0, "start", 0.5)
    failed_point.snap_failed = True
    points = [failed_point, _point(0, "end", 0.6)]

    _, report = cut_regions(
        x,
        SR,
        points,
        fade_in_ms=0.0,
        fade_out_ms=0.0,
        crossfade_ms=0.0,
        crossfade_shape="equal_power",
        regions_dropped_by_padding=2,
    )
    assert report["regions_dropped_by_padding"] == 2
    assert report["snap_failures"] == 1
    assert len(report["edit_points"]) == 2
    assert report["edit_points"][0]["region_index"] == 0
