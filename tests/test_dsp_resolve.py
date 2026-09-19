"""Behavioural tests for aud.dsp.resolve: pad -> snap -> validate joins.

Every test builds a real (synthesised) signal and checks a real numeric
result -- never a mock -- because the whole point of this layer is that a
snap either demonstrably worked or demonstrably didn't (see
contracts/plan.v1.md#the-snap-invariant: "a snap that silently fails is
worse than one that refuses").
"""

from __future__ import annotations

import numpy as np

from aud.dsp.resolve import resolve_points

SR = 44100


def _tone(seconds: float, freq: float = 220.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def _at(x: np.ndarray, sr: int, seconds: float) -> float:
    return float(x[min(round(seconds * sr), x.shape[0] - 1)])


def test_zero_crossing_snap_lands_on_a_zero_crossing_not_the_peak():
    sr = SR
    freq = 220.0
    x = _tone(2.0, freq=freq, sr=sr)
    period = 1.0 / freq

    # Quarter-period offset from t=0 lands exactly on a peak (sin(pi/2) == 1).
    nominal_s = period / 4.0
    region = [{"start_s": nominal_s, "end_s": nominal_s + 5 * period}]

    points, dropped = resolve_points(
        x,
        sr,
        region,
        pad_out_ms=0.0,
        pad_in_ms=0.0,
        snap="zero_crossing",
        snap_window_ms=(period / 2.0) * 1000.0,  # generous: at least one full crossing guaranteed
    )

    assert dropped == []
    start_point = points[0]
    assert start_point.boundary == "start"
    assert start_point.rule_applied == "zero_crossing"
    assert start_point.snap_failed is False

    nominal_amplitude = abs(_at(x, sr, start_point.nominal_s))
    resolved_amplitude = abs(_at(x, sr, start_point.resolved_s))
    print(
        f"\n[resolve] zero_crossing: nominal amplitude={nominal_amplitude:.4f}, resolved amplitude={resolved_amplitude:.6f}"
    )
    assert nominal_amplitude > 0.45  # started at the peak
    assert resolved_amplitude < 0.01  # ended up on (or a sample from) a zero crossing


def test_silence_snap_moves_the_point_into_a_planted_quiet_gap():
    sr = SR
    seconds = 1.2
    x = _tone(seconds, freq=220.0, sr=sr, amplitude=0.6)
    gap_start, gap_end = 0.40, 0.50
    x[int(gap_start * sr) : int(gap_end * sr)] = 1e-5  # a planted, near-digital-silence gap

    nominal_s = 0.38  # just before the gap, so the window has to reach for it
    region = [{"start_s": nominal_s, "end_s": 0.9}]

    points, dropped = resolve_points(
        x,
        sr,
        region,
        pad_out_ms=0.0,
        pad_in_ms=0.0,
        snap="silence",
        snap_window_ms=150.0,
    )

    assert dropped == []
    start_point = points[0]
    print(f"\n[resolve] silence: nominal_s={start_point.nominal_s}, resolved_s={start_point.resolved_s}")
    assert start_point.rule_applied == "silence"
    assert start_point.snap_failed is False
    # Zero-crossing refinement can nudge the candidate a little either side
    # of the frame it was measured on; allow a small margin either way.
    assert gap_start - 0.01 <= start_point.resolved_s <= gap_end + 0.01


def test_transient_snap_lands_before_the_onset_never_after():
    sr = SR
    seconds = 1.2
    x = np.full(int(seconds * sr), 1e-5)
    onset_s = 0.50
    onset_i = int(onset_s * sr)
    tail = _tone(seconds, freq=440.0, sr=sr, amplitude=0.7)[: x.shape[0] - onset_i]
    x[onset_i:] = tail

    nominal_s = 0.52  # deliberately placed AFTER the onset
    region = [{"start_s": nominal_s, "end_s": 0.9}]

    points, dropped = resolve_points(
        x,
        sr,
        region,
        pad_out_ms=0.0,
        pad_in_ms=0.0,
        snap="transient",
        snap_window_ms=100.0,
        onsets=[onset_s],
    )

    assert dropped == []
    start_point = points[0]
    print(f"\n[resolve] transient: onset_s={onset_s}, resolved_s={start_point.resolved_s}")
    assert start_point.rule_applied == "transient"
    assert start_point.snap_failed is False
    assert start_point.resolved_s < onset_s  # "always moves a point earlier, never later"


def test_snap_with_no_candidate_in_window_keeps_nominal_and_reports_failure():
    sr = SR
    # A constant DC offset never changes sign -- zero_crossing has nothing to find.
    x = np.full(int(1.0 * sr), 0.3)
    nominal_s = 0.5
    region = [{"start_s": nominal_s, "end_s": 0.9}]

    points, dropped = resolve_points(
        x,
        sr,
        region,
        pad_out_ms=0.0,
        pad_in_ms=0.0,
        snap="zero_crossing",
        snap_window_ms=20.0,
    )

    assert dropped == []
    start_point = points[0]
    print(f"\n[resolve] no-candidate: {start_point}")
    assert start_point.snap_failed is True
    assert start_point.rule_applied == "none"
    assert start_point.resolved_s == start_point.padded_s == nominal_s
    assert start_point.reason is not None


def test_transient_snap_fails_honestly_when_no_onset_is_in_window():
    sr = SR
    x = _tone(1.0, freq=220.0, sr=sr)
    region = [{"start_s": 0.5, "end_s": 0.9}]

    points, dropped = resolve_points(
        x, sr, region, pad_out_ms=0.0, pad_in_ms=0.0, snap="transient", snap_window_ms=20.0, onsets=[]
    )

    assert dropped == []
    start_point = points[0]
    assert start_point.snap_failed is True
    assert start_point.rule_applied == "none"
    assert start_point.reason is not None
    assert "onset" in start_point.reason


def test_snap_never_crosses_into_a_neighbouring_region():
    sr = SR
    x = _tone(3.0, freq=220.0, sr=sr)
    # Two regions with only a 10ms gap between them; ask for a much wider
    # snap window (50ms) than the gap so the invariant, not the window, is
    # what keeps each point inside its own territory.
    region0_end = 1.05
    region1_start = 1.06
    regions = [
        {"start_s": 1.00, "end_s": region0_end},
        {"start_s": region1_start, "end_s": 2.00},
    ]

    points, dropped = resolve_points(
        x, sr, regions, pad_out_ms=0.0, pad_in_ms=0.0, snap="zero_crossing", snap_window_ms=50.0
    )

    assert dropped == []
    by_region = {(p.region_index, p.boundary): p for p in points}
    region0_end_point = by_region[(0, "end")]
    region1_start_point = by_region[(1, "start")]
    print(
        f"\n[resolve] neighbour clip: region0.end resolved={region0_end_point.resolved_s}, "
        f"region1.start resolved={region1_start_point.resolved_s}"
    )
    # Never crosses INTO the neighbouring region's territory.
    assert region0_end_point.resolved_s <= region1_start
    assert region1_start_point.resolved_s >= region0_end
    # Never leaves the file, and the removal never inverts.
    assert 0.0 <= region0_end_point.resolved_s <= 3.0
    duration = x.shape[0] / sr
    assert 0.0 <= region1_start_point.resolved_s <= duration


def test_padding_that_swallows_a_region_drops_it_and_reports_which():
    sr = SR
    x = _tone(1.0, freq=220.0, sr=sr)
    # A 50 ms region; 30 ms pad_out + 30 ms pad_in == 60 ms, more than the
    # region's own length, so padding consumes the whole thing.
    regions = [{"start_s": 0.5, "end_s": 0.55}]

    points, dropped = resolve_points(
        x, sr, regions, pad_out_ms=30.0, pad_in_ms=30.0, snap="zero_crossing", snap_window_ms=20.0
    )

    assert points == []
    assert dropped == [0]


def test_none_snap_takes_the_padded_position_literally_no_search():
    sr = SR
    x = _tone(1.0, freq=220.0, sr=sr)
    nominal_s = 0.5
    regions = [{"start_s": nominal_s, "end_s": 0.9}]

    points, dropped = resolve_points(x, sr, regions, pad_out_ms=10.0, pad_in_ms=0.0, snap="none", snap_window_ms=20.0)

    assert dropped == []
    start_point = points[0]
    assert start_point.rule_applied == "none"
    assert start_point.snap_failed is False
    assert start_point.resolved_s == start_point.padded_s
    assert abs(start_point.moved_ms - 10.0) < 1e-6  # padding alone moved it, nothing searched further
