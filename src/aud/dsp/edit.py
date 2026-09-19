"""Cutting and gluing -- turns resolved edit points into a shorter signal.

`aud.dsp.resolve` decides *where* the blade falls; this module does the
actual removal and rejoining: fades at boundaries with no crossfade
partner, a crossfade at every join a removal creates. See
contracts/plan.v1.md#edit-point-resolution-shared-by-cut-and-strip_silence
and #fades-and-crossfades.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from aud.dsp.resolve import EditPoint

__all__ = ["CrossfadeExceedsGapError", "crossfade", "cut_regions", "fade_in", "fade_out"]


class CrossfadeExceedsGapError(ValueError):
    """`crossfade_ms` cannot fit at one join.

    Two distinct reasons, both named in contracts/plan.v1.md#fades-and-crossfades:
    "removal" -- the crossfade is longer than the removed span it spans, which
    would eat into kept audio on the far side; "kept_slice" -- the crossfade
    is longer than the kept material between two removals, which would make
    two crossfades overlap.

    Not raised as an `AudError` directly: `dsp/` modules take arrays and
    parameters and return arrays, they do not raise user-facing errors (see
    AGENTS.md #8). `aud.lib.render` catches this and maps it to
    `AudError(code="crossfade_exceeds_gap")`, the same boundary pattern
    `aud.schemas.NotImplementedStageError` uses for `not_implemented`.
    """

    def __init__(self, *, region_index: int, requested_ms: float, available_ms: float, reason: str) -> None:
        self.region_index = region_index
        self.requested_ms = requested_ms
        self.available_ms = available_ms
        self.reason = reason  # "removal" | "kept_slice"
        super().__init__(
            f"crossfade_ms={requested_ms} exceeds the {reason} available at region {region_index} "
            f"({available_ms:.3f} ms)."
        )


def fade_in(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Linear fade-in over the first `ms` milliseconds (clamped to the array length)."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = min(x.shape[0], round(ms / 1000.0 * sr))
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, endpoint=True)
    if x.ndim == 2:
        ramp = ramp[:, None]
    x[:n] = x[:n] * ramp
    return x


def fade_out(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Linear fade-out over the last `ms` milliseconds (clamped to the array length)."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = min(x.shape[0], round(ms / 1000.0 * sr))
    if n <= 0:
        return x
    ramp = np.linspace(1.0, 0.0, n, endpoint=True)
    if x.ndim == 2:
        ramp = ramp[:, None]
    x[-n:] = x[-n:] * ramp
    return x


def crossfade(a: np.ndarray, b: np.ndarray, sr: int, ms: float, shape: str = "equal_power") -> np.ndarray:
    """Join `a` (outgoing slice) into `b` (incoming slice) over `ms` milliseconds.

    `equal_power` is the default because two uncorrelated signals sum in
    POWER, not amplitude: a linear (equal-gain) crossfade puts both sides at
    0.5 in the middle, so the summed power is half of either side's alone --
    an audible ~3 dB dip through the join. `sin`/`cos` gains keep the sum of
    the *squared* gains constant instead, so perceived loudness stays flat
    across the seam. `linear` exists for the opposite case: two *correlated*
    signals (e.g. a splice in a continuous tone) sum in amplitude, and there
    equal-power would bump by ~3 dB instead of staying flat.

    Args:
        a: Outgoing slice, shape (n_samples, n_channels) or (n_samples,).
        b: Incoming slice, same shape convention.
        sr: Sample rate in Hz.
        ms: Crossfade length in milliseconds. `<= 0` returns `a` and `b`
            simply concatenated (no crossfade).
        shape: "equal_power" (default) or "linear".

    Returns:
        The joined array: `a` up to the overlap, the crossfaded overlap,
        then `b` after the overlap.

    Raises:
        ValueError: `shape` is not one of the two documented values, or `a`
            or `b` is shorter than the crossfade needs. Callers that have
            already validated gap sizes (see `cut_regions`) should never
            hit the length case; it is a defensive floor, not the reported
            failure -- `cut_regions` raises `CrossfadeExceedsGapError` with
            the join's region context before ever calling this.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if shape not in ("equal_power", "linear"):
        raise ValueError(f"shape must be 'equal_power' or 'linear', got {shape!r}")

    n = round(ms / 1000.0 * sr)
    if n <= 0:
        return np.concatenate([a, b], axis=0)
    if a.shape[0] < n or b.shape[0] < n:
        raise ValueError(
            f"crossfade of {ms} ms needs {n} samples on each side; got {a.shape[0]} (outgoing) "
            f"and {b.shape[0]} (incoming)"
        )

    t = np.linspace(0.0, 1.0, n, endpoint=False)
    if shape == "linear":
        gain_out = 1.0 - t
        gain_in = t
    else:
        gain_out = np.cos(t * (np.pi / 2.0))
        gain_in = np.sin(t * (np.pi / 2.0))
    if a.ndim == 2:
        gain_out = gain_out[:, None]
        gain_in = gain_in[:, None]

    mixed = a[-n:] * gain_out + b[:n] * gain_in
    return np.concatenate([a[:-n], mixed, b[n:]], axis=0)


def _seconds_to_sample(seconds: float, sr: int, n: int) -> int:
    return min(max(round(seconds * sr), 0), n)


def cut_regions(
    x: np.ndarray,
    sr: int,
    points: list[EditPoint],
    *,
    fade_in_ms: float,
    fade_out_ms: float,
    crossfade_ms: float,
    crossfade_shape: str,
    regions_dropped_by_padding: int = 0,
) -> tuple[np.ndarray, dict]:
    """Remove the spans `points` describes and glue the kept slices together.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        points: Resolved edit points from `aud.dsp.resolve.resolve_points`
            -- exactly one `start` and one `end` per surviving region.
        fade_in_ms: Fade-in at a kept boundary with no crossfade partner
            (the head of the programme, or a join with `crossfade_ms == 0`).
        fade_out_ms: Symmetric to `fade_in_ms`, at the tail / outgoing side.
        crossfade_ms: Length of the crossfade at each join a removal creates.
        crossfade_shape: "equal_power" or "linear" -- see `crossfade`.
        regions_dropped_by_padding: Count of regions padding consumed
            entirely (from `resolve_points`'s second return value), passed
            straight through into the report so a caller sees the whole
            picture in one place.

    Returns:
        (y, report). `report` matches contracts/plan.v1.md's per-stage
        render report shape: `regions_removed`, `regions_dropped_by_padding`,
        `snap_failures`, `seconds_removed`, `joins_crossfaded`, and
        `edit_points` (one dict per point, via `dataclasses.asdict` --
        field names already match the contract).

    Raises:
        CrossfadeExceedsGapError: `crossfade_ms` does not fit at some join --
            either it exceeds the removal it spans, or the kept slice
            between two removals is too short to hold crossfades on both
            of its sides.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    by_region: dict[int, dict[str, EditPoint]] = {}
    for point in points:
        by_region.setdefault(point.region_index, {})[point.boundary] = point

    spans: list[tuple[float, float, int]] = []
    for region_index, pair in sorted(by_region.items()):
        start_point = pair.get("start")
        end_point = pair.get("end")
        if start_point is None or end_point is None:
            raise ValueError(f"region {region_index} is missing a start or end edit point")
        spans.append((start_point.resolved_s, end_point.resolved_s, region_index))
    spans.sort(key=lambda span: span[0])

    n_removals = len(spans)

    # Validate every join before touching any samples: a crossfade that does
    # not fit is an error, never a silent clamp (contracts/plan.v1.md).
    for start_s, end_s, region_index in spans:
        removal_ms = (end_s - start_s) * 1000.0
        if crossfade_ms > removal_ms + 1e-9:
            raise CrossfadeExceedsGapError(
                region_index=region_index, requested_ms=crossfade_ms, available_ms=removal_ms, reason="removal"
            )

    kept_bounds: list[tuple[int, int]] = []
    cursor = 0
    for start_s, end_s, _ in spans:
        start_n = max(cursor, _seconds_to_sample(start_s, sr, n))
        end_n = max(start_n, _seconds_to_sample(end_s, sr, n))
        kept_bounds.append((cursor, start_n))
        cursor = end_n
    kept_bounds.append((cursor, n))

    if crossfade_ms > 0:
        crossfade_n = round(crossfade_ms / 1000.0 * sr)
        for seg_index, (seg_start, seg_end) in enumerate(kept_bounds):
            needed = 0
            if seg_index > 0:  # incoming side of the previous join
                needed += crossfade_n
            if seg_index < n_removals:  # outgoing side of the next join
                needed += crossfade_n
            seg_len = seg_end - seg_start
            if needed > seg_len:
                nearby_region = spans[seg_index][2] if seg_index < n_removals else spans[seg_index - 1][2]
                raise CrossfadeExceedsGapError(
                    region_index=nearby_region,
                    requested_ms=crossfade_ms,
                    available_ms=(seg_len / sr) * 1000.0,
                    reason="kept_slice",
                )

    pieces: list[np.ndarray] = []
    joins_crossfaded = 0
    last_index = len(kept_bounds) - 1
    for i, (seg_start, seg_end) in enumerate(kept_bounds):
        seg = x[seg_start:seg_end].copy()
        has_join_before = i > 0
        has_join_after = i < n_removals
        is_head = i == 0
        is_tail = i == last_index

        # Fades and crossfades never both apply to the same seam: at an
        # interior join, a fade only runs when crossfade_ms is 0 there.
        if fade_in_ms > 0 and (is_head or (has_join_before and crossfade_ms <= 0)):
            seg = fade_in(seg, sr, fade_in_ms)
        if fade_out_ms > 0 and (is_tail or (has_join_after and crossfade_ms <= 0)):
            seg = fade_out(seg, sr, fade_out_ms)

        if has_join_before and crossfade_ms > 0:
            previous = pieces.pop()
            pieces.append(crossfade(previous, seg, sr, crossfade_ms, crossfade_shape))
            joins_crossfaded += 1
        else:
            pieces.append(seg)

    y = np.concatenate(pieces, axis=0) if pieces else x[:0]

    seconds_removed = sum(end_s - start_s for start_s, end_s, _ in spans)
    report = {
        "regions_removed": n_removals,
        "regions_dropped_by_padding": regions_dropped_by_padding,
        "snap_failures": sum(1 for point in points if point.snap_failed),
        "seconds_removed": seconds_removed,
        "joins_crossfaded": joins_crossfaded,
        "edit_points": [dataclasses.asdict(point) for point in points],
    }
    return y, report
