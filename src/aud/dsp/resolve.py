"""Edit-point resolution: pad -> snap -> validate joins.

A detector (or a stored `cut` region) reports where a boundary *is*. Where
the blade should *fall* is a separate decision -- see
contracts/plan.v1.md#edit-point-resolution-shared-by-cut-and-strip_silence.
This module implements exactly that resolution, in the order the contract
fixes: pad, then snap (bounded, refusable), then hand back enough detail for
the caller (`aud.dsp.edit.cut_regions`) to validate the joins.

Like every other `dsp/` module, this one takes arrays and parameters and
returns plain data -- no configuration, no filesystem, no user-facing
errors (see AGENTS.md #8). A bad `snap` value raises a plain `ValueError`;
`aud.lib`'s stage builders are the ones that turn a bad parameter into an
`AudError` before it ever reaches here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["EditPoint", "resolve_points"]

_SNAP_MODES = ("zero_crossing", "silence", "transient", "none")

# "transient" moves a point to just BEFORE the nearest onset, never onto it
# or past it -- see contracts/plan.v1.md#snap. This is the lead-in distance;
# it is small relative to any sane snap_window_ms and is not itself a
# contract value (see plan.v1.md "Not promised": DSP internals are free to
# change between releases).
_TRANSIENT_LEAD_S = 0.002

# Short-time energy envelope used by "silence": window + hop for the local
# minimum search. Internal tuning, not a contract value.
_SILENCE_FRAME_MS = 5.0
_SILENCE_HOP_MS = 1.0


@dataclass
class EditPoint:
    """One resolved boundary of one region -- a `start` or an `end`.

    Field names match contracts/plan.v1.md's `edit_points` report entries
    exactly (region_index, boundary, nominal_s, padded_s, resolved_s,
    moved_ms, snap_requested, rule_applied, snap_failed, reason), so a
    caller can serialise one of these with `dataclasses.asdict` and get the
    contract's shape directly.
    """

    region_index: int
    boundary: str  # "start" | "end"
    nominal_s: float
    padded_s: float
    resolved_s: float
    moved_ms: float
    snap_requested: str
    rule_applied: str
    snap_failed: bool
    reason: str | None


def _get(region: Any, key: str) -> float:
    """Read `start_s`/`end_s` from either a mapping or an attribute-bearing object.

    Regions arrive two ways: as plain dicts (a `cut` stage's stored params,
    or `aud.dsp.engine`'s padding-shrunk `strip_silence` spans) and,
    potentially, as objects from `aud.core.regions.Region`. Supporting both
    means this module does not need to know or care which one it was
    handed.
    """
    if isinstance(region, dict):
        return float(region[key])
    return float(getattr(region, key))


def _mono_sum(x: np.ndarray) -> np.ndarray:
    """The mono sum used by "zero_crossing" -- see contracts/plan.v1.md#snap.

    Deliberately NOT `aud.dsp.channels.fold_to_mono` (which averages): only
    the SIGN of the folded signal matters here (finding where it crosses
    zero), never its level, so a plain sum -- cheaper than a sum-then-
    divide -- is the right computation, not the same one spelled
    differently. See `aud.dsp.channels`'s module docstring, which names
    this function as the one deliberately-unconsolidated copy.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    return np.sum(x, axis=1)


def _nearest_zero_crossing(mono: np.ndarray, sr: int, target_s: float, lo_s: float, hi_s: float) -> float | None:
    """Nearest sample to `target_s`, within [lo_s, hi_s], where `mono` changes sign.

    Returns None if the window contains no sign change (e.g. the window is
    empty, or the material never crosses zero -- a DC-offset window).
    """
    n = mono.shape[0]
    lo_i = max(0, int(np.ceil(lo_s * sr)))
    hi_i = min(n - 1, int(np.floor(hi_s * sr)))
    if hi_i <= lo_i:
        return None

    segment = mono[lo_i : hi_i + 1]
    signs = np.sign(segment)
    signs[signs == 0] = 1.0  # a sample of exactly 0.0 counts as "already crossed"
    changes = np.where(np.diff(signs) != 0)[0]
    if changes.size == 0:
        return None

    target_i = round(target_s * sr)
    target_i = min(max(target_i, lo_i), hi_i)

    candidate_segments = lo_i + changes  # index i such that the crossing is between i and i+1
    nearest_segment = candidate_segments[np.argmin(np.abs(candidate_segments - target_i))]
    i0, i1 = int(nearest_segment), int(nearest_segment) + 1
    chosen = i0 if abs(mono[i0]) <= abs(mono[i1]) else i1
    return chosen / sr


def _quietest_position(mono: np.ndarray, sr: int, lo_s: float, hi_s: float) -> float:
    """Center of the local-minimum-RMS frame within [lo_s, hi_s] -- the "silence" rule's coarse step."""
    n = mono.shape[0]
    lo_i = max(0, round(lo_s * sr))
    hi_i = min(n, round(hi_s * sr))
    if hi_i <= lo_i:
        return (lo_s + hi_s) / 2.0

    frame = max(1, min(int(sr * _SILENCE_FRAME_MS / 1000.0), hi_i - lo_i))
    hop = max(1, int(sr * _SILENCE_HOP_MS / 1000.0))

    best_center = lo_i
    best_rms: float | None = None
    start = lo_i
    while start < hi_i:
        end = min(start + frame, hi_i)
        segment = mono[start:end]
        rms = float(np.sqrt(np.mean(segment * segment))) if segment.size else 0.0
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_center = (start + end) // 2
        start += hop
    return best_center / sr


def _align_to_zero_crossing(
    mono: np.ndarray,
    sr: int,
    target_s: float,
    lo_s: float,
    hi_s: float,
    *,
    success_rule: str,
    fail_reason: str,
    success_snap_failed: bool = False,
    success_reason: str | None = None,
) -> tuple[float | None, str, bool, str | None]:
    """Run the zero-crossing floor in [lo_s, hi_s] around `target_s` and
    report the outcome honestly -- the ONE place in this module that
    handles `_nearest_zero_crossing`'s `None` case.

    Every rule below ends the same way: a bounded, provisional position
    (the padded position itself for "zero_crossing", the quietest frame
    for "silence", the lead-in point before an onset for "transient") run
    through this same floor. If the floor finds nothing, the point is a
    genuine alignment failure and must be reported as "unaligned" with
    `snap_failed=True` -- never as whatever rule was requested, and never
    with `snap_failed=False` while silently keeping the raw candidate.
    That silent-fallback shape is the one bug this module has now grown
    three separate instances of (0.6.0's coarse-miss case, "silence"'s
    failed refinement, "transient"'s failed refinement): routing every
    call site through this single function makes a fourth instance
    structurally impossible -- there is no other place left to forget the
    check.

    Returns `(resolved_s, rule_applied, snap_failed, reason)`, with
    `resolved_s is None` signalling failure -- the caller substitutes its
    own `padded_s` (this function does not know it, and "unaligned"
    always falls back to the raw padded position, never a partial
    candidate).
    """
    found = _nearest_zero_crossing(mono, sr, target_s, lo_s, hi_s)
    if found is None:
        return None, "unaligned", True, fail_reason
    return found, success_rule, success_snap_failed, success_reason


def _resolve_one(
    mono: np.ndarray,
    sr: int,
    *,
    region_index: int,
    boundary: str,
    nominal_s: float,
    padded_s: float,
    snap: str,
    snap_window_ms: float,
    lower_limit: float,
    upper_limit: float,
    duration: float,
    onsets: list[float],
) -> EditPoint:
    def make(resolved_s: float, rule_applied: str, snap_failed: bool, reason: str | None) -> EditPoint:
        moved_ms = (resolved_s - nominal_s) * 1000.0
        return EditPoint(
            region_index=region_index,
            boundary=boundary,
            nominal_s=nominal_s,
            padded_s=padded_s,
            resolved_s=resolved_s,
            moved_ms=moved_ms,
            snap_requested=snap,
            rule_applied=rule_applied,
            snap_failed=snap_failed,
            reason=reason,
        )

    if snap == "none":
        # The only value that disables zero-crossing alignment: it is the
        # only one asserting the caller already picked the sample.
        return make(padded_s, "none", False, None)

    window_s = snap_window_ms / 1000.0
    lo = max(padded_s - window_s, lower_limit, 0.0)
    hi = min(padded_s + window_s, upper_limit, duration)

    if lo >= hi:
        # No floor under the floor: there is no window left to search at
        # all, so nothing -- not even zero-crossing -- can run. "none" is
        # reserved for the caller's literal snap="none"; this is a genuine,
        # always-reportable failure to align (contracts/plan.v1.md#snap).
        reason = (
            f"no position within {snap_window_ms} ms of the padded position stays inside "
            "the region and clear of its neighbours"
        )
        return make(padded_s, "unaligned", True, reason)

    if snap == "zero_crossing":
        resolved, rule, failed, reason = _align_to_zero_crossing(
            mono,
            sr,
            padded_s,
            lo,
            hi,
            success_rule="zero_crossing",
            fail_reason=f"no zero crossing within {snap_window_ms} ms of the padded position",
        )
        return make(resolved if resolved is not None else padded_s, rule, failed, reason)

    if snap == "silence":
        # "silence" always finds A coarse candidate -- there is always a
        # quietest frame in a non-empty window -- so unlike "transient" it
        # can never fail its own coarse step. But the floor underneath it
        # (zero-crossing alignment, searched across the full [lo, hi]
        # window) can still fail: if the whole window never changes sign
        # (a DC-offset region, or a window entirely on one side of zero),
        # there is nowhere in it that is actually safe to cut.
        candidate = _quietest_position(mono, sr, lo, hi)
        resolved, rule, failed, reason = _align_to_zero_crossing(
            mono,
            sr,
            candidate,
            lo,
            hi,
            success_rule="silence",
            fail_reason=(
                f"no zero crossing within {snap_window_ms} ms of the padded position -- the quietest "
                "position found had none in its own search window either, so the floor under 'silence' "
                "failed too"
            ),
        )
        return make(resolved if resolved is not None else padded_s, rule, failed, reason)

    if snap == "transient":
        candidates = [onset for onset in onsets if lo <= onset <= hi]
        if candidates:
            nearest_onset = min(candidates, key=lambda onset: abs(onset - padded_s))
            lead = min(_TRANSIENT_LEAD_S, hi - lo)
            candidate = max(nearest_onset - lead, lo)
            # Refine towards a zero crossing, but never past `candidate` --
            # "always moves a point earlier, never later"
            # (contracts/plan.v1.md) is a promise about the FINAL resolved
            # position, not just the coarse step, so the refinement window
            # stops at the candidate rather than reusing the full [lo, hi]
            # search window. Widening the search past `candidate` on
            # failure is not an option either: it could return a position
            # at or after the onset, which is exactly what this rule
            # exists to prevent. There is no floor under the floor, so a
            # failed refinement here is a genuine alignment failure like
            # every other one in this function.
            resolved, rule, failed, reason = _align_to_zero_crossing(
                mono,
                sr,
                candidate,
                lo,
                candidate,
                success_rule="transient",
                fail_reason=(
                    f"an onset was found within {snap_window_ms} ms of the padded position, but no "
                    "zero crossing exists in the lead-in window before it -- the floor under "
                    "'transient' failed too"
                ),
            )
            return make(resolved if resolved is not None else padded_s, rule, failed, reason)

        # No onset in the window. Zero crossing is the floor under every
        # rule, not a peer of them (contracts/plan.v1.md#snap): a failed
        # coarse search must still be zero-crossing aligned, so the click
        # this rule exists to prevent does not simply move one layer down.
        #
        # "transient" itself is asymmetric across the two boundaries it
        # snaps. At `end` -- the resume point -- a miss is a genuine
        # failure: a real attack could be sitting just past the window,
        # and truncating it is exactly what this rule exists to prevent.
        # At `start` -- the trailing edge into whatever is being removed
        # -- there is structurally nothing upstream of the boundary left
        # to protect (the search window never extends past this region's
        # own end), so a miss there is the expected outcome, not a
        # failure. Either way the point still gets the same fallback
        # alignment; only `snap_failed` differs -- expressed here as which
        # (snap_failed, reason) pair is handed to the shared helper for
        # the success case.
        applies_here = boundary == "end"
        miss_reason = f"no onset within {snap_window_ms} ms of the padded position"
        if applies_here:
            success_snap_failed, success_reason = True, miss_reason
            fail_reason = miss_reason
        else:
            success_snap_failed, success_reason = (
                False,
                f"{miss_reason}; not expected at a region's start boundary, used zero_crossing instead",
            )
            fail_reason = (
                f"{miss_reason}; not expected at a region's start boundary, and no zero crossing was available either"
            )
        resolved, rule, failed, reason = _align_to_zero_crossing(
            mono,
            sr,
            padded_s,
            lo,
            hi,
            success_rule="zero_crossing_fallback",
            success_snap_failed=success_snap_failed,
            success_reason=success_reason,
            fail_reason=fail_reason,
        )
        return make(resolved if resolved is not None else padded_s, rule, failed, reason)

    raise ValueError(f"snap must be one of {_SNAP_MODES}, got {snap!r}")


def resolve_points(
    x: np.ndarray,
    sr: int,
    regions: list[Any],
    *,
    pad_out_ms: float,
    pad_in_ms: float,
    snap: str,
    snap_window_ms: float,
    onsets: list[float] | None = None,
) -> tuple[list[EditPoint], list[int]]:
    """Resolve nominal region boundaries into edit points, pad -> snap -> validate.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,) -- the
            source signal the regions were measured on.
        sr: Sample rate in Hz.
        regions: Regions sorted ascending by `start_s`, non-overlapping.
            Each entry needs `start_s`/`end_s` (dict or attribute access).
        pad_out_ms: Moves each region's `start` point later (shrinks removal).
        pad_in_ms: Moves each region's `end` point earlier (shrinks removal).
        snap: One of "zero_crossing", "silence", "transient", "none".
        snap_window_ms: How far a point may move from its padded position.
        onsets: Onset positions in seconds, needed only for `snap="transient"`.

    Returns:
        (edit_points, dropped_region_indices). `edit_points` has exactly two
        entries (`start` then `end`) per region that survived padding, in
        region order. `dropped_region_indices` lists (by index into
        `regions`) every region where `pad_out_ms + pad_in_ms` consumed the
        whole removal -- contracts/plan.v1.md's `regions_dropped_by_padding`.

    Raises:
        ValueError: `snap` is not one of the four documented values. Real
            plans never reach this: `aud.lib`'s stage builders validate
            `snap` (and `snap_window_ms`) into an `AudError` before a plan
            stage is ever built.
    """
    if snap not in _SNAP_MODES:
        raise ValueError(f"snap must be one of {_SNAP_MODES}, got {snap!r}")

    mono = _mono_sum(x)
    duration = mono.shape[0] / sr
    parsed = [(_get(region, "start_s"), _get(region, "end_s")) for region in regions]
    onset_list = list(onsets) if onsets else []

    edit_points: list[EditPoint] = []
    dropped: list[int] = []

    for i, (start_s, end_s) in enumerate(parsed):
        padded_start = start_s + pad_out_ms / 1000.0
        padded_end = end_s - pad_in_ms / 1000.0

        if padded_start >= padded_end:
            # Padding consumed the whole removal. Not an error: the region
            # is simply not removed, and the caller reports it as dropped.
            dropped.append(i)
            continue

        prev_end = parsed[i - 1][1] if i > 0 else 0.0
        next_start = parsed[i + 1][0] if i + 1 < len(parsed) else duration
        eps = 1.0 / sr  # keeps start strictly before end -- the removal never inverts

        edit_points.append(
            _resolve_one(
                mono,
                sr,
                region_index=i,
                boundary="start",
                nominal_s=start_s,
                padded_s=padded_start,
                snap=snap,
                snap_window_ms=snap_window_ms,
                lower_limit=max(prev_end, 0.0),
                upper_limit=padded_end - eps,
                duration=duration,
                onsets=onset_list,
            )
        )
        edit_points.append(
            _resolve_one(
                mono,
                sr,
                region_index=i,
                boundary="end",
                nominal_s=end_s,
                padded_s=padded_end,
                snap=snap,
                snap_window_ms=snap_window_ms,
                lower_limit=padded_start + eps,
                upper_limit=min(next_start, duration),
                duration=duration,
                onsets=onset_list,
            )
        )

    return edit_points, dropped
