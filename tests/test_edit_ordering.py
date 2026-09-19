"""Proves canonical stage order, not append order, governs rendering --
and that it matters numerically for `cut` + `loudness`.

contracts/plan.v1.md#why-editing-is-first: loudness targeted over material
`cut` later removes describes a file that no longer exists. This test
builds a signal whose loudness changes materially when the loud region is
removed, appends the stages in the WRONG order, and shows the render still
comes out right -- because `render` always applies `STAGE_ORDER`, never the
plan's append order.
"""

from __future__ import annotations

import numpy as np

from aud import lib
from aud.core.regions import new_regions, write_regions
from aud.dsp import engine, loudness
from aud.plan import new_plan, ordered

SR = 44100


def _signal_with_loud_middle() -> np.ndarray:
    seconds = 3.0
    t = np.arange(int(seconds * SR)) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    loud = slice(int(1.0 * SR), int(2.0 * SR))
    x[loud] *= 4.0  # a materially louder middle third
    return np.stack([x, x], axis=1)


def test_cut_renders_before_loudness_even_when_appended_after_it():
    x = _signal_with_loud_middle()
    target_lufs = -20.0

    # Append LOUDNESS first, CUT second -- the opposite of canonical order --
    # to prove `ordered()` (not append order) decides what actually renders.
    plan = new_plan()
    plan = lib.loudness(plan, target_lufs=target_lufs)

    regions_doc = new_regions(
        kind="silence",
        source="in.wav",
        sample_rate=SR,
        detection={"threshold_above_floor_db": 6.0, "min_len_ms": 400.0, "noise_floor_dbfs": -50.0},
        regions=[{"start_s": 1.0, "end_s": 2.0, "peak_dbfs": -1.0, "rms_dbfs": -3.0}],
    )
    plan = lib.cut(plan, write_regions(regions_doc), crossfade_ms=5.0)

    stages = ordered(plan)
    stage_names_in_render_order = [stage.stage for stage in stages]
    print(f"\n[ordering] plan.stages append order: {[s.stage for s in plan.stages]}")
    print(f"[ordering] render order:              {stage_names_in_render_order}")
    assert stage_names_in_render_order == ["cut", "loudness"]

    y, report = engine.apply_plan(x, SR, stages)
    assert [s["stage"] for s in report["stages"]] == ["cut", "loudness"]

    measured_lufs = loudness.integrated_lufs(y, SR)
    print(f"[ordering] measured LUFS of rendered (cut-then-loudness) output: {measured_lufs:.3f}")
    assert abs(measured_lufs - target_lufs) < 0.5

    # Falsifiability check: if loudness had instead been computed against the
    # ORIGINAL (uncut, louder) material -- what "loudness before cut" would
    # have produced -- delivering that same gain against the actually-cut
    # material misses the target by a wide, unmistakable margin. This is the
    # concrete evidence that the assertion above is not vacuously true.
    _, wrong_order_gain_db = loudness.normalize(x, SR, target_lufs)
    cut_only, _ = engine.apply_plan(x, SR, [s for s in stages if s.stage == "cut"])
    wrong_order_result = cut_only * (10.0 ** (wrong_order_gain_db / 20.0))
    wrong_order_lufs = loudness.integrated_lufs(wrong_order_result, SR)
    print(f"[ordering] counterfactual (loudness-before-cut) LUFS: {wrong_order_lufs:.3f}")
    assert abs(wrong_order_lufs - target_lufs) > 1.0  # meaningfully off -- the ordering really matters
