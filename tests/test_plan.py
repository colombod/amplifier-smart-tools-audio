"""Tests for aud.plan -- the mastering plan document."""

from __future__ import annotations

import pytest

from aud.plan import STAGE_ORDER, append, new_plan, ordered, read_plan, write_plan
from aud.schemas import AudError


def test_append_never_mutates() -> None:
    original = new_plan()
    assert original.stages == []
    updated = append(original, "eq", {"hpf": 40})
    assert original.stages == [], "append must not mutate the original plan's stages"
    assert original is not updated
    assert len(updated.stages) == 1


def test_append_returns_a_new_plan_object_each_time() -> None:
    plan = new_plan()
    first = append(plan, "eq", {"hpf": 40})
    second = append(first, "limit", {"ceiling_dbtp": -1.0})
    assert first is not second
    assert len(first.stages) == 1
    assert len(second.stages) == 2


def test_ordered_respects_canonical_mastering_order_regardless_of_append_order() -> None:
    plan = new_plan()
    plan = append(plan, "limit", {"ceiling_dbtp": -1.0})  # appended first
    plan = append(plan, "eq", {"hpf": 40})  # appended second
    names = [stage.stage for stage in ordered(plan)]
    assert names.index("eq") < names.index("limit")


def test_ordered_keeps_insertion_order_within_the_same_stage_name() -> None:
    plan = new_plan()
    plan = append(plan, "eq", {"hpf": 40})
    plan = append(plan, "eq", {"lpf": 18000})
    eq_stages = [stage for stage in ordered(plan) if stage.stage == "eq"]
    assert len(eq_stages) == 2
    assert eq_stages[0].params == {"hpf": 40}
    assert eq_stages[1].params == {"lpf": 18000}


def test_stage_order_is_the_canonical_mastering_chain() -> None:
    assert STAGE_ORDER == [
        "cut",
        "strip_silence",
        "stretch",
        "pitch",
        "gate",
        "expand",
        "dereverb",
        "deess",
        "eq",
        "eq_match",
        "compress",
        "saturate",
        "reverb",
        "loudness",
        "limit",
        "downmix",
        "resample",
    ]


def test_editing_stages_render_before_any_timeline_or_measurement_stage() -> None:
    """The reason behind the order, not just the order.

    `cut` carries absolute positions measured on the source timeline, so
    `stretch` -- which re-times the programme -- must run after it. And
    `loudness` averages over duration, so it must measure material that
    survives the edit. See contracts/plan.v1.md, "Why editing is first".
    """
    plan = new_plan()
    plan = append(plan, "limit", {"ceiling_dbtp": -1.0})
    plan = append(plan, "stretch", {"ratio": 0.98})
    plan = append(plan, "loudness", {"target_lufs": -14.0})
    plan = append(plan, "strip_silence", {})
    plan = append(plan, "cut", {"regions": []})  # appended last, renders first
    names = [stage.stage for stage in ordered(plan)]
    assert names[:2] == ["cut", "strip_silence"]
    assert names.index("cut") < names.index("stretch") < names.index("loudness")


def test_downmix_and_resample_render_after_every_other_stage() -> None:
    """Output-format stages sit at the very end -- see
    contracts/plan.v1.md#why-downmixresample-sit-at-the-very-end.
    """
    plan = new_plan()
    plan = append(plan, "resample", {"target_hz": 48000})
    plan = append(plan, "eq", {})
    plan = append(plan, "downmix", {})
    plan = append(plan, "limit", {"ceiling_dbtp": -1.0})
    names = [stage.stage for stage in ordered(plan)]
    assert names[-2:] == ["downmix", "resample"]
    assert names.index("limit") < names.index("downmix") < names.index("resample")


def test_round_trip_write_then_read_is_lossless() -> None:
    plan = new_plan()
    plan = append(plan, "eq", {"hpf": 40, "peaks": [[3200.0, -2.5, 1.4]]})
    plan = append(plan, "limit", {"ceiling_dbtp": -1.0})
    restored = read_plan(write_plan(plan))
    assert restored == plan


def test_read_plan_on_none_returns_new_plan() -> None:
    assert read_plan(None) == new_plan()


def test_read_plan_on_empty_string_returns_new_plan() -> None:
    assert read_plan("") == new_plan()
    assert read_plan("   \n  ") == new_plan()


def test_append_unknown_stage_raises_unknown_stage() -> None:
    plan = new_plan()
    with pytest.raises(AudError) as exc_info:
        append(plan, "not-a-real-stage", {})
    assert exc_info.value.code == "unknown_stage"
    assert exc_info.value.remedy


def test_read_plan_malformed_json_raises_bad_plan() -> None:
    with pytest.raises(AudError) as exc_info:
        read_plan("{not valid json")
    assert exc_info.value.code == "bad_plan"


def test_read_plan_malformed_shape_raises_bad_plan() -> None:
    with pytest.raises(AudError) as exc_info:
        read_plan('{"plan_format": 1}')
    assert exc_info.value.code == "bad_plan"
