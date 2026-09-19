"""D5 regression: a filler-kind `cut` must remove more than the literal span.

`aud detect fillers` reports a filler word's END timestamp systematically
175-200 ms early (see aud/lib.py's `_FILLER_TAIL_PAD_MS` and
contracts/plan.v1.md#why-filler_tail_pad_ms-exists-and-why-it-is-not-padding
for the measurement). These tests prove:

1. A filler-kind document gets a 200 ms end-of-region extension by default.
2. That extension actually removes more audio at render time, not just in
   the stored plan.
3. A caller's explicit override wins (including 0.0, which disables it).
4. Any other kind of document (silence) is completely unaffected -- `cut`'s
   general defaults do not change.
5. The extension never crosses into a neighbouring region.

Note: this is deliberately implemented as a new `filler_tail_pad_ms`
parameter, not via `pad_out_ms`/`pad_in_ms`. contracts/plan.v1.md#padding is
explicit that padding can only ever SHRINK a removal ("a control that could
also remove more would not be safe to reach for") -- repurposing it here
would contradict that documented, deliberate invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud import lib
from aud.core.regions import new_regions, write_regions
from aud.dsp import engine
from aud.plan import new_plan, ordered
from aud.schemas import AudError

SR = 44100

_FILLER_DETECTION = {
    "words": ["um", "uh"],
    "min_pause_ms": 700.0,
    "engine": "faster-whisper",
    "model": "tiny.en",
}
_SILENCE_DETECTION = {"threshold_above_floor_db": 6.0, "min_len_ms": 400.0, "noise_floor_dbfs": -50.0}


def _filler_region(start_s: float, end_s: float, text: str = "um") -> dict:
    return {"start_s": start_s, "end_s": end_s, "text": text, "confidence": 0.9}


def _silence_region(start_s: float, end_s: float) -> dict:
    return {"start_s": start_s, "end_s": end_s, "peak_dbfs": -50.0, "rms_dbfs": -55.0}


def _tone(seconds: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 0.4 * np.sin(2.0 * np.pi * freq * t)


def test_filler_kind_defaults_to_200ms_tail_extension_in_the_stored_plan():
    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2)],
    )
    plan = lib.cut(new_plan(), write_regions(doc))

    stage = plan.stages[0]
    assert stage.params["filler_tail_pad_ms"] == pytest.approx(200.0)
    assert stage.params["regions"][0]["end_s"] == pytest.approx(1.4)  # 1.2 + 200ms
    assert stage.params["regions"][0]["start_s"] == pytest.approx(1.0)  # start untouched


def test_filler_default_removes_more_audio_than_the_literal_span_at_render():
    """The acceptance criterion itself: render, and measure what actually got cut."""
    x = _tone(3.0)
    x2 = np.stack([x, x], axis=1)

    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2)],
    )
    # snap="none" + crossfade_ms=0 so the reported removal is exact, not
    # perturbed by zero-crossing alignment -- isolates the one thing under test.
    plan = lib.cut(new_plan(), write_regions(doc), snap="none", crossfade_ms=0.0)

    y, report = engine.apply_plan(x2, SR, ordered(plan))
    cut_report = report["stages"][0]
    literal_span_s = 1.2 - 1.0
    print(f"\n[cut] literal span={literal_span_s:.3f}s, seconds_removed={cut_report['seconds_removed']:.3f}s")
    assert cut_report["seconds_removed"] > literal_span_s
    assert cut_report["seconds_removed"] == pytest.approx(0.4, abs=1e-6)  # (1.4 - 1.0), not (1.2 - 1.0)
    assert y.shape[0] == pytest.approx(x2.shape[0] - round(0.4 * SR), abs=1)


def test_explicit_zero_disables_the_default():
    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2)],
    )
    plan = lib.cut(new_plan(), write_regions(doc), filler_tail_pad_ms=0.0)

    stage = plan.stages[0]
    assert stage.params["filler_tail_pad_ms"] == 0.0
    assert stage.params["regions"][0]["end_s"] == pytest.approx(1.2)


def test_explicit_override_value_wins_over_the_kind_aware_default():
    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2)],
    )
    plan = lib.cut(new_plan(), write_regions(doc), filler_tail_pad_ms=50.0)

    stage = plan.stages[0]
    assert stage.params["filler_tail_pad_ms"] == 50.0
    assert stage.params["regions"][0]["end_s"] == pytest.approx(1.25)  # 1.2 + 50ms


def test_non_filler_kind_is_unaffected_by_the_filler_default():
    """cut's general default stays 0 for every other kind -- only `filler` changes."""
    doc = new_regions(
        kind="silence",
        source="in.wav",
        sample_rate=SR,
        detection=_SILENCE_DETECTION,
        regions=[_silence_region(1.0, 1.2)],
    )
    plan = lib.cut(new_plan(), write_regions(doc))

    stage = plan.stages[0]
    assert stage.params["filler_tail_pad_ms"] == 0.0
    assert stage.params["regions"][0]["end_s"] == pytest.approx(1.2)


def test_extension_is_clamped_to_not_cross_into_the_next_region():
    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2), _filler_region(1.25, 1.5, text="uh")],
    )
    plan = lib.cut(new_plan(), write_regions(doc))

    stage = plan.stages[0]
    # Unclamped, 200ms would push the first region's end to 1.4 -- past the
    # second region's start_s (1.25). It must stop at 1.25, not overshoot.
    assert stage.params["regions"][0]["end_s"] == pytest.approx(1.25)
    # Last region has no next-region limit, so it gets the full 200ms extension.
    assert stage.params["regions"][1]["end_s"] == pytest.approx(1.7)


def test_negative_override_is_a_bad_param():
    doc = new_regions(
        kind="filler",
        source="in.wav",
        sample_rate=SR,
        detection=_FILLER_DETECTION,
        regions=[_filler_region(1.0, 1.2)],
    )
    with pytest.raises(AudError) as exc_info:
        lib.cut(new_plan(), write_regions(doc), filler_tail_pad_ms=-5.0)
    assert exc_info.value.code == "bad_param"
