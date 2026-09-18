"""Tests for aud.lib.compress's plan-document shape.

DEFECT 2: the compress stage builder emitted {"bands": [freqs...], "ratio":
X} -- a shape the DSP engine could not consume (it needs the crossover
frequencies and the per-band settings under different keys, per
contracts/plan.v1.md). These tests pin the builder's output to that
contract shape directly, without needing to render actual audio.
"""

from __future__ import annotations

from aud.lib import compress
from aud.plan import new_plan


def test_compress_emits_crossovers_hz_and_one_band_object_per_band() -> None:
    plan = compress(new_plan(), bands=[120.0, 900.0, 5500.0], ratio=3.0)
    stage = plan.stages[-1]

    assert stage.stage == "compress"
    assert stage.params["crossovers_hz"] == [120.0, 900.0, 5500.0]
    assert len(stage.params["bands"]) == len(stage.params["crossovers_hz"]) + 1


def test_compress_band_objects_carry_every_field_the_engine_needs() -> None:
    plan = compress(new_plan(), bands=[1000.0], ratio=4.0)
    stage = plan.stages[-1]

    for band in stage.params["bands"]:
        assert band.keys() == {
            "threshold_db",
            "ratio",
            "attack_ms",
            "release_ms",
            "knee_db",
            "makeup_db",
        }
        assert band["ratio"] == 4.0


def test_compress_single_crossover_yields_exactly_two_bands() -> None:
    plan = compress(new_plan(), bands=[1000.0], ratio=2.0)
    stage = plan.stages[-1]
    assert len(stage.params["bands"]) == 2
