"""Regression tests for the deterministic EQ sign-contradiction guard.

The defect this guards against, measured end-to-end (not hypothetical): with
`rel_reference_db` present, a real model read the signed measurement and its
own prose correctly ("16 kHz is -36.79 dB below reference... extreme
high-frequency rolloff... lacks presence") and then proposed a further CUT
at that exact frequency -- the opposite of what its own stated reasoning
called for. `advisor.py`'s `_eq_move_contradiction`/`_filter_contradictory_eq_moves`
catch this deterministically, for free, with no model call: every test
below is a plain function call over measurements or a synthetic proposal,
never a provider round trip.

Six induced-defect fixtures are reused from `tests/test_reference_anchored_tonal.py`
(`CLEAN_MONO`, `DEFECTS`) rather than inventing new material -- those are
the same six controlled boxy/rumbly/dull/harsh/muddy/thin cases already
proven (in that file) to move the spectrum in a known, confirmed direction,
and already proven to score 6 of 6 correct band+direction with
`rel_reference_db`.
"""

from __future__ import annotations

from typing import Any

import pytest

from aud.dsp import analysis
from aud.intelligence import advisor
from aud.schemas import AudError
from tests.test_reference_anchored_tonal import CLEAN_MONO, DEFECTS, SR, _peak_limit, _to_stereo

_REFERENCE_STEREO = _to_stereo(_peak_limit(CLEAN_MONO))


def _measurements_for(name: str, *, with_reference: bool) -> dict[str, Any]:
    """Real `analyze()` output for one of the six induced-defect fixtures."""
    _direction, _expected_hz, _check_freq, defect = DEFECTS[name]
    defect_stereo = _to_stereo(_peak_limit(defect))
    if with_reference:
        return analysis.analyze(defect_stereo, SR, reference_x=_REFERENCE_STEREO, reference_sr=SR)
    return analysis.analyze(defect_stereo, SR)


def _band_value(measurements: dict[str, Any], hz: float, key: str = "rel_reference_db") -> float | None:
    for entry in measurements["octave_band_analysis"]:
        if entry["hz"] == hz:
            return entry.get(key)
    raise AssertionError(f"no {hz} Hz band in octave_band_analysis")


def _eq_proposal(freq_hz: float, gain_db: float, *, kind: str = "peak") -> dict[str, Any]:
    if kind == "peak":
        params: dict[str, Any] = {"peaks": [[freq_hz, gain_db, 1.2]]}
    else:
        params = {"shelves": [["high", freq_hz, gain_db, 0.707]]}
    return {
        "stages": [
            {
                "stage": "eq",
                "params": params,
                "reason": f"testing a {gain_db:+.1f} dB move at {freq_hz:g} Hz",
            }
        ]
    }


# --- Fixture sanity: confirm the sign we are about to test against --------


def test_dull_fixture_16khz_is_measured_deficient_vs_reference() -> None:
    measurements = _measurements_for("dull", with_reference=True)
    deviation = _band_value(measurements, 16000.0)
    assert deviation is not None, deviation
    assert deviation < -4.0, deviation


def test_boxy_fixture_500hz_is_measured_in_excess_vs_reference() -> None:
    measurements = _measurements_for("boxy", with_reference=True)
    deviation = _band_value(measurements, 500.0)
    assert deviation is not None, deviation
    assert deviation > 4.0, deviation


# --- The reported defect, caught: a cut proposed in a DEFICIENT band ------


def test_cut_in_a_deficient_band_is_dropped_and_reported() -> None:
    """The exact shape of the reported bug: 16 kHz is ~-36 dB deficient
    (dull fixture), and the model proposes cutting it further. The move
    must not survive into the built plan, and its removal must be visible
    in the returned reasoning."""
    measurements = _measurements_for("dull", with_reference=True)
    proposal = _eq_proposal(16000.0, -6.0)
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["peaks"] == [], "the contradictory cut must not reach the built plan"

    guarded = [r for r in reasoning if r["stage"] == "eq" and r["reason"].startswith("GUARDED:")]
    assert len(guarded) == 1, reasoning
    assert "16000" in guarded[0]["reason"]
    assert "cut" in guarded[0]["reason"].lower()

    # The model's own (unmodified) reason is still recorded -- dropping a
    # move is not the same as pretending the model never said anything.
    original = [r for r in reasoning if r["stage"] == "eq" and not r["reason"].startswith("GUARDED:")]
    assert len(original) == 1


def test_cut_in_a_deficient_band_is_dropped_when_expressed_as_a_shelf() -> None:
    measurements = _measurements_for("dull", with_reference=True)
    proposal = _eq_proposal(16000.0, -6.0, kind="shelf")
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["shelves"] == []
    guarded = [r for r in reasoning if r["stage"] == "eq" and r["reason"].startswith("GUARDED:")]
    assert len(guarded) == 1, reasoning


# --- The control: a cut in an EXCESS band passes untouched -----------------
#
# This is what proves the guard is not simply rejecting every cut -- only
# a cut whose direction contradicts its own band's measurement.


def test_cut_in_an_excess_band_passes_untouched() -> None:
    measurements = _measurements_for("boxy", with_reference=True)
    proposal = _eq_proposal(500.0, -3.0)
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["peaks"] == [{"freq_hz": 500.0, "gain_db": -3.0, "q": 1.2}]
    assert not any(r["reason"].startswith("GUARDED:") for r in reasoning), reasoning


# --- A boost in a deficient band also passes untouched ---------------------


def test_boost_in_a_deficient_band_passes_untouched() -> None:
    measurements = _measurements_for("dull", with_reference=True)
    proposal = _eq_proposal(16000.0, 6.0)
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["peaks"] == [{"freq_hz": 16000.0, "gain_db": 6.0, "q": 1.2}]
    assert not any(r["reason"].startswith("GUARDED:") for r in reasoning), reasoning


# --- With no reference supplied, behaviour is unchanged --------------------


def test_no_reference_supplied_leaves_a_would_be_contradiction_untouched() -> None:
    """Same proposal as the caught case above, but measured with no
    reference file -- `rel_reference_db` is then ABSENT from every band
    (see `dsp.analysis.analyze`'s docstring), and the guard must be a
    strict no-op: never fall back to guarding against `rel_median_db`."""
    measurements = _measurements_for("dull", with_reference=False)
    assert all("rel_reference_db" not in entry for entry in measurements["octave_band_analysis"])

    proposal = _eq_proposal(16000.0, -6.0)
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["peaks"] == [{"freq_hz": 16000.0, "gain_db": -6.0, "q": 1.2}]
    assert not any(r["reason"].startswith("GUARDED:") for r in reasoning), reasoning


def test_no_measurements_at_all_leaves_behaviour_unchanged() -> None:
    """`measurements=None` (the default) -- e.g. every existing caller in
    tests/test_intelligence.py that calls `_validate_and_build_plan` with a
    single argument -- must behave exactly as before this guard existed."""
    proposal = _eq_proposal(16000.0, -6.0)
    plan, reasoning = advisor._validate_and_build_plan(proposal)

    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["peaks"] == [{"freq_hz": 16000.0, "gain_db": -6.0, "q": 1.2}]
    assert not any(r["reason"].startswith("GUARDED:") for r in reasoning), reasoning


# --- Band-to-frequency matching: nearest centre in log2 (octave) space ----


def test_nearest_band_hz_assigns_6khz_to_the_8khz_band() -> None:
    """A move at 6000 Hz sits between the 4000 Hz and 8000 Hz octave bands;
    the boundary between them is sqrt(4000*8000) ~= 5657 Hz, so 6000 Hz is
    inside (and nearer, in octaves, to) the 8000 Hz band."""
    assert advisor._nearest_band_hz(6000.0, [4000.0, 8000.0]) == 8000.0


def test_nearest_band_hz_assigns_5khz_to_the_4khz_band() -> None:
    """5000 Hz sits on the other side of the same 5657 Hz boundary."""
    assert advisor._nearest_band_hz(5000.0, [4000.0, 8000.0]) == 4000.0


# --- Pure-function unit tests: no fixtures, synthetic measurements --------


def test_eq_move_contradiction_flags_cut_in_deficient_band() -> None:
    note = advisor._eq_move_contradiction(16000.0, -6.0, {16000.0: -36.0})
    assert note is not None
    assert "16000" in note


def test_eq_move_contradiction_flags_boost_in_excess_band() -> None:
    """The mirror direction of the reported bug: boosting a band that is
    already measured in excess."""
    note = advisor._eq_move_contradiction(500.0, 3.0, {500.0: 12.0})
    assert note is not None
    assert "500" in note


def test_eq_move_contradiction_passes_cut_in_excess_band() -> None:
    assert advisor._eq_move_contradiction(500.0, -3.0, {500.0: 12.0}) is None


def test_eq_move_contradiction_passes_boost_in_deficient_band() -> None:
    assert advisor._eq_move_contradiction(16000.0, 6.0, {16000.0: -36.0}) is None


def test_eq_move_contradiction_ignores_near_zero_deviation() -> None:
    """Within +/-tolerance of zero there is nothing to contradict."""
    assert advisor._eq_move_contradiction(1000.0, -3.0, {1000.0: 0.2}) is None


def test_eq_move_contradiction_ignores_zero_gain() -> None:
    assert advisor._eq_move_contradiction(1000.0, 0.0, {1000.0: -20.0}) is None


def test_octave_reference_lookup_is_none_with_no_measurements() -> None:
    assert advisor._octave_reference_lookup(None) is None


def test_octave_reference_lookup_is_none_when_no_band_carries_it() -> None:
    measurements = {"octave_band_analysis": [{"hz": 1000.0, "rel_median_db": 1.0}]}
    assert advisor._octave_reference_lookup(measurements) is None


def test_octave_reference_lookup_maps_hz_to_rel_reference_db() -> None:
    measurements = {
        "octave_band_analysis": [
            {"hz": 1000.0, "rel_reference_db": 2.0},
            {"hz": 2000.0, "rel_reference_db": None},
        ]
    }
    assert advisor._octave_reference_lookup(measurements) == {1000.0: 2.0}


# --- _filter_contradictory_eq_moves: defensive on malformed entries -------


def test_filter_leaves_a_malformed_peak_for_the_builder_to_reject() -> None:
    """A shape the model got wrong is not this guard's job -- `eq()`'s own
    validation still raises `bad_model_plan` for it, unchanged."""
    params = {"peaks": [[16000.0, -6.0]]}  # missing q: still a 2-tuple, not 3
    filtered, notes = advisor._filter_contradictory_eq_moves(params, {16000.0: -36.0})
    assert filtered["peaks"] == [[16000.0, -6.0]]
    assert notes == []


def test_filter_drops_one_contradictory_peak_and_keeps_a_good_one() -> None:
    params = {"peaks": [[16000.0, -6.0, 1.2], [500.0, -3.0, 1.2]]}
    reference_lookup = {16000.0: -36.0, 500.0: 12.0}
    filtered, notes = advisor._filter_contradictory_eq_moves(params, reference_lookup)
    assert filtered["peaks"] == [[500.0, -3.0, 1.2]]
    assert len(notes) == 1
    assert "peak:" in notes[0]


# --- hpf/lpf are not guarded -----------------------------------------------


def test_hpf_lpf_are_never_touched_by_the_guard() -> None:
    """`eq`'s hpf/lpf corners carry no `gain_db` of their own -- see
    advisor.py's guard docstring for why they are out of scope."""
    measurements = _measurements_for("rumbly", with_reference=True)  # low end in excess
    proposal = {
        "stages": [
            {
                "stage": "eq",
                "params": {"hpf": 80.0, "lpf": None, "peaks": [], "shelves": []},
                "reason": "trim rumble below 80 Hz",
            }
        ]
    }
    plan, reasoning = advisor._validate_and_build_plan(proposal, measurements)
    eq_stage = next(s for s in plan.stages if s.stage == "eq")
    assert eq_stage.params["hpf_hz"] == 80.0
    assert not any(r["reason"].startswith("GUARDED:") for r in reasoning), reasoning


# --- A malformed proposal still raises bad_model_plan, guard or no guard --


def test_bad_model_plan_still_raised_for_an_out_of_range_peak_with_a_reference() -> None:
    measurements = _measurements_for("dull", with_reference=True)
    proposal = _eq_proposal(-100.0, -6.0)  # negative frequency: eq()'s own validation must catch this
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(proposal, measurements)
    assert excinfo.value.code == "bad_model_plan"
