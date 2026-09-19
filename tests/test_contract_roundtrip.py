"""The property that makes contracts/plan.v1.md's field names load-bearing:

    A plan document hand-written using ONLY the field names, shapes and
    defaults contracts/plan.v1.md states -- for every stage this build
    implements -- must actually render.

Before this test existed, the builders (aud.lib) and the render engine
(aud.dsp.engine) agreed with EACH OTHER on a wire shape that was not what
the contract documented (`hpf`/`lpf` instead of `hpf_hz`/`lpf_hz`, raw EQ
peak triples instead of `{freq_hz, gain_db, q}` objects, a `compress.ratio`
floor of 0 instead of the documented 1.0, contract-listed `limit`/
`compress` defaults that did not match what the engine actually assumed
for an omitted field). A caller who read only the contract -- exactly what
it promises a caller may do -- would have written a plan the tool could
not use. Asserting the two agree with each other is not the same claim;
this file asserts the tool agrees with its own published contract, by
constructing plans the way the contract itself says they look, never by
calling `aud.lib`'s builders.

Only stages actually implemented in this build's render engine
(`aud.dsp.engine._REGISTRY`) are exercised -- an unimplemented stage
correctly refuses with `not_implemented`, which is not what this test is
about.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import engine
from aud.plan import Plan, ordered
from aud.schemas import NotImplementedStageError

SR = 48000


def _noise(seconds: float = 1.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=(int(seconds * sr), 2))


def _hand_written_plan(stages: list[dict]) -> Plan:
    """Build a Plan exactly the way a third-party generator would: from the
    contract's own JSON shape, never through aud.lib's convenience builders."""
    return Plan.model_validate(
        {
            "plan_format": 1,
            "created_with": "third-party-generator/1.0",
            "stages": stages,
        }
    )


def _render(plan: Plan, x: np.ndarray) -> tuple[np.ndarray, dict]:
    try:
        return engine.apply_plan(x, SR, ordered(plan))
    except NotImplementedStageError as exc:  # pragma: no cover - guard, not the point of this test
        pytest.fail(f"stage not implemented in this build: {exc}")


# ---------------------------------------------------------------------------
# eq -- hpf_hz/lpf_hz, peaks as {freq_hz, gain_db, q} objects, shelves too
# ---------------------------------------------------------------------------


def test_hand_written_eq_with_contract_field_names_renders() -> None:
    plan = _hand_written_plan(
        [
            {
                "stage": "eq",
                "params": {
                    "hpf_hz": 40.0,
                    "lpf_hz": 18000.0,
                    "peaks": [{"freq_hz": 3200.0, "gain_db": -2.5, "q": 1.4}],
                    "shelves": [{"type": "low", "freq_hz": 100.0, "gain_db": 2.0, "q": 0.7}],
                },
            }
        ]
    )
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    eq_report = report["stages"][0]
    assert eq_report["hpf_hz"] == 40.0
    assert eq_report["lpf_hz"] == 18000.0
    assert len(eq_report["peaks_applied"]) == 1
    assert len(eq_report["shelves_applied"]) == 1
    assert eq_report["shelves_applied"][0]["type"] == "low"


def test_hand_written_eq_with_only_documented_defaults_renders() -> None:
    """params: {} is valid per the contract -- every field has a default."""
    plan = _hand_written_plan([{"stage": "eq", "params": {}}])
    y, _ = _render(plan, _noise())
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# compress -- crossovers_hz/bands, ratio >= 1.0, omitted band fields use the
# contract's defaults, and crossovers_hz: [] (single-band) renders.
# ---------------------------------------------------------------------------


def test_hand_written_compress_multiband_with_full_band_objects_renders() -> None:
    plan = _hand_written_plan(
        [
            {
                "stage": "compress",
                "params": {
                    "crossovers_hz": [300.0, 3000.0],
                    "bands": [
                        {"threshold_db": -24.0, "ratio": 2.0, "attack_ms": 20.0, "release_ms": 180.0},
                        {"threshold_db": -20.0, "ratio": 3.0, "attack_ms": 15.0, "release_ms": 150.0},
                        {"threshold_db": -18.0, "ratio": 2.5, "attack_ms": 10.0, "release_ms": 120.0},
                    ],
                },
            }
        ]
    )
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    assert len(report["stages"][0]["bands"]) == 3


def test_hand_written_compress_band_with_no_fields_uses_the_contracts_defaults() -> None:
    """An empty band object is valid (every band field has a documented
    default) -- and those defaults must be the CONTRACT's, not whatever
    dsp.dynamics.BandParams's dataclass happens to default to internally."""
    plan = _hand_written_plan(
        [
            {
                "stage": "compress",
                "params": {"crossovers_hz": [], "bands": [{}]},
            }
        ]
    )
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    band_report = report["stages"][0]["bands"][0]
    assert band_report["threshold_db"] == -24.0
    assert band_report["ratio"] == 2.0


def test_hand_written_compress_with_empty_crossovers_is_single_band_and_renders() -> None:
    """contracts/plan.v1.md: 'N crossovers produce N + 1 bands; [] is single-band.'"""
    plan = _hand_written_plan(
        [
            {
                "stage": "compress",
                "params": {"crossovers_hz": [], "bands": [{"ratio": 3.0}]},
            }
        ]
    )
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    compress_report = report["stages"][0]
    assert compress_report["crossovers_hz"] == []
    assert len(compress_report["bands"]) == 1
    assert compress_report["bands"][0]["ratio"] == 3.0


# ---------------------------------------------------------------------------
# saturate -- drive >= 0 (0.0 must be accepted), mix omitted defaults to 1.0
# ---------------------------------------------------------------------------


def test_hand_written_saturate_with_zero_drive_renders() -> None:
    """contracts/plan.v1.md: drive >= 0 -- 0.0 is a documented valid value."""
    plan = _hand_written_plan([{"stage": "saturate", "params": {"drive": 0.0, "mix": 1.0}}])
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    assert report["stages"][0]["drive"] == 0.0


def test_hand_written_saturate_with_no_params_uses_documented_defaults() -> None:
    plan = _hand_written_plan([{"stage": "saturate", "params": {}}])
    _, report = _render(plan, _noise())
    assert report["stages"][0]["drive"] == 1.0
    assert report["stages"][0]["mix"] == 1.0


# ---------------------------------------------------------------------------
# loudness -- target_lufs < 0, no undocumented lower bound
# ---------------------------------------------------------------------------


def test_hand_written_loudness_with_a_target_below_negative_sixty_renders() -> None:
    """contracts/plan.v1.md documents only '< 0' -- no lower bound at all."""
    plan = _hand_written_plan([{"stage": "loudness", "params": {"target_lufs": -70.0}}])
    y, report = _render(plan, _noise())
    assert np.all(np.isfinite(y))
    assert report["stages"][0]["target_lufs"] == -70.0


# ---------------------------------------------------------------------------
# limit -- ceiling_dbtp/lookahead_ms/release_ms/oversample, all by contract
# ---------------------------------------------------------------------------


# --- Regression guard: `ceiling_met` must not be vacuous -------------------
#
# `_noise()` (std 0.1) never comes close to a -1.0 dBTP ceiling, so
# `ceiling_met is True` on it would pass whether or not `ceiling_dbtp`,
# `lookahead_ms`, `release_ms` and `oversample` were even read off the
# hand-written params -- see the agent report and
# tests/test_dsp_limiter.py's own regression-guard comment. This is
# specifically the test that claims to exercise every one of those
# documented fields, so it should force the limiter to actually engage
# using them, not just accept the field names syntactically. Reuses
# test_dsp_limiter.py's own proven recipe (a faded plateau several dB over
# the ceiling) directly against `engine.apply_plan`, so it is exercised
# through the exact contract-shaped params a hand-written plan would use.
_FORCED_CEILING_DBTP = -1.0
_FORCED_HEADROOM_OVER_CEILING_DB = 6.0
_FORCED_FADE_MS = 50.0  # avoids resample_poly FIR ringing on a hard edge -- see test_dsp_limiter.py
_GAIN_REDUCTION_TOLERANCE_DB = 0.05  # same tolerance test_dsp_limiter.py uses for this exact recipe


def _forced_plateau_signal(sr: int, seconds: float = 1.0) -> tuple[np.ndarray, float]:
    """A constant-level plateau `_FORCED_HEADROOM_OVER_CEILING_DB` dB over
    the ceiling, faded in/out. See test_dsp_limiter.py's identical helper
    for the full rationale (measured there; holds here too since it is the
    same `dsp.limiter.brickwall` underneath).
    """
    input_peak_dbtp = _FORCED_CEILING_DBTP + _FORCED_HEADROOM_OVER_CEILING_DB
    amplitude = 10.0 ** (input_peak_dbtp / 20.0)
    n = int(seconds * sr)
    fade_n = int(_FORCED_FADE_MS / 1000.0 * sr)
    x = np.full(n, amplitude)
    ramp = np.linspace(0.0, 1.0, fade_n)
    x[:fade_n] *= ramp
    x[-fade_n:] *= ramp[::-1]
    return np.stack([x, x], axis=1), input_peak_dbtp


def test_hand_written_limit_with_every_documented_field_renders() -> None:
    x, input_peak_dbtp = _forced_plateau_signal(SR)
    plan = _hand_written_plan(
        [
            {
                "stage": "limit",
                "params": {
                    "ceiling_dbtp": -1.0,
                    "lookahead_ms": 5.0,
                    "release_ms": 50.0,
                    "oversample": 8,
                },
            }
        ]
    )
    y, report = _render(plan, x)
    assert np.all(np.isfinite(y))
    limit_report = report["stages"][0]
    print(f"\n[contract] limit stage: {limit_report}")
    # The canary: without this, `ceiling_met` below proves nothing about
    # whether the documented fields actually reached the DSP.
    assert limit_report["max_gain_reduction_db"] != 0.0, (
        "the forced plateau did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_met below is not evidence the documented fields are wired up; fix the fixture, not the assertion"
    )
    expected_reduction_db = _FORCED_CEILING_DBTP - input_peak_dbtp  # exact arithmetic: -6.0 dB
    assert abs(limit_report["max_gain_reduction_db"] - expected_reduction_db) < _GAIN_REDUCTION_TOLERANCE_DB, (
        f"measured gain reduction {limit_report['max_gain_reduction_db']:.4f} dB strayed from the expected "
        f"{expected_reduction_db:.4f} dB by more than {_GAIN_REDUCTION_TOLERANCE_DB} dB"
    )
    assert report["stages"][0]["ceiling_met"] is True


def test_hand_written_limit_with_no_params_uses_the_contracts_release_default() -> None:
    """release_ms's documented default is 50.0 -- the engine used to assume
    100.0 for an omitted field, which no hand-written contract-conformant
    plan could ever have asked for."""
    x = _noise()
    with_default = _hand_written_plan([{"stage": "limit", "params": {}}])
    explicit_50 = _hand_written_plan([{"stage": "limit", "params": {"release_ms": 50.0}}])

    y_default, _ = _render(with_default, x)
    y_explicit, _ = _render(explicit_50, x)
    assert np.array_equal(y_default, y_explicit)


# ---------------------------------------------------------------------------
# Whole-chain smoke test: every field name in one hand-written plan at once,
# for every stage this task touched.
# ---------------------------------------------------------------------------

# --- Regression guard: `ceiling_met` must not be vacuous -------------------
#
# Plain `_noise(seconds=2.0)` (std 0.1) never comes close to a -1.0 dBTP
# ceiling by the time it reaches the end of this chain, so
# `ceiling_met is True` would pass with the limiter never touching the
# signal -- see the agent report and test_dsp_limiter.py's own
# regression-guard comment. Unlike `lib.render`'s CLI-driven tests (which
# round-trip through an actual WAV file and so must stay under the file
# format's +-1.0 clip ceiling), this test drives `engine.apply_plan`
# directly on an in-memory array with no such cap, so a much smaller
# multiplier on a brief spike is enough to force real engagement.
_NOISE_HOT_SPIKE_SAMPLES = 50  # ~1ms @ 48kHz
_NOISE_HOT_SPIKE_MULTIPLIER = 10.0
# Measured on this exact chain (see agent report): multipliers below ~4
# leave the limiter fully quiet (max_gain_reduction_db == 0.0); 5-20
# reliably engage it (multiple dB of real reduction) while
# `ceiling_met` stays True (output settles ~0.03-0.05 dB under the
# ceiling, within limiter.py's own declared 0.05 dB numerical tolerance).
# 10.0 sits in the middle of that band.


def test_a_single_hand_written_plan_covering_eq_compress_saturate_loudness_limit_renders() -> None:
    plan = _hand_written_plan(
        [
            {
                "stage": "eq",
                "params": {
                    "hpf_hz": 40.0,
                    "peaks": [{"freq_hz": 3200.0, "gain_db": -2.5, "q": 1.4}],
                    "shelves": [{"type": "high", "freq_hz": 10000.0, "gain_db": 1.5, "q": 0.7}],
                },
            },
            {
                "stage": "compress",
                "params": {
                    "crossovers_hz": [120.0, 900.0, 5500.0],
                    "bands": [
                        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
                        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
                        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
                        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
                    ],
                },
            },
            {"stage": "saturate", "params": {"drive": 1.0, "mix": 1.0}},
            {"stage": "loudness", "params": {"target_lufs": -14.0}},
            {"stage": "limit", "params": {"ceiling_dbtp": -1.0}},
        ]
    )
    x = _noise(seconds=2.0)
    hot = slice(SR // 4, SR // 4 + _NOISE_HOT_SPIKE_SAMPLES)
    x[hot] *= _NOISE_HOT_SPIKE_MULTIPLIER
    y, report = _render(plan, x)
    assert np.all(np.isfinite(y))
    assert [s["stage"] for s in report["stages"]] == ["eq", "compress", "saturate", "loudness", "limit"]

    limit_report = report["stages"][-1]
    print(f"\n[contract] whole-chain limit stage: {limit_report}")
    # The canary: without this, `ceiling_met` below proves nothing -- it
    # would be equally true of a limiter that never touched the signal.
    assert limit_report["max_gain_reduction_db"] != 0.0, (
        "the hot noise fixture did not exercise the limiter (max_gain_reduction_db == 0.0) -- "
        "ceiling_met below is not evidence the whole chain works; fix the fixture, not the assertion"
    )
    assert report["stages"][-1]["ceiling_met"] is True
