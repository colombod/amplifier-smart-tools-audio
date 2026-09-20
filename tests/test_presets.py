"""Tests for aud.presets and the `aud preset` verb.

Presets are the third mechanism (alongside the pipe chain and `master`)
serving "one shell command, not a conversation". These tests pin:

1. Every preset is built through the real `aud.lib` stage builders (so an
   invalid preset is structurally impossible -- it would fail exactly like
   a hand-built plan with a bad param would).
2. `aud preset --list` / `aud preset show NAME` behave as documented at
   the CLI, including the not-a-plan vs. is-a-plan output-shape split.
3. An unknown preset name is a named, actionable error, not a KeyError.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from aud.dsp import engine
from aud.plan import STAGE_ORDER, Plan, ordered
from aud.presets import PRESET_NAMES, build_preset, list_presets
from aud.schemas import AudError

SR = 48000


def _run(args: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _noise(seconds: float = 1.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=(int(seconds * sr), 2))


def _speech_like_with_real_noise_floor(seconds: float = 3.0, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Speech-shaped bursts over a continuous, real (not digital-silence) noise floor.

    A real noise floor -- not zeros -- is the whole point: `expand`'s
    threshold is relative to the *measured* floor
    (`aud.dsp.detect.measure_noise_floor`), so a signal with no floor at all
    (pure silence between bursts) would give the stage nothing honest to
    measure against. The floor here sits around -42 dBFS RMS (a plausible
    untreated-room level); the bursts sit tens of dB above it, the same
    shape `tests/test_dsp_detect.py` uses for its own planted-gap tests.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    floor = 0.008 * rng.standard_normal((n, 2))  # ~ -42 dBFS RMS bed, both channels
    burst_env = (np.sin(2 * np.pi * 0.6 * t) > 0.3).astype(np.float64)  # ~35% duty cycle
    burst = 0.3 * np.sin(2 * np.pi * 220.0 * t)[:, None] * burst_env[:, None]
    return floor + burst


def _find_stage(report: dict, stage: str) -> dict:
    for entry in report["stages"]:
        if entry["stage"] == stage:
            return entry
    raise AssertionError(f"stage {stage!r} not found in render report: {[s['stage'] for s in report['stages']]}")


# ---------------------------------------------------------------------------
# aud.presets -- library level
# ---------------------------------------------------------------------------


def test_preset_names_cover_all_four_documented_destinations() -> None:
    assert set(PRESET_NAMES) == {"podcast", "music-streaming", "broadcast", "voiceover"}


def test_list_presets_gives_every_name_a_nonempty_description() -> None:
    entries = list_presets()
    assert {entry["name"] for entry in entries} == set(PRESET_NAMES)
    for entry in entries:
        assert entry["description"].strip(), f"{entry['name']} has an empty description"


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_build_preset_returns_a_plan_using_only_canonical_stage_names(name: str) -> None:
    plan = build_preset(name)
    assert isinstance(plan, Plan)
    assert plan.stages, f"preset {name!r} built an empty chain"
    for stage in plan.stages:
        assert stage.stage in STAGE_ORDER, f"preset {name!r} produced a non-canonical stage {stage.stage!r}"


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_build_preset_ends_with_loudness_then_limit(name: str) -> None:
    """Every preset should reach its stated target AND guarantee the ceiling."""
    names_in_canonical_order = [stage.stage for stage in ordered(build_preset(name))]
    assert "loudness" in names_in_canonical_order
    assert "limit" in names_in_canonical_order
    assert names_in_canonical_order.index("loudness") < names_in_canonical_order.index("limit")


def test_build_preset_unknown_name_raises_unknown_preset() -> None:
    with pytest.raises(AudError) as exc_info:
        build_preset("not-a-real-preset")
    assert exc_info.value.code == "unknown_preset"
    assert "not-a-real-preset" in exc_info.value.message
    for name in PRESET_NAMES:
        assert name in exc_info.value.remedy


# ---------------------------------------------------------------------------
# gate/expand: which presets use them, which deliberately don't, and the
# real measured attenuation each one produces on material with a genuine
# noise floor -- not just that the plan contains the stage name on paper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["podcast", "voiceover"])
def test_spoken_word_presets_include_a_quiet_end_repair_stage(name: str) -> None:
    """podcast/voiceover must address room tone between phrases -- a cleanup
    preset that leaves it alone ships a floor the chain's own compression
    and loudness stages are about to lift (see the module docstring)."""
    stage_names = [stage.stage for stage in build_preset(name).stages]
    assert "expand" in stage_names, f"{name} must include a gate/expand stage; got {stage_names}"
    assert "gate" not in stage_names  # this preset's documented choice is the gentler device


@pytest.mark.parametrize("name", ["broadcast", "music-streaming"])
def test_non_spoken_word_presets_deliberately_omit_gate_and_expand(name: str) -> None:
    stage_names = [stage.stage for stage in build_preset(name).stages]
    assert "gate" not in stage_names
    assert "expand" not in stage_names


@pytest.mark.parametrize("name", ["podcast", "voiceover"])
def test_expand_measurably_attenuates_a_real_noise_floor_when_rendered(name: str) -> None:
    """Not just a plan containing 'expand' on paper -- rendered against
    material with a genuine (non-digital-silence) noise floor, the stage
    must report real, non-zero attenuation. Prints the measured numbers
    (max/avg attenuation, attenuated_pct, measured floor) for inspection.
    """
    x = _speech_like_with_real_noise_floor()
    plan = build_preset(name)
    _y, report = engine.apply_plan(x, SR, ordered(plan))
    expand_report = _find_stage(report, "expand")
    print(f"\n[preset:{name}] expand report: {expand_report}")
    assert expand_report["max_attenuation_db"] < -3.0, (
        f"{name}: expected a real, audible attenuation on a planted noise floor, got {expand_report}"
    )
    assert expand_report["attenuated_pct"] > 5.0, (
        f"{name}: expected the quiet portions of the file to be measurably attenuated, got {expand_report}"
    )
    assert expand_report["noise_floor_dbfs"] < -20.0  # sane measured floor, not a degenerate reading


@pytest.mark.parametrize("name", ["podcast", "voiceover"])
def test_spoken_word_preset_descriptions_name_the_quiet_end_stage(name: str) -> None:
    """The description is where a preset's choices are argued (like the
    loudness target already is) -- gate/expand must be named there too."""
    entries = {entry["name"]: entry["description"] for entry in list_presets()}
    assert "expansion" in entries[name].lower() or "expand" in entries[name].lower()


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_preset_actually_renders_end_to_end(name: str) -> None:
    """Not just a valid plan on paper -- it must apply cleanly to real samples."""
    x = _noise()
    plan = build_preset(name)
    y, report = engine.apply_plan(x, SR, ordered(plan))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    stage_names = [s["stage"] for s in report["stages"]]
    assert stage_names, f"preset {name!r} produced no stage reports"
    limit_report = report["stages"][-1]
    assert limit_report["stage"] == "limit"
    assert limit_report["output_true_peak_dbtp"] <= -1.0 + 0.1


# ---------------------------------------------------------------------------
# `aud preset` -- CLI level
# ---------------------------------------------------------------------------


def test_cli_preset_list_is_a_wrapped_result_not_a_plan() -> None:
    proc = _run(["preset", "--list"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert set(payload) == {"result"}
    names = {entry["name"] for entry in payload["result"]}
    assert names == set(PRESET_NAMES)


def test_cli_preset_show_prints_a_raw_unwrapped_plan_document() -> None:
    proc = _run(["preset", "show", "podcast"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    # Raw plan document, like every other stage verb's stdout -- not
    # {"result": ...}.
    assert "result" not in payload
    assert "stages" in payload
    assert payload["plan_format"] == 1
    assert any(stage["stage"] == "limit" for stage in payload["stages"])


def test_cli_preset_show_unknown_name_is_a_named_error_not_a_crash() -> None:
    proc = _run(["preset", "show", "not-a-real-preset"])
    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "unknown_preset"
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr


def test_cli_preset_show_pipes_straight_into_render(tmp_path) -> None:
    """The exact worked example from the verb's own --help."""
    import soundfile as sf

    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(str(in_wav), _noise(seconds=1.0), SR, subtype="PCM_16")

    show_proc = _run(["preset", "show", "podcast"])
    assert show_proc.returncode == 0, show_proc.stderr

    render_proc = _run(["render", str(in_wav), str(out_wav)], input_text=show_proc.stdout)
    assert render_proc.returncode == 0, render_proc.stderr
    payload = json.loads(render_proc.stdout)
    assert payload["result"]["out_path"] == str(out_wav.resolve())
    assert out_wav.exists()
