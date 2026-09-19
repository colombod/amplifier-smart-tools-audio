"""End-to-end: `gate` and `expand` chained through the real CLI and rendered.

Mirrors tests/test_pipeline_all_stages.py's shape but for the two new
stages: plan | gate | expand | render, run as a real subprocess, with the
render report inspected for both stages' stats. Also proves STAGE_ORDER
(not append order) puts gate/expand ahead of compress, and pins the plan
builders' JSON shape the way tests/test_lib_compress.py pins compress's.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aud.lib import compress, expand, gate
from aud.plan import new_plan, ordered
from aud.schemas import AudError


def _run(args: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_gate_and_expand_chain_through_the_cli_and_render(tiny_wav: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.wav"

    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr

    gate_proc = _run(["gate", "--threshold", "6", "--range", "18"], input_text=plan_proc.stdout)
    assert gate_proc.returncode == 0, gate_proc.stderr

    expand_proc = _run(["expand", "--threshold", "9", "--ratio", "2"], input_text=gate_proc.stdout)
    assert expand_proc.returncode == 0, expand_proc.stderr

    plan = json.loads(expand_proc.stdout)
    stage_names = [s["stage"] for s in plan["stages"]]
    assert "gate" in stage_names
    assert "expand" in stage_names

    render_proc = _run(["render", str(tiny_wav), str(out_path)], input_text=expand_proc.stdout)
    assert render_proc.returncode == 0, render_proc.stderr
    render_result = json.loads(render_proc.stdout)["result"]
    assert render_result["out_path"] == str(out_path)
    assert out_path.exists()

    report_by_stage = {s["stage"]: s for s in render_result["report"]["stages"]}
    print(f"\n[chain] gate report:   {report_by_stage['gate']}")
    print(f"[chain] expand report: {report_by_stage['expand']}")
    assert "open_count" in report_by_stage["gate"]
    assert "max_attenuation_db" in report_by_stage["gate"]
    assert "ratio" in report_by_stage["expand"]
    assert "max_attenuation_db" in report_by_stage["expand"]


def test_gate_and_expand_render_before_compress_even_when_appended_after_it(tiny_wav: Path, tmp_path: Path) -> None:
    """STAGE_ORDER, not append order, governs rendering -- append compress
    FIRST, gate/expand AFTER, and prove canonical order still puts the two
    new repair stages ahead of compress."""
    plan = new_plan()
    plan = compress(plan, bands=[1000.0], ratio=2.0)
    plan = gate(plan, threshold_above_floor_db=12.0)
    plan = expand(plan, threshold_above_floor_db=6.0)

    append_order = [s.stage for s in plan.stages]
    render_order = [s.stage for s in ordered(plan)]
    print(f"\n[chain] append order: {append_order}")
    print(f"[chain] render order: {render_order}")
    assert append_order == ["compress", "gate", "expand"]
    assert render_order == ["gate", "expand", "compress"]


def test_gate_builder_emits_the_documented_plan_shape() -> None:
    plan = gate(new_plan(), threshold_above_floor_db=10.0, range_db=15.0, crossovers_hz=[500.0, 4000.0])
    stage = plan.stages[-1]
    assert stage.stage == "gate"
    assert stage.params["threshold_above_floor_db"] == 10.0
    assert stage.params["threshold_db"] is None
    assert stage.params["range_db"] == 15.0
    assert stage.params["crossovers_hz"] == [500.0, 4000.0]


def test_expand_builder_emits_the_documented_plan_shape() -> None:
    plan = expand(new_plan(), ratio=3.0, knee_db=4.0)
    stage = plan.stages[-1]
    assert stage.stage == "expand"
    assert stage.params["ratio"] == 3.0
    assert stage.params["knee_db"] == 4.0
    assert stage.params["crossovers_hz"] == []


def test_gate_builder_rejects_non_ascending_crossovers() -> None:
    with pytest.raises(AudError) as exc_info:
        gate(new_plan(), crossovers_hz=[2000.0, 500.0])
    assert exc_info.value.code == "bad_param"


def test_expand_builder_rejects_ratio_below_one() -> None:
    with pytest.raises(AudError) as exc_info:
        expand(new_plan(), ratio=0.5)
    assert exc_info.value.code == "bad_param"
