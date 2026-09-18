"""End-to-end regression test for DEFECT 2: every implemented stage chained
together and actually rendered, not just eyeballed.

This is the literal repro from the bug report: plan | eq | compress |
saturate | loudness | limit | render, then verify. It must not crash, and
verify must report the render actually near the requested loudness and
under the requested ceiling.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_all_implemented_stages_chain_and_render_without_crashing(tiny_wav: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "out.wav"

    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr

    eq_proc = _run(["eq", "--hpf", "40", "--peak", "3000,-3,1.2"], input_text=plan_proc.stdout)
    assert eq_proc.returncode == 0, eq_proc.stderr

    compress_proc = _run(["compress", "--bands", "120,900,5500", "--ratio", "3"], input_text=eq_proc.stdout)
    assert compress_proc.returncode == 0, compress_proc.stderr

    saturate_proc = _run(["saturate", "--drive", "1.5", "--mix", "0.25"], input_text=compress_proc.stdout)
    assert saturate_proc.returncode == 0, saturate_proc.stderr

    loudness_proc = _run(["loudness", "--target", "-14"], input_text=saturate_proc.stdout)
    assert loudness_proc.returncode == 0, loudness_proc.stderr

    limit_proc = _run(["limit", "--ceiling", "-1.0"], input_text=loudness_proc.stdout)
    assert limit_proc.returncode == 0, limit_proc.stderr

    render_proc = _run(["render", str(tiny_wav), str(out_path)], input_text=limit_proc.stdout)
    assert render_proc.returncode == 0, render_proc.stderr
    render_result = json.loads(render_proc.stdout)["result"]
    assert render_result["out_path"] == str(out_path)
    stage_names = [s["stage"] for s in render_result["report"]["stages"]]
    assert stage_names == ["eq", "compress", "saturate", "loudness", "limit"]
    assert out_path.exists()

    verify_proc = _run(["verify", str(out_path), "--target", "-14", "--ceiling", "-1.0"])
    assert verify_proc.returncode == 0, verify_proc.stderr
    verify_result = json.loads(verify_proc.stdout)["result"]

    print(f"\n[pipeline] verify: {verify_result}")
    assert verify_result["lufs_ok"] is True, verify_result
    assert verify_result["ceiling_ok"] is True, verify_result
