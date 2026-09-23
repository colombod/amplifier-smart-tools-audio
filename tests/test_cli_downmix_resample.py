"""CLI-level tests for the `downmix`/`resample` verbs: real subprocesses,
zero-credential proof, and the invalid-input envelope.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from aud.cli import registered_verbs
from aud.core.skill import CAPABILITIES

SR = 44100


def _run(args: list[str], input_text: str = "", env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_downmix_and_resample_are_registered_cli_verbs() -> None:
    verbs = registered_verbs()
    assert "downmix" in verbs
    assert "resample" in verbs


def test_downmix_and_resample_are_declared_capabilities() -> None:
    names = {verb for verb, _, _ in CAPABILITIES}
    assert "downmix" in names
    assert "resample" in names


def test_downmix_appends_a_stage_and_prints_the_plan() -> None:
    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr

    proc = _run(["downmix"], input_text=plan_proc.stdout)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert any(stage["stage"] == "downmix" for stage in plan["stages"])


def test_resample_requires_the_hz_flag() -> None:
    proc = _run(["resample"])
    assert proc.returncode == 2
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "usage_error"


def test_resample_appends_a_stage_with_the_requested_hz() -> None:
    plan_proc = _run(["plan"])
    proc = _run(["resample", "--hz", "48000"], input_text=plan_proc.stdout)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    stage = next(s for s in plan["stages"] if s["stage"] == "resample")
    assert stage["params"]["target_hz"] == 48000


def test_resample_with_a_non_positive_hz_is_a_named_bad_param() -> None:
    plan_proc = _run(["plan"])
    proc = _run(["resample", "--hz", "0"], input_text=plan_proc.stdout)
    assert proc.returncode == 1
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "bad_param"
    assert payload["error"]["remedy"]


def test_render_with_missing_input_file_is_a_named_file_not_found(tmp_path: Path) -> None:
    plan_proc = _run(["plan"])
    resample_proc = _run(["resample", "--hz", "48000"], input_text=plan_proc.stdout)
    missing = tmp_path / "does_not_exist.wav"
    out = tmp_path / "out.wav"
    proc = _run(["render", str(missing), str(out)], input_text=resample_proc.stdout)
    assert proc.returncode == 1
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "file_not_found"


def test_zero_credential_render_with_resample_and_downmix(tmp_path: Path, scrubbed_env: dict[str, str]) -> None:
    """The tool's central claim, for these two verbs specifically: no
    provider, no credential, no network -- proven via a genuinely scrubbed
    subprocess environment, not an in-process mock.
    """
    t = np.arange(SR) / SR
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    in_path = tmp_path / "in.wav"
    sf.write(str(in_path), np.stack([left, right], axis=1), SR, subtype="PCM_24")
    out_path = tmp_path / "out.wav"

    plan_proc = _run(["plan"], env=scrubbed_env)
    assert plan_proc.returncode == 0, plan_proc.stderr

    downmix_proc = _run(["downmix"], input_text=plan_proc.stdout, env=scrubbed_env)
    assert downmix_proc.returncode == 0, downmix_proc.stderr

    resample_proc = _run(["resample", "--hz", "16000"], input_text=downmix_proc.stdout, env=scrubbed_env)
    assert resample_proc.returncode == 0, resample_proc.stderr

    render_proc = _run(["render", str(in_path), str(out_path)], input_text=resample_proc.stdout, env=scrubbed_env)
    assert render_proc.returncode == 0, render_proc.stderr

    result = json.loads(render_proc.stdout)["result"]
    written, written_sr = sf.read(result["out_path"], always_2d=True)
    assert written_sr == 16000
    assert written.shape[1] == 1
