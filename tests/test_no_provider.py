"""Proof of the tool's central claim: deterministic verbs run with zero
AI-provider credentials and no external binaries reachable on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


def _run(args: list[str], env: dict[str, str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_guard_the_guard_ffmpeg_is_genuinely_unreachable(scrubbed_env: dict[str, str]) -> None:
    """If this fails, the scrubbed environment is not actually scrubbed, and
    every other test in this file would be passing for the wrong reason.
    """
    assert shutil.which("ffmpeg", path=scrubbed_env["PATH"]) is None


def test_manifest_with_no_provider_and_no_path(scrubbed_env: dict[str, str]) -> None:
    proc = _run(["manifest"], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    assert "result" in json.loads(proc.stdout)


def test_check_with_no_provider_and_no_path(scrubbed_env: dict[str, str]) -> None:
    proc = _run(["check"], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)["result"]

    assert result["deterministic_capabilities_available"] is True
    assert result["ready"] is True

    ffmpeg_req = next(r for r in result["requirements"] if r["name"] == "ffmpeg")
    assert ffmpeg_req["state"] == "absent"

    provider_req = next(r for r in result["requirements"] if r["name"] == "ai-provider")
    assert provider_req["state"] == "absent"
    for var in PROVIDER_ENV_VARS:
        assert var not in scrubbed_env


def test_config_with_no_provider_and_no_path(scrubbed_env: dict[str, str]) -> None:
    proc = _run(["config"], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)["result"]
    for key in ("sample_rate_policy", "default_ceiling_dbtp", "default_target_lufs", "oversample", "output_subtype"):
        assert key in result
        assert "value" in result[key]
        assert "source" in result[key]


def test_plan_with_no_provider_and_no_path(scrubbed_env: dict[str, str]) -> None:
    proc = _run(["plan"], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["stages"] == []


def test_stage_verb_with_no_provider_and_no_path(scrubbed_env: dict[str, str]) -> None:
    plan_proc = _run(["plan"], env=scrubbed_env)
    assert plan_proc.returncode == 0, plan_proc.stderr

    eq_proc = _run(["eq", "--hpf", "40"], env=scrubbed_env, input_text=plan_proc.stdout)
    assert eq_proc.returncode == 0, eq_proc.stderr

    plan = json.loads(eq_proc.stdout)
    assert any(stage["stage"] == "eq" for stage in plan["stages"])
