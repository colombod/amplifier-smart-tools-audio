"""End-to-end tests for the aud CLI's output-envelope contract.

Run as real subprocesses -- this is what actually proves the CLI surface
behaves as documented, not just that the library functions it calls do.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from aud.cli import registered_verbs
from aud.core.skill import CAPABILITIES


def _run(args: list[str], input_text: str = "", env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_manifest_emits_a_parseable_result_envelope() -> None:
    proc = _run(["manifest"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert set(payload) == {"result"}
    assert payload["result"]["name"] == "aud"


def test_bad_argument_emits_error_envelope_on_stdout_only() -> None:
    proc = _run(["limit", "--ceiling", "not-a-number"])
    assert proc.returncode in (1, 2)
    # The whole of stdout must be one JSON document -- nothing non-JSON alongside it.
    payload = json.loads(proc.stdout)
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "remedy"}
    assert payload["error"]["code"]
    assert payload["error"]["remedy"]


def test_missing_required_argument_is_also_a_clean_error_envelope() -> None:
    proc = _run(["compress"])  # --bands is required
    assert proc.returncode in (1, 2)
    payload = json.loads(proc.stdout)
    assert set(payload) == {"error"}


def test_unknown_verb_is_a_clean_error_envelope() -> None:
    proc = _run(["not-a-real-verb"])
    assert proc.returncode in (1, 2)
    payload = json.loads(proc.stdout)
    assert set(payload) == {"error"}


def test_plan_pipeline_yields_both_stages() -> None:
    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr

    eq_proc = _run(["eq", "--hpf", "40"], input_text=plan_proc.stdout)
    assert eq_proc.returncode == 0, eq_proc.stderr

    limit_proc = _run(["limit", "--ceiling", "-1.0"], input_text=eq_proc.stdout)
    assert limit_proc.returncode == 0, limit_proc.stderr

    plan = json.loads(limit_proc.stdout)
    stage_names = [stage["stage"] for stage in plan["stages"]]
    assert "eq" in stage_names
    assert "limit" in stage_names


def test_top_level_help_prints_the_skill_text() -> None:
    proc = _run(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert '<skill_content name="aud">' in proc.stdout


def test_verb_help_prints_full_prose_not_terse_usage() -> None:
    proc = _run(["eq", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "eq -- tone stage: parametric equalization" in proc.stdout


def test_verb_short_h_prints_terse_argparse_usage() -> None:
    proc = _run(["eq", "-h"])
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()


# --- DEFECT 1: CLI surface vs. aud.core.skill.CAPABILITIES must not drift ---


def test_cli_surface_matches_capabilities_manifest() -> None:
    """Every capability CAPABILITIES declares must have a real CLI route,
    and vice versa. 'analyze' going unreachable from the CLI while still
    being documented in --help is exactly the drift this guards against.
    """
    cli_verbs = registered_verbs()
    capability_verbs = {verb for verb, _, _ in CAPABILITIES}
    assert cli_verbs == capability_verbs


def test_analyze_verb_is_registered_and_reachable(tiny_wav: Path) -> None:
    proc = _run(["analyze", str(tiny_wav)])
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)["result"]
    assert result["sample_rate"] == 44100
    assert isinstance(result["integrated_lufs"], float)


# --- DEFECT 3: an unexpected exception must never leak a raw traceback ---


def test_missing_input_file_is_a_file_not_found_error_not_an_internal_bug(tmp_path: Path) -> None:
    """A missing/unreadable file is wrong INPUT, not an aud bug -- see AGENTS.md #4.

    Regression guard: this used to come back as {"code": "internal_error",
    "remedy": "...not something wrong with your input..."} for exactly the
    case where the input IS what is wrong.
    """
    missing = tmp_path / "does-not-exist.wav"
    proc = _run(["verify", str(missing), "--target", "-14"])
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == "file_not_found"
    assert str(missing) in payload["error"]["message"]
    assert str(missing) in payload["error"]["remedy"]
    # A handled, named user error -- not an internal bug -- so no traceback either.
    assert "Traceback" not in proc.stderr


def test_eq_match_missing_curve_file_is_a_file_not_found_error(tmp_path: Path) -> None:
    """The inverted case: a missing --curve file is the caller's input error."""
    missing_curve = tmp_path / "does-not-exist.json"
    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr
    proc = _run(["eq-match", "--curve", str(missing_curve)], input_text=plan_proc.stdout)
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "file_not_found"
    assert str(missing_curve) in payload["error"]["message"]


def test_unexpected_exception_is_wrapped_in_error_envelope_not_a_raw_traceback(tiny_wav: Path, tmp_path: Path) -> None:
    """A genuinely unforeseen exception -- not a named, mapped condition --
    still gets the internal_error catch-all treatment. Using an out_path
    whose parent path component is itself a regular file forces the
    os.mkdir call inside io.write_audio to raise NotADirectoryError, which
    nothing maps -- exactly the class of bug this catch-all exists for.
    """
    blocking_file = tmp_path / "blocking.wav"
    blocking_file.write_bytes(b"not a real wav, just needs to exist as a file")
    bogus_out = blocking_file / "sub" / "out.wav"

    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr
    proc = _run(["render", str(tiny_wav), str(bogus_out)], input_text=plan_proc.stdout)
    assert proc.returncode != 0
    # The whole of stdout must still be exactly one JSON document -- a raw
    # Python traceback on stdout would fail this parse outright.
    payload = json.loads(proc.stdout)
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == "internal_error"
    assert "remedy" in payload["error"]
    assert payload["error"]["remedy"]
    # Without --debug/AUD_DEBUG, the raw traceback is not dumped to stderr either.
    assert "Traceback" not in proc.stderr


def test_debug_flag_puts_the_real_traceback_on_stderr(tiny_wav: Path, tmp_path: Path) -> None:
    blocking_file = tmp_path / "blocking.wav"
    blocking_file.write_bytes(b"not a real wav, just needs to exist as a file")
    bogus_out = blocking_file / "sub" / "out.wav"

    plan_proc = _run(["plan"])
    assert plan_proc.returncode == 0, plan_proc.stderr
    proc = _run(
        ["render", str(tiny_wav), str(bogus_out), "--debug"],
        input_text=plan_proc.stdout,
    )
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "internal_error"
    assert "Traceback" in proc.stderr
    assert "NotADirectoryError" in proc.stderr
