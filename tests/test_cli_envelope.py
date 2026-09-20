"""End-to-end tests for the aud CLI's output-envelope contract.

Run as real subprocesses -- this is what actually proves the CLI surface
behaves as documented, not just that the library functions it calls do.
"""

from __future__ import annotations

import json
import os
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


def test_bad_argument_emits_error_envelope_on_stderr_only() -> None:
    proc = _run(["limit", "--ceiling", "not-a-number"])
    assert proc.returncode in (1, 2)
    # Stdout carries the result, stderr carries diagnostics (Amplifier Smart
    # Tools spec) -- a failing invocation has no result, so stdout must be
    # completely empty and the whole error envelope must be on stderr.
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "remedy"}
    assert payload["error"]["code"]
    assert payload["error"]["remedy"]


def test_missing_required_argument_is_also_a_clean_error_envelope() -> None:
    proc = _run(["compress"])  # --bands is required
    assert proc.returncode in (1, 2)
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}


def test_unknown_verb_is_a_clean_error_envelope() -> None:
    proc = _run(["not-a-real-verb"])
    assert proc.returncode in (1, 2)
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}


def test_failing_invocation_writes_nothing_to_stdout_and_the_envelope_to_stderr() -> None:
    """The direct regression guard for the stdout/stderr separation itself.

    A pipe chain (`aud detect silence x.wav | aud cut | aud render in out`)
    reads only stdout downstream of a failure; an error envelope leaking
    onto stdout would look like real data to that pipe instead of the loud,
    separate failure it needs to be (Amplifier Smart Tools spec: "stdout
    carries the result, stderr carries diagnostics").
    """
    proc = _run(["not-a-real-verb"])
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert proc.stderr != ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "remedy"}


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
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
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
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
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
    # Stdout must be completely empty -- a raw Python traceback (or anything
    # else) on stdout would fail this outright, and so would an error
    # envelope landing on the wrong stream.
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == "internal_error"
    assert "remedy" in payload["error"]
    assert payload["error"]["remedy"]
    # Without --debug/AUD_DEBUG, the raw traceback is not dumped either.
    assert "Traceback" not in proc.stderr


# --- write_audio failures: honest audio_write_error, not internal_error ---


def test_invalid_output_subtype_is_an_audio_write_error_not_internal_error(tiny_wav: Path, tmp_path: Path) -> None:
    """A bad `output_subtype` (here forced via AUD_OUTPUT_SUBTYPE) is a real,
    provokable `io.write_audio` failure. Before this fix, nothing in
    aud.lib caught the RuntimeError io.write_audio raises for it, so it
    fell through to the CLI's `internal_error` catch-all -- telling the
    caller to file a bug report about their own bad config value.
    """
    out_path = tmp_path / "out.wav"
    env = dict(os.environ)
    env["AUD_OUTPUT_SUBTYPE"] = "NOT_A_REAL_SUBTYPE"

    plan_proc = _run(["plan"], env=env)
    assert plan_proc.returncode == 0, plan_proc.stderr
    proc = _run(["render", str(tiny_wav), str(out_path)], input_text=plan_proc.stdout, env=env)

    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert set(payload) == {"error"}
    assert payload["error"]["code"] == "audio_write_error"
    assert payload["error"]["code"] != "internal_error"
    assert "remedy" in payload["error"]
    assert payload["error"]["remedy"]
    assert "Traceback" not in proc.stderr


def test_unwritable_destination_directory_is_an_audio_write_error(tiny_wav: Path, tmp_path: Path) -> None:
    """A destination directory this process cannot write to -- provoked
    directly with chmod, not simulated -- must map the same honest way.
    """
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir(mode=0o555)
    out_path = readonly_dir / "out.wav"
    try:
        plan_proc = _run(["plan"])
        assert plan_proc.returncode == 0, plan_proc.stderr
        proc = _run(["render", str(tiny_wav), str(out_path)], input_text=plan_proc.stdout)

        assert proc.returncode != 0
        assert proc.stdout == ""
        payload = json.loads(proc.stderr)
        assert set(payload) == {"error"}
        assert payload["error"]["code"] == "audio_write_error"
        assert payload["error"]["code"] != "internal_error"
        assert "remedy" in payload["error"]
        assert payload["error"]["remedy"]
    finally:
        readonly_dir.chmod(0o755)  # let tmp_path cleanup remove it


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
    assert proc.stdout == ""
    # With --debug, stderr carries BOTH the traceback and the error envelope
    # (in that order, see cli.py::main) -- the envelope is always the last
    # line, since `_print_error` is the last thing written.
    assert "Traceback" in proc.stderr
    assert "NotADirectoryError" in proc.stderr
    payload = json.loads(proc.stderr.strip().splitlines()[-1])
    assert payload["error"]["code"] == "internal_error"
