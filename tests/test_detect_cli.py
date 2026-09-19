"""End-to-end CLI tests for `aud detect` -- run as real subprocesses.

Mirrors tests/test_cli_envelope.py's approach: this is what actually proves
the CLI surface behaves as documented, not just that the library functions
it calls do.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _run(args: list[str], input_text: str = "", env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_detect_silence_emits_a_raw_parseable_regions_document(tiny_wav: Path) -> None:
    proc = _run(["detect", "silence", str(tiny_wav)])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    # Raw regions document, NOT wrapped in {"result": ...} -- same treatment
    # a plan document gets from stage verbs.
    assert set(payload) == {
        "regions_format",
        "created_with",
        "source",
        "sample_rate",
        "kind",
        "detection",
        "regions",
    }
    assert payload["kind"] == "silence"
    assert payload["sample_rate"] == 44100
    assert "noise_floor_dbfs" in payload["detection"]
    assert isinstance(payload["regions"], list)


def test_detect_transients_emits_a_raw_parseable_regions_document(tiny_wav: Path) -> None:
    proc = _run(["detect", "transients", str(tiny_wav)])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["kind"] == "transient"
    for region in payload["regions"]:
        assert region["start_s"] == region["end_s"]


def test_detect_silence_missing_file_is_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.wav"
    proc = _run(["detect", "silence", str(missing)])
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "file_not_found"
    assert str(missing) in payload["error"]["message"]
    assert "Traceback" not in proc.stderr


def test_detect_fillers_on_this_host_gives_speech_extra_missing_never_a_traceback(tiny_wav: Path) -> None:
    proc = _run(["detect", "fillers", str(tiny_wav)])
    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "speech_extra_missing"
    assert "aud[speech]" in payload["error"]["remedy"]
    assert "Traceback" not in proc.stderr


def test_detect_silence_output_is_valid_input_for_cut(tiny_wav: Path) -> None:
    """`cut` is a sibling lane's own work-in-progress, built independently and
    concurrently with this one; its implementation status may vary. Either
    way, detect's raw-document output must be real, parseable stdin for the
    next verb in the pipe (contracts/regions.v1.md's worked example) -- so
    this only asserts that 'cut' never rejects it as a malformed regions
    document, whatever else it does or does not yet do.
    """
    detect_proc = _run(["detect", "silence", str(tiny_wav)])
    assert detect_proc.returncode == 0, detect_proc.stderr
    cut_proc = _run(["cut"], input_text=detect_proc.stdout)
    assert "Traceback" not in cut_proc.stderr
    payload = json.loads(cut_proc.stdout)
    if cut_proc.returncode != 0:
        regions_document_error_codes = {
            "bad_regions",
            "unknown_region_field",
            "bad_region_field",
            "regions_out_of_order",
            "unknown_region_kind",
            "regions_format_unsupported",
        }
        assert payload["error"]["code"] not in regions_document_error_codes, payload


def test_detect_custom_threshold_and_min_len_are_honored(tmp_path: Path) -> None:
    """Two gaps of different lengths, with enough total quiet material (a bit
    over a quarter of the file) that the noise-floor percentile estimate
    reflects genuine background rather than being skewed by loud content --
    see tests/test_dsp_detect.py's tolerance test for the same reasoning.
    """
    sr = 44100
    rng = np.random.default_rng(42)

    def burst(seconds: float) -> np.ndarray:
        n = int(seconds * sr)
        return 0.3 * np.sin(2 * np.pi * 300 * np.arange(n) / sr) + 0.02 * rng.standard_normal(n)

    short_gap = np.zeros(int(0.15 * sr))
    long_gap = np.zeros(int(0.5 * sr))
    x = np.concatenate([burst(0.6), short_gap, burst(0.6), long_gap, burst(0.6)])
    path = tmp_path / "gap.wav"
    sf.write(str(path), x, sr, subtype="PCM_24")

    # min_len above the short gap but below the long one -- only the long gap reported.
    proc_high = _run(["detect", "silence", str(path), "--min-len", "300"])
    assert proc_high.returncode == 0, proc_high.stderr
    assert len(json.loads(proc_high.stdout)["regions"]) == 1

    # min_len below both gaps -- both reported.
    proc_low = _run(["detect", "silence", str(path), "--min-len", "100"])
    assert proc_low.returncode == 0, proc_low.stderr
    assert len(json.loads(proc_low.stdout)["regions"]) == 2


def test_detect_deterministic_with_no_provider_and_no_path(scrubbed_env: dict[str, str], tiny_wav: Path) -> None:
    """Deterministic verbs run with zero AI-provider credentials -- AGENTS.md #3."""
    proc = _run(["detect", "silence", str(tiny_wav)], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["kind"] == "silence"


def test_detect_transients_deterministic_with_no_provider_and_no_path(
    scrubbed_env: dict[str, str], tiny_wav: Path
) -> None:
    proc = _run(["detect", "transients", str(tiny_wav)], env=scrubbed_env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["kind"] == "transient"


def test_detect_verb_registered_and_documented() -> None:
    proc = _run(["detect", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "detect -- find things in the audio" in proc.stdout
