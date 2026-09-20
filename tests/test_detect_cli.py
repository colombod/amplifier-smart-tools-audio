"""End-to-end CLI tests for `aud detect` -- run as real subprocesses.

Mirrors tests/test_cli_envelope.py's approach: this is what actually proves
the CLI surface behaves as documented, not just that the library functions
it calls do.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aud.cli import _build_parser, _dispatch
from aud.dsp import speech as dsp_speech
from tests import replay


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
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "file_not_found"
    assert str(missing) in payload["error"]["message"]
    assert "Traceback" not in proc.stderr


def test_detect_fillers_on_this_host_gives_speech_extra_missing_never_a_traceback(tiny_wav: Path) -> None:
    proc = _run(["detect", "fillers", str(tiny_wav)])
    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "speech_extra_missing"
    assert "aud[speech]" in payload["error"]["remedy"]
    assert "Traceback" not in proc.stderr


def test_detect_fillers_missing_extra_fails_before_the_input_is_ever_decoded(tmp_path: Path) -> None:
    """Deviation-fix regression guard: the prerequisite check must fire

    BEFORE `path` is read, not after. Pointing the verb at a path that
    would fail on DECODE (not merely on lookup) if the code ever got that
    far, and still getting `speech_extra_missing` rather than a decode
    error, is what actually proves the ordering -- a plain missing-file
    test would pass even if the extra check ran second.
    """
    not_audio = tmp_path / "not-really-audio.wav"
    not_audio.write_bytes(b"this is not a wav file at all")
    proc = _run(["detect", "fillers", str(not_audio)])
    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "speech_extra_missing"
    assert payload["error"]["code"] not in {"file_not_found", "audio_decode_error", "internal_error"}
    assert "Traceback" not in proc.stderr


def test_detect_fillers_checks_the_speech_extra_before_touching_the_file(tmp_path: Path) -> None:
    """A missing prerequisite fails BEFORE the work: the extra-availability
    check must run before the input file is even opened, not after. Proven
    with a path that does not exist -- if the code decoded first, this
    would come back `file_not_found`; the fix means it never gets that far.
    """
    missing = tmp_path / "does-not-exist.wav"
    proc = _run(["detect", "fillers", str(missing)])
    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "speech_extra_missing"


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
    if cut_proc.returncode != 0:
        assert cut_proc.stdout == ""
        payload = json.loads(cut_proc.stderr)
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


# --- D3: the CLI's default filler vocabulary must not shadow the library's -


def test_detect_fillers_cli_words_defaults_to_none_not_a_second_vocabulary() -> None:
    """A caller who does not pass --words must get argparse's `None`, not a
    second, hard-coded vocabulary string -- that second copy is exactly what
    let 'um' (arguably the most common English filler) go undetectable by
    default (D3, lane report), because the CLI's copy always won.
    """
    parser = _build_parser()
    args = parser.parse_args(["detect", "fillers", "in.wav"])
    assert args.words is None


def test_detect_fillers_cli_default_vocabulary_is_speechs_filler_words(
    monkeypatch: pytest.MonkeyPatch, tiny_wav: Path
) -> None:
    """The effective vocabulary the CLI hands to the library, with no
    --words given, must be `None` -- so `aud.dsp.speech.FILLER_WORDS`, the
    one place that vocabulary is defined, is what actually runs.

    Proved end-to-end through the REAL `dsp.speech.detect_fillers`, with
    only faster-whisper itself replayed (a real recorded transcription,
    tests/replay.py) -- not by monkeypatching `lib.detect_fillers` and
    inventing its return value. `tiny_wav` drives the call, which is NOT
    the wav that produced `speech_short_16000__raw` -- declared explicitly
    via `source=replay.UNBOUND(...)` below, because this test only proves
    CLI -> lib -> dsp.speech vocabulary wiring, not correspondence between
    the replayed transcript and `tiny_wav`'s content.
    """
    replay.install_faster_whisper_replay(
        monkeypatch,
        "speech_short_16000__raw",
        source=replay.UNBOUND(
            "this test drives the call with `tiny_wav` (an arbitrary DSP fixture), not the "
            "recorded speech_short_16000 wav -- it proves the CLI's default vocabulary is wired "
            "through to dsp.speech.FILLER_WORDS, not that the replayed transcript corresponds to "
            "this audio"
        ),
    )

    parser = _build_parser()
    args: argparse.Namespace = parser.parse_args(["detect", "fillers", str(tiny_wav)])
    result = _dispatch("detect", args, None)

    assert result.detection["words"] == list(dsp_speech.FILLER_WORDS)


def test_detect_fillers_cli_explicit_words_still_override() -> None:
    """--words, when given, must still work -- only the *default* changed."""
    parser = _build_parser()
    args = parser.parse_args(["detect", "fillers", "in.wav", "--words", "like,basically"])
    assert args.words == "like,basically"
