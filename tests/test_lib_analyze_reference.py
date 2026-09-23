"""`lib.analyze`/`lib.advise` wiring for the reference-anchored tonal signal.

The DSP-level statistics (6/6 new, 1/6 old control) live in
`test_reference_anchored_tonal.py`. This file only guards the public
plumbing: `reference_path` reaching `dsp.analysis.analyze` through
`lib.analyze`, and `lib.advise`'s `measurements` (the dict actually shown
to the model) carrying the field when a reference was supplied to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from aud import lib
from tests import replay


def _write_tone(path: Path, freq: float, sr: int = 44100, seconds: float = 1.5) -> Path:
    t = np.arange(int(sr * seconds)) / sr
    tone = 0.2 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), np.stack([tone, tone], axis=1), sr, subtype="PCM_24")
    return path


def test_lib_analyze_omits_rel_reference_db_without_a_reference(tiny_wav: Path) -> None:
    result = lib.analyze(str(tiny_wav))
    entries = result["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    for entry in entries:
        assert "rel_reference_db" not in entry


def test_lib_analyze_adds_rel_reference_db_with_a_reference(tiny_wav: Path, tmp_path: Path) -> None:
    reference = _write_tone(tmp_path / "reference.wav", 440.0)
    result = lib.analyze(str(tiny_wav), reference_path=str(reference))
    entries = result["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    assert all("rel_reference_db" in entry for entry in entries)


def test_lib_advise_measurements_carry_rel_reference_db_when_a_reference_is_given(
    tiny_wav: Path, tmp_path: Path
) -> None:
    """`measurements` -- not just `reference_measurements` -- must carry the
    new field: it is `measurements`'s `octave_band_analysis` that
    `intelligence/prompts.py`'s system prompt tells the model to read as
    the PRIMARY tonal signal."""
    reference = _write_tone(tmp_path / "reference.wav", 440.0)
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    outcome = lib.advise(
        str(tiny_wav),
        reference_path=str(reference),
        backend=backend,
        model="fake-1",
    )
    entries = outcome["measurements"]["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    assert all("rel_reference_db" in entry for entry in entries)
    # reference_measurements is unaffected -- it is analyze(reference_path)
    # alone, informational context only, never itself reference-anchored.
    assert outcome["reference_measurements"] is not None
    for entry in outcome["reference_measurements"]["octave_band_analysis"]:
        assert "rel_reference_db" not in entry


def test_lib_advise_measurements_omit_rel_reference_db_without_a_reference(tiny_wav: Path) -> None:
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    outcome = lib.advise(str(tiny_wav), backend=backend, model="fake-1")
    entries = outcome["measurements"]["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    for entry in entries:
        assert "rel_reference_db" not in entry
    assert outcome["reference_measurements"] is None
