"""Tests for `aud.lib.curve_extract` / `curve_apply`.

Three things these guard, all findings from the spec-adherence check:

1. The payload is data, not a reference: `curve_apply` must accept a parsed
   curve directly (`curve=`), not only a path to one -- a caller with no
   filesystem in common with `aud` can still supply a curve.
2. An artifact's location is named: `curve_path`/`out_path` in the result is
   always the ABSOLUTE path the file actually landed at, even when the
   caller passed a relative one.
3. A curve write is atomic and an unwritable destination is a named,
   actionable `AudError`, never a bare traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aud import lib
from aud.schemas import AudError


def test_curve_extract_returns_an_absolute_path_even_for_a_relative_one(
    tiny_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = lib.curve_extract(str(tiny_wav), "curve.json")
    assert result["curve_path"] == str((tmp_path / "curve.json").resolve())
    assert Path(result["curve_path"]).exists()


def test_curve_apply_accepts_parsed_curve_data_directly_no_file_needed(tiny_wav: Path, tmp_path: Path) -> None:
    """The library caller path: no curve FILE ever touches disk."""
    curve_data = [[100.0, -2.0], [1000.0, 0.0], [8000.0, 3.0]]
    out_path = tmp_path / "matched.wav"
    result = lib.curve_apply(str(tiny_wav), str(out_path), curve=curve_data)
    assert Path(result["out_path"]).exists()
    assert result["out_path"] == str(out_path.resolve())


def test_curve_apply_curve_path_convenience_still_works(tiny_wav: Path, tmp_path: Path) -> None:
    """The CLI's own convenience: a caller may still hand it a file path."""
    curve_file = tmp_path / "curve.json"
    extracted = lib.curve_extract(str(tiny_wav), str(curve_file))
    out_path = tmp_path / "matched.wav"
    result = lib.curve_apply(str(tiny_wav), str(out_path), curve_path=extracted["curve_path"])
    assert Path(result["out_path"]).exists()


def test_curve_apply_with_neither_curve_nor_curve_path_is_a_named_bad_param(tiny_wav: Path, tmp_path: Path) -> None:
    with pytest.raises(AudError) as excinfo:
        lib.curve_apply(str(tiny_wav), str(tmp_path / "out.wav"))
    assert excinfo.value.code == "bad_param"


def test_curve_extract_to_an_unwritable_destination_is_a_named_bad_path_error(tiny_wav: Path, tmp_path: Path) -> None:
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir(mode=0o555)
    out_path = readonly_dir / "curve.json"
    try:
        with pytest.raises(AudError) as excinfo:
            lib.curve_extract(str(tiny_wav), str(out_path))
        assert excinfo.value.code == "bad_path"
        assert str(out_path) in excinfo.value.message
        assert excinfo.value.remedy
        # No partial/temp file left behind in the (unwritable) destination.
        assert list(readonly_dir.iterdir()) == []
    finally:
        readonly_dir.chmod(0o755)


def test_curve_extract_write_is_atomic_no_partial_file_on_success(tiny_wav: Path, tmp_path: Path) -> None:
    """Sanity check the happy path uses the same atomic helper: no stray
    temp file left alongside the real one once the write has succeeded."""
    out_path = tmp_path / "curve.json"
    lib.curve_extract(str(tiny_wav), str(out_path))
    leftovers = [p for p in tmp_path.iterdir() if p.name != "curve.json" and p.name != "in.wav"]
    assert leftovers == [], f"unexpected leftover files: {leftovers}"
