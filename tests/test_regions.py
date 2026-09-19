"""Unit tests for aud.core.regions -- contracts/regions.v1.md."""

from __future__ import annotations

import json

import pytest

from aud.core.regions import REGIONS_FORMAT, new_regions, read_regions, write_regions
from aud.schemas import AudError

_SILENCE_DETECTION = {"threshold_above_floor_db": 6.0, "min_len_ms": 400.0, "noise_floor_dbfs": -58.3}
_TRANSIENT_DETECTION = {"sensitivity": 1.0, "min_gap_ms": 50.0}
_FILLER_DETECTION = {"words": ["um", "uh"], "min_pause_ms": 700.0, "engine": "faster-whisper", "model": "base"}


def _silence_region(start: float = 12.48, end: float = 13.94) -> dict:
    return {"start_s": start, "end_s": end, "peak_dbfs": -54.1, "rms_dbfs": -57.8}


# --- round-trip ---------------------------------------------------------


def test_round_trip_is_lossless() -> None:
    doc = new_regions(
        kind="silence",
        source="in.wav",
        sample_rate=44100,
        detection=_SILENCE_DETECTION,
        regions=[_silence_region()],
    )
    text = write_regions(doc)
    reparsed = read_regions(text)
    assert write_regions(reparsed) == text
    assert reparsed.regions[0].start_s == 12.48
    assert reparsed.regions[0].end_s == 13.94
    assert reparsed.regions[0].peak_dbfs == -54.1
    assert reparsed.regions[0].rms_dbfs == -57.8
    assert reparsed.regions_format == REGIONS_FORMAT
    assert reparsed.kind == "silence"
    assert reparsed.detection == _SILENCE_DETECTION


def test_empty_regions_array_is_valid() -> None:
    doc = new_regions(kind="transient", source="in.wav", sample_rate=44100, detection=_TRANSIENT_DETECTION, regions=[])
    assert doc.regions == []
    reparsed = read_regions(write_regions(doc))
    assert reparsed.regions == []


# --- ordering / overlap --------------------------------------------------


def test_out_of_order_regions_are_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[_silence_region(5.0, 6.0), _silence_region(2.0, 3.0)],
        )
    assert exc_info.value.code == "regions_out_of_order"


def test_overlapping_regions_are_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[_silence_region(2.0, 5.0), _silence_region(4.0, 6.0)],
        )
    assert exc_info.value.code == "regions_out_of_order"


def test_touching_regions_are_accepted() -> None:
    """end_s(i) == start_s(i+1) is allowed -- the contract's constraint is <=, not <."""
    doc = new_regions(
        kind="silence",
        source="in.wav",
        sample_rate=44100,
        detection=_SILENCE_DETECTION,
        regions=[_silence_region(2.0, 3.0), _silence_region(3.0, 4.0)],
    )
    assert len(doc.regions) == 2


# --- kind-specific shape --------------------------------------------------


def test_transient_end_s_must_equal_start_s() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="transient",
            source="in.wav",
            sample_rate=44100,
            detection=_TRANSIENT_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 1.1, "strength": 0.5}],
        )
    assert exc_info.value.code == "bad_region_field"


def test_transient_zero_length_is_accepted() -> None:
    doc = new_regions(
        kind="transient",
        source="in.wav",
        sample_rate=44100,
        detection=_TRANSIENT_DETECTION,
        regions=[{"start_s": 1.0, "end_s": 1.0, "strength": 0.5}],
    )
    assert doc.regions[0].start_s == doc.regions[0].end_s == 1.0


def test_silence_region_missing_a_required_field_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 2.0, "peak_dbfs": -10.0}],  # rms_dbfs missing
        )
    assert exc_info.value.code == "bad_regions"


def test_silence_region_with_unknown_field_is_rejected() -> None:
    """A region carrying a field belonging to a different kind is unknown, not tolerated."""
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 2.0, "peak_dbfs": -10.0, "rms_dbfs": -20.0, "strength": 0.5}],
        )
    assert exc_info.value.code == "unknown_region_field"


def test_silence_end_s_must_exceed_start_s() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[{"start_s": 5.0, "end_s": 5.0, "peak_dbfs": -10.0, "rms_dbfs": -20.0}],
        )
    assert exc_info.value.code == "bad_region_field"


def test_filler_confidence_out_of_range_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="filler",
            source="in.wav",
            sample_rate=44100,
            detection=_FILLER_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 1.2, "text": "um", "confidence": 1.5}],
        )
    assert exc_info.value.code == "bad_region_field"


def test_peak_dbfs_above_zero_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 2.0, "peak_dbfs": 3.0, "rms_dbfs": -20.0}],
        )
    assert exc_info.value.code == "bad_region_field"


def test_rms_above_peak_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        new_regions(
            kind="silence",
            source="in.wav",
            sample_rate=44100,
            detection=_SILENCE_DETECTION,
            regions=[{"start_s": 1.0, "end_s": 2.0, "peak_dbfs": -20.0, "rms_dbfs": -10.0}],
        )
    assert exc_info.value.code == "bad_region_field"


# --- document-level validation --------------------------------------------


def test_unknown_top_level_field_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    raw["bogus_field"] = True
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "unknown_region_field"


def test_missing_top_level_field_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    del raw["sample_rate"]
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "bad_regions"


def test_unsupported_regions_format_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    raw["regions_format"] = 99
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "regions_format_unsupported"


def test_unknown_kind_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    raw["kind"] = "echo"
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "unknown_region_kind"


def test_detection_missing_a_required_key_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    del raw["detection"]["noise_floor_dbfs"]
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "bad_regions"


def test_detection_unknown_key_is_rejected() -> None:
    raw = json.loads(
        write_regions(
            new_regions(
                kind="silence",
                source="in.wav",
                sample_rate=44100,
                detection=_SILENCE_DETECTION,
                regions=[_silence_region()],
            )
        )
    )
    raw["detection"]["extra_key"] = 1
    with pytest.raises(AudError) as exc_info:
        read_regions(json.dumps(raw))
    assert exc_info.value.code == "unknown_region_field"


def test_not_json_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        read_regions("not json at all {")
    assert exc_info.value.code == "bad_regions"


def test_json_array_instead_of_object_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        read_regions("[]")
    assert exc_info.value.code == "bad_regions"


def test_empty_input_is_rejected() -> None:
    with pytest.raises(AudError) as exc_info:
        read_regions("")
    assert exc_info.value.code == "bad_regions"
    with pytest.raises(AudError):
        read_regions(None)


def test_created_with_is_stamped_with_the_tool_version() -> None:
    doc = new_regions(
        kind="silence", source="in.wav", sample_rate=44100, detection=_SILENCE_DETECTION, regions=[_silence_region()]
    )
    assert doc.created_with.startswith("aud/")


def test_every_error_carries_a_remedy() -> None:
    with pytest.raises(AudError) as exc_info:
        read_regions("not json")
    assert exc_info.value.remedy
    assert exc_info.value.message
