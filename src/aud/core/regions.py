"""The regions document -- contracts/regions.v1.md.

A regions document says *where things are* in a file: what every `aud
detect` verb emits and what `aud cut` consumes. It is the counterpart of
`aud.plan`'s mastering plan, versioned by its own `regions_format` integer.

This module owns the whole contract: the pydantic container, construction
from detector output (`new_regions`), and parsing/validation of a document
handed in from outside (`read_regions`). Every failure mode in
contracts/regions.v1.md's "Failure codes" table is raised here as an
`AudError` with that table's exact code -- nothing upstream re-derives
these rules.

A note on validation strategy: unlike `aud.plan.Stage`, a region's set of
*valid* fields depends on the document's `kind` (silence/transient/filler),
and "a field is missing" must be rejected rather than defaulted (every
field is a measurement -- see the contract's Region-entries section). That
per-kind dynamism does not fit a single declarative pydantic schema, so
region-level and detection-level validation is written out by hand here,
the same way `aud.plan.append` hand-validates a stage name against
`STAGE_ORDER` rather than leaning on a pydantic validator.

One place where the contract is deliberately under-specified and this
module makes a documented choice: the failure-code table has four region-
level buckets -- `bad_regions` (something required is missing, at any
level), `unknown_region_field` (something present is not recognised, at
any level), `bad_region_field` (something present has the wrong type, is
non-finite, or is out of range), and `regions_out_of_order` (ordering/
overlap). That is the split this module applies; it is not spelled out
field-by-field in the contract itself.

The other thing this module deliberately does NOT check: that `end_s`
stays within `source`'s actual duration. The contract states that
constraint on the *region* (`regions.v1.md`'s Region-entries table), but
`new_regions`/`read_regions` never open `source` and have no duration to
check against -- and `regions_source_mismatch` is explicitly documented as
`cut`'s failure to raise, at render time, against the file it is actually
handed. Detectors (`aud.dsp.detect`) only ever report positions found
inside the array they were given, which is enough to make this a
non-issue in practice without inventing a parameter this function's
documented signature does not have.
"""

from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from pydantic import BaseModel, ConfigDict

from aud.schemas import AudError

__all__ = ["REGIONS_FORMAT", "Region", "RegionsDoc", "new_regions", "read_regions", "write_regions"]

REGIONS_FORMAT = 1

_KINDS: tuple[str, ...] = ("silence", "transient", "filler")

# Per-kind fields beyond the universal start_s/end_s, and the detection keys
# that kind requires. Both are exact sets: nothing more, nothing less --
# except for detection, which also has an optional extension point; see
# `_OPTIONAL_DETECTION_KEYS` below.
_REGION_FIELDS: dict[str, tuple[str, ...]] = {
    "silence": ("peak_dbfs", "rms_dbfs"),
    "transient": ("strength",),
    "filler": ("text", "confidence"),
}
_DETECTION_KEYS: dict[str, tuple[str, ...]] = {
    "silence": ("threshold_above_floor_db", "min_len_ms", "noise_floor_dbfs"),
    "transient": ("sensitivity", "min_gap_ms"),
    "filler": ("words", "min_pause_ms", "engine", "model"),
}

# Per-kind detection keys that MAY be present but are not required -- an
# additive, backward-compatible extension point (contracts/regions.v1.md's
# "Versioning" section: a new detection key stays in regions_format 1
# because no document written before the change contains it). Kept
# separate from `_DETECTION_KEYS` so a document produced before this key
# existed still validates: only `_DETECTION_KEYS` drives the "missing
# required key" check.
_OPTIONAL_DETECTION_KEYS: dict[str, tuple[str, ...]] = {
    "filler": ("degenerate_words_dropped",),
}

_COMMON_REGION_FIELDS = ("start_s", "end_s")
_TOP_LEVEL_FIELDS = ("regions_format", "created_with", "source", "sample_rate", "kind", "detection", "regions")


class Region(BaseModel):
    """One region entry. `start_s`/`end_s` are universal; everything else
    is kind-specific and arrives through pydantic's `extra="allow"`, so
    `region.peak_dbfs`, `region.strength`, `region.text` etc. are plain
    attribute access once a document has been built or parsed -- which
    per-kind fields are actually *valid* is enforced by this module's
    validation functions, not by this model's shape.
    """

    model_config = ConfigDict(extra="allow")

    start_s: float
    end_s: float


class RegionsDoc(BaseModel):
    """A whole regions document. See the module docstring and
    contracts/regions.v1.md for what every field means and promises.
    """

    model_config = ConfigDict(extra="forbid")

    regions_format: int = REGIONS_FORMAT
    created_with: str
    source: str
    sample_rate: int
    kind: str
    detection: dict[str, Any]
    regions: list[Region]


def _tool_version() -> str:
    try:
        return _pkg_version("aud")
    except PackageNotFoundError:
        return "0.0.0"


def _err(code: str, message: str, remedy: str) -> AudError:
    return AudError(code=code, message=message, remedy=remedy)


def _finite_number(value: Any) -> float | None:
    """Return `value` as a float if it is a real, finite JSON number; else None.

    Rejects bool (a JSON `true`/`false` is not a number, even though Python's
    `bool` is an `int` subclass) and non-finite floats (NaN/inf are not valid
    JSON and must not silently round-trip through Python's `json` module).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Per-region validation
# ---------------------------------------------------------------------------


def _validate_region_field_names(kind: str, index: int, raw: dict[str, Any]) -> None:
    allowed = set(_COMMON_REGION_FIELDS) | set(_REGION_FIELDS[kind])
    present = set(raw.keys())
    unknown = sorted(present - allowed)
    if unknown:
        raise _err(
            "unknown_region_field",
            f"regions[{index}] has field(s) not valid for kind '{kind}': {', '.join(unknown)}.",
            f"For kind '{kind}', a region may only carry: {', '.join(sorted(allowed))}.",
        )
    missing = sorted(allowed - present)
    if missing:
        raise _err(
            "bad_regions",
            f"regions[{index}] (kind '{kind}') is missing required field(s): {', '.join(missing)}.",
            f"For kind '{kind}', every region needs: {', '.join(sorted(allowed))}.",
        )


def _require_number(raw: dict[str, Any], field: str, index: int) -> float:
    number = _finite_number(raw.get(field))
    if number is None:
        raise _err(
            "bad_region_field",
            f"regions[{index}].{field} must be a finite number, got {raw.get(field)!r}.",
            f"Provide a finite numeric value for {field}.",
        )
    return number


def _build_region(kind: str, index: int, raw: dict[str, Any]) -> Region:
    """Validate one raw region dict against `kind`'s rules and build a `Region`.

    Raises AudError (`bad_region_field`) for any type/range/constraint
    violation. Field presence has already been checked by
    `_validate_region_field_names` before this runs.
    """
    start_s = _require_number(raw, "start_s", index)
    end_s = _require_number(raw, "end_s", index)

    if start_s < 0:
        raise _err(
            "bad_region_field",
            f"regions[{index}].start_s must be >= 0, got {start_s}.",
            "Use a non-negative offset in seconds.",
        )
    if end_s < start_s:
        raise _err(
            "bad_region_field",
            f"regions[{index}].end_s ({end_s}) must be >= start_s ({start_s}).",
            "end_s must not be before start_s.",
        )

    if kind == "transient":
        if end_s != start_s:
            raise _err(
                "bad_region_field",
                f"regions[{index}] is a transient; end_s must equal start_s, got start_s={start_s}, end_s={end_s}.",
                "Transients are zero-length positions: set end_s == start_s.",
            )
        strength = _require_number(raw, "strength", index)
        if strength <= 0:
            raise _err(
                "bad_region_field",
                f"regions[{index}].strength must be > 0, got {strength}.",
                "Report only onsets with a positive detection-function value.",
            )
        return Region(start_s=start_s, end_s=end_s, strength=strength)

    # silence and filler both require end_s > start_s -- a zero-length span
    # is not a thing that can be found for either kind.
    if end_s <= start_s:
        raise _err(
            "bad_region_field",
            f"regions[{index}] (kind '{kind}') must have end_s > start_s, got start_s={start_s}, end_s={end_s}.",
            "A silence or filler region must have positive duration.",
        )

    if kind == "silence":
        peak_dbfs = _require_number(raw, "peak_dbfs", index)
        rms_dbfs = _require_number(raw, "rms_dbfs", index)
        if peak_dbfs > 0:
            raise _err(
                "bad_region_field",
                f"regions[{index}].peak_dbfs must be <= 0, got {peak_dbfs}.",
                "peak_dbfs is measured relative to full scale and cannot exceed 0.",
            )
        if rms_dbfs > peak_dbfs:
            raise _err(
                "bad_region_field",
                f"regions[{index}].rms_dbfs ({rms_dbfs}) must be <= peak_dbfs ({peak_dbfs}).",
                "RMS level cannot exceed peak level.",
            )
        return Region(start_s=start_s, end_s=end_s, peak_dbfs=peak_dbfs, rms_dbfs=rms_dbfs)

    # kind == "filler"
    text = raw.get("text")
    if not isinstance(text, str):
        raise _err(
            "bad_region_field",
            f"regions[{index}].text must be a string, got {text!r}.",
            "Use the recognised word, lowercased, or '' for a hesitation.",
        )
    confidence = _require_number(raw, "confidence", index)
    if not (0.0 <= confidence <= 1.0):
        raise _err(
            "bad_region_field",
            f"regions[{index}].confidence must be between 0.0 and 1.0, got {confidence}.",
            "Use a confidence in [0.0, 1.0]; use 1.0 for a measured hesitation.",
        )
    return Region(start_s=start_s, end_s=end_s, text=text, confidence=confidence)


def _validate_ordering(regions: list[Region]) -> None:
    for i in range(len(regions) - 1):
        if regions[i].end_s > regions[i + 1].start_s:
            raise _err(
                "regions_out_of_order",
                f"regions[{i}] (end_s={regions[i].end_s}) overlaps or comes after "
                f"regions[{i + 1}] (start_s={regions[i + 1].start_s}).",
                "Regions must be sorted ascending by start_s and must not overlap.",
            )


# ---------------------------------------------------------------------------
# Detection-object validation
# ---------------------------------------------------------------------------


def _validate_detection(kind: str, detection: dict[str, Any]) -> None:
    required = set(_DETECTION_KEYS[kind])
    optional = set(_OPTIONAL_DETECTION_KEYS.get(kind, ()))
    allowed = required | optional
    present = set(detection.keys())
    unknown = sorted(present - allowed)
    if unknown:
        raise _err(
            "unknown_region_field",
            f"detection has key(s) not valid for kind '{kind}': {', '.join(unknown)}.",
            f"For kind '{kind}', detection may only carry: {', '.join(sorted(allowed))}.",
        )
    missing = sorted(required - present)
    if missing:
        raise _err(
            "bad_regions",
            f"detection is missing required key(s) for kind '{kind}': {', '.join(missing)}.",
            f"For kind '{kind}', detection needs: {', '.join(sorted(required))}.",
        )
    if kind == "filler":
        words = detection.get("words")
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise _err(
                "bad_region_field",
                f"detection.words must be an array of strings, got {words!r}.",
                "Provide the filler vocabulary as a list of strings.",
            )
        for key in ("engine", "model"):
            if not isinstance(detection.get(key), str):
                raise _err(
                    "bad_region_field",
                    f"detection.{key} must be a string, got {detection.get(key)!r}.",
                    f"Provide {key} as a string.",
                )
        # Optional: absent on a document produced before this key existed
        # (see `_OPTIONAL_DETECTION_KEYS`'s docstring). If present, it must
        # be a real, non-negative count -- not merely present.
        if "degenerate_words_dropped" in detection:
            degenerate = detection["degenerate_words_dropped"]
            if isinstance(degenerate, bool) or not isinstance(degenerate, int) or degenerate < 0:
                raise _err(
                    "bad_region_field",
                    f"detection.degenerate_words_dropped must be a non-negative integer, got {degenerate!r}.",
                    "Provide degenerate_words_dropped as a non-negative integer count, or omit it.",
                )


def _validate(kind: str, sample_rate: int, detection: dict[str, Any], regions: list[Region]) -> None:
    if kind not in _KINDS:
        raise _err(
            "unknown_region_kind",
            f"'{kind}' is not a region kind.",
            f"Use one of: {', '.join(_KINDS)}.",
        )
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise _err(
            "bad_region_field",
            f"sample_rate must be a positive integer, got {sample_rate!r}.",
            "Provide the sample rate of source, in Hz, as a positive integer.",
        )
    _validate_detection(kind, detection)
    _validate_ordering(regions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def new_regions(
    kind: str,
    source: str,
    sample_rate: int,
    detection: dict[str, Any],
    regions: list[dict[str, Any]],
) -> RegionsDoc:
    """Build a fresh regions document from detector output.

    `regions` is a list of plain dicts (as `aud.dsp.detect`/`aud.dsp.speech`
    return them) -- the same shape `read_regions` accepts on input. Applies
    every validation `read_regions` applies, so what this tool emits is
    always something it would also accept back.

    Raises:
        AudError: see the module docstring's failure-code table for codes.
    """
    for index, raw in enumerate(regions):
        _validate_region_field_names(kind, index, dict(raw))
    built_regions = [_build_region(kind, index, dict(raw)) for index, raw in enumerate(regions)]
    detection_copy = dict(detection)
    _validate(kind, sample_rate, detection_copy, built_regions)
    return RegionsDoc(
        regions_format=REGIONS_FORMAT,
        created_with=f"aud/{_tool_version()}",
        source=source,
        sample_rate=sample_rate,
        kind=kind,
        detection=detection_copy,
        regions=built_regions,
    )


def write_regions(doc: RegionsDoc) -> str:
    """Serialise a regions document to its canonical JSON form."""
    return doc.model_dump_json()


def read_regions(text_or_none: str | None) -> RegionsDoc:
    """Parse and fully validate a regions document from JSON text.

    Unlike `aud.plan.read_plan`, empty/absent input is an error rather than
    a fresh empty document -- there is no such thing as "the regions we
    haven't found yet", and every regions document is bound to one
    already-measured `source` (see the module docstring).

    Raises:
        AudError: see the module docstring's failure-code table for codes.
    """
    if text_or_none is None or not text_or_none.strip():
        raise _err(
            "bad_regions",
            "No regions document provided.",
            "Pipe in a regions document produced by 'aud detect ...'.",
        )
    try:
        data = json.loads(text_or_none)
    except json.JSONDecodeError as exc:
        raise _err(
            "bad_regions",
            f"Regions document is not valid JSON: {exc}",
            "Pipe in a regions document produced by 'aud detect ...'.",
        ) from exc
    if not isinstance(data, dict):
        raise _err(
            "bad_regions",
            "Regions document must be a JSON object.",
            f"A regions document needs: {', '.join(_TOP_LEVEL_FIELDS)}.",
        )

    missing_top = [field for field in _TOP_LEVEL_FIELDS if field not in data]
    if missing_top:
        raise _err(
            "bad_regions",
            f"Regions document is missing required field(s): {', '.join(missing_top)}.",
            f"A regions document needs: {', '.join(_TOP_LEVEL_FIELDS)}.",
        )
    unknown_top = sorted(set(data.keys()) - set(_TOP_LEVEL_FIELDS))
    if unknown_top:
        raise _err(
            "unknown_region_field",
            f"Regions document has unknown top-level field(s): {', '.join(unknown_top)}.",
            f"Only these top-level fields are recognised: {', '.join(_TOP_LEVEL_FIELDS)}.",
        )

    regions_format = data["regions_format"]
    if isinstance(regions_format, bool) or not isinstance(regions_format, int):
        raise _err(
            "bad_region_field",
            f"regions_format must be an integer, got {regions_format!r}.",
            f"This build accepts regions_format {REGIONS_FORMAT}.",
        )
    if regions_format != REGIONS_FORMAT:
        raise _err(
            "regions_format_unsupported",
            f"regions_format {regions_format} is not supported by this build.",
            f"This build accepts regions_format {REGIONS_FORMAT}.",
        )

    created_with = data["created_with"]
    if not isinstance(created_with, str):
        raise _err(
            "bad_region_field",
            f"created_with must be a string, got {created_with!r}.",
            "Provide created_with as a string, e.g. 'aud/0.3.0'.",
        )
    source = data["source"]
    if not isinstance(source, str):
        raise _err(
            "bad_region_field",
            f"source must be a string, got {source!r}.",
            "Provide source as the path the detector was run against.",
        )
    kind = data["kind"]
    if not isinstance(kind, str) or kind not in _KINDS:
        raise _err(
            "unknown_region_kind",
            f"'{kind}' is not a region kind.",
            f"Use one of: {', '.join(_KINDS)}.",
        )
    detection = data["detection"]
    if not isinstance(detection, dict):
        raise _err(
            "bad_region_field",
            f"detection must be an object, got {detection!r}.",
            "Provide detection as an object with the keys documented for this kind.",
        )
    raw_regions = data["regions"]
    if not isinstance(raw_regions, list):
        raise _err(
            "bad_region_field",
            f"regions must be an array, got {raw_regions!r}.",
            "Provide regions as an array of region objects.",
        )

    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise _err(
                "bad_region_field",
                f"regions[{index}] must be an object, got {raw!r}.",
                "Each region must be a JSON object.",
            )
        _validate_region_field_names(kind, index, raw)

    built_regions = [_build_region(kind, index, raw) for index, raw in enumerate(raw_regions)]

    sample_rate = data["sample_rate"]
    _validate(kind, sample_rate, detection, built_regions)

    return RegionsDoc(
        regions_format=regions_format,
        created_with=created_with,
        source=source,
        sample_rate=sample_rate,
        kind=kind,
        detection=dict(detection),
        regions=built_regions,
    )
