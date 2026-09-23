"""The mastering plan document -- aud's central contract.

Every stage verb appends one stage to a plan and writes the plan back out.
`render` is the only verb that touches samples: it applies the plan's stages
in canonical mastering order (`STAGE_ORDER`), not the order they were
appended in, and a plan is otherwise inert data -- it can be written to a
file, piped between processes, or archived next to the render it produced.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pydantic import BaseModel, ValidationError

from aud.schemas import AudError

PLAN_FORMAT = 1

# Canonical order: editing -> repair -> tone -> dynamics -> character -> loudness -> limiting.
#
# Editing is FIRST, and not by preference. `cut` carries absolute positions measured on the
# source timeline, so every stage that alters that timeline -- `stretch` included -- has to run
# after it. And every measurement downstream is a measurement of a duration: loudness targeted
# over material that is later removed describes a file that does not exist.
#
# `gate`/`expand` sit at the FRONT of the repair group, before `dereverb`/`deess`, and -- more
# importantly -- before `compress`. Compression lifts the noise floor it is handed: it raises
# quiet material toward its threshold along with everything else, which is backwards for a stage
# whose entire job is to remove what is quiet. Gating after compression would be gating a noise
# floor the chain itself had already raised. dereverb/deess are narrowband, surgical repairs that
# do not materially change the overall noise floor, so gate/expand precede them too -- the
# surgical repairs then work on material whose dead air is already under control.
#
# `downmix`/`resample` sit at the very END, immediately before encode.
# They are output-format concerns, not mastering decisions: every stage
# before them should see the programme's real channel layout and rate
# (a de-esser's sibilant band, a compressor's crossovers, and the
# limiter's oversampling ratio are all defined against the ACTUAL sample
# rate of the material being processed), so folding channels or changing
# rate any earlier would make every measurement upstream of that point
# describe a signal that no longer exists once written -- the same
# argument contracts/plan.v1.md#why-editing-is-first makes for why `cut`/
# `strip_silence` sit at the front, mirrored for the two stages that
# change the medium's shape rather than its timeline. `downmix` precedes
# `resample` only for efficiency (resampling fewer channels costs less);
# the two are mathematically commutative order-independent linear
# operations -- resample_poly's per-channel filter is identical across
# channels, so folding channels before or after it produces the same
# result to floating-point precision (see aud.dsp.channels.downmix and
# aud.dsp.resample.resample's own docstrings).
#
# This list is the executable form of contracts/plan.v1.md's canonical order. If they disagree,
# one of them is lying to a caller.
STAGE_ORDER: list[str] = [
    "cut",
    "strip_silence",
    "stretch",
    "pitch",
    "gate",
    "expand",
    "dereverb",
    "deess",
    "eq",
    "eq_match",
    "compress",
    "saturate",
    "reverb",
    "loudness",
    "limit",
    "downmix",
    "resample",
]

_STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}


class Stage(BaseModel):
    """One entry in a plan: a stage name and its validated parameters."""

    stage: str
    params: dict


class Plan(BaseModel):
    """A mastering plan: an ordered-by-append list of stages, replayable by `render`."""

    plan_format: int = PLAN_FORMAT
    created_with: str
    stages: list[Stage]


def _tool_version() -> str:
    try:
        return _pkg_version("aud")
    except PackageNotFoundError:
        return "0.0.0"


def new_plan() -> Plan:
    """An empty mastering plan, stamped with this tool's version."""
    return Plan(created_with=f"aud/{_tool_version()}", stages=[])


def append(plan: Plan, stage: str, params: dict) -> Plan:
    """Return a NEW plan with one stage appended. Never mutates `plan`.

    Raises:
        AudError: code "unknown_stage" if `stage` is not in STAGE_ORDER.
    """
    if stage not in _STAGE_INDEX:
        raise AudError(
            code="unknown_stage",
            message=f"'{stage}' is not a mastering stage.",
            remedy=f"Use one of: {', '.join(STAGE_ORDER)}.",
        )
    new_stages = [*plan.stages, Stage(stage=stage, params=dict(params))]
    return plan.model_copy(update={"stages": new_stages})


def ordered(plan: Plan) -> list[Stage]:
    """Stages in canonical mastering order, stable within the same stage name.

    Two stages of the same name (e.g. two `eq` calls) keep their relative
    insertion order, because Python's sort is stable and both share a key.
    """
    return sorted(plan.stages, key=lambda stage: _STAGE_INDEX.get(stage.stage, len(STAGE_ORDER)))


def write_plan(plan: Plan) -> str:
    """Serialise a plan to its canonical JSON form."""
    return plan.model_dump_json()


def read_plan(text_or_stdin: str | None) -> Plan:
    """Parse a plan from JSON text (typically piped stdin).

    Empty or absent input (`None` or a blank/whitespace-only string) returns
    a fresh `new_plan()` rather than failing -- running a stage verb with no
    upstream pipe still works.

    Raises:
        AudError: code "bad_plan" if the text is not valid JSON, or is valid
            JSON that does not match the plan shape.
    """
    if text_or_stdin is None or not text_or_stdin.strip():
        return new_plan()
    try:
        data = json.loads(text_or_stdin)
    except json.JSONDecodeError as exc:
        raise AudError(
            code="bad_plan",
            message=f"Plan is not valid JSON: {exc}",
            remedy="Pipe in a plan produced by 'aud plan' or a stage verb; do not hand-edit unless the result is valid JSON.",
        ) from exc
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        raise AudError(
            code="bad_plan",
            message=f"Plan JSON does not match the plan shape: {exc}",
            remedy="A plan needs plan_format, created_with, and stages: [{stage, params}, ...].",
        ) from exc
