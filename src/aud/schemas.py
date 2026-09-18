"""Shared pydantic v2 data models and the one exception type aud raises.

This module has no internal aud dependencies -- everything else imports from
here, never the other way around.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ManifestRequirement(BaseModel):
    """One entry in the manifest's `requires` list."""

    name: str
    purpose: str
    install: str
    optional: bool = False


class Manifest(BaseModel):
    """The validated shape of SMART_TOOL.md's YAML frontmatter."""

    smart_tool_format: int
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str
    use_cases: list[str]
    platforms: list[str]
    requires: list[ManifestRequirement]


class AudError(Exception):
    """The one exception type aud raises for user-facing failures.

    Every AudError carries a machine-readable `code`, a human-readable
    `message`, and a `remedy` describing what a valid input looks like. The
    CLI renders these three fields verbatim into `{"error": {...}}`.
    """

    def __init__(self, code: str, message: str, remedy: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.remedy = remedy

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "remedy": self.remedy}


class NotImplementedStageError(ValueError):
    """Raised by aud.dsp.engine for a plan stage with no DSP implementation yet.

    A plan stage name can be valid (accepted by plan.py's STAGE_ORDER) while
    still having no DSP handler -- eq_match, deess, dereverb, reverb, stretch
    and pitch are all planned but not yet built (see dsp/engine.py). That is
    a known, named gap, not an internal bug, so it gets its own type rather
    than a bare ValueError: aud.lib catches this specific type and reports it
    as a `not_implemented` AudError instead of letting the CLI's catch-all
    wrap it as `internal_error` and tell the caller to file a bug about a
    gap the tool already knows about.

    Subclasses ValueError (rather than AudError) because `dsp/` modules take
    arrays and parameters and return arrays -- they do not raise user-facing
    errors (see AGENTS.md #8); AudError construction stays a job for
    aud.lib, above the dsp/ boundary. Subclassing ValueError also means a
    caller that only knows "engine dispatch failures are ValueErrors" (see
    dsp/engine.py's own docstring) keeps working unchanged.
    """

    def __init__(self, stage: str, implemented: tuple[str, ...]) -> None:
        self.stage = stage
        self.implemented = implemented
        super().__init__(
            f"Stage '{stage}' is not yet implemented in this DSP engine. Implemented stages: {', '.join(implemented)}."
        )
