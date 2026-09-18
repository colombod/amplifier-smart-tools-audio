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
