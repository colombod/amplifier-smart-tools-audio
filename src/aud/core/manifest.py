"""Load and validate the packaged SMART_TOOL.md manifest.

The manifest is read from disk (not re-derived from any in-memory source) so
that what ships in the wheel is exactly what gets validated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from aud.schemas import AudError, Manifest

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "SMART_TOOL.md"
_FENCE = "---"


def _read_manifest_text() -> str:
    try:
        return _MANIFEST_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise AudError(
            code="manifest_missing",
            message=f"Could not read the manifest at {_MANIFEST_PATH}: {exc}",
            remedy="Reinstall aud -- SMART_TOOL.md ships inside the package next to this module.",
        ) from exc


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith(_FENCE):
        raise AudError(
            code="bad_manifest",
            message="SMART_TOOL.md must open with a '---' frontmatter fence.",
            remedy="Start the file with '---', a YAML block, then a closing '---'.",
        )
    parts = text.split(_FENCE, 2)
    if len(parts) < 3:
        raise AudError(
            code="bad_manifest",
            message="SMART_TOOL.md is missing its closing '---' frontmatter fence.",
            remedy="Add a closing '---' line after the YAML frontmatter block.",
        )
    _, frontmatter, body = parts
    return frontmatter, body.lstrip("\n")


def load_manifest() -> Manifest:
    """Parse and validate the packaged SMART_TOOL.md frontmatter.

    Raises:
        AudError: code "manifest_missing" if the file cannot be read, or
            "bad_manifest" if the frontmatter is malformed or fails schema
            validation.
    """
    frontmatter, _ = _split_frontmatter(_read_manifest_text())
    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise AudError(
            code="bad_manifest",
            message=f"SMART_TOOL.md frontmatter is not valid YAML: {exc}",
            remedy="Fix the YAML syntax in the frontmatter block.",
        ) from exc
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        raise AudError(
            code="bad_manifest",
            message=f"SMART_TOOL.md frontmatter failed validation: {exc}",
            remedy="Check smart_tool_format, name, version, description, use_cases, platforms and requires.",
        ) from exc


def manifest_body() -> str:
    """The markdown body of SMART_TOOL.md, after the frontmatter fences."""
    _, body = _split_frontmatter(_read_manifest_text())
    return body
