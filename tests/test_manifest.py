"""Tests for aud.core.manifest -- the packaged SMART_TOOL.md contract."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from aud.core.manifest import load_manifest, manifest_body

REPO_ROOT = Path(__file__).parent.parent
_SHELL_COMMAND_WORDS = {"pip", "pip3", "uv", "uvx", "apt", "apt-get", "brew", "npm", "yarn", "pnpm", "conda", "sudo"}


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def _smart_tool_json() -> dict:
    return json.loads((REPO_ROOT / "smart-tool.json").read_text(encoding="utf-8"))


def test_manifest_validates() -> None:
    manifest = load_manifest()
    assert manifest.smart_tool_format == 1


def test_manifest_name_matches_cli_argv() -> None:
    manifest = load_manifest()
    cli_argv = _smart_tool_json()["cli_argv"]
    assert manifest.name == cli_argv[0]


def test_manifest_version_matches_pyproject() -> None:
    manifest = load_manifest()
    assert manifest.version == _pyproject_version()


def test_manifest_name_pattern() -> None:
    manifest = load_manifest()
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", manifest.name)


def test_manifest_version_pattern() -> None:
    manifest = load_manifest()
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest.version)


def test_manifest_has_requirements() -> None:
    manifest = load_manifest()
    assert len(manifest.requires) >= 1


def test_requires_install_is_a_doc_path_or_url_not_a_shell_command() -> None:
    manifest = load_manifest()
    for requirement in manifest.requires:
        first_word = requirement.install.strip().split(maxsplit=1)[0].lower()
        assert first_word not in _SHELL_COMMAND_WORDS, (
            f"{requirement.name}.install looks like a shell command: {requirement.install!r}"
        )


def test_manifest_body_is_markdown_after_frontmatter() -> None:
    body = manifest_body()
    assert body.strip().startswith("# aud")
    assert "---" not in body.splitlines()[0]
