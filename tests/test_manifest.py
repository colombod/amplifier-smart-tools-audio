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


def test_manifest_description_does_not_overclaim_every_verb_appends_to_a_plan() -> None:
    """Regression guard for the spec-adherence finding: analyze, detect, verify,
    check, config, manifest and plan itself do not append to a plan -- only
    chain stages do (src/aud/cli.py registers them as distinct non-stage verbs).
    """
    manifest = load_manifest()
    lowered = manifest.description.lower()
    assert "every verb appends to a plan" not in lowered
    assert "chain stage" in lowered, "description should scope the append claim to chain stages"


def test_ai_provider_requirement_flags_the_azure_endpoint_condition() -> None:
    """AzureOpenAIBackend (src/aud/intelligence/interface.py) rejects an Azure
    key with no AZURE_OPENAI_ENDPOINT set -- the manifest must say so, not
    imply the key alone is sufficient like the other four providers.
    """
    manifest = load_manifest()
    ai_provider = next(r for r in manifest.requires if r.name == "ai-provider")
    assert "AZURE_OPENAI_ENDPOINT" in ai_provider.purpose or any(
        r.name == "azure-openai-endpoint" for r in manifest.requires
    )


def test_azure_openai_endpoint_is_a_declared_conditional_requirement() -> None:
    manifest = load_manifest()
    azure_endpoint = next((r for r in manifest.requires if r.name == "azure-openai-endpoint"), None)
    assert azure_endpoint is not None, "expected a conditional 'azure-openai-endpoint' requirement"
    assert "AZURE_OPENAI_API_KEY" in azure_endpoint.purpose
    assert "AZURE_OPENAI_ENDPOINT" in azure_endpoint.purpose
    assert azure_endpoint.optional is True
    assert azure_endpoint.install == "docs/CONFIGURATION.md"
