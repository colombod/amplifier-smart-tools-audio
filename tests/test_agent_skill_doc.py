"""Contract test for skills/aud/SKILL.md: the Agent Skill is thin.

Guards against a deviation found by the spec's `check-spec-adherence` review:
the skill independently described provider and data-flow behaviour for
`advise`/`master` (which verbs need a credential, and that only measurements
-- never raw audio -- are sent), duplicating what `aud --help` and
`aud advise --help` already say and can drift from. Per spec: "The skill
carries the manifest's name and description, the install commands, and the
instruction to run `--help` and follow it. Nothing more."

Same shape as the sibling `vid` repo's `tests/test_agent_skill_doc.py`: pin
the removed markers so they cannot silently creep back into the skill, and
separately prove each one is still reachable from runtime help -- nothing is
lost, only relocated.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SKILL_PATH = Path(__file__).parents[1] / "skills" / "aud" / "SKILL.md"

# Content that used to live in the skill and must now only be reachable from
# runtime help (`aud --help`, `aud advise --help`) instead.
REMOVED_MARKERS = (
    "Almost all of it needs no AI provider and no credential.",  # provider gating restated
    "even they send only the",  # data-flow restated: measurements, never the audio
)


def _frontmatter_and_body() -> tuple[dict, str]:
    text = SKILL_PATH.read_text(encoding="utf-8")
    _, _, rest = text.partition("---\n")
    frontmatter_text, _, body = rest.partition("\n---\n")
    return yaml.safe_load(frontmatter_text), body


def test_frontmatter_carries_name_and_description() -> None:
    frontmatter, _ = _frontmatter_and_body()

    assert frontmatter["name"] == "aud"
    assert frontmatter["description"]


def test_body_carries_install_commands() -> None:
    _, body = _frontmatter_and_body()

    assert "uv tool install" in body
    assert "uv add" in body


def test_body_instructs_running_help_and_following_it() -> None:
    _, body = _frontmatter_and_body()

    assert "aud --help" in body
    assert "aud <verb> --help" in body
    assert "aud check" in body


def test_body_does_not_duplicate_runtime_help_content() -> None:
    _, body = _frontmatter_and_body()

    for marker in REMOVED_MARKERS:
        assert marker not in body, f"skill duplicates runtime-help content: {marker!r}"


def test_removed_provider_gating_is_reachable_via_aud_help() -> None:
    """The provider-gating claim that used to be restated in the skill is
    the same information `aud --help` renders per-verb, from CAPABILITIES.
    """
    from aud import lib

    skill_text = lib.skill()

    assert "[deterministic]" in skill_text
    assert "[model-backed]" in skill_text
    assert "ANTHROPIC_API_KEY" in skill_text


def test_removed_measurements_only_claim_is_reachable_via_advise_help() -> None:
    """The "measurements only, never raw audio" claim that used to be
    restated in the skill is the same text `aud advise --help` prints,
    sourced through the library (aud.lib.verb_help), not the CLI directly.
    """
    from aud import lib

    advise_doc = lib.verb_help("advise")

    assert advise_doc is not None
    assert "measurements only, never raw audio" in advise_doc
