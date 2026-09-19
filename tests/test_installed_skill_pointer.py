"""`skills/aud/SKILL.md` -- the file `npx skills add` installs -- must stay a
POINTER at the real documentation, never a copy of it.

The risk this guards against: it is easy, over time, for an installed skill
file to accumulate a full copy of `aud --help`'s verb-by-verb reference (or
drift into being one), at which point it is a second, unversioned source of
truth that silently goes stale the moment a verb changes. The guard is
structural -- frontmatter identity, and "does it point outward instead of
inlining the verb table" -- not a byte-diff against `--help`.

A note on a naive size check, honestly: at the time of writing this skill is
*larger* than `aud --help` (104 lines vs 83). That is not a red flag here --
`aud --help` (rendered by `aud.core.skill.render_skill`) is unusually terse,
one line per verb, because it targets an agent that will follow up with
`aud <verb> --help`. A skill file additionally needs installation
instructions, a capability/dependency table and worked examples that
`--help` deliberately omits. So this file does NOT assert "shorter than
--help" -- that ratio is the wrong signal for this tool -- it asserts the
actual structural property that matters: the skill does not inline the verb
reference table `SMART_TOOL.md` owns, and it tells the reader where the real
answers live.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aud.core.manifest import load_manifest

REPO_ROOT = Path(__file__).parent.parent
SKILL_PATH = REPO_ROOT / "skills" / "aud" / "SKILL.md"
_FENCE = "---"


def _load_skill_frontmatter_and_body() -> tuple[dict, str]:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith(_FENCE), "skills/aud/SKILL.md must open with a '---' frontmatter fence"
    parts = text.split(_FENCE, 2)
    assert len(parts) == 3, "skills/aud/SKILL.md must have a closing '---' frontmatter fence"
    _, frontmatter_text, body = parts
    frontmatter = yaml.safe_load(frontmatter_text)
    return frontmatter, body.lstrip("\n")


def test_skill_file_exists_exactly_once() -> None:
    assert SKILL_PATH.is_file()


def test_skill_frontmatter_has_the_required_fields() -> None:
    frontmatter, _ = _load_skill_frontmatter_and_body()

    assert frontmatter.get("name"), "SKILL.md frontmatter needs a non-empty 'name'"
    assert frontmatter.get("description"), "SKILL.md frontmatter needs a non-empty 'description'"
    assert frontmatter.get("license"), "SKILL.md frontmatter needs a non-empty 'license'"
    metadata = frontmatter.get("metadata")
    assert isinstance(metadata, dict), "SKILL.md frontmatter needs a 'metadata' mapping"
    assert metadata.get("repository"), "SKILL.md frontmatter needs a non-empty 'metadata.repository'"


def test_skill_name_matches_the_manifests_name() -> None:
    """The installed skill's identity must agree with SMART_TOOL.md's --
    two names for the same tool drifting apart is exactly the kind of
    silent inconsistency a pointer file is supposed to avoid.
    """
    frontmatter, _ = _load_skill_frontmatter_and_body()
    manifest = load_manifest()
    assert frontmatter["name"] == manifest.name


def test_skill_points_the_reader_at_help_and_check_instead_of_restating_them() -> None:
    _frontmatter, body = _load_skill_frontmatter_and_body()

    assert "aud --help" in body, "the skill must send the reader to `aud --help` for the real instructions"
    assert "aud <verb> --help" in body, "the skill must send the reader to `aud <verb> --help` for one verb's docs"
    assert "aud check" in body, "the skill must send the reader to `aud check` for what this host can do"


def test_skill_does_not_inline_the_verb_reference_table() -> None:
    """Structural check for "this became a copy", not a line-count ratio
    (see the module docstring for why a size comparison to `--help` is the
    wrong test for this particular tool).

    The verb reference table this must NOT contain lives in
    `src/aud/SMART_TOOL.md` under "Straight and smart paths": a markdown
    table with one row per verb, three columns (verb / kind / description),
    which renders as four `|` characters per row -- distinct from either
    legitimate 2-column table already in SKILL.md (capability/needs,
    you-know/use), which render as three `|` characters per row.
    """
    _frontmatter, body = _load_skill_frontmatter_and_body()

    assert "Straight and smart paths" not in body, (
        "SKILL.md must not carry SMART_TOOL.md's verb-reference-table section verbatim"
    )
    assert "| deterministic |" not in body, "a per-verb 'deterministic' table cell means the verb table got inlined"
    assert "| model-backed |" not in body, "a per-verb 'model-backed' table cell means the verb table got inlined"

    four_pipe_lines = [line for line in body.splitlines() if line.count("|") >= 4]
    assert four_pipe_lines == [], (
        f"SKILL.md must not contain any 3+-column table row (the verb table's shape); found: {four_pipe_lines!r}"
    )


def test_smart_tool_md_verb_table_is_still_the_shape_this_guard_assumes() -> None:
    """If SMART_TOOL.md's verb table ever stops being a 4-pipe, 3-column
    table, the discriminator above stops meaning anything -- so pin the
    assumption here, next to the guard that depends on it.
    """
    smart_tool_body = (REPO_ROOT / "src" / "aud" / "SMART_TOOL.md").read_text(encoding="utf-8")
    assert "Straight and smart paths" in smart_tool_body
    verb_row = next(line for line in smart_tool_body.splitlines() if line.startswith("| `analyze`"))
    assert verb_row.count("|") == 4
