"""Tests for `aud.core.skill.render_skill` -- the text `aud --help` prints.

Guards the four things a caller cannot get anywhere else: a hard line-count
ceiling (Agent Skills convention), copy-pasteable install commands, the
prerequisite ladder for the credential-gated and extra-gated capabilities,
and the tool's non-goals -- so `aud --help` alone is enough to know both
what to run and what NOT to reach for this tool to do.
"""

from __future__ import annotations

from aud.core.skill import render_skill

_LINE_CEILING = 500


def test_render_skill_stays_under_the_agent_skills_line_ceiling() -> None:
    text = render_skill()
    line_count = len(text.splitlines())
    assert line_count < _LINE_CEILING, f"rendered skill is {line_count} lines, must stay under {_LINE_CEILING}"


def test_render_skill_has_copy_pasteable_install_commands() -> None:
    text = render_skill()
    assert "uv tool install git+https://github.com/colombod/amplifier-smart-tools-audio" in text
    assert "aud[speech,stretch]" in text
    assert 'uv add "aud @ git+https://github.com/colombod/amplifier-smart-tools-audio"' in text
    assert "uvx --from git+https://github.com/colombod/amplifier-smart-tools-audio aud --help" in text
    assert "npx skills add colombod/amplifier-smart-tools-audio" in text


def test_render_skill_names_prerequisites_and_points_at_check() -> None:
    text = render_skill()
    assert "ffmpeg" in text
    assert "aud[speech]" in text
    assert "aud check" in text


def test_render_skill_states_non_goals() -> None:
    text = render_skill()
    lowered = text.lower()
    for phrase in ("mixing", "multitrack", "transcription", "video files", "music"):
        assert phrase in lowered, f"non-goal {phrase!r} missing from rendered skill"


def test_render_skill_still_ends_with_the_per_verb_help_pointer() -> None:
    text = render_skill()
    assert "Run 'aud <verb> --help' for the full documentation of any one verb." in text
