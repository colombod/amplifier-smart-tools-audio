"""Every documented `speech`-extra install command must carry the PyTorch
CPU-only wheel index (issue #44).

`[tool.uv.sources]` in pyproject.toml only pins `torch` to the CPU index for
THIS project's own `uv sync`/`uv run` -- it is NOT consulted when `aud` is
installed as a dependency from git (`uv tool install 'aud[speech] @
git+...'`), which is exactly how every documented install command in this
repo tells a reader to install it. Without the explicit `--index`/
`--index-strategy` flags, that command silently resolves the default
(possibly CUDA, ~8 GB on Linux) `torch` build instead of the ~187 MB CPU-only
one -- measured directly while resolving issue #44 (see this PR's body).

This module is the enforcement: every REAL install instruction (a fenced
markdown code block, or a Python string literal building CLI output) that
installs the `speech` extra must carry the CPU index nearby. A prose mention
INSIDE a sentence (a single-backtick inline code span, not a fenced block) is
deliberately exempt -- `docs/CONFIGURATION.md` uses exactly one of these to
describe what happens WITHOUT the index, as the reason the index is needed,
and that counter-example must not itself be required to carry the fix it is
explaining.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

_CPU_INDEX_MARKER = "--index https://download.pytorch.org/whl/cpu"

# The command target this guard looks for -- any line naming this is
# installing (or telling a reader to install) the `speech` extra from git.
_SPEECH_INSTALL_RE = re.compile(r"uv tool install ['\"]aud\[speech[^'\"]*['\"]")

# (path, description) -- every file this repo ships that names a `speech`
# install command, checked exhaustively so a new one added later and missed
# here fails this test's own completeness check (see
# test_every_known_speech_install_site_is_covered).
_MARKDOWN_FILES = ("README.md", "docs/CONFIGURATION.md", "skills/aud/SKILL.md")
_PYTHON_FILES = ("src/aud/lib.py", "src/aud/core/skill.py", "src/aud/dsp/speech.py")

# This test module itself necessarily contains the pattern it greps for
# (in docstrings, the regex literal, and the example command below) -- it is
# not a documentation site and is excluded from the completeness sweep.
_THIS_FILE = Path(__file__).resolve()


def _fenced_code_blocks(markdown_text: str) -> list[str]:
    """Every fenced (``` ... ```) code block's contents, concatenated text."""
    blocks = []
    in_block = False
    current: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                blocks.append("\n".join(current))
                current = []
            in_block = not in_block
            continue
        if in_block:
            current.append(line)
    return blocks


def _install_commands_in_markdown(path: Path) -> list[str]:
    """Every fenced-code-block line that looks like a `speech` install command."""
    text = path.read_text(encoding="utf-8")
    commands = []
    for block in _fenced_code_blocks(text):
        for line in block.splitlines():
            if _SPEECH_INSTALL_RE.search(line):
                commands.append(line)
    return commands


def _install_commands_in_python(path: Path) -> list[str]:
    """Every source line (Python string literal or otherwise) that looks like
    a `speech` install command, joined with a window of following lines so a
    command built from concatenated string literals across several lines
    (as lib.py and skill.py both do) is checked as one unit."""
    lines = path.read_text(encoding="utf-8").splitlines()
    commands = []
    for i, line in enumerate(lines):
        if _SPEECH_INSTALL_RE.search(line):
            window = "\n".join(lines[i : i + 4])
            commands.append(window)
    return commands


def test_every_markdown_speech_install_command_carries_the_cpu_index() -> None:
    violations = []
    for rel in _MARKDOWN_FILES:
        path = REPO_ROOT / rel
        for command in _install_commands_in_markdown(path):
            if _CPU_INDEX_MARKER not in command:
                violations.append(f"{rel}: {command!r}")
    assert not violations, (
        "every documented `speech` extra install command (in a fenced code block) must carry "
        f"{_CPU_INDEX_MARKER!r}; missing on:\n" + "\n".join(violations)
    )


def test_every_python_speech_install_command_carries_the_cpu_index() -> None:
    violations = []
    for rel in _PYTHON_FILES:
        path = REPO_ROOT / rel
        for command in _install_commands_in_python(path):
            if _CPU_INDEX_MARKER not in command:
                violations.append(f"{rel}: {command!r}")
    assert not violations, (
        "every documented `speech` extra install command (in Python source building CLI output "
        f"or an error remedy) must carry {_CPU_INDEX_MARKER!r}; missing on:\n" + "\n".join(violations)
    )


def test_the_configuration_md_counter_example_is_the_one_exempt_prose_mention() -> None:
    """Confirms the one known exemption is still exactly what this module's
    docstring claims it is -- an inline, single-backtick prose mention
    describing the problem the index fixes, not a fenced-block instruction.
    If this ever changes shape, this test's own assumption needs revisiting,
    not a silent pass."""
    text = (REPO_ROOT / "docs/CONFIGURATION.md").read_text(encoding="utf-8")
    prose_line = next(line for line in text.splitlines() if "resolves the default torch build" in line)
    assert _SPEECH_INSTALL_RE.search(prose_line), "the counter-example line must still name the install command"
    assert prose_line.strip().startswith("index below"), "the counter-example must still be prose, not a fenced block"


def test_every_known_speech_install_site_is_covered() -> None:
    """A repo-wide grep must not find a `speech` install command in a file
    this module does not already check -- otherwise a new documentation site
    could add an unguarded command and this guard would stay green by
    omission."""
    checked = {*(REPO_ROOT / p for p in _MARKDOWN_FILES), *(REPO_ROOT / p for p in _PYTHON_FILES)}
    found_elsewhere = []
    for path in REPO_ROOT.rglob("*"):
        if path in checked or path == _THIS_FILE or not path.is_file():
            continue
        if path.suffix not in (".md", ".py"):
            continue
        if "tests/fixtures/recorded" in str(path) or "/.venv" in str(path) or "/.venv-default" in str(path):
            continue
        if path.suffix == ".md" and path.name == "CHANGELOG.md":
            continue  # historical record, not a live instruction site
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _SPEECH_INSTALL_RE.search(text):
            found_elsewhere.append(str(path.relative_to(REPO_ROOT)))
    assert not found_elsewhere, (
        f"found a `speech` install command in file(s) this guard does not check: {found_elsewhere} -- "
        "add it to _MARKDOWN_FILES/_PYTHON_FILES above"
    )
