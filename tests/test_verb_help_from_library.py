"""Regression guard for `help-comes-from-library`.

AGENTS.md #2: "Every capability lives in aud.lib ... A capability reachable
only through the CLI is a defect." Before this fix, `aud <verb> --help`
prose was read by cli.py directly out of `aud.verbdoc.VERB_DOCS` -- a Python
caller with no CLI, no argparse, had no way to obtain the same documentation
`aud <verb> --help` prints, short of importing the CLI's private data
module itself.

`aud.lib.verb_help` is the fix: a plain library function, no argparse
anywhere in its call path, that returns the exact same text. This module
proves three things: every registered verb is reachable this way, what a
library caller gets is byte-identical to what the CLI actually prints (a
real subprocess, not just the same in-process data), and an unknown verb
name returns None rather than raising or crashing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from aud import lib
from aud.cli import registered_verbs
from aud.verbdoc import VERB_DOCS

_ALL_VERBS = sorted(registered_verbs())


def _run_cli_help(verb: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", verb, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_every_registered_verb_is_reachable_through_the_library() -> None:
    """A pure library call -- no CLI, no argparse -- gets every verb's doc."""
    missing = [verb for verb in _ALL_VERBS if lib.verb_help(verb) is None]
    assert not missing, f"aud.lib.verb_help has no documentation for: {missing}"


def test_library_accessor_returns_exactly_the_verbdoc_prose() -> None:
    """The library reads the same data cli.py used to read directly.

    This is a sourcing change, not a rewrite (see AGENTS.md and the fix's
    task description): `aud.verbdoc.VERB_DOCS` stays the data, `aud.lib`
    becomes the one accessor everyone -- CLI included -- goes through.
    """
    for verb in _ALL_VERBS:
        assert lib.verb_help(verb) == VERB_DOCS[verb]


def test_unknown_verb_name_returns_none_not_a_crash() -> None:
    assert lib.verb_help("not-a-real-verb") is None
    assert lib.verb_help("") is None


@pytest.mark.parametrize("verb", _ALL_VERBS)
def test_library_consumer_gets_what_the_cli_actually_prints(verb: str) -> None:
    """End-to-end proof, not just an in-process data comparison.

    Runs `aud <verb> --help` as a real subprocess and asserts its stdout is
    byte-identical to what a pure-library caller (`aud.lib.verb_help`, no
    CLI, no argparse) gets for the same verb -- the exact claim this fix
    exists to make true.
    """
    proc = _run_cli_help(verb)
    assert proc.returncode == 0, proc.stderr
    library_doc = lib.verb_help(verb)
    assert library_doc is not None
    assert proc.stdout == library_doc + "\n"
