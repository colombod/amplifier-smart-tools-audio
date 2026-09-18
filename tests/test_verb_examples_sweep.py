"""Sweep test: every registered verb's own documented example must parse.

This is the regression guard for two real bugs found by running every verb
at the exact example its own `--help` prints:

1. `advise`, `master` and `preset` were registered with NO arguments at all,
   so their own documented worked examples ("aud advise in.wav", "aud master
   in.wav out.wav", "aud preset --list") failed with `usage_error` --
   "unrecognized arguments" -- instead of the `not_implemented` envelope the
   tool's own convention promises for an unbuilt capability (see AGENTS.md
   and `cli.py`'s `_NOT_YET_BUILT`).
2. A handled, named condition (a planned DSP gap, a missing input file)
   must never come back as `internal_error` -- that code is reserved for a
   genuinely unexpected exception (see AudError's docstring and
   `cli.py::main`'s catch-all).

The test is driven off `aud.cli.registered_verbs()` (itself kept in lock
step with `aud.core.skill.CAPABILITIES` by
`test_cli_envelope.py::test_cli_surface_matches_capabilities_manifest`) and
`aud.verbdoc.VERB_DOCS`, so a newly added verb is swept automatically --
nothing here needs editing when one is added, only its own doc/parser need
to agree with each other.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from aud.cli import PLAN_OUTPUT_VERBS, registered_verbs
from aud.verbdoc import VERB_DOCS

# Placeholder filenames verbdoc.py's prose uses, e.g. "in.wav", "out.wav",
# "reference.wav", "curve.json" -- see docstring for why substitution is
# positional per verb rather than a blind text replace.
_ALL_VERBS = sorted(registered_verbs())


def _run(args: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _own_commands(verb: str, doc: str) -> list[str]:
    """Every `aud <verb> ...` command documented for `verb`, own segment only.

    Scans the WHOLE doc (not just the text right after an "Example:"/"Worked
    example:" header) because some verbs (`cut`, `strip-silence`, `preset`)
    document more than one worked invocation. Joins backslash line
    continuations first, then splits each logical command on `|` and keeps
    only the segment whose first token is this verb -- the same verb can
    appear later in someone ELSE's pipe chain (e.g. 'aud render in.wav
    out.wav' inside `preset`'s own doc) and must not be picked up there.
    """
    text = doc.replace("\\\n", " ")
    commands: list[str] = []
    seen: set[str] = set()
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        # Only a line that IS a shell command (not prose that merely
        # mentions "aud verb" mid-sentence, e.g. "...plus 'aud verify' to
        # get the same result...") is a candidate -- real example lines are
        # always formatted on their own line, starting with "aud".
        if not line.startswith("aud "):
            continue
        line = re.sub(r"\s+#.*$", "", line).strip()
        for segment in line.split("|"):
            segment = segment.strip()
            if not segment.startswith("aud "):
                continue
            rest = segment[len("aud ") :].strip()
            tokens = shlex.split(rest)
            if tokens and tokens[0] == verb and rest not in seen:
                seen.add(rest)
                commands.append(rest)
    return commands


def _substitute_paths(
    verb: str,
    tokens: list[str],
    *,
    tiny_wav: Path,
    out_wav: Path,
    curve_in: Path,
    curve_out: Path,
) -> list[str]:
    """Replace verbdoc.py's placeholder filenames with real paths.

    Positional per verb, not a blind text replace: the same placeholder
    name plays different roles in different verbs -- 'out.wav' is
    `verify`'s INPUT (the file you just rendered) but `render`'s OUTPUT.
    Getting this wrong would make a verb fail on a made-up path and be
    indistinguishable from the argument-surface bug this sweep exists to
    catch (both look like a non-"usage_error" error, but for the wrong
    reason) -- so paths are resolved from the tool's own registered
    argparse surface (cli.py), not guessed from the prose.
    """
    t = list(tokens)
    if verb in ("analyze", "verify", "advise") and len(t) >= 2:
        t[1] = str(tiny_wav)
    elif verb in ("render", "master") and len(t) >= 3:
        t[1] = str(tiny_wav)
        t[2] = str(out_wav)
    elif verb == "detect" and len(t) >= 3:
        t[2] = str(tiny_wav)  # detect <kind> PATH ...
    elif verb == "curve" and len(t) >= 4 and t[1] == "extract":
        t[2] = str(tiny_wav)
        t[3] = str(curve_out)
    elif verb == "curve" and len(t) >= 5 and t[1] == "apply":
        t[2] = str(tiny_wav)
        t[3] = str(curve_in)
        t[4] = str(out_wav)
    elif verb == "eq-match" and "--curve" in t:
        idx = t.index("--curve")
        t[idx + 1] = str(curve_in)
    return t


def _cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for verb in _ALL_VERBS:
        doc = VERB_DOCS.get(verb, "")
        for command in _own_commands(verb, doc):
            cases.append((verb, command))
    return cases


ALL_CASES = _cases()


def test_every_registered_verb_has_at_least_one_documented_example() -> None:
    """Guards the sweep itself: a verb with zero extracted commands would
    silently test nothing for that verb."""
    covered = {verb for verb, _ in ALL_CASES}
    assert covered == set(_ALL_VERBS), f"no documented example found for: {set(_ALL_VERBS) - covered}"


@pytest.mark.parametrize(("verb", "command"), ALL_CASES, ids=[f"{v}:{c}" for v, c in ALL_CASES])
def test_documented_example_parses_and_returns_a_clean_envelope(
    verb: str, command: str, tiny_wav: Path, tmp_path: Path
) -> None:
    curve_in = tmp_path / "curve_in.json"
    curve_in.write_text(json.dumps({"200": -1.0, "1000": 0.5, "6000": -0.5}), encoding="utf-8")
    curve_out = tmp_path / "curve_out.json"
    out_wav = tmp_path / "out.wav"

    tokens = _substitute_paths(
        verb,
        shlex.split(command),
        tiny_wav=tiny_wav,
        out_wav=out_wav,
        curve_in=curve_in,
        curve_out=curve_out,
    )

    proc = _run(tokens)

    # Whatever happened, stdout must be exactly one parseable JSON document --
    # never a raw traceback, never prose alongside it.
    payload = json.loads(proc.stdout)
    assert "Traceback" not in proc.stdout
    assert "Traceback" not in proc.stderr

    if isinstance(payload, dict) and "error" in payload:
        assert set(payload) == {"error"}
        error = payload["error"]
        assert set(error) == {"code", "message", "remedy"}
        # The whole point: a documented-correct invocation must never look
        # like a USAGE mistake -- that would mean the flag doesn't parse.
        assert error["code"] != "usage_error", (
            f"'aud {command}' is presented as correct usage in the verb's own "
            f"--help, but was rejected as a usage error: {error!r}"
        )
    elif verb in PLAN_OUTPUT_VERBS:
        # 'plan' and every stage verb print the plan document itself, raw
        # and unwrapped, on success -- see cli.py's module docstring.
        assert isinstance(payload, dict)
        assert "stages" in payload
    else:
        assert isinstance(payload, dict)
        assert "result" in payload
