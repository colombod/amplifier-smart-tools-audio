"""Enforces `docs/DESIGN-ENVELOPE.md`: the patented masking/loudness formulations
must never appear in `src/`. A documented constraint with no check that runs is
just a comment (see `docs/DESIGN-ENVELOPE.md`'s Honesty section for the sibling
lesson this repo already paid for). This test is that check.

What is forbidden, and why (full detail in `docs/DESIGN-ENVELOPE.md`):

- `partial_loudness` / "partial loudness" -- the iZotope/NI family's core quantity
  (US10396744B2 et al.).
- `loudness_loss` / "loudness loss" -- the same family's derived quantity
  (a maskee's loudness alone minus its loudness in the presence of a masker).
- "phon"/"phons" used as a UNIT (a loudness value expressed in phons) -- the unit
  every claim in that family is expressed in. Deliberately NOT flagged as a
  substring of an unrelated word: this codebase's `src/` carries three genuine
  matches of the bare substring "phon" today, all inside longer words that have
  nothing to do with loudness units -- see `_KNOWN_FALSE_POSITIVE_WORDS` below.
- The alone-vs-in-mix / in-isolation-vs-in-presence comparison, in either word
  order and regardless of separator style (`loudness_alone`, "loudness alone",
  "alone-loudness", "in-presence loudness", ...) -- this is the shape
  US11469731B2 claims ("comparing" a loudness "absent" the other sources against
  the loudness "in the presence of" them), and US10763812B2's ranking-and-display
  claims restate the same difference in prose ("a loudness ... occurring in
  isolation" vs. "a partial loudness ... occurring concurrently").

Detection strategy: normalize each source line by lowercasing it and replacing
underscores/hyphens with spaces, then match plain-English phrase patterns
against the normalized text. This means a `snake_case` identifier, a
`kebab-case` identifier, and an ordinary English sentence in a docstring or
comment are all caught by the same pattern -- there is exactly one place each
concept is spelled out, not one regex per naming convention.

The escape hatch: a matching line is allowed through only if that EXACT line
also contains the literal marker `# DESIGN-ENVELOPE-EXCEPTION: <reason>`. See
`docs/DESIGN-ENVELOPE.md`'s "What Step 6's metric must demonstrably do" section
for why the marker lives inline with the flagged code rather than in a side
table. As of this writing, no such marker exists anywhere in `src/` -- adding
one is a decision to argue for in a pull request, not a mechanical workaround.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
_ENVELOPE_DOC = "docs/DESIGN-ENVELOPE.md"

_EXCEPTION_MARKER = "DESIGN-ENVELOPE-EXCEPTION:"

# Measured 2026-09-25 against this exact tree (see docs/DESIGN-ENVELOPE.md):
# three pre-existing bare-substring hits on "phon" -- src/aud/verbdoc.py:113
# and src/aud/dsp/detect.py:103 ("phone in a kitchen" / "phone-in-a-kitchen"),
# and src/aud/dsp/dereverb.py:4 ("one microphone signal") -- none of them the
# unit. Recorded here, not just skipped silently, so a NEW bare-"phon" hit
# that is NOT one of these two words still gets caught by the word-boundary
# pattern below (it only excludes the pattern from matching "phone"-shaped
# words in general, on purpose -- this constant is documentation of what was
# checked, not a suppression list the pattern itself depends on).
_KNOWN_FALSE_POSITIVE_WORDS = ("phone", "microphone")


def _normalize(line: str) -> str:
    """Lowercase and collapse identifier separators to spaces, so
    `partial_loudness`, `partial-loudness` and "partial loudness" all match
    the same phrase pattern."""
    return re.sub(r"[_\-]+", " ", line).lower()


# Each entry: (label, compiled regex over NORMALIZED text, one-line reason).
# Patterns are phrase-based (word-boundary aware via \s+ / \b), which is what
# lets normalization collapse every naming convention into one match site.
_FORBIDDEN: list[tuple[str, re.Pattern[str], str]] = [
    (
        "partial_loudness",
        re.compile(r"partial\s+loudness"),
        "the iZotope/NI family's core patented quantity (US10396744B2 et al.)",
    ),
    (
        "loudness_loss",
        re.compile(r"loudness\s+loss"),
        "the iZotope/NI family's derived patented quantity",
    ),
    (
        "phon (unit)",
        re.compile(r"\bphons?\b"),
        "the loudness unit every claim in the iZotope/NI family is expressed in "
        "(NOT the same as 'phone'/'microphone'/'headphone', which this pattern does not match)",
    ),
    (
        "loudness <alone/isolation/presence>",
        re.compile(r"loudness\s+(alone|in\s+isolation|in\s+presence|in\s+the\s+presence)"),
        "the alone-vs-in-mix comparison US11469731B2 claims and US10763812B2 restates",
    ),
    (
        "<alone/isolation/presence> loudness",
        re.compile(r"(alone|isolation|presence)\s+loudness"),
        "the alone-vs-in-mix comparison, reverse word order",
    ),
]


def _iter_src_py_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _scan() -> list[tuple[str, int, str, str, str]]:
    """Returns a list of (relpath, lineno, label, reason, raw_line) violations."""
    violations: list[tuple[str, int, str, str, str]] = []
    for path in _iter_src_py_files():
        text = path.read_text(encoding="utf-8")
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            if _EXCEPTION_MARKER in raw_line:
                continue
            normalized = _normalize(raw_line)
            for label, pattern, reason in _FORBIDDEN:
                if pattern.search(normalized):
                    violations.append((str(path.relative_to(_REPO_ROOT)), lineno, label, reason, raw_line.strip()))
    return violations


def test_no_forbidden_patented_loudness_formulations_in_src():
    """Fails loud, naming the envelope doc and the specific forbidden term, if
    any `src/` file contains a patented-formulation term. See this module's
    docstring and `docs/DESIGN-ENVELOPE.md` for what is forbidden and why."""
    violations = _scan()
    if not violations:
        return

    lines = [
        f"  {relpath}:{lineno}: forbidden term '{label}' -- {reason}\n    > {raw_line}"
        for relpath, lineno, label, reason, raw_line in violations
    ]
    raise AssertionError(
        f"Found {len(violations)} use(s) of a patented-formulation term forbidden by "
        f"{_ENVELOPE_DOC}:\n\n" + "\n".join(lines) + "\n\n"
        f"Read {_ENVELOPE_DOC} before implementing a masking/loudness metric: it names "
        "exactly which formulations are off-limits (energy-domain masking is required "
        "instead) and why. If this is a genuine false positive, the fix is a narrower "
        "pattern in this test, not a suppression of the line -- see this module's "
        "docstring for the deliberate, visible escape hatch if an exception is truly "
        "warranted."
    )


def test_known_false_positive_words_still_present_and_still_not_flagged():
    """Guards the guard: asserts the three pre-existing bare-'phon' substrings
    this repo actually carries are still there and still pass clean, so this
    test suite cannot quietly stop exercising the word-boundary distinction
    the design envelope depends on (a check that cannot fail on the exact
    case it was built for is worse than no check)."""
    found_words: set[str] = set()
    for path in _iter_src_py_files():
        text_lower = path.read_text(encoding="utf-8").lower()
        for word in _KNOWN_FALSE_POSITIVE_WORDS:
            if word in text_lower:
                found_words.add(word)

    missing = set(_KNOWN_FALSE_POSITIVE_WORDS) - found_words
    assert not missing, (
        f"Expected to still find {sorted(missing)} somewhere under src/ (as of 2026-09-25 these "
        "are the known bare-'phon'-substring words this codebase carries). If they were removed "
        "or renamed, this test's premise changed -- update _KNOWN_FALSE_POSITIVE_WORDS and confirm "
        "the phon-unit pattern in test_no_forbidden_patented_loudness_formulations_in_src still has "
        "a real false positive to avoid, rather than deleting the guard."
    )

    # And the main scan must not have flagged any of them as the forbidden unit.
    violations = _scan()
    phon_violations = [v for v in violations if v[2] == "phon (unit)"]
    assert not phon_violations, (
        "The phon-unit pattern flagged a line it should not have (a known false-positive word "
        f"regressed into a real match): {phon_violations}"
    )
