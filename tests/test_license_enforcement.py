"""Enforces AGENTS.md section 1 ("aud is MIT... these dependencies are forbidden")
and the user ruling behind it: MIT-incompatible (GPL-family, copyleft) dependencies
must be structurally impossible to introduce, not merely absent today. See
`tests/license_policy.py` for the approach and why each signal was chosen.

Every evasion-pattern test below is named for the pattern it proves the checker
defeats, and is written against the REAL detection functions in `license_policy`
-- if `denylist_match`, `scan_source_text`, `classify`, or
`scan_declared_dependencies_against_denylist` were removed or weakened (e.g. the
denylist emptied, the AST walk restricted to module-level nodes only, or the
name normalization dropped), the corresponding test here goes red. That was
verified directly -- see the PR description for the red-then-green transcripts.
"""

from __future__ import annotations

from importlib.metadata import distribution

import pytest

from tests import license_policy
from tests.license_policy import (
    PYPROJECT_PATH,
    SRC_ROOT,
    DistInfo,
    Verdict,
    _classify_classifiers,
    _classify_spdx_expression,
    classify,
    denylist_match,
    scan_declared_dependencies_against_denylist,
    scan_installed_environment,
    scan_source_text,
    scan_source_tree,
)

# ---------------------------------------------------------------------------
# The real checks, run against the real tree -- "run it and report the truth"
# ---------------------------------------------------------------------------


def test_src_tree_has_no_forbidden_imports():
    """AST-scans every file under src/ for a forbidden import at any nesting
    depth, or a literal-argument importlib.import_module/__import__ call."""
    violations = scan_source_tree(SRC_ROOT)
    assert not violations, "Forbidden import(s) found in src/:\n" + "\n".join(
        f"  {v.relpath}:{v.lineno}: {v.pattern} of {v.imported_name!r} -- {v.denylist_entry.reason}" for v in violations
    )


def test_installed_environment_has_no_forbidden_or_unknown_licences():
    """Every distribution actually resolved and installed in THIS environment
    (direct or transitive) must classify ALLOWED. UNKNOWN is a failure, not a
    silent pass -- missing/ambiguous metadata is not evidence of permissiveness."""
    results = scan_installed_environment()
    bad = [(info, verdict, reason) for info, verdict, reason in results if verdict is not Verdict.ALLOWED]
    assert not bad, "Non-ALLOWED installed distribution(s):\n" + "\n".join(
        f"  {info.name}: {verdict.value.upper()} -- {reason}" for info, verdict, reason in bad
    )


def test_declared_dependencies_are_not_denylisted():
    """Static parse of the REAL pyproject.toml (main dependencies + every
    optional-dependencies extra), regardless of what is currently installed."""
    violations = scan_declared_dependencies_against_denylist(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert not violations, "Denylisted package(s) declared in pyproject.toml:\n" + "\n".join(
        f"  {name}: {entry.reason}" for name, entry in violations
    )


# ---------------------------------------------------------------------------
# Evasion pattern 1: import inside a function body, not module top level
# ---------------------------------------------------------------------------


def test_pattern1_function_body_import_is_caught():
    source = "def load_effects():\n    import pedalboard\n    return pedalboard\n"
    violations = scan_source_text(source, "synthetic/pattern1_function_body_import.py")
    assert len(violations) == 1
    (violation,) = violations
    assert violation.imported_name == "pedalboard"
    assert violation.pattern == "import"
    assert violation.lineno == 2
    assert violation.denylist_entry.root == "pedalboard"


def test_pattern1_nested_deeper_than_one_function_is_still_caught():
    """Nesting depth is not a variable the scanner tracks by design (ast.walk
    visits every node); confirm two levels of nesting still gets caught."""
    source = (
        "class Engine:\n"
        "    def render(self):\n"
        "        if True:\n"
        "            from matchering import process\n"
        "            return process\n"
    )
    violations = scan_source_text(source, "synthetic/pattern1_deep_nesting.py")
    assert len(violations) == 1
    (violation,) = violations
    assert violation.imported_name == "matchering"
    assert violation.pattern == "from-import"


# ---------------------------------------------------------------------------
# Evasion pattern 2: importlib.import_module(...) / __import__(...)
# ---------------------------------------------------------------------------


def test_pattern2_importlib_import_module_literal_is_caught():
    source = "import importlib\n\ndef load():\n    return importlib.import_module('pedalboard')\n"
    violations = scan_source_text(source, "synthetic/pattern2_import_module.py")
    assert len(violations) == 1
    (violation,) = violations
    assert violation.imported_name == "pedalboard"
    assert violation.pattern == "importlib.import_module"


def test_pattern2_bare_import_module_after_from_import_is_caught():
    source = "from importlib import import_module\n\ndef load():\n    return import_module('essentia.standard')\n"
    violations = scan_source_text(source, "synthetic/pattern2_bare_import_module.py")
    assert len(violations) == 1
    (violation,) = violations
    assert violation.imported_name == "essentia"
    assert violation.pattern == "importlib.import_module"


def test_pattern2_dunder_import_literal_is_caught():
    source = "def load():\n    return __import__('rubberband_ctypes')\n"
    violations = scan_source_text(source, "synthetic/pattern2_dunder_import.py")
    assert len(violations) == 1
    (violation,) = violations
    assert violation.imported_name == "rubberband_ctypes"
    assert violation.pattern == "__import__"


def test_pattern2_non_literal_argument_is_not_falsely_flagged():
    """The repo's OWN src/aud/lib.py calls `importlib.import_module(module_name)`
    with a variable (module_name sourced from a tuple of core package names). A
    scanner that flagged every non-literal dynamic-import call would break that
    real, legitimate code. Confirm the scanner does not do that."""
    source = (
        "import importlib\n\n"
        "def _module_available(module_name):\n"
        "    try:\n"
        "        importlib.import_module(module_name)\n"
        "    except ImportError:\n"
        "        return False\n"
        "    return True\n"
    )
    violations = scan_source_text(source, "synthetic/pattern2_non_literal.py")
    assert violations == []


# ---------------------------------------------------------------------------
# Evasion pattern 3: dependency declared in an OPTIONAL extra, not main deps
# ---------------------------------------------------------------------------


def test_pattern3_dependency_hidden_in_optional_extra_is_caught():
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
stretch = ["python-stretch>=0.3"]
evil = ["pedalboard>=0.9"]
"""
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert len(violations) == 1
    (name, entry) = violations[0]
    assert name == "pedalboard"
    assert entry.root == "pedalboard"


def test_pattern3_dependency_hidden_in_extra_with_marker_is_caught():
    """A version specifier plus an environment marker still resolves to the
    same bare package name."""
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = []

[project.optional-dependencies]
mastering = ["matchering>=2.0,<3.0; python_version >= '3.10'"]
"""
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert len(violations) == 1
    assert violations[0][0] == "matchering"


def test_pattern3_clean_extras_produce_no_violations():
    """Guards the guard: a pyproject with only permitted extras must not false-positive."""
    violations = scan_declared_dependencies_against_denylist(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert violations == [], (
        f"This repo's REAL pyproject.toml declared a denylisted dependency (main or extra): {violations}"
    )


# ---------------------------------------------------------------------------
# Evasion pattern 4: a permissive-looking package that pulls in a GPL one --
# proved with REAL recorded PyPI metadata, not a hand-built fake (AGENTS.md 3b).
# `scan_installed_environment` treats every installed distribution flatly, so
# "transitive" vs. "direct" is not a distinction the scanner needs: anything
# anywhere in the resolved graph shows up as its own distribution and is
# classified the same way. These tests prove the underlying classification
# logic actually flags real GPL-family metadata, using data transcribed
# verbatim from PyPI (see tests/license_policy.py's module docstring and the
# DENYLIST entries for the exact source URLs and retrieval date).
# ---------------------------------------------------------------------------


def test_pattern4_essentia_real_recorded_metadata_is_forbidden_via_metadata_alone():
    """essentia ships NO legacy 'License ::' classifiers at all (verified
    2026-09-26 via https://pypi.org/pypi/essentia/json) -- only a modern
    License-Expression header, 'AGPL-3.0-only'. This proves the METADATA rule
    catches it independently of the denylist: a classifier-only checker would
    see zero signal and call it UNKNOWN; the License-Expression check is what
    makes this the general rule the task asked for, not just a name match."""
    verdict = _classify_spdx_expression("AGPL-3.0-only")
    assert verdict is Verdict.FORBIDDEN


def test_pattern4_pedalboard_real_recorded_metadata_is_forbidden_via_classifier_alone():
    """Recorded 2026-09-26 via https://pypi.org/pypi/pedalboard/json. Proves the
    classifier-based general rule catches a real GPL-family package by its
    metadata, with the denylist entirely out of the picture."""
    verdict = _classify_classifiers(["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])
    assert verdict is Verdict.FORBIDDEN


def test_pattern4_matchering_real_recorded_metadata_is_forbidden_via_classifier_alone():
    """Recorded 2026-09-26 via https://pypi.org/pypi/matchering/json."""
    verdict = _classify_classifiers(["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])
    assert verdict is Verdict.FORBIDDEN


def test_pattern4_pyrubberband_metadata_alone_would_wrongly_allow_it():
    """The load-bearing negative case: pyrubberband's OWN recorded metadata
    (https://pypi.org/pypi/pyrubberband/json, 2026-09-26) is genuinely ISC. The
    metadata rule ALONE calls this ALLOWED -- proving why the name backstop in
    `classify()` must be unconditional, not just a fallback for missing metadata."""
    info = DistInfo(
        name="pyrubberband",
        license_expression=None,
        classifiers=("License :: OSI Approved :: ISC License (ISCL)",),
        license_field="ISC",
    )
    metadata_only_verdict = _classify_classifiers(list(info.classifiers))
    assert metadata_only_verdict is Verdict.ALLOWED, "the metadata signal itself really is permissive"

    combined_verdict, reason = classify(info)
    assert combined_verdict is Verdict.FORBIDDEN
    assert "denylisted despite metadata verdict allowed" in reason


# ---------------------------------------------------------------------------
# Evasion pattern 5: name-spelling variants (case, hyphenation, package alias)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "Rubber Band",
        "rubberband-ctypes",
        "pyrubberband",
        "RubberBand",
        "rubber_band",
        "python-rubberband",
    ],
)
def test_pattern5_name_spelling_variants_are_all_denylist_matched(variant):
    entry = denylist_match(variant)
    assert entry is not None, f"{variant!r} was not recognised as a Rubber Band variant"
    assert entry.root == "rubberband"


@pytest.mark.parametrize(
    ("source", "expected_name"),
    [
        ("import rubberband_ctypes\n", "rubberband_ctypes"),
        ("import pyrubberband\n", "pyrubberband"),
        ("from PyRubberBand import stretch\n", "PyRubberBand"),
    ],
)
def test_pattern5_name_variants_are_caught_in_source_imports_too(source, expected_name):
    violations = scan_source_text(source, "synthetic/pattern5_variant.py")
    assert len(violations) == 1
    assert violations[0].imported_name == expected_name
    assert violations[0].denylist_entry.root == "rubberband"


def test_pattern5_unrelated_name_is_not_falsely_matched():
    """Guards the guard: denylist matching is substring-on-normalized-name, not
    a loose fuzzy match -- confirm it does not fire on unrelated packages."""
    for clean_name in ("numpy", "scipy", "soundfile", "pyloudnorm", "pydantic", "pyyaml"):
        assert denylist_match(clean_name) is None, f"{clean_name} was falsely denylisted"


# ---------------------------------------------------------------------------
# Critical false-positive guard: scipy's OWN `License` metadata field literally
# contains "GPL-3.0-or-later" and "LGPL-2.1-or-later" (bundled OpenBLAS/gfortran
# runtime notices, under the GCC Runtime Library Exception) -- yet scipy is a
# permitted, permissive (BSD) dependency this project actually ships. Verified
# directly against the REAL installed distribution in this environment.
# ---------------------------------------------------------------------------


def test_scipy_bundled_gpl_notice_is_not_a_false_positive():
    dist = distribution("scipy")
    license_field = dist.metadata.get("License") or ""
    # Confirm the trap is real, not a stale claim in a comment.
    assert "GPL-3.0-or-later" in license_field or "LGPL-2.1-or-later" in license_field, (
        "scipy's License metadata no longer contains the bundled GPL/LGPL runtime notice text "
        "this test exists to guard against -- if scipy's packaging changed, this guard's "
        "premise changed too; re-verify before deleting it."
    )

    info = license_policy._distinfo_from_installed(dist)
    verdict, reason = classify(info)
    assert verdict is Verdict.ALLOWED, (
        f"scipy was misclassified as {verdict.value} ({reason}) -- the classifier-first "
        "priority order in classify()/tests/license_policy.py has regressed and is now "
        "keyword-matching the long bundled-notices License field."
    )
    assert "classifiers" in reason


def test_missing_licence_metadata_is_unknown_not_silently_allowed():
    info = DistInfo(name="totally-unknown-package", license_expression=None, classifiers=(), license_field=None)
    verdict, reason = classify(info)
    assert verdict is Verdict.UNKNOWN
    assert "no licence metadata at all" in reason


def test_ambiguous_long_license_field_with_no_classifiers_is_unknown():
    """A package with no classifiers and no License-Expression, but a long
    free-text License field (which could bundle anything) -- must not be
    silently allowed just because it also does not contain a forbidden keyword."""
    info = DistInfo(
        name="some-vendored-thing",
        license_expression=None,
        classifiers=(),
        license_field="Copyright (c) Example Corp. All rights reserved. " * 10,
    )
    verdict, _reason = classify(info)
    assert verdict is Verdict.UNKNOWN


# ---------------------------------------------------------------------------
# Real installed permissive dependencies must actually be ALLOWED (not just
# "not forbidden") -- a check that only ever asserted absence of FORBIDDEN
# would let UNKNOWN slip through the top-level environment test by accident.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["numpy", "scipy", "soundfile", "pyloudnorm", "pydantic", "pyyaml"])
def test_real_core_dependencies_classify_allowed(name):
    info = license_policy._distinfo_from_installed(distribution(name))
    verdict, reason = classify(info)
    assert verdict is Verdict.ALLOWED, f"{name} classified {verdict.value} ({reason})"
