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
    _ALLOWED_SPDX_IDS,
    PYPROJECT_PATH,
    SRC_ROOT,
    BundledBinaryAcknowledgement,
    DistInfo,
    Verdict,
    _classify_classifiers,
    _classify_spdx_expression,
    _scan_libs_dir,
    classify,
    denylist_match,
    iter_declared_dependency_specs,
    scan_bundled_runtime_binaries,
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


# ---------------------------------------------------------------------------
# Gap (d): "THE ALLOW-LIST IS SPECIFIED, use exactly it." Locks the SPDX
# allow-list to the exact set from the governing work item (smart_tools-c53) /
# docs/DESIGN-ENVELOPE.md -- neither a superset nor a subset. If someone edits
# `_ALLOWED_SPDX_IDS` to add or drop an id without updating this test (and the
# governing doc), this fails loud rather than silently drifting.
# ---------------------------------------------------------------------------

_SPECIFIED_ALLOW_LIST = frozenset(
    {
        "MIT",
        "MIT-0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "ISC",
        "Apache-2.0",
        "PSF-2.0",
        "Python-2.0",
        "Zlib",
        "Unlicense",
        "CC0-1.0",
        "HPND",
    }
)


def test_allow_list_matches_exactly_the_specified_set():
    specified_upper = {s.upper() for s in _SPECIFIED_ALLOW_LIST}
    assert specified_upper == _ALLOWED_SPDX_IDS, (
        f"_ALLOWED_SPDX_IDS has drifted from the specified allow-list.\n"
        f"  missing: {specified_upper - _ALLOWED_SPDX_IDS}\n"
        f"  extra:   {_ALLOWED_SPDX_IDS - specified_upper}"
    )


@pytest.mark.parametrize("spdx_id", sorted(_SPECIFIED_ALLOW_LIST))
def test_each_specified_allow_list_id_classifies_allowed(spdx_id):
    assert _classify_spdx_expression(spdx_id) is Verdict.ALLOWED, f"{spdx_id} did not classify ALLOWED"


def test_anything_off_the_allow_list_is_unknown_not_allowed():
    """ "ANYTHING ELSE FAILS, INCLUDING UNKNOWN" -- a plausible-sounding but
    unlisted permissive SPDX id (e.g. a licence real projects use, but not on
    this project's reviewed allow-list) must not slip through as ALLOWED."""
    for unlisted in ("BSL-1.0", "OFL-1.1", "WTFPL", "CDDL-1.0", "EPL-2.0"):
        verdict = _classify_spdx_expression(unlisted)
        assert verdict is not Verdict.ALLOWED, f"{unlisted} is not on the specified allow-list but classified ALLOWED"


# ---------------------------------------------------------------------------
# Gap (b): non-truncation. An earlier manual audit truncated licence strings to
# 45 characters and reported a clean tree -- hiding scipy's bundled GPL/LGPL
# runtime notices entirely. Prove the read captures the full field.
# ---------------------------------------------------------------------------


def test_scipy_license_field_read_is_not_truncated():
    dist = distribution("scipy")
    license_field = dist.metadata.get("License") or ""
    assert len(license_field) > 45_000, (
        f"scipy's License field read back as only {len(license_field)} chars -- this looks "
        "truncated. docs/DESIGN-ENVELOPE.md records the exact failure this guards: a "
        "45-character-capped read of licence metadata hid the bundled GPL/LGPL runtime "
        "notices and produced a false clean audit. The real field is 47,559 characters."
    )
    assert "GPL-3.0-or-later" in license_field
    assert "LGPL-2.1-or-later" in license_field


def test_distinfo_license_field_is_not_truncated_relative_to_raw_metadata():
    """Proves the EXTRACTION path (`_distinfo_from_installed`) specifically --
    not just the raw `importlib.metadata` read -- carries the full field through
    with no length cap of its own."""
    dist = distribution("scipy")
    raw_license_field = dist.metadata.get("License") or ""
    info = license_policy._distinfo_from_installed(dist)
    assert info.license_field is not None
    assert len(info.license_field) == len(raw_license_field)
    assert len(info.license_field) > 45_000


# ---------------------------------------------------------------------------
# Gap (a): bundled copyleft runtime binaries. numpy.libs/ and scipy.libs/ ship
# libgfortran (GPL-3.0-or-later WITH GCC-exception-3.1) and libquadmath
# (LGPL-2.1-or-later) -- MIT-compatible AS USED, per docs/DESIGN-ENVELOPE.md, but
# only because each is a REVIEWED, ENUMERATED decision, not a blanket pass for
# those two directories. A new, unreviewed bundled binary must fail.
# ---------------------------------------------------------------------------


def test_bundled_runtime_binaries_currently_on_disk_are_all_acknowledged():
    """The clean-tree criterion (gap (e)), applied to the bundled-runtime scan
    specifically: given the CURRENT tree (numpy, scipy and their bundled
    libgfortran/libquadmath present), this check MUST PASS."""
    violations = scan_bundled_runtime_binaries()
    assert violations == [], "Unacknowledged bundled runtime binaries found:\n" + "\n".join(
        f"  {v.path}: {v.reason}" for v in violations
    )


def test_bundled_runtime_scan_finds_the_real_libgfortran_and_libquadmath():
    """Guards the guard the other direction: confirm the acknowledgement list
    actually matches real filenames in THIS environment, rather than passing
    only because nothing was found to check (an empty `.libs/` scan and a
    correctly-matched non-empty one both return `[]`; this distinguishes them)."""
    from tests.license_policy import _libs_dirs_for

    libs_dirs = _libs_dirs_for(("numpy", "scipy"))
    assert libs_dirs, "Expected at least one of numpy.libs/, scipy.libs/ to exist in this environment"
    all_files = [entry.name for d in libs_dirs for entry in d.iterdir() if entry.is_file()]
    assert any(name.startswith("libgfortran-") for name in all_files), (
        f"No libgfortran binary found under {libs_dirs} -- premise of this guard changed, re-verify"
    )
    assert any(name.startswith("libquadmath-") for name in all_files), (
        f"No libquadmath binary found under {libs_dirs} -- premise of this guard changed, re-verify"
    )


def test_bundled_runtime_scan_fails_on_a_new_unacknowledged_binary(tmp_path):
    """Proves the required negative: a NEW bundled copyleft binary that is not
    on the reviewed list must fail. Synthetic -- no real new copyleft library is
    needed to prove the scan reacts to one, mirroring how `scan_source_text` is
    proved against synthetic source strings elsewhere in this file."""
    libs_dir = tmp_path / "somepkg.libs"
    libs_dir.mkdir()
    (libs_dir / "libgfortran-abc12345.so.5.0.0").write_bytes(b"acknowledged, must not fire")
    (libs_dir / "libquadmath-abc12345.so.0.0.0").write_bytes(b"acknowledged, must not fire")
    (libs_dir / "libnewcopyleftthing-deadbeef.so.1.0.0").write_bytes(b"NOT on the acknowledgement list")

    violations = _scan_libs_dir(libs_dir)

    assert len(violations) == 1, f"expected exactly 1 violation, got {violations}"
    assert "libnewcopyleftthing-deadbeef.so.1.0.0" in violations[0].path
    assert "not on the reviewed bundled-runtime acknowledgement list" in violations[0].reason
    assert "docs/DESIGN-ENVELOPE.md" in violations[0].reason


def test_bundled_runtime_scan_empty_dir_and_missing_dir_both_pass():
    """Guards the guard: an empty or absent `.libs/` directory must not be
    misread as a violation."""
    assert _scan_libs_dir(license_policy.REPO_ROOT / "definitely-does-not-exist.libs") == []


@pytest.mark.parametrize(
    "filename",
    [
        "libgfortran-040039e1-0352e75f.so.5.0.0",  # real scipy filename, recorded above
        "libgfortran-8f1e9814.so.5.0.0",  # real scipy filename (second variant)
        "libquadmath-828275a7.so.0.0.0",  # real scipy filename
        "libquadmath-96973f99-934c22de.so.0.0.0",  # real numpy filename
        "libscipy_openblas-6cdc3b4a.so",  # real scipy filename
        "libscipy_openblas64_-32a4b2a6.so",  # real numpy filename
    ],
)
def test_bundled_runtime_acknowledgement_patterns_match_real_recorded_filenames(filename):
    assert any(ack.pattern.match(filename) for ack in license_policy.BUNDLED_RUNTIME_ACKNOWLEDGEMENTS), (
        f"{filename!r} (a real filename recorded from this environment) does not match any "
        "acknowledged pattern -- the regexes have drifted from what auditwheel actually produces."
    )


def test_bundled_runtime_acknowledgement_is_an_enumerated_decision_not_a_blanket_directory_exemption():
    """Direct proof that the acknowledgement is per-FILENAME-PATTERN, not per-directory:
    an acknowledged-looking directory containing an unrelated file still flags it."""
    entry = BundledBinaryAcknowledgement(
        pattern=license_policy.BUNDLED_RUNTIME_ACKNOWLEDGEMENTS[0].pattern,
        licence=license_policy.BUNDLED_RUNTIME_ACKNOWLEDGEMENTS[0].licence,
        note=license_policy.BUNDLED_RUNTIME_ACKNOWLEDGEMENTS[0].note,
    )
    assert entry.pattern.match("libgfortran-anything.so.5.0.0")
    assert not entry.pattern.match("libpedalboard_dsp-anything.so.1.0.0")


# ---------------------------------------------------------------------------
# Gap (c): dev-group dependencies. This project's OWN pyproject.toml declares a
# PEP 735 `[dependency-groups]` `dev` group (pytest, ruff) -- the exact shape a
# denylisted package could hide in without appearing in `dependencies` or
# `optional-dependencies` at all.
# ---------------------------------------------------------------------------


def test_pattern3_dependency_as_direct_main_dependency_is_caught():
    """Transcript 1 of 3 (gap (c)): a DIRECT main dependency."""
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = ["numpy>=2.0", "pedalboard>=0.9"]
"""
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert len(violations) == 1
    name, entry = violations[0]
    assert name == "pedalboard"
    assert entry.root == "pedalboard"


def test_pattern3_dependency_as_optional_extra_is_caught_transcript_2_of_3():
    """Transcript 2 of 3 (gap (c)): an OPTIONAL EXTRA -- same assertion shape as
    `test_pattern3_dependency_hidden_in_optional_extra_is_caught` above, named
    and kept separate per the task's instruction not to generalise the claim."""
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = []

[project.optional-dependencies]
evil = ["pedalboard>=0.9"]
"""
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert len(violations) == 1
    name, entry = violations[0]
    assert name == "pedalboard"
    assert entry.root == "pedalboard"


def test_pattern3_dependency_hidden_in_dev_group_is_caught_transcript_3_of_3():
    """Transcript 3 of 3 (gap (c)): a PEP 735 `[dependency-groups]` DEV-GROUP
    dependency -- this project's own `dev` group (pytest, ruff) is exactly this
    shape (pyproject.toml), and until `iter_declared_dependency_specs` read
    `[dependency-groups]`, a denylisted package placed there was invisible to
    both `scan_declared_dependencies_against_denylist` (this signal) AND
    `scan_installed_environment` (if the group were never installed in CI)."""
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = []

[dependency-groups]
dev = ["pytest>=8.3", "pedalboard>=0.9"]
"""
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert len(violations) == 1
    name, entry = violations[0]
    assert name == "pedalboard"
    assert entry.root == "pedalboard"


def test_dependency_groups_include_group_reference_does_not_crash_the_parser():
    """PEP 735 allows a group member to be a table (`{include-group = "..."}`)
    referencing another group, not just a plain requirement string. Confirm the
    parser tolerates that shape (skips it) rather than crashing or
    misclassifying it as a package name."""
    synthetic_pyproject = """
[project]
name = "demo"
version = "0.0.0"
dependencies = []

[dependency-groups]
base = ["numpy>=2.0"]
dev = [{include-group = "base"}, "pytest>=8.3"]
"""
    specs = iter_declared_dependency_specs(synthetic_pyproject)
    assert "numpy>=2.0" in specs
    assert "pytest>=8.3" in specs
    violations = scan_declared_dependencies_against_denylist(synthetic_pyproject)
    assert violations == []


def test_real_pyproject_dev_group_is_read_and_clean():
    """Guards the guard: this repo's REAL `[dependency-groups]` `dev` group
    (pytest, ruff) must actually be read (not silently skipped) and must not
    false-positive."""
    real_text = PYPROJECT_PATH.read_text(encoding="utf-8")
    specs = iter_declared_dependency_specs(real_text)
    assert any(_spec_name_starts_with(spec, "pytest") for spec in specs), (
        "the real pyproject.toml's [dependency-groups] dev entries were not read at all"
    )
    violations = scan_declared_dependencies_against_denylist(real_text)
    assert violations == []


def _spec_name_starts_with(spec: str, prefix: str) -> bool:
    return spec.strip().lower().startswith(prefix.lower())


# ---------------------------------------------------------------------------
# Gap (f): every failure names the offending package, its licence, and the
# governing document -- not just "FORBIDDEN" or "UNKNOWN" with no pointer back
# to where the policy and the enforcement gap are recorded.
# ---------------------------------------------------------------------------


def test_forbidden_verdict_reason_names_package_licence_and_governing_doc():
    info = DistInfo(
        name="pedalboard",
        license_expression=None,
        classifiers=("License :: OSI Approved :: GNU General Public License v3 (GPLv3)",),
        license_field=None,
    )
    verdict, reason = classify(info)
    assert verdict is Verdict.FORBIDDEN
    assert "pedalboard" in reason
    assert "GNU General Public License" in reason or "GPL" in reason
    assert "docs/DESIGN-ENVELOPE.md" in reason
    assert "smart_tools-c53" in reason


def test_unknown_verdict_reason_names_package_and_governing_doc():
    info = DistInfo(name="totally-unknown-package", license_expression=None, classifiers=(), license_field=None)
    verdict, reason = classify(info)
    assert verdict is Verdict.UNKNOWN
    assert "totally-unknown-package" in reason
    assert "docs/DESIGN-ENVELOPE.md" in reason


def test_denylisted_verdict_reason_names_package_licence_and_governing_doc():
    info = DistInfo(
        name="pyrubberband",
        license_expression=None,
        classifiers=("License :: OSI Approved :: ISC License (ISCL)",),
        license_field="ISC",
    )
    verdict, reason = classify(info)
    assert verdict is Verdict.FORBIDDEN
    assert "pyrubberband" in reason
    assert "rubberband" in reason.lower() or "Rubber Band" in reason
    assert "docs/DESIGN-ENVELOPE.md" in reason


def test_allowed_verdict_reason_is_not_cluttered_with_the_governing_doc_note():
    """The doc-citation suffix is for FAILURES. A passing (ALLOWED) verdict's
    reason should stay as-is -- confirms the note is appended conditionally,
    not unconditionally."""
    info = license_policy._distinfo_from_installed(distribution("pydantic"))
    verdict, reason = classify(info)
    assert verdict is Verdict.ALLOWED
    assert "docs/DESIGN-ENVELOPE.md" not in reason


# ---------------------------------------------------------------------------
# Gap (e), stated as its own top-level test (the individual signal tests above
# already each pass on the current tree, but the acceptance criterion is
# phrased as one clean-tree assertion -- this is that assertion, literally).
# ---------------------------------------------------------------------------


def test_the_current_clean_tree_passes_every_signal():
    """ "Given the CURRENT tree (numpy, scipy and their bundled
    libgfortran/libquadmath present) / When the check runs / Then it PASSES."
    A check that fails here is wrong, however safe failing feels."""
    env_results = scan_installed_environment()
    env_bad = [(i, v, r) for i, v, r in env_results if v is not Verdict.ALLOWED]
    assert env_bad == [], f"installed-environment signal failed on the clean tree: {env_bad}"

    declared_violations = scan_declared_dependencies_against_denylist(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert declared_violations == [], f"declared-dependency signal failed on the clean tree: {declared_violations}"

    import_violations = scan_source_tree(SRC_ROOT)
    assert import_violations == [], f"src/ AST-scan signal failed on the clean tree: {import_violations}"

    bundled_violations = scan_bundled_runtime_binaries()
    assert bundled_violations == [], f"bundled-runtime signal failed on the clean tree: {bundled_violations}"
