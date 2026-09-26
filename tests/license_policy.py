"""MIT-compatible dependency enforcement: the reusable logic behind
`tests/test_license_enforcement.py`.

`aud` is MIT (AGENTS.md section 1, docs/VISION.md). Until this module existed, that
was a documented human audit with nothing enforcing it -- a licence audit that is
not a check is a comment, the same lesson `test_design_envelope_enforcement.py`
already paid for with patented-formulation vocabulary. This module is the check.

Two independent signals are combined, on purpose (belt-and-braces, per AGENTS.md):

1. **The ACTUAL licence metadata of the resolved, installed distribution graph**
   (`importlib.metadata.distributions()`) -- the general rule. This is what catches
   a GPL-family package NOBODY has thought to denylist yet (evasion pattern 4:
   a permissive-looking package that itself pulls in a GPL one shows up as its own
   installed distribution, flat, regardless of dependency depth).

   Chosen over shelling out to `uv tree` / `uv pip list`: `importlib.metadata` is
   stdlib (no subprocess, no dependency on `uv`'s CLI output format staying stable
   across versions), and it returns STRUCTURED metadata -- trove `Classifier` lines
   and, on modern packaging, a `License-Expression` (SPDX) header -- not just a
   name/version pair. `uv tree`'s text output would need a second parser for
   exactly the same information this already gives directly.

2. **A measured, named denylist** (`DENYLIST` below) as a backstop for packages
   with missing or LYING metadata -- see `pyrubberband`'s recorded case below, a
   real, load-bearing example of why the backstop cannot be "when metadata is
   silent" alone: `pyrubberband`'s OWN metadata is genuinely, honestly ISC
   (permissive) -- it is a thin ctypes wrapper; the licence problem is the
   GPL/commercial-dual Rubber Band C++ library it calls out to at runtime, which
   never appears as its own installed Python distribution at all. Metadata-only
   would ALLOW it. The name backstop is what forbids it, unconditionally, not only
   when metadata is absent.

Verified 2026-09-26 against PyPI's JSON API (`https://pypi.org/pypi/<name>/json`)
for every recorded fixture cited below; each citation says exactly what was
checked. Per this repo's standing rule (AGENTS.md 3b): these are REAL recorded
distribution metadata, transcribed verbatim from the fields actually observed --
never a hand-built fake shaped to confirm what the reader already assumes.

A third, independent layer -- `scan_source_text`/`scan_source_tree` -- AST-parses
`src/` for a forbidden import at ANY nesting depth (module top level or inside a
function body) and for a literal string argument to `importlib.import_module`/
`__import__`. This is a separate defence: it catches a forbidden package pulled in
without ever being declared as a dependency at all.

A fourth -- `scan_declared_dependencies_against_denylist` -- reads `pyproject.toml`
statically (main dependencies, every optional-dependencies extra, AND every
PEP 735 `[dependency-groups]` group such as `dev`) so a denylisted package hidden
in an extra or a dev-only group is caught even when that group is not currently
installed and so never shows up in (1).

A fifth -- `scan_bundled_runtime_binaries` -- walks `numpy.libs/`/`scipy.libs/`
(the auditwheel-bundled shared-library directories that sit alongside the
`numpy`/`scipy` packages themselves) and checks every file there against
`BUNDLED_RUNTIME_ACKNOWLEDGEMENTS`, a separate, explicitly-enumerated, reviewed
list -- see docs/DESIGN-ENVELOPE.md's "Dependency licences" section for the
per-binary licence and MIT-compatibility argument for each entry on that list.
These binaries never appear in `importlib.metadata` at all (they are not
themselves installed distributions), so signal (1) cannot see them; this is a
sixth, independent signal for exactly that gap. A binary found in either
directory that matches none of the acknowledged patterns is a **new, unreviewed
bundled runtime** and fails the check unconditionally -- the acknowledgement
list is an enumerated decision, never a blanket exemption for those two
directories.

## The sixth evasion this cannot catch

Every signal here is NAME-based (an import name, an installed distribution name, a
declared dependency name) or METADATA-based (what the package's own `PKG-INFO`
claims). None of them is CONTENT-based. Vendoring GPL source directly into a new
module under a name that is not on any denylist -- or repointing a dependency's
name to different upstream content via a `[tool.uv.sources]` git override or a
renamed fork -- defeats every check in this file simultaneously, because all four
signals key off a name that the vendoring/renaming step controls. Catching that
needs content/copyright-header fingerprinting (e.g. `scancode-toolkit`,
`licensee`), which this module does not attempt.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import Distribution, PackageNotFoundError, distribution, distributions
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class Verdict(Enum):
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    # Missing or ambiguous metadata is NOT a pass. "An unknown licence is not a
    # permissive licence" -- the task's own words, and the whole point of using
    # metadata as the general rule instead of a denylist alone.
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# The measured denylist backstop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenylistEntry:
    root: str  # canonical, already-normalized name fragment (see _normalize_pkg_name)
    reason: str


# Each `root` is matched as a substring of the NORMALIZED candidate name (lowercased,
# non-alphanumeric characters stripped) in both directions of nothing needing to be
# exact -- "pyrubberband", "rubberband-ctypes", "Rubber Band" and "rubberband_ctypes"
# all normalize to a string containing "rubberband".
DENYLIST: tuple[DenylistEntry, ...] = (
    DenylistEntry(
        "pedalboard",
        "GPL-3.0 (Spotify Pedalboard) -- forbidden by AGENTS.md's licence rule. Confirmed "
        "2026-09-26 via https://pypi.org/pypi/pedalboard/json: classifiers include "
        "'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'.",
    ),
    DenylistEntry(
        "matchering",
        "GPL-3.0 (reference-matching mastering) -- forbidden by AGENTS.md's licence rule. "
        "Confirmed 2026-09-26 via https://pypi.org/pypi/matchering/json: classifiers include "
        "'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'.",
    ),
    DenylistEntry(
        "rubberband",
        "GPL-2.0-or-later / commercial dual (the Rubber Band Library and every binding onto "
        "it: pyrubberband, rubberband-ctypes, rubberband-cli, ...) -- forbidden by AGENTS.md's "
        "licence rule. IMPORTANT, recorded 2026-09-26 via "
        "https://pypi.org/pypi/pyrubberband/json: pyrubberband's OWN metadata is genuinely "
        "'License :: OSI Approved :: ISC License (ISCL)' (license='ISC') -- a metadata-only "
        "check would ALLOW it. The wrapper's own licence is honestly permissive; what it wraps "
        "at runtime is not. This is exactly the case the name backstop exists for, checked "
        "unconditionally rather than only when metadata is missing.",
    ),
    DenylistEntry(
        "essentia",
        "AGPL-3.0-only -- forbidden by AGENTS.md's licence rule. Confirmed 2026-09-26 via "
        "https://pypi.org/pypi/essentia/json: license_expression='AGPL-3.0-only' (essentia "
        "ships no legacy 'License ::' classifiers at all -- the SPDX License-Expression header "
        "is the ONLY metadata signal for this one, which is why this module checks it).",
    ),
    DenylistEntry(
        "librosa",
        "Kept off this project's dependency stack per explicit project instruction (the "
        "permissive numpy/scipy/soundfile/pyloudnorm stack documented in docs/VISION.md does "
        "not include it) -- NOT because librosa's own licence is non-permissive. Checked "
        "2026-09-26: librosa is published under the ISC licence (permissive) per its PyPI "
        "classifiers. Flagged here as a discrepancy: if this entry is ever revisited, that is "
        "a dependency-stack policy conversation, not a licence-compliance one.",
    ),
)


def _normalize_pkg_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def denylist_match(name: str) -> DenylistEntry | None:
    """Returns the matching DenylistEntry, spelling/case/hyphenation-insensitive, or None."""
    normalized = _normalize_pkg_name(name)
    if not normalized:
        return None
    for entry in DENYLIST:
        if entry.root in normalized:
            return entry
    return None


# ---------------------------------------------------------------------------
# Metadata classification -- the general rule
# ---------------------------------------------------------------------------

# Matches GPL, LGPL, AGPL (via "Affero" and via the "General Public License" full
# name that every GPL-family trove classifier spells out), plus the other named
# copyleft families the task's ruling covers ("and similar copyleft licences").
_FORBIDDEN_RE = re.compile(
    r"\bAGPL\b|\bGPL\b|\bLGPL\b|General Public License|Affero"
    r"|Mozilla Public License|Eclipse Public License"
    r"|Common Development and Distribution License|Common Public License",
    re.IGNORECASE,
)

# Trove `License ::` classifiers that correspond to an entry on `_ALLOWED_SPDX_IDS`
# below. Deliberately NOT a superset of "permissive-sounding" classifiers: e.g.
# "Public Domain", "Universal Permissive License", "Zope Public License" and "W3C
# License" were removed because none of them is the specified allow-list -- see
# gap (d), "THE ALLOW-LIST IS SPECIFIED, use exactly it. ANYTHING ELSE FAILS,
# INCLUDING UNKNOWN." A trove classifier with no corresponding SPDX id on the list
# is intentionally left unmatched here and falls through to UNKNOWN.
_ALLOWED_CLASSIFIER_SUBSTRINGS = (
    "MIT License",  # MIT
    "BSD License",  # BSD-2-Clause / BSD-3-Clause (trove does not distinguish)
    "ISC License",  # ISC
    "Apache Software License",  # Apache-2.0
    "Python Software Foundation License",  # PSF-2.0
    "The Unlicense",  # Unlicense
    "zlib/libpng License",  # Zlib
)

# SPDX identifiers recognised as permissive when found in a modern `License-Expression`
# header (PEP 639 / Metadata 2.4) or as a short, exact `License` field value.
#
# This is the EXACT allow-list from the governing work item (smart_tools-c53) and
# docs/DESIGN-ENVELOPE.md's "Dependency licences" section -- not a superset, not a
# subset. Anything else classifies UNKNOWN, which fails the build just like FORBIDDEN
# does (see `classify()` below and `test_installed_environment_has_no_forbidden_or_unknown_licences`).
_ALLOWED_SPDX_IDS = frozenset(
    {
        "MIT",
        "MIT-0",
        "BSD-2-CLAUSE",
        "BSD-3-CLAUSE",
        "0BSD",
        "ISC",
        "APACHE-2.0",
        "PSF-2.0",
        "PYTHON-2.0",
        "ZLIB",
        "UNLICENSE",
        "CC0-1.0",
        "HPND",
    }
)

_LICENSE_FIELD_MAX_LEN_FOR_SHORT_MATCH = 200


def _classify_spdx_expression(expr: str) -> Verdict:
    if _FORBIDDEN_RE.search(expr):
        return Verdict.FORBIDDEN
    tokens = [
        t.strip().upper() for t in re.split(r"\s+(?:OR|AND|WITH)\s+|[()]", expr, flags=re.IGNORECASE) if t.strip()
    ]
    if tokens and all(t in _ALLOWED_SPDX_IDS for t in tokens):
        return Verdict.ALLOWED
    return Verdict.UNKNOWN


def _classify_classifiers(classifiers: list[str]) -> Verdict | None:
    """Returns None (no signal) if there are no `License ::` classifiers at all --
    the caller falls back to the legacy `License` field in that case."""
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    if not license_classifiers:
        return None
    if any(_FORBIDDEN_RE.search(c) for c in license_classifiers):
        return Verdict.FORBIDDEN
    if any(any(sub in c for sub in _ALLOWED_CLASSIFIER_SUBSTRINGS) for c in license_classifiers):
        return Verdict.ALLOWED
    return Verdict.UNKNOWN


def _classify_license_field(text: str) -> Verdict:
    """Legacy free-text `License` field, used only when there are no classifiers
    and no `License-Expression`.

    A long blob is deliberately NOT keyword-matched: scipy's own `License` field
    is its real BSD licence text FOLLOWED BY bundled third-party notices for
    OpenBLAS/LAPACK (BSD) and the GCC runtime (libgfortran/libquadmath), and the
    latter's text literally contains 'GPL-3.0-or-later' and 'LGPL-2.1-or-later'
    -- under the GCC Runtime Library Exception, which does not extend the GPL to
    anything scipy itself. A naive substring search over the full field would
    misclassify a package this project's own AGENTS.md explicitly permits.
    Verified 2026-09-26 against this exact installed scipy distribution -- see
    `tests/test_license_enforcement.py::test_scipy_bundled_gpl_notice_is_not_a_false_positive`.
    scipy never reaches this function in practice (it has classifiers), which is
    exactly the point: classifiers are checked FIRST, before this fallback ever
    looks at the long blob.
    """
    text = text.strip()
    if not text:
        return Verdict.UNKNOWN
    if len(text) > _LICENSE_FIELD_MAX_LEN_FOR_SHORT_MATCH:
        return Verdict.UNKNOWN
    if _FORBIDDEN_RE.search(text):
        return Verdict.FORBIDDEN
    upper = text.upper()
    if any(tok in upper for tok in ("MIT", "BSD", "ISC", "APACHE", "PSF", "PUBLIC DOMAIN", "UNLICENSE")):
        return Verdict.ALLOWED
    return Verdict.UNKNOWN


@dataclass(frozen=True)
class DistInfo:
    name: str
    license_expression: str | None
    classifiers: tuple[str, ...]
    license_field: str | None


def _distinfo_from_installed(dist: Distribution) -> DistInfo:
    meta = dist.metadata
    classifiers = tuple(meta.get_all("Classifier") or ())
    return DistInfo(
        name=meta["Name"],
        license_expression=meta.get("License-Expression"),
        classifiers=classifiers,
        license_field=meta.get("License"),
    )


#: Cited in every non-ALLOWED verdict's reason string (gap (f): "Each failure must
#: NAME the offending package, its licence, AND the governing document") -- the
#: package name and licence are already embedded earlier in the reason text; this
#: constant supplies the document.
_GOVERNING_DOC_NOTE = (
    "governing document: docs/DESIGN-ENVELOPE.md 'Dependency licences' section "
    "(policy: AGENTS.md section 1; tracked as work item smart_tools-c53)"
)


def classify(info: DistInfo) -> tuple[Verdict, str]:
    """Combines the metadata general rule with the denylist backstop. The
    denylist is checked UNCONDITIONALLY -- even when the metadata rule alone
    would say ALLOWED (see `pyrubberband` in DENYLIST's docstring above)."""
    if info.license_expression:
        verdict = _classify_spdx_expression(info.license_expression)
        reason = f"License-Expression metadata: {info.license_expression!r}"
    else:
        clf_verdict = _classify_classifiers(list(info.classifiers))
        if clf_verdict is not None:
            verdict = clf_verdict
            reason = f"trove classifiers: {[c for c in info.classifiers if c.startswith('License ::')]!r}"
        elif info.license_field:
            verdict = _classify_license_field(info.license_field)
            reason = "legacy License metadata field (no classifiers, no License-Expression present)"
        else:
            verdict = Verdict.UNKNOWN
            reason = "no licence metadata at all (no License-Expression, no classifiers, no License field)"

    denylisted = denylist_match(info.name)
    if denylisted is not None and verdict is not Verdict.FORBIDDEN:
        reason = f"denylisted despite metadata verdict {verdict.value}: {denylisted.reason}"
        verdict = Verdict.FORBIDDEN

    if verdict is not Verdict.ALLOWED:
        reason = f"package {info.name!r}: {reason} -- {_GOVERNING_DOC_NOTE}"

    return verdict, reason


def scan_installed_environment() -> list[tuple[DistInfo, Verdict, str]]:
    """Every distribution actually resolved and installed in THIS environment --
    direct or transitive, flat, regardless of depth. This is the resolved
    dependency tree; see this module's docstring for why `importlib.metadata`
    was chosen over shelling out to `uv tree`."""
    results = []
    for dist in distributions():
        info = _distinfo_from_installed(dist)
        verdict, reason = classify(info)
        results.append((info, verdict, reason))
    return results


# ---------------------------------------------------------------------------
# Declared dependencies -- main dependencies, every optional-dependencies extra,
# AND every PEP 735 `[dependency-groups]` group (e.g. `dev`) -- read statically so
# a denylisted package hidden in ANY of the three is caught even when it is not
# currently installed. This repo's own `dev` group (pytest, ruff) is exactly the
# PEP 735 shape being read here -- see pyproject.toml's `[dependency-groups]`.
# ---------------------------------------------------------------------------

_SPEC_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+")


def _spec_to_name(spec: str) -> str:
    match = _SPEC_NAME_RE.match(spec.strip())
    return match.group(0) if match else spec.strip()


def iter_declared_dependency_specs(pyproject_text: str) -> list[str]:
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    for _extra_name, extra_deps in project.get("optional-dependencies", {}).items():
        specs.extend(extra_deps)
    for _group_name, group_deps in data.get("dependency-groups", {}).items():
        for dep in group_deps:
            # PEP 735 allows a group member to be a plain requirement string OR a
            # `{include-group = "other-group"}` table referencing another group.
            # The latter carries no package name of its own -- nothing to check
            # here directly; the group it points at is scanned on its own
            # iteration of this same loop (or, if external, is out of scope for
            # a static single-file read). Only plain string specs are collected.
            if isinstance(dep, str):
                specs.append(dep)
    return specs


def scan_declared_dependencies_against_denylist(pyproject_text: str) -> list[tuple[str, DenylistEntry]]:
    violations = []
    for spec in iter_declared_dependency_specs(pyproject_text):
        name = _spec_to_name(spec)
        entry = denylist_match(name)
        if entry is not None:
            violations.append((name, entry))
    return violations


# ---------------------------------------------------------------------------
# Source-level AST scan -- catches a forbidden import regardless of whether it
# is declared as a dependency at all, at ANY nesting depth (pattern 1), via
# importlib.import_module/__import__ with a literal string argument (pattern 2),
# and under any spelling variant (pattern 5, via denylist_match's normalization).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportViolation:
    relpath: str
    lineno: int
    imported_name: str
    denylist_entry: DenylistEntry
    pattern: str  # "import" | "from-import" | "importlib.import_module" | "__import__"


def _iter_ast_import_events(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Yields (lineno, top_level_name, pattern) for every import-shaped construct
    ANYWHERE in the tree. `ast.walk` visits nodes regardless of nesting depth --
    a `def`-body import is visited exactly like a module-level one, which is what
    defeats evasion pattern 1 without needing separate top-level/nested logic."""
    events: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                events.append((node.lineno, alias.name.split(".")[0], "import"))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                events.append((node.lineno, node.module.split(".")[0], "from-import"))
        elif isinstance(node, ast.Call):
            func = node.func
            is_import_module = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
                isinstance(func, ast.Name) and func.id == "import_module"
            )
            is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
            if (is_import_module or is_dunder_import) and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    pattern = "importlib.import_module" if is_import_module else "__import__"
                    events.append((node.lineno, first_arg.value.split(".")[0], pattern))
    return events


def scan_source_text(text: str, relpath: str) -> list[ImportViolation]:
    tree = ast.parse(text, filename=relpath)
    violations = []
    for lineno, top_name, pattern in _iter_ast_import_events(tree):
        entry = denylist_match(top_name)
        if entry is not None:
            violations.append(ImportViolation(relpath, lineno, top_name, entry, pattern))
    return violations


def scan_source_tree(root: Path) -> list[ImportViolation]:
    violations: list[ImportViolation] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            violations.extend(scan_source_text(text, str(path.relative_to(REPO_ROOT))))
        except SyntaxError as exc:
            raise AssertionError(f"{path}: could not parse for licence enforcement: {exc}") from exc
    return violations


# ---------------------------------------------------------------------------
# Bundled copyleft runtime binaries -- present on disk in numpy.libs/ and
# scipy.libs/, never visible in any importlib.metadata field (they are not
# themselves installed distributions). This is a SEPARATE, explicitly-enumerated,
# reviewed acknowledgement list -- not a blanket exemption for those directories.
# See docs/DESIGN-ENVELOPE.md's "Dependency licences" section for the licence and
# MIT-compatibility argument recorded for each entry below; do not add an entry
# here without adding the matching argument there.
# ---------------------------------------------------------------------------

#: Packages whose sibling `<name>.libs/` directory (the auditwheel/delvewheel
#: convention for a wheel's bundled shared libraries) is scanned.
BUNDLED_RUNTIME_PACKAGE_NAMES: tuple[str, ...] = ("numpy", "scipy")


@dataclass(frozen=True)
class BundledBinaryAcknowledgement:
    pattern: re.Pattern[str]
    licence: str
    note: str


# Filenames carry a build-specific content hash suffix appended by auditwheel
# (e.g. `libgfortran-040039e1-0352e75f.so.5.0.0`), so each pattern matches the
# stable prefix, not an exact filename -- confirmed against the real filenames
# in THIS environment's numpy.libs/ and scipy.libs/ (see
# test_bundled_runtime_binaries_currently_on_disk_are_all_acknowledged).
BUNDLED_RUNTIME_ACKNOWLEDGEMENTS: tuple[BundledBinaryAcknowledgement, ...] = (
    BundledBinaryAcknowledgement(
        re.compile(r"^libgfortran-.*\.so(\.\d+)*$"),
        "GPL-3.0-or-later WITH GCC-exception-3.1",
        "The GCC Runtime Library Exception exists precisely to permit GCC runtime "
        "libraries to be carried into a program under ANY licence, including "
        "proprietary, without propagating GPL terms to it -- MIT-compatible as used. "
        "docs/DESIGN-ENVELOPE.md 'Dependency licences'.",
    ),
    BundledBinaryAcknowledgement(
        re.compile(r"^libquadmath-.*\.so(\.\d+)*$"),
        "LGPL-2.1-or-later",
        "Dynamic linking under LGPL-2.1 does not relicense the linking program; this "
        "project does not redistribute the binary (pip/uv fetch numpy/scipy wheels "
        "directly from PyPI), so LGPL section 6's notice/relink duties sit with "
        "numpy/scipy, not aud. Does NOT carry the GCC Runtime Library Exception -- "
        "the weaker of the two arguments on this list, recorded as such. "
        "docs/DESIGN-ENVELOPE.md 'Dependency licences'.",
    ),
    BundledBinaryAcknowledgement(
        re.compile(r"^libscipy_openblas(64_)?-.*\.so$"),
        "BSD-3-Clause",
        "Permissive outright. docs/DESIGN-ENVELOPE.md 'Dependency licences'.",
    ),
)


@dataclass(frozen=True)
class UnacknowledgedBundledBinary:
    path: str
    reason: str


def _scan_libs_dir(libs_dir: Path) -> list[UnacknowledgedBundledBinary]:
    """Testable in isolation against a synthetic directory, so the "fails on a
    new unlisted binary" acceptance criterion can be proved without needing a
    real new copyleft library to actually appear in a real wheel."""
    violations = []
    if not libs_dir.is_dir():
        return violations
    for entry in sorted(libs_dir.iterdir()):
        if not entry.is_file():
            continue
        if any(ack.pattern.match(entry.name) for ack in BUNDLED_RUNTIME_ACKNOWLEDGEMENTS):
            continue
        violations.append(
            UnacknowledgedBundledBinary(
                path=str(entry),
                reason=(
                    f"{entry.name!r} in {libs_dir} is not on the reviewed bundled-runtime "
                    "acknowledgement list (tests/license_policy.py BUNDLED_RUNTIME_ACKNOWLEDGEMENTS) "
                    "-- a new bundled binary must be reviewed and explicitly enumerated with its "
                    "licence and MIT-compatibility argument, never silently passed. "
                    f"{_GOVERNING_DOC_NOTE}"
                ),
            )
        )
    return violations


def _libs_dirs_for(names: tuple[str, ...]) -> list[Path]:
    dirs = []
    for name in names:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        site_root = Path(str(dist.locate_file("")))
        libs_dir = site_root / f"{name}.libs"
        if libs_dir.is_dir():
            dirs.append(libs_dir)
    return dirs


def scan_bundled_runtime_binaries(
    names: tuple[str, ...] = BUNDLED_RUNTIME_PACKAGE_NAMES,
) -> list[UnacknowledgedBundledBinary]:
    """Every file under `<name>.libs/` for each package in `names`, checked
    against `BUNDLED_RUNTIME_ACKNOWLEDGEMENTS`. A package with no `.libs/`
    directory (not installed, or a build with no bundled binaries) contributes
    nothing -- that is not a failure."""
    violations: list[UnacknowledgedBundledBinary] = []
    for libs_dir in _libs_dirs_for(names):
        violations.extend(_scan_libs_dir(libs_dir))
    return violations


# ---------------------------------------------------------------------------
# Standalone report -- `uv run python -m tests.license_policy`
# ---------------------------------------------------------------------------


def _format_verdict_line(info: DistInfo, verdict: Verdict, reason: str) -> str:
    return f"  {info.name:<30} {verdict.value.upper():<10} {reason}"


def main() -> int:
    print("== Installed / resolved distribution graph ==")
    results = scan_installed_environment()
    bad = [(i, v, r) for i, v, r in results if v is not Verdict.ALLOWED]
    for info, verdict, reason in sorted(results, key=lambda t: t[0].name.lower()):
        print(_format_verdict_line(info, verdict, reason))
    print(f"\nExamined {len(results)} installed distribution(s); {len(bad)} not ALLOWED.\n")

    print("== Declared dependencies (main + every optional extra) vs. denylist ==")
    declared_violations = scan_declared_dependencies_against_denylist(PYPROJECT_PATH.read_text(encoding="utf-8"))
    if declared_violations:
        for name, entry in declared_violations:
            print(f"  {name}: {entry.reason}")
    else:
        print("  none")

    print("\n== src/ AST import scan ==")
    import_violations = scan_source_tree(SRC_ROOT)
    if import_violations:
        for v in import_violations:
            print(f"  {v.relpath}:{v.lineno}: {v.pattern} of {v.imported_name!r} -- {v.denylist_entry.reason}")
    else:
        print("  none")

    print("\n== Bundled runtime binaries (numpy.libs/, scipy.libs/) vs. acknowledgement list ==")
    bundled_violations = scan_bundled_runtime_binaries()
    if bundled_violations:
        for v in bundled_violations:
            print(f"  {v.path}: {v.reason}")
    else:
        print("  none (every bundled binary present matches an acknowledged pattern)")

    return 1 if (bad or declared_violations or import_violations or bundled_violations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
