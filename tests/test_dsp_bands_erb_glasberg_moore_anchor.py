"""External-authority anchor for `hz_to_erb_rate`/`erb_rate_to_hz`
(`erb_glasberg_moore`) against Jesteadt, Wroblewski & High (2019) Table I --
plus a literal re-evaluation of Glasberg & Moore (1990)'s own printed
equation, evaluated independently of this module.

Background (issue #31): Glasberg & Moore, "Derivation of auditory filter
shapes from notched-noise data", Hearing Research 47:103-138 (1990),
disagrees with itself about one constant:

- Eq. (4), printed text p.114: `ERBrate(f) = 21.4*log10(4.37*f/1000 + 1)`
  -- what this module ships (`hz_to_erb_rate`/`erb_rate_to_hz`).
- Eq. (3), printed text p.114: `ERB(f) = 24.7*(4.37*f/1000 + 1)` -- what
  this module ships (`erb_bandwidth_hz`).
- The paper's own Fortran appendix (pp.132, 135, 138): `c1 = 24.673`,
  `c2 = 4.368`, `c3 = 2302.6/(c1*c2) ~= 21.366` -- REJECTED. This codebase
  follows the paper's PRINTED equation (what a reader checking our
  docstring against the paper actually sees), not its reference
  implementation. Recorded here because it is the one Fortran-derived
  triple that agrees closely with Jesteadt 2019's own table (see below) --
  the road not taken, not a bug.
- Eq. (4)'s own first line, re-derived from ITS OWN rounded 24.7/4.37
  (2302.6/(24.7*4.37) ~= 21.33): REJECTED as an unpublished intermediate
  quantity, not something anyone actually ships.

USER RULING (2026-09-29): keep 21.4/4.37/24.7. No shipped constant
changes. This test file exists to anchor that choice against an authority
EXTERNAL to this codebase (not just this module's own forward/inverse
pair, which a consistent mutation of both would survive), and to make the
resulting, deliberately-accepted gap a MEASURED, pinned profile rather
than something a future reader has to rediscover and mistake for a bug.

Cites:
- Jesteadt, W., Wroblewski, M., & High, R. (2019). "Contribution of
  frequency bands to the loudness of broadband sounds: Tonal and noise
  stimuli." J. Acoust. Soc. Am. 145(6):3586-3594. doi:10.1121/1.5111751.
  Table I: 15 bands, each 2-ERB wide, with edges at the paper's own stated
  ERBN values of 3, 5, ..., 31, 33 Cams (their term for ERB-rate). Open
  access via PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6584171/.
  Accessed 2026-09-29.

Fixture: `tests/fixtures/standards/jesteadt2019_table1_erbn_band_edges_hz.csv`,
carrying its own provenance header (URL, table, access date, and the
averaging rule used for Table I's independently-rounded shared edges).
Read from disk, never recomputed from this module's own formula.

What this does NOT do: it does not change the shipped constants, and it
does not claim close agreement -- the whole point is that 21.4/4.37
disagrees with this external table by a growing amount, and that amount
is exactly what gets pinned.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from aud.dsp.bands import erb_rate_to_hz, hz_to_erb_rate

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "standards"
_FIXTURE_FILENAME = "jesteadt2019_table1_erbn_band_edges_hz.csv"


def _load_jesteadt_table1() -> tuple[np.ndarray, np.ndarray]:
    """Read the Jesteadt 2019 Table I fixture into (cams, f_hz) arrays.

    The fixture's first lines are `#`-prefixed provenance comments; the
    real header (`cams,f_hz`) follows.
    """
    path = _FIXTURES_DIR / _FIXTURE_FILENAME
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("cams,"))
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    cams = np.array([float(r["cams"]) for r in rows], dtype=np.float64)
    f_hz = np.array([float(r["f_hz"]) for r in rows], dtype=np.float64)
    return cams, f_hz


def test_jesteadt_fixture_has_expected_row_count_and_cams_values():
    """Guards against a truncated or mis-transcribed fixture silently
    shrinking the comparison below to fewer rows, or the wrong Cams
    values, than the paper actually tabulates."""
    cams, f_hz = _load_jesteadt_table1()
    assert cams.shape == (16,)
    assert f_hz.shape == (16,)
    assert list(cams) == [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0]


# The MEASURED deviation profile of `erb_rate_to_hz` (our shipped 21.4/4.37)
# against the Jesteadt 2019 Table I fixture, computed once and pinned here
# (not re-derived from the module under test). A consistent mutation of
# BOTH `hz_to_erb_rate` and `erb_rate_to_hz` cannot pass this test the way
# it would pass a self-referential round-trip check, because these are
# fixed external numbers.
_EXPECTED_DIFF_HZ = {
    3.0: 0.180434,
    5.0: -0.443637,
    7.0: -1.349628,
    9.0: -1.663317,
    11.0: -1.960209,
    13.0: -2.513375,
    15.0: -3.980723,
    17.0: -5.017195,
    19.0: -6.793849,
    21.0: -9.401470,
    23.0: -13.110983,
    25.0: -16.456291,
    27.0: -22.096904,
    29.0: -29.407471,
    31.0: -39.728660,
    33.0: -51.698074,
}


def test_erb_rate_to_hz_deviation_from_jesteadt2019_is_the_known_chosen_gap():
    """Pin the measured deviation profile of our shipped 21.4/4.37 against
    Jesteadt 2019 Table I -- the external anchor issue #31 required.

    We do NOT expect close agreement: keeping the paper's printed
    constants instead of its own Fortran-appendix constants (21.366/4.368)
    is the user's explicit ruling (see module docstring), and this table
    is what makes that choice's cost visible and stated rather than
    silently absorbed. Two independent, human-readable bounds make the
    shape of the gap legible without reading the per-row table:

    - Below ~1 kHz (Cams <= 13, i.e. up to 700.5 Hz): agrees within 3 Hz.
    - At the top of the table (33 Cams = 7795 Hz): off by -52 +/- 2 Hz,
      matching issue #31's own stated measurement ("-52 Hz at 33 Cams").
    """
    cams, f_hz = _load_jesteadt_table1()
    ours = np.asarray(erb_rate_to_hz(cams), dtype=np.float64)
    diff = ours - f_hz

    for c, d in zip(cams, diff, strict=True):
        expected = _EXPECTED_DIFF_HZ[float(c)]
        assert d == pytest.approx(expected, abs=0.01), f"deviation profile changed at {c} Cams"

    below_1khz = cams <= 13.0
    assert np.all(np.abs(diff[below_1khz]) < 3.0), "expected agreement within 3 Hz below ~1 kHz"

    top_diff = float(diff[cams == 33.0][0])
    assert top_diff == pytest.approx(-52.0, abs=2.0), "expected the known ~-52 Hz gap at 33 Cams"


def test_hz_to_erb_rate_matches_jesteadt2019_edges_within_the_same_gap():
    """The forward direction (Hz -> Cams) must show the same known gap as
    the inverse direction above, not a different one -- both directions
    are driven by the same two constants."""
    _cams, f_hz = _load_jesteadt_table1()
    computed_cams = np.asarray(hz_to_erb_rate(f_hz), dtype=np.float64)
    # At 33 Cams our constants place 7795 Hz at a HIGHER ERB-rate than 33
    # (since our f(33 Cams) undershoots 7795 Hz, the converse -- our
    # ERB-rate AT 7795 Hz -- overshoots 33). The known top-end gap of
    # about -52 Hz in the inverse direction corresponds to roughly
    # +0.06 Cams of forward-direction disagreement at the top of the
    # table (measured: computed_cams[-1] ~= 33.060; the ERB-rate scale is
    # compressive, so a fixed Hz gap maps to a shrinking Cams gap as
    # frequency rises). At the bottom (3 Cams = 87 Hz) the two directions
    # agree closely (measured: computed_cams[0] ~= 2.9947).
    assert computed_cams[0] == pytest.approx(3.0, abs=0.01)
    assert computed_cams[-1] == pytest.approx(33.0, abs=0.1)


def test_printed_equation_anchor_literal_constants_not_imported_from_module():
    """Anchor `hz_to_erb_rate` against Glasberg & Moore (1990) eq. (4)'s
    PRINTED text (p.114), evaluated here with LITERAL constants -- 21.4
    and 4.37 written directly in this test, never imported from
    `aud.dsp.bands` -- so a typo that silently diverged the module's
    implementation from the paper's printed equation would be caught even
    though this test has no external table to compare against at these
    particular frequencies.

        ERBrate(f) = 21.4 * log10(4.37*f/1000 + 1)      -- eq. (4), p.114
    """
    for f in (0.0, 100.0, 1000.0, 4000.0, 10000.0, 20000.0):
        literal = 21.4 * math.log10(4.37 * f / 1000.0 + 1.0)
        module_value = float(np.asarray(hz_to_erb_rate(f)))
        assert module_value == pytest.approx(literal, abs=1e-12), f"diverges from the printed equation at f={f}"
