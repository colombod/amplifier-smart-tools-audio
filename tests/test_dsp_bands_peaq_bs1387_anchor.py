"""External-authority anchor for `hz_to_bark_peaq`/`bark_peaq_to_hz` against
ITU-R BS.1387 Tables 6 and 7.

Cites:
- ITU-R BS.1387-1 (11/2001), Annex 2, Sec 2.1.5, eq. (10), p. 36: `z/Bark =
  7*arsinh((f/Hz)/650)`, attributed to Schroeder, Atal & Hall, J. Acoust.
  Soc. Am., Vol. 66 (Dec 1979), p. 1647-1652. The same formula recurs as
  eq. (30) in Annex 2 Sec 2.2.5 (the filterbank-based ear model).
- ITU-R BS.1387-1, Annex 2, Sec 2.1.5, p. 36: Table 6 (Basic version, 109
  bands, res = 0.25 Bark) and Table 7 (Advanced version, 55 bands,
  res = 0.5 Bark), both spanning 80 Hz to 18 000 Hz.
- P. Kabal, "An Examination and Interpretation of ITU-R BS.1387:
  Perceptual Evaluation of Audio Quality", McGill University, Version 2
  (2003-12-08), Sec 2.6, eq. (9)-(12): the same construction, and its own
  statement "The band edges calculated using the procedure described
  above agree with the tabulated values in BS.1387 to within 0.003 Hz."

What this pins: `hz_to_bark_peaq`'s two constants, 650 and 7, against the
standard's own tabulated Hz values (Tables 6 and 7) -- an authority
external to this codebase. It does not validate any downstream consumer
of `bark_peaq` (e.g. `band_edges`, which partitions a scale evenly across
`[f_min, f_max]` and does not implement Sec 2.1.5's "all bands but the
last have the same width in Barks" truncation rule); the band-edge
construction below is test-local scaffolding that reproduces that rule
using the two production functions, solely to reach a value comparable to
the fixture.

Fixtures: `tests/fixtures/standards/bs1387_table{6,7}_*.csv`, each carrying
a provenance header (source, sha256 of the source PDF). Read from disk on
each parametrized case, never recomputed from this module's own formula.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from aud.dsp.bands import bark_peaq_to_hz, hz_to_bark_peaq

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "standards"

# Absolute Hz tolerance for the fixture comparison below. Measured this
# session: the baseline (unmutated) agreement between this module's
# `hz_to_bark_peaq`/`bark_peaq_to_hz` and both fixture tables is
# 0.002505 Hz (consistent with Kabal's own reported 0.003 Hz, and with the
# standard's tables being printed to 3 decimal places). The smallest of
# the four mutations checked in this session (650 -> 649.9) moves that
# error to 2.350 Hz. 0.01 Hz sits about 4x above the measured baseline
# noise and about 235x below the smallest tested mutation.
_TOLERANCE_HZ = 0.01


def _load_table(filename: str) -> dict[str, np.ndarray]:
    """Read a BS.1387 band-edge table fixture into column arrays.

    The fixture's first line is a `#`-prefixed provenance comment; the
    real header (`k,f_lower_hz,f_centre_hz,f_upper_hz,f_width_hz`) follows.
    """
    path = _FIXTURES_DIR / filename
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("k,"))
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    return {
        "f_lower_hz": np.array([float(r["f_lower_hz"]) for r in rows], dtype=np.float64),
        "f_centre_hz": np.array([float(r["f_centre_hz"]) for r in rows], dtype=np.float64),
        "f_upper_hz": np.array([float(r["f_upper_hz"]) for r in rows], dtype=np.float64),
    }


def _construct_bs1387_bands(
    resolution_bark: float, n_bands: int, f_lo_hz: float = 80.0, f_hi_hz: float = 18000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce BS.1387 Annex 2 Sec 2.1.5 eq. (11)-(12)'s band-edge
    construction (also Kabal Sec 2.6 eq. (9)-(12)), using the production
    `hz_to_bark_peaq`/`bark_peaq_to_hz` for every Hz<->Bark step:

        z_l[k] = B(f_lo) + res*k
        z_u[k] = min(B(f_lo) + res*(k+1), B(f_hi))   -- last band truncated
        z_c[k] = (B(f_l[k]) + B(f_u[k])) / 2          -- midpoint in Bark
        f_l/f_c/f_u = Binv(z_l/z_c/z_u)
    """
    k = np.arange(n_bands, dtype=np.float64)
    z_lo = float(hz_to_bark_peaq(np.array([f_lo_hz]))[0])
    z_hi_cap = float(hz_to_bark_peaq(np.array([f_hi_hz]))[0])

    z_l = z_lo + resolution_bark * k
    z_u = np.minimum(z_lo + resolution_bark * (k + 1.0), z_hi_cap)

    f_l = bark_peaq_to_hz(z_l)
    f_u = bark_peaq_to_hz(z_u)
    z_c = (hz_to_bark_peaq(f_l) + hz_to_bark_peaq(f_u)) / 2.0
    f_c = bark_peaq_to_hz(z_c)

    return f_l, f_c, f_u


@pytest.mark.parametrize(
    ("filename", "resolution_bark", "n_bands"),
    [
        ("bs1387_table6_basic_fft_bands_hz.csv", 0.25, 109),
        ("bs1387_table7_advanced_fft_bands_hz.csv", 0.5, 55),
    ],
)
def test_hz_to_bark_peaq_matches_bs1387_tabulated_band_edges(filename, resolution_bark, n_bands):
    """`hz_to_bark_peaq`/`bark_peaq_to_hz`, driven through BS.1387's own
    band-edge construction rule, reproduce the standard's own tabulated
    `f_lower`/`f_centre`/`f_upper` values (Table 6 or Table 7) within
    `_TOLERANCE_HZ` -- an external authority, never this module's own
    inverse checked against itself.
    """
    table = _load_table(filename)
    assert table["f_lower_hz"].shape == (n_bands,)

    f_l, f_c, f_u = _construct_bs1387_bands(resolution_bark, n_bands)

    err_l = np.abs(f_l - table["f_lower_hz"])
    err_c = np.abs(f_c - table["f_centre_hz"])
    err_u = np.abs(f_u - table["f_upper_hz"])
    max_err = float(max(err_l.max(), err_c.max(), err_u.max()))
    print(f"\n[bands] bark_peaq vs BS.1387 {filename}: max err = {max_err:.6f} Hz")

    assert max_err < _TOLERANCE_HZ


def test_bs1387_fixtures_have_expected_row_counts():
    """Guards against a truncated or mis-transcribed fixture silently
    shrinking the comparison above to fewer rows than the standard
    actually tabulates."""
    table6 = _load_table("bs1387_table6_basic_fft_bands_hz.csv")
    table7 = _load_table("bs1387_table7_advanced_fft_bands_hz.csv")
    assert table6["f_lower_hz"].shape == (109,)
    assert table7["f_lower_hz"].shape == (55,)
