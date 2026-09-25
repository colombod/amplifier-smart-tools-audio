"""External-authority anchor for `aud.dsp.masking.spreading_function_peaq`:
a SEPARATE, independently-written transliteration of P. Kabal's own
published MATLAB (`PQspreadCB`/`PQ_SpreadCB`, "An Examination and
Interpretation of ITU-R BS.1387: Perceptual Evaluation of Audio Quality",
McGill University, Appendix F.4), not a copy of `masking.py`'s own code.

This is the "second implementation" anchor (AGENTS.md #3b / this task's
design note): both this test's oracle and the production module implement
the same published algorithm, but from the reference MATLAB directly and
independently -- structured as the paper's own nested loops, not as
`masking.py`'s vectorised-per-band-with-trailing-axes NumPy. If production
and this oracle agree to float64 precision, that is evidence the port is
faithful to the published algorithm, not evidence the test module agrees
with itself.

Kabal's MATLAB (reproduced in `masking.py`'s own docstring, and quoted here
for direct side-by-side reading):

    e = 0.4;
    aL = 10^(-2.7 * dz);
    for m = 0:Nc-1
        aUC = 10^((-2.4 - 23/fc(m+1)) * dz);
        aUCE = aUC * E(m+1)^(0.2*dz);
        gIL = (1 - aL^(m+1)) / (1 - aL);
        gIU = (1 - aUCE^(Nc-m)) / (1 - aUCE);
        En = E(m+1) / (gIL + gIU - 1);
        aUCEe(m+1) = aUCE^e;
        Ene(m+1) = En^e;
    end
    Es(Nc) = Ene(Nc);
    aLe = aL^e;
    for m = Nc-1:-1:1
        Es(m) = aLe * Es(m+1) + Ene(m);
    end
    for m = 1:Nc-1
        r = Ene(m); a = aUCEe(m);
        for i = m+1:Nc
            r = r*a;
            Es(i) = Es(i) + r;
        end
    end
    for i = 1:Nc
        Es(i) = (Es(i))^(1/e) / Bs(i);
    end
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp.bands import band_edges, hz_to_bark_peaq
from aud.dsp.masking import spreading_function_peaq

_PLAYBACK_LEVEL_DB_SPL = 92.0


def _kabal_pq_spread_cb(power: np.ndarray, fc: np.ndarray, dz: float, e: float = 0.4) -> np.ndarray:
    """Direct, independent 0-indexed Python transliteration of Kabal's
    `PQ_SpreadCB`, BEFORE the final `Bs` division (that division is applied
    by the caller, exactly mirroring how Kabal's own `PQspreadCB` wrapper
    calls `PQ_SpreadCB` twice -- once for the real signal, once cached for
    `ones(1, Nc)` -- and divides."""
    n_c = len(power)
    a_l = 10.0 ** (-2.7 * dz)
    ene = np.zeros(n_c)
    a_uce_e = np.zeros(n_c)
    for m in range(n_c):
        a_uc = 10.0 ** ((-2.4 - 23.0 / fc[m]) * dz)
        a_uce = a_uc * power[m] ** (0.2 * dz)
        g_il = (1.0 - a_l ** (m + 1)) / (1.0 - a_l)
        g_iu = (1.0 - a_uce ** (n_c - m)) / (1.0 - a_uce)
        en = power[m] / (g_il + g_iu - 1.0)
        a_uce_e[m] = a_uce**e
        ene[m] = en**e

    es = np.zeros(n_c)
    es[n_c - 1] = ene[n_c - 1]
    a_le = a_l**e
    for m in range(n_c - 2, -1, -1):
        es[m] = a_le * es[m + 1] + ene[m]

    for m in range(n_c - 1):
        r = ene[m]
        a = a_uce_e[m]
        for i in range(m + 1, n_c):
            r = r * a
            es[i] = es[i] + r

    return es


def _kabal_spread_normalised(power: np.ndarray, fc: np.ndarray, dz: float, e: float = 0.4) -> np.ndarray:
    """`PQspreadCB`'s own two-call-plus-divide wrapper, transliterated the
    same independent way."""
    es_raw = _kabal_pq_spread_cb(power, fc, dz, e)
    bs_raw = _kabal_pq_spread_cb(np.ones_like(power), fc, dz, e)
    bs = bs_raw ** (1.0 / e)
    return (es_raw ** (1.0 / e)) / bs


def _make_bands(n_bands: int) -> dict:
    return band_edges(n_bands, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)


def _dz_for(bands: dict) -> float:
    z = hz_to_bark_peaq(np.asarray(bands["centers_hz"]))
    return float(np.mean(np.diff(z)))


@pytest.mark.parametrize("n_bands", [8, 16, 32, 55])
@pytest.mark.parametrize(
    "profile_name",
    ["flat_low", "flat_high", "single_impulse", "two_maskers", "ramp", "random_seeded"],
)
def test_production_matches_independent_kabal_transliteration(n_bands, profile_name):
    """`spreading_function_peaq` must match `_kabal_spread_normalised` (an
    independently-written port of the standard's own published algorithm)
    to float64 precision, across several band counts and masker profiles --
    not just the single impulse case the acceptance criteria name."""
    bands = _make_bands(n_bands)
    fc = np.asarray(bands["centers_hz"])
    dz = _dz_for(bands)
    calibration = 10.0 ** (_PLAYBACK_LEVEL_DB_SPL / 10.0)

    rng = np.random.default_rng(20260925)
    if profile_name == "flat_low":
        normalised = np.full(n_bands, 1e-6)
    elif profile_name == "flat_high":
        normalised = np.full(n_bands, 0.5)
    elif profile_name == "single_impulse":
        normalised = np.full(n_bands, 1e-9)
        normalised[n_bands // 2] = 0.05
    elif profile_name == "two_maskers":
        normalised = np.full(n_bands, 1e-9)
        normalised[n_bands // 4] = 0.01
        normalised[3 * n_bands // 4] = 0.2
    elif profile_name == "ramp":
        normalised = np.linspace(1e-6, 1e-2, n_bands)
    else:  # random_seeded
        normalised = rng.uniform(1e-6, 1e-1, size=n_bands)

    calibrated = normalised * calibration

    expected = _kabal_spread_normalised(calibrated, fc, dz) / calibration
    actual = spreading_function_peaq(normalised, bands, playback_level_db_spl=_PLAYBACK_LEVEL_DB_SPL)

    max_rel_err = float(np.max(np.abs((actual - expected) / np.where(expected != 0, expected, 1.0))))
    print(f"\n[masking] n_bands={n_bands} profile={profile_name}: max rel err vs Kabal oracle = {max_rel_err:.3e}")
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-300)
