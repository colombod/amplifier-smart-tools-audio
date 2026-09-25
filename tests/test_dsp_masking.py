"""Behavioural tests for `aud.dsp.masking`: the PEAQ spreading function and
the Terhardt absolute-threshold-of-hearing floor.

Every expectation here either (a) comes from a source this module's own
implementation cannot reach -- a published formula's own worked checkpoints
(Painter & Spanias 2000), an independently-measured empirical dataset
(ISO 226:2003 Table 1), or a *second*, independently-written transliteration
of the standard's own published algorithm (`test_dsp_masking_kabal_
reference.py`) -- or (b) is a structural/mathematical invariant provable
directly from the construction (e.g. "spreading a uniform excitation at the
exact reference level used to define the normaliser returns that same
uniform excitation, to float64 precision"), not a re-statement of this
module's own formula text. See `AGENTS.md` #3b and this module's own
docstring for why that distinction matters here specifically -- a real bug
(the level-dependent exponent divided by an extra, erroneous factor of 10)
was caught during this session by the independent-oracle test, not by any
test that re-derived the same formula this module implements.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from aud.dsp.bands import band_edges, hz_to_bark_peaq
from aud.dsp.masking import masking_threshold, spreading_function_peaq, threshold_in_quiet

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "standards"


def _peaq_bands(n_bands: int = 32, f_min: float = 20.0, f_max: float = 20000.0) -> dict:
    return band_edges(n_bands, scale="bark_peaq", f_min=f_min, f_max=f_max, allow_extrapolation=True)


# --- threshold_in_quiet: checkpoints, external cross-check, validation ---


@pytest.mark.parametrize(
    ("f_hz", "expected_db", "tol"),
    [
        # Worked checkpoints independently re-derived (this session) from
        # Painter & Spanias 2000 Sec. II.A eq. (1) (Proc. IEEE 88(4), citing
        # Terhardt 1979) -- see module docstring's citation chain. These are
        # NOT copied from this module's own docstring; they were computed
        # by hand from the published equation before this module's
        # implementation was checked against them.
        (20.0, 83.2, 0.5),
        (1000.0, 3.4, 0.1),
        (3300.0, -5.0, 0.1),
        (10000.0, 10.6, 0.1),
        (16000.0, 65.9, 0.2),
        (20000.0, 160.3, 0.2),
        # Two additional points off the resonance dip's own centre and
        # inflection, at a TIGHT tolerance: 3300 Hz alone under-constrains
        # the resonance term's centre-frequency and width constants (a
        # +/-0.1 kHz shift in the centre, or a +/-0.1 shift in the width,
        # moves Tq(3300) by under 0.04 dB -- MEASURED, by mutating each and
        # re-running, not assumed -- because 3300 Hz sits exactly at the
        # dip where both mutations' first-order effect vanishes). At 2000
        # and 4000 Hz the same two mutations move Tq by 0.25-0.39 dB, so a
        # 0.05 dB tolerance here actually constrains both constants.
        (2000.0, -0.2513, 0.05),
        (4000.0, -3.3875, 0.05),
    ],
)
def test_threshold_in_quiet_matches_published_checkpoints(f_hz, expected_db, tol):
    got = float(threshold_in_quiet(np.array([f_hz]))[0])
    print(f"\n[masking] Tq({f_hz} Hz) = {got:.4f} dB SPL (published checkpoint ~{expected_db})")
    assert got == pytest.approx(expected_db, abs=tol)


def _load_iso226_table() -> tuple[np.ndarray, np.ndarray]:
    path = _FIXTURES_DIR / "iso226_2003_table1_threshold_hz_db.csv"
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("f_hz"))
    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    f_hz = np.array([float(r["f_hz"]) for r in rows], dtype=np.float64)
    t_f_db = np.array([float(r["t_f_db"]) for r in rows], dtype=np.float64)
    return f_hz, t_f_db


def test_threshold_in_quiet_agrees_with_iso226_measured_threshold_within_bound():
    """Cross-check against ISO 226:2003 Table 1's `T_f` column -- an
    EMPIRICALLY MEASURED free-field threshold of hearing, independent of
    (and not derived from) the Terhardt 1979 formula this module
    implements. The two models are NOT expected to match exactly (Terhardt's
    analytic approximation does not capture the ear-canal resonance ISO226's
    measured data shows near 6-10 kHz), but a correct implementation of a
    widely-cited absolute-threshold model should agree with real measured
    human-hearing data to within a modest bound across the audible range --
    this is the closest available thing to ground truth for this quantity,
    since no numerically tabulated form of Terhardt's own formula is
    published (it is an analytic expression, not a table)."""
    f_hz, t_f_db = _load_iso226_table()
    tq = threshold_in_quiet(f_hz)
    diff = tq - t_f_db
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    print(f"\n[masking] Tq vs ISO226 T_f: max |diff| = {max_abs_diff:.2f} dB, mean |diff| = {mean_abs_diff:.2f} dB")
    for f, iso, terhardt in zip(f_hz, t_f_db, tq, strict=True):
        print(f"  {f:>8.1f} Hz: ISO226={iso:>7.2f} dB, Terhardt={terhardt:>7.2f} dB, diff={terhardt - iso:>6.2f}")
    # Measured this session: max |diff| = 12.60 dB (at 12 500 Hz, where the
    # two models diverge most -- see module docstring). 15 dB keeps margin
    # above that measured worst case while still catching a genuinely wrong
    # implementation (e.g. a sign error would produce differences an order
    # of magnitude larger almost everywhere, not just at one edge frequency).
    assert max_abs_diff < 15.0
    # The bulk of the range should agree much more closely than the single
    # worst-case point; guards against "one lucky/unlucky point passes while
    # the model is systematically off everywhere".
    assert mean_abs_diff < 5.0


@pytest.mark.parametrize(
    ("bad_f", "match"),
    [(0.0, "must be > 0"), (-1.0, "must be > 0"), (float("nan"), "must be finite"), (float("inf"), "must be finite")],
)
def test_threshold_in_quiet_rejects_invalid_frequency(bad_f, match):
    with pytest.raises(ValueError, match=match):
        threshold_in_quiet(np.array([bad_f]))


def test_threshold_in_quiet_clips_above_ceiling_rather_than_blowing_up():
    """The f^4 term diverges to physically meaningless values above the
    audible range -- module docstring's own worked example: ~66 dB at
    16 kHz vs. ~160 dB at 20 kHz for a 25% change in frequency. Past the
    ceiling, the function must clip (return the ceiling's own value), not
    extrapolate the divergence further."""
    at_ceiling = float(threshold_in_quiet(np.array([20000.0]))[0])
    past_ceiling = float(threshold_in_quiet(np.array([40000.0]))[0])
    far_past_ceiling = float(threshold_in_quiet(np.array([1_000_000.0]))[0])
    assert at_ceiling == pytest.approx(past_ceiling)
    assert at_ceiling == pytest.approx(far_past_ceiling)


# --- spreading_function_peaq: validation ---


def test_spreading_function_rejects_non_peaq_scale():
    bands = band_edges(32, scale="bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    with pytest.raises(ValueError, match="bark_peaq"):
        spreading_function_peaq(np.ones(32), bands)


def test_spreading_function_rejects_erb_scale():
    bands = band_edges(32, scale="erb_glasberg_moore", f_min=20.0, f_max=20000.0)
    with pytest.raises(ValueError, match="bark_peaq"):
        spreading_function_peaq(np.ones(32), bands)


def test_spreading_function_rejects_mismatched_band_power_shape():
    bands = _peaq_bands(32)
    with pytest.raises(ValueError, match="n_bands"):
        spreading_function_peaq(np.ones(16), bands)


@pytest.mark.parametrize(
    ("bad_power", "match"),
    [
        (np.array([-1e-6] + [1e-6] * 31), "non-negative"),
        (np.array([0.0] + [1e-6] * 31), "strictly positive"),
        (np.array([float("nan")] + [1e-6] * 31), "finite"),
        (np.array([float("inf")] + [1e-6] * 31), "finite"),
    ],
)
def test_spreading_function_rejects_invalid_band_power(bad_power, match):
    bands = _peaq_bands(32)
    with pytest.raises(ValueError, match=match):
        spreading_function_peaq(bad_power, bands)


def test_spreading_function_rejects_degenerate_single_band():
    bands = band_edges(1, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    with pytest.raises(ValueError, match="n_bands"):
        spreading_function_peaq(np.array([1e-3]), bands)


# --- spreading_function_peaq: asymmetry (acceptance criterion 1) ---


def test_impulse_spreads_asymmetrically_upward_stronger_than_downward():
    """A single-band impulse's spread masking profile is asymmetric: at
    matching Bark distance from the masker, the band ABOVE it carries more
    spread energy than the band BELOW it -- upward spread is stronger, the
    core acceptance criterion for this whole step."""
    n = 55
    bands = _peaq_bands(n, f_min=80.0, f_max=18000.0)
    peak = n // 2
    power = np.full(n, 1e-30)
    power[peak] = 0.01
    out = spreading_function_peaq(power, bands)

    for distance in (1, 2, 3, 4, 5):
        below = out[peak - distance]
        above = out[peak + distance]
        print(f"\n[masking] distance={distance}: below={below:.6e} above={above:.6e} (above/below={above / below:.3f})")
        assert above > below, (
            f"at distance {distance} bands, upward spread ({above}) was not stronger than downward ({below})"
        )


def test_measured_slopes_match_the_cited_peaq_model_within_tolerance():
    """Fit the log-linear (dB vs. Bark) decay on each side of a single
    masker, away from array edges and far enough from the masker that the
    OTHER side's contribution and the masker's own band are negligible, and
    compare the fitted slope to ITU-R BS.1387's own published slope formula
    (S_l = 27 dB/Bark fixed; S_u = -24 - 230/f_c + 0.2*L dB/Bark) -- this is
    the "slopes in dB/Bark matching the cited model within a stated
    tolerance" acceptance criterion, checked as an actual numeric fit, not
    by re-typing the formula and comparing it to itself."""
    n = 109
    bands = band_edges(n, scale="bark_peaq", f_min=80.0, f_max=15000.0)
    fc = np.asarray(bands["centers_hz"])
    z = hz_to_bark_peaq(fc)
    peak = 60

    for level_normalized in (1e-6, 1e-3, 0.3):
        power = np.full(n, 1e-30)
        power[peak] = level_normalized
        out = spreading_function_peaq(power, bands)
        out_db = 10.0 * np.log10(out)

        lo_idx = np.arange(peak - 8, peak - 2)
        hi_idx = np.arange(peak + 3, peak + 9)
        slope_lo = float(np.polyfit(z[lo_idx], out_db[lo_idx], 1)[0])
        slope_hi = float(np.polyfit(z[hi_idx], out_db[hi_idx], 1)[0])

        masker_level_db_spl = 92.0 + 10.0 * np.log10(level_normalized)
        theoretical_upper = -24.0 - 230.0 / fc[peak] + 0.2 * masker_level_db_spl

        print(
            f"\n[masking] level={level_normalized:.1e} (masker {masker_level_db_spl:.1f} dB SPL): "
            f"measured slope_lo={slope_lo:.3f} (theory 27.0), "
            f"measured slope_hi={slope_hi:.3f} (theory {theoretical_upper:.3f})"
        )
        assert slope_lo == pytest.approx(27.0, abs=0.1)
        assert slope_hi == pytest.approx(theoretical_upper, abs=0.1)
        # The asymmetry itself, restated numerically: the upper (shallower)
        # slope's magnitude must be smaller than the lower (steeper) one.
        assert abs(slope_hi) < abs(slope_lo)


# --- spreading_function_peaq: level dependence (acceptance criterion 2) ---


def test_upper_slope_shallows_as_masker_level_increases():
    """Increasing a masker's level makes the upper slope LESS steep (the
    `+0.2*L` term), i.e. masking reaches further upward at higher levels --
    the direction ITU-R BS.1387 specifies, not merely "changes somehow"."""
    n = 55
    bands = _peaq_bands(n, f_min=80.0, f_max=18000.0)
    peak = n // 2

    ratios = []
    for level in (1e-6, 1e-4, 1e-2, 1.0):
        power = np.full(n, 1e-30)
        power[peak] = level
        out = spreading_function_peaq(power, bands)
        ratio_2up_to_peak = out[peak + 2] / out[peak]
        ratios.append(ratio_2up_to_peak)
        print(f"\n[masking] level={level:.1e}: out[peak+2]/out[peak] = {ratio_2up_to_peak:.6e}")

    assert all(b > a for a, b in pairwise(ratios)), (
        f"upward-spread ratio did not increase monotonically with masker level: {ratios}"
    )


# --- spreading_function_peaq: normalisation (acceptance criterion 3) ---


def test_uniform_excitation_is_exactly_preserved_at_the_bs_reference_level():
    """The normalising factor Bs (ITU-R BS.1387 eq. 167 / Kabal's `Bs`) is,
    by the standard's OWN construction, computed at a fixed reference where
    every band's calibrated power is exactly 1.0. At exactly that real
    signal level, spreading a uniform profile must return that SAME uniform
    profile to float64 precision -- this is provable directly from the
    construction (see module docstring), not an approximation."""
    playback_level_db_spl = 92.0
    n = 32
    bands = _peaq_bands(n)
    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    reference_normalised_power = 1.0 / calibration
    uniform = np.full(n, reference_normalised_power)

    out = spreading_function_peaq(uniform, bands, playback_level_db_spl=playback_level_db_spl)
    max_abs_dev = float(np.max(np.abs(out / uniform - 1.0)))
    print(f"\n[masking] at exact Bs reference level: max |ratio - 1| = {max_abs_dev:.3e}")
    # NOT `np.allclose(out, uniform, rtol=1e-9)` (no explicit atol): `uniform`
    # here is ~6.3e-10 (1.0 / 10**9.2), so `np.allclose`'s DEFAULT atol=1e-8
    # is ~16x LARGER than the quantity being compared -- it would pass for
    # ANY `out` within 1e-8 of zero, regardless of whether the recurrence
    # computed the right answer. MEASURED: deleting the Bs normalisation
    # entirely still passes that old assertion (ratio deviates by 84%, but
    # 84% of ~6.3e-10 is ~5.3e-10, still under the 1e-8 default atol).
    # `max_abs_dev` above is already a RATIO (dimensionless, atol-immune);
    # asserting on it directly is both tighter and correct regardless of
    # the quantities' absolute scale.
    assert max_abs_dev < 1e-9


def test_uniform_excitation_deviation_is_bounded_and_matches_independent_oracle():
    """At realistic (non-reference) levels, a uniform excitation does NOT
    spread back to exactly uniform -- see module docstring's "Normalisation
    after spreading" section for why this is real, substantial, and
    expected model behaviour, not a scaling defect. What this test actually
    guards against is (a) an UNBOUNDED blow-up (a true scaling bug, as
    opposed to the model's own real level dependence) and (b) it separately
    documents the measured numbers so a future change to this deviation is
    visible rather than silent. The claim that this deviation is the
    PUBLISHED algorithm's own behaviour -- not this port's -- is established
    by `test_dsp_masking_kabal_reference.py`'s independent-oracle comparison
    at these same flat/high-level profiles, not by this test."""
    playback_level_db_spl = 92.0
    n = 32
    bands = _peaq_bands(n)

    measured = {}
    for normalized_level in (1.0, 1e-2, 1e-4, 1e-6, 1e-9):
        uniform = np.full(n, normalized_level)
        out = spreading_function_peaq(uniform, bands, playback_level_db_spl=playback_level_db_spl)
        ratio = out / uniform
        max_dev = float(np.max(np.abs(ratio - 1.0)))
        measured[normalized_level] = max_dev
        print(f"\n[masking] normalised_level={normalized_level:.1e}: max |ratio-1| = {max_dev:.4f}")

    # Measured this session (32 PEAQ bands, 92 dB SPL calibration): 5.7382
    # at full-scale (normalised power 1.0, i.e. every one of 32 bands
    # simultaneously at 92 dB SPL -- a physically extreme, not merely loud,
    # input). The OLD bound here was `< 10.0` -- so loose it did not
    # discriminate a real defect: deleting the Bs normalisation entirely
    # (see the exact-reference test above) measures 8.7281 here, comfortably
    # under the old 10.0 bound. 6.0 keeps real margin above the measured
    # 5.7382 (to absorb harmless cross-platform/NumPy-version float
    # differences) while catching that mutation's 8.7281 with room to
    # spare, and still catches a true unbounded-blow-up regression.
    assert measured[1.0] < 6.0
    # And the deviation should shrink monotonically toward the reference
    # level (decreasing normalised_level here, since the reference level
    # itself is far below all of these -- see the exact-reference test).
    levels_desc = sorted(measured, reverse=True)
    devs_desc = [measured[level] for level in levels_desc]
    assert all(a >= b for a, b in pairwise(devs_desc)), (
        f"deviation from uniform did not shrink monotonically as level decreased toward the Bs reference: {measured}"
    )


def test_spreading_function_handles_trailing_frame_axis():
    """`band_power` may carry a trailing (frame, ...) axis (matching
    `aud.dsp.bands.band_energy`'s own convention); each frame must spread
    independently and match calling the function once per frame."""
    n = 32
    bands = _peaq_bands(n)
    rng = np.random.default_rng(7)
    multi_frame = rng.uniform(1e-6, 1e-2, size=(n, 3))

    batched = spreading_function_peaq(multi_frame, bands)
    for frame_idx in range(3):
        single = spreading_function_peaq(multi_frame[:, frame_idx], bands)
        assert np.allclose(batched[:, frame_idx], single)


def test_spreading_function_matches_independent_kabal_oracle_on_a_nonuniform_profile():
    """Coverage gap this session found and closed: every other test in
    THIS file is blind to a single-band index shift inside
    `_spread_power_domain` (e.g. accidentally reading `centers_hz[m+1]`
    instead of `centers_hz[m]` for band `m`'s own upper-slope coefficient).
    MEASURED directly: applying exactly that mutation still passes all 32
    of this file's other tests (asymmetry, fitted aggregate slopes over
    6-band windows, uniform-profile normalisation) -- because a uniform
    input is symmetric under a band relabelling, and a slope FITTED over
    several neighbouring bands barely moves when just one band's
    coefficient is nudged to its immediate neighbour's (Bark spacing is
    fine enough that adjacent bands' coefficients are nearly equal). Only
    `test_dsp_masking_kabal_reference.py`'s independent transliteration,
    exercised on a NON-uniform, per-band-distinguishable profile, catches
    it (24 of its parametrised cases failed under this exact mutation).

    Rather than write a THIRD from-scratch transliteration of the same
    published algorithm (which would itself need independent verification
    before it could be trusted), this test reuses that already-independent
    oracle directly -- closing the coverage gap in this file without
    duplicating the risk of a fresh, unverified re-implementation.
    """
    # Loaded by file path (not `import test_dsp_masking_kabal_reference`):
    # pytest's own import-mode does not reliably put `tests/` on `sys.path`
    # when only ONE test file is targeted on the command line (MEASURED
    # this session -- `pytest tests/test_dsp_masking.py` alone leaves
    # `sys.path[0]` as the repo root, not `tests/`), so a bare module import
    # here would pass when the whole suite is run but fail when this file
    # is run alone. Loading by explicit file path is independent of that.
    import importlib.util

    _kabal_ref_spec = importlib.util.spec_from_file_location(
        "test_dsp_masking_kabal_reference", Path(__file__).parent / "test_dsp_masking_kabal_reference.py"
    )
    _kabal_ref_module = importlib.util.module_from_spec(_kabal_ref_spec)
    _kabal_ref_spec.loader.exec_module(_kabal_ref_module)
    _kabal_spread_normalised = _kabal_ref_module._kabal_spread_normalised

    n = 55
    bands = band_edges(n, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    fc = np.asarray(bands["centers_hz"])
    z = hz_to_bark_peaq(fc)
    dz = float(np.mean(np.diff(z)))
    playback_level_db_spl = 92.0
    calibration = 10.0 ** (playback_level_db_spl / 10.0)

    # Deliberately non-uniform and non-symmetric: three maskers of
    # different magnitudes at different positions, well away from the
    # array edges (so no edge-clipping effect confounds the comparison).
    normalised = np.full(n, 1e-9)
    normalised[10] = 0.003
    normalised[27] = 0.05
    normalised[44] = 0.0007
    calibrated = normalised * calibration

    expected = _kabal_spread_normalised(calibrated, fc, dz) / calibration
    actual = spreading_function_peaq(normalised, bands, playback_level_db_spl=playback_level_db_spl)

    max_rel_err = float(np.max(np.abs((actual - expected) / np.where(expected != 0, expected, 1.0))))
    print(f"\n[masking] non-uniform profile vs independent Kabal oracle: max rel err = {max_rel_err:.3e}")
    assert np.allclose(actual, expected, rtol=1e-9, atol=1e-300)


# --- masking_threshold ---


def test_masking_threshold_sits_the_bs1387_offset_below_the_excitation():
    """CORRECTNESS BLOCKER fix, verified directly: ITU-R BS.1387 Annex 2
    Sec 2.1.9 eq. 24-26 requires the masking threshold to sit `m[k]` dB
    BELOW the (spread) excitation pattern, where

        m[k]_dB = 3.0             for k*res <= 12
                = 0.25 * (k*res)  for k*res >  12

    (`k` = 0-indexed band number, `res` = the uniform Bark spacing between
    band centres). BEFORE this fix, `masking_threshold` returned the raw
    excitation unchanged -- threshold-minus-excitation measured at exactly
    0.00 dB at every band, 3.0-6.75 dB too permissive (the UNSAFE direction
    for a ducker: concluding material is masked when it audibly is not).

    The expected `m[k]` values here are LITERAL constants from the
    standard, typed directly into this test -- NOT imported from
    `aud.dsp.masking._masking_offset_db` -- so this test cannot pass merely
    because production and test agree with each other; it independently
    re-derives the published formula and checks production against it.
    This is also this test's mutation-proof for BOTH constants (3.0 and
    0.25) individually: mutating either one in the source makes this
    assertion fail (verified directly this session -- see the PR's mutation
    transcript)."""
    n = 109
    bands = band_edges(n, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    fc = np.asarray(bands["centers_hz"])
    z = hz_to_bark_peaq(fc)
    dz = float(np.mean(np.diff(z)))

    rng = np.random.default_rng(2026)
    power = rng.uniform(1e-6, 1e-1, size=n)

    excitation = spreading_function_peaq(power, bands)
    threshold_unfloored = masking_threshold(power, bands, apply_absolute_threshold=False)
    measured_offset_db = 10.0 * np.log10(excitation) - 10.0 * np.log10(threshold_unfloored)

    k = np.arange(n, dtype=np.float64)
    bark_offset = k * dz
    expected_offset_db = np.where(bark_offset <= 12.0, 3.0, 0.25 * bark_offset)

    max_abs_diff = float(np.max(np.abs(measured_offset_db - expected_offset_db)))
    print(f"\n[masking] threshold-minus-excitation vs eq. 24 m[k]: max |diff| = {max_abs_diff:.3e} dB")
    assert np.allclose(measured_offset_db, expected_offset_db, atol=1e-9)


def test_masking_threshold_floors_very_quiet_bands_at_absolute_threshold():
    """A vanishingly quiet signal's masking threshold must not report a
    value quieter than the absolute threshold of hearing -- nothing masks
    below the floor no one can hear anything at regardless. Compared only
    against `threshold_in_quiet`'s own independently-verified output (see
    the checkpoints/ISO226 cross-check above) -- NOT against a fresh
    `spreading_function_peaq` call, which would make this test unable to
    distinguish a `masking_threshold` defect from a `spreading_function_peaq`
    defect (the self-consistency anti-pattern this PR's review flagged and
    this file has spent eight review rounds eliminating elsewhere)."""
    n = 32
    playback_level_db_spl = 92.0
    bands = _peaq_bands(n)
    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    # Far quieter than anything close to the absolute threshold at any
    # band's centre frequency.
    vanishingly_quiet = np.full(n, 1e-30)

    floored = masking_threshold(vanishingly_quiet, bands, playback_level_db_spl=playback_level_db_spl)

    tq_db_spl = threshold_in_quiet(np.asarray(bands["centers_hz"]))
    tq_normalised = 10.0 ** (tq_db_spl / 10.0) / calibration
    assert np.allclose(floored, tq_normalised)


def test_masking_threshold_does_not_floor_a_loud_signal_at_low_and_mid_frequencies():
    """Where the absolute threshold at this calibration is modest (below
    ~8 kHz, at 92 dB SPL), a loud signal's own masking threshold already
    exceeds it and the floor must be a no-op. Checked independently of
    `spreading_function_peaq`: if the floor were a no-op, `masking_threshold`
    (which is `max(pre_floor_value, tq)`) can only read ABOVE `tq` by
    returning `pre_floor_value` itself -- so `floored > tq` at a band
    PROVES the floor did not engage there, without needing a second,
    separately-computed `pre_floor_value` to compare against."""
    n = 32
    playback_level_db_spl = 92.0
    bands = _peaq_bands(n)
    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    loud = np.full(n, 0.1)
    floored = masking_threshold(loud, bands, playback_level_db_spl=playback_level_db_spl)
    centers = np.asarray(bands["centers_hz"])
    low_mid = centers < 8000.0
    assert np.any(low_mid)

    tq_db_spl = threshold_in_quiet(centers)
    tq_normalised = 10.0 ** (tq_db_spl / 10.0) / calibration
    assert np.all(floored[low_mid] > tq_normalised[low_mid])


def test_masking_threshold_floors_near_the_top_of_the_audible_range_even_for_a_loud_signal():
    """Near 18-20 kHz the absolute threshold of hearing is itself very high
    (Terhardt's own formula, verified above: ~66 dB SPL at 16 kHz but
    ~160 dB SPL at 20 kHz -- humans are dramatically less sensitive there),
    so at a realistic 92 dB SPL calibration a merely LOUD (not extreme)
    signal can legitimately read as quieter than the absolute threshold in
    that range. This is the mirror image of the low/mid-frequency test
    above: it confirms the floor actually ENGAGES where it should (the
    result EQUALS the independently-computed `tq_normalised` there), rather
    than this module's calibration silently no-op'ing it everywhere --
    checked independently of `spreading_function_peaq`, for the same reason
    given in the test above."""
    n = 32
    playback_level_db_spl = 92.0
    bands = _peaq_bands(n)
    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    loud = np.full(n, 0.1)
    floored = masking_threshold(loud, bands, playback_level_db_spl=playback_level_db_spl)
    centers = np.asarray(bands["centers_hz"])
    top_band = centers > 18000.0
    assert np.any(top_band)

    tq_db_spl = threshold_in_quiet(centers)
    tq_normalised = 10.0 ** (tq_db_spl / 10.0) / calibration
    assert np.allclose(floored[top_band], tq_normalised[top_band])
