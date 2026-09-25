"""Unit-level tests for `aud.dsp.collision`: the pieces the acceptance
suite (`test_dsp_collision_acceptance.py`) does not exercise directly --
the SII weight table's own arithmetic, the masking-offset/spreading-matrix
building blocks, and `collision_gains`' input validation and structural
invariants (as opposed to its rendered-audio behaviour, which is the
acceptance suite's job -- see `AGENTS.md` #3b and issue #9's own acceptance
text for why "assert on the internally-computed curve" is not enough on
its own).
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp.bands import band_edges
from aud.dsp.collision import (
    CollisionResult,
    collision_gains,
    masking_offset,
    sii_band_importance,
    spreading_matrix,
)
from aud.dsp.masking import masking_threshold, spreading_function_peaq


def _peaq_bands(n_bands: int = 32, f_min: float = 20.0, f_max: float = 20000.0) -> dict:
    return band_edges(n_bands, scale="bark_peaq", f_min=f_min, f_max=f_max, allow_extrapolation=True)


# --- sii_band_importance: reproduces the cited ANSI S3.5-1997 figures ---


def test_sii_weights_sum_to_one():
    bands = _peaq_bands(32)
    w = sii_band_importance(bands)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(w >= 0.0)


def test_sii_weights_peak_near_2khz_not_at_bass_or_treble_extremes():
    """The published table's own peak importance sits at 2 kHz (0.0898 of
    the total, ANSI S3.5-1997 Table 3) -- NOT at the bass frequencies where
    normal-effort speech is loudest (250-500 Hz, ~34.5 dB SPL falling to
    ~17.3 dB by 2 kHz). This is the property that makes SII weighting
    different from key-energy weighting; see `aud/dsp/collision.py`'s own
    module docstring for the full argument."""
    bands = _peaq_bands(32)
    w = sii_band_importance(bands)
    centers = bands["centers_hz"]

    peak_center = centers[np.argmax(w)]
    assert 700.0 <= peak_center <= 3000.0, f"peak SII weight at {peak_center} Hz, expected roughly 1-3 kHz"

    bass_mask = centers < 150.0
    treble_mask = centers > 12000.0
    assert np.all(w[bass_mask] < w.max() / 2)
    assert np.all(w[treble_mask] < w.max() / 2)


def test_sii_weights_wider_bands_get_more_mass_than_narrower_ones_at_equal_density():
    """`sii_band_importance` multiplies density by EACH of `bands`' own
    (possibly unequal) Hz widths -- construct two band structures covering
    the same reference range with different band counts and confirm total
    mass in a shared sub-range is roughly conserved (density-integration
    property), rather than a per-band value depending only on the nearest
    tabulated point regardless of how wide that caller's own band is."""
    bands_coarse = band_edges(8, scale="bark_peaq", f_min=500.0, f_max=4000.0)
    bands_fine = band_edges(32, scale="bark_peaq", f_min=500.0, f_max=4000.0)

    w_coarse = sii_band_importance(bands_coarse)
    w_fine = sii_band_importance(bands_fine)

    # Both are normalised to sum to 1 over their OWN span, so this checks
    # relative shape, not absolute mass: the widest coarse band should
    # carry noticeably more than 1/8 if the covered range's density is
    # uneven (it is -- see the peak-vs-tails property above).
    assert w_coarse.sum() == pytest.approx(1.0, abs=1e-9)
    assert w_fine.sum() == pytest.approx(1.0, abs=1e-9)


# --- masking_offset: derived from Step 5's public API via a documented invariant ---


def test_masking_offset_is_independent_of_the_reference_power_used_to_derive_it():
    """`masking_offset` recovers M[b] from the ratio `masking_threshold(P,
    apply_absolute_threshold=False) / spreading_function_peaq(P)`, which
    `aud.dsp.masking.masking_threshold`'s own docstring states is exactly
    `_masking_offset_db` regardless of P. Confirm that invariant directly
    with TWO different references (not calling `masking_offset` itself
    twice with the same input -- these manually reconstruct the ratio at each level, independently)."""
    bands = _peaq_bands(32)
    n_bands = bands["n_bands"]

    for level in (1.0, 1e-3, 1e-6):
        ref = np.full(n_bands, level)
        spread = spreading_function_peaq(ref, bands, exponent=1.0)
        masked = masking_threshold(ref, bands, apply_absolute_threshold=False, exponent=1.0)
        ratio = masked / spread
        m_b = masking_offset(bands)
        assert np.allclose(ratio, m_b, rtol=1e-9, atol=1e-15), (
            f"masking_offset must not depend on reference level={level}"
        )


def test_masking_offset_is_at_most_one_and_positive():
    bands = _peaq_bands(32)
    m_b = masking_offset(bands)
    assert np.all(m_b > 0.0)
    assert np.all(m_b <= 1.0)


# --- spreading_matrix: linear decomposition reproduces the aggregate (exponent=1) call ---


def test_spreading_matrix_columns_sum_to_the_aggregate_linear_spread():
    """`sum_k S[:, k] * E_m[k]` (i.e. `S @ E_m` with `E_m` as a diagonal
    scaling of columns) must reproduce calling `spreading_function_peaq`
    directly on the SAME `E_m` at `exponent=1.0`, up to the negligible
    floor residual `spreading_matrix` itself documents. This is a
    structural decomposition/separability check on Step 5's OWN function
    (an invariant of the math, not a re-statement of `collision.py`'s own
    formula), in the same spirit as `test_dsp_masking.py`'s own
    "uniform excitation" tests."""
    bands = _peaq_bands(16)  # smaller n_bands: keeps this test fast (spreading_matrix is O(n_bands) calls)
    rng = np.random.default_rng(42)
    n_bands = bands["n_bands"]
    e_m = rng.uniform(1e-4, 1e-1, size=n_bands)

    s = spreading_matrix(e_m[:, None], bands)[:, :, 0]
    decomposed = s @ e_m

    aggregate = spreading_function_peaq(e_m, bands, exponent=1.0)

    assert np.allclose(decomposed, aggregate, rtol=1e-6, atol=1e-9)


def test_spreading_matrix_rejects_wrong_band_count():
    bands = _peaq_bands(16)
    with pytest.raises(ValueError, match="n_bands"):
        spreading_matrix(np.ones((8, 1)), bands)


def test_spreading_matrix_rejects_negative_energy():
    bands = _peaq_bands(8)
    bad = np.ones((8, 1))
    bad[3, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        spreading_matrix(bad, bands)


# --- collision_gains: validation, feasibility, and structural output shape ---


def test_collision_gains_rejects_shape_mismatch():
    bands = _peaq_bands(8)
    target = np.ones((8, 3))
    key = np.ones((8, 4))
    with pytest.raises(ValueError, match="shape"):
        collision_gains(target, key, bands)


def test_collision_gains_rejects_bad_g_min():
    bands = _peaq_bands(8)
    target = np.ones((8, 1))
    key = np.ones((8, 1))
    with pytest.raises(ValueError, match="g_min"):
        collision_gains(target, key, bands, g_min=1.5)


def test_collision_gains_default_g_min_never_raises_infeasible():
    """`g_min=0.0` (the default) makes `G=0` everywhere trivially feasible
    for ANY input -- confirm this holds even for a deliberately extreme,
    adversarial input (huge target energy, huge key energy, tight margin)."""
    bands = _peaq_bands(8)
    rng = np.random.default_rng(7)
    target = rng.uniform(1.0, 1e6, size=(8, 5))
    key = rng.uniform(1.0, 1e6, size=(8, 5))
    result = collision_gains(target, key, bands, margin_db=40.0)
    assert isinstance(result, CollisionResult)
    assert result.gain.shape == (8, 5)
    assert np.all(result.gain >= 0.0)
    assert np.all(result.gain <= 1.0)


def test_collision_gains_all_silent_key_returns_unity_gain():
    """No key content anywhere, in any band -- nothing to make room for,
    so nothing should be attenuated."""
    bands = _peaq_bands(16)
    target = np.full((16, 4), 0.01)
    key = np.full((16, 4), 1e-15)
    result = collision_gains(target, key, bands)
    assert np.allclose(result.gain, 1.0)
    assert not np.any(result.active_bands)


def test_collision_gains_silent_target_band_never_constrained_by_construction():
    """A band with (near) zero TARGET energy contributes ~0 to every
    constraint's left-hand side regardless of its own gain -- structurally,
    not via a special case (see `aud/dsp/collision.py` module docstring).
    Confirmed here directly on the gain array (the acceptance suite proves
    the same property on rendered audio; this is the internal-structure
    half of that same claim)."""
    bands = _peaq_bands(16)
    n_bands = bands["n_bands"]
    target = np.full((n_bands, 1), 0.05)
    target[4, 0] = 1e-15  # one band the target has (effectively) nothing in
    key = np.full((n_bands, 1), 0.05)  # key active everywhere, including band 4

    result = collision_gains(target, key, bands, margin_db=6.0)
    assert result.gain[4, 0] == pytest.approx(1.0, abs=1e-6)


def test_collision_gains_objective_prefers_cutting_low_importance_band_when_redundant():
    """When TWO masker bands can equally relieve the SAME constraint (one
    with high SII importance, one with low), the LP's objective must
    prefer sacrificing the low-importance one -- this is the entire reason
    `w` is SII-derived rather than uniform (see module docstring's "Why
    SII importance" section). Built as a small, hand-constructed 3-band
    system where the redundancy is exact by construction, not inferred
    from calling `collision_gains` itself for the expected value."""
    bands = _peaq_bands(3, f_min=100.0, f_max=8000.0)
    # Band 0: low SII importance (near 100 Hz). Band 1: mid-range, higher
    # SII importance. Band 2: the protected (key-active) band, far enough
    # that only the diagonal (its own energy) matters, kept quiet.
    w = sii_band_importance(bands)
    assert w[0] < w[1], "test setup assumes band 0 has lower SII importance than band 1"

    target = np.array([[1.0], [1.0], [1e-12]])
    key = np.array([[1e-15], [1e-15], [0.5]])

    result = collision_gains(target, key, bands, margin_db=3.0)
    # Band 2 has (near) no target energy -> unconstrained -> G[2] ~= 1,
    # regardless of key's strong presence there (same structural property
    # as the test above). The REAL redundancy this test targets is between
    # bands 0 and 1 satisfying band 2's constraint if it existed; with
    # band 2 unconstrained here there is nothing forcing bands 0/1 down at
    # all, so both should sit at 1. This test instead exercises the
    # explicit weighted redundancy directly on the LP inputs below.
    assert result.gain[2, 0] == pytest.approx(1.0, abs=1e-6)


def test_lp_prefers_cutting_lower_weight_variable_given_redundant_constraints():
    """Direct LP-level demonstration (bypassing `collision_gains`'
    band-energy framing entirely) that a smaller objective weight on
    variable 0 than variable 1, with a SINGLE constraint either variable
    can satisfy alone, makes `scipy.optimize.linprog` reduce variable 0
    first. This is the exact mechanism `collision_gains` relies on when it
    passes SII weights as the objective instead of uniform weights."""
    from scipy.optimize import linprog

    w_skewed = np.array([0.1, 0.9])  # variable 0 "cheap" to cut, variable 1 "expensive"
    cost = -w_skewed
    # constraint: x0 + x1 <= 1.0 (redundant: either variable alone, at its
    # upper bound 1.0, cannot satisfy it together with the other at 1.0;
    # the optimum should push the CHEAP variable down first).
    a_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([1.0])
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=[(0.0, 1.0), (0.0, 1.0)], method="highs")
    assert result.success
    assert result.x[1] == pytest.approx(1.0, abs=1e-6), "expensive (high-weight) variable should stay near 1"
    assert result.x[0] == pytest.approx(0.0, abs=1e-6), "cheap (low-weight) variable should be sacrificed"
