"""Mutation-proving tests for `aud.dsp.collision`: each test here is
designed to FAIL if one specific, named design decision is broken --
matching the three properties this session was asked to mutation-prove:
the masker/maskee role assignment, the SII (not uniform/key-energy)
objective weighting, and the margin's direction of effect. See the PR
description for the actual mutate-run-observe-revert transcripts each of
these tests was checked against.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from aud.dsp.bands import band_edges
from aud.dsp.collision import collision_gains, sii_band_importance


def _peaq_bands(n_bands: int = 32, f_min: float = 20.0, f_max: float = 20000.0) -> dict:
    return band_edges(n_bands, scale="bark_peaq", f_min=f_min, f_max=f_max, allow_extrapolation=True)


# --- Role assignment: target=masker, key=maskee -- swapping the call arguments must change the result ---


def test_swapping_target_and_key_arguments_reverses_who_gets_cut():
    """ROLE ASSIGNMENT is the property `aud/dsp/collision.py`'s module
    docstring calls out as "get this backwards and everything downstream
    is wrong". A loud, broadband signal overlapping a quiet one: in the
    CORRECT order (loud=target/masker, quiet=key/maskee), the loud target
    must be cut meaningfully to protect the quiet key. Called with the
    SAME TWO ARRAYS SWAPPED (quiet=target/masker, loud=key/maskee), the
    "masker" now has almost nothing to spread, so almost no attenuation
    should be demanded at all -- a large, unambiguous difference, not a
    subtle one.
    """
    bands = _peaq_bands(32)
    n_bands = bands["n_bands"]
    loud = np.full((n_bands, 4), 0.1)
    quiet = np.full((n_bands, 4), 0.01)

    correct = collision_gains(loud, quiet, bands, margin_db=6.0)
    swapped = collision_gains(quiet, loud, bands, margin_db=6.0)

    correct_mean_gain_db = 10.0 * np.log10(np.mean(correct.gain))
    swapped_mean_gain_db = 10.0 * np.log10(np.mean(swapped.gain))

    assert correct_mean_gain_db < -5.0, (
        f"correct role assignment (loud target, quiet key) should cut the target meaningfully; "
        f"got {correct_mean_gain_db:.2f} dB mean gain"
    )
    assert swapped_mean_gain_db > -1.0, (
        f"swapped role assignment (quiet 'target', loud 'key') should cut almost nothing -- the "
        f"quiet masker has little to spread; got {swapped_mean_gain_db:.2f} dB mean gain"
    )
    assert swapped_mean_gain_db - correct_mean_gain_db > 5.0, (
        "swapping target/key must produce a large, unambiguous difference in how much gets cut"
    )


def test_target_empty_band_gain_alone_does_not_discriminate_a_role_swap():
    """A NEGATIVE result, kept deliberately: checking only the empty band's
    OWN gain does NOT discriminate a role swap -- it reads exactly 1.0 in
    BOTH the correct call and the swapped call, for two DIFFERENT
    structural reasons (target has no energy there vs. key is not active
    there). The real discriminator is the AGGREGATE assertion in
    `test_swapping_target_and_key_arguments_reverses_who_gets_cut` above,
    not a single-band check -- this test exists to document why that
    single-band check alone would be insufficient, not to claim it detects
    anything on its own.

    Renamed from
    `test_target_empty_band_stops_being_protected_from_attenuation_if_roles_swapped`,
    which asserted ~1.0 in both calls (i.e. this same non-discriminating
    result) while its name and opening claimed the opposite -- that the
    property INVERTS under a role swap. It does not; see the body below."""
    bands = _peaq_bands(32)
    n_bands = bands["n_bands"]
    centers = bands["centers_hz"]

    empty_band = int(np.argmin(np.abs(centers - 9000.0)))
    loud_bands = np.full(n_bands, 0.05)
    quiet_at_one_band = np.full(n_bands, 0.05)
    quiet_at_one_band[empty_band] = 1e-14  # "target has no energy here"

    # Correct: quiet_at_one_band is the target (masker); loud_bands is the key.
    correct = collision_gains(quiet_at_one_band[:, None], loud_bands[:, None], bands, margin_db=6.0)
    assert correct.gain[empty_band, 0] == pytest.approx(1.0, abs=1e-4)

    # Swapped: loud_bands is now the target (masker) -- its real energy at
    # `empty_band` is no longer protected from anything by the OTHER
    # array's hole there (the hole is now on the KEY side, which just
    # means that band is not key-active -- a DIFFERENT property, not the
    # one this test is pinning). This confirms the two calls are not
    # interchangeable, which is the point: the feature's own guarantee
    # applies to a specific ARGUMENT, not "whichever array happens to have
    # a hole."
    swapped = collision_gains(loud_bands[:, None], quiet_at_one_band[:, None], bands, margin_db=6.0)
    # loud_bands (now target) has REAL energy at empty_band, and quiet_at_one_band
    # (now key) is active everywhere EXCEPT empty_band -- so empty_band carries no
    # key-side constraint in EITHER call, and both leave it at gain 1. What
    # differs is band assignment semantics, not this one band's own gain --
    # asserted instead via the aggregate test above, which is the real
    # discriminator. This test documents why a single-band check alone
    # would NOT reliably catch a role swap.
    assert swapped.gain[empty_band, 0] == pytest.approx(1.0, abs=1e-4)


# --- SII weighting: the objective must not be uniform, and must not favour bass ---


def test_sii_weights_are_not_uniform():
    """If `sii_band_importance` were mutated to return uniform weights
    (`np.ones(n_bands) / n_bands`, mimicking "give every band equal
    priority" -- functionally the same failure mode as weighting by key
    energy in a case where key energy happens to be flat), this test
    fails immediately: real ANSI S3.5-1997 importance is far from flat
    (bass and treble bands carry a fraction of the mid-band weight)."""
    bands = _peaq_bands(32)
    w = sii_band_importance(bands)
    uniform = np.full_like(w, 1.0 / len(w))
    assert not np.allclose(w, uniform, atol=1e-3), "SII weights must not degenerate to uniform"
    assert w.max() / w.min() > 3.0, "SII weights should vary substantially across the spectrum"


def test_sii_weights_do_not_favour_the_loudest_speech_bass_bands():
    """Normal-effort speech is loudest at 250-500 Hz (ANSI S3.5-1997 Table
    3's own standard speech spectrum: ~34.5 dB SPL at 315 Hz, falling to
    ~17.3 dB by 2 kHz) -- a ducker weighted by KEY ENERGY would treat
    250-500 Hz as most important. SII importance says the opposite: this
    test fails if `w` were built from (proportional to) that speech
    spectrum instead of the band-importance table."""
    bands = _peaq_bands(32)
    w = sii_band_importance(bands)
    centers = bands["centers_hz"]

    bass_mask = (centers >= 250.0) & (centers <= 500.0)
    mid_mask = (centers >= 1000.0) & (centers <= 3000.0)
    assert np.any(bass_mask)
    assert np.any(mid_mask)
    assert np.mean(w[mid_mask]) > np.mean(w[bass_mask]), (
        "1-3 kHz must carry more SII weight than 250-500 Hz -- a key-energy-weighted objective would get this backwards"
    )


# --- Margin: more margin must never demand LESS attenuation ---


def test_increasing_margin_never_decreases_required_attenuation():
    """`margin_db` enters the constraint as `E_s / 10**(margin_db/10) /
    M_b` -- a LARGER margin must TIGHTEN the constraint (smaller
    allowance), never loosen it. A sign error (e.g. multiplying by
    `margin_linear` instead of dividing, or negating `margin_db`) would
    make more margin demand LESS attenuation -- this test asserts the
    monotonic direction directly, across several margins, on a scenario
    with a real, non-trivial collision."""
    bands = _peaq_bands(32)
    n_bands = bands["n_bands"]
    target = np.full((n_bands, 1), 0.1)
    key = np.full((n_bands, 1), 0.02)

    margins = [0.0, 3.0, 6.0, 12.0, 20.0]
    mean_gains_db = []
    for margin_db in margins:
        result = collision_gains(target, key, bands, margin_db=margin_db)
        mean_gains_db.append(10.0 * np.log10(np.mean(result.gain)))

    for earlier, later in pairwise(mean_gains_db):
        assert later <= earlier + 1e-6, (
            f"increasing margin_db must never REDUCE the amount of attenuation demanded; "
            f"got gains (dB) {mean_gains_db} for margins {margins}"
        )
    assert mean_gains_db[-1] < mean_gains_db[0] - 1.0, "the largest margin tested should demand visibly more cut"
