"""Acceptance tests for issue #16 ([aud] Step 6: the collision measure).
These four tests ARE the feature's definition (see the issue text and
`aud/dsp/collision.py`'s own module docstring) -- each measures the
RENDERED output of a real STFT-analyse -> `collision_gains` -> apply-gain
-> ISTFT round trip (via `collision_test_support.py`), never the
internally-computed gain curve alone (see `AGENTS.md` and issue #9's own
acceptance text for why that distinction is load-bearing: "those
measurements are taken from the rendered audio, not from the automation
curve the code computed").

Every scenario uses REAL, physically bandlimited noise (Butterworth
bandpass, `scipy.signal.sosfiltfilt`) rather than hand-built band-energy
arrays specifically so these tests see the same practical spectral
leakage, calibration, and STFT-framing effects a real render would --
`collision_test_support.py`'s own docstring on why that harness performs
the calibration correction it does.
"""

from __future__ import annotations

import numpy as np

from aud.dsp import collision
from tests import collision_test_support as h


def _render_and_measure(target: np.ndarray, key: np.ndarray, bands: dict, margin_db: float = 6.0):
    """Runs the full pipeline and returns
    `(gain, orig_level_db, proc_level_db, diff_db)` -- `diff_db` is
    `proc_level_db - orig_level_db`, per band (mean-over-frames dB)."""
    target_power, target_spec, weights = h.analyze_band_energy(target, bands)
    key_power, _, _ = h.analyze_band_energy(key, bands)

    result = collision.collision_gains(target_power, key_power, bands, margin_db=margin_db)

    processed_spec = h.apply_band_gain(target_spec, result.gain, weights)
    y = h.resynthesize(processed_spec, len(target))

    orig_level = h.band_level_db(target_power)
    proc_power, _, _ = h.analyze_band_energy(y, bands)
    proc_level = h.band_level_db(proc_power)
    return result.gain, orig_level, proc_level, proc_level - orig_level


# --- Acceptance 1: key strong where target has (effectively) no energy -> not attenuated ---


def test_key_energy_in_target_empty_band_is_not_attenuated_on_rendered_output():
    """THE feature's own definition (issue #16): a key with strong energy
    in a band where the target has effectively no energy must produce NO
    attenuation there, measured on the rendered output -- not "gain looks
    like 1.0 in the array", the actual audio level in that band, before
    and after processing, must match."""
    rng = np.random.default_rng(101)
    bands = h.peaq_bands(32)

    target = h.band_limited_noise(rng, 3.0, 200, 800)  # target lives at 200-800 Hz only
    key = h.band_limited_noise(rng, 3.0, 8000, 10000)  # key loud where target has ~nothing

    gain, _orig_level, _proc_level, diff = _render_and_measure(target, key, bands, margin_db=6.0)

    centers = bands["centers_hz"]
    key_only_mask = (centers >= 8000.0) & (centers <= 10000.0)
    assert np.any(key_only_mask), "test setup: expected at least one band centred in 8-10 kHz"

    assert np.allclose(gain[key_only_mask], 1.0, atol=1e-4), "target-empty band must not be gained down at all"
    assert np.max(np.abs(diff[key_only_mask])) < 1.0, (
        f"rendered level in the target-empty/key-loud band changed by up to "
        f"{np.max(np.abs(diff[key_only_mask])):.2f} dB; expected < 1 dB (no attenuation)"
    )


# --- Acceptance 2: disjoint spectra -> target unchanged within tolerance ---


def test_disjoint_spectra_leave_target_unchanged_within_tolerance():
    """Target and key occupy well-separated frequency ranges everywhere --
    the rendered target must come back essentially unchanged, and total
    attenuation must be near zero."""
    rng = np.random.default_rng(202)
    bands = h.peaq_bands(32)

    target = h.band_limited_noise(rng, 3.0, 150, 600)
    key = h.band_limited_noise(rng, 3.0, 6000, 12000)

    _, orig_level, proc_level, diff = _render_and_measure(target, key, bands, margin_db=6.0)

    # Compare only bands INSIDE the target's own constructed passband
    # (150-600 Hz, known exactly since this test built the signal) --
    # not a relative-level heuristic. The filter's own rolloff/leakage
    # tail falls off only gradually in dB (a 4th-order Butterworth, not a
    # brick wall), so a level-relative mask keeps sweeping in tail bands
    # at whatever threshold is chosen; the frequency range the signal was
    # actually constructed to occupy is unambiguous.
    centers = bands["centers_hz"]
    real_content_mask = (centers >= 150.0) & (centers <= 600.0)
    assert np.any(real_content_mask)
    max_diff = np.max(np.abs(diff[real_content_mask]))
    assert max_diff < 2.0, f"disjoint spectra should leave target's own content unchanged; max diff {max_diff:.2f} dB"

    total_before = 10.0 ** (orig_level / 10.0)
    total_after = 10.0 ** (proc_level / 10.0)
    total_attenuation_db = 10.0 * np.log10(total_after.sum() / total_before.sum())
    assert abs(total_attenuation_db) < 1.0, f"total attenuation should be near zero; got {total_attenuation_db:.2f} dB"


# --- Acceptance 3: overlap only in a known range -> concentrated there ---


def test_localized_overlap_concentrates_attenuation_there_not_elsewhere():
    """Target is broadband; key overlaps it only in a known, narrow
    range. Attenuation must be concentrated in that range and measurably
    lower outside it -- the profile tracks the OVERLAP, not the key's own
    spectrum (which here IS the overlap range, so this also rules out the
    naive "gain driven by key energy alone" failure mode issue #16 names
    explicitly)."""
    rng = np.random.default_rng(303)
    bands = h.peaq_bands(32)

    target = h.white_noise(rng, 3.0) * 0.3  # broadband, energy everywhere
    key = h.band_limited_noise(rng, 3.0, 1000, 2000)  # only overlaps target here

    _, _orig_level, _proc_level, diff = _render_and_measure(target, key, bands, margin_db=6.0)

    centers = bands["centers_hz"]
    overlap_mask = (centers >= 800.0) & (centers <= 2500.0)  # the overlap range, with spreading margin
    outside_mask = (centers < 400.0) | (centers > 5000.0)  # comfortably away from the overlap
    assert np.any(overlap_mask)
    assert np.any(outside_mask)

    overlap_cut = np.mean(diff[overlap_mask])
    outside_cut = np.mean(diff[outside_mask])
    assert overlap_cut < -1.0, f"expected real attenuation inside the overlap range; got {overlap_cut:.2f} dB"
    assert outside_cut > -0.5, f"expected near-zero attenuation outside the overlap range; got {outside_cut:.2f} dB"
    assert overlap_cut < outside_cut - 1.0, (
        f"attenuation must be concentrated in the overlap range (overlap={overlap_cut:.2f} dB) "
        f"and measurably lower outside it (outside={outside_cut:.2f} dB)"
    )


# --- Acceptance 4: broadband collision -> graceful degrade toward a broadband duck ---


def test_broadband_collision_degrades_to_a_roughly_uniform_duck_never_worse_than_level_ducking():
    """Target and key are both broadband and fully overlapping -- the
    per-band collision measure should degrade gracefully toward something
    close to a simple broadband duck (roughly uniform attenuation across
    bands), and never cut MORE than a naive single-band level ducker
    would in this fully-overlapping case where a level duck is the
    correct call."""
    rng = np.random.default_rng(404)
    bands = h.peaq_bands(32)

    target = h.white_noise(rng, 3.0) * 0.3
    key = h.white_noise(rng, 3.0) * 0.1  # quieter, same broadband shape, full overlap

    _, orig_level, _proc_level, diff = _render_and_measure(target, key, bands, margin_db=6.0)

    # "Graceful degrade toward a broadband duck": the per-band attenuation
    # should be reasonably uniform, not wildly spiky (which would indicate
    # this step is still behaving like per-band gating rather than a
    # smoothly spreading masking measure).
    std_diff = float(np.std(diff))
    mean_diff = float(np.mean(diff))
    assert std_diff < abs(mean_diff), (
        f"broadband/broadband collision should be roughly uniform across bands; "
        f"std={std_diff:.2f} dB is not small relative to mean={mean_diff:.2f} dB"
    )

    # "Never worse than a level ducker where a level duck is correct": a
    # coarse single-band reference -- how much a naive full-band ducker
    # would cut the target to clear the key by the same margin, using
    # total (summed) band energy as the single-band proxy for level.
    total_target_db = 10.0 * np.log10(np.sum(10.0 ** (orig_level / 10.0)))
    total_key_power, _, _ = h.analyze_band_energy(key, bands)
    total_key_db = 10.0 * np.log10(np.sum(np.mean(total_key_power, axis=1)))
    margin_db = 6.0
    naive_full_band_cut_db = min(0.0, (total_key_db - margin_db) - total_target_db)

    assert mean_diff >= naive_full_band_cut_db - 3.0, (
        f"collision measure cut ({mean_diff:.2f} dB mean) should not be substantially deeper than "
        f"a naive full-band level ducker's cut ({naive_full_band_cut_db:.2f} dB) in this fully-overlapping case"
    )
