"""Mutation-proving tests for `aud.dsp.smoothing` (Step 8, issue #18).

Each test here is designed to FAIL if one specific, named design decision
is broken. The five mutations issue #18 asks to be run (remove frequency
smoothing; make attack/release symmetric; remove the lookahead/non-causal
step; remove the hop-limit check; break partition of unity), plus the
negation-reuse claim in `aud.dsp.smoothing`'s own module docstring, were
each ACTUALLY applied to the real source, run against the NAMED test below,
confirmed RED, and reverted (`git diff` empty, `sha256sum` matched the
pre-mutation file) -- see the PR description for the mutate-run-observe-
revert transcripts. The tests below are the PERMANENT guards that would
catch a regression of the same shape in the future; they do not themselves
contain the mutated code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aud.dsp import bands as bands_module
from aud.dsp.gate import _hold_attack_release_db
from aud.dsp.smoothing import (
    TransitionFasterThanHopError,
    duck_gain_surface,
    smooth_frequency_axis_erb_db,
    smooth_time_axis_db,
    upsample_bands_to_bins,
)

# --- Mutation guard: "remove the frequency smoothing" ---


def test_removing_frequency_smoothing_would_fail_this_isolated_bin_check():
    """NAMED TEST for the "remove frequency smoothing" mutation
    (`smooth_frequency_axis_erb_db` replaced by `lambda x, *a: x`, an
    identity pass-through): an isolated single-bin -40 dB cut, surrounded
    by unity neighbours, must come back SHALLOWER than -40 dB. An identity
    mutation leaves it at EXACTLY -40 dB, failing this assertion.
    Reproduced directly against the real function this session: mutating
    `smooth_frequency_axis_erb_db` to `return bin_gain_db` unchanged makes
    this exact assertion fail (`-40.0 > -35.0` is False) -- confirmed RED,
    then reverted (`git diff` empty afterward).
    """
    b = bands_module.band_edges(32, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins = 2048 // 2 + 1
    raw = np.zeros(n_bins)
    raw[200] = -40.0
    smoothed = smooth_frequency_axis_erb_db(raw, b, n_fft=2048, sr=48000)
    assert smoothed[200] > -35.0, (
        f"an isolated single-bin cut must be diluted by ERB smoothing, not left untouched; got {smoothed[200]}"
    )


# --- Mutation guard: "make attack and release symmetric" ---


def test_forcing_symmetric_attack_release_would_fail_this_recovery_timing_check():
    """NAMED TEST for the "make attack and release symmetric" mutation
    (calling `smooth_gain_db` with `release_ms` forced equal to
    `attack_ms`): the release-timing prediction below
    (`hold_ms + release_ms * ln(depth/tol)`) is specific to an
    asymmetric release constant SLOWER than attack. Reproduced directly
    this session: patching `smooth_time_axis_db`'s call to `smooth_gain_db`
    to pass `attack_ms` in place of `release_ms` made a duck engaged at
    -18 dB recover to within 1 dB in 10 frames (100 ms) instead of the
    ~48 frames (~484 ms) this test expects from the real (asymmetric,
    release_ms=150 ms) behaviour -- confirmed RED against this test's own
    2-frame tolerance (`assert 38.4 <= 2` failed), then reverted
    (`git diff`/`sha256sum` both confirmed byte-identical afterward).
    """
    frame_rate_hz = 100.0  # 10 ms per frame, round numbers
    hold_ms, attack_ms, release_ms = 50.0, 20.0, 150.0
    duck_db = -18.0
    tol_db = 1.0

    n_frames = 200
    target_db = np.zeros(n_frames)
    target_db[5:100] = duck_db  # engaged, then released at frame 100

    smoothed, _stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=attack_ms, hold_ms=hold_ms, release_ms=release_ms, lookahead_frames=0
    )

    recovered_frame = next(
        (f for f in range(100, n_frames - 5) if np.all(np.abs(smoothed[f : f + 5]) < tol_db)),
        None,
    )
    assert recovered_frame is not None
    measured_recovery_frames = recovered_frame - 100
    expected_recovery_ms = hold_ms + release_ms * math.log(abs(duck_db) / tol_db)
    expected_recovery_frames = expected_recovery_ms / (1000.0 / frame_rate_hz)

    assert abs(measured_recovery_frames - expected_recovery_frames) <= 2, (
        f"recovery took {measured_recovery_frames} frames; expected ~{expected_recovery_frames:.1f} frames "
        "from the asymmetric release_ms=150 -- a symmetric (attack_ms-driven) release would recover "
        "roughly 7.5x faster and fail this bound"
    )


# --- Mutation guard: "remove the lookahead / non-causal step" ---


def test_removing_lookahead_would_fail_the_transient_depth_check():
    """NAMED TEST for the "remove the lookahead/non-causal step" mutation
    (`duck_gain_surface` forcing `lookahead_frames=0`): a single-frame
    transient duck request must reach close to its intended depth WITH
    lookahead. Reproduced directly this session: forcing
    `lookahead_frames=0` inside `duck_gain_surface` delivered only ~-5 dB
    of an intended -20 dB (matching the un-lookahead-compensated figure
    measured in `tests/test_dsp_smoothing_acceptance.py`), failing the
    `< -14.0` bound below -- confirmed RED, then reverted.
    """
    b = bands_module.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_frames = 40
    n_bands = b["n_bands"]
    transient_frame = 20
    band_gain_power = np.ones((n_bands, n_frames))
    band_gain_power[:, transient_frame] = 10 ** (-20.0 / 10.0)

    bin_gain_amp, stats = duck_gain_surface(
        band_gain_power, b, n_fft=2048, hop=512, sr=48000, attack_ms=20.0, hold_ms=50.0, release_ms=150.0
    )
    assert stats["lookahead_frames"] > 0
    delivered_db = 20.0 * np.log10(bin_gain_amp[:, transient_frame].mean())
    assert delivered_db < -14.0, (
        f"expected the automatic lookahead to deliver close to -20 dB at the transient frame; got {delivered_db:.2f}"
    )


# --- Mutation guard: "remove the hop-limit check" ---


def test_removing_hop_limit_check_would_fail_this_error_assertion():
    """NAMED TEST for the "remove the hop-limit check" mutation (deleting
    `duck_gain_surface`'s `attack_ms`/`release_ms` vs `hop_ms` validation):
    an attack faster than the hop must raise `TransitionFasterThanHopError`.
    Reproduced directly this session: commenting out that validation block
    let the call below SUCCEED (return a gain surface) instead of raising,
    failing `pytest.raises` -- confirmed RED, then reverted.
    """
    b = bands_module.band_edges(8, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    band_gain = np.ones((8, 10))
    hop_ms = 512 / 48000 * 1000.0
    with pytest.raises(TransitionFasterThanHopError):
        duck_gain_surface(band_gain, b, n_fft=2048, hop=512, sr=48000, attack_ms=hop_ms / 2.0, release_ms=150.0)


# --- Mutation guard: "break partition of unity" ---


def test_breaking_partition_of_unity_would_fail_the_constant_upsample_check():
    """NAMED TEST for the "break partition of unity" mutation (e.g.
    `smooth_frequency_axis_erb_db` forgetting to divide by `row_sums`, or
    `upsample_bands_to_bins` scaling `weights` by an arbitrary constant): a
    CONSTANT band value must upsample to that SAME constant everywhere --
    the direct, minimal consequence of `sum_b weights[b, k] == 1` for every
    bin `k`. Reproduced directly this session: multiplying `weights` by 0.5
    before the final `tensordot` in `upsample_bands_to_bins` made a
    constant `-7.5` band value upsample to `-3.75` everywhere (linear gain
    domain, not dB, because the scaling happens on the dB array; the effect
    on a dB constant is a proportional dB scaling too, still visibly
    wrong), failing `np.allclose(bin_db, -7.5, ...)` -- confirmed RED, then
    reverted.
    """
    b = bands_module.band_edges(16, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    constant_db = np.full((16, 1), -7.5)
    bin_db = upsample_bands_to_bins(constant_db, b, n_fft=2048, sr=48000)
    assert np.allclose(bin_db, -7.5, atol=1e-9)


# --- Negative result, kept deliberately: gate.py's hold+ballistics ---
# --- cannot be reused for a duck merely by negating its input ---


def test_gate_hold_attack_release_cannot_be_reused_by_negation():
    """A NEGATIVE result, kept deliberately (see `aud.dsp.collision`'s own
    precedent for this convention): `aud.dsp.gate._hold_attack_release_db`
    is NOT reused by this module even via negating the input, and this
    demonstrates the specific, concrete way that fails rather than merely
    asserting it. A steady -30 dB duck request, negated to +30 dB and fed
    to the gate's own function: its `is_open_raw = target_db > -eps` test
    reads a `+30` value as "open" on every single frame (the eps-threshold
    direction is calibrated for a `<= 0` curve approaching 0 FROM BELOW,
    not a `>= 0` curve approaching 0 FROM ABOVE), so the hold counter never
    counts down and the result stays at unity (0.0) throughout -- silently
    discarding the entire duck request rather than smoothing it.
    """
    target_db = np.full(10, -30.0)
    negated = -target_db
    smoothed, open_count = _hold_attack_release_db(negated, 100.0, 5.0, 50.0, 150.0)
    assert open_count == 0, "negation should (wrongly) read every frame as 'open'/unity, never counting an engagement"
    assert np.allclose(smoothed, 0.0), (
        f"negated reuse of gate's hold+ballistics silently discards the whole duck request; got {smoothed}"
    )
