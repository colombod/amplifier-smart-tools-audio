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
    smooth_frequency_axis_db,
    smooth_time_axis_db,
    upsample_bands_to_bins,
)

# --- Mutation guard: "remove the frequency smoothing" ---


def test_removing_frequency_smoothing_would_fail_this_isolated_bin_check():
    """NAMED TEST for the "remove frequency smoothing" mutation
    (`smooth_frequency_axis_db` replaced by `lambda x, *a: x`, an
    identity pass-through): an isolated single-bin -40 dB cut, surrounded
    by unity neighbours, must come back SHALLOWER than -40 dB. An identity
    mutation leaves it at EXACTLY -40 dB, failing this assertion.
    Reproduced directly against the real function this session: mutating
    `smooth_frequency_axis_db` to `return bin_gain_db` unchanged makes
    this exact assertion fail (`-40.0 > -35.0` is False) -- confirmed RED,
    then reverted (`git diff` empty afterward).
    """
    b = bands_module.band_edges(32, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    n_bins = 2048 // 2 + 1
    raw = np.zeros(n_bins)
    raw[200] = -40.0
    smoothed = smooth_frequency_axis_db(raw, b, n_fft=2048, sr=48000)
    assert smoothed[200] > -35.0, (
        f"an isolated single-bin cut must be diluted by Bark smoothing, not left untouched; got {smoothed[200]}"
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
    `smooth_frequency_axis_db` forgetting to divide by `row_sums`, or
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


def test_gate_hold_attack_release_negation_reuse_fails_across_varying_duck_depth():
    """STRENGTHENS the test above (PR #40 review round 2): the original
    test used one STEADY -30 dB curve, which leaves open whether the
    failure is specific to a constant/deep request or general to any duck
    curve. Reproduced directly this session with a curve that VARIES --
    briefly returns near unity, dips to two different engaged depths (-30
    and -18 dB), and returns to unity again -- the same eps-threshold
    mirror-image argument applies at every value in `(-inf, 0]`, since
    `negated = -target_db >= 0` for ANY duck curve, and `negated > -eps` is
    true for every non-negative value regardless of magnitude. Confirms the
    claim is not an artifact of the original test's constant input: the
    entire time-varying request -- not just its deepest point -- is
    silently discarded to a flat 0.0, and the hold counter never once
    counts an engagement despite the curve genuinely engaging and releasing
    twice.
    """
    target_db = np.array([0.0, -5.0, -30.0, -30.0, -10.0, 0.0, -18.0, -18.0, -2.0, 0.0])
    negated = -target_db
    smoothed, open_count = _hold_attack_release_db(negated, 100.0, 5.0, 50.0, 150.0)
    assert open_count == 0, (
        "negation should (wrongly) read every frame as 'open'/unity regardless of how the duck curve varies"
    )
    assert np.allclose(smoothed, 0.0), (
        f"a duck curve that genuinely varies (engages twice, releases twice) must not collapse to a flat 0.0 "
        f"under negated reuse of gate's hold+ballistics; got {smoothed}"
    )


# --- Mutation guard: PR #40 review round 3 -- hold must latch EVERY ---
# --- engaged frame's own target, not just the depth seen when the ---
# --- engagement first began ---
#
# The review mutated `_duck_hold_attack_release_db` to latch only the
# FIRST engage depth (`held_value` set once, on the released->engaged
# transition, instead of on every engaged frame) and all 50 pre-existing
# smoothing tests still passed, because every one of them either engages
# at one CONSTANT depth throughout, or releases and RE-engages (a fresh
# engagement each time) -- never changes depth WITHIN a single continuous
# engagement. That is the gap the two tests below close: a single
# continuous duck (target never returns above `-DUCK_ENGAGE_EPS_DB`
# between the two depths, so `engage_count` must read exactly 1) whose
# depth changes mid-engagement, in each direction.
#
# Shipped `_duck_hold_attack_release_db` sets `held_value =
# float(target_db[n])` on every frame where `is_engaged_raw[n]` is True
# (that function's own source) -- so during a continuous engagement the
# held curve tracks the raw target exactly, frame for frame, and a
# mid-engagement depth change reaches the ballistics-predicted new depth
# on the timescale the CHANGE's own direction calls for. A first-depth
# latch instead freezes `held_value` at the first engaged sample and never
# updates it again while continuously engaged, so neither direction of
# change ever reaches the smoothed curve -- reproduced directly this
# session (see PR description for the mutate/run/revert transcript; the
# two tests below are the NAMED tests that transcript's RED run cites).
#
# The settle-time bound both tests assert against is the same exact
# one-pole exponential-decay formula this file's own
# `test_forcing_symmetric_attack_release_...` test and
# `tests/test_dsp_smoothing_acceptance.py::
# test_bed_returns_to_unity_after_release_following_key_gap` already use:
# `settle_ms = tau_ms * ln(gap_db / tol_db)`, `tau_ms` being whichever of
# `attack_ms`/`release_ms` the direction of change selects (see
# `aud.dsp.dynamics.smooth_gain_db`'s own docstring, quoted below).


def _settle_frames_by_formula(tau_ms: float, gap_db: float, tol_db: float, frame_rate_hz: float) -> int:
    """Frames after a step for a one-pole (dB-domain) curve to be within
    `tol_db` of its new target: `settle_ms = tau_ms * ln(gap_db / tol_db)`,
    converted to frames at `frame_rate_hz`. Derived, not measured -- the
    same formula this file's own recovery-timing tests already use.
    """
    settle_ms = tau_ms * math.log(gap_db / tol_db)
    return math.ceil(settle_ms / (1000.0 / frame_rate_hz))


def test_hold_latching_only_first_engaged_depth_would_fail_the_deepening_check():
    """NAMED TEST (deepening direction): a single continuous engagement
    whose target DEEPENS mid-engagement (-6 dB then -18 dB, never
    returning to unity in between). Reproduced directly this session:
    applying the first-engaged-depth-latch mutant to
    `_duck_hold_attack_release_db` freezes `held_value` at -6 dB forever,
    so the smoothed curve never reaches within `tol_db` of -18 dB at
    either the helper level (`smooth_time_axis_db`) or the public path
    (`duck_gain_surface`) -- both assertions below go RED under the
    mutant; confirmed, then reverted (`git diff`/`sha256sum` both
    byte-identical afterward -- see PR description for the transcript).

    Uses a DIFFERENT (hop, sr, n_fft) than the shallowing test below (per
    review instruction: cases must not all share one hop, sample rate, or
    n_fft).
    """
    # --- helper level: smooth_time_axis_db, lookahead=0 for an exact formula ---
    frame_rate_hz = 100.0  # 10 ms/frame, round numbers
    attack_ms, hold_ms, release_ms = 20.0, 10.0, 150.0
    depth1_db, depth2_db = -6.0, -18.0
    tol_db = 0.5
    n_pre, n_post = 100, 100
    target_db = np.concatenate([np.full(n_pre, depth1_db), np.full(n_post, depth2_db)])

    smoothed, stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=attack_ms, hold_ms=hold_ms, release_ms=release_ms, lookahead_frames=0
    )
    assert stats["engage_count"] == 1, "test setup: target never returns to unity, so this must be ONE engagement"
    assert smoothed[n_pre - 1] == pytest.approx(depth1_db), "test setup: must be fully settled at depth1 pre-transition"

    gap_db = abs(depth2_db - depth1_db)
    expected_recovery_frames = _settle_frames_by_formula(attack_ms, gap_db, tol_db, frame_rate_hz)
    recovered_frame = next(
        (f for f in range(n_pre, n_pre + n_post - 5) if np.all(np.abs(smoothed[f : f + 5] - depth2_db) < tol_db)),
        None,
    )
    assert recovered_frame is not None, (
        f"the smoothed curve never reached within {tol_db} dB of the deeper target {depth2_db} dB; "
        f"final value {smoothed[-1]:.4f} dB (a first-engaged-depth latch would freeze it at {depth1_db} dB)"
    )
    measured_recovery_frames = recovered_frame - n_pre
    assert abs(measured_recovery_frames - expected_recovery_frames) <= 2, (
        f"deepening took {measured_recovery_frames} frames to settle; expected ~{expected_recovery_frames} frames "
        f"from attack_ms={attack_ms} * ln({gap_db}/{tol_db})"
    )

    # --- public path: duck_gain_surface, a DIFFERENT n_fft/hop/sr ---
    n_fft, hop, sr = 1024, 256, 44100
    attack_ms2, hold_ms2, release_ms2 = 20.0, 10.0, 150.0
    n_bands = 4
    b = bands_module.band_edges(
        n_bands, scale="bark_peaq", f_min=20.0, f_max=min(20000.0, sr / 2.0 - 1.0), allow_extrapolation=True
    )
    n_pre2, n_post2 = 150, 150
    band_db = np.concatenate([np.full(n_pre2, depth1_db), np.full(n_post2, depth2_db)])
    band_gain_power = np.tile((10.0 ** (band_db / 10.0))[None, :], (n_bands, 1))

    bin_gain_amp, _stats2 = duck_gain_surface(
        band_gain_power, b, n_fft, hop, sr, attack_ms=attack_ms2, hold_ms=hold_ms2, release_ms=release_ms2
    )
    final_db = 20.0 * np.log10(bin_gain_amp[:, -5:].mean())
    assert abs(final_db - depth2_db) < tol_db, (
        f"duck_gain_surface's own output never settled near the deeper target {depth2_db} dB well after the "
        f"transition; got {final_db:.4f} dB (a first-engaged-depth latch would freeze it near {depth1_db} dB)"
    )


def test_hold_latching_only_first_engaged_depth_would_fail_the_shallowing_check():
    """NAMED TEST (shallowing direction, the reverse case the review also
    asked for): a single continuous engagement whose target becomes
    SHALLOWER mid-engagement (-18 dB then -6 dB, never returning to
    unity). Correct behaviour here is settled by the documented polarity
    rule this module reuses UNMODIFIED (`aud.dsp.dynamics.smooth_gain_db`'s
    own docstring: "`attack_coeff` is selected whenever `target < prev` ...
    and `release_coeff` otherwise (recovering toward 0 dB / unity)") -- the
    code's own condition is `target < prev`, not `target < 0`, so ANY move
    toward LESS reduction, including toward a shallower NON-ZERO depth and
    not only toward unity, takes the RELEASE coefficient. Correct
    behaviour is therefore to reach the shallower target on the RELEASE
    timescale, the mirror image of the deepening test above reaching its
    deeper target on the ATTACK timescale.

    Reproduced directly this session: the same first-engaged-depth-latch
    mutant freezes `held_value` at -18 dB forever, so this direction fails
    too -- both assertions below go RED under the mutant; confirmed, then
    reverted.

    Uses a DIFFERENT (hop, sr, n_fft) and different attack/hold/release
    constants than the deepening test above.
    """
    # --- helper level: smooth_time_axis_db, lookahead=0 for an exact formula ---
    frame_rate_hz = 50.0  # 20 ms/frame
    attack_ms, hold_ms, release_ms = 15.0, 25.0, 90.0
    depth1_db, depth2_db = -18.0, -6.0
    tol_db = 0.5
    n_pre, n_post = 100, 80
    target_db = np.concatenate([np.full(n_pre, depth1_db), np.full(n_post, depth2_db)])

    smoothed, stats = smooth_time_axis_db(
        target_db, frame_rate_hz, attack_ms=attack_ms, hold_ms=hold_ms, release_ms=release_ms, lookahead_frames=0
    )
    assert stats["engage_count"] == 1, "test setup: target never returns to unity, so this must be ONE engagement"
    assert smoothed[n_pre - 1] == pytest.approx(depth1_db), "test setup: must be fully settled at depth1 pre-transition"

    gap_db = abs(depth2_db - depth1_db)
    expected_recovery_frames = _settle_frames_by_formula(release_ms, gap_db, tol_db, frame_rate_hz)
    recovered_frame = next(
        (f for f in range(n_pre, n_pre + n_post - 5) if np.all(np.abs(smoothed[f : f + 5] - depth2_db) < tol_db)),
        None,
    )
    assert recovered_frame is not None, (
        f"the smoothed curve never reached within {tol_db} dB of the shallower target {depth2_db} dB; "
        f"final value {smoothed[-1]:.4f} dB (a first-engaged-depth latch would freeze it at {depth1_db} dB)"
    )
    measured_recovery_frames = recovered_frame - n_pre
    assert abs(measured_recovery_frames - expected_recovery_frames) <= 2, (
        f"shallowing took {measured_recovery_frames} frames to settle; expected ~{expected_recovery_frames} frames "
        f"from release_ms={release_ms} * ln({gap_db}/{tol_db})"
    )

    # --- public path: duck_gain_surface, a DIFFERENT n_fft/hop/sr ---
    n_fft, hop, sr = 4096, 1024, 16000
    attack_ms2, hold_ms2, release_ms2 = 80.0, 50.0, 200.0
    n_bands = 4
    b = bands_module.band_edges(
        n_bands, scale="bark_peaq", f_min=20.0, f_max=min(20000.0, sr / 2.0 - 1.0), allow_extrapolation=True
    )
    n_pre2, n_post2 = 60, 80
    band_db = np.concatenate([np.full(n_pre2, depth1_db), np.full(n_post2, depth2_db)])
    band_gain_power = np.tile((10.0 ** (band_db / 10.0))[None, :], (n_bands, 1))

    bin_gain_amp, _stats2 = duck_gain_surface(
        band_gain_power, b, n_fft, hop, sr, attack_ms=attack_ms2, hold_ms=hold_ms2, release_ms=release_ms2
    )
    final_db = 20.0 * np.log10(bin_gain_amp[:, -5:].mean())
    assert abs(final_db - depth2_db) < tol_db, (
        f"duck_gain_surface's own output never settled near the shallower target {depth2_db} dB well after the "
        f"transition; got {final_db:.4f} dB (a first-engaged-depth latch would freeze it near {depth1_db} dB)"
    )
