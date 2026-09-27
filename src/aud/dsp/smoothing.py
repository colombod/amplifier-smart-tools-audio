"""Two-axis smoothing of a ducking gain -- Step 8 of the masking/ducking
epic (issue #18): TIME (attack/hold/release) and FREQUENCY (ERB width).

***ROLE ASSIGNMENT -- READ THIS BEFORE TOUCHING ANY MATH BELOW***
-------------------------------------------------------------------
USER RULING, binding (2026-09-26): `aud.dsp.collision.collision_gains`
(Step 6, an STFT-domain, per-bin-via-bands psychoacoustic masking LP) and
`aud.dsp.gate.dynamic_eq` (Step 7, a crossover-band, time-domain dynamic EQ)
are STRUCTURALLY DIFFERENT gain mechanisms and stay that way -- this module
does NOT convert one representation into the other and does NOT merge them
into a single pipeline. It provides smoothing PRIMITIVES a caller applies
to EITHER stage's own raw gain curve, and a thin orchestrator specifically
for the STFT-domain (collision) stage, because that is the one with a
per-bin frequency axis to smooth in the first place.

Which axis applies to which stage
------------------------------------
- **Time (attack/hold/release)** is relevant to BOTH stages: any per-band
  dB curve, at any update rate, benefits from ballistics instead of being
  applied raw sample-by-sample. `smooth_time_axis_db` below is rate-
  agnostic (it takes a `frame_rate_hz`, not a fixed audio sample rate) for
  exactly this reason -- it is the same function whether the caller is
  smoothing `collision_gains`' per-STFT-frame band gain or `dynamic_eq`'s
  per-audio-sample band gain (`aud.dsp.gate._process_bands`'s own
  `smoothing=False` boundary, issue #17's own docstring note, is exactly
  the seam this function is built to fill for a `dynamic_eq` caller who
  wants real ballistics -- see that function's own docstring).
- **Frequency (ERB width)** applies ONLY to the STFT stage
  (`collision_gains`). `dynamic_eq`'s bands are wide crossover bands
  (typically 2-6 of them, Linkwitz-Riley split), not a dense per-bin
  spectrum -- there is no per-BIN frequency axis to smooth there in the
  first place. `collision_gains` computes one gain per (Bark/ERB) band,
  and that per-band step function has to be turned into a per-BIN gain
  surface before it can be multiplied onto a target's complex STFT
  spectrum -- `smooth_frequency_axis_erb_db` (operating directly on an
  arbitrary per-bin curve) and `upsample_bands_to_bins` (the band -> bin
  interpolation `collision_gains`' own output needs) are both specific to
  that surface.

Why the two named helpers this issue points at (`aud.dsp.dynamics.
smooth_gain_db` and `aud.dsp.gate._hold_attack_release_db`) are reused
DIFFERENTLY, not identically
------------------------------------------------------------------------
The issue text says: reuse `_smooth_gain_db` (dynamics.py:94) and
`_hold_attack_release_db` (gate.py:177) -- "both already accept an
arbitrary dB curve and can be applied per band." Both DO accept an
arbitrary dB curve. Only one of them is reused here UNMODIFIED, and the
reason is measured, not stylistic:

`aud.dsp.dynamics.smooth_gain_db` (renamed public for this reuse -- see
its own docstring) is used AS-IS, per band, for the attack/release core.
Its polarity is EXACTLY right for a duck's `target_db <= 0` curve (0 dB =
unity, negative = attenuated): it selects the ATTACK coefficient whenever
the curve is moving toward MORE reduction (`target < prev`), and RELEASE
whenever it is recovering toward unity -- see that function's own
docstring for the full argument, including why this is the SAME polarity
`dynamic_eq`'s and `collision_gains`' own curves use.

`aud.dsp.gate._hold_attack_release_db` is NOT reused, even via negating
its input, and this is demonstrated rather than asserted:

    >>> import numpy as np
    >>> from aud.dsp.gate import _hold_attack_release_db
    >>> target_db = np.array([-30.0, -30.0, -30.0])  # e.g. a steady, deep duck request
    >>> smoothed, _ = _hold_attack_release_db(-(-target_db), 100.0, 5.0, 50.0, 150.0)
    >>> # is_open_raw = target_db > -0.01 -- for a NEGATED duck curve (>= 0 always,
    >>> # since the duck's own curve is <= 0), this is True on almost every frame
    >>> # regardless of how deep the duck is, because the eps-threshold direction
    >>> # is calibrated to detect "near zero FROM BELOW" (a <=0 quantity approaching
    >>> # its ceiling), not "near zero FROM ABOVE" (a >=0 quantity approaching its
    >>> # floor) -- negating a <=0 curve produces a >=0 curve, and the SAME
    >>> # `target_db > -eps` test that meant "near-zero, i.e. unity" on the
    >>> # original curve now means "true for every non-negative value", i.e. the
    >>> # gate function reads the entire duck as permanently "open" and the hold
    >>> # counter never counts down. `tests/test_dsp_smoothing_mutations.py::
    >>> # test_gate_hold_attack_release_cannot_be_reused_by_negation` reproduces
    >>> # this exact failure directly against the real function.

Beyond that concrete break, `_hold_attack_release_db`'s hold mechanism
forces the smoothed target to a FIXED reference value (`0.0`) while held --
correct for a gate/expander, which has exactly one canonical "open"
value (unity) to latch onto, but wrong for a duck's hold, whose engaged
depth varies continuously frame to frame; latching to a fixed 0.0 during
a duck's hold window would force the bed back to unity for the whole hold
period, the opposite of what hold is for. Both of these are the SAME
class of mistake `aud.dsp.gate`'s own docstring already names for a
different pair of functions in the very same module ("dynamic_eq does NOT
reuse `_gate_curve`/`_expander_curve` ... both attenuate BELOW threshold,
the shape correct for cleaning a signal's OWN quiet passages, but the
OPPOSITE of what an external-key duck needs ... would silently invert the
entire feature"). `_duck_hold_attack_release_db` below is the mirror-image
counterpart this module writes instead, structurally parallel to
`_hold_attack_release_db` (same hold-counter shape, same one-pole
coefficient formula) but with both of the corrections above applied.

The one-pole coefficient formula, verified against a reference table
------------------------------------------------------------------------
`a = exp(-R / (tau * fs))`, where `R` is the hop in samples, `fs` the
sample rate, and `tau` the desired time constant in seconds -- equivalently
`exp(-hop_seconds / tau_seconds)`, which is exactly what this module's
functions compute when called with `frame_rate_hz = fs / R` (one update
per hop) and `tau_ms` in the usual way (`exp(-1 / (frame_rate_hz * tau_ms
/ 1000))`). At `hop = 512` samples / `fs = 48000` Hz (`hop_ms = 10.667`),
computed directly this session: `tau=5ms -> a=0.1184`, `20ms -> 0.5866`,
`50ms -> 0.8079`, `100ms -> 0.8988`, `300ms -> 0.9651` -- each within
0.001 of this module's own reference table, confirming the formula (not
re-derived from the reference table, computed independently and compared).

THE STFT ALREADY SMOOTHS, AND THAT SETS A HARD LIMIT -- the hop-resolution error
---------------------------------------------------------------------------------
Measured this session (`n_fft=2048`, `hop=512`, `sr=48000`, Hann analysis
window, WOLA -- see `aud.dsp.stft`): a single bin's step from 0 dB to -40 dB
between two adjacent STFT frames resynthesizes as roughly a 13-19 ms
90%->10% crossfade in the TIME domain, starting well before the first
attenuated frame's own center (the window's own smearing -- see
`tests/test_dsp_smoothing_time_aliasing.py`'s `test_stft_frame_step_
already_smooths_over_one_to_two_hops` for the exact measured figures on
this repo's own analysis/resynthesis pair). A caller requesting an
`attack_ms`/`release_ms` FASTER than one hop's own duration
(`hop_ms = hop / sr * 1000`) is asking this stage to deliver a transition
the STFT itself cannot resolve -- attempting it would silently deliver
something SLOWER than requested (the STFT's own smear dominates) with no
signal anything was wrong. `duck_gain_surface` below raises
`TransitionFasterThanHopError` (code `duck_transition_faster_than_hop`,
see `docs/01-library.md`) rather than doing that silently. This check
lives in `duck_gain_surface`, the STFT-specific orchestrator, not in
`smooth_time_axis_db` -- the generic time-axis helper has no "hop" concept
of its own when reused for `dynamic_eq`'s audio-sample-rate curves.

TRANSIENTS ARE UNDER-DUCKED unless the attack is non-causal
--------------------------------------------------------------
Because the STFT's own analysis window smears a frame's information
backward in time (see above), ducking that only starts reacting at the
transient's OWN frame arrives too late: earlier, overlapping frames still
carry the un-ducked signal. `aud` is an OFFLINE processor (`aud.dsp.stft`'s
own module docstring), so a non-causal (forward-looking) lookahead costs
nothing -- there is no real-time deadline to violate. `duck_gain_surface`
therefore always applies a forward MINIMUM lookahead
(`_lookahead_min_db`, the mirror image of `aud.dsp.limiter._lookahead_min`
and `aud.dsp.gate._lookahead_max`, both of which already establish this
exact "duplicate a few lines rather than import" idiom for the same
reason: different modules, different callers, no real coupling value)
sized to at least `ceil(n_fft / (2 * hop))` frames (covering the window's
own backward smear, `N/2` samples) PLUS however many frames the requested
attack itself needs to complete -- so the full duck is guaranteed to be
reached at or before the real transient, not merely started there. See
`tests/test_dsp_smoothing_acceptance.py::
test_transient_duck_reaches_target_depth_with_lookahead_but_not_without`
for the measured before/after numbers.

RELEASE FLOOR
-----------------
Music's own forward masking lasts well over 120 ms; a release much faster
than that recovers the bed's level before the masking the gain was
computed from has itself decayed, which is audible as the bed "poking
through" during material that is still, perceptually, covered. This
module's default `release_ms=150.0` is chosen to clear that floor (and
matches the pre-existing repo-wide default already used by
`aud.dsp.gate.gate`/`expand`) -- a caller may still request a faster
release explicitly, this is a documented default, not an enforced floor.

FREQUENCY AXIS: ERB width, not linear Hz -- three measured reasons
------------------------------------------------------------------------
(a) Musical noise from isolated per-bin outliers -- the ERB/Bark-spaced
    triangular interpolation below removes an isolated bin's discontinuity
    by construction (see `smooth_frequency_axis_erb_db`).
(b) TIME ALIASING: a per-bin gain multiplies a frame's SPECTRUM, which
    means the INVERSE transform of that gain circularly convolves with the
    frame's own time-domain content -- a sharp gain edge in frequency
    wraps energy around the frame in time. Measured this session
    (`n_fft=2048`, a 0/-60 dB edge at 1 kHz, Hann analysis + Hann synthesis
    at 75% overlap, `hop=512` -- this repo's own STFT defaults): a
    brick-wall edge left roughly -47 dB of energy beyond +-N/2 samples; a
    half-octave-wide raised-cosine (ERB-comparable width) edge left
    roughly -73 dB -- see
    `tests/test_dsp_smoothing_time_aliasing.py::
    test_smoothed_edge_reduces_time_domain_aliasing_vs_brick_wall` for the
    exact reproduced numbers (this module's own measurement, not a copy of
    the issue's reference table, though it lands in the same regime).
(c) Auditory filters integrate over roughly 1 ERB -- a gain that varies
    faster than that across frequency is finer detail than the ear can
    resolve as separate bands in the first place.

Method: `smooth_frequency_axis_erb_db` downsamples an arbitrary per-bin
dB curve to ERB/Bark bands (a weighted average using
`aud.dsp.bands.bin_band_weights`' own triangular partition-of-unity
kernel, normalised by each band's row sum since ROWS of that matrix do NOT
sum to 1 -- only COLUMNS do, by that module's own partition-of-unity
guarantee), then interpolates the band values back to bins with the SAME
kernel. Two passes of one already-tested kernel are the entire mechanism;
no separate low-pass filter is layered on top. `upsample_bands_to_bins`
is the second half of that same kernel exposed on its own, for a caller
(namely `duck_gain_surface`) that already HAS band-rate values (from
`collision_gains`) and only needs the upsampling half, not the
downsample-then-upsample round trip.

Do NOT copy the common mistake: `noisereduce` (MIT) smooths with a FIXED
`freq_mask_smooth_hz=500` -- about 14 ERB wide at 100 Hz but only 0.45 ERB
wide at 10 kHz (see `aud.dsp.bands`' own module docstring, which already
names this exact defect for the bin/band-edge convention this module
reuses unchanged).

Musical-noise measurement: Step 3's own proxy does not exist -- disclosed,
not silently substituted
------------------------------------------------------------------------
Issue #18's acceptance criterion 1 calls for "the musical-noise proxy
metric from Step 3." Step 3 is issue #13 ("verification harness"), whose
own acceptance text calls for building "a musical-noise proxy metric" as
part of a two-track rendered-probe harness in a SEPARATE tool, `aud-mix`.
Checked directly this session: `aud-mix` is scaffolded only (one commit,
"Scaffold aud-mix with smart-tool-creator") and contains no such metric;
`aud`'s own test suite (including `tests/test_dsp_collision_acceptance.py`,
the closest existing harness) contains no metric of this kind either. Step
3's own proxy metric was never built. Per this task's own instruction ("if
it does not exist, STOP and report; do not silently invent one"), that gap
is reported here rather than quietly filled.

The literature DOES supply a named, checkable alternative, and issue text
explicitly permits using it if verified against its source: the KURTOSIS
RATIO (Saruwatari et al., "Musical Noise Analysis for Bayesian
Minimum Mean-Square Error Speech Amplitude Estimators," Interspeech 2013,
eq. 1 and eq. 3 -- fetched and read directly this session, not taken on
citation alone): `kurtosis = mu_4 / mu_2**2` where `mu_m` is the m-th raw
(non-centered) moment of the signal's POWER-SPECTRAL values (not the
waveform, and not a centered/"excess" kurtosis); `kurtosis_ratio =
kurt(processed) / kurt(observed)`, and "this measure increases as the
amount of generated musical noise increases" (quoted from eq. 3's own
surrounding text). `tests/musical_noise.py` implements exactly these two
equations, cites the paper directly in its own docstring, and is used by
`tests/test_dsp_smoothing_acceptance.py`'s acceptance-criterion-1 test.
This is a NEWLY-INTRODUCED metric for THIS PR, not Step 3's own (which
does not exist) -- recorded here so nobody mistakes one for the other.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from aud.dsp import bands as bands_module
from aud.dsp.dynamics import smooth_gain_db

__all__ = [
    "DUCK_ENGAGE_EPS_DB",
    "TransitionFasterThanHopError",
    "duck_gain_surface",
    "smooth_frequency_axis_erb_db",
    "smooth_time_axis_db",
    "upsample_bands_to_bins",
]

_EPS_POWER = 1e-30

#: Tolerance, in dB, for treating a duck's raw target curve as "at unity"
#: (mirrors `aud.dsp.gate._OPEN_EPS_DB`'s role -- floating-point slack
#: only, not a perceptual threshold).
DUCK_ENGAGE_EPS_DB = 0.01


class TransitionFasterThanHopError(ValueError):
    """`attack_ms`/`release_ms` requests a transition faster than the STFT
    hop this gain surface is computed at can resolve.

    Not raised as an `AudError` directly: like `aud.dsp.edit.
    CrossfadeExceedsGapError`, `dsp/` modules take arrays and parameters
    and return arrays, they do not raise user-facing errors (AGENTS.md
    #8). `collision_gains`/`dynamic_eq`/this module are not yet wired to
    `aud.lib` (that wiring is Step 11, issue #21, still open) -- once it
    is, the boundary layer maps this to `AudError(code="duck_transition_
    faster_than_hop")`, the same pattern `crossfade_exceeds_gap` already
    uses. The `code` attribute is carried directly on this exception
    (rather than only implied by the class name, as `CrossfadeExceedsGapError`
    does) so a caller -- or a test -- can assert on it without needing to
    know the future `aud.lib` mapping.
    """

    code = "duck_transition_faster_than_hop"

    def __init__(self, *, name: str, requested_ms: float, hop_ms: float) -> None:
        self.requested_ms = requested_ms
        self.hop_ms = hop_ms
        super().__init__(
            f"{name}={requested_ms} ms is faster than this stage's own hop ({hop_ms:.3f} ms); "
            "the STFT's own analysis/synthesis window already smears a frame's content over "
            "roughly one hop, so a transition faster than that cannot be delivered -- it would "
            "silently render slower than requested with no signal anything was wrong. Request "
            f"{name} >= {hop_ms:.3f} ms, or use a smaller hop."
        )


def _lookahead_min_db(target_db: np.ndarray, window: int) -> np.ndarray:
    """`target_la[n] = min(target_db[n : n + window])` -- forward-looking.

    Duplicated (not imported) from the same idiom `aud.dsp.limiter.
    _lookahead_min` and `aud.dsp.gate._lookahead_max` already establish,
    for the same stated reason those two give for duplicating each other:
    "a handful of lines... callers apply it to different signals for
    different reasons, and sharing it would couple two otherwise
    independent dsp modules to each other's internals" (`aud.dsp.gate`'s
    own docstring). This one is the MINIMUM direction (mirrors the
    limiter's own forward minimum exactly, not the gate's forward maximum):
    a duck anticipates an upcoming DIP (more negative target), the same
    event a limiter anticipates, not an upcoming RISE (what a gate/
    expander's opening anticipates).
    """
    if window <= 1:
        return target_db
    origin = -(window // 2)
    return ndimage.minimum_filter1d(target_db, size=window, mode="nearest", origin=origin)


def _duck_hold_attack_release_db(
    target_db: np.ndarray,
    frame_rate_hz: float,
    hold_ms: float,
) -> tuple[np.ndarray, int]:
    """Hold-extension for a duck curve: latches the most recently ENGAGED
    (target_db < -`DUCK_ENGAGE_EPS_DB`) value through brief returns to
    unity, for `hold_ms`, before letting the raw curve drive again.

    Structurally parallel to `aud.dsp.gate._hold_attack_release_db`'s own
    hold-counter shape, but BOTH corrected for a duck's polarity -- see
    this module's own docstring, "Why the two named helpers... are reused
    DIFFERENTLY": the engagement test is inverted (`< -eps`, not `> -eps`)
    and the held value is the LAST OBSERVED engaged value (duck depth
    varies continuously), not a fixed reference constant (gate's fixed
    `0.0` is correct only because gate/expander have exactly one canonical
    "open" value to latch onto).

    Returns:
        (held_db, engage_count) -- `held_db` is `target_db` with brief
        gaps filled in by the hold latch; `engage_count` is how many times
        the held-engaged state transitioned from released to engaged (the
        duck's own analogue of `aud.dsp.gate`'s `open_count`, diagnostic
        for a caller checking whether `hold_ms` is too short).
    """
    hold_frames = max(0, round(hold_ms / 1000.0 * frame_rate_hz))
    is_engaged_raw = target_db < -DUCK_ENGAGE_EPS_DB

    held = np.empty_like(target_db)
    hold_counter = 0
    held_value = 0.0
    engage_count = 0
    was_engaged = False
    for n in range(target_db.shape[0]):
        if is_engaged_raw[n]:
            hold_counter = hold_frames
            held_value = float(target_db[n])
            is_held_engaged = True
        elif hold_counter > 0:
            hold_counter -= 1
            is_held_engaged = True
        else:
            is_held_engaged = False

        if is_held_engaged and not was_engaged:
            engage_count += 1
        was_engaged = is_held_engaged

        held[n] = held_value if is_held_engaged else target_db[n]

    return held, engage_count


def smooth_time_axis_db(
    target_db: np.ndarray,
    frame_rate_hz: float,
    *,
    attack_ms: float = 20.0,
    hold_ms: float = 50.0,
    release_ms: float = 150.0,
    lookahead_frames: int = 0,
) -> tuple[np.ndarray, dict]:
    """TIME-axis smoothing of one band's raw dB gain curve: non-causal
    lookahead, then hold, then asymmetric attack/release ballistics.

    Rate-agnostic: `frame_rate_hz` is "updates per second" for whatever
    curve is passed -- an STFT frame rate (`sr / hop`) for `collision_
    gains`' output, or the audio sample rate itself for `dynamic_eq`'s
    per-sample curve (see this module's own docstring, "Which axis applies
    to which stage"). Works on a single band's 1-D curve; a caller with a
    `(n_bands, n_frames)` array loops over axis 0 (as `aud.dsp.gate.
    _process_bands` already does per band for its own ballistics).

    Args:
        target_db: `(n_frames,)`, `<= 0` -- 0 = unity, negative = duck
            depth requested, the same convention `dynamic_eq`/
            `collision_gains` both use.
        frame_rate_hz: Updates per second (see above). Must be > 0.
        attack_ms: Time to reach a deeper duck once one is requested. >= 0.
        hold_ms: How long a duck stays engaged through a brief return to
            unity before release is allowed to start recovering it. >= 0.
        release_ms: Time to recover toward unity once hold expires. >= 0.
            Default 150.0 clears the forward-masking floor (see module
            docstring's "RELEASE FLOOR").
        lookahead_frames: Non-causal lookahead, in frames of this curve's
            own rate -- 0 (default) disables it (useful for a caller who
            has already lookahead-adjusted the curve, or genuinely wants a
            causal result). `duck_gain_surface` computes and passes this
            explicitly for the STFT stage; a `dynamic_eq` caller wanting
            the same anticipation would size it in audio samples instead.

    Returns:
        (smoothed_db, stats) where stats = {
            "engage_count": int, "attack_ms": float, "hold_ms": float,
            "release_ms": float, "lookahead_frames": int,
        }

    Raises:
        ValueError: `frame_rate_hz <= 0`; any of `attack_ms`/`hold_ms`/
            `release_ms`/`lookahead_frames` is negative.
    """
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be > 0; got {frame_rate_hz}")
    for name, value in (("attack_ms", attack_ms), ("hold_ms", hold_ms), ("release_ms", release_ms)):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if lookahead_frames < 0:
        raise ValueError(f"lookahead_frames must be >= 0, got {lookahead_frames}")

    target_db = np.asarray(target_db, dtype=np.float64)
    anticipated = _lookahead_min_db(target_db, max(1, int(lookahead_frames))) if lookahead_frames > 0 else target_db
    held, engage_count = _duck_hold_attack_release_db(anticipated, frame_rate_hz, hold_ms)
    smoothed_db = smooth_gain_db(held, frame_rate_hz, attack_ms, release_ms)

    stats = {
        "engage_count": engage_count,
        "attack_ms": float(attack_ms),
        "hold_ms": float(hold_ms),
        "release_ms": float(release_ms),
        "lookahead_frames": int(lookahead_frames),
    }
    return smoothed_db, stats


def upsample_bands_to_bins(band_values: np.ndarray, bands: dict, n_fft: int, sr: float) -> np.ndarray:
    """Interpolate `(n_bands, ...)` band-rate values onto the STFT bin grid.

    Reuses `aud.dsp.bands.bin_band_weights`' triangular partition-of-unity
    kernel in the OPPOSITE direction from its original purpose (bin -> band
    energy SUMMATION): `bin_value[k] = sum_b weights[b, k] * band_value[b]`.
    This is a genuine convex combination, not a coincidental reuse of the
    same array shape, because `sum_b weights[b, k] == 1` for every bin `k`
    (asserted by `bin_band_weights`' own tests) -- so every bin's value is
    a weighted AVERAGE of its overlapping bands' values, smoothly
    interpolated rather than stepped.

    Args:
        band_values: `(n_bands, ...)` -- e.g. a smoothed dB gain curve,
            `(n_bands, n_frames)`.
        bands: As returned by `aud.dsp.bands.band_edges`.
        n_fft: Must match whatever produced the spectrum this will apply to.
        sr: Sample rate in Hz.

    Returns:
        `(n_bins, ...)`, same trailing shape as `band_values`.

    Raises:
        ValueError: (via `bin_band_weights`) malformed `bands`, or
            `band_values.shape[0] != bands["n_bands"]`.
    """
    band_values = np.asarray(band_values, dtype=np.float64)
    weights = bands_module.bin_band_weights(bands, n_fft, sr)  # (n_bands, n_bins)
    if band_values.shape[0] != weights.shape[0]:
        raise ValueError(
            f"band_values' leading axis has length {band_values.shape[0]}, but bands has n_bands={weights.shape[0]}"
        )
    return np.tensordot(weights, band_values, axes=([0], [0]))


def smooth_frequency_axis_erb_db(bin_gain_db: np.ndarray, bands: dict, n_fft: int, sr: float) -> np.ndarray:
    """ERB/Bark-width smoothing of an ARBITRARY per-bin dB gain curve.

    Two passes of `aud.dsp.bands.bin_band_weights`' own triangular
    partition-of-unity kernel: downsample bins to bands (a weighted
    average -- ROWS of `weights` do NOT sum to 1 in general, only COLUMNS
    do, so this division by each band's own row sum is required and is
    NOT the same operation as `aud.dsp.bands.band_energy`, which sums
    rather than averages), then interpolate the band values back to bins
    with `upsample_bands_to_bins` (the same kernel's other half). See
    module docstring's "FREQUENCY AXIS" section for why ERB/Bark width
    rather than a fixed linear-Hz kernel, and for the measured time-domain
    aliasing this smoothing removes.

    Args:
        bin_gain_db: `(n_bins, ...)`, log-domain (dB) gain -- smoothing
            happens in dB (log-gain), per the issue's own instruction, not
            in linear power/amplitude.
        bands: As returned by `aud.dsp.bands.band_edges`.
        n_fft: Must match whatever produced the spectrum this will apply to.
        sr: Sample rate in Hz.

    Returns:
        `(n_bins, ...)`, same shape as `bin_gain_db`.
    """
    bin_gain_db = np.asarray(bin_gain_db, dtype=np.float64)
    weights = bands_module.bin_band_weights(bands, n_fft, sr)  # (n_bands, n_bins)
    if bin_gain_db.shape[0] != weights.shape[1]:
        raise ValueError(
            f"bin_gain_db's leading axis has length {bin_gain_db.shape[0]}, but bands/n_fft imply "
            f"n_bins={weights.shape[1]}"
        )
    row_sums = weights.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("bin_band_weights produced a band with zero total weight; degenerate bands/n_fft/sr")
    trailing_shape = (1,) * (bin_gain_db.ndim - 1)
    band_avg_db = np.tensordot(weights, bin_gain_db, axes=([1], [0])) / row_sums.reshape(-1, *trailing_shape)
    return upsample_bands_to_bins(band_avg_db, bands, n_fft, sr)


def duck_gain_surface(
    band_gain_power: np.ndarray,
    bands: dict,
    n_fft: int,
    hop: int,
    sr: float,
    *,
    attack_ms: float = 20.0,
    hold_ms: float = 50.0,
    release_ms: float = 150.0,
) -> tuple[np.ndarray, dict]:
    """The Step 8 orchestrator for the STFT-based (`collision_gains`) gain
    stage: TIME-axis ballistics per band, then FREQUENCY-axis (ERB) upsample
    to bins, producing a per-bin AMPLITUDE gain ready to multiply directly
    onto a target's complex STFT spectrum (`aud.dsp.stft.analyze`'s own
    `(n_bins, n_frames)`/`(n_bins, n_channels, n_frames)` convention --
    reshape this surface's `(n_bins, n_frames)` to add a channel axis if
    needed).

    Does NOT itself call `collision_gains` or touch `dynamic_eq` -- per the
    binding ruling (module docstring), those two gain mechanisms stay
    separate; this is the smoothing/upsampling stage a caller places AFTER
    `collision_gains` and BEFORE multiplying onto the target spectrum.

    Args:
        band_gain_power: `(n_bands, n_frames)`, LINEAR POWER in `[g_min,
            1]` -- `collision_gains`' own `CollisionResult.gain` convention
            (or any caller-built array in that same convention).
        bands: As returned by `aud.dsp.bands.band_edges` -- must match
            whatever produced `band_gain_power`.
        n_fft: STFT window length, samples -- must match the `analyze`/
            `resynthesize` call this surface will be applied within.
        hop: STFT hop, samples -- same requirement.
        sr: Sample rate in Hz.
        attack_ms: See `smooth_time_axis_db`. Must be `>= hop_ms`
            (`hop / sr * 1000`) -- see module docstring's "hop-resolution
            error".
        hold_ms: See `smooth_time_axis_db`.
        release_ms: See `smooth_time_axis_db`. Must be `>= hop_ms`.

    Returns:
        (bin_gain_amplitude, stats) where `bin_gain_amplitude` is
        `(n_bins, n_frames)`, real, non-negative, and `stats` is
        {
            "hop_ms": float, "lookahead_frames": int,
            "bands": [per-band `smooth_time_axis_db` stats dict, ...],
        }

    Raises:
        TransitionFasterThanHopError: `attack_ms` or `release_ms` is
            faster than `hop_ms` can resolve.
        ValueError: shape mismatch between `band_gain_power` and `bands`;
            non-positive `n_fft`/`hop`/`sr`.
    """
    band_gain_power = np.asarray(band_gain_power, dtype=np.float64)
    if band_gain_power.ndim != 2 or band_gain_power.shape[0] != bands["n_bands"]:
        raise ValueError(
            f"band_gain_power must be shape (n_bands={bands['n_bands']}, n_frames); got {band_gain_power.shape}"
        )
    if n_fft <= 0 or hop <= 0 or sr <= 0:
        raise ValueError(f"n_fft, hop and sr must all be > 0; got n_fft={n_fft}, hop={hop}, sr={sr}")

    hop_ms = hop / sr * 1000.0
    frame_rate_hz = sr / hop
    for name, value in (("attack_ms", attack_ms), ("release_ms", release_ms)):
        if value < hop_ms:
            raise TransitionFasterThanHopError(name=name, requested_ms=value, hop_ms=hop_ms)

    attack_frames = math.ceil(attack_ms / 1000.0 * frame_rate_hz)
    stft_smear_frames = math.ceil(n_fft / (2.0 * hop))
    lookahead_frames = stft_smear_frames + attack_frames

    n_bands = band_gain_power.shape[0]
    band_gain_db = 10.0 * np.log10(np.maximum(band_gain_power, _EPS_POWER))
    smoothed_db = np.empty_like(band_gain_db)
    band_stats: list[dict] = []
    for b in range(n_bands):
        smoothed_db[b], stats_b = smooth_time_axis_db(
            band_gain_db[b],
            frame_rate_hz,
            attack_ms=attack_ms,
            hold_ms=hold_ms,
            release_ms=release_ms,
            lookahead_frames=lookahead_frames,
        )
        band_stats.append(stats_b)

    bin_gain_db = upsample_bands_to_bins(smoothed_db, bands, n_fft, sr)
    bin_gain_amplitude = 10.0 ** (bin_gain_db / 20.0)

    stats = {
        "hop_ms": float(hop_ms),
        "lookahead_frames": int(lookahead_frames),
        "bands": band_stats,
    }
    return bin_gain_amplitude, stats
