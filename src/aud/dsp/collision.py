"""The collision measure -- Step 6 of the masking/ducking epic (see `AGENTS.md`
and `docs/DESIGN-ENVELOPE.md`): given two signals' own per-band energy over
time, compute the per-band, per-frame TARGET gain that clears room for one
signal (the KEY) inside the other (the TARGET) -- without touching a band the
TARGET never occupied in the first place.

***ROLE ASSIGNMENT -- READ THIS BEFORE TOUCHING ANY MATH BELOW***
-------------------------------------------------------------------
This tool's own vocabulary (see issue #9's acceptance text and
`docs/ARCHITECTURE.md`): the TARGET is the signal being processed and gained
down (typically a music bed); the KEY is the external sidechain signal that
drives the decision (typically speech). That is a naming convention, not yet
a masking role -- the masking role is the opposite of what a first read
suggests:

    The TARGET (music bed) is the MASKER.  It is loud, broadband, and it is
    what makes the KEY hard to hear.
    The KEY (speech) is the MASKEE.  It is what needs room made for it.

So the masking THRESHOLD -- the energy level below which a masker renders
another sound inaudible -- is computed FROM THE TARGET (the masker), spread
across frequency by the same asymmetric PEAQ kernel `aud.dsp.masking`
already implements and validates. The KEY's own per-band energy is what that
threshold is compared AGAINST, band by band, to decide how far down the
target must go. This is the cross-adaptive structure the sidechain/ducking
literature calls out explicitly (Zölzer et al., DAFX; QMUL's cross-adaptive
processing line of work): each source's processing is driven by a function
of every OTHER source's signal, `T'_n = H(sum_{i != n} s_i)` -- here n=2,
one masker.

Getting this backwards -- computing a threshold from the KEY and comparing
it against the TARGET -- silently degrades into ordinary multiband gating:
"cut the target in whatever band the key happens to be loud in," full stop,
with no reference to whether the target even has energy there to attenuate,
and no reference to whether the target's OWN energy actually masks the key
enough to warrant cutting it. Both directions can look "plausible" on
typical program material (voice-over-music is forgiving), which is exactly
why this step is the one most likely to be silently wrong -- see this
module's acceptance tests, all of which are written to fail loudly under
that reversed assignment (see `tests/test_dsp_collision_mutations.py`).

The formulation, and why it is solvable
------------------------------------------
Given, per STFT frame:

    E_m[k]  target/masker per-band energy (linear power), from `aud.dsp.bands`
    E_s[b]  key/maskee per-band energy (linear power), same band structure
    S[b,k]  spreading matrix: how much of masker band k's power reaches
            band b once spread by the asymmetric PEAQ kernel -- LEVEL
            DEPENDENT (the upper skirt steepens with the masker's own level)
            but HELD FIXED within this frame, calibrated from the frame's
            actual (pre-gain) E_m -- see `spreading_matrix` below.
    M[b]    a fixed, signal-independent offset (ITU-R BS.1387 eq. 24-26,
            reused from `aud.dsp.masking` -- see `masking_offset` below)
    w[k]    SII (ANSI S3.5-1997) band-importance weight -- see
            `sii_band_importance` below

the masking threshold this step must keep the key clear of is

    T[b] = M[b] * sum_k S[b,k] * G[k] * E_m[k]

and the requirement ("the key clears the target's threshold by `margin_db`
dB in every key-active band") is

    sum_k S[b,k] * E_m[k] * G[k]  <=  E_s[b] / 10**(margin_db / 10) / M[b]

for every band b where the key is active (see `key_active_floor`). This is
LINEAR in G -- S, M, and E_m are all frame constants once G is the unknown
-- so the per-frame problem

    minimise    sum_k w[k] * (1 - G[k])
    subject to  the constraint above, for every key-active b
                0 <= G_min <= G[k] <= 1

is a small linear program, solved once per frame with `scipy.optimize.
linprog` (BSD; already a dependency). No special case exists for "the
target has no energy in band k": if E_m[k] is at (or near) zero, band k's
column contributes ~0 to every constraint's left-hand side regardless of
G[k], so no constraint binds it, and the objective (which strictly prefers
G[k]=1, i.e. no cut, whenever nothing forces it lower) leaves it there. See
`tests/test_dsp_collision_acceptance.py::
test_key_energy_in_target_empty_band_is_not_attenuated` for the measurement
on rendered audio, not on the internally-computed gain curve.

Combination rule: linear, not PEAQ's 0.4-power pooling -- an explicit,
cited deviation
------------------------------------------------------------------------
`aud.dsp.masking.spreading_function_peaq` (Step 5) combines multiple
maskers' spread contributions in a 0.4-power domain (ITU-R BS.1387 eq. 166)
-- a deliberate, cited property of THAT model, not a free parameter. That
combination rule is nonlinear in each masker's own energy, which would make
`T[b]` nonlinear in `G` and destroy the LP above. This module instead
combines contributions ADDITIVELY in linear power -- the older
Schroeder/Atal/Hall (JASA 66:1647, Dec 1979) convention, which is on
`docs/DESIGN-ENVELOPE.md`'s ALLOW-list in its own right, not a distortion of
PEAQ. `spreading_matrix` gets this by reusing `spreading_function_peaq`
itself with `exponent=1.0` (its docstring: "Exposed for experimentation, not
because the cited model treats it as free") -- so the asymmetric slope
SHAPE, the Bark-scale validation, and the unit-preserving normalisation are
100% Step 5's already-tested code; only the cross-masker pooling rule
differs, and that difference is this module's entire reason to exist as a
separate step rather than a one-line call into Step 5.

Inherited known deviation: this module CONSUMES the over-prediction, it
does not just carry it
------------------------------------------------------------------------
`M[b]` and `S[b,k]` both come straight from `aud.dsp.masking`, which
over-predicts masking almost everywhere -- no outer/middle-ear `W[k]`
weighting, no internal-noise floor (see that module's own "Known
deviation" section for the measured per-band table, e.g. ~2.75 dB/Bark
too shallow a slope at 100 Hz). This module is the first place that
over-prediction turns into an audible decision rather than a number: a
masking threshold read as higher than the true perceptual one means this
step's LP believes the key is adequately covered sooner than it truly is,
so `collision_gains` ducks LESS than a fully eq.-7/eq.-18-compliant model
would ask for. That is the safe-by-accident direction (under-ducking, not
over-ducking), but a caller relying on `margin_db` for a hard
intelligibility guarantee should budget extra margin to compensate rather
than assume the computed threshold is exact.

Why SII importance, not key energy, weights the objective -- a second,
independent reason the naive approach is wrong
------------------------------------------------------------------------
A ducker whose cost function is driven by the KEY's own energy spends its
attenuation budget wherever normal-effort speech happens to be loudest --
250-500 Hz (ANSI S3.5-1997 Table 3 standard speech spectrum: ~34.5 dB SPL at
315 Hz, falling to ~17.3 dB by 2 kHz) -- which is NOT where speech
intelligibility actually lives. The same standard's band-importance function
(Table 3, average-speech-material column) puts 59.2% of total importance in
1-4 kHz and 78.7% in 500 Hz-4 kHz; the single most important one-third-octave
band is 2 kHz (0.0898 of the total). Weighting the objective by key energy
would therefore cut hardest exactly where cutting buys the least
intelligibility and spends the most of the target's own body. `w[k]` below
is built from that same published, 1997 (pre-2017-priority-date, ALLOW-list)
table, not from either signal's own energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from aud.dsp.masking import masking_threshold, spreading_function_peaq

__all__ = [
    "CollisionResult",
    "collision_gains",
    "masking_offset",
    "sii_band_importance",
    "spreading_matrix",
]

_EPS = 1e-30

#: ANSI S3.5-1997 Table 3, "average speech" band-importance function --
#: one-third-octave centre frequencies (Hz). This is the ANSI-preferred
#: series (matches ISO 266): the widely-copied 3159 Hz figure that appears
#: in at least one third-party transliteration of this same table is a
#: typo for the standard 3150 Hz preferred centre; corrected here rather
#: than propagated (see module docstring's own precedent in
#: `aud.dsp.masking` for correcting a copied error instead of repeating it).
_SII_MID_BAND_HZ = np.array(
    [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000],
    dtype=np.float64,
)

#: ANSI S3.5-1997 Table 3, band-importance function I_i, "average speech
#: material" column (the standard's default/most commonly cited column).
#: Verified, this session: sums to 1.0 exactly; peak 0.0898 at 2000 Hz;
#: bands 1000-4000 Hz sum to 59.25% of the total; bands 500-4000 Hz sum to
#: 78.67% -- both figures matching (to rounding) the commonly cited "59.2%"
#: / "78.7%" figures for this table.
_SII_IMPORTANCE_STANDARD = np.array(
    [
        0.0083,
        0.0095,
        0.015,
        0.0289,
        0.044,
        0.0578,
        0.0653,
        0.0711,
        0.0818,
        0.0844,
        0.0882,
        0.0898,
        0.0868,
        0.0844,
        0.0771,
        0.0527,
        0.0364,
        0.0185,
    ],
    dtype=np.float64,
)

#: One-third-octave bandwidth, as a fraction of centre frequency:
#: `2**(1/6) - 2**(-1/6) ~= 0.2316`. Cross-checked against the standard's
#: own tabulated "Column 3" bandwidth-in-dB figures (which convert to a
#: 0.2296-0.2334 fraction-of-centre range across the 18 bands, versus this
#: single analytic constant) -- close enough that using the closed form
#: here (rather than an 18th data column) does not change which bands
#: dominate the resulting weights.
_THIRD_OCTAVE_BANDWIDTH_FRACTION = 2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0)


def sii_band_importance(bands: dict) -> np.ndarray:
    """SII (ANSI S3.5-1997) band-importance weights for `bands`, NOT for
    either the target's or the key's own energy -- see module docstring's
    "Why SII importance, not key energy" section for why that distinction
    is load-bearing.

    Builds an importance-per-Hz DENSITY from the standard's 18 tabulated
    one-third-octave points (density = table importance / one-third-octave
    bandwidth at that centre, so wide and narrow reference bands are put on
    a common footing), interpolates that density log-log across frequency
    (clamped -- constant, not extrapolated -- outside the tabulated 160-8000
    Hz range, the same convention `aud.dsp.masking.threshold_in_quiet` uses
    for its own ceiling), then integrates it back across EACH of `bands`'
    own (possibly wider or narrower) bands by multiplying the density at
    that band's centre by that band's own Hz width. The result is
    normalised to sum to 1 -- an arbitrary but convenient scale, since only
    the RELATIVE weights matter to `collision_gains`' objective.

    Args:
        bands: As returned by `aud.dsp.bands.band_edges` -- needs
            `"centers_hz"` and `"edges_hz"` only; no scale requirement (this
            function does not touch Bark space).

    Returns:
        `(n_bands,)`, non-negative, summing to 1.0.
    """
    centers_hz = np.asarray(bands["centers_hz"], dtype=np.float64)
    edges_hz = np.asarray(bands["edges_hz"], dtype=np.float64)
    widths_hz = np.diff(edges_hz)

    table_bandwidth_hz = _SII_MID_BAND_HZ * _THIRD_OCTAVE_BANDWIDTH_FRACTION
    density = _SII_IMPORTANCE_STANDARD / table_bandwidth_hz

    log_f_table = np.log(_SII_MID_BAND_HZ)
    log_density_table = np.log(density)
    log_f_query = np.log(np.clip(centers_hz, _SII_MID_BAND_HZ[0], _SII_MID_BAND_HZ[-1]))
    log_density_query = np.interp(log_f_query, log_f_table, log_density_table)
    density_query = np.exp(log_density_query)

    raw_weights = density_query * widths_hz
    total = raw_weights.sum()
    if total <= 0:
        raise ValueError("sii_band_importance produced a non-positive total weight; check `bands`' edges_hz")
    return raw_weights / total


def masking_offset(bands: dict, playback_level_db_spl: float = 92.0) -> np.ndarray:
    """`M[b]` -- the fixed, per-band masking-threshold offset (ITU-R
    BS.1387 eq. 24-26), reused from `aud.dsp.masking` rather than
    re-derived, via the invariant `masking_threshold`'s own docstring
    states and this module's tests pin directly: the ratio
    `masking_threshold(P, ..., apply_absolute_threshold=False) /
    spreading_function_peaq(P, ...)` equals `10**(-m_db/10)` for ANY
    positive `P`, so it can be recovered from Step 5's PUBLIC API alone --
    no private import, no duplicated formula.

    `bands` must be PEAQ-scale (`bands["scale"] == "bark_peaq"`) -- both
    calls below require it; see `aud.dsp.masking`'s own module docstring
    for why the scale is not swappable.

    Returns:
        `(n_bands,)`, in `(0, 1]` -- a multiplicative reduction applied to
        the spread excitation to get the masking threshold.
    """
    n_bands = bands["n_bands"]
    reference = np.ones(n_bands, dtype=np.float64)
    spread = spreading_function_peaq(reference, bands, playback_level_db_spl=playback_level_db_spl, exponent=1.0)
    masked = masking_threshold(
        reference,
        bands,
        playback_level_db_spl=playback_level_db_spl,
        apply_absolute_threshold=False,
        exponent=1.0,
    )
    return masked / spread


def spreading_matrix(
    target_band_energy: np.ndarray,
    bands: dict,
    playback_level_db_spl: float = 92.0,
    energy_floor: float = 1e-12,
) -> np.ndarray:
    """`S[b, k, ...]` -- the linear-power spreading matrix from EVERY
    masker band `k` to every band `b`, calibrated per-frame from the
    target's OWN (pre-gain) energy -- see module docstring's "The
    formulation" and "Combination rule" sections for why this is linear and
    why it differs from Step 5's own aggregate function.

    Built by calling `aud.dsp.masking.spreading_function_peaq` once PER
    MASKER BAND `k` (not once per frame: every call is vectorised across
    whatever trailing/frame axes `target_band_energy` carries), with an
    impulse-like power vector -- band `k` at its real energy, every other
    band at `energy_floor` (negligible, but strictly positive, since Step 5
    rejects an exact zero) -- at `exponent=1.0` so the aggregate call
    degenerates to exactly this single masker's own linear contribution
    (see module docstring). Column `k` of the result is then that
    contribution divided back by band `k`'s own real energy, recovering the
    per-unit-energy spreading kernel while the SHAPE (in particular the
    level-dependent upper-slope term) stays calibrated to the real,
    pre-gain level -- "held fixed within a frame."

    Args:
        target_band_energy: `(n_bands, ...)`, linear power, non-negative,
            finite (as `aud.dsp.bands.band_energy` returns).
        bands: PEAQ-scale, as `spreading_function_peaq` requires.
        playback_level_db_spl: Forwarded to `spreading_function_peaq`.
        energy_floor: The negligible placeholder energy for every
            non-target band in each impulse call. Must be strictly
            positive and several orders of magnitude below any energy this
            module will realistically see; the default (1e-12 normalised
            power, i.e. -120 dBFS) satisfies that for any signal this tool
            would be asked to process.

    Returns:
        `(n_bands, n_bands) + target_band_energy.shape[1:]` -- axis 0 is
        the receiving band `b`, axis 1 is the masking band `k`, matching
        `S[b, k]` in the module docstring's formulation.
    """
    target_band_energy = np.asarray(target_band_energy, dtype=np.float64)
    n_bands = bands["n_bands"]
    if target_band_energy.shape[0] != n_bands:
        raise ValueError(
            f"target_band_energy's leading axis has length {target_band_energy.shape[0]}, "
            f"but bands has n_bands={n_bands}"
        )
    if np.any(target_band_energy < 0) or not np.all(np.isfinite(target_band_energy)):
        raise ValueError("target_band_energy must be non-negative and finite")
    if energy_floor <= 0:
        raise ValueError(f"energy_floor must be > 0; got {energy_floor}")

    trailing_shape = target_band_energy.shape[1:]
    floored_energy = np.maximum(target_band_energy, energy_floor)

    columns = np.empty((n_bands, n_bands, *trailing_shape), dtype=np.float64)
    for k in range(n_bands):
        impulse = np.full((n_bands, *trailing_shape), energy_floor, dtype=np.float64)
        impulse[k] = floored_energy[k]
        response = spreading_function_peaq(impulse, bands, playback_level_db_spl=playback_level_db_spl, exponent=1.0)
        columns[:, k] = response / floored_energy[k]

    return columns


@dataclass
class CollisionResult:
    """The per-frame target gain `collision_gains` computes, and enough of
    its own working state to audit or debug one frame without re-solving."""

    gain: np.ndarray  #: `(n_bands, n_frames)`, linear power, in `[g_min, 1]`.
    active_bands: np.ndarray  #: `(n_bands, n_frames)` bool -- which bands carried a constraint, per frame.
    solver_status: list[str]  #: length `n_frames`; `scipy.optimize.linprog`'s own `.message` per frame.


def collision_gains(
    target_band_energy: np.ndarray,
    key_band_energy: np.ndarray,
    bands: dict,
    margin_db: float = 3.0,
    g_min: float = 0.0,
    key_active_floor: float = 1e-9,
    key_active_relative_db: float = -40.0,
    playback_level_db_spl: float = 92.0,
) -> CollisionResult:
    """The collision measure: per-band, per-frame TARGET gain that keeps
    the KEY clear of the TARGET's masking threshold by `margin_db`, solved
    as the linear program in this module's docstring.

    ROLE ASSIGNMENT (see module docstring for the full argument):
    `target_band_energy` is the MASKER -- the signal being gained down.
    `key_band_energy` is the MASKEE -- the signal being protected. Swapping
    these two arguments silently reverses the entire measure; see
    `tests/test_dsp_collision_mutations.py` for a test that fails loudly
    when that swap happens.

    Args:
        target_band_energy: `(n_bands, n_frames)`, linear power -- the
            masker's (target's) own per-band energy, one column per STFT
            frame (as `aud.dsp.bands.band_energy` returns per-frame).
        key_band_energy: `(n_bands, n_frames)`, linear power, same band
            structure and frame axis as `target_band_energy` -- the
            maskee's (key's) own per-band energy.
        bands: PEAQ-scale (`bands["scale"] == "bark_peaq"`), as
            `aud.dsp.bands.band_edges(..., scale="bark_peaq")` returns --
            required transitively by `spreading_matrix`/`masking_offset`.
        margin_db: How far below the target's masking threshold the key
            must sit, in every key-active band, before this measure stops
            asking for more attenuation. Higher = more headroom = more
            attenuation where target and key collide.
        g_min: Lower bound on the returned gain (linear power). Default
            0.0 -- UNBOUNDED depth. This is deliberate: depth limiting is
            Step 7's job (see module docstring and `AGENTS.md`'s epic
            note), not this step's; `g_min=0.0` also guarantees the LP is
            always feasible (`G=0` everywhere trivially satisfies every
            constraint), so this function never raises for a solvable
            request. A caller that wants Step 6's math pre-limited can
            pass a higher `g_min` and must be prepared for `ValueError` if
            that floor makes some frame infeasible.
        key_active_floor: A band is "key-active" this frame (i.e. gets a
            constraint at all) only if `key_band_energy` there is at or
            above this ABSOLUTE linear-power floor, AND at or above
            `key_active_relative_db` below this frame's OWN loudest
            key band (see `key_active_relative_db`). Below either, the
            band is treated as though the key were silent there THIS
            FRAME -- no constraint, no attenuation demanded on its
            account. Without the absolute floor, a hand-built all-zero
            (or all-equal-floor) test signal would read every band as
            "at the frame's own peak" and so, perversely, fully active;
            without the relative floor, ordinary spectral leakage from a
            real, physically bandlimited "disjoint" signal -- see
            `tests/collision_test_support.py`'s bandpass noise, which
            leaks roughly -55 to -65 dB relative to its own passband, not
            literal digital silence -- would still register as "active"
            against a fixed, very low absolute number, and force every
            masker band reaching it toward zero gain (see module
            docstring's disjoint-spectra acceptance test, which BOTH
            floors together are what make pass on real, rendered audio
            rather than only on hand-built exact-zero arrays).
        key_active_relative_db: The RELATIVE half of the key-active test
            above: a band below this many dB relative to the frame's own
            loudest key band is inactive regardless of its absolute
            level. Mirrors `aud.dsp.gate`'s own noise-floor-relative
            threshold convention (`threshold_above_floor_db`) rather than
            inventing a second one -- a fixed absolute number is right
            for exactly one calibration, per that module's own docstring.
        playback_level_db_spl: Forwarded to `spreading_matrix`/
            `masking_offset` (see `aud.dsp.masking`'s own "Calibration").

    Returns:
        `CollisionResult`.

    Raises:
        ValueError: shape mismatch between the two energy arrays and
            `bands`; `margin_db` non-finite; `g_min` outside `[0, 1]`; or
            `scipy.optimize.linprog` reports an infeasible/unbounded/error
            status for some frame (only reachable when `g_min > 0`; see
            above).
    """
    target_band_energy = np.asarray(target_band_energy, dtype=np.float64)
    key_band_energy = np.asarray(key_band_energy, dtype=np.float64)
    n_bands = bands["n_bands"]
    if target_band_energy.ndim != 2 or target_band_energy.shape[0] != n_bands:
        raise ValueError(
            f"target_band_energy must be shape (n_bands={n_bands}, n_frames); got {target_band_energy.shape}"
        )
    if key_band_energy.shape != target_band_energy.shape:
        raise ValueError(
            f"key_band_energy shape {key_band_energy.shape} must match target_band_energy shape "
            f"{target_band_energy.shape}"
        )
    if not np.isfinite(margin_db):
        raise ValueError(f"margin_db must be finite; got {margin_db}")
    if not (0.0 <= g_min <= 1.0):
        raise ValueError(f"g_min must be in [0, 1]; got {g_min}")

    n_frames = target_band_energy.shape[1]
    w = sii_band_importance(bands)
    m_b = masking_offset(bands, playback_level_db_spl=playback_level_db_spl)
    s = spreading_matrix(target_band_energy, bands, playback_level_db_spl=playback_level_db_spl)

    margin_linear = 10.0 ** (margin_db / 10.0)
    cost = -w  # maximise sum(w*G) == minimise sum(w*(1-G)) up to a constant

    gains = np.empty((n_bands, n_frames), dtype=np.float64)
    active = np.zeros((n_bands, n_frames), dtype=bool)
    statuses: list[str] = []

    for frame in range(n_frames):
        e_m = target_band_energy[:, frame]
        e_s = key_band_energy[:, frame]
        # S[b, k] is PER UNIT masker energy (see `spreading_matrix`'s own
        # docstring) -- the constraint is on S[b, k] * E_m[k] * G[k], so the
        # masker's own energy must be folded in here, column-wise, before
        # this matrix is handed to linprog. Forgetting this step was an
        # actual bug caught this session: every band's own near-unity
        # diagonal entry (S[b, b] ~= 1, being a PER-UNIT-ENERGY term) would
        # otherwise be mistaken for its absolute contribution, throttling
        # every band's gain toward zero regardless of whether the target
        # had any real energy there at all -- see
        # tests/test_dsp_collision_regressions.py::
        # test_constraint_matrix_is_scaled_by_target_energy_not_bare_spreading_matrix.
        s_frame = s[:, :, frame] * e_m[np.newaxis, :]

        peak_e_s = np.max(e_s)
        relative_floor = peak_e_s * 10.0 ** (key_active_relative_db / 10.0)
        active_mask = (e_s >= key_active_floor) & (e_s >= relative_floor)
        active[:, frame] = active_mask

        if not np.any(active_mask):
            gains[:, frame] = 1.0
            statuses.append("no key-active bands; no constraints, G=1")
            continue

        a_ub = s_frame[active_mask, :]
        b_ub = e_s[active_mask] / margin_linear / np.maximum(m_b[active_mask], _EPS)

        result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=[(g_min, 1.0)] * n_bands, method="highs")
        if not result.success:
            raise ValueError(
                f"collision_gains: linprog failed on frame {frame} (status={result.status}): {result.message}"
            )
        gains[:, frame] = result.x
        statuses.append(result.message)

    return CollisionResult(gain=gains, active_bands=active, solver_status=statuses)
