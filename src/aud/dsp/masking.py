"""PEAQ (ITU-R BS.1387) frequency-domain masking spread and an absolute
threshold-of-hearing floor, over the band structure `aud.dsp.bands` builds.

Scope, and what this module deliberately does NOT do
------------------------------------------------------
This is Step 5 of the masking/ducking epic (see `AGENTS.md`): the spreading
function and the resulting masking threshold. It does NOT implement a
collision measure between two signals (Step 6) or a gain law (Step 7) --
both are blocked behind an unresolved patent-envelope gate at the time this
module was written (`smart_tools-bpu` / issue #22) -- and it does NOT decide
how multiple *tracks'* energies combine; it only spreads ONE signal's own
per-band energy into a masking profile for that same signal.

Why PEAQ, and why its Bark scale specifically
------------------------------------------------
`aud.dsp.bands.band_edges` requires an explicit `scale` with no default,
precisely because pairing one model's spreading slopes with a *different*
Bark convention's band edges is a silent, real error -- the two are
calibrated together in the standard that publishes them (see `bands.py`'s
own "THE SCALE IS NOT COSMETIC" section, and the ~3 Bark disagreement
between `bark_peaq` and `bark_zwicker_terhardt` measured there). Every
function in this module that takes a `bands` argument therefore REQUIRES
`bands["scale"] == "bark_peaq"` and raises if it is not -- the slopes below
are calibrated against that scale and no other.

The spreading function (ITU-R BS.1387 Annex 2 Sec 2.1.7 / 2.2.7; ported
here from the discrete algorithm in P. Kabal, "An Examination and
Interpretation of ITU-R BS.1387: Perceptual Evaluation of Audio Quality",
McGill University TSP Lab Technical Report, 2002-04-18 revision, Appendix B
(continuous derivation, eq. 166-170) and Appendix F.4 (`PQspreadCB` /
`PQ_SpreadCB`, the discrete per-band algorithm actually implemented here).
Verified independently against ITU-R BS.1387-1 (11/2001) Annex 2 Sec 2.1.7
(the slope constants) and cross-checked line-by-line against Kabal's own
published MATLAB before being ported to Python/NumPy -- see
`tests/test_dsp_masking_kabal_reference.py`, which carries a *separate*
transliteration of that MATLAB as its test oracle, not a copy of this
module's own code.

    S_l                  = 27 dB/Bark                    (lower skirt, fixed)
    S_u[k] = -24 - 230/f_c[k] + 0.2*L[k]   dB/Bark        (upper skirt)

`f_c[k]` is masker band `k`'s centre frequency in Hz; `L[k]` is that
masker's own level in dB SPL (`10*log10(E[k])` in whatever linear-power
units `E` is calibrated to -- see "Calibration" below). The lower skirt is
fixed; the upper skirt is both frequency- and level-dependent, and is
*shallower* than the lower skirt across the entire audible range (-24 to
roughly -8 dB/Bark at typical listening levels, vs. a fixed -27 dB/Bark
below) -- shallower means SLOWER decay, i.e. masking reaches further above
a masker's own frequency than below it. That asymmetry is not a detail to
approximate away: it is the entire reason a per-band-independent duck gets
this wrong (see this module's own package docstring reference in
`AGENTS.md`).

Discrepancy in the literature, noted and resolved in favour of the
Recommendation: Thiede et al. 2000 (the paper introducing PEAQ) states the
lower slope as "about 27" in one place and "24" in its own model
description; ITU-R BS.1387 itself (the ratified Recommendation, not the
paper describing it) states 27, and that is what this module implements.

Multiple maskers combine in the 0.4-power domain (eq. 166: PEAQ's own
choice, `exponent=0.4` below), not a plain linear/power sum -- this is a
property of the *cited* model, not a free parameter this module invents.

Calibration is a convention, not physics
--------------------------------------------
Every dB-SPL-denominated quantity in this module (the masker's own level
`L` inside the upper slope, and `threshold_in_quiet`'s absolute floor)
requires a mapping from this codebase's normalised linear-power band
energies (as returned by `aud.dsp.bands.band_energy`, where a full-scale
sinusoid has an amplitude of 1.0) to real dB SPL. There is no physical fact
that supplies this mapping -- it depends on how loud the *listener* actually
plays the material back, which this module cannot know. BS.1387 itself
asks the user and defaults to 92 dB SPL for a 0 dBFS sine; every function
below that needs this mapping takes `playback_level_db_spl` as an explicit
parameter (default 92.0, matching BS.1387's own default) rather than
silently assuming one. The convention used here: linear power 1.0
(0 dBFS) <-> `playback_level_db_spl` dB SPL, so
`calibrated_power = normalised_power * 10**(playback_level_db_spl / 10)`.
`spreading_function_peaq` and `masking_threshold` both take and return
NORMALISED power (matching `band_energy`'s own convention) -- calibration
happens internally and is undone again before the result is returned, so a
caller of this module never has to reason about dB SPL directly unless it
wants to.

Normalisation after spreading -- the step this epic's design notes flag as
a common silent-scaling bug, and what it actually guarantees
------------------------------------------------------------------------------
Spreading a signal's energy outward through a triangular dB-domain kernel,
then summing multiple maskers' contributions, changes the total energy
unless it is explicitly corrected for. BS.1387/Kabal's own construction
(eq. 167, `Bs` in Kabal's MATLAB) computes a normalising factor from a
*fixed, level-1 reference* (every band's power set to exactly 1.0, in the
`playback_level_db_spl`-calibrated domain -- see "Calibration" above) and
divides every real spread result by that SAME fixed factor, regardless of
the real signal's actual level. This is not an approximation this module
introduces: it is exactly what Kabal's own `PQspreadCB`/`PQ_SpreadCB`
MATLAB does (`Bs` is computed once, cached, and reused for every frame).

One consequence, MEASURED rather than assumed, and considerably LARGER than
first intuition suggests: because the upper slope is level-dependent
(`+0.2*L`), the spreading kernel's own SHAPE changes with level -- so the
normaliser, fixed at the level-1 reference, cancels the gain exactly ONLY
when the real signal is ALSO at that reference level, and increasingly
imperfectly, MULTIPLICATIVELY over every one of the ~24.5 Bark steps the
audible range spans, as the real level departs from it. A uniform (flat)
input profile therefore maps back to EXACTLY uniform output only at that
one (very quiet, near-silent) reference level; at realistic listening
levels the deviation is real and substantial, not a rounding-level effect
-- 32 PEAQ-scale bands, `playback_level_db_spl=92.0` (the default):

    normalised power   real level (92 dB SPL calibration)   max |ratio-1|
    1.0 (0 dBFS)        92.0 dB SPL                          5.74
    1e-2                72.0 dB SPL                          1.93
    1e-4                52.0 dB SPL                          0.81
    1e-6                32.0 dB SPL                          0.34
    1e-9                 2.0 dB SPL                          0.013
    (exact Bs reference)  0.0 dB SPL                         0.0 (exact)

(Corrected from an earlier draft of this docstring that said "~-92 dB SPL"
for the exact-reference row: the reference is defined as the level at which
`normalised_power * calibration == 1.0` -- i.e. `10*log10(1.0) == 0` dB SPL
exactly, by construction, not approximately, and not -92. Verified directly:
`reference_normalised_power = 1.0 / calibration`, so
`reference_normalised_power * calibration` is `1.0` for any
`playback_level_db_spl`, not just 92.)

This is real, intrinsic behaviour of a level-dependent spreading model
applied across a wide dynamic range from a fixed low-level reference -- NOT
a scaling bug this module introduced. The distinction is verified, not
asserted: `spreading_function_peaq` matches an INDEPENDENTLY-transliterated
port of Kabal's own published MATLAB bit-for-bit (`rtol=1e-9`) across flat
profiles at every level above, plus impulses, ramps and random profiles, in
`tests/test_dsp_masking_kabal_reference.py` -- so this deviation is provably
the published algorithm's own behaviour, not this port's. (A real scaling
BUG was in fact caught this way during development: an early version divided
the level-dependent exponent by an extra, erroneous factor of 10, which
UNDERSTATED this same deviation by roughly 10x and was only caught by that
independent-oracle comparison -- see the inline comment at
`_spread_power_domain`'s level-dependent term.)
`test_dsp_masking.py::test_uniform_excitation_is_exactly_preserved_at_the_bs_reference_level`
pins the one EXACT case (provable from the construction); `test_dsp_masking.py::
test_uniform_excitation_deviation_is_bounded_and_matches_independent_oracle`
pins that the (large, real) deviation at realistic levels stays within a
generous, measured envelope -- catching a genuine blow-up/regression -- while
the Kabal-oracle test above is what actually rules out a silent scaling defect.

Known deviation: outer/middle-ear weighting and internal noise are NOT
applied before the level-dependent upper slope -- measured impact
------------------------------------------------------------------------
ITU-R BS.1387's FFT-based ear model (Annex 2 Sec 2.1) applies TWO stages
BEFORE frequency-domain spreading that this module does not implement:
Sec 2.1.4 "Outer and middle ear" (a fixed, frequency-dependent weighting
`W[k]` applied to each FFT bin before grouping into critical bands, eq. 7)
and Sec 2.1.6 "Adding internal noise" (a fixed per-band additive term,
eq. 18, `0.4*3.64*(f/1000)^-0.8` dB SPL). `spreading_function_peaq` instead
spreads `band_power` exactly as received from `aud.dsp.bands.band_energy`
-- unweighted, with no internal-noise floor added.

This is a DELIBERATE, documented simplification, not an oversight:
`aud.dsp.masking` operates purely on already-grouped per-band energy (see
this module's own scope note below and `aud.dsp.bands`'s own docstring --
"does NOT implement spreading functions" there either). Applying `W[k]`
correctly requires per-FFT-bin weighting BEFORE the bin->band summation
`aud.dsp.bands.band_energy` performs; retrofitting it here, after the fact,
on already-summed band energy, would only be a coarser band-centre-only
approximation of the real per-bin integral -- not obviously better than
documenting the gap, and it would silently mix "the real thing" and "an
approximation of the real thing" under one name. Implementing it properly
is future work at the `aud.dsp.bands`/`aud.dsp.stft` layer, not this
module.

Because the upper slope's level-dependent term is `+0.2*L` and `L` is
computed here directly from the UNWEIGHTED band power, the omission of
`W[k]` shifts `L` by approximately `W[k]` itself (for signal levels well
above the internal-noise floor, where the additive noise term is
negligible) -- making the upper slope's level term wrong by roughly
`0.2*W[k]` dB/Bark. `W[k]` (ITU-R BS.1387 eq. 7 / Kabal eq. 6:
`W_dB(f) = -0.6*3.64*(f/1000)^-0.8 + 6.5*exp(-0.6*(f/1000-3.3)^2) -
0.001*(f/1000)^3.6`) is NEGATIVE almost everywhere (the ear is less
sensitive than the 1 kHz reference at most frequencies), so this module's
slope is correspondingly too SHALLOW (masking over-predicted -- the same
unsafe-for-a-ducker direction as the missing eq. 24 offset, see "Known
deviation" below) almost everywhere, and too STEEP only in the narrow
~2-5 kHz region where `W[k]` is positive. Measured directly against eq. 7
above (re-derived and independently cross-checked against Kabal's own
worked checkpoints -- `W(1 kHz) = -1.9` dB and peak `+5.6` dB near 3.3 kHz,
both reproduced to 3 significant figures by this same formula):

    f            W[k]        0.2*W[k]  (the missing slope correction, dB/Bark)
    50 Hz       -23.98 dB     -4.80
    80 Hz       -16.46 dB     -3.29
    100 Hz      -13.77 dB     -2.75   (matches the ~2.8 dB/Bark figure this
                                       deviation was originally measured at)
    200 Hz       -7.89 dB     -1.58
    500 Hz       -3.74 dB     -0.75
    1000 Hz      -1.91 dB     -0.38
    2000 Hz      +1.09 dB     +0.22
    3300 Hz      +5.59 dB     +1.12   (the one region where this module's
                                       slope is too STEEP, not too shallow)
    5000 Hz      +0.22 dB     +0.04
    8000 Hz      -2.20 dB     -0.44
    10000 Hz     -4.33 dB     -0.87
    15000 Hz    -17.39 dB     -3.48
    18000 Hz    -33.25 dB     -6.65

A caller needing PEAQ-exact slopes at bass or high-treble frequencies
should be aware of this; a caller only needing the asymmetry itself
(upward masking reaches further than downward) and approximate slope
magnitudes in the 1-5 kHz region is well served by the current
implementation.

Two further behaviours, decided and documented (not silently left
implicit)
--------------------------------------------------------------------------
- **The upper slope can turn POSITIVE at extreme levels, and this module
  does NOT clamp it.** `S_u = -24 - 230/f_c + 0.2*L` is positive whenever
  `L > 120 + 1150/f_c` dB SPL -- reachable, at the default calibration,
  only when `playback_level_db_spl` itself is set above roughly 120 dB SPL
  (since normalised `band_power` is capped at 1.0 == `playback_level_db_spl`
  dB SPL). DECISION: leave it unclamped. This is faithful to the FFT-based
  ear model this module ports (Kabal's `PQ_SpreadCB`, Appendix F.4, has no
  clamp); BS.1387's OTHER (filter-bank-based) ear model does clamp its
  spreading kernel, but this module implements the FFT-based one, and
  mixing a clamp from the other model in would be an uncited addition, not
  a port. 120+ dB SPL is itself outside any realistic playback level (well
  past instantaneous hearing-damage thresholds), so the reachability
  condition is stated here rather than guarded against defensively.
- **`apply_absolute_threshold=True` (the default) declares everything above
  a CALIBRATION-DEPENDENT crossover frequency "masked" -- that frequency is
  NOT a fixed "~15 kHz"; it moves with `playback_level_db_spl`, and stating
  one number without the calibration hides that.** Measured directly (32
  PEAQ-scale bands spanning 20 Hz-20 kHz -- this module's own reference band
  count -- comparing `threshold_in_quiet`'s floor against the eq. 24-offset
  masking threshold of the loudest possible masker, 0 dBFS in every OTHER
  band, the most masking-favorable case obtainable): the absolute-threshold
  floor DOMINATES above roughly 16.0 kHz at `playback_level_db_spl=70`,
  rising to roughly 17.8 kHz at `playback_level_db_spl=100` (both converge,
  in the many-band limit, to ~16.06 kHz and ~17.89 kHz respectively -- the
  32-band figures above already sit within ~120 Hz of that limit). For
  context, `threshold_in_quiet(15 kHz)` is only 51.0 dB SPL -- far below any
  realistic calibration, so "~15 kHz" was measurably wrong, not merely
  imprecise: at every playback level tested, the true crossover sits above
  16 kHz, never at 15 kHz. `threshold_in_quiet` grows very steeply above
  that (measured: 65.9 dB SPL at 16 kHz, 160.3 dB SPL at 20 kHz -- see
  `threshold_in_quiet`'s own docstring). DECISION: keep the default `True`.
  For this module's ducking use case the direction is SAFE, unlike the eq. 24 masking-offset
  omission this PR fixes: content genuinely above the threshold of hearing
  at these frequencies is, by definition, inaudible regardless of any
  masker, so reporting it as "masked" (safe to duck) cannot itself cause
  audible artifacts. This is a different failure mode from over-predicting
  masking of clearly audible mid-range content, which IS unsafe -- the
  reason Blocker 1 in this module's own history was a correctness bug and
  this default is not.

What is NOT implemented (documented, not silently dropped)
----------------------------------------------------------
- PEAQ's own further pattern-adaptation, and time-domain (forward-masking)
  smoothing stages are NOT implemented here -- they are part of PEAQ's
  *quality assessment* pipeline, not needed to produce a per-frame masking
  profile for a ducker. The design note for this epic explicitly scopes
  Step 5 to "Implement PEAQ spreading and STOP." (Outer/middle-ear
  weighting and internal noise are ALSO not implemented -- see "Known
  deviation" above, which is the one item in this list with a measured
  numeric impact rather than being purely out of scope.)
- Multi-masker combination uses PEAQ's own 0.4-power law rather than a
  simple linear (energy) sum. This is the CONSERVATIVE choice for a ducker
  (it under-predicts combined masking relative to measured "excess
  masking", per Lutfi 1983 -- see this epic's design notes) and is also
  literally what the cited model specifies; it is not a free choice this
  module is making beyond what PEAQ itself does.
"""

from __future__ import annotations

import numpy as np

from aud.dsp.bands import hz_to_bark_peaq

__all__ = [
    "masking_threshold",
    "spreading_function_peaq",
    "threshold_in_quiet",
]

#: PEAQ's own default calibration (ITU-R BS.1387 Annex 2 Sec 2.1.3): in the
#: absence of other information, a 0 dBFS sine is assumed to correspond to
#: this many dB SPL. See module docstring's "Calibration" section.
_DEFAULT_PLAYBACK_LEVEL_DB_SPL = 92.0

#: PEAQ's own combination exponent (ITU-R BS.1387 Annex 2 Sec 2.1.7 eq. 166 /
#: Kabal Appendix B eq. 166, `e` in Kabal's MATLAB): multiple maskers'
#: spread contributions add in this power domain, not linearly.
_DEFAULT_COMBINE_EXPONENT = 0.4

#: Fixed lower-skirt slope, dB/Bark (ITU-R BS.1387 Annex 2 Sec 2.1.7 eq. 170;
#: Kabal Appendix B eq. 170, and Appendix F.4's `aL = 10^(-2.7*dz)`, where
#: 2.7 == 27/10 -- the /10 converts a dB slope to a power-domain ratio per
#: unit Bark, see `_spread_power_domain` below). Frequency- and
#: level-INDEPENDENT: this is what "fixed" means in the standard.
_LOWER_SLOPE_DB_PER_BARK = 27.0

#: Upper-skirt slope's frequency-independent term and per-Hz coefficient,
#: dB/Bark (ITU-R BS.1387 Annex 2 Sec 2.1.7 eq. 170: `S_u = -24 - 230/f_c +
#: 0.2*L`). Kabal Appendix F.4 codes the same constants divided by 10 as
#: `-2.4` and `-23` (`aUC = 10^((-2.4 - 23/fc)*dz)`) for the same
#: dB->power-ratio reason as the lower slope above.
_UPPER_SLOPE_CONST_DB_PER_BARK = -24.0
_UPPER_SLOPE_FC_COEFF_HZ_DB_PER_BARK = -230.0

#: Upper-skirt slope's level-dependence coefficient, dB/Bark per dB SPL
#: (ITU-R BS.1387 eq. 170's `+ 0.2*L` term).
_UPPER_SLOPE_LEVEL_COEFF = 0.2

#: Terhardt (1979) absolute-threshold-of-hearing formula's three
#: coefficients (ITU-R BS.1387 does not define this; the wider psychoacoustic
#: literature does). Cited here as given in J. D. Painter & A. Spanias,
#: "Perceptual Coding of Digital Audio", Proceedings of the IEEE, 88(4),
#: April 2000, Sec. II.A eq. (1), which in turn cites E. Terhardt,
#: "Calculating Virtual Pitch", Hearing Research 1, 155-182, 1979
#: (Painter & Spanias's own reference [46]). Verified independently in this
#: session against Painter & Spanias's own worked checkpoints (their Fig. 2
#: reads ~83 dB @ 20 Hz through ~-5 dB @ 3.3 kHz) and, separately, against
#: real measured threshold-of-hearing data (ISO 226:2003 Table 1's `T_f`
#: column, an EMPIRICAL free-field threshold, not derived from this formula
#: at all) -- see `tests/test_dsp_masking.py` and
#: `tests/fixtures/standards/iso226_2003_table1_threshold_hz_db.csv`.
_TQ_LOW_FREQ_COEFF = 3.64
_TQ_RESONANCE_COEFF = 6.5
_TQ_RESONANCE_CENTRE_KHZ = 3.3
_TQ_RESONANCE_WIDTH = 0.6
_TQ_HIGH_FREQ_COEFF = 1.0e-3

#: ITU-R BS.1387 Annex 2 Sec 2.1.9 "Masking threshold", eq. 24: the masking
#: threshold sits this many dB BELOW the (spread) excitation pattern, as a
#: piecewise-linear function of Bark distance from the first band
#: (`k * res`, 0-indexed band number `k` times the uniform Bark spacing
#: `res` == this module's `dz`):
#:     m[k]_dB = 3.0               for k*res <= 12
#:             = 0.25 * (k*res)    for k*res >  12
#: Independently confirmed (this session) against P. Kabal's report eq. 112
#: (same piecewise formula, same 3.0/0.25/12 constants) and against the
#: standard's own PDF text (Rec. ITU-R BS.1387, Annex 2 Sec 2.1.9, eq. 24 --
#: literally the section and equation numbers this comment cites, not a
#: paraphrase). Kabal's eq. 112 states the breakpoint occurs at `z_L + 12`
#: Bark, where `z_L = B(f_L)` is the Bark value of the lowest band EDGE
#: (Kabal Appendix F "zL = B(fL)"); since bands are uniformly Bark-spaced
#: starting at that edge, band k's own Bark offset from z_L is exactly
#: `k*res` -- matching the eq. 24 condition on `k*res` directly, with no
#: separate `z_L` term needed here.
#: THIS WAS THE MISSING PIECE (a real correctness bug, not a style
#: preference): without it, `masking_threshold` returned the raw spread
#: excitation UNCHANGED (threshold-minus-excitation measured at exactly
#: 0.00 dB at every band), i.e. 3.0-6.75 dB too permissive -- the UNSAFE
#: direction for a ducker, since it means concluding material is masked
#: when BS.1387's own model says it is not.
_MASKING_OFFSET_FLOOR_DB = 3.0
_MASKING_OFFSET_BREAKPOINT_BARK = 12.0
_MASKING_OFFSET_SLOPE_DB_PER_BARK = 0.25

#: Above this frequency, `_TQ_HIGH_FREQ_COEFF * (f/1000)**4` diverges to
#: physically-meaningless values (~66 dB by 16 kHz, >1000 dB by 40 kHz --
#: MEASURED: the term alone is 65.536 dB at 16 kHz and 2560.0 dB at 40 kHz;
#: full Tq(16 kHz) is 65.9 dB, matching the checkpoint in
#: `tests/test_dsp_masking.py`. An earlier draft of this comment said
#: ">100 dB by 16 kHz", which is wrong by roughly 35 dB -- corrected here;
#: see module docstring). `threshold_in_quiet` clips its input frequency at
#: this ceiling before evaluating the formula; it does not extrapolate past
#: it. Matches `aud.dsp.bands.band_edges`'s own default `f_max`, the
#: conventional upper edge of the audible range.
_TQ_FREQUENCY_CEILING_HZ = 20000.0


def threshold_in_quiet(f_hz: np.ndarray, f_ceiling_hz: float = _TQ_FREQUENCY_CEILING_HZ) -> np.ndarray:
    """Absolute threshold of hearing in quiet, dB SPL (Terhardt 1979 via
    Painter & Spanias 2000 eq. (1) -- see module docstring for the exact
    citation chain):

        Tq(f) = 3.64*(f/1000)^-0.8 - 6.5*exp(-0.6*(f/1000 - 3.3)^2)
                + 1e-3*(f/1000)^4

    Args:
        f_hz: Frequency in Hz. Non-positive values are invalid (the
            formula's first term diverges at f=0 and is undefined for
            f<0); zero or negative input raises.
        f_ceiling_hz: Input frequency is clipped to this ceiling before the
            formula is evaluated -- the `(f/1000)^4` term diverges to
            physically meaningless values above roughly 16 kHz (e.g.
            ~66 dB at 16 kHz vs. ~160 dB at 20 kHz, a factor-of-2.4 change
            for a 25% change in frequency), so this is NOT extrapolation
            the way `band_edges(..., allow_extrapolation=True)` is --
            evaluating the formula past its ceiling is simply wrong, not
            merely unvalidated. Default 20 kHz, the conventional upper edge
            of the audible range and `band_edges`'s own default `f_max`.

    Returns:
        Same shape as `f_hz`: threshold SPL in dB. Reference checkpoints
        (this module's own docstring, verified independently against
        Painter & Spanias's Fig. 2 and ISO 226:2003's measured data --
        see `tests/test_dsp_masking.py`): ~83 dB @ 20 Hz, ~3.4 dB @ 1 kHz,
        ~-5.0 dB @ 3.3 kHz (the formula's minimum), ~10.6 dB @ 10 kHz.

    Raises:
        ValueError: `f_hz` is non-finite, or <= 0.
    """
    f_hz = np.asarray(f_hz, dtype=np.float64)
    if not np.all(np.isfinite(f_hz)):
        raise ValueError(f"f_hz must be finite; got f_hz={f_hz}")
    if np.any(f_hz <= 0):
        raise ValueError(f"f_hz must be > 0 (the formula diverges at f=0); got f_hz={f_hz}")

    f_khz = np.minimum(f_hz, f_ceiling_hz) / 1000.0
    low = _TQ_LOW_FREQ_COEFF * f_khz**-0.8
    resonance = _TQ_RESONANCE_COEFF * np.exp(-_TQ_RESONANCE_WIDTH * (f_khz - _TQ_RESONANCE_CENTRE_KHZ) ** 2)
    high = _TQ_HIGH_FREQ_COEFF * f_khz**4
    return low - resonance + high


def _validate_peaq_bands(bands: dict) -> tuple[np.ndarray, float]:
    """Shared precondition for every function below that takes a `bands`
    dict: it must be a PEAQ-scale band structure (see module docstring's
    "why PEAQ, and why its Bark scale specifically"). Returns `(z_centers,
    dz)` -- the Bark-space band centres and their (validated-uniform)
    spacing, both needed by `spreading_function_peaq`.
    """
    if bands.get("scale") != "bark_peaq":
        raise ValueError(
            f"masking functions require bands['scale'] == 'bark_peaq' (got {bands.get('scale')!r}); "
            "the spreading slopes in this module are calibrated to that scale specifically -- pairing "
            "them with a different Bark convention silently misapplies dB-per-Bark constants across "
            "scales that differ by up to ~3 Bark at the top of the audible range. Build `bands` with "
            "aud.dsp.bands.band_edges(..., scale='bark_peaq')."
        )
    n_bands = bands["n_bands"]
    if n_bands < 2:
        raise ValueError(
            f"masking functions require n_bands >= 2 (a spreading function needs neighbours); got {n_bands}"
        )

    centers_hz = np.asarray(bands["centers_hz"], dtype=np.float64)
    z_centers = hz_to_bark_peaq(centers_hz)
    dz_steps = np.diff(z_centers)
    dz = float(dz_steps[0])
    if not np.allclose(dz_steps, dz, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "masking functions require uniformly Bark-spaced band centres (as band_edges always "
            f"produces); got non-uniform spacing {dz_steps}. This bands dict was not built by band_edges."
        )
    return centers_hz, dz


def _spread_power_domain(power: np.ndarray, centers_hz: np.ndarray, dz: float, exponent: float) -> np.ndarray:
    """Core PEAQ spreading recurrence, in the `exponent`-power domain,
    BEFORE the final `**(1/exponent)` root and BEFORE the `Bs` unit-area
    normalisation -- ported from P. Kabal's `PQ_SpreadCB` (Appendix F.4;
    see module docstring). Returns an array the same shape as `power`
    (leading axis = band, any trailing axes carried through elementwise).

    Not applying the normalising division here (callers apply it once,
    since it depends only on `centers_hz`/`dz` and not on `power`) is what
    lets `spreading_function_peaq` compute the normaliser exactly once per
    `bands` rather than recomputing it -- and, separately, is what makes
    "spread of a uniform excitation is uniform" testable as calling this
    function twice (once for the real signal, once for `np.ones_like`) and
    comparing, rather than baking the normaliser silently into a single
    non-invertible step.
    """
    n_bands = centers_hz.shape[0]
    power = np.asarray(power, dtype=np.float64)

    a_lower = 10.0 ** (-_LOWER_SLOPE_DB_PER_BARK / 10.0 * dz)
    a_lower_e = a_lower**exponent

    a_upper_const = 10.0 ** (
        (_UPPER_SLOPE_CONST_DB_PER_BARK + _UPPER_SLOPE_FC_COEFF_HZ_DB_PER_BARK / centers_hz) / 10.0 * dz
    )  # (n_bands,), frequency-dependent, level-independent part of the upper slope

    ene = np.empty_like(power)
    a_upper_level_e = np.empty_like(power)
    for m in range(n_bands):
        # Level-dependent factor: the slope's "+0.2*L" term is in dB/Bark,
        # with L = 10*log10(power) (dB SPL) -- NOT power itself. Converting
        # dB/Bark to a power ratio per dz normally divides by 10 (as it does
        # for the constant/frequency terms above), but substituting L's own
        # factor of 10 back in cancels that division exactly:
        #     10**(0.2*L*dz/10) = 10**(0.2*dz*10*log10(power)/10)
        #                       = 10**(0.2*dz*log10(power)) = power**(0.2*dz)
        # So the exponent applied to `power` is `_UPPER_SLOPE_LEVEL_COEFF *
        # dz`, with NO additional "/10" -- that additional division was
        # measured, this session, to silently mistune the level dependence
        # by a factor of 10 (max relative error 0.85 against the independent
        # Kabal-transliteration oracle in
        # tests/test_dsp_masking_kabal_reference.py before this comment was
        # added and the bug fixed). Do not "simplify" this by reusing
        # `a_upper_const`'s `/10.0` pattern.
        a_upper_m = a_upper_const[m] * power[m] ** (_UPPER_SLOPE_LEVEL_COEFF * dz)
        g_lower = (1.0 - a_lower ** (m + 1)) / (1.0 - a_lower)
        g_upper = (1.0 - a_upper_m ** (n_bands - m)) / (1.0 - a_upper_m)
        normalised_masker = power[m] / (g_lower + g_upper - 1.0)
        ene[m] = normalised_masker**exponent
        a_upper_level_e[m] = a_upper_m**exponent

    spread = np.empty_like(power)
    spread[n_bands - 1] = ene[n_bands - 1]
    for m in range(n_bands - 2, -1, -1):
        spread[m] = a_lower_e * spread[m + 1] + ene[m]

    for m in range(n_bands - 1):
        running = ene[m]
        a_m = a_upper_level_e[m]
        for i in range(m + 1, n_bands):
            running = running * a_m
            spread[i] = spread[i] + running

    return spread


def _masking_offset_db(n_bands: int, dz: float) -> np.ndarray:
    """ITU-R BS.1387 Sec 2.1.9 eq. 24 -- see `_MASKING_OFFSET_FLOOR_DB`'s own
    comment for the full citation. Returns `m[k]` in dB, shape `(n_bands,)`,
    for 0-indexed band number `k = 0 .. n_bands-1`.
    """
    bark_offset_from_first_band = np.arange(n_bands, dtype=np.float64) * dz
    return np.where(
        bark_offset_from_first_band <= _MASKING_OFFSET_BREAKPOINT_BARK,
        _MASKING_OFFSET_FLOOR_DB,
        _MASKING_OFFSET_SLOPE_DB_PER_BARK * bark_offset_from_first_band,
    )


def spreading_function_peaq(
    band_power: np.ndarray,
    bands: dict,
    playback_level_db_spl: float = _DEFAULT_PLAYBACK_LEVEL_DB_SPL,
    exponent: float = _DEFAULT_COMBINE_EXPONENT,
) -> np.ndarray:
    """Spread `band_power` across bands via the PEAQ (ITU-R BS.1387)
    frequency-domain masking model -- see module docstring for the full
    citation and the asymmetric-slope model this implements.

    Known deviation: this spreads UNWEIGHTED band power (no outer/middle-ear
    `W[k]` weighting applied first), which over-predicts masking almost
    everywhere -- see module docstring's "Known deviation" section for the
    measured per-frequency table.

    Args:
        band_power: `(n_bands, ...)`, linear power (as `aud.dsp.bands.
            band_energy` returns; 0 dBFS full-scale sine == 1.0), matching
            `bands["n_bands"]` on axis 0. Must be non-negative and finite.
        bands: As returned by `aud.dsp.bands.band_edges(..., scale=
            "bark_peaq")`. Any other scale raises -- see module docstring.
        playback_level_db_spl: See module docstring's "Calibration"
            section. Only affects the upper slope's level-dependent term
            (`0.2*L`); the lower slope is level-independent by definition.
        exponent: PEAQ's multi-masker combination exponent (default 0.4,
            eq. 166). Exposed for experimentation, not because the cited
            model treats it as free.

    Returns:
        Same shape as `band_power`: the spread masking-excitation profile,
        in the SAME normalised linear-power units as the input (calibration
        is applied and undone internally -- see module docstring). Band `b`
        of the result is influenced by every OTHER band's energy, weighted
        by the asymmetric triangular-in-dB spreading kernel, then
        normalised so that a uniform input profile at the Bs reference
        level maps EXACTLY to itself, and approximately to itself at other
        levels (see module docstring's "Normalisation after spreading" and
        `test_dsp_masking.py::test_uniform_excitation_is_exactly_preserved_at_the_bs_reference_level`).

    Raises:
        ValueError: `bands` is not PEAQ-scale or has fewer than 2 bands
            (see `_validate_peaq_bands`); `band_power`'s leading axis does
            not match `bands["n_bands"]`; `band_power` is negative,
            non-finite, or contains an exact zero (an exact zero is
            rejected as a semantically degenerate masker level, NOT because
            of a numerical domain error -- see the raised message, and the
            correction of this exact claim in an earlier draft).
    """
    centers_hz, dz = _validate_peaq_bands(bands)
    band_power = np.asarray(band_power, dtype=np.float64)
    n_bands = bands["n_bands"]
    if band_power.shape[0] != n_bands:
        raise ValueError(f"band_power's leading axis has length {band_power.shape[0]}, but bands has n_bands={n_bands}")
    if not np.all(np.isfinite(band_power)):
        raise ValueError("band_power must be finite")
    if np.any(band_power < 0):
        raise ValueError("band_power must be non-negative (it is a power/energy quantity)")
    if np.any(band_power == 0):
        raise ValueError(
            "band_power must be strictly positive: at exactly zero, this masker's implicit level "
            "L = 10*log10(power) is -infinity, which is not a physically meaningful masker level for this "
            "model. (Note this is a semantic guard, not a numerical-domain error: the exponent actually "
            "applied to power in the recurrence, _UPPER_SLOPE_LEVEL_COEFF * dz, is POSITIVE -- dz is always "
            "a positive Bark step -- so power**that_exponent evaluates to a well-defined 0.0 at power=0, not "
            "NaN or a ZeroDivisionError. An earlier version of this message wrongly claimed the exponent was "
            "negative and undefined at zero; it is not.) Add a small floor (e.g. the quietest representable "
            "digital sample's power) before calling."
        )

    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    calibrated_power = band_power * calibration

    spread_raw = _spread_power_domain(calibrated_power, centers_hz, dz, exponent)
    ones_like_bands = np.ones(n_bands, dtype=np.float64)
    normaliser_raw = _spread_power_domain(ones_like_bands, centers_hz, dz, exponent)
    # normaliser_raw has shape (n_bands,); broadcast against spread_raw's
    # possible trailing (frame, ...) axes.
    normaliser_shape = (n_bands,) + (1,) * (spread_raw.ndim - 1)
    normaliser = normaliser_raw.reshape(normaliser_shape) ** (1.0 / exponent)

    spread_calibrated = spread_raw ** (1.0 / exponent) / normaliser
    return spread_calibrated / calibration


def masking_threshold(
    band_power: np.ndarray,
    bands: dict,
    playback_level_db_spl: float = _DEFAULT_PLAYBACK_LEVEL_DB_SPL,
    apply_absolute_threshold: bool = True,
    exponent: float = _DEFAULT_COMBINE_EXPONENT,
) -> np.ndarray:
    """`spreading_function_peaq`'s excitation pattern, weighted down by the
    ITU-R BS.1387 Sec 2.1.9 eq. 24-26 masking offset `m[k]` (see
    `_MASKING_OFFSET_FLOOR_DB`'s own comment), then optionally floored (per
    band) by `threshold_in_quiet` at that band's centre frequency -- the
    combination Painter & Spanias describe as the conventional
    `MAX(spread_pattern, Tq)` (Sec. II.C; see module docstring's citation
    for `threshold_in_quiet`), applied to the OFFSET-WEIGHTED pattern, not
    the raw excitation.

    Known deviation: inherited from `spreading_function_peaq` -- the spread
    excitation this thresholds is UNWEIGHTED band power (no outer/middle-ear
    `W[k]` applied first), which over-predicts masking almost everywhere --
    see module docstring's "Known deviation" section for the measured table.

    Args:
        band_power, bands, playback_level_db_spl, exponent: See
            `spreading_function_peaq`.
        apply_absolute_threshold: If `False`, this is
            `spreading_function_peaq`'s output with the eq. 24-26 masking
            offset applied but with NO absolute-threshold floor -- the
            absolute-threshold-of-hearing term is optional per this epic's
            design notes ("optional absolute threshold of hearing floor").
            The eq. 24-26 offset itself is NOT optional: it is what makes
            this function's return value a masking THRESHOLD rather than a
            bare excitation pattern, in either case.

    Returns:
        Same shape as `band_power`, same normalised linear-power units.
        `10*log10(spreading_function_peaq(...)) -
        10*log10(masking_threshold(..., apply_absolute_threshold=False))`
        equals `_masking_offset_db(...)` exactly, at every band -- see
        `tests/test_dsp_masking.py::
        test_masking_threshold_sits_the_bs1387_offset_below_the_excitation`.

    Raises:
        As `spreading_function_peaq`.
    """
    spread = spreading_function_peaq(band_power, bands, playback_level_db_spl=playback_level_db_spl, exponent=exponent)

    n_bands = bands["n_bands"]
    _, dz = _validate_peaq_bands(bands)
    m_db = _masking_offset_db(n_bands, dz)
    m_shape = (n_bands,) + (1,) * (spread.ndim - 1)
    masked = spread * 10.0 ** (-m_db.reshape(m_shape) / 10.0)

    if not apply_absolute_threshold:
        return masked

    centers_hz = np.asarray(bands["centers_hz"], dtype=np.float64)
    tq_db_spl = threshold_in_quiet(centers_hz)
    calibration = 10.0 ** (playback_level_db_spl / 10.0)
    tq_normalised_power = 10.0 ** (tq_db_spl / 10.0) / calibration
    tq_shape = (n_bands,) + (1,) * (masked.ndim - 1)
    return np.maximum(masked, tq_normalised_power.reshape(tq_shape))
