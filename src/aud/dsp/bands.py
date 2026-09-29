"""Perceptual band mapping: Hz<->Bark/ERB conversions, band-edge construction,
and bin->band energy summation over an STFT magnitude/power array.

This module is deliberately narrow, matching `aud.dsp.stft`'s own boundary:
it maps frequencies onto a perceptual scale and sums spectral energy into
bands. It does NOT implement spreading functions, masking thresholds, gain
laws or ducking -- those are a later step in the same epic (multi-track /
sidechain work lives in a separate tool; see AGENTS.md).

Why this scale, not linear Hz
------------------------------
A fixed-Hz band width is wrong across the audible range in a way that is
easy to reproduce by accident. `noisereduce` (MIT) defaults to
`freq_mask_smooth_hz=500`: that is ~14 ERB wide at 100 Hz but only ~0.45 ERB
wide at 10 kHz -- far too coarse in the bass, far too fine in the treble.
(Its bin conversion also divides by `n_fft/2` instead of `n_fft`, which is
twice the true bin spacing -- also not reproduced here; see
`aud.dsp.stft.stft_properties`'s `bin_hz = sr / n_fft` for the convention
this module matches.) Bark and ERB widen with frequency roughly the way
critical-band masking and cochlear filtering actually do, which is the
entire reason a "perceptual band" module exists.

THE SCALE IS NOT COSMETIC -- read this before calling `band_edges`
--------------------------------------------------------------------
`band_edges` takes `scale` as a REQUIRED parameter (no default) and stamps
it into the returned dict's `"scale"` key. This is not decoration: the next
step in this epic hangs its spreading-function slopes (27 dB/Bark below a
masker, -24 - 230/f_c + 0.2*L dB/Bark above it) directly off the **Bark**
scale, and 1 Bark is emphatically not a fixed multiple of 1 ERB --
the ratio itself changes with frequency:

    dE/dz (ERB per Bark, measured here via central-difference on the
    Zwicker&Terhardt forward map against Glasberg-Moore ERB-rate):
    ~2.7-2.9 at 100 Hz, ~1.2-1.3 at 1 kHz, ~1.9-2.0 at 10 kHz.

A caller who builds ERB bands and then reuses a per-Bark dB/Bark slope
unchanged gets a silently wrong (too narrow or too wide) skirt -- no
exception, no failing test, just wrong ducking three-to-one at 100 Hz vs.
1 kHz. Making the scale a required, visible field on the band structure is
the guard against that: a downstream reader can always ask "which scale did
I get" instead of assuming.

Two Bark realizations, on purpose
------------------------------------
"Bark" names a family, not one formula, and the family members disagree by
multiple Bark at the extremes (measured below). Both are implemented so a
caller can pick deliberately rather than inherit whichever one a library
happened to expose:

- `bark_peaq` -- PEAQ / ITU-R BS.1387 convention, `z = 7*asinh(f/650)`.
  Closed form BOTH ways (`f = 650*sinh(z/7)`), so it round-trips to float64
  precision with no search. Diverges from the classical 24-band table by up
  to ~3 Bark at the top of the audible range (measured below) -- it is a
  smooth perceptual-model approximation, not a classical-Bark substitute.
- `bark_zwicker_terhardt` -- Zwicker & Terhardt 1980 (JASA 68(5):1523),
  `z(f) = 13*atan(0.00076f) + 3.5*atan((f/7500)^2)`. This formula
  APPROXIMATES the earlier-published classical 24-critical-band table (not
  the reverse), so it is (expectedly) the closest match to that table
  (~0.20 Bark max error, measured below). It has no closed-form inverse;
  `bark_zwicker_terhardt_to_hz` bisects.

**`band_edges` therefore defaults its callers to nothing**: `scale` has no
default value at all, so a caller must say which of the two (or
`erb_glasberg_moore`) it means. Where this module itself needs one non-band
default (the low-level scalar `hz_to_bark`/`bark_to_hz` convenience pair),
it defaults to `bark_peaq`, per the task note that a closed form both ways
is the one to reach for casually -- but that convenience default is
DELIBERATELY NOT reused as `band_edges`'s default, because the accuracy
tradeoff above is real and `band_edges` is the entry point whose choice
actually propagates into a spreading-function slope downstream.

Zwicker 24-band table agreement (measured, see tests/test_dsp_bands.py)
--------------------------------------------------------------------------
Checked against the classical band-edge table (0, 100, 200, ..., 15500 Hz,
by definition z=0..24 Bark at those edges):

    bark_zwicker_terhardt: max |z_computed - z_table| ~= 0.20 Bark
    bark_peaq:             max |z_computed - z_table| ~= 3.06 Bark (at 15500 Hz)

>15.5 kHz is EXTRAPOLATION, not silent
-----------------------------------------
The classical Bark table is tabulated only to 24 Bark = 15.5 kHz
(`_BARK_TABULATED_LIMIT_HZ`). `band_edges` raises `ValueError` if a Bark
`scale` is asked to cover `f_max` above that limit, unless the caller passes
`allow_extrapolation=True` -- in which case the closed-form/bisected formula
is evaluated anyway (all of them are defined, just unvalidated, above
15.5 kHz) and the returned dict's `"extrapolated"` field is set `True` so a
caller cannot lose track of it. `erb_glasberg_moore` carries no such gate:
the task's citation does not name an equivalent tabulated limit for ERB.

Bin -> band weights: triangular partition-of-unity, not linear-Hz smoothing
------------------------------------------------------------------------------
`bin_band_weights` builds one triangular filter per band, in the style
RNNoise (22 Bark-ish bands, BSD-3) and DeepFilterNet (32 ERB bands, MIT)
both use -- approach only, no code copied from either. Filter `i` peaks at
band `i`'s center (the scale-space midpoint of `band_edges`' hard edges,
converted back to Hz) and falls linearly to zero at the neighbouring bands'
centers. The two end bands are "flattened" on their outward side (constant
weight 1 from `f_min` out to the first center, and from the last center out
to `f_max`) rather than tapering to zero there -- with plain (untapered)
triangles, the classical construction only sums to 1 in the *interior*
between the first and last centers, silently discarding some energy at the
very top and bottom of the covered range. The flattened-end construction
proved here still sums to exactly 1 in every interior segment (two adjacent
triangles' rising and falling ramps are complementary linear functions of
frequency, so they sum to 1 by construction -- see the module's internal
proof in `bin_band_weights`), AND is exactly 1 in the two end segments too
(a flat weight, not a ramp), so `sum_b w_b(k) == 1` holds for literally
every bin, in-range or not -- including bins below `f_min` or above `f_max`
(e.g. a `sr` whose Nyquist falls below the top band edge; see the "Nyquist
below top edge" test). No masking or clamping code is needed for those
out-of-range bins: the flat ends already extend to +/-infinity in the
formula, so out-of-range bins land on the nearest edge band's constant-1
region for free.

32 bands is the parameterised default reference point (never hardcoded
here, but see tests): Trackspacer, iZotope Neutron Unmask and DeepFilterNet
(ERB) all independently converged on 32; RNNoise uses 22. `band_edges`
leaves `n_bands` entirely to the caller.

Energy conservation follows for free from partition-of-unity: summed over
ALL bands, `sum_b band_energy_b = sum_b sum_k w_b(k)*E(k) = sum_k E(k) *
sum_b w_b(k) = sum_k E(k) * 1 = total energy` -- exactly, not
approximately, whenever the weights sum to 1 per bin (verified directly by
`test_partition_of_unity_sums_to_one_everywhere`, and the energy-sum
equality is checked independently in
`test_band_energy_conserves_total_energy`).

Which functions validate their numeric input, and why some don't
--------------------------------------------------------------------
`bark_zwicker_terhardt_to_hz` (an ITERATIVE SEARCH) and `band_edges` /
`bin_band_weights` (which CONSTRUCT A RANGE FROM ENDPOINTS AND A COUNT)
validate their arguments and raise on non-finite or degenerate input,
because those code paths can silently return a bracket edge, an
extrapolated point, or a mis-shapen array dressed up as a real answer with
no signal that anything was wrong -- issues #25, #26 and #29 are three
independent, previously-shipped instances of exactly that in this file.

A fourth instance of the same class was found -- by actually measuring
every remaining "propagates honestly" claim below against negative input,
rather than inferring it from the shape of the code -- in `critical_bandwidth_hz`,
`erb_bandwidth_hz`, and `band_energy`. All three are now guarded too (see
each function's own docstring for the measured counter-example that forced
this). **The rule that decides each case**: a non-guard is safe only if the
operation keeps a bad value VISIBLE in the output (NaN, +-inf, or an
obviously-invalid result such as a negative Hz/Bark value); "closed-form,
no search loop" is NOT the test, because an even function or a weighted sum
can absorb a bad value into a small, plausible, WRONG positive number with
no signal anything was wrong. Measured counter-examples that forced the
three functions above out of the "deliberate accept" bucket:

- `critical_bandwidth_hz(-1000.0)` returns `162.2167...`, bit-identical to
  `critical_bandwidth_hz(1000.0)` -- the formula squares `f`, so it is an
  EVEN function of frequency and hides the sign of every negative input,
  not just large ones.
- `erb_bandwidth_hz(-5.0)` returns `24.1603...`, a plausible ERB bandwidth
  value (close to `erb_bandwidth_hz(0.0) == 24.7`) -- the formula is linear
  in `f` with a small coefficient (`4.37/1000`), so small-magnitude negative
  input stays positive and looks legitimate; it only turns visibly negative
  once `f < -1000/4.37 ~= -228.83` Hz. Silently plausible for the entire
  `(-228.83, 0)` Hz range is not an edge case, it is most of the range a
  caller's typo or off-by-one would actually produce.
- `band_energy` with one negative `spectrum` bin, or one negative `weights`
  entry, does not propagate a visible negative sum -- a weighted sum over
  several other positive bins/weights routinely absorbs one negative
  contributor into a smaller but still-positive, still-plausible band
  energy (measured: one bin set to `-5.0` changed a clean band value of
  `19.45` to `13.58`, with every band in the result still `>= 0`). A sum is
  exactly the shape that can hide a bad value the way an even function does.

The remaining five one-line closed-form scalar maps (`hz_to_bark_peaq`,
`bark_peaq_to_hz`, `hz_to_bark_zwicker_terhardt`, `hz_to_erb_rate`,
`erb_rate_to_hz`) do NOT validate, deliberately, and this is now backed by a
measured sweep of each one across 15 negative magnitudes from -0.001 Hz/Bark
to -1e9 Hz/Bark (not just one large-magnitude probe): every one of them
returns either a negative value (Bark/ERB-rate/Hz can never legitimately be
negative, so this is always visible) or NaN, at every magnitude tested, with
no silent sign-cancelling range **at any of the swept magnitudes**. Each is a
single published formula evaluated once with no search and no accumulated
state: NaN in produces NaN out, +-inf in produces whatever that one
formula's own limit is (finite or infinite), and because each of these five
is odd (or reduces to an odd function's sign for any nonzero negative input)
rather than even or a sum, a negative input cannot be absorbed back into a
plausible positive answer. None of these five can produce a PLAUSIBLE WRONG
ANSWER the way a search, a range construction, or an even/summing formula
can: the output is always the literal, honestly-computed image of the input
under that one formula, and a caller can already tell the answer is unusable
by inspecting it for non-finiteness or an impossible sign. This is the same
precedent `bark_peaq_to_hz` established in PR #26 ("the honest sibling"):
propagate, don't add a guard whose only job would be to reject a value that
is already self-evidently unusable -- but that precedent is now applied
per-function, backed by a measured sweep, rather than assumed for the whole
group.

**Two of the five have one float64-resolution exception the magnitude sweep
above does not probe** (it starts at -0.001, far above either bound): very
close to zero, `hz_to_erb_rate` and `erb_rate_to_hz` each have a tiny window
where `1 + (small coefficient)*input` rounds to exactly `1.0` at float64
precision, so the sign IS lost -- the function returns exactly `0.0`,
indistinguishable from input `0.0`. Measured directly (see each function's
own docstring): `hz_to_erb_rate` for `-1.27e-14 < f < 0`;
`erb_rate_to_hz` for `-5.16e-16 < erb_rate < 0`. Outside those windows
(i.e. at every magnitude the sweep above actually covers) the sign is
visible as documented. Vanishingly small in absolute terms and not a
practical concern for any real caller, but the literal boundary rather than
an assumption.

`band_energy`'s new guard sits in a per-frame hot path (see AGENTS.md #8:
`dsp/` modules take arrays and return arrays without raising user-facing
errors); this cost is measured directly (see the function's own docstring)
and judged worth paying because the alternative is exactly the silent
plausible-wrong-answer class this module exists to remove.

The five non-guarded closed-form maps above accept a Python/NumPy `bool` as
a number without complaint (e.g. `bark_peaq_to_hz(True)` returns `93.17`
Hz, identical to passing `1.0`) -- `np.asarray`/arithmetic treat `bool` as
`0`/`1`, and this module does not special-case it out. Recorded here as a
fact about current behaviour, not changed: none of these five functions
guard their input (see above), and a `bool` is, numerically, a perfectly
well-defined `0` or `1`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BARK_SCALES",
    "ERB_SCALES",
    "SCALES",
    "band_edges",
    "band_energy",
    "bark_peaq_to_hz",
    "bark_to_hz",
    "bark_zwicker_terhardt_to_hz",
    "bin_band_weights",
    "critical_bandwidth_hz",
    "erb_bandwidth_hz",
    "erb_rate_to_hz",
    "hz_to_bark",
    "hz_to_bark_peaq",
    "hz_to_bark_zwicker_terhardt",
    "hz_to_erb_rate",
]

# Top edge of the classical Zwicker 24-critical-band table (24 Bark). Above
# this, any Bark formula is extrapolation -- see module docstring.
_BARK_TABULATED_LIMIT_HZ = 15500.0


# --- Bark, PEAQ / ITU-R BS.1387 convention -- closed form both ways ---


def hz_to_bark_peaq(f: np.ndarray) -> np.ndarray:
    """Hz -> Bark, PEAQ/ITU-R BS.1387 convention: `z = 7*asinh(f/650)`.

    Closed form both ways (see `bark_peaq_to_hz`); the most convenient of
    the three Bark realizations here, but NOT the closest match to the
    classical 24-band table -- see module docstring.
    """
    f = np.asarray(f, dtype=np.float64)
    return 7.0 * np.arcsinh(f / 650.0)


def bark_peaq_to_hz(z: np.ndarray) -> np.ndarray:
    """Bark -> Hz, PEAQ/ITU-R BS.1387 convention: `f = 650*sinh(z/7)`.

    Exact algebraic inverse of `hz_to_bark_peaq`.
    """
    z = np.asarray(z, dtype=np.float64)
    return 650.0 * np.sinh(z / 7.0)


# --- Bark, Zwicker & Terhardt 1980 (JASA 68(5):1523) -- no closed-form inverse ---


def hz_to_bark_zwicker_terhardt(f: np.ndarray) -> np.ndarray:
    """Hz -> Bark, Zwicker & Terhardt 1980 (JASA 68(5):1523):

        z(f) = 13*atan(0.00076*f) + 3.5*atan((f/7500)**2)

    This formula approximates the earlier-published classical 24-critical-
    band table (not the reverse), so it is (measured) the formula that
    matches that table most closely among the three Bark realizations here
    -- see module docstring.
    Strictly increasing in `f` for `f >= 0`, which is what makes bisection
    in `bark_zwicker_terhardt_to_hz` well-defined.
    """
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def critical_bandwidth_hz(f: np.ndarray) -> np.ndarray:
    """Critical bandwidth CB(f) in Hz, Zwicker & Terhardt 1980:

        CB(f) = 25 + 75*(1 + 1.4*(f/1000)**2)**0.69

    Reported for reference/sanity-checking only -- `band_edges` does not
    use this to size bands; it only spaces edges evenly in Bark/ERB-rate.
    Measured against the task's stated reference values: CB(100 Hz) ~= 101,
    CB(1000 Hz) ~= 162, CB(10000 Hz) ~= 2305 Hz.

    Guards against negative/non-finite `f` (added after this module's own
    "propagates honestly" precedent was measured to be false here): the
    formula squares `f`, so it is an EVEN function of frequency and returns
    a value BIT-IDENTICAL to the positive input -- `critical_bandwidth_hz(-1000.0)
    == critical_bandwidth_hz(1000.0)` exactly, for every negative `f`, not
    just large-magnitude ones. That hides a negative-frequency input behind
    a completely plausible positive bandwidth, which is exactly the silent
    plausible-wrong-answer class this module guards against elsewhere; it
    is not something a caller could notice by inspecting the output.

    Raises:
        ValueError: `f` is non-finite (NaN/inf) or negative.
    """
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError(f"f must be finite; got f={f}")
    if np.any(f < 0.0):
        raise ValueError(
            f"f={float(np.min(np.atleast_1d(f)))} Hz is negative; critical_bandwidth_hz squares f, "
            "so a negative frequency silently returns the SAME bandwidth as its positive counterpart "
            "instead of signalling anything is wrong. Pass a non-negative frequency."
        )
    return 25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69


# Bisection seed for `bark_zwicker_terhardt_to_hz`'s geometric bracket
# expansion below: 50 kHz comfortably exceeds every documented request
# (the audible range, and every extrapolation up to a 192 kHz transfer's
# 96 kHz Nyquist -- see the function's own docstring) while still keeping
# the expansion loop's iteration count a meaningful figure rather than
# padding. Named here -- rather than left as a literal duplicated in the
# loop below and in tests/test_dsp_bands.py's regression pin -- so that a
# change to this value changes what that pin measures instead of the test
# comparing a copy of itself against another copy (AGENTS.md #3b).
_BARK_BRACKET_SEED_HZ = 50_000.0

# Iteration ceiling for the same expansion loop. The worst case this
# function ever reaches (see the loop's own comment) takes 47 iterations;
# this cap is that measured worst case's actual safety margin, not a copy
# of it -- see
# test_bark_zwicker_terhardt_to_hz_bracket_expansion_needs_46_doublings_at_the_asymptote_boundary's
# headroom assertion, which reads this constant rather than a literal 200.
_BARK_BRACKET_EXPANSION_CAP = 200

# Bisection convergence parameters for the SECOND loop below (narrowing the
# bracket once it covers z), not the expansion loop above. These used to be
# caller-facing `tol`/`max_iter` keyword arguments; issue #29 found that an
# invalid choice (max_iter<=0, non-finite/negative tol) silently returned a
# bracket edge dressed up as a real answer, with no exception -- e.g.
# max_iter=0 returned 25000 Hz (the bracket's own seed/2) for every input.
# They are SOLVER INTERNALS: a caller who does not know a bisection is
# running here has no principled basis for choosing either value, and
# nothing in this codebase ever passed either one explicitly (verified by
# grep across src/ and tests/ before removing them -- see the PR). Fixed
# here as named constants instead of a tunable surface; removing the
# surface ends this defect class in this function rather than extending it
# with a validated-but-still-present guard.
_BARK_INVERSE_TOL_HZ = 1e-9
_BARK_INVERSE_MAX_ITER = 60


def bark_zwicker_terhardt_to_hz(z: np.ndarray) -> np.ndarray:
    """Bark -> Hz, Zwicker & Terhardt 1980, by bisection.

    `hz_to_bark_zwicker_terhardt` has no closed-form inverse. The formula
    saturates as `f -> infinity` (both `atan` terms approach `pi/2`), so its
    range has a hard ceiling of `13*pi/2 + 3.5*pi/2` (~25.918 Bark); a `z` at
    or beyond that ceiling -- or below 0, the formula's value at `f=0` --
    has no finite Hz value in range and is rejected outright.

    Below the ceiling, the bisection bracket is expanded geometrically from
    a `_BARK_BRACKET_SEED_HZ` (50 kHz) seed until it actually covers `z`,
    then bisected using `hz_to_bark_zwicker_terhardt`'s strict
    monotonicity. A fixed `[0, 50_000]` Hz bracket used to be hardcoded
    here on the assumption that no caller would ever request `z` above
    ~25.5 Bark (the audible range's own ceiling) -- but
    `band_edges(..., allow_extrapolation=True)` lets a
    caller ask for `f_max` well above 50 kHz (e.g. 96 kHz, the Nyquist of a
    192 kHz transfer), and the fixed bracket silently capped every returned
    edge at ~50 kHz with no exception. `_BARK_INVERSE_MAX_ITER` (60) halves
    whatever bracket is found to <2^-60 of its width -- a RELATIVE bound,
    not an absolute one. Within the audible range (20 Hz-24 kHz) the
    bracket stays small, so this also reaches `_BARK_INVERSE_TOL_HZ` in
    absolute Hz (dense round-trip: ~3.6e-10 Hz, re-measured at ~4.3e-10 Hz
    over a 200,000-point 20 Hz-20 kHz sweep when the tol/max_iter
    parameters were removed -- see the PR). Near the asymptote the forward
    map itself saturates (both `atan` terms flatten toward their limits),
    so `hz_to_bark_zwicker_terhardt`'s derivative there is minuscule
    (~1.7e-14 at f=1e9 Hz, measured): float64 rounding noise in evaluating
    the forward formula, not the number of bisection halvings, is what
    limits how precisely `z` can pin down `f` there. Measured: the ~0.102
    Hz absolute error at f=1e9 Hz is IDENTICAL from `max_iter=60` through
    `max_iter=2000`, including with `tol=0` -- more halvings buy nothing
    once the bracket has narrowed past what the saturated forward function
    can resolve at that magnitude. Within the audible range (20 Hz-24 kHz)
    the forward map is nowhere near saturated, so the bisection reaches
    `_BARK_INVERSE_TOL_HZ` in absolute Hz as stated above.

    Raises:
        ValueError: `z` is non-finite (NaN/inf), negative (below the
            formula's `z(0 Hz) = 0` floor -- a negative Bark request used
            to silently converge on ~0 Hz, the bisection bracket's own
            floor, dressed up as a real answer), or at or beyond the
            formula's asymptote (see above).
    """
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError(f"z must be finite; got z={z}")

    if np.any(z < 0.0):
        raise ValueError(
            f"z={float(np.min(np.atleast_1d(z)))} Bark is negative; the Zwicker & Terhardt scale "
            "starts at 0 Bark (z(0 Hz) = 0) and has no Hz value below it. Request a non-negative "
            "Bark value."
        )

    # 13*atan(x) -> 13*pi/2 and 3.5*atan(x**2) -> 3.5*pi/2 as f -> infinity.
    asymptote = 13.0 * (np.pi / 2.0) + 3.5 * (np.pi / 2.0)
    if np.any(z >= asymptote):
        raise ValueError(
            f"z={float(np.max(np.atleast_1d(z)))} Bark is at or beyond the Zwicker & Terhardt "
            f"formula's asymptote (~{asymptote:.6f} Bark, its limit as f -> infinity); no finite "
            "Hz value maps to it. Request a lower Bark value (or a lower f_max)."
        )

    lo = np.zeros_like(z)
    hi = np.full_like(z, _BARK_BRACKET_SEED_HZ)
    # Geometric bracket expansion. The worst case -- z at the largest
    # representable float64 strictly below `asymptote` -- needs exactly 46
    # doublings (measured) before `hi` exceeds it; because this loop checks
    # BEFORE it doubles, observing that as a `break` takes the loop's 47th
    # iteration, not its 46th -- a cap of 46 would exit `range(46)` without
    # ever running that check. The guard above already rejects every z that
    # could need more than 46 doublings, so 47 iterations is this loop's
    # true worst case, and `_BARK_BRACKET_EXPANSION_CAP` below has ~4x that
    # headroom, not ~4x the doubling count. See
    # `test_bark_zwicker_terhardt_to_hz_bracket_expansion_needs_46_doublings_at_the_asymptote_boundary`
    # for a regression pin on the 46/47 figures themselves. There is
    # deliberately no "ran out of doublings" fallback branch here: given
    # the guard, that branch would
    # be unreachable and therefore untestable, which is worse than no
    # branch at all -- see AGENTS.md and the review that caught this.
    for _ in range(_BARK_BRACKET_EXPANSION_CAP):
        too_low = hz_to_bark_zwicker_terhardt(hi) < z
        if not np.any(too_low):
            break
        hi = np.where(too_low, hi * 2.0, hi)

    for _ in range(_BARK_INVERSE_MAX_ITER):
        mid = 0.5 * (lo + hi)
        too_high = hz_to_bark_zwicker_terhardt(mid) > z
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.all(hi - lo < _BARK_INVERSE_TOL_HZ):
            break
    return 0.5 * (lo + hi)


# --- ERB, Glasberg & Moore 1990 (Hearing Research 47:103) ---

# DECISION RECORD (issue #31) -- which constant "21.4" means, and why
# ---------------------------------------------------------------------
# Glasberg & Moore, "Derivation of auditory filter shapes from notched-noise
# data", Hearing Research 47:103-138 (1990), p.114, disagrees with itself
# about this constant. Four candidate values appear, and only one is
# shipped here:
#
#   CHOSEN:  21.4 / 4.37 / 24.7  -- eq. (4) and eq. (3), PRINTED TEXT, p.114:
#            ERBrate(f) = 21.4*log10(4.37*f/1000 + 1)      -- eq. (4)
#            ERB(f)     = 24.7*(4.37*f/1000 + 1)            -- eq. (3)
#            This is what a reader checking this module's docstring against
#            the actual printed paper will see. No shipped behaviour
#            changed when this was decided -- keeping the printed equation
#            was always what this module implemented.
#
#   REJECTED: 21.366 / 4.368 / 24.673 -- the paper's OWN Fortran appendix
#            (pp. 132, 135, 138): `c1 = 24.673`, `c2 = 4.368`,
#            `c3 = 2302.6/(c1*c2) ~= 21.366`. Rejected because we follow
#            the paper's printed equation, not its reference
#            implementation -- but recorded here because it is the
#            better-anchored choice: Jesteadt, Wroblewski & High, JASA
#            145(6):3586 (2019), Table I (doi:10.1121/1.5111751, open on
#            PMC as PMC6584171) tabulates ERBN band edges at 3, 5, ..., 33
#            Cams in integer Hz, and 21.366/4.368 agrees with that table to
#            within about 1 Hz up to ~5 kHz (see
#            tests/test_dsp_bands_erb_glasberg_moore_anchor.py for the
#            measured comparison against BOTH constant choices). This
#            module's shipped 21.4/4.37 drifts steadily against the same
#            table, reaching about -52 Hz at 33 Cams (7795 Hz) -- a known,
#            measured, and deliberately accepted gap, not drift or a bug.
#
#   REJECTED: ~21.33 -- eq. (4) evaluated from ITS OWN rounded printed
#            constants (2302.6/(24.7*4.37)). This is an intermediate
#            quantity that appears in neither the paper's printed text nor
#            its Fortran appendix; nobody publishes it. Rejected as not a
#            real candidate, only an artifact of composing two already-
#            rounded numbers.
#
# A future reader who notices 21.4 doesn't reproduce Jesteadt 2019's table,
# or doesn't match the paper's own Fortran code, should read this as the
# recorded, chosen gap above -- not rediscover the disagreement and assume
# it is a bug. See tests/test_dsp_bands_erb_glasberg_moore_anchor.py for
# the external anchor (Jesteadt 2019 Table I) and the literal printed-
# equation anchor (independent of this module's own constants).
#
# SOURCE CAVEAT: the G&M 1990 PDF this issue was originally raised against
# is a third-party mirror (audres.org), whose OCR text layer misreads
# eq. (4)'s "21.4" as "22.4". Two independent vision reads of 300-600 dpi
# crops of that same mirror both read 21.4, not 22.4. The publisher's copy
# (Elsevier/ScienceDirect, doi:10.1016/0378-5955(90)90170-T) is NOT open
# access -- confirmed via the Unpaywall API (`is_oa: false`, no repository
# copy, checked 2026-09-29) -- so it could not be read directly. As
# independent corroboration from a DIFFERENT, non-audres.org source:
# Simpson, Terrell & Reiss, "A Practical Step-by-Step Guide to the
# Time-Varying Loudness Model of Moore, Glasberg and Baer", AES Convention
# Paper 8873 (2013), hosted on the last author's own Queen Mary University
# of London page (joshreiss.github.io), restates both equations verbatim,
# citing Moore: eq. (4) `ERB = 24.7*(0.00437*f + 1)`, eq. (5)
# `n = 21.4*log10(0.00437*fc + 1)` -- 21.4, 4.37 (as 0.00437), and 24.7,
# matching what this module ships. This is a secondary restatement, not a
# first-hand read of the original 1990 paper's own typeset page -- no
# constant was changed on the strength of either the OCR layer or this
# secondary source; both point the same way, and this reads as
# corroboration of the two vision reads, not a replacement for them.


def hz_to_erb_rate(f: np.ndarray) -> np.ndarray:
    """Hz -> ERB-rate ("Cams"), Glasberg & Moore 1990 (Hearing Research
    47:103), eq. (4), PRINTED TEXT, p.114 -- see the DECISION RECORD
    immediately above this function for why 21.4/4.37 (not the paper's
    own Fortran-appendix 21.366/4.368, and not the ~21.33 re-derived from
    eq. (4)'s own rounded constants) is what this module ships:

    ERBrate(f) = 21.4*log10(4.37*f/1000 + 1)

    Deliberately does not validate `f` (see module docstring) -- a negative
    input's sign stays visible in the output (negative ERB-rate) across
    virtually the entire negative range. One float64-resolution exception,
    measured directly: for `-1.27e-14 < f < 0` (`4.37*f/1000` too small in
    magnitude to move `1.0 + 4.37*f/1000` off exactly `1.0` at float64
    precision), the sign is lost and this returns exactly `0.0`, identical
    to `hz_to_erb_rate(0.0)`. Below that bound (`f <= -1.27e-14`) the sign
    is visible again (a negative result). This is a vanishingly small
    blind spot, not a practical concern for any real caller, but it is the
    literal boundary, not an approximation.
    """
    f = np.asarray(f, dtype=np.float64)
    return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)


def erb_rate_to_hz(erb_rate: np.ndarray) -> np.ndarray:
    """ERB-rate -> Hz: exact algebraic inverse of `hz_to_erb_rate`. See the
    DECISION RECORD above `hz_to_erb_rate` for why 21.4/4.37 is shipped
    (the paper's printed equation) rather than its own Fortran-appendix
    21.366/4.368 or the ~21.33 re-derivation.

    f = (10**(E/21.4) - 1)/0.00437

    Deliberately does not validate `erb_rate` (see module docstring) -- a
    negative input's sign stays visible in the output (negative Hz) across
    virtually the entire negative range, with the same class of float64-
    resolution exception as `hz_to_erb_rate`: measured directly, for
    `-5.16e-16 < erb_rate < 0` (`erb_rate/21.4` too small in magnitude to
    move `10**(erb_rate/21.4)` off exactly `1.0` at float64 precision),
    the sign is lost and this returns exactly `0.0`. Below that bound
    (`erb_rate <= -5.16e-16`) the sign is visible again. Vanishingly small,
    not the practical range any real caller passes, but the literal
    boundary rather than an approximation.
    """
    erb_rate = np.asarray(erb_rate, dtype=np.float64)
    return (10.0 ** (erb_rate / 21.4) - 1.0) / 0.00437


def erb_bandwidth_hz(f: np.ndarray) -> np.ndarray:
    """Equivalent rectangular bandwidth ERB(f) in Hz, Glasberg & Moore 1990,
    eq. (3), PRINTED TEXT, p.114 -- see the DECISION RECORD above
    `hz_to_erb_rate` for why 24.7/4.37 (not the paper's own Fortran-
    appendix 24.673/4.368) is what this module ships:

        ERB(f) = 24.7*(4.37*f/1000 + 1)

    Reference/sanity-checking only, matching `critical_bandwidth_hz`'s role
    for the Bark side. Measured against the task's stated reference values:
    ERB(100 Hz) ~= 35, ERB(1000 Hz) ~= 133, ERB(10000 Hz) ~= 1104 Hz.

    Guards against negative/non-finite `f` (added after this module's own
    "propagates honestly" precedent was measured to be false here): the
    formula is linear in `f` with a small coefficient (`4.37/1000`), so it
    is NOT even (unlike `critical_bandwidth_hz`) but still hides small-
    magnitude negative input -- `erb_bandwidth_hz(-5.0) == 24.1603...`, a
    perfectly plausible ERB value close to `erb_bandwidth_hz(0.0) == 24.7`.
    It only turns visibly negative once `f < -1000/4.37 ~= -228.83` Hz;
    silently plausible across the entire `(-228.83, 0)` Hz range is most of
    the range a caller's typo or off-by-one would actually produce, not an
    edge case.

    Raises:
        ValueError: `f` is non-finite (NaN/inf) or negative.
    """
    f = np.asarray(f, dtype=np.float64)
    if not np.all(np.isfinite(f)):
        raise ValueError(f"f must be finite; got f={f}")
    if np.any(f < 0.0):
        raise ValueError(
            f"f={float(np.min(np.atleast_1d(f)))} Hz is negative; erb_bandwidth_hz's small linear "
            "coefficient means negative frequencies down to about -228.83 Hz silently return a "
            "plausible-looking positive bandwidth instead of signalling anything is wrong. Pass a "
            "non-negative frequency."
        )
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


# --- Named scales, dispatch, and the band-edge / bin-mapping API ---

_BARK_VARIANTS: dict[str, tuple] = {
    "bark_peaq": (hz_to_bark_peaq, bark_peaq_to_hz),
    "bark_zwicker_terhardt": (hz_to_bark_zwicker_terhardt, bark_zwicker_terhardt_to_hz),
}
_ERB_VARIANTS: dict[str, tuple] = {
    "erb_glasberg_moore": (hz_to_erb_rate, erb_rate_to_hz),
}
_ALL_SCALES: dict[str, tuple] = {**_BARK_VARIANTS, **_ERB_VARIANTS}

#: Every scale name this module knows, Bark realizations followed by ERB.
SCALES: tuple[str, ...] = tuple(_ALL_SCALES.keys())
#: Just the Bark realizations -- the ones gated by `_BARK_TABULATED_LIMIT_HZ`.
BARK_SCALES: tuple[str, ...] = tuple(_BARK_VARIANTS.keys())
#: Just the ERB realization(s).
ERB_SCALES: tuple[str, ...] = tuple(_ERB_VARIANTS.keys())

_DEFAULT_BARK_VARIANT = "bark_peaq"


def _scale_funcs(scale: str) -> tuple:
    if scale not in _ALL_SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose one of {SCALES}")
    return _ALL_SCALES[scale]


def hz_to_bark(f: np.ndarray, variant: str = _DEFAULT_BARK_VARIANT) -> np.ndarray:
    """Hz -> Bark using the named `variant` (default `"bark_peaq"`, the one
    closed-form-both-ways realization -- convenient for casual scalar use;
    see module docstring for why `band_edges` does NOT reuse this default).
    """
    if variant not in _BARK_VARIANTS:
        raise ValueError(f"unknown Bark variant {variant!r}; choose one of {BARK_SCALES}")
    forward, _ = _BARK_VARIANTS[variant]
    return forward(f)


def bark_to_hz(z: np.ndarray, variant: str = _DEFAULT_BARK_VARIANT) -> np.ndarray:
    """Bark -> Hz using the named `variant` (default `"bark_peaq"`); see `hz_to_bark`."""
    if variant not in _BARK_VARIANTS:
        raise ValueError(f"unknown Bark variant {variant!r}; choose one of {BARK_SCALES}")
    _, inverse = _BARK_VARIANTS[variant]
    return inverse(z)


def band_edges(
    n_bands: int,
    scale: str,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    allow_extrapolation: bool = False,
) -> dict:
    """Construct `n_bands` contiguous bands, evenly spaced in `scale`-space,
    covering `[f_min, f_max]` Hz.

    `scale` has NO default -- see the module docstring's "THE SCALE IS NOT
    COSMETIC" section. Pass one of `SCALES` explicitly.

    Args:
        n_bands: Number of bands (>= 1). 32 is a common reference point
            (Trackspacer, iZotope Neutron Unmask, DeepFilterNet all use 32;
            RNNoise uses 22) but is never assumed or defaulted here.
        scale: One of `SCALES` -- REQUIRED, see above.
        f_min: Bottom of the covered range in Hz (> 0).
        f_max: Top of the covered range in Hz (> f_min).
        allow_extrapolation: For a Bark `scale`, `f_max` above
            `_BARK_TABULATED_LIMIT_HZ` (15500 Hz = 24 Bark, the top of the
            classical table) is extrapolation, not a validated conversion --
            see module docstring. `False` (default) raises `ValueError`;
            `True` proceeds and the returned dict's `"extrapolated"` is
            `True`. Ignored for `erb_glasberg_moore` (no such gate exists
            for ERB in the task's source material).

    Returns:
        {
            "scale": scale,             # exactly as passed -- see docstring
            "n_bands": n_bands,
            "f_min": float(f_min),
            "f_max": float(f_max),
            "edges_hz": ndarray, shape (n_bands + 1,), strictly increasing.
                Band `i` covers `[edges_hz[i], edges_hz[i+1]]`; adjacent
                bands share their boundary exactly (same float), so the
                edges are non-overlapping and gapless by construction.
            "centers_hz": ndarray, shape (n_bands,) -- the scale-space
                midpoint of each band, converted back to Hz. This is what
                `bin_band_weights` uses as each band's triangle peak.
            "extrapolated": bool -- True iff a Bark `scale` was asked to
                cover `f_max` beyond the tabulated 15.5 kHz limit.
        }

    Raises:
        ValueError: `n_bands` is not an integer, or is `< 1`; `f_min`/
            `f_max` are not real numbers, are non-finite (inf/NaN), or
            `f_min` is outside `(0, f_max)`; `allow_extrapolation` is not a
            `bool`; unknown `scale`; or a Bark `scale` with `f_max` beyond
            the tabulated limit and `allow_extrapolation=False`.
    """
    if not isinstance(n_bands, (int, np.integer)) or isinstance(n_bands, bool):
        raise ValueError(f"n_bands must be an integer; got n_bands={n_bands!r} (type {type(n_bands).__name__})")
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1; got {n_bands}")
    for _name, _value in (("f_min", f_min), ("f_max", f_max)):
        if not isinstance(_value, (int, float, np.integer, np.floating)) or isinstance(_value, bool):
            raise ValueError(f"{_name} must be a real number; got {_name}={_value!r} (type {type(_value).__name__})")
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        raise ValueError(f"f_min/f_max must be finite; got f_min={f_min}, f_max={f_max}")
    if not (0.0 < f_min < f_max):
        raise ValueError(f"f_min/f_max must satisfy 0 < f_min < f_max; got f_min={f_min}, f_max={f_max}")
    if not isinstance(allow_extrapolation, bool):
        raise ValueError(
            f"allow_extrapolation must be a bool; got allow_extrapolation={allow_extrapolation!r} "
            f"(type {type(allow_extrapolation).__name__}) -- a truthy non-bool (e.g. a non-empty "
            "string, even the string 'False') would silently be treated as True regardless of its "
            "apparent meaning"
        )
    forward, inverse = _scale_funcs(scale)

    extrapolated = scale in BARK_SCALES and f_max > _BARK_TABULATED_LIMIT_HZ
    if extrapolated and not allow_extrapolation:
        raise ValueError(
            f"f_max={f_max} Hz exceeds the classical Bark table's tabulated limit "
            f"({_BARK_TABULATED_LIMIT_HZ} Hz = 24 Bark) for scale={scale!r}; beyond that, "
            "the formula is extrapolation, not a validated conversion. Pass "
            "allow_extrapolation=True to proceed anyway, or lower f_max."
        )

    z_min = float(forward(np.asarray(f_min, dtype=np.float64)))
    z_max = float(forward(np.asarray(f_max, dtype=np.float64)))
    z_edges = np.linspace(z_min, z_max, n_bands + 1)
    edges_hz = np.asarray(inverse(z_edges), dtype=np.float64)

    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    centers_hz = np.asarray(inverse(z_centers), dtype=np.float64)

    return {
        "scale": scale,
        "n_bands": int(n_bands),
        "f_min": float(f_min),
        "f_max": float(f_max),
        "edges_hz": edges_hz,
        "centers_hz": centers_hz,
        "extrapolated": bool(extrapolated),
    }


def _bin_frequencies_hz(n_fft: int, sr: float) -> np.ndarray:
    """The rfft bin frequency grid `bin_band_weights` maps onto bands: bin
    `k` sits at `k * sr / n_fft`, `n_bins = n_fft // 2 + 1` -- the same
    convention as `aud.dsp.stft.analyze` (`stft_properties`'s
    `bin_hz = sr / n_fft`; see module docstring for the `noisereduce`
    `sr/(n_fft/2)` bug this deliberately does not reproduce).

    Factored out of `bin_band_weights` so it can be asserted directly in
    tests (element-wise against `k * sr / n_fft`) rather than only inferred
    indirectly from where energy lands in one band at one configuration --
    that indirect check was measured to pass unchanged against an
    off-by-one bin index AND an off-by-one denominator (see
    tests/test_dsp_bands.py).
    """
    n_bins = n_fft // 2 + 1
    return np.arange(n_bins, dtype=np.float64) * sr / n_fft


def bin_band_weights(bands: dict, n_fft: int, sr: float) -> np.ndarray:
    """Triangular partition-of-unity weights mapping each STFT bin to bands.

    Args:
        bands: As returned by `band_edges`.
        n_fft: Window/FFT length in samples -- must match whatever produced
            the spectrum this will be applied to (`aud.dsp.stft.analyze`).
        sr: Sample rate in Hz. Typed as `float`, not `int`: the runtime
            check below (`isfinite` and `> 0`) has always accepted a
            fractional `sr`, so `int` was simply the wrong annotation for
            this function's own behaviour.

    Returns:
        `(n_bands, n_bins)` array, `n_bins = n_fft // 2 + 1` (the same rfft
        convention as `aud.dsp.stft.analyze`), where column `k` (bin `k`,
        at frequency `k * sr / n_fft` -- NOT `sr / (n_fft / 2)`, the
        `noisereduce` bug this module deliberately does not reproduce; see
        module docstring) sums to exactly 1 across all bands: bin `k`'s
        energy is fully accounted for, never doubled or dropped.

    Construction (see module docstring's "Bin -> band weights" section for
    the full partition-of-unity argument): band `i`'s triangle peaks at
    `bands["centers_hz"][i]` and falls linearly to zero at its neighbours'
    centers, EXCEPT the two end bands, whose outward side is flattened to a
    constant weight of 1 (from `f_min` out to the first center, and from
    the last center out to `f_max` and beyond) rather than tapering to
    zero. Two adjacent triangles' complementary linear ramps sum to 1
    across every interior segment; the flattened ends make the two outer
    segments (and everything beyond `f_min`/`f_max`) sum to 1 as well, with
    no separate clamping step required -- so this holds for every bin the
    STFT can produce, whether or not `[f_min, f_max]` covers `[0, sr/2]`.
    One consequence of the flattened ends: a caller cannot tell that flat-
    region energy apart from genuine in-band energy, so DC offset (bin 0)
    or ultrasonic content (above the last center) reads as full-strength
    band-0/band-N energy indistinguishable from a real signal there.

    Raises:
        ValueError: `bands` is not a dict, or is missing one of its
            required keys (`n_bands`, `f_min`, `f_max`, `centers_hz` -- a
            hand-built dict, or an unrelated object, that used to fail with
            an unnamed `TypeError`/`KeyError` instead); `n_fft` is not a
            positive integer, or `sr` is not a real, positive finite
            number; `bands["n_bands"]` does not match
            `len(bands["centers_hz"])` (a hand-built or hand-edited `bands`
            dict with mismatched fields, which would otherwise silently
            produce weights for the wrong number of bands, or an `IndexError`
            instead of a documented `ValueError` when centers are too few);
            `bands["f_min"]`/`bands["f_max"]`/`bands["centers_hz"]` contain a
            non-finite value (inf/NaN); or `bands["centers_hz"]` (together
            with `f_min`/`f_max`) is not strictly increasing -- a degenerate
            `band_edges` request (e.g. `n_bands` so large, relative to the
            covered scale span, that two centers collide at float64
            precision) that would otherwise silently divide by zero.
    """
    if not isinstance(bands, dict):
        raise ValueError(f"bands must be a dict as returned by band_edges(); got type {type(bands).__name__}")
    _required_keys = {"n_bands", "f_min", "f_max", "centers_hz"}
    _missing_keys = _required_keys - bands.keys()
    if _missing_keys:
        raise ValueError(
            f"bands is missing required key(s) {sorted(_missing_keys)}; expected a dict as "
            "returned by band_edges(), with keys n_bands/f_min/f_max/centers_hz"
        )
    if not isinstance(n_fft, (int, np.integer)) or n_fft <= 0:
        raise ValueError(f"n_fft must be a positive integer; got n_fft={n_fft!r}")
    if not isinstance(sr, (int, float, np.integer, np.floating)) or isinstance(sr, bool):
        raise ValueError(f"sr must be a real number; got sr={sr!r} (type {type(sr).__name__})")
    if not (np.isfinite(sr) and sr > 0):
        raise ValueError(f"sr must be a positive finite number; got sr={sr!r}")

    n_bands = bands["n_bands"]
    f_min = bands["f_min"]
    f_max = bands["f_max"]
    centers = np.asarray(bands["centers_hz"], dtype=np.float64)

    if len(centers) != n_bands:
        raise ValueError(
            f"bands['n_bands']={n_bands} does not match len(bands['centers_hz'])={len(centers)}; "
            "a bands dict must have exactly n_bands centers"
        )

    control = np.concatenate(([f_min], centers, [f_max]))
    if not np.all(np.isfinite(control)):
        raise ValueError(
            f"bands['f_min']/['f_max']/['centers_hz'] must all be finite; got "
            f"f_min={f_min}, f_max={f_max}, centers_hz={centers}"
        )
    if np.any(np.diff(control) <= 0):
        raise ValueError(
            "band centers are not strictly increasing between f_min and f_max; "
            f"n_bands={n_bands} is degenerate for this scale/frequency range "
            "(centers have collided at float64 precision)"
        )

    bin_hz = _bin_frequencies_hz(n_fft, sr)
    n_bins = bin_hz.shape[0]

    weights = np.zeros((n_bands, n_bins), dtype=np.float64)
    for i in range(n_bands):
        lo, peak, hi = control[i], control[i + 1], control[i + 2]
        rising = np.ones_like(bin_hz) if i == 0 else (bin_hz - lo) / (peak - lo)
        falling = np.ones_like(bin_hz) if i == n_bands - 1 else (hi - bin_hz) / (hi - peak)
        weights[i] = np.clip(np.minimum(rising, falling), 0.0, 1.0)

    return weights


def band_energy(spectrum: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum per-bin energy (or magnitude) into bands using `bin_band_weights`.

    Args:
        spectrum: Real, non-negative array whose axis 0 is the bin axis
            (matching `aud.dsp.stft.analyze`'s `(n_bins, ...)` convention --
            pass `np.abs(spectrum)**2` for power or `np.abs(spectrum)` for
            magnitude; this function does not care which, it only sums).
            May carry extra trailing axes (frames, channels, ...).
        weights: `(n_bands, n_bins)`, as returned by `bin_band_weights`.

    Returns:
        `(n_bands, ...)` -- band `b` = `sum_k weights[b, k] * spectrum[k, ...]`.
        Summed again over `b`, this equals `spectrum`'s own per-bin sum
        exactly (to float64 rounding), because `weights` sums to 1 per bin
        (see `bin_band_weights`) -- this is the energy-conservation property
        this module's tests assert directly.

    Guards against non-finite or negative `spectrum`/`weights` entries
    (added after this module's own "propagates honestly" precedent was
    measured to be false here): this function is a weighted SUM, and a sum
    is exactly the shape that can absorb one bad (negative) contributor
    into a smaller but still-positive, still-plausible band value -- it
    does not reliably surface as a visible negative in the output the way
    a single closed-form formula does. Measured: one negative `spectrum`
    bin changed a clean band value of `19.45` to `13.58`, with every band
    in the result still `>= 0` -- no signal anything was wrong. This is a
    hot per-frame path (see AGENTS.md #8), so the added cost is measured
    directly rather than assumed, both for the guard itself and for the
    cheaper equivalent it was rewritten to below.

    **Guard cost, measured (2049-bin, 32-band, float64, this machine --
    see the PR for the harness)**: the guard is `a.min() >= 0 and
    np.isfinite(a.max())` per array rather than an elementwise
    `isfinite(a) & (a >= 0)` boolean pass -- two O(n) reductions with no
    boolean temporaries instead of two temporaries plus a third `np.all`
    reduction. At a single frame (the realistic per-hop call for a 48 kHz/
    hop-512 stream), added overhead over the unguarded sum dropped from
    **~47 us to ~24 us** (about a 2x reduction, repeated runs ranged
    ~1.4x-2.3x depending on system noise); at 64 frames in one call,
    **~149 us to ~74 us**. Against a 48 kHz/hop-512 frame budget of
    10,667 us, that is **~0.44% -> ~0.22%** of one frame's time budget --
    real, but small next to the alternative (a silent plausible-wrong band
    value). The two reductions are equivalent to the elementwise check:
    `a.min() >= 0` is False for ANY negative entry or a NaN (NaN
    propagates through `min`, and `nan >= 0` is False) and for -inf (it
    sorts as the minimum); `isfinite(a.max())` catches the one case the
    min check cannot -- a +inf entry, which is not the array's minimum
    unless every entry is +inf. An empty `spectrum`/`weights` is a
    no-op here (matching this function's pre-guard behaviour): `.min()`/
    `.max()` raise on a zero-size array, so size 0 short-circuits before
    either reduction runs.

    Raises:
        ValueError: `spectrum`'s bin axis (0) does not match `weights`'s
            bin axis (1) -- they came from a different `n_fft`; or either
            array contains a non-finite (NaN/inf) or negative value.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if spectrum.shape[0] != weights.shape[1]:
        raise ValueError(
            f"spectrum's bin axis (0) has length {spectrum.shape[0]}, but weights has "
            f"{weights.shape[1]} bins; they must come from the same n_fft"
        )
    # Two reductions per array on the happy path, no boolean temporaries --
    # measured cheaper than the elementwise `isfinite(a) & (a >= 0)` pass
    # this replaced (see the PR this guard shipped in for the before/after
    # timing). Equivalent to the elementwise check: `a.min() >= 0` is False
    # whenever ANY element is negative OR NaN (NaN propagates through
    # `min`, and `nan >= 0` is False), and covers -inf the same way (-inf
    # sorts as the array's minimum, so `-inf >= 0` is False); `isfinite(a.max())`
    # is False whenever the array's maximum is +inf (the one case the min
    # check cannot see -- a +inf entry is not the minimum unless every
    # other entry is also +inf). An empty array is a happy-path no-op here,
    # matching this function's pre-guard behaviour (an empty `spectrum`/
    # `weights` never raised): `array.min()`/`.max()` raise on a zero-size
    # array, so size 0 is checked first and short-circuits to "valid".
    # Only on the rare failure path do we pay for a second, elementwise
    # pass to say which condition failed.
    if not (spectrum.size == 0 or (spectrum.min() >= 0.0 and np.isfinite(spectrum.max()))):
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum must be finite")
        raise ValueError(
            "spectrum must be non-negative (it is a magnitude/power quantity); a negative bin can be "
            "silently absorbed into a smaller-but-still-plausible positive band energy by the "
            "weighted sum below, with no signal anything was wrong"
        )
    if not (weights.size == 0 or (weights.min() >= 0.0 and np.isfinite(weights.max()))):
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        raise ValueError(
            "weights must be non-negative (as returned by bin_band_weights); a negative weight can be "
            "silently absorbed into a smaller-but-still-plausible positive band energy by the "
            "weighted sum below, with no signal anything was wrong"
        )
    return np.tensordot(weights, spectrum, axes=([1], [0]))
