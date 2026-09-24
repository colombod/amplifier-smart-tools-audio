"""Behavioural tests for perceptual band mapping (Hz<->Bark/ERB, band edges,
bin->band energy summation).

All tests use pure numeric inputs -- no committed binary fixtures, no mocks
(AGENTS.md #3b: these are pure array functions, so real generated values are
the correct substitute, and there is nothing here a hand-written fake could
usefully stand in for).
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import bands

# --- 1. Hz<->scale<->Hz round trips: dense, exhaustive, every audible frequency ---

# 1 Hz steps across the audible range -- these are pure functions, so this is
# cheap and it is the whole point of testing this step early (task spec).
_DENSE_FREQS_HZ = np.arange(20.0, 20001.0, 1.0)


@pytest.mark.parametrize(
    "variant",
    ["bark_peaq", "bark_zwicker_terhardt"],
)
def test_bark_round_trip_dense(variant):
    z = bands.hz_to_bark(_DENSE_FREQS_HZ, variant=variant)
    back = bands.bark_to_hz(z, variant=variant)
    max_err = float(np.max(np.abs(back - _DENSE_FREQS_HZ)))
    print(f"\n[bands] {variant} dense round-trip max err = {max_err:.3e} Hz")
    # Zwicker&Terhardt inverts by bisection; give it a looser (but still
    # tiny) tolerance than the two closed-form pairs.
    tol = 1e-6 if variant == "bark_zwicker_terhardt" else 1e-8
    assert max_err < tol


def test_erb_round_trip_dense():
    e = bands.hz_to_erb_rate(_DENSE_FREQS_HZ)
    back = bands.erb_rate_to_hz(e)
    max_err = float(np.max(np.abs(back - _DENSE_FREQS_HZ)))
    print(f"\n[bands] erb_glasberg_moore dense round-trip max err = {max_err:.3e} Hz")
    assert max_err < 1e-8


def test_bark_zwicker_terhardt_round_trip_scalar_and_extremes():
    """Bisection-specific: exercise it at 0 Hz and near its search bracket's
    interior extremes, not just the dense sweep's 20 Hz floor."""
    for f in (0.0, 1.0, 24000.0):
        z = bands.hz_to_bark_zwicker_terhardt(f)
        back = float(bands.bark_zwicker_terhardt_to_hz(z))
        assert back == pytest.approx(f, abs=1e-6)


# --- 2. Agreement with the published Zwicker 24-band critical-band table ---

# The classical Zwicker 24-critical-band table's edges (Hz). By definition
# edge i sits at exactly i Bark (i = 0..24) -- this IS what "24 critical
# bands" means. This is the published reference artifact the task calls
# out, not a value re-derived from this module's own formulas.
_ZWICKER_24_BAND_EDGES_HZ = np.array(
    [
        0,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ],
    dtype=np.float64,
)
_ZWICKER_24_BAND_EXPECTED_BARK = np.arange(25.0)


@pytest.mark.parametrize(
    ("variant", "tolerance_bark"),
    [
        # Zwicker&Terhardt IS the formula the table was derived from --
        # tightest agreement, measured ~0.20 Bark.
        ("bark_zwicker_terhardt", 0.25),
        # PEAQ is a smooth closed-form approximation for a different
        # application (perceptual audio quality models); it diverges from
        # the classical table by ~3 Bark at the top of the range. Included
        # here so the disagreement is measured and reported, not hidden.
        ("bark_peaq", 3.1),
    ],
)
def test_agreement_with_zwicker_24_band_table(variant, tolerance_bark):
    computed = bands.hz_to_bark(_ZWICKER_24_BAND_EDGES_HZ, variant=variant)
    err = np.abs(computed - _ZWICKER_24_BAND_EXPECTED_BARK)
    max_err = float(np.max(err))
    print(f"\n[bands] {variant} vs Zwicker 24-band table: max err = {max_err:.3f} Bark")
    assert max_err < tolerance_bark


def test_peaq_disagreement_with_table_is_concentrated_at_the_high_end():
    """Document, rather than hide, where PEAQ's larger table disagreement
    actually comes from: it grows toward 15.5 kHz, it is not a uniform
    offset across the whole range."""
    computed = bands.hz_to_bark(_ZWICKER_24_BAND_EDGES_HZ, variant="bark_peaq")
    err = np.abs(computed - _ZWICKER_24_BAND_EXPECTED_BARK)
    assert err[-1] > err[5], "PEAQ's table disagreement should grow toward the top of the range"


# --- 3. Bandwidth reference values (CB / ERB), task's stated reference figures ---


@pytest.mark.parametrize(
    ("f", "expected_cb"),
    [(100.0, 101.0), (1000.0, 162.0), (10000.0, 2305.0)],
)
def test_critical_bandwidth_matches_reference_values(f, expected_cb):
    assert bands.critical_bandwidth_hz(f) == pytest.approx(expected_cb, abs=1.0)


@pytest.mark.parametrize(
    ("f", "expected_erb"),
    [(100.0, 35.0), (1000.0, 133.0), (10000.0, 1104.0)],
)
def test_erb_bandwidth_matches_reference_values(f, expected_erb):
    assert bands.erb_bandwidth_hz(f) == pytest.approx(expected_erb, abs=1.0)


# --- 4. Derived spans and the Bark<->ERB ratio (why the scale must be explicit) ---


def test_span_20hz_to_20khz_matches_derived_reference_values():
    span_zt = bands.hz_to_bark_zwicker_terhardt(20000.0) - bands.hz_to_bark_zwicker_terhardt(20.0)
    span_erb = bands.hz_to_erb_rate(20000.0) - bands.hz_to_erb_rate(20.0)
    print(f"\n[bands] 20Hz-20kHz span: Z&T={span_zt:.2f} Bark, ERB={span_erb:.2f} Cams")
    assert span_zt == pytest.approx(24.4, abs=0.1)
    assert span_erb == pytest.approx(40.9, abs=0.1)


@pytest.mark.parametrize(
    ("f", "expected_erb_per_bark"),
    [(100.0, 2.8), (1000.0, 1.2), (10000.0, 2.1)],
)
def test_erb_per_bark_ratio_varies_with_frequency(f, expected_erb_per_bark):
    """THE reason `band_edges` requires an explicit `scale`: a fixed
    per-Bark slope (e.g. the next epic step's dB/Bark spreading skirts) is
    NOT a fixed multiple of ERB -- the ratio itself changes with frequency.
    Measured here as a central-difference derivative ratio (dE/dz, ERB per
    Bark) using the Zwicker&Terhardt Bark map against Glasberg-Moore ERB-rate,
    against the task's stated approximate reference ratios.
    """
    h = 1e-3

    def d_bark(fc):
        return (bands.hz_to_bark_zwicker_terhardt(fc + h) - bands.hz_to_bark_zwicker_terhardt(fc - h)) / (2 * h)

    def d_erb(fc):
        return (bands.hz_to_erb_rate(fc + h) - bands.hz_to_erb_rate(fc - h)) / (2 * h)

    erb_per_bark = float(d_erb(f) / d_bark(f))
    print(f"\n[bands] at {f} Hz: 1 Bark ~= {erb_per_bark:.2f} ERB (expected ~{expected_erb_per_bark})")
    # A loose tolerance: this is a local-derivative sanity check, not a
    # precise reproduction of the task's own approximate figures.
    assert erb_per_bark == pytest.approx(expected_erb_per_bark, rel=0.25)


# --- 5. Band-edge construction: monotonic, non-overlapping, gapless ---


@pytest.mark.parametrize("scale", bands.SCALES)
@pytest.mark.parametrize("n_bands", [1, 2, 8, 32, 256])
def test_band_edges_monotonic_nonoverlapping_gapless(scale, n_bands):
    f_max = 15000.0 if scale in bands.BARK_SCALES else 20000.0
    result = bands.band_edges(n_bands, scale, f_min=20.0, f_max=f_max)
    edges = result["edges_hz"]
    assert edges.shape == (n_bands + 1,)
    assert np.all(np.diff(edges) > 0), "edges must be strictly increasing (no overlap, no zero-width band)"
    assert edges[0] == pytest.approx(20.0, abs=1e-6)
    assert edges[-1] == pytest.approx(f_max, abs=1e-6)
    # Gapless-by-construction: band i's right edge IS band i+1's left edge
    # (literally the same array element), so there is nothing further to
    # assert here beyond strict monotonicity -- recorded for the reader.
    assert result["centers_hz"].shape == (n_bands,)
    assert np.all(result["centers_hz"] > edges[:-1])
    assert np.all(result["centers_hz"] < edges[1:])
    assert result["scale"] == scale


def test_band_edges_scale_is_required_not_defaulted():
    """`scale` has no default -- calling positionally without it must fail,
    not silently pick one (see module docstring's "not cosmetic" section)."""
    with pytest.raises(TypeError):
        bands.band_edges(32)  # type: ignore[call-arg]


def test_band_edges_rejects_bad_n_bands_and_range():
    with pytest.raises(ValueError, match="n_bands"):
        bands.band_edges(0, "bark_zwicker_terhardt")
    with pytest.raises(ValueError, match="f_min"):
        bands.band_edges(8, "bark_zwicker_terhardt", f_min=100.0, f_max=50.0)
    with pytest.raises(ValueError, match="unknown scale"):
        bands.band_edges(8, "not_a_real_scale")


@pytest.mark.parametrize("bad_f_max", [float("inf"), float("nan"), float("-inf")])
def test_band_edges_rejects_non_finite_f_max(bad_f_max):
    """A non-finite `f_max` used to bypass the `0 < f_min < f_max` guard for
    `+inf` specifically (`0 < f_min < inf` is True) and produce NaN edges
    and NaN weights downstream with no exception -- reject it up front."""
    with pytest.raises(ValueError, match="finite"):
        bands.band_edges(8, "erb_glasberg_moore", f_max=bad_f_max, allow_extrapolation=True)


@pytest.mark.parametrize("bad_f_min", [float("nan"), float("-inf"), float("inf")])
def test_band_edges_rejects_non_finite_f_min(bad_f_min):
    with pytest.raises(ValueError, match="finite"):
        bands.band_edges(8, "erb_glasberg_moore", f_min=bad_f_min, f_max=20000.0)


def test_bin_band_weights_rejects_non_finite_bands_dict():
    """`bin_band_weights` takes a `bands` dict as documented input, but
    nothing stops a caller from constructing one directly (bypassing
    `band_edges`'s own guard) with a non-finite f_max/centers_hz -- it must
    reject that too, rather than silently produce NaN weights."""
    degenerate = {
        "n_bands": 2,
        "f_min": 20.0,
        "f_max": float("inf"),
        "centers_hz": np.array([100.0, 500.0]),
    }
    with pytest.raises(ValueError, match="finite"):
        bands.bin_band_weights(degenerate, n_fft=1024, sr=48000)

    degenerate_centers = {
        "n_bands": 2,
        "f_min": 20.0,
        "f_max": 5000.0,
        "centers_hz": np.array([100.0, float("nan")]),
    }
    with pytest.raises(ValueError, match="finite"):
        bands.bin_band_weights(degenerate_centers, n_fft=1024, sr=48000)


# --- 6. The >15.5 kHz Bark extrapolation gate ---


@pytest.mark.parametrize("scale", bands.BARK_SCALES)
def test_bark_extrapolation_above_15500hz_is_gated(scale):
    with pytest.raises(ValueError, match="extrapolation"):
        bands.band_edges(32, scale, f_min=20.0, f_max=20000.0)

    result = bands.band_edges(32, scale, f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    assert result["extrapolated"] is True
    assert result["edges_hz"][-1] == pytest.approx(20000.0, abs=1e-6)


def test_bark_extrapolation_not_triggered_within_table_limit():
    result = bands.band_edges(32, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    assert result["extrapolated"] is False


def test_bark_tabulated_limit_hz_is_pinned():
    """Regression pin for `_BARK_TABULATED_LIMIT_HZ`. Measured: mutating it
    to `16500.0` (a same-byte-length change, so it cannot be caught by
    accident via stale-bytecode reasoning -- see AGENTS.md's mutation-sweep
    method) leaves all 676 existing tests green, because every existing
    `f_max` used above (15500.0, at-or-below both values, and 20000.0,
    safely above both) sits on the same side of the real limit and the
    mutant alike. Pin the constant's own value directly -- the same
    pattern that closed the bracket-seed gap above -- since a derived
    expectation cannot police the constant it derives from."""
    assert bands._BARK_TABULATED_LIMIT_HZ == 15500.0, (
        "the Bark extrapolation gate's tabulated limit changed; if this is intentional, "
        "update this pin and the docstrings/messages that cite 15500 Hz / 24 Bark"
    )


@pytest.mark.parametrize("scale", bands.BARK_SCALES)
def test_bark_extrapolation_gate_brackets_15500hz_tightly(scale):
    """Behavioural companion to the pin above: bracket the gate tightly
    enough that a 1000 Hz move of the threshold (in either direction)
    breaks one side. Fixed literals here, not read from the live
    constant -- deriving the bracket from `bands._BARK_TABULATED_LIMIT_HZ`
    would move in lockstep with a mutation to it and catch nothing, the
    same failure mode the pin above documents for the bracket-seed gap."""
    result = bands.band_edges(8, scale, f_min=20.0, f_max=15499.0)
    assert result["extrapolated"] is False, f"{scale}: expected NOT extrapolated at 15499 Hz"

    with pytest.raises(ValueError, match="extrapolation"):
        bands.band_edges(8, scale, f_min=20.0, f_max=15501.0)


def test_erb_has_no_extrapolation_gate_above_15500hz():
    """The task's citation names no tabulated limit for ERB -- confirm the
    ERB path is genuinely ungated, not merely untested."""
    result = bands.band_edges(32, "erb_glasberg_moore", f_min=20.0, f_max=20000.0)
    assert result["extrapolated"] is False


def test_bark_zwicker_terhardt_extrapolation_reaches_f_max_above_50khz():
    """Previously-broken case: `bark_zwicker_terhardt_to_hz` used to bisect
    on a fixed `[0, 50_000]` Hz bracket, so any `f_max` above 50 kHz was
    silently capped at ~50 kHz instead of reached -- no exception. 96 kHz is
    the Nyquist of a 192 kHz transfer, a real extrapolation request this
    module must support once `allow_extrapolation=True` is given."""
    result = bands.band_edges(32, "bark_zwicker_terhardt", f_max=96000.0, allow_extrapolation=True)
    assert result["edges_hz"][-1] == pytest.approx(96000.0, abs=1e-4)
    assert result["extrapolated"] is True


def test_bark_zwicker_terhardt_no_collapsed_edges_at_high_band_count_above_50khz():
    """At 256 bands and f_max=96 kHz, the fixed-50kHz-bracket bug collapsed
    the top three edges together (all clamped to ~50 kHz) instead of
    spacing them out to 96 kHz. Edges must stay strictly increasing (hence
    all distinct) for every band count, not just modest ones."""
    result = bands.band_edges(256, "bark_zwicker_terhardt", f_max=96000.0, allow_extrapolation=True)
    edges = result["edges_hz"]
    assert edges.shape == (257,)
    assert np.all(np.diff(edges) > 0), "edges must be strictly increasing -- no collapsed/duplicate edges"
    assert len(np.unique(edges)) == 257
    assert edges[-1] == pytest.approx(96000.0, abs=1e-4)


def test_bark_zwicker_terhardt_to_hz_rejects_z_at_or_beyond_asymptote():
    """The formula saturates at 13*pi/2 + 3.5*pi/2 (~25.918 Bark) as
    f -> infinity; no finite Hz value maps to a Bark value at or beyond
    that ceiling, so it must be rejected rather than returned as some
    arbitrarily large (and practically meaningless) Hz value."""
    asymptote = 13.0 * (np.pi / 2.0) + 3.5 * (np.pi / 2.0)
    with pytest.raises(ValueError, match="asymptote"):
        bands.bark_zwicker_terhardt_to_hz(np.array(asymptote))
    with pytest.raises(ValueError, match="asymptote"):
        bands.bark_zwicker_terhardt_to_hz(np.array(30.0))


def test_bark_zwicker_terhardt_to_hz_rejects_non_finite_z():
    """Previously-broken case: `nan >= asymptote` is False, so a NaN `z`
    silently passed the asymptote guard, then made both bracket checks
    (`too_low`, `too_high`) false on every iteration -- so bisection never
    narrowed and `lo` stayed pinned at the geometric-expansion loop's last
    `hi` value, converging on whatever the 50 kHz seed had grown to (a
    specific, wrong, plausible-looking frequency) instead of raising. `+inf`
    hit the asymptote guard by luck (`inf >= asymptote` is True) but for the
    wrong reason -- confirm both are now rejected up front, explicitly,
    scalar and embedded in an array."""
    with pytest.raises(ValueError, match="finite"):
        bands.bark_zwicker_terhardt_to_hz(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        bands.bark_zwicker_terhardt_to_hz(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        bands.bark_zwicker_terhardt_to_hz(float("-inf"))
    with pytest.raises(ValueError, match="finite"):
        bands.bark_zwicker_terhardt_to_hz(np.array([5.0, float("nan"), 15.0]))
    # Same guard reachable through the hz_to_bark/bark_to_hz dispatch layer.
    with pytest.raises(ValueError, match="finite"):
        bands.bark_to_hz(float("nan"), "bark_zwicker_terhardt")
    # bark_peaq's closed-form inverse legitimately propagates NaN -> NaN
    # (no bisection involved) -- must NOT be broken by this guard, since it
    # lives only in bark_zwicker_terhardt_to_hz.
    assert np.isnan(bands.bark_to_hz(float("nan"), "bark_peaq"))


def test_bark_zwicker_terhardt_to_hz_asymptote_message_has_no_nan_after_finiteness_fix():
    """Previously: an array containing both a NaN and a genuinely-too-large
    z made `np.max` propagate NaN into the asymptote error message (`z=nan
    Bark is at or beyond ...`), hiding the real out-of-range value. Now the
    finiteness guard fires first and unconditionally, so the asymptote
    message can never contain a NaN placeholder for a mixed array."""
    with pytest.raises(ValueError, match="finite") as excinfo:
        bands.bark_zwicker_terhardt_to_hz(np.array([5.0, float("nan"), 30.0]))
    assert "nan Bark" not in str(excinfo.value)


# Golden boundary value for the CURRENT production seed (50 kHz) and its
# measured 46-doublings/47th-iteration-break worst case -- see the
# mutation-proof transcript in this PR's report for how this figure was
# obtained by mutating the PRODUCTION loop's cap (200 -> 45/46/47), not by
# reimplementing its arithmetic. This is a fixed literal, not derived from
# `bands._BARK_BRACKET_SEED_HZ` at test time: deriving it from the live
# constant would make the expected value move in lockstep with a seed
# mutation, which is exactly what let seeds 100_000/25_000 slip through
# undetected in the first place (see the seed pin below, which is what
# actually closes that gap) -- and it would ALSO let a seed mutation that
# happens to need the same 46 doublings (e.g. 60_000, measured) move in
# lockstep too, since both production and this literal would then agree
# on a "new" wrong answer. A fixed literal has neither failure mode.
_BARK_BOUNDARY_DOUBLINGS_AT_PRODUCTION_SEED = 46
_Z_MAX_BOUNDARY_HZ = 3.5184372088832e18


def test_bark_zwicker_terhardt_to_hz_bracket_expansion_needs_46_doublings_at_the_asymptote_boundary():
    """Regression pin for the geometric bracket-expansion loop's comment in
    `bark_zwicker_terhardt_to_hz`: for the largest representable float64
    `z` strictly below the asymptote (the worst case the function ever
    accepts -- `z >= asymptote` is rejected outright by the guard above
    this loop), expanding from the production seed needs exactly 46
    doublings before `hi` exceeds it. Because the loop CHECKS before it
    DOUBLES, observing that as a `break` takes the loop's 47th iteration,
    not its 46th.

    Measured by mutating the PRODUCTION loop's cap directly (`range(200)`
    edited in place to `range(44)`/`range(45)`/`range(46)`/`range(47)`,
    each run, then restored byte-identical): cap 44 and cap 45 both leave
    `hi` short of the true crossing point and `bark_zwicker_terhardt_to_hz`
    returns a wrong result (1.7592186044416e+18 Hz at cap 45, roughly HALF
    of the correct answer); cap 46 already matches the cap-200 answer
    exactly, and cap 47 changes nothing further. So the 47th iteration's
    check only *observes* that `hi` is already big enough and breaks -- it
    does not correct anything, and the FIRST (largest) cap that returns a
    WRONG answer is 45, not 46. An earlier version of this docstring
    wrongly claimed a cap of 46 "would leave the loop silently exhausted
    at the wrong `hi`"; it does not -- only caps of 45 or below do. The
    shipped production comment itself does not make that claim (it only
    says the 47th check would not run under a cap of 46, which is true)
    and needs no change.

    `_Z_MAX_BOUNDARY_HZ` above is `50_000 * 2**46` -- but it is ALSO
    `100_000 * 2**45` and `25_000 * 2**47`, so on its own it cannot tell a
    production seed change apart from the seed staying put: a seed
    mutation to `100_000` (needing only 45 doublings) or `25_000` (needing
    47) reaches this exact same float64 value and the boundary assertion
    below would not notice (measured). The seed pin immediately below
    closes that gap directly, by pinning the seed's OWN value rather than
    trying to detect its effect through the boundary arithmetic -- which
    also covers the seed values (e.g. `60_000`, measured) that happen to
    need the SAME 46 doublings as production: those change the boundary
    value (`60_000 * 2**46 != _Z_MAX_BOUNDARY_HZ`) but would just as
    happily satisfy a boundary check re-derived from the mutated seed, so
    only a direct pin on the seed's value is proof against every case.

    The primary assertion below calls the REAL production function, not a
    copy of its arithmetic, so a regression that silently shrinks the
    bracket-expansion cap (e.g. `range(200)` to `range(45)`) is caught
    here even though it still returns a finite, positive,
    plausible-looking number. The geometric-expansion-only loop further
    below is documentation, not the test's teeth: it independently pins
    the 46-doublings/47th-iteration-break figures this docstring and the
    production comment both cite, starting from the production seed
    (read, not copied) so a change to the seed, the formula, or the
    asymptote is caught by it too.
    """
    # Pin the seed's own value FIRST: a change here is exactly finding 1's
    # failure mode (a production seed change with a golden boundary value
    # that cannot detect it), and this catches every variant of it --
    # including the ones (like 60_000) that still need 46 doublings and so
    # would otherwise satisfy any check re-derived from the live constant.
    assert bands._BARK_BRACKET_SEED_HZ == 50_000.0, (
        "bark_zwicker_terhardt_to_hz's bracket-expansion seed changed; if this is "
        "intentional, update this pin, _Z_MAX_BOUNDARY_HZ, and their comments together"
    )

    asymptote = 13.0 * (np.pi / 2.0) + 3.5 * (np.pi / 2.0)
    z_max = np.nextafter(asymptote, -np.inf)

    # The real product call -- this is the test's teeth. A cap-45 mutant
    # returns 1.7592186044416e+18 here, which fails this assertion outright.
    result = bands.bark_zwicker_terhardt_to_hz(z_max)
    assert result == pytest.approx(_Z_MAX_BOUNDARY_HZ, rel=1e-9)

    # Documentation-only regression pin for the 46/47 figures the comment
    # in bands.py and this docstring both cite -- NOT the test's teeth
    # (see AGENTS.md #3b: a test that only reimplements the thing it
    # checks cannot detect a change in the thing). Starts from the
    # PRODUCTION seed (read, not a copied literal) so this loop tracks
    # whatever the seed pin above already confirmed it to be.
    hi = bands._BARK_BRACKET_SEED_HZ
    doublings = 0
    break_iteration = None
    for i in range(200):
        too_low = bool(bands.hz_to_bark_zwicker_terhardt(np.array(hi)) < z_max)
        if not too_low:
            break_iteration = i + 1  # 1-indexed, matches "loop iteration" in the comment
            break
        hi *= 2.0
        doublings += 1

    assert doublings == _BARK_BOUNDARY_DOUBLINGS_AT_PRODUCTION_SEED
    assert break_iteration == _BARK_BOUNDARY_DOUBLINGS_AT_PRODUCTION_SEED + 1
    # Real headroom check against the PRODUCTION cap (read it, not a copy
    # of the literal 200): a cap-47 mutant (zero headroom over the
    # measured 47-iteration worst case) makes this fail (47 < 47 is
    # False) even though the function still returns the correct answer.
    assert break_iteration < bands._BARK_BRACKET_EXPANSION_CAP, (
        "the production expansion cap must retain real headroom over the measured worst case"
    )


# --- 6b. The default Bark variant (hz_to_bark/bark_to_hz's dispatch default) is pinned ---


def test_default_bark_variant_is_pinned():
    """Regression pin for `_DEFAULT_BARK_VARIANT`: measured, mutating it to
    `"bark_zwicker_terhardt"` leaves all 676 existing tests green, because
    every existing round-trip/table-agreement test states its `variant`
    explicitly (see section 1/2 above) -- nothing exercises the unqualified
    default. That default is what every caller of `hz_to_bark`/`bark_to_hz`
    without an explicit `variant=` actually gets, and the module docstring's
    "THE SCALE IS NOT COSMETIC" section is explicit that PEAQ's ITU-R
    BS.1387 spreading-function constants (S_l = 27 dB/Bark, S_u = -24 -
    230/f_c + 0.2L) are calibrated ON the PEAQ Bark scale and must not be
    paired with Zwicker & Terhardt bands -- pin the constant's own value
    directly, the same pattern used for the bracket seed and the
    tabulated limit above."""
    assert bands._DEFAULT_BARK_VARIANT == "bark_peaq", (
        "the default Bark variant changed; if this is intentional, update this pin, "
        "the callers that rely on an unqualified hz_to_bark/bark_to_hz being PEAQ, and "
        'the module docstring\'s "THE SCALE IS NOT COSMETIC" section'
    )


def test_unqualified_hz_to_bark_matches_peaq_not_zwicker_terhardt():
    """Behavioural companion to the pin above: an unqualified `hz_to_bark`
    call (no `variant=` stated -- the one call site the module docstring
    warns is exactly where a silently-swapped default would defeat the
    required-`scale` protection on `band_edges`) must agree with the PEAQ
    realization and disagree with Zwicker & Terhardt, at a frequency where
    the two diverge unambiguously. Measured ~4.26 Bark apart at 20000 Hz
    (PEAQ is a smooth closed-form approximation that diverges from the
    classical-table-matching Zwicker & Terhardt formula most sharply at
    the top of the audible range -- see section 2's table-agreement
    tests)."""
    f = 20000.0
    default = float(bands.hz_to_bark(f))
    peaq = float(bands.hz_to_bark(f, variant="bark_peaq"))
    zwicker = float(bands.hz_to_bark(f, variant="bark_zwicker_terhardt"))

    assert abs(peaq - zwicker) > 4.0, "expected the two variants to diverge unambiguously at 20 kHz"
    assert default == pytest.approx(peaq)
    assert default != pytest.approx(zwicker, abs=1.0)


# --- 7. Bin -> band weights: partition of unity, asserted directly ---


def _assert_partition_of_unity(weights: np.ndarray, atol: float = 1e-10):
    column_sums = weights.sum(axis=0)
    max_dev = float(np.max(np.abs(column_sums - 1.0)))
    print(f"\n[bands] partition-of-unity max deviation from 1.0 = {max_dev:.3e}")
    assert np.allclose(column_sums, 1.0, atol=atol)


@pytest.mark.parametrize("n_bands", [1, 2, 32, 256])
def test_partition_of_unity_sums_to_one_everywhere(n_bands):
    sr = 48000
    n_fft = 2048
    result = bands.band_edges(n_bands, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)
    assert weights.shape == (n_bands, n_fft // 2 + 1)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    _assert_partition_of_unity(weights)


def test_partition_of_unity_holds_below_f_min_and_above_f_max():
    """Bins below f_min or above f_max are NOT clamped by special-case code
    (see module docstring) -- they fall into the flattened end-bands'
    constant-1 region automatically. Confirm bin 0 (0 Hz, below f_min=20)
    goes entirely to band 0, and the top bins (above f_max=8000) go
    entirely to the last band."""
    sr = 48000
    n_fft = 2048
    result = bands.band_edges(16, "bark_zwicker_terhardt", f_min=20.0, f_max=8000.0)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)
    bin_hz = np.arange(n_fft // 2 + 1) * sr / n_fft

    below = bin_hz < 20.0
    above = bin_hz > 8000.0
    assert np.any(below), "test setup should actually exercise the below-f_min tail"
    assert np.any(above), "test setup should actually exercise the above-f_max tail"
    assert np.allclose(weights[0, below], 1.0)
    assert np.allclose(weights[1:, below], 0.0)
    assert np.allclose(weights[-1, above], 1.0)
    assert np.allclose(weights[:-1, above], 0.0)
    _assert_partition_of_unity(weights)


def test_nyquist_below_top_band_edge():
    """A sample rate whose Nyquist falls below the requested f_max: the top
    band(s) simply never receive a bin (no bin reaches that high), which is
    legitimate, documented behaviour -- not a crash, and partition of unity
    still holds for every bin that DOES exist."""
    sr = 8000  # Nyquist = 4000 Hz
    n_fft = 512
    result = bands.band_edges(16, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0, allow_extrapolation=False)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)
    n_bins = n_fft // 2 + 1
    assert weights.shape == (16, n_bins)
    bin_hz = np.arange(n_bins) * sr / n_fft
    assert bin_hz[-1] < result["f_max"], "test setup should actually put Nyquist below f_max"
    _assert_partition_of_unity(weights)
    # The very top bands (centered well above Nyquist) get zero weight from
    # every bin that actually exists -- legitimate, not an error.
    assert np.allclose(weights[-1, :], 0.0)


@pytest.mark.parametrize(
    ("n_fft", "sr"),
    [(2048, 48000), (512, 44100), (4096, 96000), (1023, 48000)],
)
def test_bin_hz_matches_k_times_sr_over_n_fft_exactly(n_fft, sr):
    """Assert the bin frequency grid directly against `k * sr / n_fft`,
    exact element-wise comparison -- not inferred indirectly from where
    energy happens to land in one band at one configuration.

    `test_bin_frequencies_use_sr_over_n_fft_not_the_noisereduce_bug` (below)
    only fails once the bin-spacing error is large (~2-5%): it was measured
    to pass unchanged against BOTH `(np.arange(n_bins) + 1) * sr / n_fft`
    (an off-by-one bin index) and `np.arange(n_bins) * sr / (n_fft - 1)` (an
    off-by-one denominator) -- see this PR's mutation-proof transcripts.
    This test asserts the grid itself, so it catches both directly.

    `(1023, 48000)` is the ODD `n_fft` case: a fourth mutant,
    `np.linspace(0.0, sr / 2.0, n_bins)`, is exact for every EVEN `n_fft`
    above (the top bin genuinely sits at Nyquist when `n_fft` is even, so
    `linspace`'s endpoint-anchored spacing happens to coincide with
    `k * sr / n_fft`) and only diverges once `n_fft` is odd -- the last bin
    then sits at `((n_fft-1)/2) * sr / n_fft`, strictly below `sr / 2`. At
    `n_fft=1023`, `sr=48000` the true last-bin frequency is
    23976.539589... Hz, not the mutant's 24000.0 Hz -- see this PR's fourth
    mutation transcript, and
    `test_bin_band_weights_odd_n_fft_energy_placement_independent_of_grid_helper`
    below for the same fix asserted through `bin_band_weights`/`band_energy`
    rather than the grid helper in isolation.
    """
    n_bins = n_fft // 2 + 1
    bin_hz = bands._bin_frequencies_hz(n_fft, sr)
    k = np.arange(n_bins, dtype=np.float64)
    expected = k * sr / n_fft

    assert bin_hz.shape == (n_bins,)
    assert np.array_equal(bin_hz, expected)
    assert bin_hz[0] == 0.0, "bin 0 must be exactly DC"
    assert bin_hz[-1] == (n_bins - 1) * sr / n_fft


def test_bin_band_weights_odd_n_fft_energy_placement_independent_of_grid_helper():
    """Second, independent check that the odd-`n_fft` grid case above
    actually propagates into `bin_band_weights`/`band_energy` -- not just
    that `_bin_frequencies_hz` returns the right array in isolation.

    At `n_fft=1023` (odd), `sr=48000`, bin 511's TRUE frequency is
    23976.539589... Hz; the `np.linspace(0.0, sr / 2.0, n_bins)` mutant
    would instead compute it as `sr / 2 == 24000.0` Hz, a ~23.46 Hz gap.
    Two hand-built bands straddle exactly that gap: centers at 23800 Hz and
    23990 Hz. Under the TRUE frequency, bin 511 falls inside the interior
    ramp between the two centers, splitting its energy between both bands;
    under the mutant's 24000.0 Hz, bin 511 sits past the second center and
    clips to a full 1.0/0.0 split instead. The two outcomes are not close
    to each other, so a regression in this integration -- not just in the
    grid helper -- is caught here.
    """
    sr = 48000
    n_fft = 1023
    n_bins = n_fft // 2 + 1
    k = n_bins - 1
    assert (n_bins, k) == (512, 511)

    true_freq = k * sr / n_fft
    mutant_freq = sr / 2.0
    assert true_freq == pytest.approx(23976.539589, abs=1e-5)
    assert mutant_freq - true_freq == pytest.approx(23.460411, abs=1e-5)

    lo, peak = 23800.0, 23990.0
    hand_built_bands = {
        "n_bands": 2,
        "f_min": 20000.0,
        "f_max": 24010.0,
        "centers_hz": np.array([lo, peak]),
    }
    weights = bands.bin_band_weights(hand_built_bands, n_fft=n_fft, sr=sr)
    assert weights.shape == (2, n_bins)

    # Computed from the true frequency directly (not copied from the
    # implementation) -- what the interior ramp split must equal.
    expected_band1 = (true_freq - lo) / (peak - lo)
    expected_band0 = (peak - true_freq) / (peak - lo)
    assert weights[1, k] == pytest.approx(expected_band1, abs=1e-9)
    assert weights[0, k] == pytest.approx(expected_band0, abs=1e-9)
    assert 0.0 < weights[0, k] < 1.0, "true grid must split bin 511 across both bands"
    assert 0.0 < weights[1, k] < 1.0

    # If the grid were instead sr/2-linspace-based, bin 511 would compute
    # as `mutant_freq` (24000.0), past band 1's center -- clipped to a full
    # 1.0/0.0 split, not the partial split just asserted above.
    mutant_rising = (mutant_freq - lo) / (peak - lo)
    assert mutant_rising > 1.0, "test setup should put the mutant frequency past band 1's center"
    assert weights[1, k] != pytest.approx(1.0, abs=1e-6)
    assert weights[0, k] != pytest.approx(0.0, abs=1e-6)

    _assert_partition_of_unity(weights)

    spectrum = np.zeros(n_bins)
    spectrum[k] = 1.0
    per_band = bands.band_energy(spectrum, weights)
    print(f"\n[bands] odd n_fft={n_fft}: bin {k} true freq={true_freq:.6f} Hz, per-band energy={per_band}")
    assert per_band[1] == pytest.approx(expected_band1, abs=1e-9)
    assert per_band[0] == pytest.approx(expected_band0, abs=1e-9)


def test_bin_frequencies_use_sr_over_n_fft_not_the_noisereduce_bug():
    """Guard against the specific documented mistake: `noisereduce` computes
    bin spacing as `sr/(n_fft/2)`, which is TWICE the true spacing
    `sr/n_fft`. This exercises `bin_band_weights`/`band_energy` directly --
    it does not merely recompute the two candidate spacing constants and
    compare them to each other, because that tests nothing about the
    implementation (a doubled bin spacing inside `bin_band_weights` would
    still leave both constants intact and this test green).

    Construction: bin `k`'s TRUE frequency (`k * sr / n_fft`) sits deep in
    band 8's triangular peak (weight ~0.998); bin `2k`'s TRUE frequency sits
    above `f_max`, deep in band 9's flat end region (weight exactly 1.0). If
    `bin_band_weights` used the `noisereduce` spacing `sr/(n_fft/2)`, bin
    `k`'s computed frequency would be `2 * (k * sr / n_fft)` -- exactly bin
    `2k`'s TRUE frequency -- so a spectrum with all its energy in bin `k`
    would be placed almost entirely into band 9 instead of band 8. Verified
    (see PR mutation-proof transcript): forcing that exact spacing bug into
    `bin_band_weights` makes this test fail with `per_band[8] == 0.0` and
    `per_band[9] == 1.0`, while it passes against the real implementation.
    """
    sr = 48000
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    assert n_bins == 1025
    bin_hz_spacing = sr / n_fft  # 23.4375 Hz, matches stft_properties
    wrong_bin_hz_spacing = sr / (n_fft / 2)  # 46.875 Hz -- the noisereduce bug
    assert bin_hz_spacing == pytest.approx(23.4375)
    assert wrong_bin_hz_spacing == pytest.approx(46.875)

    n_bands = 10
    result = bands.band_edges(n_bands, "erb_glasberg_moore", f_min=20.0, f_max=20000.0)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)

    k = 437
    k_doubled = 2 * k
    assert k_doubled < n_bins, "test setup should keep bin 2k in range"
    true_freq_k = k * bin_hz_spacing
    true_freq_k_doubled = k_doubled * bin_hz_spacing
    # What the noisereduce-bug spacing would compute for bin k -- exactly
    # bin 2k's true frequency, by construction (2 * k * sr/n_fft == k * sr/(n_fft/2)).
    buggy_freq_k = k * wrong_bin_hz_spacing
    assert buggy_freq_k == pytest.approx(true_freq_k_doubled)
    print(
        f"\n[bands] bin {k} true freq = {true_freq_k:.4f} Hz "
        f"(buggy spacing would compute {buggy_freq_k:.4f} Hz = bin {k_doubled}'s true freq)"
    )

    spectrum = np.zeros(n_bins)
    spectrum[k] = 1.0
    per_band = bands.band_energy(spectrum, weights)
    print(f"\n[bands] per-band energy for a spectrum concentrated in bin {k}: {per_band}")

    expected_band = 8  # where bin k's TRUE frequency actually falls
    wrong_band = 9  # where the noisereduce-bug spacing would misplace it
    assert expected_band != wrong_band
    assert per_band[expected_band] == pytest.approx(1.0, abs=0.05), (
        f"bin {k} (true freq {true_freq_k:.4f} Hz) should land almost entirely in band "
        f"{expected_band} (center {result['centers_hz'][expected_band]:.2f} Hz) -- a bin-spacing "
        "error would move this energy to the wrong band and this assertion would catch it"
    )
    assert per_band[wrong_band] == pytest.approx(0.0, abs=0.05), (
        f"bin {k} must NOT be misplaced into band {wrong_band}; that is exactly where the "
        f"sr/(n_fft/2) bug would put it, since it would compute bin {k}'s frequency as "
        f"{buggy_freq_k:.4f} Hz (bin {k_doubled}'s true frequency, which lands in band {wrong_band})"
    )


def test_bin_band_weights_accepts_fractional_sr_without_truncating_it():
    """`bin_band_weights`'s `sr` is annotated `int`, but its runtime check
    has always only required `np.isfinite(sr) and sr > 0` -- it silently
    accepts a fractional sample rate (a resampled or measured rate, e.g.
    48000.5 Hz) and does real, `sr`-dependent arithmetic with it
    (`_bin_frequencies_hz`: `k * sr / n_fft`). A mutant that coerces `sr`
    with `int(sr)` before that arithmetic is invisible to every other test
    in this file, because every other test's `sr` is already a whole
    number -- this is the one place a genuinely fractional `sr` is
    exercised through `bin_band_weights` itself (not just
    `_bin_frequencies_hz` in isolation, as
    `test_bin_hz_matches_k_times_sr_over_n_fft_exactly` above does).

    Construction: at `n_fft=2048`, the top bin (`k=1024`) sits at
    `1024 * sr / 2048 == sr / 2`. At `sr=48000.5` that is 24000.25 Hz;
    truncating to `int(sr)=48000` would instead place it at exactly
    24000.0 Hz -- a 0.25 Hz shift. Two hand-built band centers (23990.0,
    24010.0) straddle BOTH candidate frequencies inside the same interior
    ramp, so the two hypotheses predict two different, non-close weights
    (0.5125 vs 0.5 for band 1) rather than the same saturated 0.0/1.0 the
    "noisereduce"-bug test above needed -- computed independently from the
    documented triangular-ramp formula, not copied from the implementation.
    """
    sr = 48000.5
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    k = n_bins - 1
    assert (n_bins, k) == (1025, 1024)

    true_freq = k * sr / n_fft
    truncated_freq = k * int(sr) / n_fft
    assert true_freq == pytest.approx(24000.25, abs=1e-9)
    assert truncated_freq == pytest.approx(24000.0, abs=1e-9)
    assert true_freq != pytest.approx(truncated_freq, abs=1e-6)

    c0, c1 = 23990.0, 24010.0
    hand_built_bands = {
        "n_bands": 2,
        "f_min": 23980.0,
        "f_max": 24020.0,
        "centers_hz": np.array([c0, c1]),
    }
    weights = bands.bin_band_weights(hand_built_bands, n_fft=n_fft, sr=sr)
    assert weights.shape == (2, n_bins)

    # Computed from the true fractional frequency directly (not copied
    # from the implementation) -- what the interior ramp split must equal.
    expected_band1 = (true_freq - c0) / (c1 - c0)
    expected_band0 = (c1 - true_freq) / (c1 - c0)
    assert weights[1, k] == pytest.approx(expected_band1, abs=1e-9)
    assert weights[0, k] == pytest.approx(expected_band0, abs=1e-9)

    # If `sr` were coerced with `int(sr)` internally, bin k would land at
    # `truncated_freq` instead, giving a measurably different, non-close split.
    mutant_band1 = (truncated_freq - c0) / (c1 - c0)
    assert mutant_band1 == pytest.approx(0.5, abs=1e-9), "test setup sanity check"
    assert weights[1, k] != pytest.approx(mutant_band1, abs=1e-6)
    assert weights[0, k] != pytest.approx(1.0 - mutant_band1, abs=1e-6)

    _assert_partition_of_unity(weights)


def test_bin_band_weights_rejects_degenerate_band_collision():
    """Two centers that collide (or are out of order) must raise -- not
    silently divide by zero. This directly constructs a degenerate `bands`
    dict (a repeated center) rather than hunting for an `n_bands`/range
    combination that happens to collide at float64 precision -- the guard
    in `bin_band_weights` is oblivious to how its input was produced, so
    this exercises it precisely without depending on incidental floating
    point behaviour at some particular scale/frequency choice."""
    degenerate_bands = {
        "scale": "bark_peaq",
        "n_bands": 3,
        "f_min": 100.0,
        "f_max": 5000.0,
        "edges_hz": np.array([100.0, 1000.0, 1000.0, 5000.0]),
        "centers_hz": np.array([500.0, 1000.0, 1000.0]),  # repeated center
        "extrapolated": False,
    }
    with pytest.raises(ValueError, match="degenerate"):
        bands.bin_band_weights(degenerate_bands, n_fft=2048, sr=48000)


def test_bin_band_weights_rejects_n_bands_centers_mismatch():
    """Previously: a hand-built `bands` dict with `n_bands` not matching
    `len(centers_hz)` raised no exception at all when there were TOO MANY
    centers (it silently returned `(n_bands, n_bins)` weights for the wrong
    mapping -- partition-of-unity still held exactly, so it looked healthy
    while being wrong), and raised the wrong exception type (`IndexError`,
    not the documented `ValueError`) when there were too FEW."""
    too_many_centers = {
        "n_bands": 3,
        "f_min": 20.0,
        "f_max": 20000.0,
        "centers_hz": np.linspace(100.0, 15000.0, 10),
    }
    with pytest.raises(ValueError, match="n_bands"):
        bands.bin_band_weights(too_many_centers, n_fft=1024, sr=48000)

    too_few_centers = {
        "n_bands": 5,
        "f_min": 20.0,
        "f_max": 20000.0,
        "centers_hz": np.array([100.0, 500.0]),
    }
    with pytest.raises(ValueError, match="n_bands"):
        bands.bin_band_weights(too_few_centers, n_fft=1024, sr=48000)


@pytest.mark.parametrize("bad_n_fft", [0, -2, -1024])
def test_bin_band_weights_rejects_bad_n_fft(bad_n_fft):
    """Previously: `n_fft=0` produced an all-NaN weights array behind a
    bare `RuntimeWarning` (divide by zero), and a negative `n_fft` produced
    a silent, empty `(n_bands, 0)` array -- neither raised."""
    result = bands.band_edges(8, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    with pytest.raises(ValueError, match="n_fft"):
        bands.bin_band_weights(result, n_fft=bad_n_fft, sr=48000)


@pytest.mark.parametrize("bad_sr", [0, -48000, float("nan"), float("inf")])
def test_bin_band_weights_rejects_bad_sr(bad_sr):
    """Previously: `sr=0` and a negative `sr` were both silently accepted
    (bin_hz collapses to all-zero, or runs backwards -- neither is a valid
    sample rate)."""
    result = bands.band_edges(8, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    with pytest.raises(ValueError, match="sr"):
        bands.bin_band_weights(result, n_fft=1024, sr=bad_sr)


# --- 8. Bin -> band energy summation: conserves total energy ---


def test_band_energy_conserves_total_energy_white_noise():
    sr = 48000
    n_fft = 2048
    rng = np.random.default_rng(0)
    n_bins = n_fft // 2 + 1
    n_frames = 20
    # A real, non-negative "power" array -- exactly what a caller would
    # pass after |STFT|**2, but generated directly here since this function
    # only cares that it is real and non-negative.
    power = rng.uniform(0.0, 1.0, size=(n_bins, n_frames)) ** 2

    result = bands.band_edges(32, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)
    per_band = bands.band_energy(power, weights)

    assert per_band.shape == (32, n_frames)
    total_before = power.sum(axis=0)
    total_after = per_band.sum(axis=0)
    max_rel_err = float(np.max(np.abs(total_after - total_before) / np.maximum(total_before, 1e-12)))
    print(f"\n[bands] band-energy conservation max relative err = {max_rel_err:.3e}")
    assert np.allclose(total_after, total_before, rtol=1e-9, atol=1e-9)


def test_band_energy_conserves_total_energy_single_frame_1d():
    sr = 48000
    n_fft = 1024
    n_bins = n_fft // 2 + 1
    rng = np.random.default_rng(1)
    power = rng.uniform(0.0, 2.0, size=(n_bins,))

    result = bands.band_edges(24, "erb_glasberg_moore", f_min=20.0, f_max=20000.0)
    weights = bands.bin_band_weights(result, n_fft=n_fft, sr=sr)
    per_band = bands.band_energy(power, weights)

    assert per_band.shape == (24,)
    assert per_band.sum() == pytest.approx(power.sum(), rel=1e-9)


def test_band_energy_rejects_mismatched_bin_axis():
    result = bands.band_edges(8, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    weights = bands.bin_band_weights(result, n_fft=2048, sr=48000)
    wrong_spectrum = np.ones(weights.shape[1] + 1)
    with pytest.raises(ValueError, match="bin axis"):
        bands.band_energy(wrong_spectrum, weights)


# --- 9. Determinism ---


def test_band_edges_and_weights_are_deterministic():
    r1 = bands.band_edges(32, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    r2 = bands.band_edges(32, "bark_zwicker_terhardt", f_min=20.0, f_max=15500.0)
    assert np.array_equal(r1["edges_hz"], r2["edges_hz"])
    assert np.array_equal(r1["centers_hz"], r2["centers_hz"])

    w1 = bands.bin_band_weights(r1, n_fft=2048, sr=48000)
    w2 = bands.bin_band_weights(r2, n_fft=2048, sr=48000)
    assert np.array_equal(w1, w2)
