# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The plan document has its
own version, independent of the package version — see [contracts/plan.v1.md](contracts/plan.v1.md)
and [contracts/regions.v1.md](contracts/regions.v1.md).

## [Unreleased]

### Added

- **`aud.dsp.stft`** -- STFT analysis / weighted-overlap-add (WOLA) resynthesis: the transform
  spine for future spectral processing (masking, denoising, de-essing-by-band, ...), library-only
  in this step -- no CLI verb, no plan stage, no engine handler. Uses `scipy.signal.ShortTimeFFT`
  (the legacy `scipy.signal.stft`/`istft` trio was marked legacy in scipy 1.12); `resynthesize`
  always supplies its own explicit `dual_win` -- the analysis window scaled by the measured
  constant-overlap-add sum of `window**2` -- rather than `ShortTimeFFT`'s own automatically-solved
  canonical dual, because the automatic dual is invertible for almost any window and would hide
  exactly the WOLA failure mode this module needs to surface (a window/hop pair that reconstructs
  fine alone but fails badly as its own WOLA pair -- e.g. Hann x Hann at 50% hop). `scipy` floor
  raised `>=1.14` -> `>=1.15`: `ShortTimeFFT.istft` had two correctness bugs fixed in that release.
  A null (unity-gain) round trip reconstructs to roughly -307 to -309 dB relative to signal peak
  (float64 machine-epsilon floor is ~-313 dB); `stft_properties`/`check_cola_nola` report
  COLA/NOLA compliance programmatically rather than assuming it from a table -- verified against
  known-failing cases (a symmetric, non-periodic Hann window; Hann x Hann at 50% hop).
- **`aud.dsp.bands`** -- perceptual band mapping (Hz<->Bark/ERB conversions, band-edge
  construction, bin->band energy summation), library-only in this step -- no CLI verb, no plan
  stage, no engine handler. Two Bark realizations (PEAQ/ITU-R BS.1387, closed form both ways;
  Zwicker & Terhardt 1980, bisected inverse) plus Glasberg & Moore 1990 ERB-rate. `band_edges`'s
  `scale` parameter has **no default** -- it is stamped into the returned band structure so a
  caller can never lose track of which scale it got, because 1 Bark is not a fixed multiple of
  1 ERB (measured here: ~2.8 ERB at 100 Hz, ~1.2 at 1 kHz, ~2.1 at 10 kHz) and the next epic
  step's dB/Bark spreading slopes depend on getting this right. Measured against the classical
  Zwicker 24-critical-band table: Zwicker & Terhardt ~0.20 Bark max error, PEAQ ~3.1 Bark (a
  smooth perceptual-model approximation, not a classical-Bark substitute). The classical table's
  15.5 kHz (24 Bark) limit is enforced as an explicit `allow_extrapolation` gate rather than
  silently extrapolated. `bin_band_weights` builds triangular partition-of-unity filters
  (flattened-end construction, so `sum_b w_b(k) == 1` holds for every bin, in-range or not) in
  the style RNNoise/DeepFilterNet use (approach only, no code copied); `band_energy` then
  conserves total energy exactly as a consequence. Hardened after adversarial review: a
  non-finite `z` (NaN/inf) passed to `bark_zwicker_terhardt_to_hz` now raises `ValueError`
  instead of silently converging on a plausible-looking frequency; a `bands` dict whose
  `n_bands` does not match `len(centers_hz)` is rejected rather than silently mapping the
  wrong number of bands; and `bin_band_weights` now validates `n_fft` (must be a positive
  integer) and `sr` (must be positive and finite) instead of accepting `n_fft=0`/negative
  `n_fft` (all-NaN weights behind a bare `RuntimeWarning`) or `sr<=0` (a silently wrong bin
  grid).
- **Removed the Traunmuller 1990 Bark variant (`bark_traunmuller` / `hz_to_bark_traunmuller` /
  `bark_traunmuller_to_hz`) before it ever shipped a release or gained a caller.** Its main
  rational-approximation expression (26.81, 1960, 0.53) and inverse constant (26.28) were
  confirmed against Traunmuller's own Stockholm University page and by symbolic algebra, but the
  four low/high-end correction constants (0.15, 0.22, and the 2 / 20.1 Bark branch thresholds)
  could not be traced to the primary source -- Traunmuller 1990, JASA 88(1):97, is paywalled, and
  the publisher, ResearchGate, Unpaywall and Semantic Scholar all refused access. Two independent
  secondary sources (Voicebox at Imperial, phonR/tidynorm) agree on the constants, but nobody has
  actually read the paper. A partially-verified formula is shipped guesswork; since nothing in
  the codebase called this variant, it was cheaper to remove the whole thing than to keep
  carrying an unverified half-version.

### Fixed

- **`aud.dsp.masking.masking_threshold`** was missing ITU-R BS.1387 Annex 2 Sec 2.1.9's eq. 24-26
  masking offset entirely: threshold-minus-excitation measured at exactly 0.00 dB at every band,
  3.0-6.75 dB too permissive (concluding material is masked when it audibly is not -- the unsafe
  direction for a ducker). Now applies `m[k] = 3.0 dB` for `k*res <= 12`, else `0.25*(k*res)`, per
  band, with a dedicated test re-deriving the formula independently of the module's own constants
  and mutation-proving both 3.0 and 0.25. Also corrected three wrong numbers in the module's own
  docstrings/error text (the exact-Bs-reference level is 0 dB SPL, not "~-92"; `Tq` measures 65.9 dB
  at 16 kHz, not ">100"; the zero-power guard is a semantic check, not a negative-exponent domain
  error), documented the measured impact of not applying outer/middle-ear weighting and internal
  noise before the level-dependent upper slope (0.2x the omitted `W[k]`, from -6.65 to +1.12 dB/Bark
  across the audible range), and recorded two deliberate, previously-undocumented decisions (no
  clamp on the upper slope's rare positive-going case above ~120 dB SPL; keep the absolute-threshold
  floor default `True`). Two of six mutations run against this module during review (deleting the
  `Bs` normalisation; a one-band index shift in the upper-slope frequency term) passed all of
  `tests/test_dsp_masking.py`'s existing assertions undetected -- an `np.allclose` call with no
  explicit `atol` against a ~6e-10-magnitude reference, and a `< 10.0` bound measured at 5.7382 for
  the correct implementation, were both loose enough to admit the mutated values (8.7281, 84%
  deviation). Fixed by asserting on the already-computed ratio deviation directly, tightening the
  bound, and adding a test against `test_dsp_masking_kabal_reference.py`'s independently-transliterated
  oracle on a non-uniform profile (the index-shift mutation is invisible to every uniform-profile or
  aggregate-slope-fit assertion in this file, by construction).
- Refresh Google's default to current stable `gemini-3.5-flash-lite` in place of retired
  `gemini-2.0-flash`. Its default minimal thinking better fits the existing 2,000-token
  advice cap. Explicit model selections, request settings and stored records are unchanged.
- `render` now re-reads the final encoded output when a plan targets loudness and exposes
  verification plus machine-readable warnings for a missed or unmeasurable target. Uses
  `verify`'s existing 0.5 LU tolerance, honors the last loudness stage, and does not infer
  a target for plans without one. No automatic compression or other DSP changes.
  Silence's loudness-stage measurements now serialize as null rather than `-Infinity`.

- **`advise`'s tonal diagnosis no longer anchors to the file's own median when a reference is
  given.** `rel_median_db` (added in 0.12.0) compares each octave band to THIS FILE's own overall
  median -- an anchor the very defect being diagnosed can move, and one dominated by the
  material's natural spectral shape rather than by any actual defect. Measured on six controlled
  induced-defect fixtures (boxy/rumbly/dull/harsh/muddy/thin): taking the largest `|rel_median_db|`
  identified the right band AND direction in only 1 of 6 cases -- for 4 of 6 it picked 31.5 Hz
  simply because that band sits naturally ~12 dB below the midrange in the test material,
  reporting the quietest band, not the defective one.
  - `dsp.analysis.reference_anchored_band_deviation_db(band_energy_db, reference_band_energy_db)`
    is a new pure function: per-band deltas against a reference, anchored by the MEDIAN OF THE
    DELTAS (never either file's own per-band median), which removes the overall level/gain
    difference between the two files while leaving genuine band-shape differences intact.
    Measured 6 of 6 correct band-and-direction on the same six fixtures.
  - `analyze`'s (`dsp.analysis` and `aud.lib`) `octave_band_analysis` entries carry a new
    `rel_reference_db` field whenever a reference is supplied (`reference_x`/`reference_sr` at the
    DSP layer; `reference_path` on `lib.analyze` and the `analyze` CLI verb, matching `advise`,
    `master` and `eq-match`'s existing convention). Without a reference, the field is **absent**
    from every entry -- never `null`, never zero-filled -- so a caller can tell "not computed"
    from "computed as zero".
  - `advise`/`master`: when `reference_path` is given, `rel_reference_db` is now the PRIMARY tonal
    signal in the advisor's system prompt; `rel_median_db` is explicitly demoted to a fallback for
    when no reference is available. No behaviour change when no reference is given.
  - No default/built-in target contour was added: with no reference there is still no taste-free
    anchor, and a built-in target curve is genre- and material-dependent -- that judgement stays
    out of the tool.
  - Two variants were tried and rejected on the same fixtures (see
    `reference_anchored_band_deviation_db`'s docstring): deviation from a smooth 2nd-order
    polynomial fit across log-frequency (1 of 6), and subtracting each file's own median from the
    raw per-band delta instead of the median of the deltas (5 of 6 -- still fails on a broad tilt).
  - Regression coverage: `tests/test_reference_anchored_tonal.py` generates six induced-defect
    fixtures from a common pink-ish-plus-harmonics source (seed 7), confirms each defect is
    actually present (Welch PSD delta vs. the clean reference) before asserting anything can find
    it, asserts 6 of 6 on the new signal, and pins a DELIBERATE negative control asserting the old
    (file's-own-median) signal scores exactly 1 of 6 on the same fixtures -- so a future change
    that silently reintroduces it as the primary signal is caught even with no reference present.
    This is deterministic DSP, tested for free with no model call.
- **`advise` no longer lets a proposed EQ move contradict the very measurement it cites.** Measured
  end-to-end: given a file missing its top end (`rel_reference_db` -35.9 dB deficient at 16 kHz), a
  real model correctly read and narrated the deficiency ("extreme high-frequency rolloff... lacks
  presence") and then proposed *cutting* that band further -- the opposite of its own stated
  reasoning. The prompt already states the sign convention explicitly; this needed a check that
  runs for free, not a stronger (unverifiable) prompt.
  - `aud.intelligence.advisor._eq_move_contradiction`/`_filter_contradictory_eq_moves`/
    `_nearest_band_hz`/`_octave_reference_lookup`: new pure functions. A proposed `peaks`/`shelves`
    move is checked against the nearest octave band's `rel_reference_db` (band matching: nearest
    centre in log2/octave space -- a move at 6 kHz maps to the 8 kHz band, since their shared
    boundary is `sqrt(4000*8000) ~= 5657` Hz). A cut whose band measured deficient, or a boost
    whose band measured in excess, is a contradiction.
  - Response to a contradiction: the offending move is DROPPED (never sign-corrected -- rewriting
    what the model proposed into something it never said is not a fix, it is the tool
    misrepresenting its own provenance) and reported as an additional `{"stage": "eq", "reason":
    "GUARDED: ..."}` entry in the same `reasoning` list every other stage's rationale already
    travels in -- visible on stderr (`aud advise`'s own reasoning printout) and in every library
    caller's `"stages"` key, exactly like any other stage's reason. One wrong sign on one band no
    longer discards an otherwise-good chain (a correct `loudness`/`limit`, or a correct move on a
    different band).
  - Guards `peaks` and `shelves` only -- each carries an explicit `(freq_hz, gain_db)` pair that
    maps onto one band's signed measurement. `hpf`/`lpf` are NOT guarded: both are broadband corner
    frequencies with no `gain_db` of their own (the attenuation varies continuously with frequency
    rather than being one number at one band), so there is no well-defined sign to compare.
  - A strict no-op with no reference supplied (`rel_reference_db` absent from every band) or with
    no measurements passed at all -- `_validate_and_build_plan`'s new `measurements` parameter
    defaults to `None`. Never falls back to guarding against `rel_median_db`.
  - Regression coverage: `tests/test_advisor_eq_guard.py`, all deterministic (no model call),
    reusing the six induced-defect fixtures from `tests/test_reference_anchored_tonal.py`. Includes
    the caught case (a cut proposed on the `dull` fixture's deficient 16 kHz band), the control
    that proves the guard is not simply rejecting every cut (a cut on the `boxy` fixture's *excess*
    500 Hz band passes untouched), a boost-in-deficient-band control, and the no-reference/no-
    measurements no-op cases.

### Added

- **Two new deterministic plan stages: `downmix` and `resample`.** Both are output-format
  decisions rather than mastering ones, so they sit at the very end of canonical order,
  immediately before the file is written (`contracts/plan.v1.md#why-downmixresample-sit-at-the-very-end`).
  - `downmix` folds a multichannel programme to one channel by taking the arithmetic mean across
    channels (sum-and-divide, never a plain sum) -- provably unable to push a sample outside
    [-1.0, 1.0] for in-range input. Its render report always includes the mean pairwise channel
    correlation and an `antiphase_detected` flag, so a caller can tell a good fold from an
    accidental near-silent one caused by out-of-phase channels, rather than getting a quiet file
    with no explanation.
  - `resample` converts to a target sample rate via `scipy.signal.resample_poly`'s polyphase
    resampler (its own anti-aliasing filter, never hand-rolled decimation).
  - Consolidated four independent one-line mono-fold copies (`aud.dsp.eqmatch._mono`,
    `aud.dsp.speech`'s pre-whisper fold, `aud.dsp.reverb`'s IR downmix) into one shared
    `aud.dsp.channels.fold_to_mono`; `aud.dsp.resolve._mono_sum` is deliberately left alone (it
    sums rather than averages -- a different computation, not the same one spelled differently).
- **`sample_rate_policy` (docs/CONFIGURATION.md) is now wired to `render`.** It was previously
  documented and accepted by `aud config`/`AUD_SAMPLE_RATE_POLICY` but never read at render time.
  `"preserve"` (default) writes at the input's own rate; an integer resamples the rendered output
  to that rate. An explicit `resample` stage in the plan always takes precedence.
## [0.12.0] - 2026-09-21

Three measured defects in `advise`'s diagnosis, found by controlled measurement
(induced spectral defects, re-rendered and re-measured, not just graded on the
proposed plan's parameters) and fixed together because all three sit in the
same seam -- what the advisor is shown and how it is told to read it.

### Fixed

- **The advisor was never told `shelves` exist.** `lib.eq` has supported shelf
  filters since they were added to `contracts/plan.v1.md`, but
  `intelligence/prompts.py`'s stage reference only listed `hpf`/`lpf`/`peaks` --
  the chooser could not name the correct instrument for a broad top- or
  bottom-end tilt ("dull"/"no air", "rumbly"/"boomy"), only a narrow bell.
  Measured before the fix: 0 of 6 shelf-appropriate cases across two model
  tiers ever produced a shelf. `eq`'s stage reference now documents `shelves`
  and includes guidance on choosing a peak (one isolated band) vs. a shelf
  (several consecutive bands moving together at the spectrum's edge).
- **A model tier applied a reflexive high-pass to clean material.** At
  `--model claude-opus-5`, `advise` proposed a 25 Hz high-pass on verified-clean
  pink noise 3/3, citing a trivial DC offset and the quietest band as if they
  were defects -- a tool that always finds something is not diagnosing. The
  system prompt now explicitly forbids a reflexive EQ move and instructs the
  model to propose no `eq` stage at all when every band's relative deviation
  is small.
- **Tonal diagnosis compared absolute dB values with no derived comparison,
  and `octave_band_energy_db`'s dict keys sort lexicographically once
  serialized with `sort_keys=True`** (1000.0 next to 125.0, 16000.0 next to
  2000.0), so a naive "compare to the next entry" walk over that report
  compares the wrong neighbours. `analyze()` now also returns
  `octave_band_analysis`: a numerically-ordered list (immune to key
  resorting) carrying two derived, already-computed comparisons per band --
  `rel_median_db` (vs. the file's own overall median; primary signal) and
  `neighbour_contrast_db` (vs. immediate octave neighbours; secondary, with a
  documented blind spot on a defect spanning two adjacent bands). The system
  prompt now directs the advisor to ground every `eq` reason in the signed
  `rel_median_db` value, not the raw absolute level.

Re-measured against the fixed tool with real model calls (not assumed):
a broad top-end tilt now gets a high shelf, a broad bottom-end tilt now gets a
low shelf (reachable even at the default model tier, not only at a top-tier
model), and the same verified-clean control that previously got an invented
high-pass now proposes no `eq` stage at all. Regression tests replay these
real recorded responses (`tests/fixtures/recorded/anthropic/advise-{dull-shelf,
clean-pink}-opus5.json`, `advise-rumbly-shelf-haiku.json`) so a future prompt
change that reintroduces any of the three defects fails a test, not just a
manual check.

## [0.11.1] - 2026-09-20

### Fixed

- **Agent Skills description exceeded 1024 character limit.** The skill's frontmatter
  description was 1093 characters, breaching Agent Skills spec compliance. Trimmed to 713
  characters by removing DSP-stage enumeration, measurement-metric list, and pipe-chaining
  mechanics — mechanics that runtime help already documents — while preserving the boundary
  clause (not video, not multitrack, not transcription, not generation) that stops
  agents reaching for the wrong tool. Added regression test covering all `skills/*/SKILL.md`
  frontmatter descriptions to prevent re-introduction.

## [0.11.0] - 2026-09-20

Ten spec deviations closed, and the README turned from a tutorial into a landing page.

### Fixed

- **Error envelopes went to stdout.** Every `AudError` was printed to stdout, so a caller piping
  a chain got the error mixed into the stream carrying results. Errors now go to **stderr**;
  stdout carries only the result. Verified: a bad flag leaves stdout empty and puts the envelope
  on stderr, exit 2.
- **The library required a shared filesystem.** `curve_apply` took only a path and read it
  inside the library, so a caller in another process could not use it. It now accepts the curve
  as **data**; the CLI reads the file and passes the content, which is a command-line
  convenience rather than a change to what the library accepts.
- **Artifact locations had no stated base.** `render`, `curve extract` and `curve apply` echoed
  back whatever relative path the caller passed. Each destination is now resolved to an
  **absolute path** before writing, and that resolved path is what the result reports —
  including `master`'s propagated render result.
- **A typo in a setting was reported as our bug.** A bad `AUD_*` value raised a bare `ValueError`
  that surfaced as `internal_error` telling the user to file a bug report. It is now
  `bad_config`, naming the variable, the bad value and the expected type —
  `AUD_OVERSAMPLE is 'four'; 'oversample' must be an integer.` — which is what
  `docs/CONFIGURATION.md` already promised. Curve output is written atomically, and an
  unwritable destination is a named `bad_path` with no partial file left behind.
- **A partial analysis looked like a complete one.** `analyze` caught every exception from the
  loudness-range measurement and returned `loudness_range_lu: null` with no status or reason —
  so a genuine measurement limit and a real bug were indistinguishable. The bare `except` is
  gone, an explicit precondition decides when the metric is unavailable, and a reason is
  reported. This also fixed a crash: the integrated-loudness call was **unguarded**, so `analyze`
  failed outright on any file below pyloudnorm's 0.4 s gating block.
- **Prerequisites were checked after the work.** `advise` analysed the input before looking for a
  credential, and `detect fillers` decoded the entire file before checking whether the speech
  extra was installed. Both now fail first: a nonexistent path still reports
  `provider_credential_missing` / `speech_extra_missing` rather than `file_not_found`.
- **`aud --help` had no install and no non-goals**, though the manifest carried them. Both are
  now in the rendered tool skill (104 lines, against the 500-line Agent Skills ceiling).
- **Capability help was incomplete**, and two entries were lying. Every verb's `--help` now
  carries **Kind** (deterministic or model-backed), **Result** and **Failures** with real error
  codes traced from the source. `cut` and `strip-silence` still claimed to be unbuilt and to
  return `not_implemented` — they have rendered since 0.4.0. Swept the whole file; no other
  stale claim remains.
- **The manifest promised something untrue.** "Every verb appends to a plan" is false for
  `manifest`, `check`, `analyze`, `config` and `plan`. It now says every **chain stage** appends
  to a plan, and names the read-only and lifecycle verbs separately. The manifest also now
  declares the **Azure endpoint** prerequisite, which the runtime enforces and the manifest had
  omitted.

### Changed

- **The agent skill is thin, the way the spec asks.** `skills/aud/SKILL.md` went from 104 lines
  to 58: frontmatter, a short summary, install, and the instruction to run `aud --help` then
  `aud <verb> --help`. The prerequisite matrix, pipelines and preset names it used to carry —
  all of which could drift from runtime help — now live in `aud --help`, which is their proper
  home. The `description` is untouched: that field is the discovery surface, it describes intent
  rather than mechanics, and it cannot drift.
- **The README is a landing page, not a tutorial.** 249 lines to 104, matching the rest of the
  fleet: what it is, the full install ladder, an Interface section, and pointers into `docs/`.
  It now carries **`npx skills add colombod/amplifier-smart-tools-audio`**, which was missing
  entirely despite the repo shipping a skill — nobody could install it into another harness.
  Also removed a status box still claiming `detect`, `cut` and `strip-silence` were unimplemented
  in 0.3.0.

### Verified

- 418 tests pass, 0 skipped (was 382). ruff clean. Conformance 16 PASS / 0 FAIL.
- Each of the ten deviations re-checked against its own evidence by running the tool, not by
  reading the diff — including the three config coercion paths, the two prerequisite orderings,
  and the stdout/stderr split.

### Known limits

- `smart-tool-creator` no longer exposes the `check-spec-adherence` verb that produced the
  original report, so the ten fixes were verified individually against that report's own
  file:line evidence rather than by re-running it.
- Config type-checking covers the environment-variable tier. The config-file tier is unchecked:
  `sample_rate_policy` legitimately takes either a string or an int and `output_subtype` needs
  enum validation, so a generic check needs per-key logic.

## [0.10.0] - 2026-09-19

An installable skill, the two replay rules moved from prose into code, gate and expander
finished, and the third instance of one bug — fixed by making a fourth structurally
unwritable.

### Added

- **`skills/aud/SKILL.md`** — the file `npx skills add` installs. It is a POINTER, not a copy
  of `--help`: it says loudly what the tool is for and the asks that should make you reach for
  it, what each optional tier unlocks, and then sends the reader to `aud --help`,
  `aud <verb> --help` and `aud check` for everything else. A test pins that structure.
- **`podcast` and `voiceover` presets now include `expand`**, the gentler quiet-end device.
  Rendered against material with a real −45.10 dBFS noise floor: **−4.60 dB** max attenuation
  on `podcast`, **−3.93 dB** on `voiceover`, each over ~51% of the programme. `broadcast` and
  `music-streaming` deliberately omit it, and the code says why — a music noise floor is often
  intentional (room tone, a decaying reverb tail) and there is no "between phrases" to target.

### Fixed

- **The third instance of one bug, and the last one writable.** When `snap: "transient"` found
  an onset but the zero-crossing refinement then failed, the code kept the raw candidate while
  reporting `rule_applied: "transient"`, `snap_failed: false` — claiming the alignment floor
  ran when it had not. 0.6.0 fixed this shape where the coarse search found nothing; the pass
  before this one fixed it for `snap: "silence"`. Three instances means the SHAPE was the
  defect, so all five zero-crossing call sites now route through one helper that is the only
  place `_nearest_zero_crossing`'s `None` case is handled. A fourth instance cannot be written
  without going through its honesty contract.
- **An existing "happy path" test was quietly validating the same bug.** Fixing the above broke
  `test_transient_snap_lands_before_the_onset_never_after` — its pre-onset material was a flat
  near-zero constant that never crosses zero, so it had only ever passed *because* the silent
  fallback succeeded. Its signal is corrected to exercise a genuine refinement.
- **`snap: "silence"`** had the same latent shape and is fixed with it: a failed refinement now
  reports `unaligned` and `snap_failed: true` instead of `silence` and `false`.
- **A write failure said it was our bug.** A bad subtype or an unwritable destination reported
  `internal_error` — telling the user to file a bug report against us for their full disk. Now
  `audio_write_error` with an actionable remedy, verified against a provoked invalid subtype
  and a permission-denied directory.

### Changed

- **The two replay rules are enforced, not documented.** `install_faster_whisper_replay` now
  REQUIRES the caller to name the source it drives with and verifies the sha256 itself; a test
  that replays only the recorded answer must say so through `replay.UNBOUND("reason")`, which
  rejects a blank reason — so an exemption is a greppable decision rather than an omission.
  And a new guard reads the RECORDINGS themselves and pins every awkward truth they carry: a
  word with `start == end`, silence returning one hallucinated word while a tone returns zero
  segments, the Signalsmith reciprocal, the float32 read-back, the attributes that are really
  methods, and both Anthropic content shapes. If a future re-recording comes back clean, the
  guard fails saying which coverage was lost instead of passing quietly.

### Verified

- **Model tier is now a pinned property, not an anecdote.** All four recorded Anthropic
  responses replay through the real validator and plan builder: the sonnet-thinking response on
  hissy material yields a chain containing `expand` and cites the noise-floor gap; the haiku
  response on the *same* material does not; both clean-material responses correctly reach for
  neither. A prompt change that breaks that reasoning now fails a test.
- The silence-snap and transient-snap fixes were each verified in BOTH directions — stashed,
  observed failing against the old code, restored, observed passing. A regression test never
  seen to fail is not yet a regression test.
- OpenAI, Google and Azure response parsing now has coverage, and all three already raise a
  clean `provider_request_failed` rather than an opaque `KeyError` on a malformed response.
- 382 tests pass, 0 skipped (was 330). Conformance 16 PASS / 0 FAIL.

### Known limits

- Those three non-Anthropic backend tests are written against each provider's **documented**
  response shape, not a recorded live one. That is precisely the gap that let the Anthropic
  `content[0]["text"]` defect ship, and only a real recorded call closes it.
- Transient detection still produces a bounded number of spurious onsets on a pure sustained
  tone. The docstring and `contracts/regions.v1.md` now state what it is and is not suited to
  rather than leaving it as a shrug.

## [0.9.0] - 2026-09-19

Every hand-written mock in the test suite is gone, replaced by replays of real recorded runs.

### Changed

- **No test fakes a third-party library any more.** `tests/replay.py` is one shared harness that
  plays back real captured responses: faster-whisper transcriptions, python-stretch input/output
  pairs, and Anthropic request/response envelopes. It fails loudly on a missing recording rather
  than falling back to a made-up value.
- **`tests/fixtures/recorded/`** holds those captures — 6.2 MB, recorded in a throwaway container
  against real piper-tts speech, each with provenance: library version, source audio sha256,
  capture date, and the command that produced it. `RECORDING.md` documents how to re-record when
  a library version moves.
- **The three Signalsmith tests stopped skipping.** They skipped because `python_stretch` is not
  installed on the development host; the recording is, so they replay. 330 tests, **0 skips.**

### Why

A fake built from the fields your own reader consults can only confirm the shape you already
assumed. This repo paid for that three times in one day, with 284 tests passing throughout:

- the hand-written whisper transcript could not produce a word with `start == end` and carried no
  sample rate — the two defects that made `detect fillers` wrong on every real file;
- the hand-written provider fake always returned a single text block, so every reasoning model
  was unusable in production while the suite was green;
- the hand-written Signalsmith stub reproduced our own belief about `timeFactor`, which was the
  reciprocal of the truth — it confirmed the bug instead of catching it.

The rule is now in AGENTS.md, with the two traps that make a replay worthless: a replay is bound
to its **exact input bytes** (the same speech at 16 kHz and at 48-kHz-resampled-to-16 kHz gave
different transcripts — 168 words versus 106), and a replay **tidier than the real library is a
fake again**, so the recordings keep the awkward parts.

### Verified

Only real data makes these assertions possible:

- a **real** degenerate `' um,'` at 24.0 s on the correct resampled path — the whole document now
  survives it and the other 13 real fillers come through, with the drop counted;
- sample-rate independence asserted across **real** 16 kHz and 48 kHz recordings of identical
  content, each checked against its source wav's sha256;
- the Anthropic parser exercised against both **real** content shapes, `['text']` and
  `['thinking', 'text']` — the second is the shape that raised `KeyError` in production;
- the Signalsmith reciprocal settled by the **real** recorded table (`timeFactor` 1.2 set
  literally gives out/in 0.8333; set as 1/f gives 1.2000), plus a byte-exact replay of real audio;
- silence really returns one hallucinated word, and a 1 kHz tone really returns zero segments —
  both now handled deliberately rather than by assumption.
- 330 tests pass, 0 skipped (was 315 + 3 skips). Conformance 16 PASS / 0 FAIL.

### Known limits

- The four Anthropic recordings have no automated re-record script; RECORDING.md documents the
  manual recapture. The two library recordings do have one.
- Two disclosed synthetic values remain, both testing OUR code rather than a library's: a
  negative-duration word (no real recording ever produced one — only zero-duration), and a
  backend returning deliberately invalid text, which no successful recording can supply because
  recordings only capture calls that worked.

## [0.8.0] - 2026-09-19

Five defects, every one found by running the tool for real in a throwaway container against
real synthesised speech, and every one invisible to a test suite of 284 passing tests. Nothing
here was caught by reading code.

### Fixed

- **`detect fillers` ignored the sample rate.** `speech.detect_fillers` accepted `sr` and never
  used it; faster-whisper assumes 16 kHz for array input. Identical audio gave a filler at
  2.800–2.980 s at 16 kHz and at **3.840–4.060 s at 22.05 kHz** — a measured 1.3714 stretch
  against the expected 1.3781. At 44.1 kHz every timestamp would be 2.76× out, at 48 kHz 3×,
  and those positions feed straight into `cut`, so the tool would have cut the wrong part of
  the file. Now resampled to 16 kHz before transcription; verified to preserve real duration
  exactly at 16 k, 22.05 k, 44.1 k and 48 kHz.
- **One zero-duration word destroyed the entire document.** faster-whisper can emit a word with
  `start == end`; the parser passed it through and validation then rejected the WHOLE regions
  document, losing every other correct detection with it. Seen live in **3 runs out of 5** on a
  58-second file. Degenerate words are now dropped at the parser and COUNTED in the document's
  `detection` block, so the caller can see it happened.
- **The CLI's default vocabulary could not find "um".** Two filler-word lists existed — the
  library's and a second hard-coded string in the CLI — and the CLI's always won. `um`, the
  commonest English filler and one this tool's own manifest advertises, was not in it. Measured
  cost on identical audio: **1 of 3 fillers found with defaults, 2 of 3 with the library's
  list.** There is now one list, and a test asserts the CLI's effective default IS that list.
- **`--factor` was inverted on the Signalsmith engine.** `--factor 1.2` produced `out/in =
  0.8333` — 7.287 s of speech became 6.072 s when asked to lengthen. Signalsmith's `timeFactor`
  is the reciprocal of ours. Worst of all it is selected AUTOMATICALLY when the `stretch` extra
  is present, so installing an optional quality tier silently reversed the meaning of the flag,
  and the stats reported `measured_factor: 0.833` beside the requested `1.2` with nothing
  comparing them. Fixed, and the measured ratio is now CHECKED against the request on both
  engines — a stat that is computed, reported and never checked is an alarm nobody wired up.
- **Cutting a filler left an audible remnant.** Whisper closes a word 175–200 ms before the
  vowel stops (measured: `um` truth 0.599–0.938 s, returned 0.600–0.740 s). `cut` took that
  literally, so transcribing the output showed *"So, um,"* had become *"So, hmm,"* rather than
  disappearing. Filler regions now carry a kind-aware `filler_tail_pad_ms` (200 ms, justified
  by that measurement) which the caller can override. `pad_out_ms` was deliberately NOT
  repurposed: the contract promises padding can only ever shrink a cut, and inverting that for
  every other caller to fix this would have been the wrong trade.
- **Two ceiling checks disagreed.** `lib.verify()` used a zero-tolerance comparison while
  `limiter.brickwall` used a documented 0.05 dB one, so a limiter working exactly as designed
  could be reported as failing. The tolerance is now defined once and imported by both.

### Verified

- 315 tests pass (was 284). Conformance 16 PASS / 0 FAIL.
- The container run that found all of this also confirmed what works: the parser DID survive
  contact with real faster-whisper objects, `aud[speech,stretch]` installs in 16.6 s with no
  compiler and no credential, and the full `detect fillers | cut --snap transient | render`
  chain ran end to end on real speech with the removed durations reconciling exactly
  (7.286625 s → 7.097313 s, the 10 ms difference being one crossfade).
- Filler detection cost, so it can be budgeted: a 142 MiB model, ~15 s first-run download,
  ~1.2 s fixed cost per invocation, and roughly 1.6–5× realtime on CPU.

### Known limits

- **Three tests skip on a machine without the `stretch` extra**, and they are the ones covering
  the Signalsmith engine. The inversion fix is proven against a mock built from the measured
  behaviour of the real library, not against the library itself. That gap is named in the skip
  reason rather than hidden, and closing it needs a container.
- Only Anthropic has been exercised live among the four model backends.

## [0.7.0] - 2026-09-19

The quiet end of dynamics, presets, and the contract finally matching the code. Plus two
defects that only a live provider call could have found.

### Added

- **`gate` and `expand`** — the other half of dynamics. The chain could pull loud material
  down and had nothing for the quiet end, which for cleanup work is usually the stage that
  matters most: room tone between phrases, mic hiss, air conditioning. `gate` is hard
  (attenuate by `range_db` below the threshold — a duck, not a kill, because a gate that slams
  to digital black sounds worse than one that ducks 20 dB); `expand` is gentle (a downward
  ratio, usually the right tool for a voice). Both are multiband via the existing
  Linkwitz-Riley split, both take their threshold RELATIVE to the measured noise floor, both
  have lookahead so a phrase's onset survives, and both sidechain off a high-passed copy so
  rumble cannot hold them open. They sit with the repair stages, BEFORE compression: gating
  after compression is backwards, because the compressor has already lifted the floor the gate
  exists to remove.
- **`aud preset --list` and `aud preset show <name>`** — `podcast` (−16 LUFS, Apple Podcasts),
  `music-streaming` (−14, the Spotify/Apple/YouTube convention), `broadcast` (−23, EBU R128)
  and `voiceover`. Each emits a plan document, so `aud preset show podcast | aud render in.wav
  out.wav` is one command. Every preset is built through the same `aud.lib` stage builders as
  everything else — a preset that hardcoded a params dict would be a fourth source of truth.
- **`shelves` are reachable.** The contract promised them and the engine applied them, but no
  CLI or builder path existed — a documented capability nobody could call.

### Fixed

- **The contract and the code now agree.** Eleven divergences, every one resolved by moving the
  CODE to the published document: `eq` stored `hpf`/`lpf` and raw peak triples where the
  contract says `hpf_hz`/`lpf_hz` and `{freq_hz, gain_db, q}` objects; `compress.ratio` accepted
  values below 1.0 (upward compression, a different device); per-band defaults came from a
  dataclass rather than the contract; a contract-conformant single-band plan
  (`crossovers_hz: []`) CRASHED at render; `limit.oversample` was documented, defaulted, and
  silently never read. CLI flag spellings stay short (`--hpf`) because the contract explicitly
  does not promise them — flags are ergonomics, the document is the interface.
- **A reasoning-capable model was unusable.** The Anthropic backend read `content[0]["text"]`,
  but `content` is a list of BLOCKS and a thinking model returns a `thinking` block first — so
  every such model came back as "response did not have the expected shape". Now takes the first
  text block.
- **The advisor could not reach the new stages**, and would not have weighed them if it could:
  `gate`/`expand` were absent from its whitelist and vocabulary, and nothing told it that
  compression and loudness LIFT a noise floor, so hiss tolerable in the source is audible in
  the master. Both fixed.

### Verified

Gate and expander, measured:

- **Chatter**: an envelope crossing the threshold at 20 Hz for 2 s opened the gate **79 times
  with `hold_ms=0` and once with `hold_ms=100`**. That is the single most recognisable way a
  gate sounds broken, and `open_count` now ships in the stats so a caller can see it.
- **Onset survival**: a sharp onset after silence measured −1.94 dB in its first 5 ms;
  with 15 ms lookahead the output was **−1.98 dB (0.04 dB lost)**, with no lookahead
  **−21.94 dB (20 dB lost)** — the attack simply clipped off.
- **Floor-relative threshold**: the same material 20 dB quieter measured a floor 20.00 dB lower
  and gated **the identical 28.39%** of the programme. A fixed dBFS threshold fails that.
- **Multiband isolation**: a loud 150 Hz tone under a quiet 6 kHz tone, split at 1 kHz — low
  band 0.0% attenuated, high band 99.98% at −18.0 dB.
- **Sidechain**: 40 Hz rumble held the gate open (0.00 dB of gating) until an 80 Hz sidechain
  high-pass was engaged, after which the gap closed by **19.99 dB**.

The advisor, run LIVE against Anthropic, both directions:

- Hissy material (floor −30.5 dBFS against −13.06 LUFS, 17.4 dB separation) → it chose
  `expand`, citing *"only ~17.4 dB separation (well under the 25 dB clean threshold), so the
  quiet passages need gentle downward expansion before loudness/limiting amplify that floor"*.
- Clean material (52.8 dB separation) → no gate, no expander. The negative case matters as much
  as the positive one: a gate on clean material is damage.
- All four presets rendered to their stated targets: −16.00, −14.00, −23.00, −16.00 LUFS, ceiling met in every case.
- A plan hand-written using ONLY the contract's own field names now renders. That was the
  property that was broken, and it is asserted directly.
- 282 tests pass (was 233). Conformance 16 PASS / 0 FAIL.

### Known limits

- **Model tier changes the answer.** On the hissy file the default (`claude-haiku-4-5`) did not
  reach for the gate even with the noise-floor rule in the prompt; `claude-sonnet-5` did, and
  cited the right numbers. The prompt was not the problem. A cheap default produces cheaper
  mastering decisions, and `--model` is how you buy a better one.
- Only Anthropic has been exercised live. OpenAI, Gemini and Azure OpenAI are still
  reviewed-but-uncalled — and the block-parsing defect just fixed is exactly the kind of thing
  waiting in them.
- Presets do not yet use `gate`/`expand`.

## [0.6.1] - 2026-09-19

The first live provider call `aud` has ever made, and the two defects it found immediately.

### Fixed

- **The default Anthropic model did not exist.** `aud advise` against a perfectly valid key
  returned `HTTP 404 model: claude-3-5-haiku-20241022`. The default had been chosen by reading,
  never by calling, and no test could have caught it: every intelligence test injects a fake
  backend through the Protocol, which is exactly what makes the suite free and exactly what
  makes it blind here. Default is now a model verified to answer.
- **A wrong model was reported as a credential problem.** A 404, or any provider error naming
  the model, now returns `provider_model_unavailable` with a remedy pointing at `--model` and
  `AUD_MODEL` — instead of `provider_request_failed` telling the caller to go and check the one
  thing that was working. A default model name is perishable; the error it produces should say
  so rather than sending someone to debug their key.

### Verified

Run for real against Anthropic, not with a fake:

```
$ aud advise in.wav --target -14
aud advise: provider=anthropic model=claude-haiku-4-5-20251001
  - loudness: Integrated loudness is -11.759 LUFS, requiring a gain reduction of
    approximately 2.24 dB to reach the target of -14.0 LUFS.
  - limit: True peak is currently -8.763 dBTP; after loudness adjustment (2.24 dB gain),
    peaks will rise to approximately -6.5 dBTP, exceeding the -1.0 dBTP ceiling, so
    limiting is required.
{"plan_format":1,"created_with":"aud/0.6.1","stages":[...]}
```

- `aud master in.wav out.wav --target -14` ran end to end as ONE shell command on the default
  model and verified at **-14.00 LUFS, -11.00 dBTP** against -14 and -1.0.
- The reasoning is grounded in the measurements the deterministic analysis produced, not in
  anything the model was told about the audio — it never sees audio.
- The retired-model path was re-run deliberately and now returns `provider_model_unavailable`.
- 233 tests pass. Conformance 16 PASS / 0 FAIL.

### Known limits

- Only the Anthropic backend has now been exercised live. OpenAI, Gemini and Azure OpenAI are
  still reviewed-but-uncalled, and their default model names carry the same perishability that
  just bit the Anthropic one.

## [0.6.0] - 2026-09-19

The smart tier arrives, and the last audible defect in edit placement is closed. `aud` is now
what it set out to be: deterministic DSP that needs no credential, with judgement available on
top for a caller that wants the decision made for it.

### Added

- **`aud advise in.wav`** — runs the deterministic `analyze`, hands the MEASUREMENTS to a model,
  and returns a mastering plan with a stated reason per stage. It never touches samples, and it
  emits a plan document, so it pipes straight into `render`. The model sees a JSON measurement
  report — LUFS, true peak, crest factor, band balance, sibilance, noise floor — never audio.
- **`aud master in.wav out.wav --target -14`** — advise, render and verify in one shell command,
  emitting one document containing the chosen plan, the per-stage report, and the measured
  result against the target. `--dry-run` shows the plan and renders nothing.
- **One intelligence interface, no SDK.** `IntelligenceBackend.complete(system, user, *, model,
  max_tokens) -> str` with plain-HTTPS implementations for Anthropic, OpenAI, Gemini and Azure
  OpenAI. The base install stays numpy/scipy/soundfile/pyloudnorm: a caller with no credentials
  pays nothing for a capability they cannot use.
- **The model's output is treated as untrusted input.** A proposed plan is applied through the
  same `aud.lib` builders and validators every hand-written plan goes through, so an invented
  stage name, an out-of-range ratio or malformed JSON is rejected with a clear code instead of
  failing at render.
- **A visual explainer** of the chain in `docs/images/`, wired into the README and the head of
  ARCHITECTURE.md, which now opens with install and unfolds the five phases from there.

### Fixed

- **A failed `transient` snap silently dropped the click protection.** When no onset was found,
  the edit point fell back reporting `rule: "none"` — and `"none"` is the one snap value the
  contract says disables zero-crossing alignment, because it is the only one asserting the
  caller already picked the sample. So the raw, unaligned position shipped. Zero crossing is the
  FLOOR, not a peer: the coarse rules decide WHERE, alignment happens regardless. A failed
  search now still aligns and reports `zero_crossing_fallback`; `"none"` is reserved for a
  caller's explicit instruction.
- **`snap_failed` was firing where no onset could exist.** A silence region's START is where
  speech stopped, so a transient search there was guaranteed to miss — half of all edit points
  reported a failure by construction, which trains a caller to ignore the flag that would have
  shown them the real problem. It is now per-boundary: a miss at the resume point is a genuine
  failure, a miss at the trailing edge is the expected outcome.

### Verified

- **The click regression, measured both directions**: cutting a region with `snap="transient"`
  from material containing no onsets gave a join discontinuity of **1.199600** before the fix
  and **0.018805** after — against the signal's own natural sample step of 0.018806. The
  unaligned splice was ~64× worse. On the original reproduction, `snap_failures` went 2 → 0.
- **The no-credential refusal, run for real** with all five provider variables unset:
  `aud advise` exits 1 with `provider_credential_missing` naming every variable that would
  satisfy it, while `aud analyze`, `aud detect silence` and a full deterministic chain all
  still exit 0.
- **`master` end to end** with an injected advisor: analyzed at −10.1 LUFS / −6.7 dBTP, chose a
  chain, rendered, and verified at **−14.01 LUFS / −6.64 dBTP** against −14 and −1.0.
- 233 tests pass (was 197). Conformance 16 PASS / 0 FAIL.

### Known limits

- **No live provider call has ever been made.** Every intelligence test injects a fake backend
  through the Protocol — no network, no credential, no cost — so the four HTTPS request builders
  are reviewed but unexercised. The first real call will be the first real test of them.
- `snap: "silence"` has the same latent shape as the `transient` defect just fixed: if the
  zero-crossing refinement finds no sign change it falls back to the raw candidate while still
  reporting `silence` and `snap_failed: false`. No test exercises that path yet.
- The Signalsmith path in `stretch`/`pitch` and `detect fillers`' live model call remain
  unexecuted anywhere; both need a throwaway container.

## [0.5.0] - 2026-09-19

The chain is complete. Every stage the manifest has claimed since 0.1.0 now renders: the six
that were still refusing — `eq_match`, `deess`, `dereverb`, `reverb`, `stretch`, `pitch` —
have real DSP behind them, written on numpy and scipy. Nothing in `STAGE_ORDER` raises
`not_implemented` any more.

### Added

- **`eq_match` and `curve extract` / `curve apply`** — measure a reference recording's 1/3-octave
  spectral profile, store it as NUMBERS (a stored curve replays with no access to the reference
  file, the same rule the `eq` stage already follows), and apply the clamped difference as a
  linear-phase FIR. Linear phase because this is a tone match on a finished programme and phase
  smearing is the one thing you do not want here.
- **`deess`** — a dynamic de-esser, not a static notch: the sibilant band is split out, detected
  on its own, and gain-reduced only while sibilance is actually present.
- **`dereverb`** — STFT-domain late-field estimation and subtraction, with a spectral floor and
  time/frequency smoothing against musical noise.
- **`reverb`** — a four-line feedback delay network with damping and pre-delay, plus convolution
  against a user-supplied impulse response. Defaults are deliberately subtle: the manifest says
  *controlled ambience*, and this is a mastering tool, not a plate.
- **`stretch` and `pitch`** — a phase vocoder with identity phase locking, resampling for pitch.
  `python_stretch` (Signalsmith, MIT) is used instead when the `stretch` extra is installed; its
  absence is not an error, because the built-in tier genuinely works, and the stats say which
  engine ran.

### Fixed

- Three stage builders wrote field names the plan contract does not use — `reverb` emitted
  `amount`/`decay` against the contract's `mix`/`decay_s`/`predelay_ms`, `stretch` emitted
  `factor` against `ratio`, and `deess`/`dereverb` emitted `amount`/`freq` against
  `amount_db`/`freq_hz`. None had ever rendered, so none could have been caught by running the
  tool; all now match the contract.

### Verified

Measured, not asserted:

- **`eq_match`**: a dull source against a bright reference closed **98.5%** of the spectral
  distance (13.558 dB RMS before, 0.202 dB after), monotonic in strength
  (13.561 / 6.792 / 0.186 dB at 0.0 / 0.5 / 1.0), and the boost clamp held at 11.98 dB against a
  12.0 dB limit on a band where the reference had nothing but noise floor.
- **`deess`**: 9.5 dB measured reduction on a sibilant burst against 10 dB requested, while the
  non-sibilant passage of the same signal moved **0.0007 dB** — that gap is the whole difference
  between a de-esser and a shelf.
- **`dereverb`**: against synthetic ground truth (a dry signal convolved with a 400 ms decaying
  IR), the energy-envelope decay rate moved from −27.9 dB/s toward the dry reference's, to
  −43.2 dB/s. Already-dry material changes by 1.9 dB at `amount_db=10` — recorded as a real
  bounded limit, not hidden.
- **`reverb`**: decay time rises monotonically with `room_size` (0.170 / 0.331 / 0.521 s at
  0.15 / 0.50 / 0.90); damping drops the tail's spectral centroid from 10028 Hz to 239 Hz; peak
  never exceeded 1.05 across a sweep of room size against mix.
- **`stretch` / `pitch`**: stretching 2.0× gave a duration ratio of 2.0000 with **0.00 cents** of
  pitch drift; +12 semitones doubled the measured fundamental exactly with the length unchanged.
- **The whole chain in one shell command** — `detect silence | cut | stretch | pitch | dereverb |
  deess | eq | eq-match | compress | saturate | reverb | loudness | limit | render` — ran all
  thirteen stages in a single pass and landed at **−14.00 LUFS, −2.81 dBTP** against a −1.0
  ceiling, each stage reporting its own measurements.
- 197 tests pass (was 165). Conformance 16 PASS / 0 FAIL.

### Known limits

- **A failed `transient` snap drops zero-crossing alignment.** When no onset is found the point
  falls back to `rule: "none"`, which per the contract is the one value that disables alignment —
  so the click protection every other mode gets for free silently disappears. Separately,
  `snap_failed` fires on every silence-region START by construction, since a region start is
  where speech stopped and no onset can be there. Both are filed; neither is fixed here.
- The `limit` stage still reports `max_gain_reduction_db: 0.0` on material that never reaches the
  ceiling, and the true-peak assertions remain tolerance-blind — see the 0.3.1 notes.
- The Signalsmith path in `stretch`/`pitch`, and `detect fillers`' live model call, have still
  never been executed on any machine: installs on the development host are not permitted and both
  need a throwaway container.

## [0.4.0] - 2026-09-19

Detection and editing become real. `detect` finds things in a recording and emits a regions
document; `cut` and `strip_silence` remove them, with the blade placed deliberately rather than
wherever the detector's boundary happened to land. `aud detect silence in.wav | aud cut | aud
render in.wav out.wav` is now one shell command that does the whole job.

### Added

- **`aud detect silence | transients | fillers`** — emits a regions document (raw on stdout, the
  same treatment a stage verb gives a plan) per [contracts/regions.v1.md](contracts/regions.v1.md).
  The silence threshold is dB above the *measured* noise floor, never a fixed dBFS constant: a
  fixed number is wrong for every recording it was not tuned on. `detect fillers` needs the
  `speech` extra and refuses by name when it is absent.
- **`cut` and `strip_silence` render.** `cut` takes positions piped in from `detect`;
  `strip_silence` stores a rule, so the same plan still means something on next week's episode.
  Both run at the front of the chain, because cutting changes the timeline every later stage
  measures.
- **Edit-point resolution** — pad, then snap, then validate joins. Snap modes `zero_crossing`
  (default), `silence`, `transient`, `none`, bounded by `snap_window_ms`, with fades and an
  equal-power crossfade at every join. A snap that finds no valid candidate keeps the nominal
  position and records `snap_failed: true`; it never moves a point across the region's other
  boundary, into a neighbour, or out of the file.
- **A per-edit-point report.** Every point carries its nominal position, padded position,
  resolved position, distance moved, the rule that moved it and whether the snap failed — so
  where the blade actually landed is checkable rather than asserted.

### Verified

Measured on synthesised signals, not asserted:

- **Silence detection**: three gaps planted at 1.0–1.8 s, 3.0–3.6 s, 4.5–5.2 s in a 6.2 s file
  were returned as exactly those bounds, against a measured noise floor of −66.07 dBFS. The same
  signal 20 dB quieter yields identical regions — the property a fixed dBFS threshold fails.
- **Transient detection**: four planted onsets located within 6–15 ms.
- **The click test, which is the point of snapping.** Cutting 440.5 cycles out of a 440 Hz tone
  leaves the two sides in antiphase — the worst case for a naive splice. With `--snap none` the
  join produces a sample-to-sample step of **0.998875**, 31.9× the tone's own smooth baseline of
  0.031340. With `--snap zero_crossing` the step is **0.031340** — the discontinuity is gone
  entirely, not merely reduced.
- **`snap transient` preserves attacks**: cut ends nominally at 1.800/3.600/5.200 s resolved to
  1.8033/3.6028/5.2057 s, each landing just before the onsets at 1.806/3.607/5.208 s.
- **End to end**: `detect silence | cut --snap silence --pad-in 80 --pad-out 80 --crossfade 10 |
  render` took 6.200 s to 4.548 s, removing 1.62 s across three crossfaded joins with zero snap
  failures, and left the maximum sample-to-sample step unchanged from the source.
- **Ordering holds**: `detect silence | cut | loudness --target -16 | render` applies `cut` first
  and `verify` measures −16.0 LUFS on the cut material.
- **Failures are loud**: a crossfade longer than the material available names the join and both
  lengths (`crossfade_exceeds_gap`); `detect fillers` without the extra returns
  `speech_extra_missing` with the exact install command.
- 165 tests pass (was 96). Conformance 16 PASS / 0 FAIL.

### Known limits

- **`detect fillers`' live path has never been executed.** The faster-whisper call is written
  against its documented API and the word-to-region parsing is tested against a fake transcript,
  but the real model has not run — installs on the development host are not permitted, and this
  needs a throwaway container to verify. Treat it as unproven.
- Transient detection on a pure sustained tone produced spurious onsets from STFT bin-leakage
  until the threshold margin was widened; it is adequate for edit placement, not a
  publication-grade onset detector.
- Installing the `stretch` extra still buys nothing: `python_stretch` imports and its native
  extension loads, but the `stretch` render stage is not built.

## [0.3.1] - 2026-09-18

Two contract-violation defect classes found by installing the published 0.3.0 build on a clean
container and invoking every verb at the exact example its own `--help` documents, both fixed;
plus a sweep test driven off the CLI's own registered-verb list and `verbdoc.py` so this class of
bug cannot silently recur when a new verb is added.

### Fixed

- **`advise`, `master` and `preset` were registered with no arguments at all.** Their own
  `--help` worked examples (`aud advise in.wav`, `aud master in.wav out.wav`, `aud preset --list`)
  failed with `usage_error` -- "unrecognized arguments" -- instead of the `not_implemented`
  envelope the tool's own convention promises for an unbuilt capability (the pattern `curve
  extract`, `detect` and `strip-silence` already followed correctly). All three now parse their
  full documented argument surface -- `advise PATH`, `master IN OUT`, `preset --list` / `preset
  show NAME` -- and return `not_implemented`, same as before, but for the right reason. No DSP or
  verb behaviour changed; only the argument parser.
- **A planned DSP gap was reported as an internal aud bug.** Rendering a plan containing
  `eq_match`, `deess`, `dereverb`, `reverb`, `stretch` or `pitch` raised a bare `ValueError` from
  `dsp/engine.py`'s dispatcher, which the CLI's catch-all wrapped as `{"code": "internal_error",
  ..., "remedy": "...report it..."}` -- telling the caller to file a bug about a gap the tool
  already knew about and named in its own message. `dsp/engine.py` now raises a dedicated
  `NotImplementedStageError` (a `ValueError` subclass, so existing engine-level callers are
  unaffected); `aud.lib.render` maps it to the existing `not_implemented` code with a remedy
  naming the implemented stages. The generic catch-all is untouched and still guards genuinely
  unexpected exceptions.
- **The inverted case: a missing input file was also reported as an internal aud bug**, with a
  remedy asserting "not something wrong with your input" for exactly the case where the input
  *is* what is wrong (`aud eq-match --curve /nonexistent.json`, and the same for `analyze`,
  `verify`, `render`, `curve extract`, `curve apply`). `aud.lib` now maps a missing/unreadable
  file to `file_not_found` and an undecodable one to `audio_decode_error`, each with a remedy
  naming the path, before either can reach the catch-all.

### Added

- `aud/schemas.py::NotImplementedStageError` -- carries the stage name and the tuple of
  currently-implemented stages.
- `aud.lib.load_json_file` -- shared JSON-file read/parse helper (mapping `file_not_found` /
  `bad_param`), used by `curve_apply` and by the CLI's `eq-match --curve` handling so both get the
  same honest envelope for a missing or malformed file.
- `tests/test_verb_examples_sweep.py` -- for every verb `aud.cli.registered_verbs()` reports, runs
  the tool as a real subprocess at the exact invocation(s) `verbdoc.py` documents as correct, and
  asserts the result parses as one clean JSON document that is never a `usage_error` for a
  documented-correct invocation and never a raw traceback without `--debug`. Driven off the CLI's
  own verb registry and `verbdoc.py`, so a newly added verb is covered automatically.

## [0.3.0] - 2026-09-18

One idea enters the design: **an edit point is resolved, not taken literally.** A detector says
where a boundary *is*; where the blade should *fall* is a separate decision with its own failure
modes, and 0.2.0 conflated the two. This release names the layer that separates them, specifies
its controls on both editing stages, and registers them on the CLI. It also settles the
error-code convention that had been left contradictory between the contracts and the code.

This release is contracts, argument surface and documentation. **No edit-point resolution, fade
or crossfade code exists** — see [Verified](#verified-1).

### Added

- **Edit-point resolution, as a named concept** — `docs/ARCHITECTURE.md` §2c, "Edit points are
  resolved, not taken literally", and a shared parameter section in `contracts/plan.v1.md`. A
  region carries a nominal position; resolution moves it, within a bounded window, to somewhere
  it is safe to cut. Named rather than scattered across flags, because the alternative is six
  interacting options nobody can reason about together.
- **Eight edit-point params on both `cut` and `strip_silence`**, identical sets, specified with
  type, default, unit and validation rule:
  - `pad_out_ms` / `pad_in_ms` — programme kept either side of the removal. Padding only ever
    **shrinks** what is removed; it can never extend a cut. Defaults differ by stage on purpose:
    `0.0` for `cut` (an explicit list a caller measured and means literally), `80.0` for
    `strip_silence` (an energy threshold's boundary sits systematically *inside* the speech, so
    padding corrects a known bias).
  - `snap` — `zero_crossing` (default and floor: removes the sample discontinuity that clicks),
    `silence` (the local energy minimum — land where there is least to damage), `transient`
    (just before the nearest onset — never truncate an attack), `none` (literal).
    `zero_crossing` composes *under* the other two rather than beside them: they place the
    point coarsely, it aligns the point to a sample.
  - `snap_window_ms` — bounds the search. `> 0` and `≤ 1000.0`; outside that,
    `snap_window_invalid`.
  - `fade_out_ms` / `fade_in_ms` — for a kept boundary with no crossfade partner. Default `0.0`,
    because with a crossfade in place the seam is already handled and both would double-dip.
  - `crossfade_ms` / `crossfade_shape` — `equal_power` default, `linear` available.
- **The snap invariant, stated**: a resolved point never moves past the region's other boundary,
  into a neighbouring region, or outside the file. **If no acceptable point exists in the
  window, the resolver keeps the nominal position and records that it did** — it never widens
  the window and never quietly substitutes a rule. A snap that silently fails is worse than one
  that refuses: the file still plays, the caller believes the edit was placed well, and they
  find out after delivery.
- **The crossfade-consumes-material rule**: a crossfade needs the two kept slices to overlap by
  its length on the source timeline, and that overlap comes out of the removal. A crossfade
  longer than the removal it spans, or than the kept slice between two removals, is
  `crossfade_exceeds_gap` — an error, **not a silent clamp**. A clamp changes the sound at one
  join out of many with nothing in the output naming which one.
- **Why `equal_power` is the default**, written down rather than asserted: two uncorrelated
  signals sum in power, not amplitude, so a linear crossfade dips about 3 dB through the middle.
  `linear` is kept because the argument inverts for correlated material, where equal power bumps
  +3 dB instead.
- **A per-edit-point render report shape** in `contracts/plan.v1.md`: nominal position, padded
  position, resolved position, signed distance moved, the rule requested, the rule applied,
  whether the snap failed, and why. Plus stage-level `regions_removed`,
  `regions_dropped_by_padding` and `snap_failures`. Without this the feature is unfalsifiable —
  a snap that worked and a snap that quietly did nothing both produce a file.
- **`contracts/regions.v1.md` now carries the nominal-versus-resolved distinction** as a named
  section: a regions document holds nominal positions, resolution happens at render, the
  document is not rewritten by it, and a `resolved_s` field would be a category error because
  the resolved position depends on parameters that live in the *plan*.
- **How the two detectors compose** (`docs/ARCHITECTURE.md` §2c): onsets are an advisory
  constraint on where a silence boundary may be reported — a silence boundary next to an onset
  must not be trimmed into it — and never a source of silence regions. No algorithm specified;
  the detection functions remain explicitly not promised.
- **`dsp/resolve`** added to the module layout as the home for resolution, keeping `dsp/edit` to
  "resolved edit points in, arrays out" with no placement logic of its own.
- **CLI surface**: `--pad-out`, `--pad-in`, `--snap`, `--snap-window`, `--fade-out`,
  `--fade-in`, `--crossfade`, `--crossfade-shape` on both `cut` and `strip-silence`, registered
  through one shared helper so the two cannot drift. Value spellings are the document's
  spellings (`zero_crossing`, not `zero-crossing`) so there is no translation layer.
- **`VERB_DOCS` for `cut` and `strip-silence` rewritten** to explain each snap mode in plain
  language, the padding direction, the power-summing argument, and the report — each ending with
  a worked example ("trim the pauses without chopping the start of a word") and a reading of it
  in prose. `detect` gains a note on why onsets matter even when you only asked about silence.

### Changed

- **Error codes are lowercase `snake_case` everywhere. This is settled.** The contracts
  documented `E_PLAN_*` / `E_REGIONS_*`; the code has always raised `bad_plan`, `unknown_stage`,
  `bad_param`, `not_implemented`, `usage_error`, `internal_error`. **The code wins** — it is
  what callers observe in the JSON envelope and it is already load-bearing in the tests and the
  CLI. Both contracts were rewritten to match, and `AGENTS.md` now states the convention
  explicitly so the next contributor does not reopen it.
  - Plan: `bad_plan`, `plan_format_unsupported`, `unknown_field`, `unknown_stage`,
    `duplicate_stage`, `unknown_param`, `bad_param`, plus the new `snap_window_invalid` and
    `crossfade_exceeds_gap`.
  - Regions: `bad_regions`, `regions_format_unsupported`, `unknown_region_field`,
    `unknown_region_kind`, `bad_region_field`, `regions_out_of_order`, `regions_not_cuttable`,
    `regions_source_mismatch`, `speech_extra_missing`.
  - `E_PLAN_PARAM_TYPE` and `E_PLAN_PARAM_RANGE` **merge** into `bad_param`, and the regions
    pair into `bad_region_field`. The remedy for both was always "supply a valid value for this
    field", the message names the field and the constraint, and a caller branching on the two
    took the same branch. The regions codes stay distinct from the plan codes
    (`unknown_region_field`, not `unknown_field`) so a caller piping `detect | cut | render` can
    tell from the code alone which document was rejected.
- **`strip_silence`'s `pad_ms` is replaced by `pad_out_ms` / `pad_in_ms`.** Keeping all three
  would give one stage two overlapping ways to say the same thing with an undefined interaction.
  The CLI's `--pad` is replaced by `--pad-in` / `--pad-out`.
- **This stays `plan_format: 1`**, and the reasoning is on the record in the contract's
  versioning section rather than left to be reconstructed — including the uncomfortable part.
  Seven of the eight new params are plainly additive. The eighth change is a **rename**, which
  the contract's own list calls breaking. It stays in format 1 because the contract's *test* is
  "can a document that already exists render differently", and no released `aud` has ever
  rendered a `strip_silence` stage: such a plan does not render today and will not render
  tomorrow, so its behaviour cannot change because it has none. The rule "a rename is breaking"
  is a conservative proxy for that test; where proxy and test disagree, the test governs. **The
  escape is recorded with its expiry**: the moment `strip_silence` renders in a released
  version, it is gone, and any later rename is `plan_format: 2`.
- **Manifest** — a `use_case` for "trim the pauses without chopping the start of a word", the
  `cut` / `strip-silence` verb rows extended, and a new "An edit point is not the detector's
  boundary" section with the control table and a worked command. Version `0.3.0` here and in
  `pyproject.toml`, which a test asserts equal.
- **Chain topology diagram** now shows resolution as an explicit step between nominal positions
  and blades.

### Verified

Stated honestly. This release is design and argument-surface wiring; none of it is DSP.

**Verified, by running it:**

- `uv run ruff check --fix .` clean, `uv run ruff format .` clean, `uv run pytest -q`
  **63 passed** — the same 63 as 0.2.0. No test was added, because nothing testable was built.
- The conformance kit from `microsoft/amplifier-smart-tools` reports **16 PASS / 0 FAIL / 0 SKIP**
  against this commit, run against an installed `aud` on `PATH`.
- `aud cut` and `aud strip-silence` parse every new flag and still exit non-zero with
  `{"error": {"code": "not_implemented", ...}}` — checked by invoking each one with the full
  documented argument set. A caller writing the eventual command gets an answer about the
  capability, never a usage error about a flag that is going to exist.
- The CLI surface and `CAPABILITIES` still match exactly
  (`test_cli_surface_matches_capabilities_manifest`), and the manifest version still matches
  `pyproject.toml` (`test_manifest_version_matches_pyproject`).

**NOT implemented — no claim is made:**

- **No edit-point resolution code exists.** There is no zero-crossing search, no energy-envelope
  minimum, no onset avoidance, no window clipping, no invariant enforcement. `dsp/resolve` is a
  row in a table, not a file.
- **No padding, fade or crossfade code exists.** `pad_out_ms`, `fade_in_ms`, `crossfade_ms` and
  `crossfade_shape` are documented and parseable and applied by nothing.
- **No render report has ever been produced for an edit point.** The report shape in
  `contracts/plan.v1.md` was written by hand. Nothing has round-tripped through it, and
  `snap_failures` has never been a number computed from anything.
- **The defaults are unmeasured.** `snap_window_ms: 20.0`, `pad_*_ms: 80.0` for `strip_silence`,
  `crossfade_ms: 10.0` — none has been tried against audio. They are reasoned defaults, and the
  first implementation may well find them wrong.
- **The error codes are documented, not raised.** `snap_window_invalid`,
  `crossfade_exceeds_gap`, `regions_not_cuttable`, `bad_regions` and the rest of the renamed set
  appear in the contracts; the only codes any code path emits today are still `not_implemented`,
  `bad_param`, `bad_plan`, `unknown_stage`, `usage_error`, `internal_error`, `bad_manifest` and
  `manifest_missing`. The convention is now consistent; the coverage is not yet complete.
- Everything listed as unimplemented in 0.2.0 — detection DSP, editing DSP, `faster-whisper`
  behaviour, `core/regions` — is still unimplemented and still unclaimed.

## [0.2.0] - 2026-09-18

Two capabilities enter the design: **detection and editing** (find things in a recording, then
cut them out, in the same pipeable chain), and **one command, not a conversation** — stated as a
named principle rather than left implicit in the examples. This release is contracts, ordering,
manifest and verb surface. The signal processing behind detection and editing is **not written**;
see [Verified](#verified) below.

### Added

- **Regions document contract, format 1** — `contracts/regions.v1.md`. A second document
  contract alongside the plan: what `aud detect` emits and what `aud cut` consumes, which is
  what lets the two pipe together. Top-level shape (`regions_format`, `created_with`, `source`,
  `sample_rate`, `kind`, `detection`, `regions`), the three kinds and their per-region fields,
  ordering and disjointness guarantees, the failure codes, what is promised, what is not, and
  the rule that a breaking change means `regions_format: 2`. Versions independently of
  `plan_format`.
- **Two editing stages in `plan.v1.md`** — `cut` (an explicit list of regions, equal-power
  crossfade at every join) and `strip_silence` (a *rule*: threshold above the measured noise
  floor, minimum length, silence kept, padding, crossfade). The split is deliberate: `cut`
  carries positions and is bound to one file, `strip_silence` carries a policy and is reusable
  across a season of episodes.
- **`detect`, `cut` and `strip-silence` verbs** registered in the CLI, `CAPABILITIES` and
  `VERB_DOCS`, with their full argument surface and prose documentation. They return
  `{"error": {"code": "not_implemented", ...}}` today, as `advise`, `master` and `preset`
  already do — a caller writing the eventual command gets an answer about the capability rather
  than a usage error about a flag that will exist.
- **`speech` optional-dependency group** — `faster-whisper>=1.0,<2.0`, needed only by
  `detect fillers` for word-level timings. MIT, pip-installable with no compiler, CPU-capable,
  idempotent cached model download: an agent can install it unattended and expect it to succeed.
  A **local** model — no AI provider, no credential, no network call once cached. Declared in
  the manifest's `requires` and documented in `docs/CONFIGURATION.md`.
- **"One command, not a conversation"** as a named section in `docs/ARCHITECTURE.md`, with a
  line in `README.md` and the manifest. Three mechanisms presented as one deliberate set — the
  pipe chain, `aud master`, `aud preset show | aud render` — plus `detect | cut | render` as the
  fourth face of the same idea. The cost it avoids is stated: every agent round trip is latency,
  tokens, and another chance to lose the thread.
- **`docs/VISION.md`** gains a section on cleanup of spoken-word recordings belonging in the
  same chain as mastering rather than in a separate tool — because a loudness target measured
  before an external tool cuts the file is a number describing a file that no longer exists.
- **A test that guards the ordering's reason**, not only its spelling:
  `test_editing_stages_render_before_any_timeline_or_measurement_stage`.

### Changed

- **Canonical stage order now has thirteen entries**, with `cut` and `strip_silence` at the
  **front**, ahead of repair and tone — and ahead of `stretch`, which is itself a timeline
  change. Two reasons, either sufficient: every measurement downstream is a measurement of a
  timeline (loudness is an average over duration), and region positions are offsets into the
  *source* timeline. Updated in `contracts/plan.v1.md`, `src/aud/plan.py`, `docs/ARCHITECTURE.md`,
  `README.md` and the manifest together.
- **This stays `plan_format: 1`**, and the reasoning is on the record in the contract's
  versioning section rather than left to be reconstructed. The test is not "did the array
  change" but "can a document that already exists render differently": no pre-0.2.0 plan can
  contain `cut` or `strip_silence` (unknown stage names are rejected), and the relative order of
  the eleven pre-existing names is untouched, so every stored format-1 plan sorts and renders
  exactly as before. The versioning rules were tightened to say this precisely — *inserting* a
  new stage name is additive; *reordering names already in the format* is breaking.
- **Manifest** — `description` and `use_cases` extended so that "trim the silences", "get rid of
  the umms and ehms" and "tighten up this recording" are recognisable asks. Version 0.2.0 here
  and in `pyproject.toml`, which a test asserts equal.
- **The transcription refusal is narrowed rather than dropped.** `aud` uses speech recognition
  internally to locate filler words; it does not return transcripts. Both the manifest and
  `docs/VISION.md` now say that, because the previous blanket "do not use for speech
  transcription" would have been false the moment `detect fillers` existed.

### Fixed

- `contracts/plan.v1.md` documented `created_with` as `"aud <version>"`; the code has always
  emitted `"aud/<version>"` (`src/aud/plan.py::new_plan`). The contract is corrected to match
  the behaviour. No code changed — the document was wrong, not the tool.

### Verified

Stated honestly, because most of this release is design and one line of it is code.

**Verified, by running it:**

- `uv run ruff check --fix .` clean, `uv run ruff format .` clean, `uv run pytest -q`
  **63 passed** (62 before, plus the new ordering-reason test).
- The conformance kit from `microsoft/amplifier-smart-tools` reports **16 PASS / 0 FAIL / 0 SKIP**
  against this commit, run against an installed `aud` on `PATH`.
- `aud detect silence|transients|fillers`, `aud cut` and `aud strip-silence` parse their
  documented flags and exit non-zero with `{"error": {"code": "not_implemented", ...}}` — checked
  by invoking each one.
- The CLI surface and `CAPABILITIES` still match exactly
  (`test_cli_surface_matches_capabilities_manifest`), and the manifest version still matches
  `pyproject.toml` (`test_manifest_version_matches_pyproject`).

**NOT implemented — no claim is made:**

- **No detection DSP exists.** There is no onset detector, no noise-floor estimator, no silence
  finder. Nothing has been run against audio to see whether the parameters in the contract are
  the right parameters.
- **No editing DSP exists.** No region removal, no equal-power crossfade, no `strip_silence`.
  `render` does not know these stages; the plan can carry them and nothing applies them.
- **`faster-whisper` has not been installed or run.** The `speech` extra is declared. Its
  install behaviour on a clean machine, its model-download path, and whether the word timings it
  returns are good enough to locate a filler word are all untested. The licence (MIT) and the
  packaging claims were checked; the behaviour was not.
- **No regions document has ever been produced by this tool.** The format in
  `contracts/regions.v1.md` was written by hand. Nothing has round-tripped through it, and the
  first implementation may well find the shape awkward to write even though it reads well.
- The `dsp/detect`, `dsp/edit`, `core/regions` and `core/speech` modules named in
  `docs/ARCHITECTURE.md` do not exist as files. They are a specified layout, not a description
  of the tree.

## [0.1.0] - 2026-09-18

Initial scaffold: the package definition, the manifest, the contracts and the documentation that
describe what `aud` is. Implementation of the DSP and the CLI verbs is in progress alongside
this entry.

### Added

- **Package definition** — `pyproject.toml`, MIT, Python 3.11+, static version. Runtime
  dependencies are permissively licensed only: `numpy` (BSD), `scipy` (BSD), `soundfile`
  (BSD-3), `pyloudnorm` (MIT), `pydantic` (MIT), `pyyaml` (MIT). Optional `stretch` extra backed
  by `python-stretch` (Signalsmith Stretch, MIT).
- **Smart tool manifest** — `src/aud/SMART_TOOL.md` and the `smart-tool.json` descriptor:
  mastering and cleanup of a finished stereo or mono programme, deterministic verbs plus two
  model-backed ones (`advise`, `master --auto`), with `ai-provider` and `ffmpeg` declared as
  optional requirements.
- **Plan document contract, format 1** — `contracts/plan.v1.md`. Top-level shape, the eleven
  canonical stage names and their order, the parameter schema of every stage, the failure codes,
  what is promised, what is not, and the rule that a breaking change means `plan_format: 2`.
- **Documentation** — `docs/VISION.md` (the problem, the audience, the licence-over-convenience
  decision and its cost, what the tool refuses to do), `docs/ARCHITECTURE.md` (plan document as
  the central contract, canonical ordering, library-is-the-tool, the single-render rule, chain
  topology, module layout, where intelligence attaches), `docs/CONFIGURATION.md` (four-tier
  settings resolution, inverted credential resolution, the absent/null/wrong-type distinction).
- **Repository conventions** — `AGENTS.md` and `CONTRIBUTING.md`: the forbidden GPL-family
  dependencies named explicitly, the zero-credential requirement and how it is proven, the
  error-with-remedy shape, the ban on hardcoded model names, and the lint/format/test commands.
- **CI** — three jobs: the test suite with an explicit assertion that no provider credential is
  present in the environment; the external conformance kit run against this commit; and an
  install-from-git job, which is the only job that would notice a git reference that works
  locally and nowhere else.
- **MIT licence**, `.gitignore`, `.python-version` pinned to 3.11.

### Verified

Stated honestly, because the code is being written alongside these documents.

**Established by research, before any of it was written down here:**

- The mature ready-made options are GPL-family and therefore excluded on purpose: `pedalboard`
  is GPL-3.0, `matchering` is GPL-3.0, Rubber Band is GPL-2.0-or-later with a commercial dual
  licence. The permissive alternatives (`numpy`, `scipy`, `soundfile`, `pyloudnorm`,
  `python-stretch`) are what the dependency list is built from.
- Topology facts the architecture rests on: Linkwitz-Riley 4th-order crossovers recombine flat;
  a single full-band limiter after recombination is the only place a true-peak ceiling can be
  guaranteed, because independently limited bands can sum above it; true peak requires
  oversampling to detect; loudness normalisation is gain only and guarantees no ceiling; EBU R128
  recommends a maximum true peak of −1 dBTP.

**Not yet verified — no claim is made:**

- The test suite has not been run. There is no passing-suite claim attached to this release.
- The conformance kit has not been run against this repository. `pass`/`fail` counts are
  unknown.
- CI has never executed: the workflow is written but this repository has no remote yet, so no
  job in it has produced a result.
- No audio has been rendered. No measured output — loudness, true peak, or otherwise — exists to
  compare against a target, so nothing is claimed about what the chain sounds like or whether it
  hits the numbers it is asked for.
- `uv build`, wheel installation and install-from-git have not been exercised for this package.

[Unreleased]: https://github.com/colombod/amplifier-smart-tools-audio/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/colombod/amplifier-smart-tools-audio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/colombod/amplifier-smart-tools-audio/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/colombod/amplifier-smart-tools-audio/releases/tag/v0.1.0
