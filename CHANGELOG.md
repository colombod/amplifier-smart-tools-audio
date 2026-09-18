# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The plan document has its
own version, independent of the package version — see [contracts/plan.v1.md](contracts/plan.v1.md)
and [contracts/regions.v1.md](contracts/regions.v1.md).

## [Unreleased]

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
