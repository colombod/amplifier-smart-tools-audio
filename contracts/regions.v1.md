# Contract: the regions document, format 1

This is the second document `aud` passes between verbs on stdin and stdout, alongside the
[mastering plan](plan.v1.md). It is a contract: what is written here is what a caller may rely
on.

A regions document says **where things are in a file**. It is what every `aud detect` verb
emits, and what `aud cut` consumes. Those two facts are the whole reason it exists as a document
rather than as an internal data structure: they are what lets detection and editing compose in a
single shell command.

```bash
aud detect silence in.wav | aud cut | aud render in.wav out.wav
```

Status: **regions_format 1**. Scope: what the document *is*. How the positions in it were
arrived at — the onset detector, the noise-floor estimator, the recogniser — is not part of this
contract; see [Not promised](#not-promised).

> **Implementation status.** This contract was written down before the signal processing behind
> it, so that a generator could be built against it and so that the implementation had something
> to be wrong about. That is history now: `aud detect`, `aud cut` and `aud strip-silence` are
> implemented and render. The document below is the shape they actually produce and consume.

## The positions in here are nominal

**A regions document holds nominal positions, not edit points.** `start_s` and `end_s` say where
a detector found a boundary. They do not say where a blade should fall, and the two are not the
same number.

A detector answers a measurement question — where did the energy cross the threshold, where did
the onset function peak. Where to cut is a separate decision with its own failure modes: a blade
through the attack of a word truncates it, and a blade at a non-zero sample value clicks. Those
are properties of the material either side of the boundary, not of the boundary itself, and
neither is knowable from this document alone.

So **resolution happens at render**, inside `cut` and `strip_silence`, governed by their
`pad_*`, `snap*`, `fade_*` and `crossfade_*` params — see
[plan.v1.md](plan.v1.md#edit-point-resolution-shared-by-cut-and-strip_silence). Three things
follow, and all three are load-bearing:

1. **This document is not rewritten by resolution.** A stored regions document still says what
   was found, which is what makes it re-usable with different edit-point settings and what makes
   a detector's output checkable against the file.
2. **Nothing here needs a `resolved_s` field, and adding one would be a category error** — the
   resolved position depends on parameters that live in the plan, not on anything the detector
   measured.
3. **Where the blade actually fell is reported by `render`**, per edit point, alongside the
   nominal position it came from and the rule that moved it. That report is the join between the
   two documents, and it is in [plan.v1.md](plan.v1.md#what-the-render-report-says-about-every-edit-point).

A caller that wants the positions taken literally asks for it: `snap: "none"` with zero padding
resolves every edit point to exactly the number in this document.

## The document

```json
{
  "regions_format": 1,
  "created_with": "aud/0.3.0",
  "source": "in.wav",
  "sample_rate": 44100,
  "kind": "silence",
  "detection": {
    "threshold_above_floor_db": 6.0,
    "min_len_ms": 400.0,
    "noise_floor_dbfs": -58.3
  },
  "regions": [
    {"start_s": 12.480, "end_s": 13.940, "peak_dbfs": -54.1, "rms_dbfs": -57.8},
    {"start_s": 41.002, "end_s": 42.117, "peak_dbfs": -52.6, "rms_dbfs": -56.9}
  ]
}
```

### Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `regions_format` | integer | yes | The format version. `1` for this contract. |
| `created_with` | string | yes | `"aud/<version>"`, e.g. `"aud/0.3.0"`. Provenance for a human reading a stored document; never used to decide behaviour. |
| `source` | string | yes | The path the detector was run against, exactly as it was given. See [Positions are bound to one file](#positions-are-bound-to-one-file). |
| `sample_rate` | integer | yes | `> 0`. The sample rate of `source` at detection time. |
| `kind` | string | yes | One of `"silence"`, `"transient"`, `"filler"`. Applies to **every** region in the document. |
| `detection` | object | yes | The parameters that actually produced this document, including anything measured. Keys are fixed per `kind`, below. |
| `regions` | array | yes | Zero or more region entries. An empty array is valid and means *nothing was found* — it is not an error. |

Unknown top-level fields are **rejected**, not ignored — the same rule, for the same reason, as
the plan document. A document carrying a key this format does not define was written by
something that believed it meant something; accepting it silently would cut a file on a
description the writer did not give.

A document is **homogeneous**: one `kind`, one `source`. Region entries do not carry their own
`kind`, because a document that mixed silences and onsets would have no single meaning for `cut`
to act on. Two kinds means two documents.

### Region entries

Every region, of every kind, carries:

| Field | Type | Constraint |
|---|---|---|
| `start_s` | float | `≥ 0`, finite. Seconds from the start of `source`. |
| `end_s` | float | `≥ start_s`, finite, and `≤` the duration of `source`. |

Plus the fields its `kind` defines, below. All of them are required for that kind: a region
missing a field its kind defines is rejected rather than defaulted, because every one of those
fields is a **measurement**, and there is no honest default for a measurement that was not made.

Regions are **sorted ascending by `start_s`** and **non-overlapping**: for consecutive entries,
`regions[i].end_s ≤ regions[i+1].start_s`. Both are promises of documents `aud` emits and
requirements of documents it accepts. Overlapping regions have no single correct interpretation
under `cut` — the overlap could be removed once or twice, and both are guesses.

### `kind: "silence"`

A span quiet enough to be treated as a pause.

| Field | Type | Constraint |
|---|---|---|
| `peak_dbfs` | float | `≤ 0`. Highest sample peak inside the region. |
| `rms_dbfs` | float | `≤ peak_dbfs`. RMS level across the region. |

`end_s > start_s` — a zero-length silence is not a thing that can be found.

`detection` for this kind:

| Key | Type | Meaning |
|---|---|---|
| `threshold_above_floor_db` | float | How far above the measured noise floor still counted as silence. |
| `min_len_ms` | float | Shorter quiet spans were not reported. |
| `noise_floor_dbfs` | float | **The floor that was actually measured in this file.** |

`noise_floor_dbfs` is in the document because the threshold that produced these regions is
relative to it, and a reader that cannot see the floor cannot check the work. See
[Why the silence threshold is relative](#why-the-silence-threshold-is-relative).

### `kind: "transient"`

An onset — the instant a new sound starts.

| Field | Type | Constraint |
|---|---|---|
| `strength` | float | `> 0`. Height of the detection-function peak at this onset, in that function's own units. Comparable **within one document**, not across documents or versions. |

`end_s == start_s` for every transient region. An onset is a position in time, not a span, and
the document says so rather than inventing a width. The consequence is deliberate and is stated
under [`cut` rejects a transients document](#cut-rejects-a-transients-document).

`detection` for this kind:

| Key | Type | Meaning |
|---|---|---|
| `sensitivity` | float | The peak-picking sensitivity used; higher reports more onsets. |
| `min_gap_ms` | float | Onsets closer together than this were merged into one. |

### `kind: "filler"`

A filler word — "umm", "uh", "ehm" — or a hesitation long enough to be treated as one.

| Field | Type | Constraint |
|---|---|---|
| `text` | string | The recognised word, lowercased. For a hesitation with no word in it, the empty string `""`. |
| `confidence` | float | `0.0 ≤ x ≤ 1.0`. The recogniser's confidence in `text`. For a hesitation (`text` is `""`), `1.0` — the silence was measured, not guessed. |

`end_s > start_s`.

`detection` for this kind:

| Key | Type | Meaning |
|---|---|---|
| `words` | array of string | The filler vocabulary that was searched for. |
| `min_pause_ms` | float | Pauses at least this long were reported as hesitations. |
| `engine` | string | The recognition backend, e.g. `"faster-whisper"`. |
| `model` | string | The model identifier that backend was run with. |
| `degenerate_words_dropped` | integer, optional | How many recognised words the backend reported with `start == end` (a zero-duration word), and were therefore dropped rather than turned into an invalid `end_s > start_s` region. `0` if none were. **Optional**: a document from a build predating this field will not have it; a reader must not require it. |

`engine` and `model` are recorded because a filler document is the one kind whose contents
depend on a model, and a reader deserves to know which one. This is a **local** recogniser, not
an AI provider: `detect fillers` needs no credential and makes no network call at run time. It
needs the `speech` extra installed, and refuses by name when it is absent — see
[Producing a regions document](#producing-a-regions-document).

A word-timing backend occasionally reports a word with no duration at all. Rather than let one
such word invalidate the *entire* document (every other, correctly-timed, filler word in the
file would be lost with it), that word is dropped and counted in `degenerate_words_dropped`
instead — see [Additive, compatible change](#versioning).

## Positions are bound to one file

Every `start_s` and `end_s` is an offset into `source`, on the timeline `source` had when it was
measured. A regions document is therefore **not portable between files**, and not portable
across an edit of the same file.

Two things follow, and both are load-bearing:

1. `cut` renders against the file the regions came from. Rendering a `cut` plan against a
   different file removes the wrong audio, silently and plausibly. `source` and `sample_rate`
   are in the document so that the mismatch can be caught rather than guessed at.
2. **Every stage that alters the timeline must run after `cut`** — which is why `cut` and
   `strip_silence` sit at the front of the canonical stage order. `stretch` is the example that
   makes this concrete: it is a timeline change, and if it ran first, every position measured on
   the original file would point somewhere else. The ordering rule is in
   [plan.v1.md](plan.v1.md#canonical-stage-order).

## Why the silence threshold is relative

The threshold that decides what counts as silence is expressed as **dB above the noise floor
measured in this file**, not as a fixed dBFS number.

A fixed number is right for exactly one recording: the one it was tuned on. A quiet condenser
in a treated room may floor at −70 dBFS, where a phone recording in a kitchen floors at
−38 dBFS. A fixed −45 dBFS threshold finds nothing in the first file — every pause is "loud"
enough to keep — and swallows whole words in the second. The failure is worse than a wrong
answer: in the first case the verb reports no silences and looks like it worked, and in the
second it deletes speech.

Measuring the floor first makes the same `threshold_above_floor_db` mean the same thing in both
files: *this is quieter than this recording's own background*. The measured floor is then
written into `detection.noise_floor_dbfs`, so the judgement is reproducible by a reader who has
only the document.

## `cut` rejects a transients document

`cut` removes spans. A transient region is zero-length, so a transients document describes
nothing to remove, and cutting it would either do nothing or — if a width were invented — remove
audio nobody asked about. Piping `aud detect transients` into `aud cut` therefore fails with
`regions_not_cuttable` and a remedy, rather than succeeding vacuously.

Transients exist to be *read* — by a caller deciding where to make an edit, and later by stages
that want to know where the attacks are. A vacuous success would be the worst of the three
outcomes, because it looks like the edit happened.

## Producing a regions document

| Verb | Emits `kind` | Notes |
|---|---|---|
| `aud detect silence PATH` | `"silence"` | Deterministic. No credential, no extra. |
| `aud detect transients PATH` | `"transient"` | Deterministic. No credential, no extra. |
| `aud detect fillers PATH` | `"filler"` | Deterministic. Needs the `speech` extra. |

All three are **read-only**. They open the file, measure, and print a regions document. They do
not append to a mastering plan, they do not accept one on stdin, and they write nothing to disk.

`detect fillers` needs word-level timings, which needs speech recognition, which is the `speech`
extra (`faster-whisper`). With the extra absent it **refuses**, naming the extra and how to
install it:

```json
{"error": {"code": "speech_extra_missing",
           "message": "'detect fillers' needs word-level speech timings, and the 'speech' extra is not installed.",
           "remedy": "Install aud with the speech extra -- see docs/CONFIGURATION.md. The other detect verbs need nothing extra."}}
```

It does not degrade to an energy-only guess. A hesitation detector dressed up as a filler-word
detector would report regions the caller would reasonably assume were words, and `cut` would
remove them.

## Promised

Within `regions_format: 1`, a caller may rely on all of this:

1. **The seven top-level fields** — `regions_format`, `created_with`, `source`, `sample_rate`,
   `kind`, `detection`, `regions` — with those names, those types and those meanings.
2. **The three kind values** and, for each, the exact set of per-region fields listed above.
   Fields are not renamed, retyped or dropped within format 1.
3. **The `detection` keys documented for each kind**, including `noise_floor_dbfs` for silence
   and `engine`/`model` for fillers.
4. **Ordering and disjointness**: ascending by `start_s`, non-overlapping. Guaranteed on output,
   required on input.
5. **`end_s == start_s` for transients, `end_s > start_s` for silences and fillers.**
6. **An empty `regions` array is a successful result**, not a failure. "I looked and there was
   nothing" is an answer.
7. **Rejection, not tolerance.** Unknown top-level fields, unknown kinds, missing per-region
   fields, wrong types, out-of-range values, unsorted or overlapping regions all fail the run.
8. **The failure shape**: `{"error": {"code", "message", "remedy"}}` on stdout with a non-zero
   exit, and the codes listed below.
9. **Round-trip stability**: a regions document read and written back unchanged parses to the
   same regions with the same fields.
10. **`detect` never writes.** No verb that emits this document modifies the file it measured.

### Failure codes

Codes are **lowercase `snake_case`**, the same convention the plan document uses and the same
one the process emits in the JSON envelope — see [AGENTS.md](../AGENTS.md).

| Code | Condition |
|---|---|
| `bad_regions` | Not JSON, not an object, or a required top-level field is missing |
| `regions_format_unsupported` | `regions_format` is not an integer this build accepts |
| `unknown_region_field` | An unknown top-level field, an unknown `detection` key, or an unknown per-region field |
| `unknown_region_kind` | `kind` is not one of the three names |
| `bad_region_field` | A field of the wrong JSON type, non-finite, or outside the documented constraint (including `end_s < start_s`) |
| `regions_out_of_order` | Regions not ascending by `start_s`, or two regions overlapping |
| `regions_not_cuttable` | A zero-length region was handed to `cut` — typically a transients document |
| `regions_source_mismatch` | `cut` rendered against a file whose sample rate or duration contradicts the document |
| `speech_extra_missing` | `detect fillers` with the `speech` extra not installed |

`bad_region_field` covers both a wrong type and an out-of-range value, which the two codes here
used to split, for the same reason `bad_param` does in the plan contract: one remedy, one
branch. The codes are kept distinct from the plan document's (`unknown_region_field`, not
`unknown_field`) so that a caller piping `detect | cut | render` can tell from the code alone
which of the two documents it handed over was rejected.

Every failure names the region index and the field, and carries a `remedy` saying what a valid
value would be. The caller is usually a program; a message it cannot act on is a message that
fails twice.

## Not promised

None of this is a contract, and all of it may change in a patch release:

- **Which regions are found.** The onset detector, the noise-floor estimator and the recogniser
  will improve. The same file may yield a different set of regions between versions. If you need
  a specific set of regions to be stable forever, keep the document, not the command that made
  it.
- **Onset detection immunity to a continuous, unmodulated tone.** `detect_transients`'s
  spectral-flux detector is suited to real programme material (speech, music, foley -- anything
  with an actual noise floor and genuine onsets); it is not suited to a continuous, perfectly
  sustained, laboratory pure tone with no noise floor at all, where STFT bin-leakage drift alone
  can clear the adaptive threshold and produce a handful of spurious onsets (measured: up to 3 on
  a 3-second 440 Hz sine at the default sensitivity). This is a disclosed limit of a lightweight
  detector on an input class real recordings never actually are, not a promise it is immune to
  every input -- see `aud.dsp.detect.detect_transients`'s docstring and
  `tests/test_dsp_detect.py::test_transients_do_not_swamp_a_realistic_steady_state_background`.
- **The units and scale of `transient.strength`.** It is comparable within one document and
  nowhere else.
- **The recogniser's choice of model.** `detection.model` reports what ran; it does not promise
  that the same model will run next time.
- **Report wording.** Any prose a detect verb prints alongside its numbers is for humans and is
  not a parsing surface.
- **CLI flag names and spellings.** They map onto these fields, and the document's names are
  what is promised.
- **Anything a future format adds.** Fields not listed above do not exist in format 1.

## Versioning

- **Additive, compatible change** — a new `detection` key, or a new `kind` with its own
  per-region fields — stays in format 1 and is documented in this file and in `CHANGELOG.md`.
  It stays compatible because no document written before the change contains the new key or the
  new kind, so nothing already stored changes meaning.
- **Breaking change** — removing or renaming a field, changing a type, changing the meaning of
  `start_s`/`end_s`, or relaxing the ordering and disjointness guarantees — means a **new
  integer**: `regions_format: 2`, specified in a new `contracts/regions.v2.md`. This file is not
  edited in place to describe different behaviour, because stored documents and third-party
  generators still read it.
- `regions_format` versions **independently of `plan_format`** and of the package version. A
  regions change does not force a plan change, and neither forces the other's integer to move.
- A build states which `regions_format` integers it accepts; a document it does not accept fails
  with `regions_format_unsupported` and a remedy, never a best-effort parse.
