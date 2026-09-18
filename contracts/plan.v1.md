# Contract: the mastering plan document, format 1

This is the document `aud` passes between verbs on stdin and stdout, and the thing another
program may generate, parse, store or diff. It is a contract: what is written here is what a
caller may rely on.

Status: **plan_format 1**. Scope: what the document *is*. What the stages *sound like* is not
part of this contract — see [Not promised](#not-promised).

## The document

```json
{
  "plan_format": 1,
  "created_with": "aud 0.1.0",
  "stages": [
    {"stage": "deess",    "params": {"amount_db": 6.0, "freq_hz": 6500.0}},
    {"stage": "eq",       "params": {"hpf_hz": 40.0, "peaks": [{"freq_hz": 3200.0, "gain_db": -2.5, "q": 1.4}]}},
    {"stage": "compress", "params": {
      "crossovers_hz": [120.0, 900.0, 5500.0],
      "bands": [
        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0},
        {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 20.0, "release_ms": 180.0}
      ]
    }},
    {"stage": "loudness", "params": {"target_lufs": -14.0}},
    {"stage": "limit",    "params": {"ceiling_dbtp": -1.0}}
  ]
}
```

### Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `plan_format` | integer | yes | The format version. `1` for this contract. |
| `created_with` | string | yes | `"aud <version>"`, e.g. `"aud 0.1.0"`. Provenance for a human reading a stored plan; never used to decide behaviour. |
| `stages` | array | yes | Zero or more stage entries. An empty array is valid. |

Unknown top-level fields are **rejected**, not ignored. A document carrying a key this format
does not define was written by something that believed it meant something; accepting it silently
would render a chain that is not the chain the caller described.

### Stage entries

Each element of `stages` is an object with **exactly two** keys:

| Field | Type | Meaning |
|---|---|---|
| `stage` | string | One of the canonical stage names below. |
| `params` | object | Parameters for that stage. May be `{}` — every stage's parameters have defaults. |

## Canonical stage order

```json
["stretch", "pitch", "dereverb", "deess", "eq", "eq_match",
 "compress", "saturate", "reverb", "loudness", "limit"]
```

`render` applies stages in this order regardless of the order they appear in `stages`. The array
order records how the plan was assembled; it does not control processing. The render report
states the order actually applied.

Stage names in the document use underscores; the CLI verb that appends them may not. `aud
eq-match` appends the stage `eq_match`. The document's spelling is the contract.

At most **one entry per stage name**. A plan carrying two `eq` entries is rejected
(`E_PLAN_DUPLICATE_STAGE`) rather than merged or last-one-wins, because both of those rules are
guesses about intent. Every stage's parameters are expressive enough to say the whole thing in
one entry — an EQ with four bands is one `eq` stage with four `peaks`.

## Stage parameters

Units are in the field names: `_db` decibels, `_dbtp` decibels true peak, `_hz` hertz, `_ms`
milliseconds, `_s` seconds, `_lufs` LUFS. All numbers must be finite; `NaN` and `Infinity`
(which are not JSON anyway, only extensions some encoders emit) are rejected.

### `stretch`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `ratio` | float | `1.0` | `> 0`. Output duration ÷ input duration; `2.0` is twice as long. |

### `pitch`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `semitones` | float | `0.0` | `-24.0 ≤ x ≤ 24.0`. Positive is up. |

### `dereverb`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `amount_db` | float | `6.0` | `≥ 0`. Maximum reduction applied to the estimated reverberant component. |

### `deess`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `amount_db` | float | `6.0` | `≥ 0`. Maximum gain reduction in the sibilant band. |
| `freq_hz` | float | `6500.0` | `> 0`, below Nyquist. Centre of the sibilant band. |

### `eq`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `hpf_hz` | float or null | `null` | `> 0`, below Nyquist. High-pass corner; `null` is no high-pass. |
| `lpf_hz` | float or null | `null` | `> 0`, below Nyquist, and `> hpf_hz` when both are set. |
| `peaks` | array of object | `[]` | Each: `{"freq_hz": float > 0, "gain_db": float, "q": float > 0}` |
| `shelves` | array of object | `[]` | Each: `{"type": "low" \| "high", "freq_hz": float > 0, "gain_db": float, "q": float > 0}` |

### `eq_match`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `curve` | array of `[freq_hz, gain_db]` pairs | required | At least 2 points, frequencies strictly ascending, all finite. |
| `amount` | float | `1.0` | `0.0 ≤ x ≤ 1.0`. How much of the curve to apply. |
| `max_gain_db` | float | `12.0` | `≥ 0`. Clamp on any single point of the curve, in both directions. |

`curve` is the same shape `aud curve` emits when it extracts a curve from a reference file, so
the output of one can be pasted into the other.

### `compress`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `crossovers_hz` | array of float | `[]` | Strictly ascending, all `> 0` and below Nyquist. `N` crossovers produce `N + 1` bands; `[]` is single-band. |
| `bands` | array of object | required | Exactly `len(crossovers_hz) + 1` entries, low band first. |

Each band object:

| Param | Type | Default | Constraint |
|---|---|---|---|
| `threshold_db` | float | `-24.0` | `≤ 0` |
| `ratio` | float | `2.0` | `≥ 1.0`. `1.0` is no compression in that band. |
| `attack_ms` | float | `20.0` | `≥ 0` |
| `release_ms` | float | `180.0` | `> 0` |
| `knee_db` | float | `6.0` | `≥ 0` |
| `makeup_db` | float | `0.0` | any finite value |

The document is always explicit about every band. The CLI's scalar flags are a convenience over
that: `aud compress --bands 120,900,5500 --ratio 2.5` sets `crossovers_hz` to those three
frequencies and expands `--ratio` into four identical band objects. A generator writing a plan
directly does not get the shorthand and does not need it.

> Note the naming seam: the CLI flag `--bands` takes **crossover frequencies**; the document
> field `bands` holds **per-band settings**. The document's names are the contract.

### `saturate`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `drive` | float | `1.0` | `≥ 0` |
| `mix` | float | `1.0` | `0.0 ≤ x ≤ 1.0`. Dry/wet; `0.0` is bypass. |

### `reverb`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `mix` | float | `0.15` | `0.0 ≤ x ≤ 1.0` |
| `decay_s` | float | `1.2` | `> 0` |
| `predelay_ms` | float | `0.0` | `≥ 0` |

### `loudness`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `target_lufs` | float | `-14.0` | `< 0`. Integrated loudness target (ITU-R BS.1770). |

Reaching this target is a gain change. It does not guarantee a ceiling — `limit` does.

### `limit`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `ceiling_dbtp` | float | `-1.0` | `≤ 0`. **True** peak, measured oversampled, not sample peak. |
| `lookahead_ms` | float | `5.0` | `≥ 0` |
| `release_ms` | float | `50.0` | `> 0` |
| `oversample` | integer | `4` | One of `1`, `2`, `4`, `8`. |

There is one limiter and it is full-band, after the multiband compressor recombines. A plan
cannot ask for per-band limiting, because independently limited bands can sum above the ceiling
and the guarantee would be false.

## Promised

Within `plan_format: 1`, a caller may rely on all of this:

1. **The three top-level fields** — `plan_format`, `created_with`, `stages` — with those names,
   those types, and those meanings.
2. **The stage entry shape**: exactly `stage` and `params`.
3. **The canonical stage names** and the eleven-element order above. Names do not get renamed
   within format 1.
4. **Every parameter name, type, unit and default documented here.** A default may not change
   within format 1: a stored plan renders the same way with any `aud` that accepts format 1.
5. **Ordering semantics**: array order is a record of assembly; render order is canonical. At
   most one entry per stage name.
6. **Rejection, not tolerance.** Unknown top-level fields, unknown stage names, unknown params,
   wrong types, out-of-range values and duplicate stages all fail the run. `aud` never drops a
   field it does not recognise and continues, because the caller asked for something and got
   something else.
7. **The failure shape**: `{"error": {"code", "message", "remedy"}}` on stdout with a non-zero
   exit, and the codes listed below.
8. **Round-trip stability**: a plan read and written back unchanged parses to the same stages
   with the same params.

### Failure codes

| Code | Condition |
|---|---|
| `E_PLAN_MALFORMED` | Not JSON, not an object, or a required top-level field is missing |
| `E_PLAN_FORMAT_UNSUPPORTED` | `plan_format` is not an integer this build accepts |
| `E_PLAN_UNKNOWN_FIELD` | Unknown top-level field, or a stage entry with keys other than `stage` and `params` |
| `E_PLAN_UNKNOWN_STAGE` | `stage` is not one of the canonical names |
| `E_PLAN_DUPLICATE_STAGE` | The same stage name appears more than once |
| `E_PLAN_UNKNOWN_PARAM` | A param name not defined for that stage |
| `E_PLAN_PARAM_TYPE` | A param of the wrong JSON type, or non-finite |
| `E_PLAN_PARAM_RANGE` | Right type, outside the documented constraint |

Every failure names the stage and the field, and carries a `remedy` saying what a valid value
would be. The caller is usually a program; a message it cannot act on is a message that fails
twice.

## Not promised

None of this is a contract, and all of it may change in a patch release:

- **DSP coefficients and internal topology.** Filter coefficients, envelope detector maths,
  oversampling filter design, the exact shape of the saturation curve. The same plan may sound
  slightly different between versions as the implementation improves. If you need a specific
  rendering to be reproducible forever, keep the rendered file, not the plan.
- **Report wording.** The prose `render`, `analyze` and `verify` produce alongside their numbers
  is for humans and is not a parsing surface.
- **The measurement report's schema.** Analysis output is a separate surface from the plan
  document and is not covered by this contract.
- **CLI flag names and spellings.** They map onto these fields, and the document's names are
  what is promised. A flag may be renamed or gain a shorthand.
- **Ordering *within* `stages` as written by `aud`.** Append order is preserved, but a
  generator must not depend on `aud` emitting a particular order.
- **Anything a future format adds.** Fields not listed above do not exist in format 1.

## Versioning

- **Additive, compatible change** — a new optional param with a default that preserves current
  behaviour, or a new stage name appended to the canonical order — stays in format 1 and is
  documented in this file and in `CHANGELOG.md`.
- **Breaking change** — removing or renaming a field, changing a type, changing a default,
  changing the order, or changing rejection semantics — means a **new integer**: `plan_format: 2`,
  specified in a new `contracts/plan.v2.md`. This file is not edited in place to describe
  different behaviour, because stored plans and third-party generators still read it.
- A build states which `plan_format` integers it accepts; a plan it does not accept fails with
  `E_PLAN_FORMAT_UNSUPPORTED` and a remedy, never a best-effort parse.
