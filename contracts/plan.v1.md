# Contract: the mastering plan document, format 1

This is the document `aud` passes between verbs on stdin and stdout, and the thing another
program may generate, parse, store or diff. It is a contract: what is written here is what a
caller may rely on.

Status: **plan_format 1**. Scope: what the document *is*. What the stages *sound like* is not
part of this contract — see [Not promised](#not-promised).

`aud` has a second document contract, the [regions document](regions.v1.md): where things are in
a file, as emitted by `aud detect` and consumed by `aud cut`. The two version independently.

## The document

```json
{
  "plan_format": 1,
  "created_with": "aud/0.3.0",
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
| `created_with` | string | yes | `"aud/<version>"`, e.g. `"aud/0.3.0"`. Provenance for a human reading a stored plan; never used to decide behaviour. |
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
["cut", "strip_silence",
 "stretch", "pitch", "gate", "expand", "dereverb", "deess", "eq", "eq_match",
 "compress", "saturate", "reverb", "loudness", "limit", "downmix", "resample"]
```

`render` applies stages in this order regardless of the order they appear in `stages`. The array
order records how the plan was assembled; it does not control processing. The render report
states the order actually applied.

### Why `gate`/`expand` sit ahead of `compress`, not after it

`gate` and `expand` remove or reduce what is quiet; `compress` (multiband dynamics) raises what
is quiet toward its threshold along with everything else it processes. Placed after `compress`,
a gate would be gating a noise floor the chain itself had already lifted \u2014 the two stages would
be fighting each other on the exact material each one is defined by. Placed before it, `compress`
receives a programme whose dead air is already under control, which is the only order in which
neither stage undoes the other's work.

`dereverb`/`deess` are narrowband, surgical repairs (an estimated reverb tail, a sibilant band)
that do not materially change the overall noise floor, so their relative position to
`gate`/`expand` is not load-bearing the way the position relative to `compress` is. They sit
immediately after `gate`/`expand` so that repair work happens once, together, ahead of tone and
dynamics \u2014 continuing the same principle that puts editing ahead of everything: process the
timeline/level problems before stages that measure or shape what's left of them.

### Why editing is first

`cut` and `strip_silence` remove material. They are at the **front** of the order, ahead of
repair and tone, and that position is not a preference — it is the only position at which the
rest of the chain is describing the file that will actually exist.

Two separate reasons, both of which bite:

1. **Every measurement downstream is a measurement of a timeline.** Integrated loudness is an
   average over duration; the limiter's ceiling is enforced against the peaks present. If
   `loudness` targets −14 LUFS over material that `cut` later removes, the number it hit
   describes a file that no longer exists, and the delivered file misses the target it was
   asked for. `verify` would then honestly report a failure that the plan looks like it should
   have passed.
2. **Region positions are offsets into the source timeline.** A `cut` region came from a
   detector that measured the original file (see [regions.v1.md](regions.v1.md)). Any stage
   that changes the timeline invalidates those offsets. `stretch` is the concrete case: it is
   a timeline change, and it sits immediately *after* the two editing stages for exactly this
   reason.

Between the two: `cut` runs before `strip_silence`. `cut` works from absolute positions measured
on the source, so it must see the source timeline; `strip_silence` detects at render time and is
therefore happy to run on whatever is left.

### Why `downmix`/`resample` sit at the very end

Both change the medium's shape rather than shape the sound: `downmix` folds the channel count,
`resample` changes the sample rate. Neither is a mastering decision, and both sit at the tail of
canonical order, after `limit` and immediately before the file is written, for the mirror image
of [why editing is first](#why-editing-is-first): every stage *before* them should measure and
process the programme's real, currently-mastered channel layout and rate. A de-esser's sibilant
band, a compressor's crossover frequencies, and the limiter's oversampling ratio are all defined
against the sample rate of the material actually being processed at that point in the chain;
folding channels or changing rate any earlier would mean those stages are shaping a signal that
is not the one the caller is asking to master.

`downmix` precedes `resample` in the canonical order for efficiency only (resampling fewer
channels costs less), not correctness: `resample_poly`'s anti-alias filter is applied identically
to every channel, so folding channels before or after resampling produces the same result to
floating-point precision. Either ordering is mathematically sound; only one is cheaper.

Stage names in the document use underscores; the CLI verb that appends them may not. `aud
eq-match` appends the stage `eq_match`. The document's spelling is the contract.

At most **one entry per stage name**. A plan carrying two `eq` entries is rejected
(`duplicate_stage`) rather than merged or last-one-wins, because both of those rules are
guesses about intent. Every stage's parameters are expressive enough to say the whole thing in
one entry — an EQ with four bands is one `eq` stage with four `peaks`.

## Stage parameters

Units are in the field names: `_db` decibels, `_dbtp` decibels true peak, `_hz` hertz, `_ms`
milliseconds, `_s` seconds, `_lufs` LUFS. All numbers must be finite; `NaN` and `Infinity`
(which are not JSON anyway, only extensions some encoders emit) are rejected.

### `cut`

Remove an explicit list of regions from the programme. The list is normally piped in as a
[regions document](regions.v1.md); the stage stores it.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `regions` | array of object | required | Each: `{"start_s": float ≥ 0, "end_s": float > start_s}`. Ascending by `start_s` and non-overlapping. May be `[]`, which renders unchanged. |
| `filler_tail_pad_ms` | float | `200.0` if the piped-in document's `kind` was `"filler"`, else `0.0` | `≥ 0`, finite. Extends every region's stored `end_s` by this many ms (clamped at build time to not cross into the next region) before it is written into `regions` above. Recorded here for transparency/reproducibility; not itself replayed at render time. |

**Plus the whole edit-point resolution set** — `pad_out_ms`, `pad_in_ms`, `snap`,
`snap_window_ms`, `fade_out_ms`, `fade_in_ms`, `crossfade_ms`, `crossfade_shape` — defined in
full under [Edit-point resolution](#edit-point-resolution-shared-by-cut-and-strip_silence).

#### Why `filler_tail_pad_ms` exists, and why it is not padding

`aud detect fillers` (faster-whisper) reports a filler word's END timestamp systematically
175-200 ms **early** — it closes the word before the vowel actually decays. Measured against
exact ground truth: "um" truth `0.599-0.938s`, returned `0.600-0.740s` (end error -198.4 ms);
"uh" truth `2.765-3.154s`, returned `2.800-2.980s` (end error -174.2 ms). Start timestamps are
accurate to within tens of milliseconds; the bias is specific to the end boundary.

Left uncompensated, `cut` removes exactly the reported (too-short) span and an audible remnant
of the filler survives on playback — "um" becomes "hmm" rather than disappearing.

**This is deliberately not done via `pad_out_ms`/`pad_in_ms`.** [Padding](#padding) can only ever
*shrink* what is removed — "a control that could also remove more would not be safe to reach
for" is the whole point of that design, and reusing it here would invert it. Compensating a
recogniser's known-early boundary needs the opposite: an *extension* past the reported position.
So `filler_tail_pad_ms` instead extends the raw `end_s` of each region, per region, at the one
point `cut` still knows the document's `kind` — before padding/snap ever run — and does not
change `pad_out_ms`/`pad_in_ms`'s defaults or shrink-only meaning for `cut` in general, or for any
other kind of region (`silence` regions, or a hand-authored literal position, are unaffected: the
default is `0.0` unless the document's `kind` is `"filler"`). A caller that knows better can
always override it, including with `0.0` to disable it entirely.

`start_s` and `end_s` are offsets into the **source** file's timeline, which is why `cut` is
first in canonical order. A plan containing a `cut` stage is bound to the file its regions were
measured on; rendering it against a different file removes the wrong audio. That binding is a
property of the stage, not a defect of it — see
[Positions are bound to one file](regions.v1.md#positions-are-bound-to-one-file).

Those positions are **nominal**. `cut` does not take them literally: each one is resolved into
an edit point before anything is removed, by the rules below. A join with `crossfade_ms: 0` and
`snap: "none"` is a hard splice at an arbitrary sample value and will click.

### `strip_silence`

Remove or shorten the silences in the programme. Unlike `cut`, this stage carries **no
positions**: it detects at render time, using the parameters below.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `threshold_above_floor_db` | float | `6.0` | `≥ 0`. How far above the file's **measured** noise floor still counts as silence. |
| `min_len_ms` | float | `400.0` | `> 0`. Silences shorter than this are left alone. |
| `keep_ms` | float | `150.0` | `≥ 0`. How much silence is left behind in place of each removed one. `0.0` removes it entirely. |

**Plus the whole edit-point resolution set** — `pad_out_ms`, `pad_in_ms`, `snap`,
`snap_window_ms`, `fade_out_ms`, `fade_in_ms`, `crossfade_ms`, `crossfade_shape` — defined in
full under [Edit-point resolution](#edit-point-resolution-shared-by-cut-and-strip_silence). The
boundaries this stage detects at render time are nominal in exactly the same sense as the ones
`cut` is handed, and are resolved by exactly the same rules.

The threshold is relative to the measured noise floor rather than an absolute dBFS value,
because an absolute value is correct for exactly one recording — the one it was tuned on. The
argument in full, with the two ways a fixed number fails, is in
[regions.v1.md](regions.v1.md#why-the-silence-threshold-is-relative).

### Edit-point resolution (shared by `cut` and `strip_silence`)

A detector reports where a boundary **is**. Where the blade should **fall** is a separate
decision, and both ways of getting it wrong are audible: a blade through the attack of a word or
a note truncates it, and a blade at a non-zero sample value clicks.

So every position an editing stage receives is **nominal**, and both editing stages resolve each
nominal position into an **edit point** before removing anything. Two edit points per region:
`start`, where removal begins and the outgoing kept slice ends, and `end`, where removal ends
and the incoming kept slice begins.

Resolution runs in a fixed order, and the order matters because the steps compose:

1. **Pad** — move each point by its padding. Arithmetic, no search.
2. **Snap** — move the *padded* point, within `snap_window_ms` of it, to somewhere it is safe to
   cut. Bounded and refusable.
3. **Validate the joins** — check each crossfade has the material it needs.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `pad_out_ms` | float | `cut`: `0.0` · `strip_silence`: `80.0` | `≥ 0`, finite. Programme kept at the **end of the outgoing slice**: the removal starts this much later. |
| `pad_in_ms` | float | `cut`: `0.0` · `strip_silence`: `80.0` | `≥ 0`, finite. Programme kept at the **start of the incoming slice**: the removal ends this much earlier. |
| `snap` | string | `"zero_crossing"` | Exactly one of `"zero_crossing"`, `"silence"`, `"transient"`, `"none"`. Anything else: `bad_param`. |
| `snap_window_ms` | float | `20.0` | `> 0` and `≤ 1000.0`. How far a point may be moved from its padded position. Outside that range: `snap_window_invalid`. |
| `fade_out_ms` | float | `0.0` | `≥ 0`, finite. Fade-out at a kept boundary that has **no crossfade partner**. |
| `fade_in_ms` | float | `0.0` | `≥ 0`, finite. Fade-in at a kept boundary that has no crossfade partner. |
| `crossfade_ms` | float | `10.0` | `≥ 0`, finite. Length of the crossfade at a join between two kept slices. Longer than the material available: `crossfade_exceeds_gap`. |
| `crossfade_shape` | string | `"equal_power"` | Exactly one of `"equal_power"`, `"linear"`. Anything else: `bad_param`. |

The two stages take the identical set. **The padding defaults differ, and deliberately.** `cut`
is handed an explicit list that a caller measured and means literally, so widening someone's
stated edit by 80 ms unasked would be a surprise — it defaults to none. `strip_silence` finds
its own boundaries from an energy threshold, and an energy threshold's boundary sits
*systematically inside* the speech: the tail of a word crosses the threshold while the word is
still going. Padding is the correction for a known bias, so it is on by default there.

#### Padding

`pad_out_ms` moves the `start` point later; `pad_in_ms` moves the `end` point earlier. Both
therefore **shrink what is removed, and neither can ever extend a cut**. That direction is the
whole point: padding is the control you reach for when speech sounds clipped at a join, and a
control that could also remove *more* would not be safe to reach for.

If `pad_out_ms + pad_in_ms` is at least the length of the region, padding has consumed the whole
removal. The region is then **not removed**, and the render report records it as dropped with
that reason. It is not an error: on `strip_silence` a marginal silence being padded back out of
existence is the parameters working as asked, and failing the entire render over one of them
would be worse than reporting it. It is not silent either — a caller that wanted those regions
gone can see exactly which ones survived and why.

#### Snap

`snap` says what a point is moved *towards*. `snap_window_ms` bounds how far.

| `snap` | What it does | What it protects against |
|---|---|---|
| `"zero_crossing"` | Move to the nearest zero crossing. **The default, and the floor.** | The sample discontinuity at a splice, which is a click. Costs nothing: sub-millisecond, always available in programme material. |
| `"silence"` | Move to the quietest place in the window — the local minimum of the short-time energy envelope — then align that to the nearest zero crossing. | Cutting through something loud. The blade lands where there is least to damage. |
| `"transient"` | Move to just **before** the nearest onset in the window, then align to the nearest zero crossing on the earlier side. | Truncating an attack. A sound that has begun and is then cut off reads as a glitch; removing it whole does not. |
| `"none"` | Take the nominal (padded) position literally. No search, and no zero-crossing alignment either. | Nothing. It exists so a caller who has already chosen exact sample positions can have them honoured. |

Zero-crossing alignment is a **floor under the other two, not a peer of them**. `silence` and
`transient` decide *where* the point belongs, coarsely; zero crossing decides how it is finally
aligned, at sub-millisecond scale. They answer different questions, so composing them is not a
fallback — and `none` is the only value that turns the alignment off, because it is the only
value that says the caller has already decided.

Because zero crossing is the floor, it runs even when `silence` or `transient` cannot find a
coarse candidate. **A failed coarse search does not fall all the way to the raw, unaligned
position — it falls one level, to the floor.** When that happens, `rule_applied` is
`"zero_crossing_fallback"`: the rule that actually ran, distinct from both the rule requested and
from `"none"`. `"none"` is reserved for the caller's literal `snap: "none"` and is never used to
report an outcome — conflating an instruction with a failure is what let an unaligned splice ship
silently. Only if the fallback *also* finds nothing — the window itself never changes sign — does
a point take the raw padded position, and that state is reported as `"unaligned"`, with
`snap_failed` always `true`: there is no floor under the floor.

`transient` always moves a point **earlier, never later**. The failure it exists to prevent is a
sound that has already started being cut off part-way through, and the fix for that is always to
place the blade before the sound began, at either boundary. At `end` that shortens the removal;
at `start` it lengthens it — removing a whole sound rather than leaving a truncated fragment of
one. Transient detection earns its place beside silence detection for exactly this: without
knowing where the attacks are, there is no way to avoid landing on one.

**`transient` is asymmetric across the two boundaries it snaps, and that asymmetry matters for
`snap_failed`.** At `end` — the resume point — a miss is a genuine failure: a real attack could be
sitting just past the search window, and truncating it is exactly what this rule exists to
prevent, so a miss there sets `snap_failed: true`. At `start` — the trailing edge into whatever is
being removed — the search window never extends past this region's own end (see
[the snap invariant](#the-snap-invariant)), so there is structurally nothing upstream of the
boundary left to protect; a miss there is the expected outcome, not a failure, and `snap_failed`
stays `false`. Both cases still get `rule_applied: "zero_crossing_fallback"` (or `"unaligned"` if
even the floor finds nothing) — only `snap_failed` differs between them. Without this distinction,
`snap_failed` fires on essentially every silence-region start (there is rarely a new sound
beginning at the trailing edge into a gap), training a caller to ignore the flag — and a caller
that ignores `snap_failed` will not notice the boundary where it actually matters.

#### The snap invariant

A resolved point must satisfy **all** of these. The search window is the intersection of them:

1. It stays inside the window: `|resolved − padded| ≤ snap_window_ms`.
2. It never crosses the region's **other** boundary. The removal stays non-empty and never
   inverts.
3. It never moves into a **neighbouring region**. Regions are sorted and non-overlapping
   (see [regions.v1.md](regions.v1.md)); the window is clipped at the neighbours on both sides.
4. It never leaves the file: `0 ≤ resolved ≤ duration`.

**If no acceptable point exists inside that window, the resolver keeps the position it had and
records that it did.** It does not widen the window. It also does not quietly stay unaligned when
zero crossing itself could still run — zero crossing is the floor under `silence` and `transient`
(see [Snap](#snap)), so a failed coarse search still falls to the floor rather than to the raw
position. The record is `"snap_failed": true` on that edit point (except at a `transient`
`start` boundary, where the rule does not apply by construction — see [Snap](#snap)), with the
rule that actually ran (`rule_applied`, which may legitimately differ from `snap_requested`) and a
reason. Only when the floor itself finds nothing does a point take the raw, unaligned position.

A snap that silently fails is worse than one that refuses, because the caller believes the edit
was placed well and the file says otherwise — and it says so only on playback, after delivery.

#### Fades and crossfades

Two different jobs, and they do not overlap:

- `crossfade_ms` / `crossfade_shape` govern a **join between two kept slices** — the seam a
  removal creates.
- `fade_out_ms` / `fade_in_ms` govern a kept boundary with **no partner to cross into**: the
  head of the programme, its tail, or any join where `crossfade_ms` is `0`. They default to `0`
  because with the default crossfade in place the seam is already handled, and applying both
  would attenuate the join twice.

**`equal_power` is the default because two uncorrelated signals sum in power, not in
amplitude.** Under a linear (equal-gain) crossfade, both sides are at 0.5 in the middle, so the
summed power is half of either side's — an audible dip of about 3 dB right through the join.
Equal power holds the sum of the *squared* gains constant instead, so perceived loudness stays
flat across the seam.

`linear` is available rather than merely documented as a mistake, because the argument inverts
when the two sides are **correlated** — a crossfade over a continuous tone, or over the same
material offset by a few samples. Correlated signals sum in amplitude, so equal power *bumps* by
about 3 dB where linear is flat. Two cases, two shapes; the default is the one that matches what
a splice in a programme usually joins.

**A crossfade consumes material on both sides of the join.** To cross the outgoing slice into
the incoming one over `crossfade_ms`, both must still exist for that long across the seam —
which means the two kept slices must **overlap by the crossfade length** on the source timeline,
and that overlap can only come out of the material being removed.

Two consequences, and both are errors rather than clamps:

- `crossfade_ms` greater than the **length of the removal** it spans would consume kept audio on
  the far side — audio nobody asked to touch. `crossfade_exceeds_gap`.
- `crossfade_ms` greater than the **kept slice between two removals** would make two crossfades
  overlap each other, leaving a slice that is entirely crossfade. `crossfade_exceeds_gap`.

Neither is silently clamped. Clamping would change the sound the caller asked for, at one join
out of many, with nothing in the output saying which one — and the difference is audible only in
the delivered file. The failure names the region index and the length that would fit.

#### What the render report says about every edit point

The resolution is where the value of this layer is, so the report is where it is **falsifiable**.
A `cut` or `strip_silence` stage contributes an `edit_points` array to the render report, one
entry per resolved boundary:

```json
{
  "stage": "strip_silence",
  "regions_removed": 11,
  "regions_dropped_by_padding": 1,
  "snap_failures": 0,
  "edit_points": [
    {
      "region_index": 3,
      "boundary": "end",
      "nominal_s": 13.940000,
      "padded_s": 13.860000,
      "resolved_s": 13.829932,
      "moved_ms": -110.068,
      "snap_requested": "transient",
      "rule_applied": "transient",
      "snap_failed": false,
      "reason": null
    },
    {
      "region_index": 7,
      "boundary": "start",
      "nominal_s": 41.002000,
      "padded_s": 41.082000,
      "resolved_s": 41.081750,
      "moved_ms": 79.750,
      "snap_requested": "transient",
      "rule_applied": "zero_crossing_fallback",
      "snap_failed": false,
      "reason": "no onset within 20.0 ms of the padded position; not expected at a region's start boundary, used zero_crossing instead"
    }
  ]
}
```

The second entry shows the common case: a `transient` search at a `start` boundary found no
onset — expected, since the trailing edge into a removed span rarely has one — so `zero_crossing`
ran as the floor instead. `rule_applied` differs from `snap_requested` and `reason` explains why,
but `snap_failed` stays `false` because the rule did not apply here by construction (see
[Snap](#snap)). Had this been an `end` boundary instead, the same miss would have set
`snap_failed: true` while still resolving to the same fallback-aligned position.

| Field | Type | Meaning |
|---|---|---|
| `region_index` | integer | Index into the regions this stage acted on, so an entry can be traced back to one. |
| `boundary` | string | `"start"` or `"end"`. |
| `nominal_s` | float | The position as received — from the regions document, or from render-time detection. |
| `padded_s` | float | After padding, before any search. Separating the two makes it visible which control moved the point. |
| `resolved_s` | float | Where the blade actually fell. |
| `moved_ms` | float | `(resolved_s − nominal_s) × 1000`. Signed; negative is earlier. |
| `snap_requested` | string | The `snap` value asked for. |
| `rule_applied` | string | The rule that actually placed the point: one of `zero_crossing`, `silence`, `transient`, `zero_crossing_fallback` (the floor ran because the requested coarse rule found nothing), `unaligned` (nothing ran at all — the raw padded position), or `none`. Equal to `snap_requested` on a plain success. `none` appears **only** when `snap_requested` is `"none"` — it is never used to report an outcome. |
| `snap_failed` | boolean | `true` when the requested rule found nothing acceptable in the window **and the rule applied at this boundary** (see [Snap](#snap) for the one documented exception: `transient` at a `start` boundary). |
| `reason` | string or null | Non-null when `snap_failed` is `true`, when an invariant clipped the window, or when `rule_applied` differs from `snap_requested` (a fallback ran, whether or not it counted as a failure). |

Without this, smart placement is unfalsifiable: a caller has no way to tell a snap that worked
from a snap that quietly did nothing, because both produce a file. `snap_failures` is there so
that checking costs one integer rather than a scan.

**Why this stage holds a policy and `cut` holds positions.** A plan is meant to be re-run across
episode 1 and episode 40. `strip_silence` carries a rule, so it means the same thing on every
file it is applied to. `cut` carries positions, so it means something only on the file they were
measured from. Both are useful and they are deliberately not the same stage: collapsing them
would either make a reusable plan file-specific, or make an explicit edit silently
approximate.

### `stretch`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `ratio` | float | `1.0` | `> 0`. Output duration ÷ input duration; `2.0` is twice as long. |

### `pitch`

| Param | Type | Default | Constraint |
|---|---|---|---|
| `semitones` | float | `0.0` | `-24.0 ≤ x ≤ 24.0`. Positive is up. |

### `gate`

Hard noise gate: below threshold, attenuate by a fixed `range_db`; at or above it, unchanged.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `threshold_above_floor_db` | float | `12.0` | `≥ 0`. Primary threshold control: dB above this file's own measured noise floor — the same convention `strip_silence` and the `detect silence` stage use. |
| `threshold_db` | float or null | `null` | Absolute dBFS escape hatch. When set, overrides `threshold_above_floor_db` entirely. |
| `range_db` | float | `20.0` | `≥ 0`. Maximum attenuation below threshold, in dB. Not silence — a duck. `0.0` disables the stage. |
| `attack_ms` | float | `2.0` | `≥ 0`. Time to open (move toward 0 dB) once triggered. |
| `hold_ms` | float | `50.0` | `≥ 0`. Minimum time the gate stays open after last being triggered, before release may begin closing it. This is what prevents chatter on material that hovers at the threshold. |
| `release_ms` | float | `150.0` | `≥ 0`. Time to close (move toward `-range_db`) once hold expires. |
| `lookahead_ms` | float | `3.0` | `≥ 0`. How far ahead the detector looks so a fast onset's attack survives. |
| `sidechain_hpf_hz` | float or null | `80.0` | `> 0`, or `null` to disable. Highpass corner applied to the LEVEL DETECTOR only — the signal itself is unaffected — so low-frequency rumble cannot hold the gate open. |
| `crossovers_hz` | array of float | `[]` | Strictly ascending, all `> 0` and below Nyquist. `N` crossovers produce `N + 1` independently-gated bands; `[]` is full-band, the same convention `compress` uses. |

### `expand`

Soft-knee downward expander — the gentle counterpart to `gate`: below threshold, attenuate
proportionally rather than by a fixed amount.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `threshold_above_floor_db` | float | `6.0` | `≥ 0`. Same convention as `gate`'s. |
| `threshold_db` | float or null | `null` | Absolute dBFS escape hatch; overrides `threshold_above_floor_db` when set. |
| `ratio` | float | `2.0` | `≥ 1.0`. `1.0` is no expansion; `2.0` means output moves 2 dB for every 1 dB the input drops below threshold. Below `1.0` is upward expansion, a different device, and is rejected. |
| `knee_db` | float | `6.0` | `≥ 0`. Width of the soft knee centred on the threshold. |
| `attack_ms` | float | `5.0` | `≥ 0`. Time to open once triggered. |
| `hold_ms` | float | `50.0` | `≥ 0`. Same purpose as `gate`'s. |
| `release_ms` | float | `150.0` | `≥ 0`. Time to move toward the target reduction once hold expires. |
| `lookahead_ms` | float | `3.0` | `≥ 0`. Same purpose as `gate`'s. |
| `sidechain_hpf_hz` | float or null | `80.0` | `> 0`, or `null` to disable. Same purpose as `gate`'s. |
| `crossovers_hz` | array of float | `[]` | Same convention as `gate`'s / `compress`'s. |

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

### `downmix`

Fold a multichannel programme down to one channel. Takes no parameters: the fold rule (the
arithmetic mean across channels -- sum-and-divide, never a plain sum) and the antiphase-detection
threshold are fixed, not caller-configurable.

| Param | Type | Default | Constraint |
|---|---|---|---|
| *(none)* | | | `params` is always `{}`. |

A no-op when the input is already mono. The render report for this stage always includes
`input_channels`, the mean pairwise Pearson correlation the fold measured across channels
(`correlation`, `null` when undefined), and `antiphase_detected` -- `true` when that correlation
is at or below the antiphase threshold, meaning the fold has cancelled most of the programme's
energy rather than merely combined it. This is reported, not refused: a caller asking to downmix
genuinely out-of-phase material still gets the file, plus the evidence that something is
suspicious about the source.

### `resample`

Convert to a target sample rate.

| Param | Type | Default | Constraint |
|---|---|---|---|
| `target_hz` | integer | required | `> 0`. |

A no-op when the input is already at `target_hz`. Uses a polyphase resampler with its own
anti-aliasing filter (never hand-rolled decimation, never a filter stripped out to save time).
The render report for this stage always includes `source_hz`, `target_hz`, `input_samples` and
`output_samples`.

## Promised

Within `plan_format: 1`, a caller may rely on all of this:

1. **The three top-level fields** — `plan_format`, `created_with`, `stages` — with those names,
   those types, and those meanings.
2. **The stage entry shape**: exactly `stage` and `params`.
3. **The canonical stage names** and the fifteen-element order above. Names do not get renamed
   within format 1, and the **relative order of stage names already in the format does not
   change** within it.
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

Codes are **lowercase `snake_case`**. That is the tool-wide convention, and it is what the
process actually emits in the JSON envelope — see [AGENTS.md](../AGENTS.md).

| Code | Condition |
|---|---|
| `bad_plan` | Not JSON, not an object, a required top-level field is missing, or the JSON does not match the plan shape |
| `plan_format_unsupported` | `plan_format` is not an integer this build accepts |
| `unknown_field` | Unknown top-level field, or a stage entry with keys other than `stage` and `params` |
| `unknown_stage` | `stage` is not one of the canonical names |
| `duplicate_stage` | The same stage name appears more than once |
| `unknown_param` | A param name not defined for that stage |
| `bad_param` | A param of the wrong JSON type, non-finite, or outside the documented constraint |
| `snap_window_invalid` | `snap_window_ms` is `≤ 0` or `> 1000.0` |
| `crossfade_exceeds_gap` | `crossfade_ms` exceeds the removal it spans, or the kept slice between two removals |

`bad_param` covers both a wrong type and an out-of-range value, which the two codes here used to
split. The merge is deliberate: the remedy for both is "supply a valid value for this field",
the `message` names the field, the value found and the constraint, and a caller that branched
differently on the two would take the same branch either way.

`snap_window_invalid` and `crossfade_exceeds_gap` are separate from `bad_param` because their
remedies genuinely differ. A bad `snap_window_ms` is one number out of range and fixable from
the plan alone; `crossfade_exceeds_gap` depends on the *regions*, so its remedy is either a
shorter crossfade or a different set of regions, and the failure names which region and what
length would fit.

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
  behaviour, or a **new stage name inserted anywhere in the canonical order** — stays in format 1
  and is documented in this file and in `CHANGELOG.md`.
- **Breaking change** — removing or renaming a field, changing a type, changing a default,
  **reordering stage names that already exist in the format**, or changing rejection semantics —
  means a **new integer**: `plan_format: 2`, specified in a new `contracts/plan.v2.md`. This file
  is not edited in place to describe different behaviour, because stored plans and third-party
  generators still read it.

### On the record: why `cut` and `strip_silence` did not move the integer

0.2.0 added two stage names at the **front** of the canonical order. That looks like the
"changing the order" case, and it is worth being explicit about why it is not, because the test
is not "did the array change" but "**can a document that already exists render differently**".

- No plan written before 0.2.0 can contain `cut` or `strip_silence` — a plan carrying an unknown
  stage name is rejected, so such a document was never producible.
- The relative order of the eleven pre-existing names is untouched. `stretch` still precedes
  `pitch`; `loudness` still precedes `limit`.
- Therefore every stored format-1 plan sorts into exactly the sequence it sorted into before,
  and renders identically.

The insertion point is load-bearing for new plans and irrelevant to old ones, which is what
makes it additive. Had the two stages been inserted between, say, `compress` and `saturate` in a
way that *also* swapped two existing names, that would have been format 2 — the distinction is
in the pre-existing names, not in the array's length or the position of the new entries.

- A build states which `plan_format` integers it accepts; a plan it does not accept fails with
  `plan_format_unsupported` and a remedy, never a best-effort parse.

### On the record: why edit-point resolution did not move the integer either

0.3.0 added eight params to `cut` and `strip_silence`, and **renamed one**: `strip_silence`'s
`pad_ms` became the pair `pad_out_ms` / `pad_in_ms`. Seven of the eight are plainly additive —
new optional params with defaults, on stages whose other defaults did not move. The rename is
not, and it is worth being precise about why it still stays in format 1, because the list above
says a rename is breaking.

Apply the contract's own **test** rather than its shorthand: *can a document that already exists
render differently?*

- At the time of the rename, no released version of `aud` had ever rendered a `strip_silence`
  stage. `render` did not know it; `aud strip-silence` returned `not_implemented` and never
  appended one.
- So a plan carrying `strip_silence` with `pad_ms` was producible only by hand or by a
  third-party generator — and it did not render. Its behaviour could not change, because it had
  none.
- The eleven implemented stages were untouched, in name, type and default.

The general rule "a rename is breaking" is a **conservative proxy** for that test, and it is the
right proxy nearly always. Where the proxy and the test disagree, the test governs, because the
test is the thing the format integer actually protects.

> **That escape has expired, exactly as written.** The condition named here was: *the moment
> `strip_silence` renders in a released version, it is gone.* It renders now. From this release
> on, a stored plan carrying `strip_silence` **has** behaviour, and any rename of its params —
> or of `cut`'s — is `plan_format: 2`, with no appeal to the test above. The reasoning is kept
> on the record because it was sound when it was made, not because it still licenses anything.

`cut` keeps `crossfade_ms` under its own name and its own default of `10.0`. Nothing that a
released `aud` implements changed spelling or value.

### On the record: why `gate`/`expand` did not move the integer either

Two new stage names, inserted in the middle of the canonical order rather than at the front —
worth applying the contract's own test explicitly, since the general rule ("reordering existing
names is breaking") is a proxy for it, not the thing itself.

- No plan written before this release can contain `gate` or `expand` — an unknown stage name is
  rejected, so such a document was never producible.
- The relative order of every pre-existing name is untouched: `pitch` still precedes `dereverb`,
  `dereverb` still precedes `deess`, and so on through `limit`. `gate`/`expand` occupy a gap
  between two names that were already adjacent; no existing pair had anything inserted between
  it and a *different* existing pair moved to compensate.
- Therefore every stored format-1 plan sorts into exactly the sequence it sorted into before, and
  renders identically. Only a plan containing the two new names is affected, and no such plan
  existed until this release.

### On the record: why `downmix`/`resample` did not move the integer either

Two more new stage names, this time appended at the very **end** of the canonical order rather
than inserted in the middle -- the simplest case the contract's own test covers, but worth
naming for the same reason the two above are: consistency in how this file justifies each
addition, not just in the outcome.

- No plan written before this release can contain `downmix` or `resample` -- an unknown stage
  name is rejected, so such a document was never producible.
- The relative order of every pre-existing name is untouched: `loudness` still precedes `limit`,
  and every pair before it is exactly as it was. `downmix`/`resample` are added after the last
  existing name, not between two existing ones, so there is no gap to reason about the way
  `gate`/`expand`'s insertion needed.
- Therefore every stored format-1 plan sorts into exactly the sequence it sorted into before, and
  renders identically. Only a plan containing the two new names is affected, and no such plan
  existed until this release.
