# Guide

Getting from a rough file to a finished one. This is the narrative path; the
[CLI reference](02-cli.md) has every flag and the [library reference](01-library.md) has the
Python surface.

![The aud mastering chain: measure, find and cut, repair, shape, finish](images/chain-01.png)

## Measure before you touch anything

```bash
aud analyze in.wav
```

One JSON document on stdout: loudness in LUFS, true peak, crest factor, per-band energy, a
sibilance estimate and an ambience estimate. Nothing is modified, nothing is spent, no
credential is needed. It is safe to call freely, including from an agent in a loop.

Advising a chain without measuring first is guessing. `analyze` turns "it sounds bad" into
numbers you can act on and cite back.

## One command, not a conversation

Get the whole job done in a **single shell command**. Do not run a stage, read the result,
decide the next one, run that — every round trip back through the caller is latency, tokens and
another chance to lose the thread, and none of it buys anything, because the whole chain can be
written down before any of it runs.

Three ways in, for three kinds of caller:

| You know | Use |
|---|---|
| what the chain should be | `aud plan \| aud eq ... \| aud render in.wav out.wav` |
| that a known-good chain exists | `aud preset show podcast \| aud render in.wav out.wav` |
| only what the result should be like | `aud master in.wav out.wav` — model-backed, decides for you |

`aud preset --list` reports the available names; `podcast`, `music-streaming`, `broadcast` and
`voiceover` ship today. Every preset is built through the same stage builders as a hand-written
chain, so it is validated identically — there is no separate, preset-only parameter path.

## Build a chain by piping

Every stage verb reads a mastering plan on stdin, appends one stage, and writes the plan back
out. Nothing touches a sample until `render`:

```bash
aud plan \
  | aud deess --amount 6 \
  | aud eq --hpf 40 --peak 3200,-2.5,1.4 \
  | aud compress --bands 120,900,5500 --ratio 2.5 \
  | aud saturate --drive 1.5 --mix 0.25 \
  | aud loudness --target -14 \
  | aud limit --ceiling -1.0 \
  | aud render in.wav out.wav
```

That is one decode, one filter graph, one encode. Do not run each stage as its own render —
every extra render is another round of quantisation and another chance to clip. Composing costs
nothing: a plan is JSON and no sample is read until `render`.

`render` applies stages in canonical order regardless of the order you appended them:

```
editing (cut, strip-silence) -> repair (gate, expand, de-ess, de-verb) -> tone (EQ, EQ-match)
  -> dynamics (multiband compression) -> character (saturation, ambience)
  -> loudness -> limiting
```

## Find things, then cut them — still one command

`aud detect` finds things and prints a **regions document**
([contracts/regions.v1.md](../contracts/regions.v1.md)) — where the silences are, where the
onsets are, where the "umm"s are. `aud cut` consumes one. So finding and removing is one
invocation, not three turns:

```bash
aud detect silence in.wav | aud cut | aud render in.wav out.wav
```

Or carry a rule instead of a list, which is reusable across every episode:

```bash
aud plan | aud strip-silence --min-len 400 --keep 150 | aud render in.wav out.wav
```

`cut` stores **positions**, which are bound to the file they were measured from.
`strip-silence` stores a **rule**, which can be re-applied to next week's recording.

Three things worth knowing:

- **Editing renders first**, ahead of repair and tone. Cutting changes the timeline everything
  downstream measures — target −14 LUFS across material a cut later removes and the number you
  hit describes a file that no longer exists.
- **The silence threshold is relative to the file's measured noise floor**, not a fixed dBFS
  value. A fixed number is right for exactly one recording: a treated room may floor at
  −70 dBFS and a phone in a kitchen at −38 dBFS, and one number finds nothing in the first file
  and eats words in the second. The measured floor is written into the regions document, so the
  judgement can be checked.
- **A detector's boundary is not where the blade falls.** The position is *nominal*; the edit
  point is resolved from it — padded, then snapped inside a bounded window to a zero crossing, a
  quiet spot, or just before an onset — because a cut through the attack of a word truncates it
  and a cut at a non-zero sample clicks. A snap that finds nothing acceptable keeps the original
  position and **says so in the report** rather than failing quietly:

  ```bash
  # trim the pauses without chopping the start of a word
  aud detect silence in.wav \
    | aud cut --pad-in 80 --pad-out 80 --snap transient --snap-window 60 \
    | aud render in.wav out.wav
  ```

  The render report carries one `edit_points` entry per boundary, each naming the nominal
  position, the padded position, the rule applied, the resolved position, how far it moved, and
  `snap_failed` when nothing acceptable was found.

`detect fillers` needs the optional `speech` extra. Without it, that one verb refuses and names
the extra — before decoding your file, not after — and it never falls back to an energy-only
guess, because those regions would look like words and `cut` would remove them.

## Keep the chain, re-use it

A plan is a JSON document ([contracts/plan.v1.md](../contracts/plan.v1.md)), so it can be saved,
read, diffed and re-run:

```bash
aud plan | aud loudness --target -14 | aud limit --ceiling -1.0 > podcast.plan.json

aud render ep-01.wav ep-01-master.wav < podcast.plan.json
aud render ep-02.wav ep-02-master.wav < podcast.plan.json
```

## Check the result against what you asked for

```bash
aud verify out.wav
```

Loudness normalisation is a gain change and does not guarantee a ceiling — that is the
limiter's job. `verify` measures the finished file so the claim is checked rather than assumed.

## Match one file to another

```bash
aud curve extract reference.wav reference.curve.json
aud plan | aud eq-match --curve reference.curve.json | aud render in.wav out.wav
```

Extracting a curve from a reference recording and applying it to this week's episode is how
"make it match last week" becomes a measurement rather than an opinion.

## Capabilities

| Verb | | What it does |
|---|---|---|
| `analyze` | deterministic | What is actually in the file: LUFS, true peak, crest, bands, sibilance, ambience |
| `detect transients` `detect silence` | deterministic | Where things are: onsets, quiet spans. Emits a regions document, not a plan |
| `detect fillers` | deterministic, needs `aud[speech]` | "umm", "uh", "ehm" and long hesitations, with word-level timings |
| `plan` | deterministic | Start an empty chain, or load one from a file |
| `cut` | deterministic | Editing stage: remove a listed set of regions, crossfaded at every join |
| `strip-silence` | deterministic | Editing stage: remove or shorten the silences, by a rule rather than a list |
| `gate` `expand` | deterministic | Repair stage: remove or reduce what sits below the floor |
| `deess` `dereverb` | deterministic | Repair stage: sibilance and room |
| `eq` `eq-match` `curve` | deterministic | Tone stage, including extracting a curve from one file and applying it to another |
| `compress` | deterministic | Multiband compression and dynamic range control |
| `saturate` `reverb` | deterministic | Character stage |
| `stretch` `pitch` | deterministic | Retime or re-pitch |
| `loudness` `limit` | deterministic | Loudness target and true-peak brickwall ceiling |
| `render` | deterministic | Apply the whole chain in one pass |
| `verify` | deterministic | Measure a render against the targets it was asked for |
| `preset` | deterministic | Named chains for common destinations |
| `check` `config` `manifest` | deterministic | What this host has, effective settings, the manifest |
| `advise` | **model-backed** | Read the measurements and say what the chain should be, and why |
| `master` | **model-backed** | Choose the chain, apply it, and verify the result |

## Deterministic paths need no credentials

Everything marked deterministic above runs with no AI provider configured and no credential
present. It spends nothing. That includes `detect fillers`: the `speech` extra is a **local**
model, not a provider — it needs an install, never a credential.

Only `advise` and `master --auto` need a provider. Without one they refuse and say so, naming
the remedy, rather than guessing a chain — and they refuse *before* decoding your file, not
after. Any one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY` or
`AZURE_OPENAI_API_KEY` satisfies the requirement; Azure additionally needs
`AZURE_OPENAI_ENDPOINT`. See [CONFIGURATION.md](CONFIGURATION.md). `aud` reads credentials; it
never writes or stores them.

Even when a model chooses the chain, **it never touches samples**. It reads measurements and
emits a plan; the same deterministic engine renders that plan as it would one you typed by hand.

## Output contract

Results go to stdout. Diagnostics, progress and the error envelope
`{"error": {"code", "message", "remedy"}}` go to stderr — so a failure never contaminates a
stream you are piping into the next verb. Exit 0 on success, 1 on a named failure, 2 on a bad
invocation.

A plan on stdout is the plan document itself, not wrapped in `{"result": ...}`, so stage verbs
pipe into each other without unwrapping — and so is a regions document, which is why `detect`
pipes straight into `cut`.

## From Python

The library is the tool. `aud.lib` holds every capability and the CLI is a thin wrapper over
it — anything available in the shell is available in Python, with the same arguments and the
same results. See the [library reference](01-library.md).
