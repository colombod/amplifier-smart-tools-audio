---
smart_tool_format: 1
name: aud
version: 0.11.1
description: >-
  Anything to do with finishing an audio file the user already has -- .wav, .flac, .aiff, .mp3.
  Reach for it when the ask sounds like "make this sound finished", "this is too quiet for
  YouTube", "get rid of the harsh S sounds", "the room sounds boxy", "make this podcast match
  last week's episode", "level these three files to each other", "tighten the low end",
  "hit -14 LUFS without clipping", "trim the silences", "get rid of the umms and ehms", or
  "tighten up this recording". It measures what is actually there (loudness, true peak,
  spectral balance, crest factor, sibilance, ambience), FINDS things in it (onsets, silent
  spans, filler words with their timings), then runs an EDITING and MASTERING chain over it:
  cut listed regions, strip or shorten silences, de-ess, de-verb, parametric EQ, EQ-match
  against a reference recording, MULTIBAND compression and multiband dynamic range control,
  saturation, controlled ambience, loudness targeting and true-peak brickwall limiting. Every
  CHAIN STAGE (cut, strip-silence, gate, expand, deess, dereverb, eq, eq-match, compress,
  saturate, reverb, stretch, pitch, loudness, limit) appends to a plan, and one render applies
  the whole chain in a single pass, so a chain can be inspected and re-run, and the whole job is
  ONE shell command rather than a round trip per stage. The read-only/reporting verbs (analyze,
  detect, verify, check, config, manifest) and the plan-lifecycle verbs (plan, preset) do not
  append to a plan themselves -- see the verb table below for which is which. This is mastering
  and cleanup, NOT mixing: it works on a finished stereo or mono programme, not on multitrack
  stems. It uses speech recognition internally to locate filler words, but it does NOT produce
  transcripts -- do NOT use it for video files, for transcription as an output, or for music
  generation.
use_cases:
  - Measure a file honestly before touching it -- LUFS, true peak, crest factor, spectral balance
  - Bring a podcast or music file to a loudness target without clipping or inter-sample peaks
  - Trim the silences out of a spoken-word recording, or shorten them to a fixed length
  - Trim the pauses without chopping the start of a word -- edits are padded, snapped to a safe
    cut point and crossfaded, rather than falling wherever the detector put the boundary
  - Get rid of the "umms" and "ehms" and long hesitations in an interview or a voice-over
  - Tighten up a rambling recording -- find the pauses and fillers, cut them, master the result
  - Find where the onsets are in a recording, to decide where an edit should land
  - Tame harsh sibilance on a vocal or spoken-word recording
  - Reduce a boxy or reverberant room on a recording made in the wrong space
  - Make one recording sit in the same tonal balance as a reference recording
  - Extract the EQ curve of a recording you like and apply it to another file
  - Even out dynamics per frequency band with multiband compression, not one blunt full-band squeeze
  - Retime or re-pitch a programme without changing the other
  - Verify that a finished render actually meets the loudness and ceiling it was asked for
platforms:
  - linux
  - macos
  - windows
requires:
  - name: ai-provider
    purpose: >-
      Backs the verbs that choose a chain rather than apply one -- `advise` and `master --auto`
      read the measurements and decide what the chain should be and why. Without it those two
      verbs refuse, saying so, rather than guessing a chain; every other verb -- analysing,
      building a chain by hand, rendering, verifying, extracting and applying EQ curves -- keeps
      working with no credential configured at all. Any one of ANTHROPIC_API_KEY, OPENAI_API_KEY,
      GOOGLE_API_KEY, GEMINI_API_KEY or AZURE_OPENAI_API_KEY satisfies it -- AZURE_OPENAI_API_KEY
      additionally needs the separate azure-openai-endpoint requirement below; the other four
      need nothing beyond the key itself. This tool stores no credentials of its own.
    optional: true
    install: docs/CONFIGURATION.md
  - name: azure-openai-endpoint
    purpose: >-
      Required in addition to AZURE_OPENAI_API_KEY, and only when that key is the credential in
      use: Azure OpenAI needs a resource endpoint to reach a deployment, which the other four
      accepted credentials do not. Without AZURE_OPENAI_ENDPOINT set, `advise` and `master`
      refuse with `provider_config_incomplete` even though AZURE_OPENAI_API_KEY is present.
      AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION are separate, already-defaulted
      settings, not part of this requirement. Irrelevant to every other provider and to every
      deterministic verb.
    optional: true
    install: docs/CONFIGURATION.md
  - name: faster-whisper
    purpose: >-
      Provides the word-level speech timings that `detect fillers` needs to locate "umm", "uh"
      and "ehm" in a recording. Installed as the `speech` extra of this package, not as a host
      program. Without it, exactly one capability is lost: `detect fillers` refuses, naming the
      extra. Nothing else changes -- `detect silence`, `detect transients`, `cut`,
      `strip-silence` and the whole mastering chain are unaffected, because none of them needs
      to know what was said. It is a LOCAL model: no AI provider, no credential, and no network
      call once the model is cached. faster-whisper is MIT; a GPL-family recogniser would
      relicense this tool and is excluded on purpose.
    optional: true
    install: docs/CONFIGURATION.md
  - name: ffmpeg
    purpose: >-
      Decodes and encodes the compressed formats libsndfile does not handle on its own -- mp3,
      m4a, ogg. WAV, FLAC and AIFF need nothing beyond the bundled libsndfile, so the whole tool
      works on uncompressed material with ffmpeg absent. `aud check` reports which state this
      host is in.
    optional: true
    install: https://ffmpeg.org/download.html
---

# aud

Master and clean up audio. `aud` measures a finished programme, then runs it through a
mastering chain and tells you what it did.

**The library is the tool.** `aud.lib` holds every capability; the CLI is a thin wrapper over
it. Anything you can do from the shell you can do from Python.

## Read this first: one command, not a conversation

Get the whole job done in **one shell command**. Do not run a stage, read the result, decide the
next one, run that. Every round trip back through the caller is latency, tokens, and another
chance to lose the thread — and none of them buy anything, because the chain is a document that
can be written in full before any of it runs.

Every stage verb reads a mastering plan on stdin, appends one stage, and writes the plan back
out. Nothing touches a sample until `render`. So a chain is built by piping:

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

That is one decode, one filter graph, one encode. Do not run each stage as its own render:
every extra render is another round of quantisation and another chance to clip.

There are three ways into that one command, for three kinds of caller:

| You know | Use |
|---|---|
| what you want the chain to be | `aud plan \| aud eq ... \| aud compress ... \| aud render in.wav out.wav` |
| only what the file should end up like | `aud master in.wav out.wav` — model-backed, decides for you |
| that a known-good chain exists for this destination | `aud preset show podcast \| aud render in.wav out.wav` |

Detection composes into the same idea. `detect` emits a regions document and `cut` consumes one,
so finding the silences and removing them is still **one** command, not three turns:

```bash
aud detect silence in.wav | aud cut | aud render in.wav out.wav
```

## The chain order is the point

`render` applies stages in canonical order regardless of the order you appended them, and says
so in its report:

`editing (cut, strip-silence) -> repair (de-ess, de-verb) -> tone (EQ, EQ-match) ->
dynamics (multiband compression) -> character (saturation, ambience) -> loudness -> limiting`

**Editing is first, and that is not a preference.** Cutting changes the timeline everything
downstream measures. Integrated loudness is an average over duration: target −14 LUFS across
material that a cut later removes and the number you hit describes a file that no longer exists.
Region positions are also offsets into the source timeline, so `stretch` — itself a timeline
change — has to come after the cut, not before it.

Multiband is the default for dynamics, not an option bolted on: `compress` splits at
Linkwitz-Riley crossovers, runs an independent gain computer per band, and recombines. A
single-band squeeze is what you get by asking for one band, deliberately.

## An edit point is not the detector's boundary

A detector says where a silence *is*. Where the blade should *fall* is a separate decision, and
getting it wrong is audible twice over: a cut through the attack of a word truncates it, and a
cut at a non-zero sample value clicks. So `cut` and `strip-silence` never take a position
literally — every boundary is **resolved** first, with the same controls on both verbs:

| | |
|---|---|
| `--pad-out` `--pad-in` | Keep programme either side of the removal, so speech is not clipped at the join. Padding only ever **shrinks** what is removed. |
| `--snap` | `zero_crossing` (default, removes the click), `silence` (land where there is least to damage), `transient` (land just before an attack, never through one), `none` (take the position literally). |
| `--snap-window` | How far a point may move. A snap that finds nothing acceptable inside the window **keeps the original position and says so in the report** -- it never widens the window or quietly swaps rules. |
| `--fade-out` `--fade-in` | Fades at a kept boundary with no crossfade partner. |
| `--crossfade` `--crossfade-shape` | The join itself. `equal_power` by default, because two uncorrelated signals sum in power and a linear crossfade dips ~3 dB through the middle. |

```bash
# trim the pauses without chopping the start of a word
aud detect silence in.wav \
  | aud cut --pad-in 80 --pad-out 80 --snap transient --snap-window 60 \
  | aud render in.wav out.wav
```

The render report says, per edit point, the nominal position, where it resolved to, how far it
moved, by which rule, and whether a requested snap failed.

## Straight and smart paths

| verb | | what it does |
|---|---|---|
| `analyze` | deterministic | what is actually in the file: LUFS, true peak, crest, bands, sibilance, ambience |
| `detect transients` `detect silence` | deterministic | where things are: onsets, quiet spans. Emits a regions document, not a plan |
| `detect fillers` | deterministic, **needs `aud[speech]`** | "umm", "uh", "ehm" and long hesitations, with word-level timings |
| `plan` | deterministic | start an empty chain, or load one from a file |
| `cut` | deterministic | editing stage: remove a listed set of regions. Every boundary is padded, snapped to a safe cut point and crossfaded |
| `strip-silence` | deterministic | editing stage: remove or shorten the silences, by a rule rather than a list. Same padding, snapping and crossfading |
| `deess` `dereverb` | deterministic | repair stage |
| `eq` `eq-match` `curve` | deterministic | tone stage, including extracting a curve from one file and applying it to another |
| `compress` | deterministic | multiband compression and dynamic range control |
| `saturate` `reverb` | deterministic | character stage |
| `stretch` `pitch` | deterministic | retime or re-pitch |
| `loudness` `limit` | deterministic | loudness target and true-peak brickwall ceiling |
| `render` | deterministic | apply the whole chain in one pass |
| `verify` | deterministic | measure a render against the targets it was asked for |
| `preset` | deterministic | named chains for common destinations |
| `check` `config` `manifest` | deterministic | what this host has, effective settings, this manifest |
| `advise` | **model-backed** | read the measurements and say what the chain should be, and why |
| `master` | **model-backed** | choose the chain, apply it, and verify the result |

Everything marked deterministic runs with no AI provider and no credentials, spends nothing,
and is safe to call freely. `detect fillers` is deterministic in that same sense — it needs no
provider and no credential — but it does need the optional `speech` extra installed, and it
refuses by name when that is absent rather than degrading to a guess.

## Output and failure contract

One JSON document on stdout. Success is `{"result": ...}`; failure is
`{"error": {"code", "message", "remedy"}}` with a non-zero exit. Progress and diagnostics go
to stderr, never stdout. A plan on stdout is the plan document itself, so verbs pipe into each
other without unwrapping — and so is a regions document, which is why `detect` pipes straight
into `cut`.

Two document contracts, versioned independently: the mastering plan
(`contracts/plan.v1.md`) and the regions document (`contracts/regions.v1.md`).

## What it will not do

Mixing. It takes a finished stereo or mono programme. It has no multitrack, no stems, no
panning, no bus routing. If the ask is "turn the guitar down", that is a mix change and this is
the wrong tool.

Transcription as an output. `detect fillers` runs a recogniser to find out *where* the filler
words are; the transcript is a means, and it is not returned. If the ask is "what does this say",
use a transcription tool.
