---
smart_tool_format: 1
name: aud
version: 0.1.0
description: >-
  Anything to do with finishing an audio file the user already has -- .wav, .flac, .aiff, .mp3.
  Reach for it when the ask sounds like "make this sound finished", "this is too quiet for
  YouTube", "get rid of the harsh S sounds", "the room sounds boxy", "make this podcast match
  last week's episode", "level these three files to each other", "tighten the low end", or
  "hit -14 LUFS without clipping". It measures what is actually there (loudness, true peak,
  spectral balance, crest factor, sibilance, ambience), then runs a MASTERING chain over it:
  de-ess, de-verb, parametric EQ, EQ-match against a reference recording, MULTIBAND compression
  and multiband dynamic range control, saturation, controlled ambience, loudness targeting and
  true-peak brickwall limiting. Every verb appends to a plan and one render applies the whole
  chain in a single pass, so a chain can be inspected and re-run. This is mastering and cleanup,
  NOT mixing: it works on a finished stereo or mono programme, not on multitrack stems. Do NOT
  use it for video files, speech transcription, or music generation.
use_cases:
  - Measure a file honestly before touching it -- LUFS, true peak, crest factor, spectral balance
  - Bring a podcast or music file to a loudness target without clipping or inter-sample peaks
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
      GOOGLE_API_KEY, GEMINI_API_KEY or AZURE_OPENAI_API_KEY satisfies it. This tool stores no
      credentials of its own.
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

## Read this first: chain the verbs, do not orchestrate them

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

## The chain order is the point

Mastering is an ordered chain. `render` applies stages in canonical mastering order regardless
of the order you appended them, and says so in its report:

`repair (de-ess, de-verb) -> tone (EQ, EQ-match) -> dynamics (multiband compression) ->
character (saturation, ambience) -> loudness -> limiting`

Multiband is the default for dynamics, not an option bolted on: `compress` splits at
Linkwitz-Riley crossovers, runs an independent gain computer per band, and recombines. A
single-band squeeze is what you get by asking for one band, deliberately.

## Straight and smart paths

| verb | | what it does |
|---|---|---|
| `analyze` | deterministic | what is actually in the file: LUFS, true peak, crest, bands, sibilance, ambience |
| `plan` | deterministic | start an empty chain, or load one from a file |
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
and is safe to call freely.

## Output and failure contract

One JSON document on stdout. Success is `{"result": ...}`; failure is
`{"error": {"code", "message", "remedy"}}` with a non-zero exit. Progress and diagnostics go
to stderr, never stdout. A plan on stdout is the plan document itself, so verbs pipe into each
other without unwrapping.

## What it will not do

Mixing. It takes a finished stereo or mono programme. It has no multitrack, no stems, no
panning, no bus routing. If the ask is "turn the guitar down", that is a mix change and this is
the wrong tool.
