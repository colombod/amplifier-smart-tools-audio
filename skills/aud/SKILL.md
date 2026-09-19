---
name: aud
description: >-
  Anything to do with finishing an audio file the user already has — .wav, .flac,
  .aiff, .mp3. Reach for it when the ask sounds like "make this sound finished",
  "this is too quiet for YouTube", "get rid of the harsh S sounds", "the room
  sounds boxy", "there's hiss between the sentences", "cut out the long pauses",
  "get rid of the umms and ahs", "make this episode match last week's", "level
  these three files to each other", or "hit -14 LUFS without clipping". It
  MEASURES first — loudness, true peak, crest factor, spectral balance,
  sibilance, noise floor — then runs an editing and mastering chain: cut
  silences and filler words, de-ess, de-verb, noise gate, expander, EQ, EQ-match
  against a reference recording, multiband compression, saturation, controlled
  ambience, loudness targeting and true-peak limiting. Chain the verbs with
  pipes — the whole job is one pass and one shell command. This is MASTERING and
  CLEANUP, not mixing: it works on a finished stereo or mono programme. Do NOT
  use it for video files, for multitrack stems, for transcription as an output,
  or for generating music.
license: MIT
metadata:
  repository: https://github.com/colombod/amplifier-smart-tools-audio
---

# aud

Someone hands you a recording that is *nearly* right — too quiet, too harsh, too
boxy, full of pauses, or not matching the episode they released last week. `aud`
is the tool for that last mile.

**Almost all of it needs no AI provider and no credential.** The DSP is written
into the tool, on numpy and scipy. Only `advise` and `master` call a model, and
even they send only the **measurements**, never the audio.

## Install

```bash
uv tool install git+https://github.com/colombod/amplifier-smart-tools-audio
```

Filler-word detection and the higher-quality stretch engine are optional extras:

```bash
uv tool install 'aud[speech,stretch] @ git+https://github.com/colombod/amplifier-smart-tools-audio'
```

As a library — the CLI is a thin wrapper and `aud.lib` holds every capability:
`uv add "aud @ git+https://github.com/colombod/amplifier-smart-tools-audio"`.
Already installed? `uv tool upgrade aud`; `aud manifest` reports what you have.

## What needs setting up, and what it unlocks

| capability | needs |
|---|---|
| `analyze`, `detect silence`/`transients`, every chain stage, `render`, `verify`, `preset` | **nothing** |
| `.mp3`, `.m4a`, `.ogg` (`.wav`/`.flac`/`.aiff` need nothing) | **ffmpeg** |
| `detect fillers` — find the "umm"s | `aud[speech]`: a **local** 142 MB model, fetched once, no credential |
| `stretch`/`pitch` at the higher tier | `aud[stretch]` — the built-in vocoder works without it |
| `advise`, `master` | a provider key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY` or `AZURE_OPENAI_API_KEY` |

**Run `aud check` first.** It reports what THIS machine has and prints the exact
command for each gap. Never tell a user a capability is unavailable without
running it, and never silently substitute another approach for a missing tier.

## Read this before running anything

**`aud --help` prints the real instructions** — every verb, the chain order, the
sharp edges. It is written for you, not for a human skimming a man page. Read it
first, then confirm each argument with `aud <verb> --help` rather than from
memory. The flags are not guessable: `cut` alone takes padding, four snap modes,
a search window, fades and a crossfade shape.

**`aud analyze <file>` before changing anything.** It costs nothing, needs no
credential, and turns "it sounds bad" into numbers you can act on and cite back.
Advising a chain without measuring first is guessing.

## The one thing worth knowing up front

**Chain the verbs. Do not call them one at a time.**

Every verb except `render` takes a plan on stdin, appends one stage, and passes
it on. Nothing touches a sample until `render`, which applies the whole chain in
a **single pass**:

```bash
aud detect silence in.wav \
  | aud cut --snap transient --crossfade 10 \
  | aud deess --amount 6 \
  | aud compress --bands 150,1200,6000 --ratio 2.5 \
  | aud loudness --target -14 \
  | aud limit --ceiling -1.0 \
  | aud render in.wav out.wav
```

Six operations, **one decode and one encode**. Rendering between each step is the
obvious approach and the expensive one — every extra render is another
quantisation and another chance to clip. Composing costs nothing: a plan is JSON
and no sample is read until `render`.

Three ways in, for three callers: the pipe chain when you know what the file
needs; `aud preset show podcast | aud render in.wav out.wav` when you want a
known-good chain (`music-streaming`, `broadcast`, `voiceover` too); and
`aud advise in.wav` or `aud master in.wav out.wav` when you want the decision
made for you — `advise` returns a plan with a stated reason per stage, so you can
show the user why before anything is written.
