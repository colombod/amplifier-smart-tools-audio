---
name: aud
description: >-
  Anything to do with finishing an audio file the user already has — .wav,
  .flac, .aiff, .mp3. Reach for it when the ask sounds like "make this sound
  finished", "this is too quiet for YouTube", "get rid of the harsh S sounds",
  "the room sounds boxy", "there's hiss between the sentences", "cut out the
  long pauses and umms", "make this episode match last week's", or "hit -14 LUFS
  without clipping". It measures loudness, true peak and spectral balance first,
  then runs an editing and mastering chain in one pass. This is MASTERING and
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

## Read this first

Run **`aud --help`** for the real instructions — every verb, the chain order,
prerequisites, install commands and the sharp edges. It is written for an agent
to act on, not for a human skimming a man page, so it is the source of truth
here, not this file. Then run **`aud <verb> --help`** for the full
documentation of any one verb before using it — the flags are not guessable.
Run **`aud check`** first to see what this host actually has and can reach;
never assume a capability is missing without running it.
