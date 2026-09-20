# aud

Master and clean up audio from the command line or from Python. `aud` measures a finished
programme — loudness, true peak, crest factor, spectral balance, sibilance, ambience — finds
what is in it — onsets, silences, filler words — then runs it through an editing and mastering
chain and reports what it did. Chainable: every chain stage appends to a plan, and one `render`
applies the whole chain in a single pass.

`aud` is an [Amplifier Smart Tool](https://github.com/microsoft/amplifier-smart-tools): a
library with a manifest and a thin CLI over the top, whose model-backed capabilities sit behind
an interface. Most of it is deterministic signal processing and needs no AI provider at all;
`advise` and `master` call a model to choose a chain, and they are the only two that do.

**This is mastering and cleanup, not mixing.** It works on a finished stereo or mono file — no
stems, no multitrack, no panning, no bus routing. It is MIT, and stays MIT by writing its own
DSP rather than depending on the mature GPL libraries in this space; see
[docs/VISION.md](docs/VISION.md) for why that trade was made.

![The aud mastering chain, animated: a waveform measured, cut, repaired, shaped and limited](docs/images/chain-animation.gif)

*One waveform through the whole chain: measured, cut, repaired, split into bands and compressed,
brought to −14 LUFS, then flattened against a −1.0 dBTP ceiling it never crosses.*

## Installation

Prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [ffmpeg](https://ffmpeg.org/download.html) only for compressed formats — `.wav`, `.flac` and
  `.aiff` need nothing.
- An AI provider key only for `advise` and `master`; see
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md). Every other capability runs without one.

```bash
uv tool install git+https://github.com/colombod/amplifier-smart-tools-audio
```

With the optional extras — filler-word detection (a local speech model, no credential) and the
higher-quality time-stretch engine:

```bash
uv tool install 'aud[speech,stretch] @ git+https://github.com/colombod/amplifier-smart-tools-audio'
```

To use it as a library:

```bash
uv add "aud @ git+https://github.com/colombod/amplifier-smart-tools-audio"
```

To run it once without installing:

```bash
uvx --from git+https://github.com/colombod/amplifier-smart-tools-audio aud --help
```

To teach a coding agent how to use it, install the [skill](skills/aud/SKILL.md):

```bash
npx skills add colombod/amplifier-smart-tools-audio
```

To update:

```bash
uv tool upgrade aud
npx skills update aud   # add --global if the skill was installed globally
```

To uninstall:

```bash
uv tool uninstall aud
npx skills remove aud   # add --global if the skill was installed globally
```

Verify an install with `aud manifest`, which needs no credentials. `aud check` reports what this
host can actually do and prints the exact command for each gap.

## Interface

```bash
# Print the tool's manifest as JSON
aud manifest

# The whole job, one shell command
aud detect silence in.wav \
  | aud cut --snap transient --crossfade 10 \
  | aud deess --amount 6 \
  | aud compress --bands 150,1200,6000 \
  | aud loudness --target -14 \
  | aud limit --ceiling -1.0 \
  | aud render in.wav out.wav
```

`aud --help` is the full reference — every verb, the chain order, prerequisites and the sharp
edges — and `aud <verb> --help` documents one verb, including its result and its failures.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the plan document, the render pass and
the intelligence seam fit together, [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for settings
and credentials, and [contracts/](contracts/) for the plan and regions document formats.

## Licence

MIT. See [LICENSE](LICENSE).
