# aud

[Branded website source and publishing guide](site/README.md)

Master and clean up audio from the command line or from Python. `aud` measures a finished
programme — loudness, true peak, crest factor, spectral balance, sibilance, ambience — finds
what is in it — onsets, silences, filler words — then runs it through an editing and mastering
chain and reports what it did.

`aud` is an [Amplifier Smart Tool](https://github.com/microsoft/amplifier-smart-tools): a
library with a manifest and a thin CLI over the top. Almost all of it is deterministic signal
processing that needs no AI provider at all; two capabilities, `advise` and `master --auto`,
call a model, and they are the only two that do.

**This is mastering and cleanup, not mixing.** It works on a finished stereo or mono file. It
has no stems, no multitrack, no panning, no bus routing.

![The aud mastering chain, animated: a waveform measured, cut, repaired, shaped and limited](docs/images/chain-animation.gif)

*One waveform through the whole chain: measured, cut, repaired, split into bands and compressed,
brought to −14 LUFS, then flattened against a −1.0 dBTP ceiling it never crosses.*

## Installation

Prerequisites:

- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [ffmpeg](https://ffmpeg.org/download.html), only for `.mp3`, `.m4a` and `.ogg` — `.wav`,
  `.flac` and `.aiff` need nothing.
- An AI provider credential, only for `advise` and `master`. Everything else runs without one.

```bash
uv tool install git+https://github.com/colombod/amplifier-smart-tools-audio
```

With the optional extras — `speech` for `aud detect fillers` (faster-whisper, MIT: a **local**
model, no provider and no credential), `stretch` for the higher-quality time-stretch engine
(Signalsmith Stretch, MIT):

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

Verify an install with `aud manifest`, which needs no credentials. Then run `aud check`: it
reports what *this* host can actually do — whether `ffmpeg` is present, which optional extras
are installed, whether any provider credential is configured — and prints the exact command
for each gap.

## Interface

```bash
# Print the tool's manifest as JSON
aud manifest

# Measure a file. Nothing is modified
aud analyze in.wav

# The whole job in one command
aud master in.wav out.wav
```

`aud --help` prints the tool's skill: every capability, install and prerequisites, the chain
order, and the sharp edges. `aud <verb> --help` documents one capability in full — whether it
is deterministic or model-backed, every argument, a worked invocation, the result, and the
failures.

Results go to stdout; diagnostics and the error envelope
`{"error": {"code", "message", "remedy"}}` go to stderr. Exit 0 on success, 1 on a named
failure, 2 on a bad invocation.

## Who it is for

- Someone with a finished file that is too quiet for a platform, too harsh, too boxy, or does
  not match the episode they released last week.
- An agent acting on that person's behalf, which needs measurements it can read and a plan it
  can inspect before anything is written.
- A product that needs all of the above under a permissive licence. `aud` is MIT, and stays MIT
  by writing its own DSP rather than depending on the mature GPL libraries in this space. See
  [docs/VISION.md](docs/VISION.md) for why that trade was made.

## Documentation

| Document | What is in it |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | Getting from a rough file to a finished one: measure, chain, cut, verify |
| [docs/01-library.md](docs/01-library.md) | The Python surface — `aud.lib` holds every capability |
| [docs/02-cli.md](docs/02-cli.md) | Every verb, every flag, the I/O and exit-code contract |
| [src/aud/SMART_TOOL.md](src/aud/SMART_TOOL.md) | The manifest — what the tool is and what it needs |
| [docs/VISION.md](docs/VISION.md) | Why this exists, what it will not do |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How it is built: plan document, chain topology, module layout |
| [docs/DESIGN-ENVELOPE.md](docs/DESIGN-ENVELOPE.md) | Patented loudness formulations to avoid, and the energy-domain approach used instead |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Settings, credentials, precedence |
| [contracts/plan.v1.md](contracts/plan.v1.md) | The plan document another program may parse |
| [contracts/regions.v1.md](contracts/regions.v1.md) | The regions document `detect` emits and `cut` consumes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, lint, test, conformance |

## Licence

MIT. See [LICENSE](LICENSE).
