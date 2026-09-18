# Configuration

Two subjects, with deliberately **opposite** precedence rules:

- **Settings** — how `aud` behaves by default. Highest wins: argument → config file →
  environment → built-in default.
- **Credentials** — how `advise` and `master` reach a model provider. The order is inverted:
  environment first, then a credentials file.

Ask the tool rather than guessing:

```bash
aud config
```

`config` prints every effective setting, its value, and **where that value came from**. If a
setting is not what you expect, that report says which tier won.

```bash
aud check
```

`check` reports host capability: whether `ffmpeg` is present, whether the optional `stretch`
extra is installed, and whether any provider credential is visible. It never prints a credential
value — only whether one was found and which variable name supplied it.

## Settings

### Precedence — highest wins

| Tier | Source | Example |
|---|---|---|
| 1 (highest) | Explicit argument | `aud limit --ceiling -0.3` |
| 2 | Config file | `~/.config/aud/config.toml` |
| 3 | Environment | `AUD_DEFAULT_CEILING_DBTP=-0.3` |
| 4 (lowest) | Built-in default | `-1.0` |

The config file sits **above** the environment on purpose. A config file is a deliberate,
inspectable statement of how this machine should behave; an environment variable is often
inherited from a shell, a CI runner or a parent process that knew nothing about `aud`. The
deliberate statement should not be silently overridden by an inherited one.

Credentials invert this, for the opposite reason — see below.

### The settings

| Setting | Type | Default | Environment variable |
|---|---|---|---|
| `sample_rate_policy` | `"preserve"` or an integer in Hz | `"preserve"` | `AUD_SAMPLE_RATE_POLICY` |
| `default_ceiling_dbtp` | float, dBTP, ≤ 0 | `-1.0` | `AUD_DEFAULT_CEILING_DBTP` |
| `default_target_lufs` | float, LUFS, < 0 | `-14.0` | `AUD_DEFAULT_TARGET_LUFS` |
| `oversample` | integer, one of 1, 2, 4, 8 | `4` | `AUD_OVERSAMPLE` |
| `output_subtype` | `"PCM_16"`, `"PCM_24"`, `"PCM_32"`, `"FLOAT"` | `"PCM_24"` | `AUD_OUTPUT_SUBTYPE` |

**`sample_rate_policy`** — `"preserve"` renders at the input's sample rate. An integer resamples
to that rate. Preserve is the default because resampling is a signal-processing decision, not a
convenience, and it should be asked for.

**`default_ceiling_dbtp`** — the true-peak ceiling the limiter enforces when `--ceiling` is not
given. `-1.0` follows the EBU R128 recommended maximum true peak. This is a *true* peak in
dBTP, measured oversampled, not a sample peak in dBFS — the two are not the same number and a
file can satisfy one while violating the other.

**`default_target_lufs`** — the integrated loudness the `loudness` stage aims for when
`--target` is not given. `-14.0` is the common streaming target. Reaching a loudness target is
a gain change and does not guarantee a ceiling; the limiter does that.

**`oversample`** — the oversampling factor used to detect and control inter-sample peaks.
Higher catches more inter-sample peaks and costs more time. Below 4 the true-peak measurement
becomes an underestimate; `1` disables oversampling entirely and reduces the limiter to a
sample-peak limiter, which is almost never what you want.

**`output_subtype`** — the quantisation applied once, at write. `PCM_24` gives headroom against
requantisation on anything done downstream. Use `PCM_16` only for a final consumer deliverable
that must be 16-bit.

### The config file

Default location, TOML:

```toml
# ~/.config/aud/config.toml
sample_rate_policy = "preserve"
default_ceiling_dbtp = -1.0
default_target_lufs = -16.0
oversample = 4
output_subtype = "PCM_24"
```

`$XDG_CONFIG_HOME` is honoured where set; on Windows the equivalent location under `%APPDATA%`
is used. `AUD_CONFIG` overrides the path outright. `aud config` prints the resolved path, so
there is no need to deduce it.

A missing config file is not an error — every tier below it still resolves.

### Absent, null, and wrong-type are three different conditions

They are distinguished deliberately, and only one of them is fatal.

| Condition | Meaning | What happens |
|---|---|---|
| **Absent** — the key is not present in this tier | This tier has no opinion | Resolution continues to the next tier down |
| **Null** — the key is present with an explicit null | This tier explicitly asks for the built-in default | Resolution **stops**; the built-in default is used, and `aud config` reports it as an explicit unset |
| **Wrong type** — present, but not the declared type or outside its range | The configuration is wrong | **Fatal.** `aud` exits non-zero with `{"error": {"code", "message", "remedy"}}` naming the file, the key, the value found and the type expected |

A wrong type is never coerced and never ignored. `oversample = "four"`, `output_subtype = 24`
and `default_ceiling_dbtp = "-1.0 dB"` all stop the run. Silently falling back to a default
would mean a chain rendering with parameters nobody chose, and no indication that it happened —
the audio would be wrong rather than the command.

Null is distinct from absent so that a config file can positively state "ignore whatever is in
the environment, use the shipped default" — which absence cannot express, because absence falls
through to the environment.

## Credentials

`advise` and `master --auto` are the only verbs that call a model. Every other verb runs with
nothing configured, spends nothing, and needs nothing from this section.

**`aud` reads credentials. It never writes or stores them.** There is no `aud login`, no
keychain entry, nothing persisted by this tool. It reads the variable your provider already
documents.

### Accepted variables

Any **one** of these satisfies the requirement:

| Variable | Provider |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Google |
| `GEMINI_API_KEY` | Google Gemini |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |

These are the names each provider publishes; `aud` does not invent an `AUD_`-prefixed
credential variable, so a machine already set up for one of these providers needs no additional
setup for `aud`.

### Precedence — inverted

| Tier | Source |
|---|---|
| 1 (highest) | Environment variable |
| 2 | Credentials file, `~/.config/aud/credentials.toml`, mode `0600` |

The environment wins for credentials precisely because it loses for settings. A credential in
the environment is usually the *narrow*, deliberate one: injected by a CI secret store, a
container runtime, a secret manager or a per-session shell. A credentials file on disk is the
broad standing fallback. The specific, short-lived one should win over the general, long-lived
one — which is the opposite of the argument that makes the config file beat the environment for
settings.

### Setting up a provider

Make one of the variables above visible in the environment `aud` runs in. How is your system's
business, not this tool's: a shell profile, a `direnv` file, a systemd unit, a CI secret, a
container secret, your organisation's secret manager. `aud` reads whatever is present when it
starts.

If you would rather keep it on disk, create the credentials file yourself:

- Path: `~/.config/aud/credentials.toml` (`AUD_CONFIG`'s directory if that is set).
- Format: TOML, one key per provider variable name, value the credential string.
- Permissions: **mode `0600`** — owner read/write only. `aud` refuses to read a credentials
  file that is group- or world-readable, and says so, rather than using a credential your
  machine is handing out. Fix the mode and re-run.

Obtaining a key is the provider's process, documented by the provider. This tool does not
provision credentials.

### When no credential is present

`advise` and `master --auto` **refuse**. They exit non-zero with
`{"error": {"code", "message", "remedy"}}`, the remedy naming the accepted variables and
pointing back at this document. They do not guess a chain, and they do not silently fall back to
a preset — a chain nobody chose is worse than no chain, because it looks like a decision.

Everything else keeps working. That is the promise the tool is built around, and it is checked
rather than assumed: CI asserts that none of the five variables is present in the environment
before running the deterministic suite.

## Optional extra: speech recognition (`aud[speech]`)

Not configuration either, and **not a credential**. One verb — `aud detect fillers` — needs
word-level speech timings to locate "umm", "uh" and "ehm". Those come from `faster-whisper`, a
**local** model, installed as an extra of this package:

```bash
uv tool install 'aud[speech] @ git+https://github.com/colombod/amplifier-smart-tools-audio'
```

It is worth being precise about what this is and is not:

- **No AI provider, no credential, no network call at run time.** The model is downloaded once
  and cached; after that `detect fillers` runs offline. It has nothing to do with the provider
  variables above, and `advise`/`master` have nothing to do with this extra.
- **`faster-whisper` is MIT**, and CPU-capable without CUDA. The alternatives were rejected on
  install cost, not on quality: `whisper.cpp` bindings need a build toolchain, and OpenAI's
  `whisper` package pulls in torch. An agent should be able to install this unattended and have
  it work.
- **With the extra absent, exactly one verb is lost.** `detect fillers` exits non-zero with
  `speech_extra_missing`, naming the extra and pointing here. `detect silence`, `detect
  transients`, `cut`, `strip-silence` and the whole mastering chain are unaffected — none of
  them needs to know what was said.

It does **not** degrade to an energy-only guess when the extra is missing. A hesitation detector
presented as a filler-word detector would report regions a caller would reasonably take for
words, and `cut` would then remove them.

## Optional host dependency: ffmpeg

Not configuration, but the other thing `check` reports. WAV, FLAC and AIFF need nothing beyond
the bundled libsndfile. mp3, m4a and ogg are decoded and encoded through `ffmpeg`, which must be
on `PATH`. Install instructions are the project's own:
<https://ffmpeg.org/download.html>. With `ffmpeg` absent, `aud` works fully on uncompressed
material and declines compressed formats with a remedy naming the missing program.
