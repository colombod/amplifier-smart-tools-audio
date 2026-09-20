# Library Reference

Every capability of `aud` is reachable from `aud.lib`. The CLI (`cli.py`) is a thin wrapper over
it: argument parsing and printing only, no logic of its own. Anything you can do from the shell
you can do by importing this module directly:

```python
from aud import lib
from aud.plan import new_plan

plan = lib.eq(new_plan(), hpf=40.0, peaks=[(3200.0, -2.5, 1.4)])
plan = lib.loudness(plan, target_lufs=-14.0)
plan = lib.limit(plan, ceiling_dbtp=-1.0)
result = lib.render(plan, "in.wav", "out.wav")
```

Functions that touch samples (`analyze`, `render`, `verify`, `curve_extract`, `curve_apply`, the
`detect_*` functions) import `aud.dsp` lazily, inside the function body, so importing `aud.lib`
itself never requires numpy/scipy/soundfile/pyloudnorm to have a working DSP backend behind it —
only to be installed. If `aud.dsp` is not available, these functions raise
`AudError(code="not_implemented")` rather than a bare `ImportError` or a silent fake result.

## Manifest, check, config, skill

```python
def manifest() -> dict
def check() -> dict
def config(**overrides: Any) -> dict
def skill() -> str
```

- `manifest()` — the validated `SMART_TOOL.md` frontmatter (`smart_tool_format`, `name`,
  `version`, `description`, `use_cases`, `platforms`, `requires`), as a plain dict.
- `check()` — host readiness: whether the core DSP dependencies import, whether `ffmpeg` is on
  `PATH`, and which of the five provider environment variables is set. A credential is reported
  by name and boolean only — its value is never read into the result.
- `config(**overrides)` — every setting `aud` knows about, and which tier each came from
  (`argument` > `config_file` > `environment` > `default`). Pass a setting as a keyword argument
  with a non-`None` value to make it win at the `argument` tier. **0.11.0:** a config-file or
  `AUD_*` environment value that does not match the setting's declared type raises
  `AudError(code="bad_config")` naming the setting, the value found and the type expected —
  never silently coerced, never silently replaced by the default. See
  [docs/CONFIGURATION.md](CONFIGURATION.md) for the full precedence rules and the settings table.
- `skill()` — the full `aud --help` text, generated from `aud.core.skill.CAPABILITIES`, the one
  place the verb surface is enumerated for documentation.

## Measurement: `analyze`, `verify`

```python
def analyze(path: str) -> dict
def verify(path: str, target_lufs: float | None = None, ceiling_dbtp: float | None = None) -> dict
```

- `analyze(path)` — read-only measurement of what is actually in a file: integrated LUFS, true
  peak (dBTP), crest factor, per-band spectral energy, sibilance, and an ambience/reverb-tail
  estimate. **0.11.0:** `integrated_lufs` and `loudness_range_lu` need at least 0.4 s of audio
  (pyloudnorm's BS.1770 gating block); below that they are `null` and the result carries a
  `loudness_unavailable_reason` string naming why, rather than the call failing or silently
  returning a wrong number. Every other field measures normally regardless.
- `verify(path, target_lufs=None, ceiling_dbtp=None)` — re-measures a file and, for whichever of
  `target_lufs`/`ceiling_dbtp` is given, reports whether it was actually met (`lufs_ok`,
  `ceiling_ok`). The ceiling check uses `aud.dsp.limiter.CEILING_TOLERANCE_DB`, the same
  tolerance the limiter itself uses to decide it is done, since true peak is an oversampled
  estimate rather than an exact quantity.

Both raise `AudError(code="file_not_found")` or `AudError(code="audio_decode_error")` for a
missing or undecodable `path`.

## Detection: `detect_silence`, `detect_transients`, `detect_fillers`

```python
def detect_silence(path: str, threshold_above_floor_db: float = 6.0, min_len_ms: float = 400.0) -> Any
def detect_transients(path: str, sensitivity: float = 1.0, min_gap_ms: float = 50.0) -> Any
def detect_fillers(path: str, words: list[str] | None = None, min_pause_ms: float = 700.0) -> Any
```

All three are read-only: they open `path` and write nothing. Each returns a **regions document**
(`kind="silence"` / `"transient"` / `"filler"`) — see
[contracts/regions.v1.md](../contracts/regions.v1.md) — never a plan. `detect_silence`'s
threshold is dB above *this file's own measured noise floor*, not an absolute dBFS value, for
the same reason `strip_silence` below uses the same convention.

`detect_fillers` needs the optional `speech` extra (`faster-whisper`, a local model — no
provider, no credential, no network call once cached). **0.11.0:** the extra is checked *before*
`path` is decoded — a missing prerequisite fails immediately, naming the extra, rather than
paying for a decode the call was always going to refuse. With the extra absent this raises
`AudError(code="speech_extra_missing")`; it never degrades to an energy-only guess.

## The plan/chain stages

```python
def cut(plan, regions_text, *, pad_out_ms=0.0, pad_in_ms=0.0, snap="zero_crossing",
        snap_window_ms=20.0, fade_out_ms=0.0, fade_in_ms=0.0, crossfade_ms=10.0,
        crossfade_shape="equal_power", filler_tail_pad_ms=None) -> Plan
def strip_silence(plan, *, threshold_above_floor_db=6.0, min_len_ms=400.0, keep_ms=150.0,
                   pad_out_ms=80.0, pad_in_ms=80.0, snap="zero_crossing", snap_window_ms=20.0,
                   fade_out_ms=0.0, fade_in_ms=0.0, crossfade_ms=10.0,
                   crossfade_shape="equal_power") -> Plan
def gate(plan, *, threshold_above_floor_db=12.0, threshold_db=None, range_db=20.0, attack_ms=2.0,
         hold_ms=50.0, release_ms=150.0, lookahead_ms=3.0, sidechain_hpf_hz=80.0,
         crossovers_hz=None) -> Plan
def expand(plan, *, threshold_above_floor_db=6.0, threshold_db=None, ratio=2.0, knee_db=6.0,
           attack_ms=5.0, hold_ms=50.0, release_ms=150.0, lookahead_ms=3.0,
           sidechain_hpf_hz=80.0, crossovers_hz=None) -> Plan
def deess(plan, amount_db=6.0, freq_hz=6500.0) -> Plan
def dereverb(plan, amount_db=6.0) -> Plan
def eq(plan, hpf=None, lpf=None, peaks=None, shelves=None) -> Plan
def eq_match(plan, curve=None, reference_path=None, amount=1.0, max_gain_db=12.0) -> Plan
def compress(plan, bands, ratio=2.5) -> Plan
def saturate(plan, drive=1.0, mix=0.25) -> Plan
def reverb(plan, amount=0.15, decay=1.2, predelay_ms=0.0) -> Plan
def stretch(plan, factor=1.0) -> Plan
def pitch(plan, semitones=0.0) -> Plan
def loudness(plan, target_lufs=-14.0) -> Plan
def limit(plan, ceiling_dbtp=-1.0) -> Plan
```

Every one of these is a plan builder: it validates its own parameters, appends one stage to
`plan`, and returns a **new** `Plan` (plans are never mutated in place). None of them touches a
sample — that happens once, later, in `render`. A stage builder's own argument names are this
library/CLI's convenience naming (e.g. `compress`'s `bands`/`ratio` expand into the plan
document's full per-band `crossovers_hz`/`bands` form; `reverb`'s `amount`/`decay` map to the
document's `mix`/`decay_s`) — the field names the *document* actually stores are
[contracts/plan.v1.md](../contracts/plan.v1.md)'s, not these argument names.

`render` (below) applies a plan's stages in **canonical mastering order** — editing → repair →
tone → dynamics → character → loudness → limiting (`aud.plan.STAGE_ORDER`) — regardless of the
order they were appended in. Every stage name in `STAGE_ORDER` has a working DSP handler in this
release; there is no stage `aud` itself builds that renders as `not_implemented`.

A bad parameter raises `AudError(code="bad_param")` naming the field and what a valid value looks
like; an unrecognised stage name raises `AudError(code="unknown_stage")`.

## Presets: `preset_list`, `preset_show`

```python
def preset_list() -> list[dict[str, str]]
def preset_show(name: str) -> Plan
```

Named, pre-built chains for common destinations (`podcast`, `music-streaming`, `broadcast`,
`voiceover`), built through the exact same stage builders above — there is no separate,
preset-only parameter path. `preset_show` raises `AudError(code="unknown_preset")` for a name
`preset_list()` does not report.

## Render: `render`

```python
def render(plan: Plan, in_path: str, out_path: str) -> dict
```

The one function that touches samples on a plan's behalf: reorders `plan`'s stages into
canonical order, applies all of them in a single decode/filter/encode pass, and writes the
result. Returns `{"out_path": ..., "report": {...}}` — the report names every stage applied, in
order, with its own measured effect (gain applied, gain reduction, true peak before/after, edit
points for `cut`/`strip_silence`, and so on per stage).

**0.11.0:** `out_path` in the result is always the **absolute** path the file was actually
written at (resolved before the write, via `Path(...).expanduser().resolve()`), never an
unresolved echo of a relative `out_path` the caller passed in. The write itself is the tool's own
atomic write path (see `curve_extract` below for the same guarantee on a text artifact).

Raises `AudError` with code `file_not_found`, `audio_decode_error`, `audio_write_error`,
`bad_plan`, `crossfade_exceeds_gap`, `bad_param` (an `eq_match` curve incompatible with this
file's sample rate), or `not_implemented` (a stage name in the plan that this build's render
engine does not register — not currently reachable for any stage `aud` itself builds).

## Curve: `curve_extract`, `curve_apply`

```python
def curve_extract(path: str, out: str) -> dict
def curve_apply(path: str, out_path: str, *, curve: Any = None, curve_path: str | None = None) -> dict
```

- `curve_extract(path, out)` — extracts `path`'s spectral profile and saves it as JSON at `out`.
  Returns `{"curve_path": ...}` — **0.11.0:** always the absolute path actually written, resolved
  the same way `render`'s `out_path` is. The write is atomic: a failure never leaves a partial
  file at the destination, and raises `AudError(code="bad_path")` naming the path and the reason.
- `curve_apply(path, out_path, *, curve=None, curve_path=None)` — applies a spectral curve to
  `path`, writing the result to `out_path`. **Changed in 0.11.0:** `curve` is the data-only
  interface a library caller should use directly — a bare array of `[freq_hz, gain_db]` pairs, or
  the rich dict `curve_extract`/`eqmatch.spectrum_profile` produce — pass the curve's actual
  content, not a path to it. `curve_path` is the CLI's own convenience: when `curve` is not given,
  the curve is read from that file; when both are given, `curve_path` is used only to name the
  source in an error message. Exactly one of `curve`/`curve_path` must resolve to data, else
  `AudError(code="bad_param")`. Returns `{"out_path": ...}`, again always absolute.

Both raise `file_not_found`/`audio_decode_error` for `path`, and `bad_param` for a curve with
fewer than two points, a malformed point, or frequencies that are not strictly ascending.

## Model-backed: `advise`, `master`

```python
def advise(path, *, target_lufs=-14.0, ceiling_dbtp=-1.0, reference_path=None, model=None,
           backend=None) -> dict
def master(in_path, out_path, *, target_lufs=-14.0, ceiling_dbtp=-1.0, reference_path=None,
           model=None, dry_run=False, backend=None) -> dict
```

- `advise(path, ...)` — runs the same measurement `analyze` does, then asks a model to choose a
  mastering chain (which stages, what parameters, and why), grounded in those numbers. The model
  never sees or touches the audio itself, only the measurement report, and its proposed chain is
  validated through the exact same stage builders every other verb uses before it becomes part of
  the returned plan — an invalid proposal is `AudError(code="bad_model_output")` or
  `AudError(code="bad_model_plan")`, never a corrupted plan that would fail later at render.
  **0.11.0:** the provider credential is resolved (and validated) *before* `path` (and
  `reference_path`) is decoded, so a caller with no credential configured is refused before
  paying for the decode, not after. Returns
  `{"plan", "measurements", "reference_measurements", "provider", "model", "stages"}`.
- `master(in_path, out_path, ...)` — the one-shot path: calls `advise` internally, renders that
  plan in a single pass, then verifies the result. `dry_run=True` stops after `advise`: the plan
  and its reasoning are returned, nothing is rendered, and `out_path` is never touched — neither
  is it touched if `advise` refuses for lack of a credential or a valid model response.

`backend` is a dependency-injection seam for tests (see **Intelligence**, below); real callers
leave it `None` and get the provider selected from whichever credential is configured.

## The plan document and the regions document

Every stage builder above returns a `Plan` (`aud.plan.Plan`); every `detect_*` function returns a
regions document. Both are the values passed between capabilities and between processes — a plan
is written to stdout by one verb and read from stdin by the next, and a plan is otherwise inert
data: it can be written to a file, piped, or archived next to the render it produced.

- **The plan document** — its shape, its field names, its defaults, and its canonical stage
  order are specified in [contracts/plan.v1.md](../contracts/plan.v1.md). That contract, not this
  page, is the source of truth for what a stage's `params` actually contain.
- **The regions document** — what `detect_silence`/`detect_transients`/`detect_fillers` emit and
  what `cut` consumes is specified in [contracts/regions.v1.md](../contracts/regions.v1.md).

`aud.plan.new_plan()`, `aud.plan.append()`, `aud.plan.ordered()`, `aud.plan.read_plan()` and
`aud.plan.write_plan()` are the functions behind plan construction and (de)serialisation, if a
caller wants to build one without going through a stage builder.

## Errors: `AudError`

```python
class AudError(Exception):
    def __init__(self, code: str, message: str, remedy: str) -> None: ...
    def to_dict(self) -> dict[str, str]: ...
```

The one exception type `aud` raises for every user-facing failure (`aud.schemas.AudError`). Every
instance carries a machine-readable `code`, a human-readable `message`, and a `remedy` describing
what a valid input looks like — `to_dict()` is what the CLI renders verbatim into
`{"error": {"code", "message", "remedy"}}` (see [02-cli.md](02-cli.md)'s I/O contract). Codes are
lowercase `snake_case`, never a second convention:

| Code | Raised by |
|---|---|
| `audio_decode_error` | A file exists but is not decodable audio. |
| `audio_write_error` | An output path cannot be written (permissions, missing directory, unsupported subtype). |
| `bad_config` | A config-file or `AUD_*` environment value does not match its setting's declared type. |
| `bad_manifest` / `manifest_missing` | The packaged `SMART_TOOL.md` fails to parse or validate. |
| `bad_model_output` / `bad_model_plan` | `advise`'s model response could not be parsed, or its proposed chain fails stage-builder validation. |
| `bad_param` | A stage or capability argument is out of range, the wrong shape, or otherwise invalid. |
| `bad_path` | A text artifact (e.g. a curve JSON) could not be written atomically. |
| `bad_plan` | Plan JSON (from stdin or `--from`) is not valid JSON, or does not match the plan shape. |
| `crossfade_exceeds_gap` | A `cut`/`strip_silence` crossfade is longer than the material available at that join. |
| `file_not_found` | A given path does not exist. |
| `internal_error` | An exception `aud` did not already turn into a named `AudError` — an internal bug, not a usage error. |
| `not_implemented` | A capability needs `aud.dsp`, which is not available in this build. |
| `provider_config_incomplete` | `AZURE_OPENAI_API_KEY` is set but `AZURE_OPENAI_ENDPOINT` is not. |
| `provider_credential_missing` | `advise`/`master` need a provider credential and none of the accepted environment variables is set. |
| `provider_model_unavailable` | The provider rejected the resolved model name (HTTP 404, or a response naming the model). |
| `provider_request_failed` | A provider request failed for any other reason (network, rate limit, malformed response shape). |
| `regions_not_cuttable` | `cut` was given a `kind="transient"` regions document; transients are zero-length and describe nothing to remove. |
| `snap_window_invalid` | `snap_window_ms` is not `> 0` and `<= 1000`. |
| `speech_extra_missing` | `detect_fillers` needs the optional `speech` extra and it is not installed. |
| `unknown_preset` | `preset_show`'s `name` is not one `preset_list()` reports. |
| `unknown_stage` | A stage name is not one of `aud.plan.STAGE_ORDER`. |
| `usage_error` | The CLI's own argument parsing failed (not raised by the library itself). |

## Intelligence: the model-backed seam

`advise` and `master` are the only capabilities that call a model, and the only seam between
`aud` and any provider SDK is `aud.intelligence.interface.IntelligenceBackend`:

```python
class IntelligenceBackend(Protocol):
    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str: ...
```

One method: a system prompt and a user prompt in, raw text back. No provider SDK type crosses
this boundary in either direction, and no provider SDK is a dependency of this package — every
backend is plain HTTPS via `urllib`, so the base install never grows a cost for a caller with no
AI-provider credential configured. A test implements this Protocol directly with a canned string
and passes it as `advise`'s `backend=` parameter — no network, no credential, no cost.

```python
def resolve_backend(model: str | None = None) -> tuple[IntelligenceBackend, str, str]
```

Picks a backend from whichever provider credential is configured. Provider precedence is the
order of `PROVIDER_ENV_VARS` — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` /
`GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY` — the first one present in the environment wins. Model
precedence: an explicit `model` argument (the CLI's `--model`) > the `AUD_MODEL` environment
variable > `DEFAULT_MODELS[provider]`. Raises `AudError(code="provider_credential_missing")` if
none of `PROVIDER_ENV_VARS` is set.

Four backends ship today, one per provider: `AnthropicBackend`, `OpenAIBackend`, `GoogleBackend`,
`AzureOpenAIBackend` — all in `aud.intelligence.interface`. `AzureOpenAIBackend` additionally
needs `AZURE_OPENAI_ENDPOINT` in the environment (raises `provider_config_incomplete` without
it); `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_API_VERSION` are separately defaulted. A default
model name (`DEFAULT_MODELS`) is perishable and will go stale over time — a provider rejecting it
is reported as `provider_model_unavailable`, naming `--model`/`AUD_MODEL` as the fix, rather than
a generic request failure, since the likeliest cause is the tool's own stale default rather than
the caller's credential.

## Configuration

Settings (`sample_rate_policy`, `default_ceiling_dbtp`, `default_target_lufs`, `oversample`,
`output_subtype`) and credentials (the five provider environment variables above) are two
separate subjects with **opposite** precedence rules, and neither is restated here — see
[docs/CONFIGURATION.md](CONFIGURATION.md) for the full precedence tables, the config file
location, and the absent/null/wrong-type distinction `config()` (above) enforces.
