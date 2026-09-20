# CLI Reference

The CLI (`cli.py`) is a thin wrapper over the [library](01-library.md): one subcommand per
capability, taking the same arguments under related names, and doing nothing the library does
not — see `AGENTS.md #2`, "the library is the tool". What each argument *means*, what a
capability returns, and what it raises is documented there; this page covers only what the CLI
adds: the invocation shape, the help ladder, and what reaches stdout, stderr, and the exit code.

## Help

```
aud -h                  argparse's own usage line: the verb list, no descriptions
aud --help              the tool's skill (lib.skill()), written for an agent driving it
aud <verb> --help       one verb in full: what it does, when to reach for it, every
                        parameter with its default, what it prints, every failure code
aud <verb> -h           argparse's own per-verb usage line (positional args, options) --
                        NOT the curated doc above; only the long-form --help gets that
```

`aud --help` is intercepted before argparse ever runs, so it never fails from a missing
subcommand: `main()` checks whether the *first* token is exactly `--help` and, if so, prints
`lib.skill()` and exits 0 — the same text `aud.core.skill.CAPABILITIES` (the one place the verb
surface is enumerated) renders. `aud <verb> --help` is a second interception: if the first token
names a known verb and `--help` appears anywhere after it, `cli.py` prints that verb's entry from
`aud.verbdoc.VERB_DOCS` and exits 0, without argparse parsing the rest of the command line at
all. Neither interception cares whether the verb's own required arguments are present — a
`--help` request always answers, even for a call that would otherwise fail as a usage error.

`aud -h` and `aud <verb> -h` fall through to argparse's own built-in `-h` handling instead
(`add_help=True`), which prints a bare usage line and the registered options/positionals — not
the curated per-verb documentation. Reach for `--help`, not `-h`, when driving `aud` as an agent
or looking for the full explanation of a verb.

## The I/O contract

```
Success:    {"result": ...} on stdout.                                    exit 0
Failure:    {"error": {"code", "message", "remedy"}} on stderr.            exit 1
Bad invocation (argparse/usage error): same error shape, on stderr.        exit 2
```

Results go to **stdout**; every diagnostic — an error envelope, `advise`'s per-stage reasoning —
goes to **stderr**. This is deliberate and is the reason stage verbs pipe into each other
cleanly: `aud <verb> ... | aud render ...` must never have the second command mistake an error
envelope arriving on the channel it expected to read a plan from. `usage_error` (argparse
rejecting the command line itself — an unknown verb, a missing required argument, an invalid
choice) exits 2; every `AudError` the library itself raises exits 1; a genuinely unexpected
exception is reported as `internal_error` (still exit 1) rather than a raw traceback, with the
real traceback available on stderr behind `--debug`/`AUD_DEBUG=1`.

Two real, captured transcripts, stdout and stderr read back separately:

```
$ aud manifest 1>out.txt 2>err.txt; echo "exit=$?"
exit=0
$ cat out.txt
{"result": {"name": "aud", "version": "0.11.0", ...}}
$ cat err.txt
                                                        (empty)

$ aud analyze /no/such/file.wav 1>out.txt 2>err.txt; echo "exit=$?"
exit=1
$ cat out.txt
                                                        (empty)
$ cat err.txt
{"error": {"code": "file_not_found", "message": "Audio file not found: /no/such/file.wav",
"remedy": "Check that '/no/such/file.wav' exists and is a readable audio file."}}

$ aud bogus-verb 1>out.txt 2>err.txt; echo "exit=$?"
exit=2
$ cat out.txt
                                                        (empty)
$ cat err.txt
{"error": {"code": "usage_error", "message": "argument verb: invalid choice: 'bogus-verb' ...",
"remedy": "Run 'aud --help' for the list of verbs, or 'aud <verb> --help' for one verb's full
documentation."}}
```

**One documented exception to the envelope**, and it is the reason the pipeline works: `plan` and
every stage verb (`cut`, `strip-silence`, `gate`, `expand`, `deess`, `dereverb`, `eq`, `eq-match`,
`compress`, `saturate`, `reverb`, `stretch`, `pitch`, `loudness`, `limit`, and `advise`) print the
**plan document itself**, raw and unwrapped, to stdout on success — not `{"result": {...plan...}}`.
`detect` is the read-only counterpart: it prints a **regions document**, raw and unwrapped. Both
still follow the error/exit-code rules above on failure. Confirmed by piping stage verbs with no
audio file at all — nothing touches a sample until `render`:

```
$ aud plan | aud deess --amount 6 | aud eq --hpf 40
{"plan_format":1,"created_with":"aud/0.11.0","stages":[
  {"stage":"deess","params":{"amount_db":6.0,"freq_hz":6500.0}},
  {"stage":"eq","params":{"hpf_hz":40.0,"lpf_hz":null,"peaks":[],"shelves":[]}}]}
```

Stage verbs read a plan document on stdin (when stdin is not a TTY) and write the updated plan
document to stdout, so they pipe: `aud plan | aud deess --amount 6 | aud render in.wav out.wav`.
`detect` emits a regions document that `cut` consumes the same way:
`aud detect silence in.wav | aud cut | aud render in.wav out.wav`. `render` is the only verb that
touches a sample — everything before it in a pipeline is inert JSON changing hands.

## `aud manifest`

```
aud manifest
```

No arguments. Prints `{"result": <SMART_TOOL.md frontmatter>}`. No reachable failure — the
packaged manifest is validated once at import time.

## `aud check`

```
aud check
```

No arguments. Prints `{"result": {"tool", "version", "python_version", "ready",
"deterministic_capabilities_available", "requirements": [...]}}`. Always succeeds; an absent
dependency is reported in the result, not raised.

## `aud analyze`

```
aud analyze PATH
```

`PATH` (positional). Prints `{"result": {...}}` with the measurement fields described in
[01-library.md](01-library.md). Failures: `file_not_found`, `audio_decode_error`.

## `aud config`

```
aud config [--sample-rate-policy STR] [--default-ceiling-dbtp FLOAT]
           [--default-target-lufs FLOAT] [--oversample INT] [--output-subtype STR]
```

Each flag overrides the named setting at the `argument` tier for this call only; omitted flags
default to `None` and fall through to the config file / environment / built-in default. Prints
`{"result": {<setting>: {"value", "source"}, ...}}`. Always succeeds for the CLI's own argument
handling; a bad config-file or environment value (not an argument) is `bad_config`.

## `aud plan`

```
aud plan [--from PATH]
```

Prints an empty plan document, or the one loaded from `--from`, raw and unwrapped. Failures:
`file_not_found`, `bad_plan`.

## Stage verbs

Every stage verb below reads a plan on stdin (defaulting to a fresh empty plan when stdin is a
TTY or empty), appends exactly one stage, and prints the updated plan raw and unwrapped. Argument
meanings, defaults, and validation are in [01-library.md](01-library.md); shown here only as the
literal flag spelling the parser accepts. All share the failure `bad_plan` (stdin is not a valid
plan document) in addition to what is listed.

### `aud cut`

```
aud cut [--regions PATH] [--pad-out F] [--pad-in F] [--snap MODE] [--snap-window F]
        [--fade-out F] [--fade-in F] [--crossfade F] [--crossfade-shape SHAPE]
        [--filler-tail-pad F]
```

`--regions` reads a regions document from a file; omitted, it reads one from stdin instead of a
plan (see the pipe example above). `MODE` is one of `zero_crossing`/`silence`/`transient`/`none`;
`SHAPE` is `equal_power`/`linear`. Failures: `bad_param`, `snap_window_invalid`,
`regions_not_cuttable` (at build time); `crossfade_exceeds_gap` (at `render` time, once the
material is actually available to check against).

### `aud strip-silence`

```
aud strip-silence [--threshold F] [--min-len F] [--keep F] [--pad-out F] [--pad-in F]
                   [--snap MODE] [--snap-window F] [--fade-out F] [--fade-in F]
                   [--crossfade F] [--crossfade-shape SHAPE]
```

Same edit-point flags as `cut`, with different padding defaults (80 ms vs 0). Failures:
`bad_param`, `snap_window_invalid` (at build time); `crossfade_exceeds_gap` (at render time).

### `aud gate`

```
aud gate [--threshold F] [--threshold-abs F] [--range F] [--attack F] [--hold F]
         [--release F] [--lookahead F] [--sidechain-hpf F] [--bands F1,F2,...]
```

Failures: `bad_param`.

### `aud expand`

```
aud expand [--threshold F] [--threshold-abs F] [--ratio F] [--knee F] [--attack F]
           [--hold F] [--release F] [--lookahead F] [--sidechain-hpf F] [--bands F1,F2,...]
```

Failures: `bad_param`.

### `aud deess`

```
aud deess [--amount F] [--freq F]
```

Failures: `bad_param`.

### `aud dereverb`

```
aud dereverb [--amount F]
```

Failures: `bad_param`.

### `aud eq`

```
aud eq [--hpf F] [--lpf F] [--peak FREQ,GAIN,Q ...] [--shelf TYPE,FREQ,GAIN,Q ...]
```

`--peak` and `--shelf` are each repeatable (append one band per occurrence). Failures:
`bad_param`.

### `aud eq-match`

```
aud eq-match (--curve PATH | --reference PATH) [--strength F] [--max-gain-db F]
```

Exactly one of `--curve`/`--reference` is required (argparse mutually-exclusive group — supplying
neither or both is a `usage_error`, exit 2, not a library-level `bad_param`). Failures beyond
that: `bad_param`, `file_not_found`, `audio_decode_error` (for `--reference`).

### `aud compress`

```
aud compress --bands F1,F2,... [--ratio F]
```

`--bands` is required. Failures: `bad_param`.

### `aud saturate`

```
aud saturate [--drive F] [--mix F]
```

Failures: `bad_param`.

### `aud reverb`

```
aud reverb [--amount F] [--decay F] [--predelay F]
```

Failures: `bad_param`.

### `aud stretch`

```
aud stretch [--factor F]
```

Failures: `bad_param`.

### `aud pitch`

```
aud pitch [--semitones F]
```

Failures: `bad_param`.

### `aud loudness`

```
aud loudness [--target F]
```

Failures: `bad_param`.

### `aud limit`

```
aud limit [--ceiling F]
```

Failures: `bad_param`. `lookahead_ms`/`release_ms`/`oversample` are plan-document-only fields
(see [01-library.md](01-library.md)); this verb does not expose CLI flags for them.

## `aud curve`

```
aud curve extract PATH OUT
aud curve apply PATH CURVE OUT
```

Two subcommands, neither reads or writes a plan. `extract` prints
`{"result": {"curve_path": OUT}}`; `apply` reads the curve from the `CURVE` JSON file (the CLI's
convenience over the library's data-first `curve=` parameter — see
[01-library.md](01-library.md)) and prints `{"result": {"out_path": OUT}}`. Both paths in the
result are always absolute (0.11.0). Failures: `file_not_found`, `audio_decode_error`,
`bad_param` (malformed or incompatible curve JSON), `audio_write_error`.

## `aud render`

```
aud render IN_PATH OUT_PATH
```

Reads the plan on stdin, applies it in canonical order in a single pass, and prints
`{"result": {"out_path": ..., "report": {...}}}` — `out_path` is always absolute. Failures:
`file_not_found`, `audio_decode_error`, `audio_write_error`, `bad_plan`,
`crossfade_exceeds_gap`, `bad_param` (an `eq_match` curve incompatible with this file's sample
rate), `not_implemented` (not currently reachable for any stage `aud` itself builds).

## `aud verify`

```
aud verify PATH [--target F] [--ceiling F]
```

Prints `{"result": {"measured": {...}, ...}}`; `lufs_ok`/`ceiling_ok` appear only when the
corresponding flag was given. Read-only. Failures: `file_not_found`, `audio_decode_error`.

## `aud detect`

```
aud detect transients PATH [--sensitivity F] [--min-gap F]
aud detect silence PATH [--threshold F] [--min-len F]
aud detect fillers PATH [--words STR] [--min-pause F]
```

Prints a regions document, raw and unwrapped, for whichever subcommand was given. Read-only.
Failures: `file_not_found`, `audio_decode_error`, and — `fillers` only —
`speech_extra_missing`.

## `aud advise`

```
aud advise PATH [--target F] [--reference PATH] [--model NAME]
```

Model-backed; needs one of the five provider credentials (see [01-library.md](01-library.md)).
Prints the chosen plan raw and unwrapped on stdout, so it pipes into `render` like a hand-built
chain; the per-stage reasoning and which provider/model answered go to **stderr**, one line per
stage, because stdout must stay a clean plan document. Failures: `file_not_found`,
`audio_decode_error`, `provider_credential_missing`, `provider_config_incomplete`,
`provider_model_unavailable`, `provider_request_failed`, `bad_model_output`, `bad_model_plan`.

## `aud master`

```
aud master IN_PATH OUT_PATH [--target F] [--ceiling F] [--reference PATH] [--model NAME]
           [--dry-run]
```

Model-backed. Prints `{"result": {"plan", "stages", "provider", "model", "measurements",
"render", "out_path", "verify", ...}}` — with `--dry-run`, the same shape minus
`render`/`out_path`/`verify`, plus `"dry_run": true`, and `OUT_PATH` is never written. Every
failure that happens before rendering leaves `OUT_PATH` untouched. Failures: `file_not_found`,
`audio_decode_error`, `provider_credential_missing`, `provider_config_incomplete`,
`provider_model_unavailable`, `provider_request_failed`, `bad_model_output`, `bad_model_plan`,
`audio_write_error`.

## `aud preset`

```
aud preset --list
aud preset show NAME
```

`--list` prints `{"result": [{"name", "description"}, ...]}`. `show NAME` prints the named
preset's plan document, raw and unwrapped, ready to pipe into `render`. Failures:
`unknown_preset`.
