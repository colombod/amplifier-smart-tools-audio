# Working in this repository

Conventions for anyone — human or agent — changing `aud`. These are constraints, not
preferences: several of them are load-bearing for claims the tool makes in its manifest.

## Repository map

```
smart-tool.json          # descriptor: where the manifest is, how to launch the CLI
pyproject.toml           # package definition; the version here is authoritative
src/aud/SMART_TOOL.md    # the manifest -- what the tool is and what it needs
src/aud/core/            # io, analysis, plan, engine, config, errors
src/aud/dsp/             # filters, crossover, dynamics, limiter, saturation, loudness
contracts/plan.v1.md     # the plan document other programs may parse
docs/                    # VISION (why), ARCHITECTURE (how), CONFIGURATION (settings)
tests/
```

The manifest is the description of record. If a change makes the manifest wrong, the change is
not finished until the manifest is right.

## 1. The licence constraint is a hard rule

`aud` is MIT. It stays MIT by writing its own DSP. These dependencies are **forbidden**, no
matter how much work they would save:

| Forbidden | Licence |
|---|---|
| `pedalboard` | GPL-3.0 |
| `matchering` | GPL-3.0 |
| Rubber Band, `pyrubberband`, `rubberband-cli` | GPL-2.0-or-later / commercial dual |

Adding any of them relicenses the tool and everything that embeds it. That is not a dependency
decision an implementation task gets to make; it is a relicensing proposal and must be argued as
one. See [docs/VISION.md](docs/VISION.md).

Permitted today: `numpy` (BSD), `scipy` (BSD), `soundfile` (BSD-3), `pyloudnorm` (MIT),
`pydantic` (MIT), `pyyaml` (MIT), and the optional `python-stretch` (Signalsmith Stretch, MIT).
**Any new dependency needs its licence checked and named in the pull request** — MIT, BSD, ISC,
Apache-2.0 or PSF. Anything GPL-family, or unlicensed, is a no.

## 2. The library is the tool

Every capability lives in `aud.lib`. `cli.py` parses arguments, calls the library, serialises
the result. **A capability reachable only through the CLI is a defect.** Parameter validation,
default resolution, stage ordering and report construction all belong in the library, because
the Python caller, the shell caller and any future adapter must get the same tool.

Test for it: if `cli.py` were deleted, would anything but argument parsing be lost?

## 3. Deterministic verbs run with zero credentials — and that is proven, not assumed

Everything except `advise` and `master --auto` must work with no provider configured and no
credential present.

The proof is a **scrubbed-subprocess test**: launch the verb in a subprocess with the
environment stripped to a bare `PATH`, and assert it succeeds. Not a mock, not a monkeypatched
`os.environ` — an inherited variable is exactly the failure mode being tested for, and an
in-process test cannot see it. CI additionally asserts that none of the five provider variables
is present before the suite runs, so the check cannot pass for the wrong reason.

Corollary: **never import a provider or agent engine at module level.** Import it inside the two
verbs that need it. A top-level import makes every deterministic path depend on the provider
being installed and can rewrite environment state for unrelated code.

## 4. Failures name what went wrong and the remedy

The caller is usually an agent, and an agent cannot act on a stack trace.

```json
{"error": {"code": "E_PLAN_PARAM_RANGE",
           "message": "compress.bands[0].ratio is 0.5; ratio must be >= 1.0",
           "remedy": "Set ratio to 1.0 or greater; 1.0 means no compression in that band."}}
```

- One JSON document on stdout, nothing else. Progress and diagnostics go to stderr.
- Failure exits non-zero.
- `message` says what was found, `remedy` says what to do about it.
- Absent, null and wrong-type are three different conditions and only the third is fatal —
  see [docs/CONFIGURATION.md](docs/CONFIGURATION.md). Never coerce a wrong type into a default;
  the audio would be wrong instead of the command.

## 5. Never hardcode a model name

The model comes from configuration or from the host's provider setup. A string like
`"claude-3-5-sonnet-20241022"` or `"gpt-4o"` in the source is a bug: it goes stale, it overrides
what the user configured, and it breaks against providers that have never heard of it.

## 6. Manifest rules the conformance kit enforces

- `requires[].install` is a **reference to documentation, never a command**. `docs/CONFIGURATION.md`
  and `https://ffmpeg.org/download.html` are valid; anything starting with `pip`, `uv`, `brew`,
  `apt`, `curl` or containing shell metacharacters fails conformance.
- **Static version in `pyproject.toml`**, matching `version:` in the manifest. A dynamic version
  does not fail the `manifest-version-matches-package` check — it makes it **SKIP**, which
  silently removes a check rather than passing one. Bump both together.
- **Exactly one `SMART_TOOL.md`** under the distribution root. A second one anywhere in the tree
  fails conformance.
- `cli_argv[0]` must resolve on `PATH` for the kit to run the tool, so conformance runs against
  an installed `aud`, not a bare checkout.

## 7. Changing the plan document is changing a contract

[contracts/plan.v1.md](contracts/plan.v1.md) is what other programs parse. Adding an optional
param with a behaviour-preserving default stays in format 1 and updates that file. Renaming a
field, changing a type, changing a **default**, or changing rejection semantics is a breaking
change: it needs `plan_format: 2` and a new `contracts/plan.v2.md`. Do not edit `plan.v1.md` to
describe different behaviour — stored plans and third-party generators still read it.

## 8. Code conventions

- Python 3.11+. `ruff` is configured in `pyproject.toml`: line length 120, and **relative
  imports are banned** (`ban-relative-imports = "all"`) — import `from aud.dsp.filters import ...`,
  not `from .filters import ...`.
- `dsp/` modules take arrays and parameters and return arrays. They do not read configuration,
  touch the filesystem, or raise user-facing errors. Keep that boundary.
- Float64 from decode to encode; quantise exactly once, at write.
- All file I/O specifies `encoding="utf-8"` explicitly.

## 9. Before calling anything done

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

All three clean. Then, if the change touched the manifest, the descriptor, the CLI surface or
the packaging, run the conformance kit as well — see [CONTRIBUTING.md](CONTRIBUTING.md).

Do not report a suite as passing that you have not run. An honest "not yet verified" is
recoverable; a false green is not.
