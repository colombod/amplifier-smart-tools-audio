# Contributing

## Prerequisites

- `git`
- [`uv`](https://docs.astral.sh/uv/) — manages the Python toolchain, the environment and the
  dependencies. Python 3.11 is pinned in `.python-version`; `uv` will fetch it if the host does
  not have it.

Optional, and only for what they enable:

- `ffmpeg` — for mp3, m4a and ogg. WAV, FLAC and AIFF work without it.
- An AI provider credential — only `advise` and `master --auto` use one. Everything else must
  work without, and the test suite checks that it does.

## Setup

```bash
git clone https://github.com/colombod/amplifier-smart-tools-audio
cd amplifier-smart-tools-audio
uv sync
```

`uv sync` creates `.venv` with the runtime dependencies and the `dev` group (`pytest`, `ruff`).

Sanity check:

```bash
uv run aud manifest
uv run aud check
```

## Lint, format, test

Exactly these three, and all three clean before a pull request:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest
```

Notes that save a round trip:

- Relative imports are banned by configuration. Use `from aud.core.plan import ...`.
- `ruff format` is the formatter; do not hand-format around it.
- CI runs `ruff check .` and `ruff format --check .` — the non-mutating forms. If you skipped
  the local run, CI fails on formatting rather than on anything interesting.

## Conformance

`aud` is an [Amplifier Smart Tool](https://github.com/microsoft/amplifier-smart-tools) and must
pass the spec's machine-checkable conformance kit. Run it from the repository root:

```bash
uv run -- uv run --no-project https://raw.githubusercontent.com/microsoft/amplifier-smart-tools/main/conformance/run.py .
```

The kit is fetched fresh each time on purpose: the bar is what the specification says today, not
what it said when someone last vendored a copy.

Two things to know before reading a failure:

- **`aud` must resolve on `PATH`.** The kit launches the tool through the `cli_argv` in
  `smart-tool.json`; the outer `uv run` supplies the project environment so that `aud` is
  findable. If you get a "not found on PATH" check failure, that is why — install the project
  (`uv sync`, or `uv tool install .`) rather than editing the descriptor.
- **A `SKIP` is not a pass.** The kit reports honestly: a rule it cannot evaluate is skipped
  with a reason. A check that starts skipping after your change has usually stopped being
  checkable — for example, making the version dynamic in `pyproject.toml` turns
  `manifest-version-matches-package` from PASS into SKIP. Treat a new SKIP as a regression.

## Before opening a pull request

1. Lint, format and tests clean, run locally.
2. Conformance run if the change touched the manifest, `smart-tool.json`, the CLI surface or
   packaging.
3. Read [AGENTS.md](AGENTS.md) — the licence rule, the library-is-the-tool rule, the
   zero-credential rule and the plan-contract rule are constraints, not style.
4. **Name the licence of any new dependency** in the pull request description. MIT, BSD, ISC,
   Apache-2.0 and PSF are fine. GPL-family is not: see [docs/VISION.md](docs/VISION.md) for why
   that is settled rather than open.
5. If the change alters the plan document, say which: an additive param stays in
   `contracts/plan.v1.md`; anything breaking needs `plan_format: 2` and a new contract file.
6. Update `CHANGELOG.md` under `## [Unreleased]`.
7. Claim only what you ran. If something is untested, say so in the description — that is
   information a reviewer can use, and a false green is not.

## Licence of contributions

By contributing you agree your contribution is licensed under the MIT licence, the same terms
as the project. See [LICENSE](LICENSE).
