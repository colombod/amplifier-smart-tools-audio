# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The plan document has its
own version, independent of the package version — see [contracts/plan.v1.md](contracts/plan.v1.md).

## [Unreleased]

## [0.1.0] - 2026-09-18

Initial scaffold: the package definition, the manifest, the contracts and the documentation that
describe what `aud` is. Implementation of the DSP and the CLI verbs is in progress alongside
this entry.

### Added

- **Package definition** — `pyproject.toml`, MIT, Python 3.11+, static version. Runtime
  dependencies are permissively licensed only: `numpy` (BSD), `scipy` (BSD), `soundfile`
  (BSD-3), `pyloudnorm` (MIT), `pydantic` (MIT), `pyyaml` (MIT). Optional `stretch` extra backed
  by `python-stretch` (Signalsmith Stretch, MIT).
- **Smart tool manifest** — `src/aud/SMART_TOOL.md` and the `smart-tool.json` descriptor:
  mastering and cleanup of a finished stereo or mono programme, deterministic verbs plus two
  model-backed ones (`advise`, `master --auto`), with `ai-provider` and `ffmpeg` declared as
  optional requirements.
- **Plan document contract, format 1** — `contracts/plan.v1.md`. Top-level shape, the eleven
  canonical stage names and their order, the parameter schema of every stage, the failure codes,
  what is promised, what is not, and the rule that a breaking change means `plan_format: 2`.
- **Documentation** — `docs/VISION.md` (the problem, the audience, the licence-over-convenience
  decision and its cost, what the tool refuses to do), `docs/ARCHITECTURE.md` (plan document as
  the central contract, canonical ordering, library-is-the-tool, the single-render rule, chain
  topology, module layout, where intelligence attaches), `docs/CONFIGURATION.md` (four-tier
  settings resolution, inverted credential resolution, the absent/null/wrong-type distinction).
- **Repository conventions** — `AGENTS.md` and `CONTRIBUTING.md`: the forbidden GPL-family
  dependencies named explicitly, the zero-credential requirement and how it is proven, the
  error-with-remedy shape, the ban on hardcoded model names, and the lint/format/test commands.
- **CI** — three jobs: the test suite with an explicit assertion that no provider credential is
  present in the environment; the external conformance kit run against this commit; and an
  install-from-git job, which is the only job that would notice a git reference that works
  locally and nowhere else.
- **MIT licence**, `.gitignore`, `.python-version` pinned to 3.11.

### Verified

Stated honestly, because the code is being written alongside these documents.

**Established by research, before any of it was written down here:**

- The mature ready-made options are GPL-family and therefore excluded on purpose: `pedalboard`
  is GPL-3.0, `matchering` is GPL-3.0, Rubber Band is GPL-2.0-or-later with a commercial dual
  licence. The permissive alternatives (`numpy`, `scipy`, `soundfile`, `pyloudnorm`,
  `python-stretch`) are what the dependency list is built from.
- Topology facts the architecture rests on: Linkwitz-Riley 4th-order crossovers recombine flat;
  a single full-band limiter after recombination is the only place a true-peak ceiling can be
  guaranteed, because independently limited bands can sum above it; true peak requires
  oversampling to detect; loudness normalisation is gain only and guarantees no ceiling; EBU R128
  recommends a maximum true peak of −1 dBTP.

**Not yet verified — no claim is made:**

- The test suite has not been run. There is no passing-suite claim attached to this release.
- The conformance kit has not been run against this repository. `pass`/`fail` counts are
  unknown.
- CI has never executed: the workflow is written but this repository has no remote yet, so no
  job in it has produced a result.
- No audio has been rendered. No measured output — loudness, true peak, or otherwise — exists to
  compare against a target, so nothing is claimed about what the chain sounds like or whether it
  hits the numbers it is asked for.
- `uv build`, wheel installation and install-from-git have not been exercised for this package.

[Unreleased]: https://github.com/colombod/amplifier-smart-tools-audio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/colombod/amplifier-smart-tools-audio/releases/tag/v0.1.0
