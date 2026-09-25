# Design Envelope: staying clear of the patented masking/loudness formulations

This document exists because of one ruling: **we stay clear of patented work and stick to
things we can ship as part of MIT work.** That is not a clearance opinion about whether a
specific implementation infringes a specific claim -- we have not obtained one and this
document does not pretend to be one (see "Honesty and limits" at the end). It is a design
decision to avoid a family of formulations entirely, so the question of infringement never
has to be assessed in the first place.

If you are implementing or reviewing a masking/collision/ducking metric (Step 6 and
anything downstream of it) and you have not read this file, stop and read it first. The
enforced check described below (`tests/test_design_envelope_enforcement.py`) will fail your
build if you cross the line, but the check can only catch *terms*; it cannot catch *intent*.
Read this before you write the code, not after the test fails.

## The rule in one sentence

Compute masking thresholds in the **energy domain** (linear power, band energies, a
spreading matrix), from openly published psychoacoustic standards, implemented from the
specification. Never compute a loudness in **phons**. Never compute a difference between
"a source's loudness alone" and "that source's loudness in the presence of / within the
mix". Those two formulations, and the mechanisms built on them, are what the AVOID-list
below names.

## AVOID-list -- do not implement, do not port, do not "adapt"

All patent characterizations below were checked against Google Patents claim text on
2026-09-25. All seven entries are active; none has expired.

### The iZotope / Native Instruments family

Family lineage: provisional **62/516,601**, filed 2017-06-07 -> PCT/US2018/034336
(2018-05-24) -> US10396744B2 -> US10763812B2 (divisional) -> US10972065B2 -> US11469731B2.

1. **US10396744B2** (iZotope, now Native Instruments USA Inc.; granted 2019-08-27;
   anticipated expiry 2038-05-24). Claim 1 requires a **multi-track** recording, determining
   **partial loudness**, determining **loudness loss** from it, and (limb F) **creating a
   processed recording based on that loudness loss**. Claim 1 itself does not recite
   subtraction as a formula; "alone minus in-presence" is the *description's* definition of
   loudness loss, not a claim limitation in those words.
2. **US10763812B2** (same family; granted 2020-09-01; expiry 2038-05-24). **This is not a
   processing patent.** All three independent claims (1, 12, 13) are **ranking-and-display**:
   identify masking events, identify a subset with greater associated loudness loss, and
   cause a representation "to be displayed to a user on a graphical user interface." There is
   no audio-processing step in the claims. The loudness-loss formula does still appear in the
   claim language: "a difference between a loudness of the sound in the first audio recording
   occurring in isolation and a partial loudness ... occurring concurrently."
3. **US11469731B2** (same family; granted 2022-10-11; expiry 2038-05-24). Claim 1 recites
   (B) loudness of the first instrument **absent** the others, (C) loudness **in the
   presence of** the others, (D) **comparing** the two, (E) applying measures based on the
   comparison. Broader than "difference": claim 1 recites *comparing*, and difference is only
   in dependent claim 15. Dependent claim 20 treats **causing a GUI display of the comparison
   result** as itself a form of act (E) -- i.e. a metering/visualization UI can fall inside
   this claim's scope, not only an audio-processing chain.
4. **US10972065B2** -- a fourth family member. Its claims have not been examined for this
   document; it is recorded here as present and unexamined, not characterized. Treat it as in
   scope for the same avoidance until someone actually reads its claims.

**The asymmetry worth knowing, because it cuts against the intuitive reading:** '812 is
display-only and '731 counts a GUI display as a "measure" satisfying a claim limitation. That
means the **masking-meter / visualization** space is more encumbered by this family than
energy-domain audio processing is, not less. If anyone later proposes a "masking meter" UI
that ranks or displays masked events by loudness loss, that proposal needs its own scrutiny
-- staying in the energy domain for the *processing* path does not automatically clear a
*display* path built on the same partial-loudness/loudness-loss quantities.

### Two narrower, non-family patents

5. **EP2963647B1** (Harman; granted 2019-07-31). Claim 1 requires two signals tied
   specifically: the first "emanating from an environment" via an input device, the second
   "received from an audio playback device"; the system claim additionally requires the VAD
   to receive a control signal "from a voice separator." This is narrower than a generic
   "maintain a level difference between two signals" claim. It is also **lapsed in most
   states** -- in force only in DE and GB (11th-year fees paid 2025-05-20), latest possible
   expiry 2035-06-08, with a UPC opt-out registered 2023.
6. **US9881635B2** (Dolby; granted 2018-01-30; adjusted expiry 2031-04-09; priority
   2010-03-08). Claim 1 requires the attenuation control value to be indicative of
   **similarity** between speech-channel and non-speech-channel content **and** to be
   "generated based on at least one speech enhancement likelihood value" -- both conditions,
   conjunctively. Per the abstract this **scales an existing ducking gain** rather than
   triggering attenuation outright. Independent claim 9 drops "similarity" but keeps the
   speech-enhancement-likelihood limitation.
7. **US11327710B2** (Adobe; granted 2022-05-10; adjusted expiry 2038-02-01). All three
   independent claims are **conjunctive**: per-slice foreground metrics, computing entries of
   a **summed-area table** over an observation window, a total metric derived from two
   entries of that table, **and** "adding a key frame to a track" usable for audio ducking.
   Windowed foreground metrics *alone* is not what is claimed -- the summed-area-table
   construction and the keyframe-emission step are both required.

### What this means for implementation

Do not implement, port, or "adapt" (i.e. rename variables and keep the mechanism of):

- Partial loudness in phons, or anything derived from it.
- Loudness loss = a maskee's loudness alone minus its loudness in the presence of a masker
  (any framing of "alone" vs. "in the mix" / "in isolation" vs. "concurrently" / "absent"
  vs. "in the presence of," acted on to drive a processing or display decision).
- Moore-Glasberg-Baer partial-loudness models used for this purpose (the Ward/Reiss/Athwal
  2012 line of work) -- this is the mechanism the '744/'812/'731/'065 family describes.
- A VAD/attenuator that maintains a target **level difference** between two signals gated by
  where the signals come from (EP2963647B1's shape).
- Attenuating non-speech content based on **similarity to speech-channel content**, combined
  with a speech-enhancement-likelihood value (US9881635B2's shape).
- Auto-ducking keyframes derived from windowed foreground metrics via a **summed-area table**
  (US11327710B2's shape).

Also, separately from patents: do not read, port, or paraphrase the GPL/AGPL implementations
of this processor class -- see "Licence hazards" below.

## ALLOW-list -- what we build from instead

Energy-domain masking thresholds from long-published standards, implemented from the
specification, in **linear power** -- never in phons, never as a partial-loudness delta.

| Source | Date |
|---|---|
| ITU-R BS.1387 (PEAQ) | First published 1998 (BS.1387-0, 12/98); current text BS.1387-2, 05/2023 |
| Schroeder, Atal & Hall spreading function, JASA 66:1647-1652 | December 1979 |
| Terhardt, "Calculating virtual pitch," *Hearing Research* 1:155-182 | March 1979 |
| Zwicker & Terhardt Bark, JASA 68:1523-1525 | 1980 |
| Traunmuller Bark, JASA 88:97-100 | 1990 |
| Glasberg & Moore ERB, *Hearing Research* 47:103-138 | 1990 |
| ANSI S3.5 band-importance weights (SII I_i) | Approved 1997-06-06; reaffirmed through 2020 |
| MPEG-1 psychoacoustic model, ISO/IEC 11172-3 -- **as a specification only** | Published 1993-08-12 |

Every one of these was published 20-40 years before the iZotope family's 2017-06-07 priority
date. That is a meaningful defensive posture (see "Honesty and limits" below for what it does
*not* prove).

**Mandatory caveat when quoting BS.1387's spreading-function constants: name the ear model.**
`S_l = 27 dB/Bark`, `S_u = -24 - 230/f_c + 0.2*L` (dB/Bark) belong to BS.1387's **FFT-based
ear model** (eq. 15/16, Sec 2.1.7 of the 1998 text). The **filter-bank ear model in the same
standard uses different constants**: `31 dB/Bark`, and
`s = min(-4, -24 - 230/f_c + 0.2*L)`. Quoting one set of constants without naming which ear
model they belong to is a silent error -- the two models are not interchangeable and the
numbers are not the same.

**Threshold in quiet is attributed to ISO 389-7, not to Terhardt.** BS.1387 eq. (7) uses a
Terhardt-*shaped* formula (0.6 scaling factor, an `f^3.6` term), but the standard itself
attributes the threshold-in-quiet values to **ISO 389-7 (1996)**. Cite what the standard
actually cites, not the shape of the formula.

**BS.1387's own Bark scale is `z = 7*asinh(f/650)`, attributed to Schroeder, Atal & Hall
1979 -- not the Zwicker-Terhardt formula.** (This repository already anchors this
distinction: see `src/aud/dsp/bands.py`'s two named Bark realizations,
`bark_peaq` vs. `bark_zwicker_terhardt`, and `tests/test_dsp_bands_peaq_bs1387_anchor.py`.
That anchor is reused, not duplicated, by this envelope.)

## Licence hazards (a separate axis from patents)

A technique can be unpatented and still unshippable in an MIT tool, if the only available
reference *implementation* is copyleft. Do not read, port, or paraphrase code from:

| Project | Licence | Note |
|---|---|---|
| nih-plug's `spectral_compressor` plugin | GPL-3.0-or-later | The nih-plug **framework** itself is ISC -- only the `plugins/spectral_compressor` plugin and its VST3 export bindings are GPLv3. Do not label nih-plug as a whole as GPL. |
| freespacer (`psychosomaticdragon/freespacer`) | GPL-3 (full v3 text; no per-file notice, so v3-only vs. "or-later" is unstated) | Two unrelated repos share this name; this identification is an assumption, recorded here rather than left implicit. |
| spectralcarve (best match: `rcptr2/gabcis-spectralcarve-pro`) | AGPL-3 | Same caveat: identity match is an assumption, recorded rather than implicit. |
| MPEG-1 `dist10` reference code (ISO/IEC 11172-3 / ISO 13818-3 "Distribution 10", 1997) | No licence grant | Carries **no explicit grant of rights to copy, modify or redistribute**; provided without fee "as is," with warranty and non-infringement disclaimers. Fraunhofer's own included note says the files "have not officially been released for public distribution." Use the **specification only**; do not copy the reference code. |

## Dependency licences (the third axis, and an audit rather than a check)

**User ruling, 2026-09-25, verbatim: "we MUST ONLY HAVE DEPENDENCIES COMPATIBLE WITH MIT
LICENSE".**

The criterion is **MIT-compatibility**, not absence of the string "GPL" -- and that
distinction is load-bearing, not pedantic. A keyword check would fail on numpy and scipy,
this project's own DSP foundation, for the bundled GCC runtime libraries described below.

**Measured state, audited 2026-09-25** -- an untruncated scan of every licence field and OSI
classifier across all 20 resolved distributions: **copyleft 0, unknown 0, provably permissive
20.**

| Licence | Distributions |
|---|---|
| MIT | aud, pydantic, pydantic_core, pyyaml, pyloudnorm, pytest, pluggy, iniconfig, ruff, future, annotated-types, typing-inspection |
| MIT-0 | cffi |
| BSD-3-Clause | numpy, scipy, soundfile, pycparser |
| BSD-2-Clause | pygments |
| Apache-2.0 OR BSD-2-Clause | packaging |
| PSF-2.0 | typing_extensions |

**Absent by name**, confirmed: `pedalboard`, `matchering`, `rubberband`, `librosa`,
`essentia`, `pyrubberband`, `soundstretch`.

**Bundled runtimes -- present on disk in the wheels, and not visible in any metadata field.**
`scipy.libs/` and `numpy.libs/` ship:

| Bundled library | Size | Licence | Why it is MIT-compatible as used |
|---|---|---|---|
| `libgfortran-*.so.5.0.0` | 2767 KB (also a 2651 KB variant in scipy) | **GPL-3.0-or-later WITH GCC-exception-3.1** | The GCC Runtime Library Exception exists precisely to permit GCC runtime libraries to be carried into a program under any licence, including proprietary, without propagating GPL terms to it. |
| `libquadmath-*.so.0.0.0` | 265 KB / 245 KB | **LGPL-2.1-or-later** | Dynamic linking under LGPL-2.1 does not relicense the linking program -- narrower than "no obligation at all," which would be wrong: **section 6 places notice and source/relink conditions on whoever distributes the combined work.** Those conditions sit with numpy and scipy, not with aud, because **this project does not redistribute these binaries** -- pip/uv fetches their wheels directly from PyPI. Note also that libquadmath does **not** carry the GCC Runtime Library Exception that covers libgfortran above; this row rests on dynamic linking plus non-redistribution alone, and is the weaker of the two arguments. |
| `libscipy_openblas*.so` | ~24 MB | BSD-3-Clause | Permissive outright. |

Both copyleft-family entries are MIT-compatible **as used**. That is the whole reason the
criterion is stated as MIT-compatibility: the honest answer here is "GPL-3.0-or-later is
present on disk and is fine," which a string match cannot reach.

**The method trap, recorded because it nearly produced a false clean result.** A first audit
truncated licence strings to 45 characters and reported a clean tree, **missing the bundled
runtimes entirely** -- because scipy's `License` field is **47,559 characters**: the BSD text
plus a bundled-third-party section. A length-capped lexical scan of licence metadata is
unreliable; read the whole field.

**This is an audit, not an enforced check.** Nothing in the build fails if an incompatible
dependency is added tomorrow. The vocabulary guard described below reads `src/` only -- it
does not read `pyproject.toml`, `uv.lock` or `importlib.metadata`, and it will not notice a
dependency at all. Enforcement is tracked as work item `smart_tools-c53`.

## What Step 6's metric must demonstrably do

The collision/masking metric must operate in the **energy domain**: linear power, band
energies (`E_{m,k}`), and a spreading matrix (`S`) applied in linear power. It must compute
**no** loudness in phons, and it must compute **no** difference (or comparison, in the
US11469731B2 sense -- see above, "comparing" is broader than "difference") between a
source's loudness alone and its loudness in the presence of / within the mix.

**The forbidden *terms* are mechanically enforced; the semantic constraint above is not
mechanically checkable and rests on review.** What runs is a vocabulary guard, not a semantic
one: a function that computes the exact patented shape under neutral names passes it cleanly --

```python
def _masking_penalty(db_solo: float, db_together: float) -> float:
    return db_solo - db_together
```

-- so nothing but a reviewer stands between that code and the codebase. This is the same limit
stated at the top of this document (the check "can only catch *terms*; it cannot catch
*intent*"), restated here where the check itself is described, so the two readings cannot
drift apart.

**`tests/test_design_envelope_enforcement.py`** scans every `.py` file under `src/` for the
forbidden vocabulary (`partial_loudness`, `loudness_loss`, `phon`/`phons` as a unit, and the
alone-vs-in-mix/isolation-vs-presence comparison naming), across identifier and prose
spellings (`snake_case`, `kebab-case`, plain words), case-insensitively. It deliberately does
**not** flag "phon" as a substring of an unrelated word (`phone`, `microphone`) -- see that
test file's own docstring for how it avoids that false positive, and its module-level
constant `_KNOWN_FALSE_POSITIVE_WORDS` naming the two words (across three source lines) that
exist in this codebase today.

**The escape hatch, and why it is shaped this way.** A line that matches a forbidden pattern
is allowed through only if that exact line carries the literal marker
`# DESIGN-ENVELOPE-EXCEPTION: <reason>`. That marker is deliberately **inline with the flagged
code**, not in a side-table or a config file, so:

- it shows up in the diff of the PR that introduces it, where a reviewer is already looking;
- it is greppable (`grep -rn "DESIGN-ENVELOPE-EXCEPTION" src/`) without needing to know this
  document exists;
- it cannot be added silently -- the marker text itself names this document by convention
  (any reviewer who sees it and doesn't know what it means has this file one search away).

There is no exception on record today. Adding one is a decision to argue for in a PR
description, not a mechanical workaround for a failing test.

## Telling a future contributor "no" specifically enough that they don't re-derive this

If you are about to propose a "better" masking metric that uses perceptual loudness rather
than energy: the reason this was rejected is not "energy-domain is better DSP." It may well
not be. The reason is that **the specific mechanism of comparing a masked source's loudness
alone against its loudness in a mix, in phons, to drive a processing or display decision, is
the subject of an active patent family (US10396744B2 / US10763812B2 / US11469731B2 /
US10972065B2, priority 2017-06-07, running to 2038-05-24)** -- see the AVOID-list above for
exactly which claim elements matter and why the corrected, narrower readings still catch this
shape. You do not need to re-derive the psychoacoustics literature to know this: the
citations in the ALLOW-list above are the openly published alternative, and they predate the
patent family by 20-40 years.

## Honesty and limits

**This is design rationale. It is not a legal clearance.** Nobody involved in writing this
document is qualified to render a clearance opinion, and none was sought. The prior-art
lineage -- every technique on the ALLOW-list published 20-40 years before the 2017 priority
date -- is a reason to **prefer** these formulations over the patented ones. It is **not**
proof that no claim reads on our specific combination of energy-domain masking, banding, and
whatever gain law is applied downstream of it. Avoiding the specific mechanism the patents
describe is a materially different, and much cheaper, activity than obtaining a clearance
opinion that no claim could possibly apply -- and this document only claims to do the former.

The verification behind this document also has limits, recorded here rather than left
implicit:

- The EPO Register and USPTO Patent Center were not reachable during verification. Expiry
  dates above are **Google Patents estimates**; patent-term adjustments and terminal
  disclaimers were not independently checked against the primary registers.
- Google Patents labels its own computed status fields "an assumption and is not a legal
  conclusion." That caveat applies to every status/expiry claim sourced from it above.
- The Terhardt 1979 paper and the ANSI S3.5 tables are paywalled and were not read directly;
  their dates and roles above are taken from citation records (including BS.1387's own
  citation of Terhardt), not from reading the primary text.

If a future contributor needs an actual freedom-to-operate opinion, that is a different
activity than this document, requiring counsel -- this document's job is to make sure the
question is avoided by design rather than needed at all.
