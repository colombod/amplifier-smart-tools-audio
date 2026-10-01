# Recorded library behaviour — replay fixtures

These are **recordings of real runs**, not hand-written fakes. A hand-written fake
can only ever contain the fields its author already knew the code reads, which is
exactly how two defects in `detect fillers` survived a green test suite. Everything
here was produced by calling the real library and serialising whatever came back,
with attributes enumerated via `dir()`/`vars()` rather than assumed.

- **Recorded:** 2026-09-19 (UTC), in a throwaway Incus container (Digital Twin
  Universe), destroyed immediately afterwards. Nothing was installed on the host.
  (The `openai_whisper/` fixtures below were recorded differently, on 2026-10-01 — see
  that section for why.)
- **Recorded by:** `scripts/01_gen_speech.sh`, `scripts/03_record_stretch.py`,
  `scripts/04_record_end_to_end.sh`, `scripts/05_record_openai_whisper.py` — all four
  are in `scripts/`, byte-identical to what ran. (`scripts/02_record_whisper.py`, which
  recorded the now-deleted `faster_whisper/` fixtures, was deleted alongside them.)
- **No provider credential was present.** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`, `GEMINI_API_KEY` and `AZURE_OPENAI_API_KEY` were never
  forwarded into the container, and were not present when `05_record_openai_whisper.py`
  ran directly in this repo's `.venv` either. Neither library needs one.

`anthropic/` in this directory was **not** produced by this run and is not covered
by this file.

## Versions recorded

| Component | Version |
|---|---|
| `aud` | 0.8.0 (original run, `faster-whisper`/`python-stretch`); 0.12.0 (the `openai_whisper/` re-recording) |
| `openai-whisper` | 20250625 |
| `torch` | 2.14.1+cpu |
| `python-stretch` | 0.3.1 (`Signalsmith.abi3.so`, nanobind) |
| `numpy` / `scipy` / `soundfile` | 2.4.6 / 1.17.1 / 0.13.1 (current, re-recording run) |
| Python | 3.11.15 (the `openai_whisper/` re-recording); 3.12.3, Ubuntu 24.04 (original run) |
| Whisper model | `base` — the value of `aud.dsp.speech._MODEL_SIZE_DEFAULT`, unchanged |
| Compute type | CPU-only, float32 (`fp16=False` passed explicitly — see `aud.dsp.speech`'s docstring) |
| Speech source | `piper-tts`, voice `en_US-lessac-medium` (MIT) |

`faster-whisper` 1.2.1 / `ctranslate2` 4.8.2 recorded the original `faster_whisper/`
fixtures, since DELETED (issue #44) — kept here only as a historical record of what
produced the now-superseded data referenced in the old-vs-new comparison below.

## What is here

### `wav/` — the source audio

One line of deliberately filler-heavy speech, synthesised once and then resampled,
so the **content is identical across rates and only the rate differs**:

> "So, um, the first thing we need to do is, uh, check the levels. And then, erm,
> we can start recording."

| File | Rate | Duration | Notes |
|---|---|---|---|
| `speech_native.wav` | 22050 | 8.070 s | piper's direct output; everything else derives from it |
| `speech_short_{16000,22050,44100,48000}.wav` | 4 rates | 8.070 s | mono PCM16, `ffmpeg -ac 1 -ar <sr>` |
| `speech_long_16000.wav` | 16000 | 64.564 s | the native file tiled 8× (`-stream_loop 7`) |
| `nospeech_silence_16000.wav` | 16000 | 5 s | digital silence |
| `nospeech_tone_16000.wav` | 16000 | 5 s | steady 1 kHz sine |

`SHA256SUMS.txt` covers **all eleven** files generated in the container, including the
three long variants (22050/44100/48000) that were transcribed but **not shipped**, to
keep this directory small. Regenerate them from `speech_native.wav`:

```bash
for SR in 22050 44100 48000; do
  ffmpeg -y -stream_loop 7 -i speech_native.wav -ac 1 -ar $SR -c:a pcm_s16le speech_long_$SR.wav
done
```

Byte-identity of a regenerated file depends on the ffmpeg build; check against
`SHA256SUMS.txt` before assuming it.

### `openai_whisper/` — 10 transcription recordings (issue #44, superseded `faster_whisper/`)

**`faster_whisper/` (18 recordings, `faster-whisper`/`ctranslate2`) was DELETED** when the
`speech` extra's engine was replaced with `openai-whisper` (issue #44: `faster-whisper`'s `av`
dependency bundles a GPL-family FFmpeg build — see AGENTS.md section 1 and
`docs/DESIGN-ENVELOPE.md`'s "Dependency licences"). No test references it any more. This section
replaces the equivalent one that used to describe it; the "old vs new" table below is the direct
comparison, kept so the engine swap's behavioural consequences stay visible rather than silently
overwritten.

**Recorded differently from the rest of this directory**: not in a throwaway DTU container, but
directly in this repository's own `.venv` (with the `speech` extra installed per this PR) —
`tests/fixtures/recorded/scripts/05_record_openai_whisper.py`, byte-identical to what ran. No
provider credential was present or read either way (openai-whisper needs none). Recorded
2026-10-01 (UTC): `openai-whisper` 20250625, `torch` 2.14.1+cpu, Python 3.11.15, CPU only
(no CUDA used; `fp16=False` passed explicitly). Model `base` — the value of
`aud.dsp.speech._MODEL_SIZE_DEFAULT`, unchanged.

**Scope, disclosed**: only the SHORT fixture matrix (`speech_short_{16000,22050,44100,48000}`,
both `raw` and `resampled_16k` paths) plus `nospeech_silence_16000`/`nospeech_tone_16000` were
re-recorded. The `speech_long_*` recordings (faster-whisper's 64s file) were NOT re-recorded —
a real attempt on this host did not complete within several minutes of wall-clock CPU time per
file, which was judged out of budget for this change; every test that used to depend on a
`speech_long_*` recording was rewritten against the short-fixture matrix instead (see
`tests/test_dsp_detect.py` and `tests/test_recorded_awkwardness_guard.py` for exactly how). If
`speech_long_*` fixtures are wanted later, `05_record_openai_whisper.py`'s `jobs` list is the
place to add them back.

Called directly, never through `aud`, in the shape `aud.dsp.speech.detect_fillers` uses:
`whisper.load_model("base").transcribe(audio_float32, word_timestamps=True, fp16=False)`. The
same two input paths as before are recorded for every file (`__raw` / `__resampled_16k`).

`_index.json` is the summary table:

| source | path | segs | words | degenerate | last word end |
|---|---|---|---|---|---|
| short 16000 | raw | 2 | 21 | 0 | 7.72 |
| short 16000 | resampled_16k | 2 | 21 | 0 | 7.72 |
| short 22050 | raw | 2 | 21 | 0 | **10.66** (true 7.72; ×1.381 ≈ 22050/16000) |
| short 22050 | resampled_16k | 2 | 21 | 0 | 7.72 |
| short 44100 | raw | 2 | 11 | **4** | 11.4 (badly mistimed/misrecognized, not just scaled) |
| short 44100 | resampled_16k | 2 | 21 | 0 | 7.72 |
| short 48000 | raw | 3 | 26 | **5** | 23.3 (badly mistimed/misrecognized, not just scaled) |
| short 48000 | resampled_16k | 2 | 21 | 0 | 7.72 |
| nospeech silence | raw | **0** | **0** | 0 | — |
| nospeech tone | raw | 0 | 0 | — | — |

**Old (faster-whisper) vs new (openai-whisper) — same fixtures, both real, measured directly:**

| Property | faster-whisper (deleted) | openai-whisper (current) |
|---|---|---|
| Resample-path short fixtures (16/22/44/48k → 16k) | 2 segs / 21 words / 7.72s, all four identical | same: 2 segs / 21 words / 7.72s, all four identical |
| `short_22050` raw (no resample) mistiming | 10.16s (×1.378) | 10.66s (×1.381 ≈ same ratio, small recognition variance) |
| `short_44100`/`short_48000` raw (no resample) | mistimed but still mostly-correct words, some degenerate | badly mangled recognition (e.g. "Hain, a-sa-sa-la-na-ta-pa-la" gibberish) AND degenerate words — the uncorrected-rate path is unusable with either engine, worse with this one |
| Degenerate (`start == end`) words | present (13 across all runs, incl. one real degenerate FILLER word on a `speech_long_48000` resampled path) | present (9 across the short-fixture matrix: 4 on `short_44100__raw`, 5 on `short_48000__raw`) — **none of them are filler words** in this round's fixtures; no real degenerate FILLER word was captured (see the scope note above re: `speech_long_*`) |
| **Silence (`nospeech_silence_16000`)** | **hallucinated ONE word** ("the one most important behavioural quirk" the old fixtures existed to pin) | **returns ZERO segments, ZERO words** — a genuine, measured behavioural improvement; no hallucination on digital silence |
| Pure tone (`nospeech_tone_16000`) | 0 segments (already correct) | 0 segments (unchanged) |
| Word object shape | `faster_whisper.Word`, attribute-access dataclass (`.start`/`.end`/`.word`/`.probability`) | plain Python `dict` (`word["start"]`/etc.) — bridged through the new `aud.dsp.speech._WordAdapter` so `_words_to_regions` keeps its attribute-access contract |
| Word-end offset vs perceptual end | not independently re-measured this round (not practical to measure "perceptual" end without a human listener) | not independently re-measured this round, same reason — disclosed as unverified in both directions, not claimed fixed or unfixed |

What the recognised word really carries: `word`, `start`, `end`, `probability` — and nothing
else, as a plain dict (confirmed directly against a real installation, not assumed from docs).

### `python_stretch/` — the timeFactor convention, measured

`Signalsmith.Stretch()` → `.preset(1, 22050)` → set `timeFactor` → `.process()` on a
1.000 s (22050-sample) mono signal. Both directions recorded:

| `timeFactor` SET | input | output | out/in |
|---|---|---|---|
| 0.5 | 22050 | 44100 | 2.0000 |
| 0.8 | 22050 | 27563 | 1.2500 |
| 1.0 | 22050 | 22050 | 1.0000 |
| 1.2 | 22050 | 18375 | **0.8333** |
| 1.5 | 22050 | 14700 | 0.6667 |
| 2.0 | 22050 | 11025 | 0.5000 |

and the reciprocal setting (`timeFactor = 1/f`, which is what
`aud.dsp.timepitch._stretch_signalsmith` does):

| requested factor f | `timeFactor` SET | output | out/in |
|---|---|---|---|
| 0.5 | 2.000000 | 11025 | 0.5000 |
| 0.8 | 1.250000 | 17640 | 0.8000 |
| 1.0 | 1.000000 | 22050 | 1.0000 |
| 1.2 | 0.833333 | 26460 | 1.2000 |
| 1.5 | 0.666667 | 33075 | 1.5000 |
| 2.0 | 0.500000 | 44100 | 2.0000 |

`timeFactor` is the **reciprocal** of the output/input duration ratio. `aud`'s
inversion is right, and the `1.2 → 0.8333` figure in its source comment is
reproduced here exactly. Output length is `round(input / timeFactor)` with **no**
latency padding at any factor tested.

Other measured facts: `timeFactor` reads back as float32 (`0.8` → `0.800000011920929`),
so a replay must compare with a tolerance. `setTransposeSemitones(-12/-7/0/+7/+12)`
leaves length unchanged (out/in = 1.0000 in all five cases). `inputLatency`,
`outputLatency`, `blockSamples`, `intervalSamples` are **methods**, not attributes;
the only public data attributes are `sampleRate` and `timeFactor`.

`audio_pair/` holds real input/output sample data (`.npy` float32 plus `.wav`) for
`timeFactor` 0.5, 1.0 and 2.0, so a replay can assert on actual samples rather than
only on lengths.

### `end_to_end/` — captured expected output

`aud detect fillers <wav>` for each source file, with the full regions document,
exit code, stderr, and the `aud manifest` of the build that produced it. All four
short rates produce the **same** four regions (first: `um`, 0.56–0.70 s), which is the
product-level evidence that the resample fix holds. Both no-speech files produce
zero regions.

## How to re-record from scratch

```bash
# 1. One throwaway container. No host installs, no credentials forwarded.
amplifier-digital-twin launch tests/fixtures/recorded/scripts/aud-record-fixtures.yaml \
  --name aud-record-fixtures
amplifier-digital-twin check-readiness aud-record-fixtures

# 2. Push the scripts and run them in order (02_record_whisper.py, which recorded
# the now-deleted faster_whisper/ fixtures, no longer exists -- see "openai_whisper/"
# above for how that engine's fixtures are recorded instead).
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/01_gen_speech.sh /rec/01_gen_speech.sh --mode 0755
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/03_record_stretch.py /rec/03_record_stretch.py
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/04_record_end_to_end.sh /rec/04_record_end_to_end.sh --mode 0755

amplifier-digital-twin exec --stream aud-record-fixtures -- bash /rec/01_gen_speech.sh
amplifier-digital-twin exec --stream aud-record-fixtures -- \
  bash -c '$(uv tool dir)/aud/bin/python /rec/03_record_stretch.py'     # seconds
amplifier-digital-twin exec --stream --timeout 1200 aud-record-fixtures -- \
  bash /rec/04_record_end_to_end.sh

# 3. Pull back, then destroy.
amplifier-digital-twin file-pull -r aud-record-fixtures \
  /rec/out/python_stretch /rec/out/end_to_end \
  tests/fixtures/recorded/
amplifier-digital-twin destroy aud-record-fixtures
```

**`openai_whisper/` is recorded differently** (see that section above for why): with the
`speech` extra installed (`uv sync --extra speech`), run
`uv run python tests/fixtures/recorded/scripts/05_record_openai_whisper.py` directly in the
repo checkout. It writes straight into `tests/fixtures/recorded/openai_whisper/`, no DTU, no
file-push/file-pull round trip needed. The `base` model (~139 MiB) is downloaded from
HuggingFace on first use and is deliberately **not** stored here, for either engine.

`anthropic/` has no equivalent script: each of its files is one real,
successful `POST https://api.anthropic.com/v1/messages` call, captured by hand
in a session with a real credential (never committed) and saved as
`{"provenance", "request", "response"}` -- see any file in `anthropic/` for the
exact shape a new one must match. Re-capturing one is: make the real call with
the model you want to test, save the request body and the raw response JSON
verbatim in that shape, and name it `<scenario>-<model-slug>.json`.

Three more were added recording the advisor-diagnosis fix (0.12.0), each with
its driving wav shipped alongside it in `wav/` (`provenance.fixture_path`
names it, `provenance.fixture_generation` says how it was made -- pink noise,
optionally through a real `aud.lib.eq`+`render` shelf):

| File | Fixture | What it proves |
|---|---|---|
| `advise-dull-shelf-opus5.json` | `wav/dull_shelf_48000.wav` (-9 dB high shelf @ 8 kHz induced) | the advisor reaches for a HIGH shelf on a broad top-end tilt, once `eq`'s stage reference documents `shelves` at all |
| `advise-rumbly-shelf-haiku.json` | `wav/rumbly_shelf_48000.wav` (+8 dB low shelf @ 70 Hz induced) | the fix is reachable at the DEFAULT model tier, not only a top-tier one |
| `advise-clean-pink-opus5.json` | `wav/control_pink_48000.wav` (unmodified) | the same model tier that previously applied a reflexive 25 Hz high-pass to this exact file now proposes no `eq` stage at all |

## Detecting version drift

Every replay in `tests/replay.py` is only as current as the version pinned in
`provenance` above. Nothing here is re-verified automatically against the
*installed* library -- there usually isn't one on this host at all (that is
the entire reason these recordings exist). Instead, drift is made **visible**
in one place:

```bash
uv run python -c "from tests import replay; print(replay.recorded_versions())"
```

`tests/test_replay_harness.py::test_recorded_versions_are_pinned_and_visible`
pins this dict to exact literal values. When a recording is refreshed against
a newer `openai-whisper`/`torch`/`python-stretch`/Anthropic API version, that test is
meant to be **edited deliberately** as part of the same change -- a failing
assertion there means "this replay's version is no longer what the test
suite claims it is", not a flake to silence.

## What these recordings prove — and what they do not

**They do prove**, for the versions in the table above, on CPU/float32:

- The recognised word is a plain dict carrying exactly `word`, `start`, `end`,
  `probability`. A fake with a fifth key is inventing; a fake missing `probability`
  is incomplete.
- openai-whisper silently mis-times non-16 kHz audio the same way faster-whisper did
  — measured, not argued: 7.72 s of speech reported as ending at 10.66 s (22050, raw,
  unresampled) on the short fixture.
- Degenerate `start == end` words are **real**, not hypothetical, with this engine
  too. Nine of them, across the two badly-mistimed short fixtures (`short_44100__raw`:
  4, `short_48000__raw`: 5) — though NONE of this round's real degenerate words happen
  to also be filler words (unlike one of faster-whisper's, which was); see
  `tests/test_dsp_detect.py`'s synthetic-degenerate-filler-word test for how that
  specific, still-real risk is covered without a matching real recording.
- **On silence, openai-whisper DOES return nothing** — zero segments, zero words.
  This is a genuine, measured behavioural DIFFERENCE from faster-whisper (which
  returned one segment with one hallucinated word on the identical fixture). Code
  that assumed "no speech" always means "empty result" was wrong about
  faster-whisper and is right about openai-whisper — on this one fixture.
- Signalsmith's `timeFactor` is the reciprocal of the duration ratio, exactly, at
  six factors, with no latency padding (python_stretch, unaffected by this engine
  swap).

**They do not prove:**

- Anything about other versions. These are `openai-whisper==20250625` /
  `torch==2.14.1+cpu` / `python-stretch==0.3.1`. A replay built on them will keep
  passing after an upgrade breaks the real thing — the replay is only as current as
  its last re-record.
- Anything about GPU or float16. This was recorded CPU-only with `fp16=False` passed
  explicitly; word timings and hallucination behaviour can differ on other compute
  types or with `fp16=True`.
- Determinism of whisper's output, or anything about the `speech_long_*` fixture
  family (not re-recorded this round — see the scope note above). The *same* audio at
  different rates produced materially different recognition on the mistimed raw path
  even within the short fixtures (e.g. short_16000 → "So, um, ..."; short_44100 raw →
  "This man, he nourch his emotions...", unrelated gibberish), so exact-output
  assertions on a *different* input than the recorded one are unsafe.
- That `aud` handles any of it correctly. `end_to_end/` records what `aud` 0.8.0 did,
  not what it should do. If a recording and the product disagree, the recording is
  the evidence and the product is the suspect — but only after checking the
  recording was made on the input you think it was (every file carries the source
  wav's sha256 for exactly this reason).
