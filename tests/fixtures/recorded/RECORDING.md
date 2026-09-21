# Recorded library behaviour — replay fixtures

These are **recordings of real runs**, not hand-written fakes. A hand-written fake
can only ever contain the fields its author already knew the code reads, which is
exactly how two defects in `detect fillers` survived a green test suite. Everything
here was produced by calling the real library and serialising whatever came back,
with attributes enumerated via `dir()`/`vars()` rather than assumed.

- **Recorded:** 2026-09-19 (UTC), in a throwaway Incus container (Digital Twin
  Universe), destroyed immediately afterwards. Nothing was installed on the host.
- **Recorded by:** `scripts/01_gen_speech.sh`, `scripts/02_record_whisper.py`,
  `scripts/03_record_stretch.py`, `scripts/04_record_end_to_end.sh` — all four are
  in `scripts/`, byte-identical to what ran.
- **No provider credential was present.** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`, `GEMINI_API_KEY` and `AZURE_OPENAI_API_KEY` were never
  forwarded into the container. Neither library needs one.

`anthropic/` in this directory was **not** produced by this run and is not covered
by this file.

## Versions recorded

| Component | Version |
|---|---|
| `aud` | 0.8.0 (`uv tool install "aud[speech,stretch] @ git+https://github.com/colombod/amplifier-smart-tools-audio"`) |
| `faster-whisper` | 1.2.1 |
| `ctranslate2` | 4.8.2 |
| `python-stretch` | 0.3.1 (`Signalsmith.abi3.so`, nanobind) |
| `numpy` / `scipy` / `soundfile` | 2.5.3 / 1.18.1 / 0.13.1 |
| Python | 3.12.3 (Ubuntu 24.04, CPU only, no CUDA) |
| Whisper model | `base` — the value of `aud.dsp.speech._MODEL_SIZE_DEFAULT` |
| Compute type | ctranslate2 fell back to **float32** ("target device does not support efficient float16") |
| Speech source | `piper-tts`, voice `en_US-lessac-medium` (MIT) |

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

### `faster_whisper/` — 18 transcription recordings

Called directly, never through `aud`, in the shape `aud.dsp.speech.detect_fillers`
uses: `WhisperModel("base").transcribe(audio_float32, word_timestamps=True)`.

Two input paths are recorded for every file, because `aud` has one of each:

- `__raw` — the array at **its own** sample rate handed straight to `transcribe()`.
  faster-whisper has no sample-rate parameter and assumes 16 kHz regardless, so this
  is the mis-timing path (defect D1).
- `__resampled_16k` — the array put through `scipy.signal.resample_poly` to 16 kHz
  first, which is what `aud.dsp.speech._resample_to_whisper_rate` does today.

Each file carries `provenance` (versions, model, source wav sha256, ISO timestamp,
the exact snippet), `input` (duration, rate, channels, samples fed), `type_inventory`
(the full `dir()` of `TranscriptionInfo`, `Segment` and `Word`), `summary`,
and then `info`, `segments` and `words_flat` serialised in full.

`_index.json` is the summary table. Key rows:

| source | path | segs | words | degenerate | last word end |
|---|---|---|---|---|---|
| short 16000 | raw | 2 | 21 | 0 | 7.72 |
| short 22050 | raw | 1 | 20 | 0 | **10.16** (true 7.72; ×1.378 = 22050/16000) |
| short 44100 | raw | 1 | 17 | **3** | 20.32 |
| short 48000 | raw | 3 | 27 | **4** | 23.48 |
| short {16,22,44,48}k | resampled_16k | 2 | 21 | 0 | 7.72 (all four identical) |
| long 44100 | raw | 16 | 67 | **5** | 177.54 |
| long 48000 | resampled_16k | 9 | 106 | **1** | 64.26 |
| nospeech silence | raw | 1 | **1** | 0 | 4.98 |
| nospeech tone | raw | 0 | 0 | — | — |

What the `Word` object really carries: `start`, `end`, `word`, `probability` — and
nothing else. It is a **dataclass** (not a NamedTuple): `_fields` is empty,
`_asdict()` exists but is deprecated, `__dict__` is present. `TranscriptionInfo`
carries `language`, `language_probability`, `duration`, `duration_after_vad`,
`all_language_probs` (99 languages), `transcription_options`, `vad_options`.

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

# 2. Push the four scripts and run them in order.
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/01_gen_speech.sh /rec/01_gen_speech.sh --mode 0755
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/02_record_whisper.py /rec/02_record_whisper.py
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/03_record_stretch.py /rec/03_record_stretch.py
amplifier-digital-twin file-push aud-record-fixtures \
  tests/fixtures/recorded/scripts/04_record_end_to_end.sh /rec/04_record_end_to_end.sh --mode 0755

amplifier-digital-twin exec --stream aud-record-fixtures -- bash /rec/01_gen_speech.sh
amplifier-digital-twin exec --stream --timeout none aud-record-fixtures -- \
  bash -c '$(uv tool dir)/aud/bin/python /rec/02_record_whisper.py'     # ~6 min
amplifier-digital-twin exec --stream aud-record-fixtures -- \
  bash -c '$(uv tool dir)/aud/bin/python /rec/03_record_stretch.py'     # seconds
amplifier-digital-twin exec --stream --timeout 1200 aud-record-fixtures -- \
  bash /rec/04_record_end_to_end.sh

# 3. Pull back, then destroy.
amplifier-digital-twin file-pull -r aud-record-fixtures \
  /rec/out/faster_whisper /rec/out/python_stretch /rec/out/end_to_end \
  tests/fixtures/recorded/
amplifier-digital-twin destroy aud-record-fixtures
```

The whisper `base` model (~142 MiB) is downloaded from HuggingFace on first use and
is deliberately **not** stored here.

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
a newer `faster-whisper`/`python-stretch`/Anthropic API version, that test is
meant to be **edited deliberately** as part of the same change -- a failing
assertion there means "this replay's version is no longer what the test
suite claims it is", not a flake to silence.

## What these recordings prove — and what they do not

**They do prove**, for the versions in the table above, on CPU/float32:

- The `Word` object carries exactly four fields. A fake with a fifth is inventing;
  a fake missing `probability` is incomplete.
- faster-whisper silently mis-times non-16 kHz audio by `sr/16000` — measured, not
  argued: 7.72 s of speech reported as ending at 10.16 s (22050) and 23.48 s (48000).
- Degenerate `start == end` words are **real**, not hypothetical. Thirteen of them,
  across four different runs — including one on the *correct* resampled path
  (`speech_long_48000__resampled_16k`, word `" um,"` at 24.0 s), which is a filler
  word that `_words_to_regions` will drop.
- On silence, faster-whisper does **not** return nothing — it returns one segment
  with one hallucinated word. Code that treats "no speech" as "empty result" is
  wrong about this library.
- Signalsmith's `timeFactor` is the reciprocal of the duration ratio, exactly, at
  six factors, with no latency padding.

**They do not prove:**

- Anything about other versions. These are `faster-whisper==1.2.1` /
  `python-stretch==0.3.1`. A replay built on them will keep passing after an upgrade
  breaks the real thing — the replay is only as current as its last re-record.
- Anything about GPU or float16. ctranslate2 fell back to float32 here; word
  timings and hallucinations can differ on other compute types.
- Determinism of whisper's output. These are single runs. The *same* audio at
  different rates produced different word counts even on the correct path
  (long 16000 → 168 words; long 48000 resampled to 16 kHz → 106), so exact-output
  assertions on a *different* input than the recorded one are unsafe.
- That `aud` handles any of it correctly. `end_to_end/` records what `aud` 0.8.0 did,
  not what it should do. If a recording and the product disagree, the recording is
  the evidence and the product is the suspect — but only after checking the
  recording was made on the input you think it was (every file carries the source
  wav's sha256 for exactly this reason).
