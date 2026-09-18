# Architecture

The mechanics. Why this tool exists and what it refuses to do is [VISION.md](VISION.md).

## 1. The plan document is the central contract

Nothing in `aud` is a hidden session state. A chain is a JSON document:

```json
{
  "plan_format": 1,
  "created_with": "aud 0.1.0",
  "stages": [
    {"stage": "deess", "params": {"amount_db": 6.0}},
    {"stage": "loudness", "params": {"target_lufs": -14.0}},
    {"stage": "limit", "params": {"ceiling_dbtp": -1.0}}
  ]
}
```

Every stage verb reads a plan on stdin, appends exactly one stage, and writes the plan back to
stdout. `render` is the only verb that opens an audio file for processing. The full schema —
what is promised, what is not, and how it versions — is [contracts/plan.v1.md](../contracts/plan.v1.md).

Three things follow from this, and they are the reason for the design:

- **A chain is reviewable before it runs.** Particularly important when a model proposed it.
- **A chain is reusable.** The same document renders episode 1 and episode 40, byte-identically
  in its parameters.
- **A chain is diffable.** "What changed between these two masters" is a text diff, not an
  archaeology exercise.

## 2. Canonical stage ordering

Mastering is an ordered chain. The order stages were *appended* is a property of how someone
typed a pipeline; the order they are *applied* is a property of what the signal needs. `render`
sorts stages into canonical order and states in its report that it did so:

```
["stretch", "pitch", "dereverb", "deess", "eq", "eq_match",
 "compress", "saturate", "reverb", "loudness", "limit"]
```

Grouped, that is: **repair → tone → dynamics → character → loudness → limiting.**

The ordering is not arbitrary. Repair before tone, because de-essing a resonance you are about
to cut wastes gain reduction. Tone before dynamics, because a compressor's detector hears the
EQ. Loudness before limiting, because loudness is a gain change and the limiter must be the last
thing that sees the signal. Anything appended after `limit` in a pipeline still renders before
it — there is nothing downstream of the ceiling.

## 3. The library is the tool

`aud.lib` holds every capability. The CLI parses arguments, calls the library, and serialises
the result to JSON. That is all it does.

**A capability that exists only in the CLI wrapper is a defect**, not a shortcut. If parameter
validation, default resolution, stage ordering or report formatting lives in `cli.py`, then the
Python caller gets a different tool from the shell caller, and the MCP or SDK adapter someone
writes later gets a third. One implementation, several thin surfaces over it.

The test for whether the boundary has been respected: could you delete `cli.py` and lose nothing
but argument parsing?

## 4. The single-render rule

The whole chain is applied in **one pass**: one decode, one filter graph, one encode.

Running each stage as its own render would be simpler to implement and is wrong. Each render is
another quantisation to the output subtype, another round of dither or truncation noise, and
another opportunity for an intermediate stage to exceed full scale and clip on write — with the
clipping baked in before the limiter ever sees it. Stage-by-stage rendering also destroys the
guarantee the limiter exists to provide: a ceiling enforced on an intermediate file says nothing
about the ceiling of the final one.

Internally, samples are carried as float64 from decode to encode and quantised exactly once, at
write.

## 5. Chain topology

```
  in.wav
    |
  [decode]  io: libsndfile, or ffmpeg for compressed formats
    |
    |  float64, channels preserved
    v
  REPAIR ......  dereverb -> deess
    |
  TONE ........  eq (biquad cascade) -> eq_match (measured curve -> filter bank)
    |
  DYNAMICS ....  multiband compression
    |            +-----------------------------------------------+
    |            |  Linkwitz-Riley 4th-order crossovers           |
    |            |  (two cascaded 2nd-order Butterworths/branch)  |
    |            |                                               |
    |            |        +--> band 0 --> gain computer 0 --+    |
    |            |        +--> band 1 --> gain computer 1 --+    |
    |   ---------+--------+--> band 2 --> gain computer 2 --+--(sum)--+
    |            |        +--> band N --> gain computer N --+    |    |
    |            +-----------------------------------------------+    |
    |                                                                 |
    v  <--------------------------------------------------------------+
  CHARACTER ...  saturate -> reverb
    |
  LOUDNESS ....  measure (ITU-R BS.1770 / pyloudnorm), apply ONE gain
    |
  LIMIT .......  single full-band, oversampled, lookahead true-peak brickwall
    |
  [encode]  quantise once, to the configured output subtype
    |
  out.wav  +  a report of what was measured and what was applied
```

### Why multiband, and why Linkwitz-Riley

Multiband is the default for dynamics, not an option bolted on. One full-band gain computer
means a kick drum ducks the vocal and a loud S ducks the bass. Splitting the signal lets each
region be controlled by its own energy. A single-band squeeze is what you get by asking for one
band, deliberately.

The crossover is Linkwitz-Riley 4th-order — two cascaded 2nd-order Butterworth sections per
branch. That is the standard choice because the summed low and high branches recombine with flat
magnitude, so a band with no gain reduction applied returns the original signal rather than a
peak or a notch at the crossover frequency. Bands are recombined by summation.

### Why exactly one limiter, on the sum

Compression is multiband; **limiting is not.** The brickwall runs once, full-band, after
recombination.

The reason is arithmetic: per-band brickwall limiting does not guarantee the final ceiling,
because independently limited bands can sum above it. Three bands each held at −1 dBTP can sum
to well above −1 dBTP. The only place a ceiling can be enforced is the last point where the
signal exists as one stream.

This is the same split the established tools make: FabFilter separates Pro-MB (multiband
dynamics) from Pro-L2 (the final limiter); iZotope Ozone separates the Dynamics module from the
Maximizer. It is not a simplification we made — it is the correct topology.

### True peak is not sample peak

A sample peak is the largest value in the array. A **true peak** is the largest value the
reconstructed analogue waveform reaches between samples, and it can exceed the sample peak. You
cannot see it without oversampling: upsample (4× by default), measure there, and control there.
It matters because downstream lossy encoders reconstruct that inter-sample peak, and a file that
measures 0.0 dBFS at sample rate can clip a consumer's decoder.

The limiter therefore works oversampled and with lookahead, so gain reduction is applied
*before* the peak arrives rather than after it. EBU R128 recommends a maximum true peak of
−1 dBTP, which is the default ceiling here.

### Loudness normalisation is gain, and gain only

The `loudness` stage measures integrated loudness and applies a single gain to reach the target.
It does not and cannot guarantee a ceiling — raising a programme by 6 dB raises its peaks by
6 dB. Enforcing the ceiling is the limiter's job, which is why `limit` is downstream of
`loudness` in canonical order and why `verify` measures the finished file rather than trusting
the arithmetic.

## 6. Module layout

Two packages under `src/aud/`. The rule that separates them: **`dsp/` takes arrays and
parameters and returns arrays.** It never reads configuration, never touches the filesystem,
never raises a user-facing error, and does not know that plans exist.

| Module | Responsibility |
|---|---|
| `core/io` | Decode and encode. libsndfile via `soundfile`; ffmpeg for compressed formats. Float64 in memory, quantise on write. |
| `dsp/filters` | Biquad primitives: peaking, shelf, high-pass, low-pass. The building block everything tonal is made of. |
| `dsp/crossover` | Linkwitz-Riley 4th-order band splitting and flat recombination. |
| `dsp/dynamics` | Envelope detection and gain computers: threshold, ratio, knee, attack, release. Per band. |
| `dsp/limiter` | Oversampled, lookahead true-peak brickwall. One instance, full-band, last. |
| `dsp/saturation` | Waveshaping with drive and dry/wet mix. |
| `dsp/loudness` | ITU-R BS.1770 measurement (`pyloudnorm`) and the gain that reaches a target. |
| `core/analysis` | The measurement report: loudness, true peak, crest factor, band energies, sibilance, ambience. What `analyze` returns and what the model reads. |
| `core/engine` | Validate the plan, sort into canonical order, build the graph, run it once, emit the report. |

`core/plan`, `core/config` and `core/errors` carry the plan document, settings resolution
(see [CONFIGURATION.md](CONFIGURATION.md)) and the `{code, message, remedy}` error shape.

## 7. Where intelligence attaches

```
   analyze  ------>  measurements (JSON)
                          |
                          v
                     advise / master --auto     <-- the only model call
                          |
                          v
                     a plan document
                          |
                          v
                     render (deterministic)  ------>  audio
```

`advise` and `master --auto` read the measurement report and choose stages and parameters, with
a stated reason. **They never touch samples.** Their entire output is a plan document, which
then goes through the same engine as a plan typed by hand.

That boundary is deliberate and load-bearing:

- A model failure produces a bad *plan*, visible before rendering, rather than corrupted audio.
- The expensive path is bounded — one call on a few kilobytes of measurements, not on audio.
- Everything else in the tool keeps working with no provider configured, because nothing else
  in the tool has a model in its call path.
- The model is chosen by configuration, never hardcoded, so the same tool runs against whichever
  provider the host already has.

Model backends are imported lazily, inside the two verbs that need them. Importing a provider
at module load would make the deterministic verbs depend on a provider being installed, which
is precisely the property this tool claims not to have.
