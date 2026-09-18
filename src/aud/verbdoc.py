"""Full prose documentation for every aud verb.

`aud <verb> --help` prints VERB_DOCS[verb] verbatim; `aud <verb> -h` falls
through to argparse's terse, auto-generated usage instead. Keep every entry
here in sync with the verb's actual arguments in cli.py.
"""

from __future__ import annotations

VERB_DOCS: dict[str, str] = {
    "analyze": """\
analyze -- measure what is actually in a file

What it does:
  Reads a finished audio file and reports honest measurements: integrated
  loudness (LUFS), true peak (dBTP), crest factor, per-band spectral
  balance, sibilance energy, and ambience/reverb tail. It changes nothing.

When to reach for it:
  Before touching a file at all -- "how loud is this really", "is there
  clipping risk", "does this sound boxy" are all answered by analyze, not
  by guessing from ear alone.

Parameters:
  path (positional)  Path to the audio file to measure.

Example:
  aud analyze podcast_ep12.wav
""",
    "plan": """\
plan -- start (or load) a mastering plan

What it does:
  Prints an empty mastering plan document to stdout, or loads an existing
  one from a file with --from. A plan is inert data: a list of stages with
  their parameters. Nothing is rendered until 'aud render' consumes it.

When to reach for it:
  The first verb in every pipeline. Every stage verb after it reads a plan
  on stdin and writes an updated plan back out.

Parameters:
  --from PATH   Load an existing plan document from PATH instead of
                starting empty.

Example:
  aud plan | aud deess --amount 6 | aud render in.wav out.wav
""",
    "deess": """\
deess -- repair stage: tame harsh sibilance

What it does:
  Appends a de-essing stage to the plan on stdin. At render time this
  reduces energy in the sibilant band (around --freq) when it spikes above
  the rest of the signal, by up to --amount dB.

When to reach for it:
  "Get rid of the harsh S sounds", vocal or spoken-word recordings with
  bright, splashy sibilance.

Parameters:
  --amount FLOAT   Maximum reduction in dB, 0-24. Default 6.
  --freq FLOAT     Center frequency of the sibilant band in Hz. Default 6000.

Example:
  aud plan | aud deess --amount 8 --freq 7000 | aud render in.wav out.wav
""",
    "dereverb": """\
dereverb -- repair stage: reduce room ambience

What it does:
  Appends a de-reverb stage to the plan on stdin. At render time this
  reduces the reverberant tail of a recording made in an untreated room.

When to reach for it:
  "The room sounds boxy", recordings made in a live or reflective space
  that need to sound drier and closer.

Parameters:
  --amount FLOAT   Percentage of ambience reduction, 0-100. Default 50.

Example:
  aud plan | aud dereverb --amount 60 | aud render in.wav out.wav
""",
    "eq": """\
eq -- tone stage: parametric equalization

What it does:
  Appends a parametric EQ stage to the plan on stdin: an optional
  high-pass corner, an optional low-pass corner, and any number of peaking
  bands (repeat --peak for each one).

When to reach for it:
  "Tighten the low end" (--hpf), "roll off the hiss" (--lpf), or correcting
  a specific frequency problem with a peaking band.

Parameters:
  --hpf FLOAT           High-pass corner frequency in Hz. Must be > 0, and
                        below --lpf if both are given.
  --lpf FLOAT           Low-pass corner frequency in Hz. Must be > 0.
  --peak FREQ,GAIN,Q    A peaking band as freq_hz,gain_db,q. Q must be > 0.
                        Repeatable.

Example:
  aud plan | aud eq --hpf 40 --peak 3200,-2.5,1.4 --peak 120,1.5,0.8 | aud render in.wav out.wav
""",
    "eq-match": """\
eq-match -- tone stage: match a reference's tonal balance

What it does:
  Appends an EQ-match stage to the plan on stdin, using a spectral curve
  previously produced by 'aud curve extract'. --mix controls how much of
  the match is applied.

When to reach for it:
  "Make this podcast match last week's episode", matching a new take to an
  established reference recording.

Parameters:
  --curve PATH   Path to a curve JSON file produced by 'aud curve extract'.
                 Required.
  --mix FLOAT    How much of the match to apply, 0.0-1.0. Default 1.0.

Example:
  aud curve extract reference.wav ref_curve.json
  aud plan | aud eq-match --curve ref_curve.json --mix 0.8 | aud render in.wav out.wav
""",
    "curve": """\
curve -- extract or apply a spectral curve

What it does:
  Two subcommands, neither of which touches a plan:
    curve extract PATH OUT          Save PATH's spectral profile to OUT (JSON).
    curve apply PATH CURVE OUT      Apply CURVE (JSON) to PATH, writing OUT.

When to reach for it:
  "Extract the EQ curve of a recording you like and apply it to another
  file" -- when you want to reuse a tonal balance without going through
  eq-match's plan-based pipeline, e.g. to inspect or share the curve itself.

Parameters:
  extract PATH OUT     PATH to analyze, OUT is the curve JSON to write.
  apply PATH CURVE OUT PATH to process, CURVE is the curve JSON to apply,
                       OUT is the rendered file to write.

Example:
  aud curve extract reference.wav curve.json
  aud curve apply in.wav curve.json out.wav
""",
    "compress": """\
compress -- dynamics stage: multiband compression

What it does:
  Appends a multiband compression stage to the plan on stdin. --bands are
  Linkwitz-Riley crossover frequencies (strictly ascending, below the
  Nyquist frequency at 44100 Hz); the signal is split into len(bands) + 1
  bands, each compressed independently at --ratio, then recombined.

When to reach for it:
  "Even out dynamics per frequency band" -- when a single full-band
  compressor squeezes the whole signal together instead of just the band
  that actually needs it. Multiband is the default for dynamics in this
  tool, not an opt-in.

Parameters:
  --bands F1,F2,...   Ascending crossover frequencies in Hz. Required.
  --ratio FLOAT       Compression ratio per band, > 0. Default 2.5.

Example:
  aud plan | aud compress --bands 120,900,5500 --ratio 2.5 | aud render in.wav out.wav
""",
    "saturate": """\
saturate -- character stage: harmonic saturation

What it does:
  Appends a saturation stage to the plan on stdin: adds harmonic
  distortion controlled by --drive, blended in at --mix.

When to reach for it:
  Adding warmth, weight or "glue" to a mix or master that sounds clean but
  thin or sterile.

Parameters:
  --drive FLOAT   Saturation drive, > 0. Higher is more distortion. Default 1.0.
  --mix FLOAT     Wet/dry blend, 0.0-1.0. Default 0.25.

Example:
  aud plan | aud saturate --drive 1.5 --mix 0.25 | aud render in.wav out.wav
""",
    "reverb": """\
reverb -- character stage: controlled ambience

What it does:
  Appends a reverb stage to the plan on stdin: adds a controlled amount of
  ambience with a given decay time, the opposite of dereverb.

When to reach for it:
  A recording is too dry/close and needs a touch of room to sit naturally.

Parameters:
  --amount FLOAT   Wet amount, 0.0-1.0. Default 0.2.
  --decay FLOAT    Decay time in seconds, > 0. Default 1.5.

Example:
  aud plan | aud reverb --amount 0.15 --decay 1.2 | aud render in.wav out.wav
""",
    "stretch": """\
stretch -- retime a programme without changing its pitch

What it does:
  Appends a time-stretch stage to the plan on stdin. --factor 1.0 is no
  change; > 1.0 lengthens, < 1.0 shortens.

When to reach for it:
  Fitting a programme to a fixed slot length, or nudging tempo without
  affecting pitch.

Parameters:
  --factor FLOAT   Stretch factor, > 0. Default 1.0.

Example:
  aud plan | aud stretch --factor 0.98 | aud render in.wav out.wav
""",
    "pitch": """\
pitch -- re-pitch a programme without changing its timing

What it does:
  Appends a pitch-shift stage to the plan on stdin, in semitones.

When to reach for it:
  Correcting or creatively shifting pitch without altering duration.

Parameters:
  --semitones FLOAT   Shift amount, -24 to 24. Default 0.0.

Example:
  aud plan | aud pitch --semitones -2 | aud render in.wav out.wav
""",
    "loudness": """\
loudness -- set the loudness stage's target

What it does:
  Appends a loudness-targeting stage to the plan on stdin: at render time,
  gain is applied so the integrated loudness lands at --target LUFS.

When to reach for it:
  "This is too quiet for YouTube", "hit -14 LUFS", "level these files to
  each other".

Parameters:
  --target FLOAT   Target integrated loudness in LUFS, -60 to 0. Default -14.0.

Example:
  aud plan | aud loudness --target -14 | aud limit --ceiling -1.0 | aud render in.wav out.wav
""",
    "limit": """\
limit -- set the true-peak brickwall ceiling

What it does:
  Appends a true-peak limiting stage to the plan on stdin: at render time,
  a brickwall limiter keeps inter-sample peaks at or below --ceiling dBTP.

When to reach for it:
  Always, as the last stage before render, to guarantee no clipping or
  inter-sample overs -- especially right after a loudness stage.

Parameters:
  --ceiling FLOAT   True-peak ceiling in dBTP, must be <= 0.0. Default -1.0.

Example:
  aud plan | aud loudness --target -14 | aud limit --ceiling -1.0 | aud render in.wav out.wav
""",
    "render": """\
render -- apply a whole plan to a file in one pass

What it does:
  Reads the plan on stdin, reorders its stages into canonical mastering
  order (repair -> tone -> dynamics -> character -> loudness -> limit)
  regardless of append order, and applies all of them in a single
  decode/filter/encode pass. Prints a {"result": ...} envelope, not a plan
  -- it is the end of the pipeline.

When to reach for it:
  The last verb in every chain. Never render each stage separately: every
  extra render is another round of quantization and another chance to clip.

Parameters:
  in_path (positional)   Source audio file.
  out_path (positional)  Destination audio file to write.

Example:
  aud plan | aud eq --hpf 40 | aud limit --ceiling -1.0 | aud render in.wav out.wav
""",
    "verify": """\
verify -- measure a render against the targets it was asked for

What it does:
  Measures a file and, if --target/--ceiling are given, reports whether it
  actually meets them (lufs_ok, ceiling_ok).

When to reach for it:
  "Verify that a finished render actually meets the loudness and ceiling it
  was asked for" -- confirming a render before shipping it.

Parameters:
  path (positional)   File to verify.
  --target FLOAT      Expected integrated loudness in LUFS, if any.
  --ceiling FLOAT     Expected true-peak ceiling in dBTP, if any.

Example:
  aud verify out.wav --target -14 --ceiling -1.0
""",
    "preset": """\
preset -- named chains for common destinations

What it does:
  Named, pre-built mastering chains for common destinations (e.g. podcast
  hosting, streaming platforms) that would otherwise be built by hand with
  plan/deess/eq/compress/loudness/limit.

Status:
  Not yet built in this release. Every deterministic verb it would chain
  together already works standalone; use them directly until presets land.

Example:
  aud preset --list   # (planned)
""",
    "check": """\
check -- report what this host has installed and can reach

What it does:
  Reports Python version, whether the core DSP dependencies import, whether
  ffmpeg is reachable on PATH, and which AI-provider credential (if any) is
  set -- by name and boolean only, never by value.

When to reach for it:
  Diagnosing why a verb behaves unexpectedly, or confirming a host is ready
  before scripting against it.

Parameters:
  (none)

Example:
  aud check
""",
    "config": """\
config -- report effective settings and where each came from

What it does:
  Reports the effective value of every setting aud knows about, and which
  tier it was resolved from: argument > config file
  (~/.config/aud/config.toml) > environment (AUD_*) > built-in default.

When to reach for it:
  Debugging "why is my render using -14 LUFS when I set AUD_DEFAULT_TARGET_LUFS".

Parameters:
  --sample-rate-policy STR     Override the sample_rate_policy setting.
  --default-ceiling-dbtp FLOAT Override the default_ceiling_dbtp setting.
  --default-target-lufs FLOAT  Override the default_target_lufs setting.
  --oversample INT             Override the oversample setting.
  --output-subtype STR         Override the output_subtype setting.

Example:
  aud config
""",
    "manifest": """\
manifest -- print this tool's validated manifest

What it does:
  Parses and validates the packaged SMART_TOOL.md frontmatter and prints it
  as a {"result": ...} envelope.

When to reach for it:
  Confirming what aud claims to be able to do, or checking its declared
  requirements, straight from the source of truth.

Parameters:
  (none)

Example:
  aud manifest
""",
    "advise": """\
advise -- read the measurements and say what the chain should be

What it does:
  Analyzes a file, then asks a model to recommend a mastering chain and
  explain why, without applying it.

Requires:
  One of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY
  or AZURE_OPENAI_API_KEY.

Status:
  Not yet built in this release. Use 'aud analyze' plus the deterministic
  stage verbs to build a chain by hand until this lands.

Example:
  aud advise in.wav   # (planned)
""",
    "master": """\
master -- choose the chain, apply it, and verify the result

What it does:
  The one-shot model-backed path: analyzes a file, decides on a chain,
  renders it, and verifies the result against sensible targets.

Requires:
  One of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY
  or AZURE_OPENAI_API_KEY.

Status:
  Not yet built in this release. Use 'aud plan | ... | aud render' plus
  'aud verify' to get the same result by hand until this lands.

Example:
  aud master in.wav out.wav   # (planned)
""",
}
