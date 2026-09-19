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
    "detect": """\
detect -- find things in the audio, without changing anything

What it does:
  Three read-only detectors. Each one measures the file and prints a
  REGIONS document (contracts/regions.v1.md) -- not a plan. A regions
  document says where things are: {start_s, end_s} plus fields specific to
  what was found.

    detect transients PATH   Onset positions, with a strength per onset.
    detect silence PATH      Quiet spans, with peak and RMS level.
    detect fillers PATH      "umm", "uh", "ehm" and long hesitations.

  Nothing here appends to a mastering plan and nothing here writes a file.

When to reach for it:
  "Where are the pauses", "get rid of the umms and ehms", "tighten up this
  recording". Pipe the result into 'aud cut' to remove what was found, in
  the same command:

    aud detect silence in.wav | aud cut | aud render in.wav out.wav

The silence threshold is relative, on purpose:
  --threshold is dB ABOVE this file's MEASURED noise floor, not an absolute
  dBFS level. A fixed dBFS threshold is correct for exactly one recording
  -- the one it was tuned on. A treated room may floor at -70 dBFS and a
  phone in a kitchen at -38 dBFS; one fixed number finds nothing in the
  first file and eats words in the second. The measured floor is reported
  in the document, so the judgement can be checked.

Why transients matter even when you only asked about silence:
  A silence boundary that sits right next to an onset must not be trimmed
  INTO the onset -- that truncates an attack, which is the one editing
  mistake that always sounds like a fault rather than a choice. So onset
  information acts as an advisory constraint on where a silence boundary
  may be reported; it is never a source of silence regions itself. The
  same knowledge is what 'cut --snap transient' uses at render time.

'detect fillers' needs the speech extra:
  Word-level timings need speech recognition. That is the optional
  'speech' extra (faster-whisper, MIT). It is a LOCAL model -- no AI
  provider, no credential, no network call at run time. With the extra
  absent, 'detect fillers' REFUSES and names it; it never falls back to an
  energy-only guess, because those regions would look like words and 'cut'
  would remove them.

Parameters:
  transients PATH  --sensitivity FLOAT  Peak-picking sensitivity. Default 1.0.
                   --min-gap FLOAT      Merge onsets closer than this, ms. Default 50.
  silence PATH     --threshold FLOAT    dB above the measured noise floor. Default 6.0.
                   --min-len FLOAT      Ignore silences shorter than this, ms. Default 400.
  fillers PATH     --words STR          Comma-separated filler vocabulary.
                   --min-pause FLOAT    Report pauses at least this long, ms. Default 700.

Example:
  aud detect silence in.wav --threshold 6 --min-len 400
""",
    "cut": """\
cut -- editing stage: remove an explicit list of regions

What it does:
  Appends a cut stage to the plan. At render time the listed regions are
  removed, each boundary is RESOLVED into an actual cut point (see below),
  and each resulting join is crossfaded. The regions come from a regions
  document -- piped in from 'aud detect' on stdin, or read from a file
  with --regions.

When to reach for it:
  You already know what to remove: a detect run, or a list you produced
  some other way.

    aud detect silence in.wav | aud cut | aud render in.wav out.wav

An edit point is NOT the position in the document:
  The regions document says where a boundary IS. Where the blade should
  FALL is a separate decision, and getting it wrong is audible twice: a
  cut through the attack of a word truncates it, and a cut at a non-zero
  sample value clicks. So every position is treated as NOMINAL and
  resolved before anything is removed -- padded, then snapped inside a
  bounded window, then joined.

  --snap picks what the point is moved towards:

    zero_crossing  Nearest zero crossing. THE DEFAULT, and the floor: it
                   costs nothing, moves the point by under a millisecond,
                   and removes the sample discontinuity that clicks.
    silence        The quietest place in the window -- the local minimum
                   of the energy envelope. The blade lands where there is
                   least to damage.
    transient      Just BEFORE the nearest onset, so an attack is never
                   cut through. Always moves the point EARLIER: a sound
                   that has begun and is then chopped reads as a glitch,
                   where removing it whole does not.
    none           Take the position literally. No search, no alignment.
                   For a caller that has already chosen exact samples.

  'silence' and 'transient' finish with a zero-crossing alignment, because
  they answer a different question (WHERE the edit belongs, coarsely) from
  the one zero crossing answers (how it is finally aligned, to a sample).

  A snap is BOUNDED and REFUSABLE. It never moves past the region's other
  boundary, into a neighbouring region, or outside the file. If nothing
  acceptable exists inside --snap-window it KEEPS the original position and
  records that it did, in the render report. It never widens the window and
  never quietly substitutes another rule -- a snap that silently fails is
  worse than one that refuses, because the file still plays and you find
  out after delivery.

Padding only ever shrinks a cut:
  --pad-out keeps programme at the end of the outgoing side, --pad-in at
  the start of the incoming side. Both make the removal SMALLER; neither
  can extend one. That is why they are safe to reach for when a join
  sounds clipped. 'cut' defaults both to 0: the regions are a list you
  measured and mean literally, so widening them unasked would surprise.

Why equal power is the default crossfade shape:
  Two uncorrelated signals sum in POWER, not amplitude. Under a linear
  crossfade both sides sit at gain 0.5 in the middle, so the sum is half
  the power of either -- an audible ~3 dB dip through every join. Equal
  power holds the summed power constant instead. 'linear' is offered
  because the argument inverts for CORRELATED material (a join across a
  sustained tone), where equal power bumps +3 dB and linear is flat.

  A crossfade consumes material on BOTH sides of the join: the two kept
  slices must overlap by its length, and that overlap comes out of what is
  being removed. A crossfade longer than the gap is an ERROR
  (crossfade_exceeds_gap), not something silently clamped.

Why it renders FIRST, ahead of everything else:
  Cutting changes the timeline that every later stage measures. If
  'loudness' targets -14 LUFS across material that 'cut' then removes, the
  number it hit describes a file that no longer exists. Region positions
  are also offsets into the SOURCE timeline, so any stage that re-times
  the programme ('stretch') has to run after the cut, not before it.

  Consequence worth knowing: a plan containing a cut stage is bound to the
  file its regions were measured on. It is not reusable across episodes
  the way a tonal plan is. Use 'strip-silence' for the reusable version.

Parameters:
  --regions PATH          Regions document to cut. Default: read from stdin.
  --pad-out FLOAT         Programme kept at the end of the outgoing side, ms.
                          Default 0.
  --pad-in FLOAT          Programme kept at the start of the incoming side,
                          ms. Default 0.
  --snap MODE             zero_crossing | silence | transient | none.
                          Default zero_crossing.
  --snap-window FLOAT     How far a point may move, ms. Default 20. Must be
                          > 0 and <= 1000, else snap_window_invalid.
                          20 is ample for zero_crossing; 'silence' and
                          'transient' usually want 50-150.
  --fade-out FLOAT        Fade at a kept boundary with NO crossfade partner
                          (programme head/tail, or --crossfade 0), ms.
                          Default 0.
  --fade-in FLOAT         As above, incoming side, ms. Default 0.
  --crossfade FLOAT       Crossfade at each join, ms. Default 10.
  --crossfade-shape SHAPE equal_power | linear. Default equal_power.

What the render report tells you:
  Per edit point: the nominal position, where it resolved to, how far it
  moved, which rule placed it, and whether a requested snap failed. Without
  that, smart placement is unfalsifiable -- a snap that worked and a snap
  that quietly did nothing both produce a file.

Status:
  Not yet built in this release: the flags above parse, and the verb
  returns {"error": {"code": "not_implemented", ...}}. No padding, snap,
  fade or crossfade code exists yet. See contracts/plan.v1.md for the
  stage's parameters and contracts/regions.v1.md for what it consumes.

Worked example -- trim the pauses without chopping the start of a word:
  aud detect silence in.wav \\
    | aud cut --pad-in 80 --pad-out 80 --snap transient --snap-window 60 \\
    | aud render in.wav out.wav                                    # (planned)

  Read it as: find the quiet spans; keep 80 ms of programme either side of
  each one so nothing sounds clipped; then, if an onset lies within 60 ms
  of where the blade would land, move the blade to just before it rather
  than through it; join with the default 10 ms equal-power crossfade.
""",
    "strip-silence": """\
strip-silence -- editing stage: remove or shorten the silences

What it does:
  Appends a strip_silence stage to the plan. Unlike 'cut', it carries NO
  positions: it detects the silences at render time using the parameters
  below, then removes or shortens each one. The boundaries it finds are
  NOMINAL and are resolved into real cut points by exactly the same rules
  'cut' uses -- see 'aud cut --help' for the full explanation of --snap,
  the padding direction, and why equal power is the default shape.

When to reach for it:
  "Trim the silences", "tighten up this recording" -- and especially when
  the same treatment should apply to every episode. Because it stores a
  rule rather than positions, the plan means the same thing on any file:

    aud plan | aud strip-silence --min-len 400 --keep 150 | aud render in.wav out.wav

  That is the difference from 'cut', which stores positions and is
  therefore bound to one file.

The threshold is relative, on purpose:
  --threshold is dB ABOVE the file's MEASURED noise floor, not an absolute
  dBFS level -- see 'aud detect --help' for why a fixed number is wrong
  for every recording but the one it was tuned on.

Why padding is ON by default here and off on 'cut':
  An energy threshold's boundary sits systematically INSIDE the speech --
  the tail of a word drops below the threshold while the word is still
  going, and the next word's attack crosses back over it a few
  milliseconds after it has started. Padding corrects a known bias, so it
  defaults to 80 ms each side. 'cut' is handed a list you measured and
  mean literally, so it defaults to none.

  Padding only ever SHRINKS the removal. If --pad-out plus --pad-in is at
  least as long as a silence, that silence is not removed at all, and the
  render report says which ones and why -- failing the whole render over
  one marginal pause would be worse, and dropping it silently worse still.

Parameters:
  --threshold FLOAT       dB above the measured noise floor. Default 6.0.
  --min-len FLOAT         Leave silences shorter than this alone, ms.
                          Default 400.
  --keep FLOAT            Silence left behind in place of each one, ms.
                          Default 150. 0 removes it entirely.
  --pad-out FLOAT         Programme kept at the end of the outgoing side, ms.
                          Default 80.
  --pad-in FLOAT          Programme kept at the start of the incoming side,
                          ms. Default 80.
  --snap MODE             zero_crossing | silence | transient | none.
                          Default zero_crossing.
  --snap-window FLOAT     How far a point may move, ms. Default 20. Must be
                          > 0 and <= 1000, else snap_window_invalid.
  --fade-out FLOAT        Fade at a kept boundary with NO crossfade partner,
                          ms. Default 0.
  --fade-in FLOAT         As above, incoming side, ms. Default 0.
  --crossfade FLOAT       Crossfade at each join, ms. Default 10.
  --crossfade-shape SHAPE equal_power | linear. Default equal_power.

Status:
  Not yet built in this release: the flags above parse, and the verb
  returns {"error": {"code": "not_implemented", ...}}. No detection,
  padding, snap, fade or crossfade code exists yet. See
  contracts/plan.v1.md for the stage's parameters.

Worked example -- trim the pauses without chopping the start of a word:
  aud plan \\
    | aud strip-silence --min-len 400 --keep 150 --snap transient --snap-window 60 \\
    | aud render in.wav out.wav                                    # (planned)

  Read it as: find pauses of at least 400 ms; leave 150 ms of silence in
  place of each; keep the default 80 ms of programme either side; and if an
  onset lies within 60 ms of where the blade would land, move it to just
  before that onset rather than through it.
""",
    "gate": """\
gate -- repair stage: hard noise gate

What it does:
  Appends a gate stage to the plan on stdin. At render time, whenever the
  signal drops below --threshold it is ducked by --range dB; at or above
  threshold it is untouched. Unlike a static effect this only engages where
  the file is actually quiet.

When to reach for it:
  Room tone between phrases, mic hiss, HVAC rumble, a breath in the wrong
  place -- material that should be well below the programme, not merely
  attenuated a little. For a gentler, proportional treatment (usually the
  better first choice for voice), reach for 'aud expand' instead.

The threshold is relative, on purpose:
  --threshold is dB ABOVE this file's MEASURED noise floor, not an absolute
  dBFS level -- see 'aud detect --help' for why a fixed number is wrong for
  every recording but the one it was tuned on. --threshold-abs is an
  absolute dBFS escape hatch for a caller that already knows one; when
  given, it overrides --threshold entirely.

Why --range, not silence:
  A gate that slams to digital black is more noticeable than the noise it
  removed. --range is a duck (default 20 dB), not a kill; 0 dB disables
  the stage, a very large value approaches a hard mute.

Why --hold matters:
  Material hovering right at the threshold will open and close many times
  a second without it (chatter) -- the single most recognisable way a gate
  sounds broken. --hold keeps the gate held open for that long after it was
  last triggered, before --release is allowed to start closing it.

Multiband, and a sidechain highpass:
  --bands splits into Linkwitz-Riley bands (ascending crossover
  frequencies) and gates each independently, for the same reason
  'aud compress' does: full-band, a signal loud in one band and at the
  noise floor in another either gates the wrong thing or nothing at all.
  --sidechain-hpf (default 80 Hz) highpasses only the LEVEL DETECTOR, not
  the signal itself, so low-frequency rumble cannot hold the gate open for
  content that, once the rumble is filtered out, is not actually there.

Parameters:
  --threshold FLOAT       dB above the measured noise floor. Default 12.0.
  --threshold-abs FLOAT   Absolute dBFS threshold; overrides --threshold
                          when given. Default: unset.
  --range FLOAT           Maximum attenuation below threshold, dB, >= 0.
                          Default 20.0.
  --attack FLOAT          Time to open once triggered, ms, >= 0. Default 2.0.
  --hold FLOAT            Minimum time held open before release may begin,
                          ms, >= 0. Default 50.0.
  --release FLOAT         Time to close once hold expires, ms, >= 0.
                          Default 150.0.
  --lookahead FLOAT       How far ahead the detector looks so an onset
                          survives, ms, >= 0. Default 3.0.
  --sidechain-hpf FLOAT   Highpass corner for the level detector only, Hz.
                          Default 80.0.
  --bands F1,F2,...       Ascending crossover frequencies, Hz. Default: full-band.

Example:
  aud plan | aud gate --threshold 12 --range 20 | aud render in.wav out.wav
""",
    "expand": """\
expand -- repair stage: soft-knee downward expander

What it does:
  Appends an expand stage to the plan on stdin. At render time, material
  below --threshold is attenuated proportionally: output moves --ratio dB
  for every 1 dB the input drops, blended in over --knee dB around the
  threshold -- gentler than 'aud gate''s hard step, and usually the better
  first choice for voice material.

When to reach for it:
  Taming background noise and room tone without a hard on/off character.
  Reach for 'aud gate' instead when a harder, more decisive cut is wanted.

Parameters shared with 'aud gate':
  --threshold, --threshold-abs, --attack, --hold, --release, --lookahead,
  --sidechain-hpf, --bands all mean exactly what they mean there -- see
  'aud gate --help' for the full explanation of the relative threshold,
  why --hold prevents chatter, multiband splitting and the sidechain
  highpass.

Parameters:
  --threshold FLOAT       dB above the measured noise floor. Default 6.0.
  --threshold-abs FLOAT   Absolute dBFS threshold; overrides --threshold
                          when given. Default: unset.
  --ratio FLOAT           Expansion ratio, >= 1.0. 1.0 is no expansion; 2.0
                          means output moves 2 dB per 1 dB input drop below
                          threshold. Default 2.0.
  --knee FLOAT            Soft-knee width around the threshold, dB, >= 0.
                          Default 6.0.
  --attack FLOAT          Time to open once triggered, ms, >= 0. Default 5.0.
  --hold FLOAT            Minimum time held open before release may begin,
                          ms, >= 0. Default 50.0.
  --release FLOAT         Time to move toward the target once hold expires,
                          ms, >= 0. Default 150.0.
  --lookahead FLOAT       How far ahead the detector looks so an onset
                          survives, ms, >= 0. Default 3.0.
  --sidechain-hpf FLOAT   Highpass corner for the level detector only, Hz.
                          Default 80.0.
  --bands F1,F2,...       Ascending crossover frequencies, Hz. Default: full-band.

Example:
  aud plan | aud expand --threshold 6 --ratio 2 | aud render in.wav out.wav
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
  --amount FLOAT   Maximum reduction in dB, >= 0. Default 6.
  --freq FLOAT     Center frequency of the sibilant band in Hz. Default 6500.

Example:
  aud plan | aud deess --amount 8 --freq 7000 | aud render in.wav out.wav
""",
    "dereverb": """\
dereverb -- repair stage: reduce room ambience

What it does:
  Appends a de-reverb stage to the plan on stdin. At render time this
  estimates the reverberant tail of a recording made in an untreated room
  and reduces it -- a moderate-improvement tool for a moderately live
  room, not a heavy-reverb remover. Larger --amount values trade more
  suppression for more risk of audible artifacts and damage to sustained,
  non-reverberant material (see dsp/dereverb.py's module docstring).

When to reach for it:
  "The room sounds boxy", recordings made in a live or reflective space
  that need to sound drier and closer.

Parameters:
  --amount FLOAT   Maximum reduction applied to the estimated reverberant
                   component, in dB, >= 0. Default 6.

Example:
  aud plan | aud dereverb --amount 6 | aud render in.wav out.wav
""",
    "eq": """\
eq -- tone stage: parametric equalization

What it does:
  Appends a parametric EQ stage to the plan on stdin: an optional
  high-pass corner, an optional low-pass corner, any number of peaking
  bands (repeat --peak for each one), and any number of shelving bands
  (repeat --shelf for each one).

When to reach for it:
  "Tighten the low end" (--hpf), "roll off the hiss" (--lpf), correcting a
  specific frequency problem with a peaking band, or a broad tonal tilt
  with a shelf (--shelf).

Parameters:
  --hpf FLOAT              High-pass corner frequency in Hz. Must be > 0,
                           and below --lpf if both are given.
  --lpf FLOAT              Low-pass corner frequency in Hz. Must be > 0.
  --peak FREQ,GAIN,Q       A peaking band as freq_hz,gain_db,q. Q must be
                           > 0. Repeatable.
  --shelf TYPE,FREQ,GAIN,Q A shelving band as type,freq_hz,gain_db,q.
                           TYPE is 'low' or 'high'; Q must be > 0.
                           Repeatable.

Example:
  aud plan | aud eq --hpf 40 --peak 3200,-2.5,1.4 --shelf low,80,3.0,0.7 | aud render in.wav out.wav
""",
    "eq-match": """\
eq-match -- tone stage: match a reference's tonal balance

What it does:
  Appends an EQ-match stage to the plan on stdin: a spectral curve is
  diffed against this file's own measured spectrum at render time, clamped
  per band, and applied as a linear-phase FIR. The curve comes from either
  a JSON file previously written by 'aud curve extract' (--curve), or a
  reference file measured right now, at plan-build time (--reference) --
  either way the plan stores plain numbers, never a file path, so it
  replays identically with no access to the reference file later.

When to reach for it:
  "Make this podcast match last week's episode", matching a new take to an
  established reference recording.

Parameters:
  --curve PATH       Path to a curve JSON file produced by 'aud curve
                     extract'. Exactly one of --curve/--reference required.
  --reference PATH   A reference audio file, measured now. Exactly one of
                     --curve/--reference required.
  --strength FLOAT   How much of the match to apply, 0.0 (none, input
                     unchanged) to 1.0 (full). Default 1.0.
  --max-gain-db FLOAT  Clamp on the correction in either direction, per
                     band, in dB. Default 12.0.

Example:
  aud curve extract reference.wav ref_curve.json
  aud plan | aud eq-match --curve ref_curve.json --strength 0.8 | aud render in.wav out.wav
  aud plan | aud eq-match --reference reference.wav --strength 0.8 | aud render in.wav out.wav
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
  --ratio FLOAT       Compression ratio per band, >= 1.0 (1.0 is no
                      compression in that band). Default 2.5.

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
  --drive FLOAT   Saturation drive, >= 0. Higher is more distortion; 0
                  disables the shaping. Default 1.0.
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
  --amount FLOAT     Wet amount, 0.0-1.0. Default 0.15.
  --decay FLOAT      Decay time in seconds, > 0. Default 1.2.
  --predelay FLOAT   Delay before the reverb tail begins, ms, >= 0. Default 0.

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
  --target FLOAT   Target integrated loudness in LUFS, must be < 0. Default -14.0.

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

  Advanced (plan-document only, not exposed as CLI flags -- see
  contracts/plan.v1.md#limit): lookahead_ms (default 5.0), release_ms
  (default 50.0), oversample (one of 1/2/4/8, default 4). A hand-written
  plan may set these directly.

Example:
  aud plan | aud loudness --target -14 | aud limit --ceiling -1.0 | aud render in.wav out.wav
""",
    "render": """\
render -- apply a whole plan to a file in one pass

What it does:
  Reads the plan on stdin, reorders its stages into canonical order
  (editing -> repair -> tone -> dynamics -> character -> loudness -> limit)
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
  Named, pre-built mastering chains for common destinations (podcast
  hosting, music streaming, broadcast, voiceover) that would otherwise be
  built by hand with plan/eq/compress/deess/saturate/loudness/limit. Every
  preset is built through those same stage builders, so it is validated
  exactly like a hand-built chain -- there is no separate, preset-only
  parameter path.

    preset --list       List the available preset names, each with a
                        one-line description naming the loudness
                        convention it targets.
    preset show NAME    Print the named preset's chain, as a plan
                        document, raw on stdout -- unwrapped, like every
                        stage verb, so it pipes straight into 'render'.

When to reach for it:
  "Master this for podcast hosting" or "get this ready for streaming"
  without hand-assembling the chain and picking numbers yourself.

Parameters:
  --list          List the available preset names and descriptions.
  show NAME       Print the named preset's chain. NAME is one of the
                  names 'preset --list' reports (e.g. 'podcast',
                  'music-streaming', 'broadcast', 'voiceover').

Example:
  aud preset --list
  aud preset show podcast | aud render in.wav out.wav
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
  Runs the same measurement 'aud analyze' would, then asks a model to
  choose a mastering chain -- which stages, what parameters, and why --
  grounded in those numbers. The model never sees or touches the audio
  itself, only the measurement report; its proposed chain is validated
  through the exact same stage builders every other verb uses ('aud eq',
  'aud compress', ...) before it becomes the plan you get back. Prints the
  plan itself on stdout, unwrapped, so it pipes straight into 'aud render'
  -- the per-stage reasoning goes to stderr, since stdout has to stay a
  clean plan document for the pipe to work.

When to reach for it:
  "What should I do to this file" without wanting to build the chain by
  hand, or as a starting point to edit before rendering.

Requires:
  One of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY
  or AZURE_OPENAI_API_KEY. Without one, this refuses with
  {"error": {"code": "provider_credential_missing", ...}} naming the
  accepted variables -- see docs/CONFIGURATION.md. Every deterministic verb
  keeps working regardless.

What the model may choose from:
  Exactly eight stages -- deess, dereverb, eq, compress, saturate, reverb,
  loudness, limit -- the ones a measurement report alone can justify.
  Editing (cut/strip-silence) needs detected regions, not measurements, so
  it is out of scope for advise; run 'aud detect' and build those by hand.

Parameters:
  path (positional)   Path to the audio file to measure and advise on.
  --target FLOAT      Target integrated loudness in LUFS, told to the
                       model. Default -14.0.
  --reference PATH    Optional reference file. Its measurements (never its
                       audio) are given to the model too, to inform tonal
                       choices such as EQ peaks.
  --model NAME        Override the model name for whichever provider is
                       configured. Default: a per-provider built-in,
                       overridable by the AUD_MODEL environment variable
                       too.

Example:
  aud advise in.wav --target -14 | aud render in.wav out.wav
""",
    "master": """\
master -- choose the chain, apply it, and verify the result

What it does:
  The one-shot model-backed path: runs 'advise' internally, renders the
  chain it proposes in a single pass, then verifies the render against the
  loudness target and true-peak ceiling it was asked for. Prints one JSON
  {"result": ...} document containing the chosen plan, the per-stage reason
  the model gave, the render report, and the verify block.

When to reach for it:
  You know only what the file should end up like, not how to get there --
  "master this to -14 LUFS" with no chain to hand-build.

Requires:
  One of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY
  or AZURE_OPENAI_API_KEY. Without one, this refuses with
  {"error": {"code": "provider_credential_missing", ...}} before touching
  out_path -- see docs/CONFIGURATION.md.

Parameters:
  in_path (positional)   Source audio file.
  out_path (positional)  Destination audio file to write. Untouched if
                         --dry-run is given, or if advise refuses.
  --target FLOAT         Target integrated loudness in LUFS. Default -14.0.
  --ceiling FLOAT        Target true-peak ceiling in dBTP. Default -1.0.
  --reference PATH       Optional reference file; measured, not rendered
                         with -- informs the model's tonal choices.
  --model NAME           Override the model name. See 'aud advise --help'.
  --dry-run              Choose and print the plan; render nothing.

Example:
  aud master in.wav out.wav --target -14
  aud master in.wav out.wav --dry-run
""",
}
