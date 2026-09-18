# Vision

Context and intent. How it is built is [ARCHITECTURE.md](ARCHITECTURE.md); this document is
about what problem `aud` solves, for whom, and what it refuses to be.

## The problem

Someone finishes a recording and it is nearly right. It is six decibels quieter than everything
else in the feed. The S sounds are sharp enough to hurt. The room it was recorded in is audible
and boxy. Episode 12 does not sound like episode 11. None of this is a creative problem — it is
a measurement-and-correction problem, and it is the last mile between a file that exists and a
file that can be published.

Two kinds of tool already solve it, and neither fits the caller we care about:

- **A DAW or a mastering plug-in chain.** Excellent, and built for a human with ears, a mouse
  and an afternoon. It cannot be invoked, cannot be scripted over a hundred files, and cannot
  explain itself to a program.
- **The mature Python libraries.** They exist and they are good. They are also GPL-family,
  which decides where they can be used before any technical question is asked.

So there is a gap: a scriptable, inspectable, permissively licensed mastering tool whose output
is a document rather than an opinion.

## Finishing a spoken-word recording includes cleaning it up

The same person with the same nearly-right file has a second problem, and it is not a different
problem. The interview has four seconds of dead air before every answer. There are nine "umm"s
in the first minute. The recording is twenty-eight minutes long and eight of those are pauses.
Nobody describes that as a separate job from "make this publishable" — they say "tighten this
up", and they mean both.

So cutting belongs in the same chain as mastering, not in a tool beside it. Splitting them would
mean two passes over the audio, two renders, two quantisations — and, worse, a loudness number
measured against material that the other tool then removes. Loudness is an average over
duration; a cut invalidates it. The only way that stays correct is if the cut and the loudness
target are stages of one ordered chain, with the cut first. That is not a convenience of
combining them; it is the reason they cannot sensibly be apart.

It does mean `aud` reaches for speech recognition to find filler words, which is the one place
it touches language. It stays a means and never becomes an output: `aud` will tell you there is
an "umm" at 4:12.380 and remove it. It will not hand you a transcript. Transcription is a
different product with different quality bars, and claiming it here would be claiming something
this tool is not built to be good at.

## Who it is for

- **Someone with a finished file.** Podcast episode, voice-over, a mix that came back from
  somewhere else, an audiobook chapter, a live recording. They want it to meet a target and to
  stop sounding like the room.
- **An agent working on that person's behalf.** This is the caller the design actually
  optimises for. An agent needs numbers it can read, a plan it can inspect before anything is
  written, and failures that name a remedy rather than a stack trace.
- **A product that has to ship the result.** Permissively licensed, no daemon, no GUI, no
  network call unless a model was explicitly asked for.

## The decision that shapes everything: licence over convenience

The obvious way to build this is to depend on what already works. We did not, and it is worth
being plain about the cost.

| The convenient dependency | Licence | Why it is excluded |
|---|---|---|
| `pedalboard` | GPL-3.0 | A ready-made, fast, well-tested effects chain |
| `matchering` | GPL-3.0 | Reference-matching mastering, essentially solved |
| Rubber Band | GPL-2.0-or-later / commercial dual | The standard high-quality time-stretch |

Each would have removed weeks of work. Taking any of them makes `aud` GPL-family too, and that
propagates to everything that embeds it. The whole point is a tool a permissively licensed
product can take without a licence conversation, so the dependency list is instead:

`numpy` (BSD) · `scipy` (BSD) · `soundfile` (BSD-3) · `pyloudnorm` (MIT)

and **the DSP is written here.** Time-stretch, when it is wanted, is an optional extra backed by
`python-stretch` (Signalsmith Stretch, MIT) — never Rubber Band.

**What that trade costs:** our crossovers, gain computers, limiter and EQ-match are younger and
less proven than the alternatives. There is no decade of field use behind them. Every one of
those stages had to be understood well enough to write, which is slower than calling one.

**What it buys:** anyone can vendor this. And the thing we were forced to do — understand and
implement each stage — is the same thing that lets the tool report what it did in terms of the
actual processing rather than "a plug-in ran". Measurement and explanation stopped being a
feature bolted on afterwards and became a property of having written the chain.

This is a settled decision, not an open question. A proposal to add a GPL-family dependency is
a proposal to relicense the tool, and should be made in those words.

## What it believes

- **Measure first, and say the numbers.** Every stage decision is traceable to something that
  was measured. "It sounds better" is not an output this tool produces.
- **The plan is the artefact.** A chain is a document you can read, save, diff, review and
  re-run across a hundred files. What happened to the audio is not locked inside a session file.
- **One render.** The whole chain is applied in a single pass because every extra render is
  another quantisation and another chance to clip.
- **Intelligence chooses, it does not touch.** A model reads measurements and proposes a plan.
  The same deterministic engine then renders that plan. A model failure can produce a bad plan —
  which you can see before rendering — but it cannot produce a corrupted file.
- **Refuse rather than guess.** No provider configured means `advise` declines and explains.
  It does not invent a chain.

## What it will deliberately not do

- **Mixing.** It takes a finished stereo or mono programme. No stems, no multitrack, no panning,
  no bus routing, no "turn the guitar down". That is a different tool and a different input.
- **Restore material that has been destroyed.** Hard-clipped peaks, codec-mangled high end and
  material recorded into the noise floor are missing information, not hidden information. `aud`
  reports what it finds and declines to pretend; it is not a resynthesis or generative tool.
- **"Make it sound good" with no measurement behind it.** There is no magic button. Even
  `master --auto` runs on measurements, produces a plan you can read, and verifies the result
  against the targets it was given.
- **Video, transcription as an output, or generating audio.** Wrong tool in all three cases; the
  manifest says so, so a dispatcher does not have to guess. `detect fillers` runs a recogniser
  to find out *where* the filler words are — the transcript is a means and is not returned. "What
  does this say" is a question for a transcription tool.
- **Hold credentials.** It reads the provider variable you already have. There is no login, no
  keychain entry, and nothing of yours persisted anywhere by this tool.
