"""Composes the `aud --help` / skill text from a single source of truth.

CAPABILITIES is the only place the verb surface is enumerated for help text.
cli.py's argument parser is built independently, but every verb registered
there must appear here too -- see tests/test_cli_envelope.py and the "no
drift" discipline this module exists to enforce.
"""

from __future__ import annotations

# (verb, kind, one-line description). kind is "deterministic" or "model-backed".
CAPABILITIES: tuple[tuple[str, str, str], ...] = (
    (
        "analyze",
        "deterministic",
        "Measure loudness, true peak, spectral balance, crest factor, sibilance and ambience.",
    ),
    (
        "detect",
        "deterministic",
        "Find things in the audio -- onsets, silences, filler words. Emits a regions document, not a plan.",
    ),
    ("plan", "deterministic", "Start an empty mastering plan, or load one from a file."),
    (
        "cut",
        "deterministic",
        "Editing stage: remove a listed set of regions -- padded, snapped to a safe cut point, crossfaded.",
    ),
    (
        "strip-silence",
        "deterministic",
        "Editing stage: remove or shorten the silences, with the same padding, snapping and crossfading.",
    ),
    ("gate", "deterministic", "Repair stage: hard noise gate -- below threshold, duck by a fixed amount."),
    (
        "expand",
        "deterministic",
        "Repair stage: soft-knee downward expander -- the gentle counterpart to gate.",
    ),
    ("deess", "deterministic", "Repair stage: tame harsh sibilance."),
    ("dereverb", "deterministic", "Repair stage: reduce room ambience."),
    ("eq", "deterministic", "Tone stage: parametric EQ -- high-pass, low-pass, peaking bands, shelving bands."),
    ("eq-match", "deterministic", "Tone stage: match this file's tonal balance to a reference curve."),
    ("curve", "deterministic", "Extract a spectral curve from a file, or apply a saved curve to another file."),
    ("compress", "deterministic", "Dynamics stage: multiband compression across Linkwitz-Riley crossovers."),
    ("saturate", "deterministic", "Character stage: harmonic saturation."),
    ("reverb", "deterministic", "Character stage: controlled ambience."),
    ("stretch", "deterministic", "Retime a programme without changing its pitch."),
    ("pitch", "deterministic", "Re-pitch a programme without changing its timing."),
    ("loudness", "deterministic", "Set the loudness stage's target, in LUFS."),
    ("limit", "deterministic", "Set the true-peak brickwall ceiling, in dBTP."),
    ("render", "deterministic", "Apply a whole plan to a file in a single decode/filter/encode pass."),
    ("verify", "deterministic", "Measure a render against the loudness and ceiling it was asked for."),
    ("preset", "deterministic", "Named chains for common destinations."),
    ("check", "deterministic", "Report what this host has installed and can reach."),
    ("config", "deterministic", "Report effective settings and which tier each came from."),
    ("manifest", "deterministic", "Print this tool's validated SMART_TOOL.md manifest."),
    (
        "advise",
        "model-backed",
        "Read the measurements and say what the chain should be, and why -- needs a provider credential.",
    ),
    (
        "master",
        "model-backed",
        "Choose the chain, apply it, and verify the result -- needs a provider credential.",
    ),
)


def render_skill() -> str:
    """The full `aud --help` text, wrapped in a <skill_content> envelope."""
    width = max(len(verb) for verb, _, _ in CAPABILITIES)
    verb_lines = [f"  {verb.ljust(width)}  [{kind:<13}]  {desc}" for verb, kind, desc in CAPABILITIES]
    lines = [
        "aud -- master and clean up a finished audio file",
        "",
        "Measures what is actually there (loudness, true peak, spectral balance, crest",
        "factor, sibilance, ambience), finds what is in it (onsets, silences, filler",
        "words), then runs it through an editing and mastering chain: cut, strip",
        "silence, de-ess, de-verb, parametric EQ, EQ-match against a reference,",
        "multiband compression, saturation, controlled reverb, loudness targeting and",
        "true-peak limiting.",
        "",
        "Read this first: one command, not a conversation. Chain the verbs in a single",
        "shell command rather than running one, reading the result, and deciding the",
        "next -- every round trip back through a caller is latency, tokens, and another",
        "chance to lose the thread. Every stage verb reads a mastering plan on stdin,",
        "appends one stage, and writes the plan back out; nothing touches a sample",
        "until 'render':",
        "",
        "  aud plan \\",
        "    | aud deess --amount 6 \\",
        "    | aud eq --hpf 40 --peak 3200,-2.5,1.4 \\",
        "    | aud compress --bands 120,900,5500 --ratio 2.5 \\",
        "    | aud loudness --target -14 \\",
        "    | aud limit --ceiling -1.0 \\",
        "    | aud render in.wav out.wav",
        "",
        "Detection composes into the same one command. 'detect' emits a regions",
        "document and 'cut' consumes one, so finding the silences and removing them is",
        "still a single invocation:",
        "",
        "  aud detect silence in.wav | aud cut | aud render in.wav out.wav",
        "",
        "'render' applies stages in canonical order regardless of the order you appended",
        "them: editing -> repair -> tone -> dynamics -> character -> loudness -> limit.",
        "Editing is first because cutting changes the timeline everything downstream",
        "measures -- loudness targeted over material you later remove describes a file",
        "that no longer exists.",
        "",
        "Verbs marked [deterministic] run with no AI provider and no credentials, spend",
        "nothing, and are safe to call freely. Verbs marked [model-backed] need one of",
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY or",
        "AZURE_OPENAI_API_KEY.",
        "",
        "Install:",
        "  uv tool install git+https://github.com/colombod/amplifier-smart-tools-audio",
        "  uv tool install 'aud[speech,stretch] @ git+https://github.com/colombod/amplifier-smart-tools-audio'",
        "    # optional extras: faster-whisper for 'detect fillers', python-stretch for",
        "    # the higher-quality stretch/pitch engine",
        '  uv add "aud @ git+https://github.com/colombod/amplifier-smart-tools-audio"   # as a library',
        "  uvx --from git+https://github.com/colombod/amplifier-smart-tools-audio aud --help   # try once, no install",
        "  npx skills add colombod/amplifier-smart-tools-audio   # the agent skill",
        "",
        "Prerequisites: nothing at all for the deterministic verbs above. ffmpeg only for",
        "mp3/m4a/ogg -- wav/flac/aiff need nothing. 'aud[speech]' (a LOCAL model, no",
        "credential, no network call once cached) only for 'detect fillers'. A provider",
        "credential only for 'advise' and 'master'. Run 'aud check' first: it reports what",
        "this host actually has and names the exact gap, if any -- never assume a",
        "capability is missing without running it.",
        "",
        "Not for: mixing (no multitrack stems, no bus routing, no panning), using",
        "transcription as the output (detect fillers locates filler words, it does not",
        "return a transcript), video files, or generating music. This works on a",
        "finished stereo or mono programme.",
        "",
        "Verbs:",
        *verb_lines,
        "",
        'One JSON document on stdout: {"result": ...} on success, or',
        '{"error": {"code", "message", "remedy"}} with a non-zero exit on failure.',
        "A plan is the exception -- 'plan' and every stage verb print the plan document",
        "itself, unwrapped, so verbs pipe into each other. So is a regions document:",
        "'detect' prints one raw, which is why it pipes straight into 'cut'.",
        "",
        "'detect fillers' additionally needs the optional speech extra (faster-whisper,",
        "a LOCAL model -- no provider, no credential). Without it that one verb refuses",
        "and names the extra; nothing else is affected.",
        "",
        "Run 'aud <verb> --help' for the full documentation of any one verb.",
    ]
    body = "\n".join(lines)
    return f'<skill_content name="aud">\n{body}\n</skill_content>'
