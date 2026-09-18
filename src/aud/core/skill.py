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
    ("plan", "deterministic", "Start an empty mastering plan, or load one from a file."),
    ("deess", "deterministic", "Repair stage: tame harsh sibilance."),
    ("dereverb", "deterministic", "Repair stage: reduce room ambience."),
    ("eq", "deterministic", "Tone stage: parametric EQ -- high-pass, low-pass, peaking bands."),
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
    ("advise", "model-backed", "Read the measurements and say what the chain should be, and why."),
    ("master", "model-backed", "Choose the chain, apply it, and verify the result."),
)


def render_skill() -> str:
    """The full `aud --help` text, wrapped in a <skill_content> envelope."""
    width = max(len(verb) for verb, _, _ in CAPABILITIES)
    verb_lines = [f"  {verb.ljust(width)}  [{kind:<13}]  {desc}" for verb, kind, desc in CAPABILITIES]
    lines = [
        "aud -- master and clean up a finished audio file",
        "",
        "Measures what is actually there (loudness, true peak, spectral balance, crest",
        "factor, sibilance, ambience), then runs it through a mastering chain: de-ess,",
        "de-verb, parametric EQ, EQ-match against a reference, multiband compression,",
        "saturation, controlled reverb, loudness targeting and true-peak limiting.",
        "",
        "Read this first: chain the verbs, do not orchestrate them. Every stage verb",
        "reads a mastering plan on stdin, appends one stage, and writes the plan back",
        "out -- nothing touches a sample until 'render':",
        "",
        "  aud plan \\",
        "    | aud deess --amount 6 \\",
        "    | aud eq --hpf 40 --peak 3200,-2.5,1.4 \\",
        "    | aud compress --bands 120,900,5500 --ratio 2.5 \\",
        "    | aud loudness --target -14 \\",
        "    | aud limit --ceiling -1.0 \\",
        "    | aud render in.wav out.wav",
        "",
        "'render' applies stages in canonical mastering order regardless of the order",
        "you appended them: repair -> tone -> dynamics -> character -> loudness -> limit.",
        "",
        "Verbs marked [deterministic] run with no AI provider and no credentials, spend",
        "nothing, and are safe to call freely. Verbs marked [model-backed] need one of",
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY or",
        "AZURE_OPENAI_API_KEY.",
        "",
        "Verbs:",
        *verb_lines,
        "",
        'One JSON document on stdout: {"result": ...} on success, or',
        '{"error": {"code", "message", "remedy"}} with a non-zero exit on failure.',
        "A plan is the exception -- 'plan' and every stage verb print the plan document",
        "itself, unwrapped, so verbs pipe into each other.",
        "",
        "Run 'aud <verb> --help' for the full documentation of any one verb.",
    ]
    body = "\n".join(lines)
    return f'<skill_content name="aud">\n{body}\n</skill_content>'
