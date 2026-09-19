"""Prompt construction for the advisor.

The model sees exactly two things: a fixed system prompt naming the eight
stages it may choose from and their JSON parameter shapes, and a user prompt
carrying `aud analyze`'s measurement report (and, optionally, a reference
file's own measurement report). Never raw audio, never a sample.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["system_prompt", "user_prompt"]

# Field names here match aud.lib's builder kwargs exactly (deess, dereverb,
# eq, compress, saturate, reverb, loudness, limit all take these), so a
# validated proposal applies with a plain **params call -- see
# aud/intelligence/advisor.py.
_STAGE_REFERENCE = """\
Available stages and their exact JSON parameter shapes -- only these eight;
no others exist for this decision:

  gate      {"threshold_above_floor_db": float >= 0, "range_db": float >= 0,
             "hold_ms": float >= 0}
  expand    {"threshold_above_floor_db": float >= 0, "ratio": float >= 1.0,
             "knee_db": float >= 0}
  deess     {"amount_db": float >= 0, "freq_hz": float > 0}
  dereverb  {"amount_db": float >= 0}
  eq        {"hpf": float > 0 or null, "lpf": float > 0 or null,
             "peaks": [[freq_hz, gain_db, q], ...] or null}
  compress  {"bands": [ascending crossover Hz, at least one],
             "ratio": float >= 1.0}
  saturate  {"drive": float >= 0, "mix": float in [0, 1]}
  reverb    {"amount": float in [0, 1], "decay": float > 0,
             "predelay_ms": float >= 0}
  loudness  {"target_lufs": float < 0}
  limit     {"ceiling_dbtp": float <= 0.0}

`gate` and `expand` work on the QUIET end and are the right answer to an
audible noise floor between phrases. `expand` is gentler and usually correct
for a voice; `gate` is for harder cases. Both take their threshold RELATIVE
to the measured noise floor, so read `noise_floor_dbfs` from the report --
a larger `threshold_above_floor_db` gates more aggressively. Neither belongs
in a chain whose noise floor is already low; say so rather than adding one.
"""


def system_prompt() -> str:
    """The fixed instructions: role, vocabulary, output shape, and rules."""
    return (
        "You are the mastering advisor inside `aud`, an audio mastering tool. "
        "You are given honest numeric measurements of a finished recording -- "
        "never the audio itself -- and you choose a mastering chain: which "
        "stages to apply, with what parameters, and why.\n\n"
        f"{_STAGE_REFERENCE}\n"
        "Respond with EXACTLY one JSON object and nothing else -- no markdown "
        "fences, no prose before or after it. The shape is:\n\n"
        '{"stages": [{"stage": "<name>", "params": {...}, '
        '"reason": "<short, grounded reason>"}, ...]}\n\n'
        "Rules:\n"
        "- Only use stage names from the list above. Do not invent one, and do "
        "not propose editing (cutting/trimming) -- this decision is measurement-"
        "based only, with no information about where anything is in time.\n"
        "- Include each stage at most once.\n"
        "- 'reason' must cite a specific number from the measurements you were "
        "given (e.g. 'crest factor is 18.2 dB, so...'), not a generic remark. "
        "A stage with no stated, measurement-grounded reason is not useful.\n"
        "- Order the stages however you like; aud reorders them into its own "
        "canonical mastering order before rendering, regardless of the order "
        "you list them in.\n"
        "- Usually include 'loudness' and 'limit' so the render actually "
        "reaches the requested target and ceiling, unless the measurements "
        "say it is already there.\n"
        "- If a stage would do nothing useful for this file, leave it out "
        "entirely rather than including it with a null-ish/no-op parameter.\n"
        "- CHECK THE NOISE FLOOR BEFORE ANYTHING ELSE, because every later "
        "stage makes it worse: compression and loudness both LIFT a noise "
        "floor, so hiss or room tone that was tolerable in the source is "
        "audible in the master. Compare 'noise_floor_dbfs' against "
        "'integrated_lufs'. Roughly 40 dB or more of separation is a clean "
        "recording -- do not gate it. Under about 25 dB is an audible floor "
        "that the chain is about to amplify: reach for 'expand' (gentler, "
        "usually right for a voice) or 'gate' (harder cases), and cite the "
        "two numbers and their difference in your reason."
    )


def user_prompt(
    measurements: dict[str, Any],
    *,
    target_lufs: float,
    ceiling_dbtp: float,
    reference_measurements: dict[str, Any] | None,
) -> str:
    """The per-call context: measurements, targets, and an optional reference."""
    parts = [
        "Measurements for the file to master:",
        json.dumps(measurements, indent=2, sort_keys=True),
        "",
        f"Target integrated loudness: {target_lufs} LUFS.",
        f"Target true-peak ceiling: {ceiling_dbtp} dBTP.",
    ]
    if reference_measurements is not None:
        parts += [
            "",
            "Measurements for a reference file to tonally resemble:",
            json.dumps(reference_measurements, indent=2, sort_keys=True),
            "",
            "Use the reference's octave-band balance and loudness to inform "
            "your 'eq' peaks and targets; do not propose a stage outside the "
            "allowed list to match it.",
        ]
    parts += [
        "",
        "Propose the mastering chain now, as one JSON object.",
    ]
    return "\n".join(parts)
