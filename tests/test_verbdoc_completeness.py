"""Structural completeness guard for `aud.verbdoc.VERB_DOCS`.

Every entry must carry, beyond its narrative "What it does"/"When to reach
for it" prose: a Kind section (deterministic vs model-backed), a Result
section (what a caller gets back on success) and a Failures section (the
non-zero-exit conditions this verb can actually produce). This is the
regression guard for the spec-adherence finding that some entries had none
of the three, and that 'cut'/'strip-silence' falsely claimed to be
"Not yet built in this release" when both are fully implemented -- see
AGENTS.md and src/aud/dsp/engine.py's module docstring.
"""

from __future__ import annotations

from aud.verbdoc import VERB_DOCS

_MODEL_BACKED_VERBS = {"advise", "master"}


def test_every_verb_has_a_kind_result_and_failures_section() -> None:
    missing: dict[str, list[str]] = {}
    for verb, doc in VERB_DOCS.items():
        gaps = [section for section in ("Kind:", "Result:", "Failures:") if section not in doc]
        if gaps:
            missing[verb] = gaps
    assert not missing, f"verb docs missing required sections: {missing}"


def test_model_backed_verbs_are_labelled_model_backed() -> None:
    for verb in _MODEL_BACKED_VERBS:
        doc = VERB_DOCS[verb]
        kind_section = doc.split("Kind:", 1)[1].split("\n\n", 1)[0]
        assert "model-backed" in kind_section.lower(), f"{verb} should be labelled model-backed"


def test_deterministic_verbs_are_not_labelled_model_backed() -> None:
    for verb, doc in VERB_DOCS.items():
        if verb in _MODEL_BACKED_VERBS:
            continue
        kind_section = doc.split("Kind:", 1)[1].split("\n\n", 1)[0]
        assert "model-backed" not in kind_section.lower(), f"{verb} should not be labelled model-backed"


def test_cut_and_strip_silence_no_longer_claim_to_be_unimplemented() -> None:
    """Regression guard: both stages are fully implemented (see
    src/aud/dsp/engine.py's `_apply_cut` / `_apply_strip_silence` and their
    registration in the stage registry) -- the docs must say so, not claim
    'not_implemented'/'Not yet built' for either.
    """
    for verb in ("cut", "strip-silence"):
        doc = VERB_DOCS[verb]
        assert "not yet built" not in doc.lower(), f"{verb} still claims to be unimplemented"
        assert '"code": "not_implemented"' not in doc, f"{verb} still documents a not_implemented status"


def test_cut_and_strip_silence_document_the_real_render_report_fields() -> None:
    for verb, expected_field in (("cut", "edit_points"), ("strip-silence", "edit_points")):
        doc = VERB_DOCS[verb]
        result_section = doc.split("Result:", 1)[1].split("\n\n", 1)[0]
        assert expected_field in result_section, f"{verb}'s Result section should name {expected_field!r}"


# --- Regression guard for capability-skills-complete: every documented
# optional argument states its default -------------------------------------
#
# 'detect fillers --words' documented its purpose but omitted its default
# (the reviewer's finding). The audit that fix required found the same gap
# on 15 more optional arguments across 5 more verbs -- not a single miss,
# a pattern. Each (verb, flag) pair below is one that was previously silent
# on its default; this pins it so it cannot regress silently.
_PREVIOUSLY_UNDOCUMENTED_DEFAULTS: tuple[tuple[str, str], ...] = (
    # detect's "fillers" sub-row packs flag+type on one line ("--words STR"),
    # unlike every other verb's two-space-indented row start -- the type
    # token makes the marker unique without needing the generic row logic.
    ("detect", "--words STR"),
    ("eq", "--hpf"),
    ("eq", "--lpf"),
    ("eq", "--peak"),
    ("eq", "--shelf"),
    ("verify", "--target"),
    ("verify", "--ceiling"),
    ("advise", "--reference"),
    ("master", "--reference"),
    ("master", "--model"),
    ("master", "--dry-run"),
    ("config", "--sample-rate-policy"),
    ("config", "--default-ceiling-dbtp"),
    ("config", "--default-target-lufs"),
    ("config", "--oversample"),
    ("config", "--output-subtype"),
)


def test_every_previously_undocumented_default_now_states_one() -> None:
    for verb, marker in _PREVIOUSLY_UNDOCUMENTED_DEFAULTS:
        doc = VERB_DOCS[verb]
        params_section = doc.split("Parameters:", 1)[1].split("\n\n", 1)[0]
        # A bare "--flag" marker must start its own row (preceded by a
        # newline and leading whitespace only), not an incidental mention
        # inside another parameter's prose (e.g. master's out_path entry
        # mentions "--dry-run" before --dry-run's own row is documented).
        # A marker that already includes its type token (detect's packed
        # "--words STR" row) is unique enough to search for directly.
        row_marker = marker if " " in marker else "\n  " + marker
        assert row_marker in params_section, f"{verb}'s Parameters section no longer documents {marker} as its own row"
        after_flag = params_section.split(row_marker, 1)[1]
        next_flag_pos = after_flag.find("\n  --", 1)
        flag_text = after_flag if next_flag_pos == -1 else after_flag[:next_flag_pos]
        assert "default" in flag_text.lower(), f"{verb}'s {marker} still does not state a default"
