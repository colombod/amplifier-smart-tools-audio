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
