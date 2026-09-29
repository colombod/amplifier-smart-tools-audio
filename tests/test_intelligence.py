"""Unit tests for aud.intelligence: the provider-agnostic seam, provider
selection, validation of untrusted model output, and the real
`AnthropicBackend` response-parsing boundary.

No hand-written fake provider anywhere in this file. Two different
techniques replace the previous `FakeBackend`:

- Tests of `advisor._parse_model_json`/`_validate_and_build_plan` call
  those pure functions directly with literal strings -- this is testing
  aud's OWN parsing/validation logic with plain Python values, not
  simulating any third party's behaviour, so no replay or fake is needed.
- Tests of `advise()`'s wiring (does it forward model/max_tokens, does the
  user prompt carry reference measurements) and of the real
  `AnthropicBackend.complete()` use `tests/replay.py`'s recorded-response
  replays -- see RECORDING.md. The `content: ["thinking", "text"]` shape
  that once broke this exact parser is replayed from a REAL captured
  response, not reconstructed by hand.

Honest exception, named rather than hidden: `OpenAIBackend`, `GoogleBackend`
and `AzureOpenAIBackend` have never been called live (only Anthropic has --
see RECORDING.md's "anthropic/... was not produced by this run" and the
shared findings this project keeps: "Only Anthropic of four provider
backends has been called live"). There is no recording to replay for the
other three. The tests below construct request/response bodies from each
provider's PUBLICLY DOCUMENTED API shape (the URL already cited in each
backend class's own docstring) rather than a real captured call -- this is
a deliberately weaker guarantee than a replay, and is not the same thing
AGENTS.md SS3b's "never hand-write a mock" rule forbids: that rule is about
guessing a third-party LIBRARY's internal object shape (attributes, method
vs. property) without ever having run it, which a docs-shaped HTTP JSON
body is not. It IS still an assumption that could be wrong if the real API
drifts from its own docs, and that is named here rather than left silent.
What these tests DO prove: aud's own parsing code, exercised against a
well-shaped and a malformed body of the documented shape, raises a named
`AudError` for the malformed one rather than an opaque KeyError -- the
exact class of defect the Anthropic `content[0]["text"]` bug was. What they
do NOT prove: that the real provider actually returns bodies shaped this
way today.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from aud.intelligence import advisor
from aud.intelligence.interface import (
    DEFAULT_MODELS,
    PROVIDER_ENV_VARS,
    AnthropicBackend,
    AzureOpenAIBackend,
    GoogleBackend,
    OpenAIBackend,
    resolve_backend,
)
from aud.schemas import AudError
from tests import replay

# ---------------------------------------------------------------------------
# _parse_model_json / _validate_and_build_plan: aud's OWN pure logic,
# exercised directly with literal inputs -- no backend, no third party.
# ---------------------------------------------------------------------------


def test_non_json_output_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json("not json at all")
    assert excinfo.value.code == "bad_model_output"


def test_top_level_shape_must_be_exactly_stages() -> None:
    bad = {"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "x"}], "notes": "extra"}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_empty_stage_list_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan({"stages": []})
    assert excinfo.value.code == "bad_model_plan"


def test_non_list_stages_is_rejected() -> None:
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan({"stages": "eq"})
    assert excinfo.value.code == "bad_model_plan"


def test_stage_entry_missing_a_required_key_is_rejected() -> None:
    bad = {"stages": [{"stage": "limit", "params": {}}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unknown_stage_name_is_rejected() -> None:
    bad = {"stages": [{"stage": "cut", "params": {}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_stage_name_not_in_allowed_advise_vocabulary_is_rejected_even_if_it_is_a_real_plan_stage() -> None:
    """'eq_match' is a real render stage but not one advise may choose --
    it needs a stored curve, which advise never holds."""
    bad = {"stages": [{"stage": "eq_match", "params": {"curve": []}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_duplicate_stage_is_rejected() -> None:
    bad = {
        "stages": [
            {"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "x"},
            {"stage": "loudness", "params": {"target_lufs": -16.0}, "reason": "y"},
        ]
    }
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_params_not_an_object_is_rejected() -> None:
    bad = {"stages": [{"stage": "limit", "params": [1, 2, 3], "reason": "x"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_blank_reason_is_rejected() -> None:
    bad = {"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "   "}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_out_of_range_param_is_rejected() -> None:
    """ceiling_dbtp must be <= 0.0 -- this reuses aud.lib.limit's own validator."""
    bad = {"stages": [{"stage": "limit", "params": {"ceiling_dbtp": 3.0}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_unexpected_param_name_is_rejected() -> None:
    """compress requires 'bands'; an unknown kwarg is a TypeError from the builder, caught and re-tagged."""
    bad = {"stages": [{"stage": "compress", "params": {"frobnicate": 1.0}, "reason": "because"}]}
    with pytest.raises(AudError) as excinfo:
        advisor._validate_and_build_plan(bad)
    assert excinfo.value.code == "bad_model_plan"


def test_well_formed_proposal_is_accepted_and_carries_reasoning() -> None:
    good = json.dumps(
        {
            "stages": [
                {
                    "stage": "eq",
                    "params": {"hpf": 60.0, "lpf": None, "peaks": [[3200.0, -2.0, 1.2]]},
                    "reason": "crest factor 14 dB and a quiet noise floor allow a gentle hpf at 60 Hz.",
                },
                {
                    "stage": "limit",
                    "params": {"ceiling_dbtp": -1.0},
                    "reason": "true peak is -6.2 dBTP today; -1.0 dBTP ceiling is the standard R128 max.",
                },
            ]
        }
    )
    plan, reasoning = advisor._validate_and_build_plan(advisor._parse_model_json(good))
    assert [s.stage for s in plan.stages] == ["eq", "limit"]
    assert len(reasoning) == 2
    assert all(r["reason"] for r in reasoning)


# ---------------------------------------------------------------------------
# Bounded repair for trailing content (issue #35): _parse_model_json accepts
# valid JSON followed by non-JSON trailing content (a model that answers
# correctly and then keeps writing), but still refuses two distinct JSON
# values. The trailing-prose case below replays a REAL recorded response
# (tests/fixtures/recorded/anthropic/advise-clean-haiku-trailing-prose.json)
# -- captured while reproducing this issue: 20 trials each against a clean
# control and a defective ("boxy") fixture, same model
# (claude-haiku-4-5-20251001) as the original report, using
# `expertise-probe`'s own audio_defects fixtures. 6/20 (30%) and 8/20 (40%)
# failed this exact way -- ALL 14 failures were trailing prose after a
# closing code fence, NONE were two distinct JSON objects. The two-object
# refusal case is the one exception in this file to "never hand-author a
# mock" (AGENTS.md SS3b): no real call in either 40-trial run produced it, so
# there is no recording to replay for it, and it is labelled SYNTHETIC below.
# ---------------------------------------------------------------------------


def test_trailing_prose_after_valid_json_is_accepted() -> None:
    """Replays advise-clean-haiku-trailing-prose.json: a valid ```json
    plan followed by the model's own closing fence and then unrequested
    "**Rationale:**" bullets. Before this fix, `json.loads` rejected the
    WHOLE response with `JSONDecodeError: Extra data: line 15 column 1` --
    issue #35's exact reported error, on this exact recorded text.
    """
    recording = replay.load_recording("anthropic", "advise-clean-haiku-trailing-prose")
    raw_text = replay.recorded_text_block(recording)
    assert "Rationale" in raw_text  # confirms this fixture really does trail prose after the fence
    proposal = advisor._parse_model_json(raw_text)
    plan, reasoning = advisor._validate_and_build_plan(proposal)
    assert [s.stage for s in plan.stages] == ["loudness", "limit"]
    assert len(reasoning) == 2


def test_advise_wiring_accepts_the_same_real_trailing_prose_response() -> None:
    """The same real recording as above, this time through the full
    `advise()` wiring (ReplayAdviceBackend), proving the fix holds at the
    level a real caller actually uses, not only at the isolated parser.
    """
    backend = replay.ReplayAdviceBackend("advise-clean-haiku-trailing-prose")
    plan, reasoning = advisor.advise(
        {"integrated_lufs": -20.0, "true_peak_dbtp": -7.99, "noise_floor_dbfs": -23.08},
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    assert [s.stage for s in plan.stages] == ["loudness", "limit"]
    assert len(reasoning) == 2


def test_trailing_whitespace_only_is_accepted() -> None:
    """Degenerate case of the same bug: a valid plan followed by nothing
    but whitespace (no remainder at all) must still parse."""
    good = '{"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "x"}]}\n\n   \n'
    proposal = advisor._parse_model_json(good)
    assert proposal["stages"][0]["stage"] == "limit"


def test_two_different_json_objects_is_refused() -> None:
    """SYNTHETIC -- hand-built, not a real recording (see the section
    docstring above: no real call in 40 trials ever produced this shape).
    The tool's existing refusal for genuine ambiguity must survive this
    fix: two DIFFERENT proposed plans is not something aud may silently
    pick between, regardless of how the first one is spelled.
    """
    two_json_values = (
        '{"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "first plan"}]}\n'
        '{"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "second plan"}]}'
    )
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(two_json_values)
    assert excinfo.value.code == "bad_model_output"
    assert "second" in excinfo.value.message.lower() or "distinct" in excinfo.value.message.lower()


# ---------------------------------------------------------------------------
# PR #41 review round 2, Fix #1 + Fix #2: a second plan ANYWHERE in the
# remainder (fenced, unfenced, or after prose) must refuse; ordinary trailing
# prose that merely OPENS like a JSON scalar must not. All SYNTHETIC --
# hand-built adversarial/regression inputs, not real recordings (see the
# section docstring above for why the one real recording this PR has,
# advise-clean-haiku-trailing-prose.json, is exercised separately below and
# is unaffected by this rule -- it carries no second value of any kind).
# ---------------------------------------------------------------------------

_FIRST_PLAN = '{"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "first plan"}]}'
_SECOND_PLAN = '{"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "second plan"}]}'


def test_second_plan_after_prose_is_refused() -> None:
    """Fix #1: 'prose, then {B}' -- round 1 accepted this (remainder does not
    BEGIN with JSON), silently discarding the model's second proposed plan."""
    text = f"{_FIRST_PLAN}\nAlternatively, you could instead try: {_SECOND_PLAN}"
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(text)
    assert excinfo.value.code == "bad_model_output"


def test_second_plan_in_a_second_fenced_block_is_refused() -> None:
    """Fix #1: a second fenced ```json block containing a full plan, after
    the first plan, must refuse -- round 1 accepted it."""
    text = f"{_FIRST_PLAN}\n\nOr, alternatively:\n```json\n{_SECOND_PLAN}\n```\n"
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(text)
    assert excinfo.value.code == "bad_model_output"


def test_second_plan_unfenced_after_an_unfenced_first_plan_is_refused() -> None:
    """Fix #1: 'unfenced A, then fenced B' -- the first plan is unfenced (no
    code fence at all), and a second, fenced plan follows it. Round 1
    accepted this because raw_decode of the first plan consumes the whole
    JSON object, leaving the fence+B as unexamined remainder."""
    text = f"{_FIRST_PLAN}\nHere is a second option:\n```\n{_SECOND_PLAN}\n```"
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(text)
    assert excinfo.value.code == "bad_model_output"


def test_bare_array_immediately_trailing_is_now_accepted() -> None:
    """INTENTIONAL VERDICT CHANGE (PR #41 review round 3, finding #1):
    through review round 2, '[1,2]' immediately after the first plan, with
    nothing else, was refused -- the whole-remainder check treated ANY
    compound (object OR array) value filling the remainder as ambiguous.
    A plan is never an array (`_validate_and_build_plan` requires a JSON
    OBJECT carrying `stages`), so an array can never be plan-shaped,
    whatever position it appears in or however much of the remainder it
    fills. This was also inconsistent with round 2's OWN stated rule that
    '[1] citations' survive -- a bare array should not flip to a refusal
    just because it happens to be the whole remainder instead of embedded
    in prose. See CHANGELOG for the explicit call-out of this change."""
    text = f"{_FIRST_PLAN}\n[1, 2]"
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_duplicate_plan_immediately_trailing_is_still_refused() -> None:
    """Baseline preserved: an exact duplicate of the first plan immediately
    following, with nothing else -- already correctly refused before this
    fix (see also test_two_different_json_objects_is_refused for two
    DIFFERENT plans; this is the exact-duplicate variant)."""
    text = f"{_FIRST_PLAN}\n{_FIRST_PLAN}"
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(text)
    assert excinfo.value.code == "bad_model_output"


@pytest.mark.parametrize(
    "trailing",
    [
        pytest.param("1. The low end could use a touch more warmth, but that's a matter of taste.", id="leading-digit"),
        pytest.param("true to the source material, this file needed very little intervention.", id="leading-true"),
        pytest.param(
            '"Less is more" applies here -- a light touch was all this file needed.', id="leading-quoted-string"
        ),
    ],
)
def test_trailing_prose_opening_like_a_json_scalar_is_accepted(trailing: str) -> None:
    """Fix #2: trailing prose that merely OPENS with a character JSON also
    uses for a bare scalar (digit, `true`, a quote) must NOT be treated as a
    second value -- round 1's `raw_decode(remainder)` parsed each of these
    as a bare JSON number/boolean/string and wrongly refused them."""
    text = f"{_FIRST_PLAN}\n{trailing}"
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_trailing_prose_with_a_footnote_style_citation_is_accepted() -> None:
    """Adversarial, hand-written (SYNTHETIC): '[1]' is valid JSON (a
    one-element array) but this is an ordinary footnote citation, not a
    second plan -- an unfenced array is deliberately never scanned for
    inside prose, only whole-remainder or fenced (see
    _contains_second_json_value's docstring)."""
    text = f"{_FIRST_PLAN}\nThis follows the loudness convention described in [1] and [2]."
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_trailing_prose_with_sic_braces_is_accepted() -> None:
    """Adversarial, hand-written (SYNTHETIC): '{sic}' uses curly braces but
    is not valid JSON (bareword, no quotes, no value) -- must not be
    mistaken for a second JSON object."""
    text = f"{_FIRST_PLAN}\nThe measurement report says 'compressor' {{sic}} throughout."
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_trailing_fenced_non_json_code_is_accepted() -> None:
    """Adversarial, hand-written (SYNTHETIC): a fenced code block whose
    content is not JSON at all (here, Python) must not be treated as a
    second value merely because it is fenced."""
    text = f'{_FIRST_PLAN}\nFor reference, here is how you would call it:\n```python\nprint("hello")\n```\n'
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


# ---------------------------------------------------------------------------
# PR #41 review round 3, finding #1: OVER-refusal. Round 2 refused on ANY
# trailing JSON object (fenced or not) -- but a parameter echo, an empty
# defaults object, or an unrelated note object are not plans. Only a
# PLAN-SHAPED object (carrying the `stages` key `_validate_and_build_plan`
# requires) is genuinely ambiguous with the first plan. All SYNTHETIC.
# ---------------------------------------------------------------------------

_PARAM_ECHO = '{"freq_hz": 120, "gain_db": -3}'


def test_trailing_prose_quoting_an_unrelated_param_object_is_accepted() -> None:
    """Finding #1: a JSON object quoted in prose that is NOT plan-shaped (no
    `stages` key) -- here, an eq param echo -- was wrongly refused through
    review round 2's "any object anywhere" rule."""
    text = f"{_FIRST_PLAN}\nThe eq stage uses {_PARAM_ECHO} for the low shelf."
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_trailing_prose_with_empty_defaults_object_is_accepted() -> None:
    """Finding #1: an empty `{}` ("keeps its defaults") is a JSON object but
    carries no `stages` key -- not plan-shaped, must be accepted."""
    text = _FIRST_PLAN + "\nThe reverb stage keeps its defaults ({})."
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_trailing_fenced_non_plan_object_is_accepted() -> None:
    """Finding #1: a FENCED JSON object that is not plan-shaped (no `stages`
    key) must be accepted -- round 2's fenced-block branch refused any
    fenced object/array regardless of shape. Also doubles as evidence for
    finding #2: the fenced-block branch was removed and this case is still
    correctly accepted by the brace scan alone, which sees the object's
    braces the same way whether or not backticks surround them."""
    text = _FIRST_PLAN + '\nFor reference:\n```json\n{"note": 1}\n```\n'
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_whole_remainder_bare_array_citation_is_accepted() -> None:
    """Finding #1: round 2 accepted '[1]' EMBEDDED in prose (a footnote
    citation, see test_trailing_prose_with_a_footnote_style_citation_is_accepted)
    but still refused it when it was the ENTIRE remainder with nothing
    else -- contradicting its own stated footnote-citation exception. A
    plan is never an array, so the whole-remainder check must never treat
    ANY array as plan-shaped, matching the embedded case exactly."""
    text = f"{_FIRST_PLAN}\n[1]"
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_whole_remainder_bare_scalar_only_is_accepted() -> None:
    """Finding #3 (untested branch): every round-1/round-2 scalar test
    (test_trailing_prose_opening_like_a_json_scalar_is_accepted) always had
    prose trailing AFTER the scalar, so a remainder that is a bare scalar
    with NOTHING else was never actually exercised. `_is_plan_shaped`
    requires a `dict`, so a scalar is rejected outright regardless of
    position -- but that was, until now, a claim, not a measurement."""
    text = f"{_FIRST_PLAN}\n42"
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


@pytest.mark.parametrize(
    ("case_id", "trailing", "expect_refused"),
    [
        ("immediately-second-object", _SECOND_PLAN, True),
        ("duplicate-object", _FIRST_PLAN, True),
        ("bare-array", "[1, 2]", False),  # round 3: INTENTIONAL verdict change, was True -- see CHANGELOG
        ("prose-then-object", f"Alternatively: {_SECOND_PLAN}", True),
        ("second-fenced-block", f"Or:\n```json\n{_SECOND_PLAN}\n```", True),
        ("unfenced-then-fenced", f"A second option:\n```\n{_SECOND_PLAN}\n```", True),
        ("leading-digit-prose", "1. The low end could use more warmth.", False),
        ("leading-true-prose", "true to the source, minimal intervention.", False),
        ("leading-quoted-string-prose", '"Less is more" applied here.', False),
        ("footnote-citation", "See [1] and [2] for background.", False),
        ("sic-braces", "the report says 'compressor' {sic} throughout.", False),
        ("fenced-non-json", '```python\nprint("hi")\n```', False),
        ("empty-trailing-whitespace", "\n\n   \n", False),
        ("param-echo-object", f"uses {_PARAM_ECHO} for the low shelf.", False),  # round 3, finding #1
        ("empty-defaults-object", "keeps its defaults ({}).", False),  # round 3, finding #1
        ("fenced-non-plan-object", '```json\n{"note": 1}\n```', False),  # round 3, findings #1 + #2
        ("bare-array-single-citation", "[1]", False),  # round 3, finding #1
        ("bare-scalar-only", "42", False),  # round 3, finding #3
    ],
)
def test_second_value_detection_full_verdict_table(case_id: str, trailing: str, expect_refused: bool) -> None:
    """One parametrized table covering every case named in PR #41 review
    rounds 2 and 3's tables plus this pass's own adversarial inputs, so the
    verdict for each is pinned individually and reportable as one table
    (see the PR description for the rendered version)."""
    text = f"{_FIRST_PLAN}\n{trailing}"
    if expect_refused:
        with pytest.raises(AudError) as excinfo:
            advisor._parse_model_json(text)
        assert excinfo.value.code == "bad_model_output", case_id
    else:
        proposal = advisor._parse_model_json(text)
        assert proposal["stages"][0]["stage"] == "limit", case_id


# ---------------------------------------------------------------------------
# advise(): wiring, using a REAL recorded response replayed through a
# recording-backed IntelligenceBackend (tests/replay.py). Never a
# hand-authored plan -- these are the model's actual real answers.
# ---------------------------------------------------------------------------


def test_well_formed_real_response_is_accepted_and_carries_reasoning() -> None:
    """Replays anthropic/advise-clean-sonnet-thinking.json -- a REAL
    response (eq, loudness, limit), captured from a 'thinking'-capable
    model, after AnthropicBackend already stripped the thinking block.
    """
    backend = replay.ReplayAdviceBackend("advise-clean-sonnet-thinking")
    plan, reasoning = advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    assert [s.stage for s in plan.stages] == ["eq", "loudness", "limit"]
    assert len(reasoning) == 3
    assert reasoning[0]["stage"] == "eq"


def test_wrapped_code_fence_is_really_present_in_the_haiku_recording_and_is_stripped() -> None:
    """anthropic/advise-clean-haiku.json's real response is genuinely
    wrapped in ```json fences (Haiku's own formatting choice, not
    something this test constructs) -- proving the fence-stripping code
    against real fenced output, not a hand-wrapped string.
    """
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    raw_text = replay.recorded_text_block(backend._recording)
    assert raw_text.strip().startswith("```")  # confirms the fixture really is fenced

    plan, _reasoning = advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    assert [s.stage for s in plan.stages] == ["loudness", "limit"]


def test_backend_is_called_with_the_requested_model_and_max_tokens() -> None:
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model="fake-model-1",
        max_tokens=999,
    )
    assert backend.calls[0]["model"] == "fake-model-1"
    assert backend.calls[0]["max_tokens"] == 999
    assert "aud" in backend.calls[0]["system"]


def test_reference_measurements_reach_the_user_prompt() -> None:
    backend = replay.ReplayAdviceBackend("advise-clean-haiku")
    advisor.advise(
        {"integrated_lufs": -11.76, "true_peak_dbtp": -8.76},
        target_lufs=-14.0,
        ceiling_dbtp=-1.0,
        reference_measurements={"integrated_lufs": -10.0},
        backend=backend,
        model="fake",
    )
    assert "reference" in backend.calls[0]["user"].lower()
    assert "-10.0" in backend.calls[0]["user"]


# ---------------------------------------------------------------------------
# The real AnthropicBackend, transport replayed -- proves the actual
# content-block parsing that broke on a 'thinking' response, against a
# REAL captured response of each shape.
# ---------------------------------------------------------------------------


def test_anthropic_backend_parses_a_plain_text_content_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replays anthropic/advise-clean-haiku.json -- content: ["text"]."""
    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-haiku")
    backend = AnthropicBackend(api_key="test-key")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)


def test_anthropic_backend_real_call_confirms_disable_parallel_tool_use_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replays advise-clean-haiku-tool-use-disable-parallel.json -- a REAL
    captured call (PR #41 review round 2, Fix #1) proving the ACTUAL request
    sent to Anthropic carries `tool_choice.disable_parallel_tool_use: true`,
    and the API accepted it and returned exactly one tool_use block that
    parses into a valid plan. Not just that our code claims to send the
    field -- that a live call with it present succeeded."""
    recording = replay.load_recording("anthropic", "advise-clean-haiku-tool-use-disable-parallel")
    assert recording["request"]["tool_choice"]["disable_parallel_tool_use"] is True
    assert [b["type"] for b in recording["response"]["content"]] == ["tool_use"]

    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del timeout
        sent = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        assert sent["tool_choice"]["disable_parallel_tool_use"] is True
        return _FakeHTTPResponse(json.dumps(recording["response"]).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    backend = AnthropicBackend(api_key="fake")
    text = backend.complete(
        "system prompt",
        "user prompt",
        model=recording["request"]["model"],
        response_schema=advisor._ADVISE_RESPONSE_SCHEMA,
    )
    real_input = recording["response"]["content"][0]["input"]
    assert json.loads(text) == real_input


def test_anthropic_backend_refuses_more_than_one_matching_tool_use_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SYNTHETIC -- hand-built, not a real recording. PR #41 review round 2,
    Fix #1: two matching tool_use blocks in one response is the same
    ambiguity `advisor._parse_model_json` already refuses for two distinct
    JSON values -- round 1 silently returned only the first. This should
    never happen for real once `disable_parallel_tool_use` is honoured (see
    the request-shape assertion in
    test_anthropic_backend_uses_a_forced_tool_call_when_response_schema_is_given),
    but the refusal is defence in depth against a future API version or
    tool_choice mode that returns more than one anyway."""
    recording = replay.load_recording("anthropic", "advise-clean-haiku-tool-use")
    real_input = recording["response"]["content"][0]["input"]
    schema = advisor._ADVISE_RESPONSE_SCHEMA
    two_tool_use_response = {
        **recording["response"],
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": schema.name, "input": real_input},
            {"type": "tool_use", "id": "toolu_2", "name": schema.name, "input": real_input},
        ],
    }

    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del request, timeout
        return _FakeHTTPResponse(json.dumps(two_tool_use_response).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    backend = AnthropicBackend(api_key="fake")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system prompt", "user prompt", model=recording["request"]["model"], response_schema=schema)
    assert excinfo.value.code == "bad_model_output"


def test_anthropic_backend_skips_a_leading_thinking_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replays anthropic/advise-clean-sonnet-thinking.json -- content:
    ["thinking", "text"], the exact shape that raised KeyError on
    `content[0]["text"]` before this was fixed. Found by calling a real
    reasoning-capable model; no hand-written fake could have produced it.
    """
    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-sonnet-thinking")
    assert [b["type"] for b in recording["response"]["content"]] == ["thinking", "text"]

    backend = AnthropicBackend(api_key="test-key")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)


def test_anthropic_backend_replay_fails_loudly_for_a_missing_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(replay.RecordingNotFoundError):
        replay.install_anthropic_urlopen_replay(monkeypatch, "does-not-exist")


# ---------------------------------------------------------------------------
# Strict structured output (issue #35): a forced tool call makes trailing
# content structurally impossible rather than something to detect and
# repair. Replays a REAL captured response (advise-clean-haiku-tool-use.json)
# from a REAL call made with `response_schema` set -- proves both that the
# REQUEST our code sends actually carries `tools`/`tool_choice` (the
# constraint was really requested) and that the returned text is the real
# parsed `input`, never raw free text with room for prose to trail.
# ---------------------------------------------------------------------------


def test_anthropic_backend_uses_a_forced_tool_call_when_response_schema_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = replay.load_recording("anthropic", "advise-clean-haiku-tool-use")
    assert [b["type"] for b in recording["response"]["content"]] == ["tool_use"]  # confirms the real shape

    captured: dict[str, object] = {}

    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeHTTPResponse(json.dumps(recording["response"]).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    backend = AnthropicBackend(api_key="test-key")
    schema = advisor._ADVISE_RESPONSE_SCHEMA
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"], response_schema=schema)

    sent = captured["body"]
    # disable_parallel_tool_use is Fix #1's prevention half (PR #41 review
    # round 2): asks the API for at most one tool_use block, belt-and-
    # suspenders with the multi-block refusal tested separately below.
    assert sent["tool_choice"] == {  # type: ignore[index]
        "type": "tool",
        "name": schema.name,
        "disable_parallel_tool_use": True,
    }
    assert sent["tools"][0]["name"] == schema.name  # type: ignore[index]
    assert sent["tools"][0]["input_schema"] == schema.schema  # type: ignore[index]

    real_input = recording["response"]["content"][0]["input"]
    assert json.loads(text) == real_input  # a fresh json.dumps of the API's own already-parsed input


def test_anthropic_backend_ignores_response_schema_argument_absence_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `response_schema` -> the request must NOT carry tools/tool_choice
    -- the free-text path (still bounded-repaired by advisor._parse_model_json)
    is unchanged for a caller that does not opt in."""
    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-haiku")
    backend = AnthropicBackend(api_key="test-key")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)


# ---------------------------------------------------------------------------
# OpenAI / Google / Azure OpenAI response-shape parsing -- docs-shaped, not
# recorded live (see the module docstring's honest caveat above). Each pair
# proves the same two things the Anthropic replay tests prove for the real
# recorded shape: a well-formed body parses to the expected text, and a
# malformed one raises AudError(code="provider_request_failed") -- never an
# opaque KeyError/IndexError -- which is exactly the class of defect the
# real content[0]["text"] bug on Anthropic was.
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _install_docs_shaped_response(monkeypatch: pytest.MonkeyPatch, body: dict) -> None:
    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del request, timeout
        return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def _install_docs_shaped_response_capturing_request(
    monkeypatch: pytest.MonkeyPatch, body: dict, captured: dict[str, object]
) -> None:
    """Like `_install_docs_shaped_response`, but also records the request
    body our code actually sent in `captured["body"]`, so a test can
    assert the schema-constraint field was really requested, not just
    that our code claims to send it."""

    def _fake_urlopen(request: object, timeout: float | None = None) -> _FakeHTTPResponse:
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def test_openai_backend_parses_the_documented_chat_completions_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://platform.openai.com/docs/api-reference/chat -- documented
    shape, not a recorded live call (see module docstring)."""
    _install_docs_shaped_response(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hello from openai"}}]}
    )
    backend = OpenAIBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gpt-4o-mini") == "hello from openai"


def test_openai_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docs_shaped_response(monkeypatch, {"choices": []})  # documented field, unexpectedly empty
    backend = OpenAIBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gpt-4o-mini")
    assert excinfo.value.code == "provider_request_failed"


def test_openai_backend_requests_json_object_mode_when_response_schema_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """response_format: json_object (GA on Chat Completions) guarantees
    the whole response is one valid JSON value -- eliminates issue #35's
    trailing-content defect. Full per-field json_schema strict mode is not
    implemented (see OpenAIBackend.complete's comment)."""
    captured: dict[str, object] = {}
    _install_docs_shaped_response_capturing_request(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}, captured
    )
    backend = OpenAIBackend(api_key="fake-key")
    backend.complete("system", "user", model="gpt-4o-mini", response_schema=advisor._ADVISE_RESPONSE_SCHEMA)
    assert captured["body"]["response_format"] == {"type": "json_object"}  # type: ignore[index]


def test_openai_backend_omits_response_format_with_no_response_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    _install_docs_shaped_response_capturing_request(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}, captured
    )
    backend = OpenAIBackend(api_key="fake-key")
    backend.complete("system", "user", model="gpt-4o-mini")
    assert "response_format" not in captured["body"]  # type: ignore[operator]


def test_google_backend_parses_the_documented_generate_content_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://ai.google.dev/api/generate-content -- documented shape, not
    a recorded live call (see module docstring)."""
    _install_docs_shaped_response(
        monkeypatch, {"candidates": [{"content": {"parts": [{"text": "hello from google"}]}}]}
    )
    backend = GoogleBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gemini-2.0-flash") == "hello from google"


def test_google_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docs_shaped_response(monkeypatch, {"candidates": [{"content": {"parts": []}}]})
    backend = GoogleBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gemini-2.0-flash")
    assert excinfo.value.code == "provider_request_failed"


def test_google_backend_skips_a_leading_thought_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """With thinking enabled, Gemini returns a part carrying `"thought":
    true` ahead of the actual answer part -- the SAME hazard class as
    AnthropicBackend's `content[0]["text"]` bug (see interface.py). Docs-
    shaped (Google has never been called live -- see module docstring),
    grounded in Gemini's own docs:
    https://ai.google.dev/gemini-api/docs/generate-content/thinking.
    """
    _install_docs_shaped_response(
        monkeypatch,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "internal reasoning, not the answer"},
                            {"text": "hello from google"},
                        ]
                    }
                }
            ]
        },
    )
    backend = GoogleBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gemini-2.0-flash") == "hello from google"


def test_google_backend_all_thought_parts_is_a_named_error_not_the_thought_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_docs_shaped_response(
        monkeypatch, {"candidates": [{"content": {"parts": [{"thought": True, "text": "only thinking"}]}}]}
    )
    backend = GoogleBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gemini-2.0-flash")
    assert excinfo.value.code == "provider_request_failed"


def test_google_backend_joins_two_non_thought_text_parts_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """SYNTHETIC, docs-shaped (Google has never been called live -- see
    module docstring). PR #41 review round 2, Fix #3: Gemini can split one
    JSON response across multiple non-thought parts (e.g. '{\"stages\": ['
    as one part, the rest as a second) -- round 1 took only the FIRST
    non-thought part and returned a truncated fragment that failed to
    parse downstream. The fix joins ALL non-thought parts, in order."""
    _install_docs_shaped_response(
        monkeypatch,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"stages": ['},
                            {"text": '{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "x"}]}'},
                        ]
                    }
                }
            ]
        },
    )
    backend = GoogleBackend(api_key="fake")
    text = backend.complete("system", "user", model="gemini-2.0-flash")
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "limit"


def test_google_backend_joins_text_parts_skipping_a_leading_thought_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """SYNTHETIC, docs-shaped. The thought-skip (existing, pre-this-fix
    behaviour) and the multi-part join (Fix #3) must compose: a thought
    part ahead of TWO non-thought parts must still join only the latter."""
    _install_docs_shaped_response(
        monkeypatch,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "internal reasoning, not the answer"},
                            {"text": '{"stages": ['},
                            {"text": '{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "y"}]}'},
                        ]
                    }
                }
            ]
        },
    )
    backend = GoogleBackend(api_key="fake")
    text = backend.complete("system", "user", model="gemini-2.0-flash")
    proposal = advisor._parse_model_json(text)
    assert proposal["stages"][0]["stage"] == "loudness"


def test_google_backend_joined_parts_containing_two_complete_plans_is_refused_downstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SYNTHETIC. If joining non-thought parts happens to concatenate TWO
    complete plans, that is exactly the ambiguity
    `advisor._parse_model_json` already refuses -- GoogleBackend's job is
    only to reassemble the text faithfully, never to judge its content."""
    first = '{"stages": [{"stage": "limit", "params": {"ceiling_dbtp": -1.0}, "reason": "first plan"}]}'
    second = '{"stages": [{"stage": "loudness", "params": {"target_lufs": -14.0}, "reason": "second plan"}]}'
    _install_docs_shaped_response(
        monkeypatch,
        {"candidates": [{"content": {"parts": [{"text": first}, {"text": "\n"}, {"text": second}]}}]},
    )
    backend = GoogleBackend(api_key="fake")
    text = backend.complete("system", "user", model="gemini-2.0-flash")
    with pytest.raises(AudError) as excinfo:
        advisor._parse_model_json(text)
    assert excinfo.value.code == "bad_model_output"


def test_google_backend_requests_json_mime_type_when_response_schema_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _install_docs_shaped_response_capturing_request(
        monkeypatch, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}, captured
    )
    backend = GoogleBackend(api_key="fake-key")
    backend.complete("system", "user", model="gemini-2.0-flash", response_schema=advisor._ADVISE_RESPONSE_SCHEMA)
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"  # type: ignore[index]


def test_azure_openai_backend_parses_the_documented_chat_completions_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """https://learn.microsoft.com/azure/ai-services/openai/reference --
    documented shape, not a recorded live call (see module docstring)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-resource.openai.azure.com")
    _install_docs_shaped_response(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "hello from azure"}}]}
    )
    backend = AzureOpenAIBackend(api_key="fake-key")
    assert backend.complete("system", "user", model="gpt-4o-mini") == "hello from azure"


def test_azure_openai_backend_malformed_response_is_a_named_error_not_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-resource.openai.azure.com")
    _install_docs_shaped_response(monkeypatch, {"choices": [{"message": {}}]})  # missing "content"
    backend = AzureOpenAIBackend(api_key="fake-key")
    with pytest.raises(AudError) as excinfo:
        backend.complete("system", "user", model="gpt-4o-mini")
    assert excinfo.value.code == "provider_request_failed"


def test_azure_openai_backend_requests_json_object_mode_when_response_schema_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_object mode is supported since API version 2023-12-01-preview
    -- well before this backend's default (2024-06-01, or whatever
    AZURE_OPENAI_API_VERSION overrides it to). Full json_schema strict
    mode needs 2024-08-01-preview+ and is NOT implemented here -- see
    AzureOpenAIBackend.complete's comment."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-resource.openai.azure.com")
    captured: dict[str, object] = {}
    _install_docs_shaped_response_capturing_request(
        monkeypatch, {"choices": [{"message": {"role": "assistant", "content": "{}"}}]}, captured
    )
    backend = AzureOpenAIBackend(api_key="fake-key")
    backend.complete("system", "user", model="gpt-4o-mini", response_schema=advisor._ADVISE_RESPONSE_SCHEMA)
    assert captured["body"]["response_format"] == {"type": "json_object"}  # type: ignore[index]


# --- Provider selection -------------------------------------------------------


def test_provider_env_vars_match_the_manifest_order() -> None:
    assert PROVIDER_ENV_VARS == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
    )


def test_resolve_backend_raises_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AudError) as excinfo:
        resolve_backend(None)
    assert excinfo.value.code == "provider_credential_missing"
    for var in PROVIDER_ENV_VARS:
        assert var in excinfo.value.message or var in excinfo.value.remedy


def test_resolve_backend_picks_first_configured_provider_in_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    _backend, provider, model = resolve_backend(None)
    assert provider == "openai"
    assert model == DEFAULT_MODELS["openai"]


def test_resolve_backend_honours_explicit_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    _backend, provider, model = resolve_backend("claude-custom")
    assert provider == "anthropic"
    assert model == "claude-custom"


def test_resolve_backend_honours_aud_model_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("AUD_MODEL", "claude-from-env")
    _backend, provider, model = resolve_backend(None)
    assert provider == "anthropic"
    assert model == "claude-from-env"


def test_gemini_api_key_also_selects_the_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    _backend, provider, _model = resolve_backend(None)
    assert provider == "google"


@pytest.mark.parametrize("key_name", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_google_supported_default_and_explicit_selections_are_preserved(monkeypatch, key_name) -> None:
    # Resolution only: never call the provider or pretend to prove live availability.
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AUD_MODEL", raising=False)
    monkeypatch.setenv(key_name, "unused-resolution-only")
    backend, provider, model = resolve_backend()
    assert isinstance(backend, GoogleBackend)
    assert provider == "google"
    assert model == "gemini-3.5-flash-lite"

    monkeypatch.setenv("AUD_MODEL", "gemini-2.0-flash")
    assert resolve_backend()[2] == "gemini-2.0-flash"
    assert resolve_backend("user-selected-model")[2] == "user-selected-model"
    assert resolve_backend("gemini-2.0-flash")[2] == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Model-tier-dependent gate/expand behaviour, pinned against REAL recorded
# advise responses (never a hand-authored plan -- see tests/replay.py and
# RECORDING.md's anthropic/ note).
#
# Measured on identical prompts and identical, real measurements
# (noise_floor_dbfs -30.51 vs integrated_lufs -13.06 -- a ~17.4 dB gap,
# well under the ~40 dB "clean" separation the system prompt names):
# claude-sonnet-5 reached for 'expand' and cited the two numbers; claude-
# haiku-4-5 did not reach for either 'gate' or 'expand' on the same
# material. On genuinely clean material (noise_floor_dbfs -64.54 vs
# integrated_lufs -11.76, a ~52.8 dB gap), neither model proposes one.
# Before this test, that was anecdote from one interactive session. These
# assertions run the real recorded text through aud's OWN validator and
# plan builder (advisor._validate_and_build_plan, via advisor.advise()),
# so a prompt change that broke this reasoning would fail a real test
# rather than requiring someone to notice the story stopped being true.
# ---------------------------------------------------------------------------


def _advise_from_recording(recording_name: str) -> tuple[list[str], list[dict[str, str]]]:
    backend = replay.ReplayAdviceBackend(recording_name)
    plan, reasoning = advisor.advise(
        {"integrated_lufs": -13.06, "true_peak_dbtp": -6.95, "noise_floor_dbfs": -30.51},
        target_lufs=-16.0,
        ceiling_dbtp=-1.0,
        reference_measurements=None,
        backend=backend,
        model=backend.recorded_model,
    )
    return [s.stage for s in plan.stages], reasoning


def test_hissy_material_sonnet_thinking_response_yields_a_quiet_end_stage() -> None:
    """Replays anthropic/advise-hissy-sonnet-thinking.json -- the real
    response that reached for 'expand' and cited noise_floor_dbfs vs
    integrated_lufs. Pins that this recorded response, run through aud's
    real validator, actually produces a chain containing a quiet-end
    repair stage -- not just that the recorded text mentions one.
    """
    stage_names, reasoning = _advise_from_recording("advise-hissy-sonnet-thinking")
    assert "expand" in stage_names
    assert "gate" not in stage_names  # the model chose the gentler device, not the harder one
    expand_reason = next(r["reason"] for r in reasoning if r["stage"] == "expand")
    assert "17.4" in expand_reason or "noise_floor" in expand_reason.lower()


def test_hissy_material_haiku_response_does_not_reach_for_a_quiet_end_stage() -> None:
    """Replays anthropic/advise-hissy-haiku.json -- the real response from
    a smaller/faster model on the SAME material and prompt that DID NOT
    propose gate/expand. This is the pinned half of the model-tier finding:
    identical measurements, different model, different chain.
    """
    stage_names, _reasoning = _advise_from_recording("advise-hissy-haiku")
    assert "gate" not in stage_names
    assert "expand" not in stage_names


@pytest.mark.parametrize("recording_name", ["advise-clean-haiku", "advise-clean-sonnet-thinking"])
def test_clean_material_responses_never_reach_for_a_quiet_end_stage(recording_name: str) -> None:
    """Replays both clean-material recordings (haiku and sonnet-thinking):
    on a ~52.8 dB noise-floor/loudness separation, well above the system
    prompt's ~40 dB clean-recording guideline, neither real model proposed
    gate or expand. This is the control for the two tests above -- proves
    the quiet-end stage tracks the ACTUAL noise floor, not just the model.
    """
    stage_names, _reasoning = _advise_from_recording(recording_name)
    assert "gate" not in stage_names
    assert "expand" not in stage_names


def test_azure_openai_requires_an_endpoint_too(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
    with pytest.raises(AudError) as excinfo:
        resolve_backend(None)
    assert excinfo.value.code == "provider_config_incomplete"
