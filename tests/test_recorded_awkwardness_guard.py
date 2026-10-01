"""Guards the RECORDINGS, not the code under test -- that is the point.

Every other test file in this suite asks "does `aud` behave correctly against
a real recorded run". This file asks a different question: "do the recordings
themselves still contain the awkward, inconvenient truths RECORDING.md
documents, and does the replay harness still hand them back unaltered?"

Why that question needs its own test: nothing else in the suite would notice
if a *future* re-recording (a newer faster-whisper, a newer python_stretch, a
different Anthropic response) happened to come back tidier than reality --
no degenerate word, no hallucinated silence transcript, an exact-float
`timeFactor`, attributes where the real library has methods. Every behavioural
test would keep passing, having quietly stopped testing what AGENTS.md SS3b
calls "the awkward parts". A replay that is tidier than the real library is a
fake again -- and the only way to catch that shape of regression is to pin
the awkward properties directly, so a future re-recording that loses one
fails LOUDLY, naming which property and which recording, instead of the
coverage evaporating in silence.

So: the first half of this file reads the raw recording JSON and asserts each
documented quirk is still there, by name, in the recording it lives in. The
second half installs the replay (`tests/replay.py`) over each recording and
asserts the replayed surface reproduces the same quirk unaltered -- proving
the harness is not quietly normalising anything on the way out.
"""

from __future__ import annotations

import pytest

from tests import replay

# ---------------------------------------------------------------------------
# Part 1: the RECORDINGS still contain their documented awkward truths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recording_name", "expected_degenerate_count"),
    [
        ("speech_short_44100__raw", 4),
        ("speech_short_48000__raw", 5),
    ],
)
def test_recording_degenerate_word_counts_match_recording_md(
    recording_name: str, expected_degenerate_count: int
) -> None:
    """RECORDING.md: re-recorded against openai-whisper for issue #44, the
    short-fixture matrix really produced 9 degenerate words across these two
    runs (none of them fillers -- see test_dsp_detect.py). If a re-recording
    changes either count, RECORDING.md's claim needs re-verifying, not just
    this test updating.
    """
    recording = replay.load_recording("openai_whisper", recording_name)
    degenerate = [w for w in recording["words_flat"] if w["start"] == w["end"]]
    assert len(degenerate) == expected_degenerate_count, (
        f"{recording_name} must still contain {expected_degenerate_count} degenerate word(s) -- "
        f"found {len(degenerate)}. If this recording was refreshed, RECORDING.md's degenerate-word "
        f"claim needs re-verifying, not just this test updating."
    )


def test_recording_degenerate_words_sum_to_nine_across_the_short_fixture_matrix() -> None:
    index = replay.load_recording("openai_whisper", "_index")
    total = sum(run["degenerate_word_count"] for run in index["runs"])
    assert total == 9, "RECORDING.md's openai-whisper degenerate-word total must still hold"


def test_recording_silence_returns_zero_segments_not_a_hallucination() -> None:
    """RECORDING.md: unlike faster-whisper (which returned one segment with
    one hallucinated word on digital silence -- see RECORDING.md's
    old-vs-new comparison), openai-whisper returns NOTHING on the same
    silence fixture -- measured directly while resolving issue #44. A
    recording that came back non-empty on a re-record would silently erase
    this real, measured behavioural difference between the two engines.
    """
    recording = replay.load_recording("openai_whisper", "nospeech_silence_16000__raw")
    assert recording["segments"] == []
    assert recording["words_flat"] == []


def test_recording_pure_tone_returns_zero_segments() -> None:
    """The other half of the same fact: a pure tone also returns zero
    segments. Pinned alongside the silence case because openai-whisper's
    behaviour is now IDENTICAL on both (unlike faster-whisper, where
    silence and tone diverged) -- see RECORDING.md.
    """
    recording = replay.load_recording("openai_whisper", "nospeech_tone_16000__raw")
    assert recording["segments"] == []
    assert recording["words_flat"] == []


def test_recording_signalsmith_timefactor_is_the_reciprocal_of_the_requested_ratio() -> None:
    """RECORDING.md: `timeFactor` is the reciprocal of the duration ratio,
    exactly, at six factors -- e.g. requesting 1.2 sets timeFactor to
    0.8333... and yields out/in == 1.2, not 0.8333.
    """
    recording = replay.load_stretch_recording()
    entries = recording["time_factor_reciprocal"]
    assert len(entries) == 6
    for entry in entries:
        requested = entry["requested_duration_factor"]
        assert entry["ratio_out_over_in"] == pytest.approx(requested, rel=1e-3)
        assert entry["time_factor_set"] == pytest.approx(1.0 / requested, rel=1e-6)
        # The one entry AGENTS.md/RECORDING.md calls out by exact figure:
        if requested == 1.2:
            assert entry["time_factor_set"] == pytest.approx(0.8333333134651184, abs=1e-9)


def test_recording_signalsmith_timefactor_reads_back_as_float32_not_the_set_double() -> None:
    """RECORDING.md: `timeFactor` reads back through float32 -- setting 0.8
    reads back as 0.800000011920929, not the Python float exactly. A
    recording (or replay) that stored the double precisely would hide a
    real precision detail a caller might depend on.
    """
    recording = replay.load_stretch_recording()
    entry = next(e for e in recording["time_factor_direct"] if e["requested_duration_factor"] == 0.8)
    read_back = entry["stretch_state_after_process"]["timeFactor"]
    assert read_back == pytest.approx(0.800000011920929, abs=1e-15)
    assert read_back != 0.8, "the float32 read-back must differ from the exact double that was set"


def test_recording_signalsmith_latency_fields_are_methods_not_attributes() -> None:
    """RECORDING.md: `inputLatency`, `outputLatency`, `blockSamples`,
    `intervalSamples` are METHODS on the real object, not attributes --
    only `sampleRate` and `timeFactor` are public data attributes.
    """
    recording = replay.load_stretch_recording()
    inventory = recording["api_inventory"]["Stretch_instance_after_preset"]
    callables = set(inventory["public_callables"])
    non_callable_names = {entry["name"] for entry in inventory["public_non_callable"]}

    for name in ("inputLatency", "outputLatency", "blockSamples", "intervalSamples"):
        assert name in callables, f"{name} must still be recorded as a method (callable), not an attribute"
        assert name not in non_callable_names

    assert non_callable_names == {"sampleRate", "timeFactor"}


def test_recording_anthropic_has_both_a_text_only_and_a_thinking_plus_text_response() -> None:
    """RECORDING.md / AGENTS.md SS3b: a hand-written provider fake always
    returned a single text block, so a reasoning model's `thinking`-first
    response was unusable in production while the suite stayed green. Both
    shapes must still be present among the recordings, or that exact
    regression could recur invisibly.
    """
    text_only = {
        name: [b["type"] for b in replay.load_recording("anthropic", name)["response"]["content"]]
        for name in ("advise-clean-haiku", "advise-hissy-haiku")
    }
    thinking_first = {
        name: [b["type"] for b in replay.load_recording("anthropic", name)["response"]["content"]]
        for name in ("advise-clean-sonnet-thinking", "advise-hissy-sonnet-thinking")
    }
    for name, types in text_only.items():
        assert types == ["text"], f"{name} must still be a text-only response"
    for name, types in thinking_first.items():
        assert types == ["thinking", "text"], f"{name} must still lead with a 'thinking' block before 'text'"


# ---------------------------------------------------------------------------
# Part 2: the REPLAY surfaces every one of those properties unaltered.
# A harness that filters or normalises any of them on the way out has become
# a fake again, even though the underlying recording is still honest.
# ---------------------------------------------------------------------------


def test_replay_surfaces_the_degenerate_word_unaltered(monkeypatch: pytest.MonkeyPatch) -> None:
    words = replay.load_words("speech_short_48000__raw")
    degenerate = [w for w in words if w.start == w.end]
    assert len(degenerate) == 5
    assert all(w.word.strip() for w in degenerate), "a degenerate word still carries real recognised text"

    # And through the full installed replay surface, not just load_words:
    installed = replay.install_openai_whisper_replay(
        monkeypatch,
        "speech_short_48000__raw",
        source=replay.UNBOUND("this guard checks the replayed ANSWER only, not correspondence to any audio"),
    )
    all_words = [w for seg in installed.result()["segments"] for w in seg["words"]]
    degenerate_via_segments = [w for w in all_words if w["start"] == w["end"]]
    assert len(degenerate_via_segments) == 5


def test_replay_surfaces_silence_and_tone_zero_segments_unaltered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silence = replay.install_openai_whisper_replay(
        monkeypatch,
        "nospeech_silence_16000__raw",
        source=replay.UNBOUND("proving the replayed segment/word count, not correspondence to driving audio"),
    )
    assert silence.result()["segments"] == []

    with pytest.MonkeyPatch.context() as mp:
        tone = replay.install_openai_whisper_replay(
            mp,
            "nospeech_tone_16000__raw",
            source=replay.UNBOUND("proving the replayed segment count, not correspondence to driving audio"),
        )
        assert tone.result()["segments"] == []


def test_replay_surfaces_signalsmith_timefactor_reciprocal_unaltered(monkeypatch: pytest.MonkeyPatch) -> None:
    replay.install_python_stretch_replay(monkeypatch)
    import python_stretch as ps  # the installed replay module

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    stretch.timeFactor = 1.0 / 1.2  # aud's own inversion for a requested factor of 1.2
    assert stretch.timeFactor == pytest.approx(0.8333333134651184, abs=1e-9)


def test_replay_surfaces_the_float32_timefactor_readback_unaltered(monkeypatch: pytest.MonkeyPatch) -> None:
    replay.install_python_stretch_replay(monkeypatch)
    import python_stretch as ps

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)
    stretch.timeFactor = 0.8
    assert stretch.timeFactor == pytest.approx(0.800000011920929, abs=1e-15)
    assert stretch.timeFactor != 0.8


def test_replay_surfaces_the_methods_not_attributes_quirk_unaltered(monkeypatch: pytest.MonkeyPatch) -> None:
    replay.install_python_stretch_replay(monkeypatch)
    import python_stretch as ps

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 22050)

    assert callable(stretch.inputLatency)
    assert callable(stretch.outputLatency)
    assert callable(stretch.blockSamples)
    assert callable(stretch.intervalSamples)
    with pytest.raises(TypeError, match="float\\(\\) argument must be a string or a real number"):
        float(stretch.inputLatency)  # type: ignore[arg-type]  # exactly the quirk under test


def test_replay_surfaces_anthropic_text_only_and_thinking_first_unaltered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The transport replay must hand back the recorded response bytes
    verbatim -- content-block order and count included -- for both shapes.
    """
    import json
    import urllib.request

    for name, expected_types in (
        ("advise-clean-haiku", ["text"]),
        ("advise-clean-sonnet-thinking", ["thinking", "text"]),
    ):
        with pytest.MonkeyPatch.context() as mp:
            replay.install_anthropic_urlopen_replay(mp, name)
            request = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(
                    {
                        "model": replay.load_recording("anthropic", name)["request"]["model"],
                        "system": "s",
                        "messages": [{"role": "user", "content": "u"}],
                    }
                ).encode("utf-8"),
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                body = json.loads(response.read())
            assert [block["type"] for block in body["content"]] == expected_types


def test_replay_surfaces_anthropic_thinking_response_through_the_real_backend_unaltered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One more layer up: the real `AnthropicBackend.complete()` -- not just
    the transport -- must still correctly skip the leading `thinking` block
    rather than choking on it (the exact regression AGENTS.md SS3b names).
    """
    from aud.intelligence.interface import AnthropicBackend

    recording = replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-sonnet-thinking")
    backend = AnthropicBackend(api_key="[REDACTED:SECRET]")
    text = backend.complete("system prompt", "user prompt", model=recording["request"]["model"])
    assert text == replay.recorded_text_block(recording)
    assert text, "the extracted text must be a real, non-empty string"
