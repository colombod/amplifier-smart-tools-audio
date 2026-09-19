"""Tests for the replay harness itself (tests/replay.py).

Not a test of `aud` -- a test that the mechanism this whole test suite now
depends on (RECORDING.md's contract: fail loudly, never invent a value,
surface version drift) actually behaves that way.
"""

from __future__ import annotations

import pytest

from tests import replay


def test_load_recording_fails_loudly_for_a_missing_recording() -> None:
    with pytest.raises(replay.RecordingNotFoundError):
        replay.load_recording("faster_whisper", "does-not-exist")


def test_wav_path_fails_loudly_for_an_unshipped_long_variant() -> None:
    """The 22050/44100/48000 Hz 'long' source wavs were transcribed but
    deliberately not committed (RECORDING.md, to keep the fixture directory
    small) -- asking for one must fail loudly, not silently return a
    similar-but-wrong file.
    """
    with pytest.raises(replay.RecordingNotFoundError):
        replay.wav_path("speech_long_48000.wav")


def test_assert_matches_recorded_source_fails_loudly_on_a_mismatched_wav() -> None:
    """The one rule this harness cannot relax: a replay bound to the wrong
    input bytes must fail loudly, not silently proceed."""
    recording = replay.load_recording("faster_whisper", "speech_short_16000__resampled_16k")
    wrong_wav = replay.wav_path("speech_short_48000.wav")  # real file, just the wrong one
    with pytest.raises(replay.RecordingNotFoundError):
        replay.assert_matches_recorded_source(recording, wrong_wav)


def test_assert_matches_recorded_source_accepts_the_real_matching_wav() -> None:
    recording = replay.load_recording("faster_whisper", "speech_short_16000__resampled_16k")
    right_wav = replay.wav_path("speech_short_16000.wav")
    replay.assert_matches_recorded_source(recording, right_wav)  # must not raise


def test_install_faster_whisper_replay_requires_a_source_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """`source` has no default -- a caller MUST say which source drives the
    call, or declare `UNBOUND(...)` explicitly. This is the structural form
    of the trap RECORDING.md warns about: a replay that never says what it
    is bound to is testing nothing about correspondence to real audio.
    """
    with pytest.raises(TypeError):
        replay.install_faster_whisper_replay(monkeypatch, "speech_short_16000__raw")  # type: ignore[call-arg]


def test_install_faster_whisper_replay_verifies_the_sha256_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing a `Path` as `source` must be checked against the recording's
    own provenance -- the wrong wav must fail loudly, not silently bind.
    """
    wrong_wav = replay.wav_path("speech_short_48000.wav")  # real file, just the wrong one
    with pytest.raises(replay.RecordingNotFoundError):
        replay.install_faster_whisper_replay(monkeypatch, "speech_short_16000__resampled_16k", source=wrong_wav)


def test_install_faster_whisper_replay_accepts_the_real_matching_source(monkeypatch: pytest.MonkeyPatch) -> None:
    right_wav = replay.wav_path("speech_short_16000.wav")
    replayed = replay.install_faster_whisper_replay(monkeypatch, "speech_short_16000__resampled_16k", source=right_wav)
    assert replayed.source == right_wav


def test_unbound_rejects_an_empty_reason() -> None:
    """An exemption with no reason is a gap wearing the shape of a decision."""
    with pytest.raises(ValueError, match="non-empty reason"):
        replay.UNBOUND("")
    with pytest.raises(ValueError, match="non-empty reason"):
        replay.UNBOUND("   ")


def test_install_faster_whisper_replay_accepts_an_explicit_unbound_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`source=UNBOUND("reason")` is the visible, greppable opt-out -- no
    sha256 check runs, and the reason travels with the installed replay.
    """
    unbound = replay.UNBOUND("this test does not drive the call with real audio at all")
    replayed = replay.install_faster_whisper_replay(monkeypatch, "speech_short_16000__raw", source=unbound)
    assert replayed.source is unbound
    assert isinstance(replayed.source, replay.UnboundSource)
    assert replayed.source.reason == "this test does not drive the call with real audio at all"


def test_anthropic_urlopen_replay_fails_loudly_on_a_model_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request-shape check (model name) is meant to catch a caller
    reusing a recording with the wrong model -- prove it actually fires.
    """
    import urllib.request

    replay.install_anthropic_urlopen_replay(monkeypatch, "advise-clean-haiku")
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=b'{"model": "not-the-recorded-model", "system": "s", "messages": [{"role": "user", "content": "u"}]}',
        method="POST",
    )
    with pytest.raises(AssertionError, match="does not match"):
        urllib.request.urlopen(request)


def test_recorded_versions_are_pinned_and_visible() -> None:
    """One place a version bump becomes a visible diff instead of a silent
    assumption -- see RECORDING.md's 'Versions recorded' table. When these
    recordings are refreshed against a newer library, this assertion is
    meant to be edited deliberately, not left to drift unnoticed.
    """
    versions = replay.recorded_versions()
    assert versions == {
        "faster_whisper": "1.2.1",
        "ctranslate2": "4.8.2",
        "python_stretch": "0.3.1",
        "anthropic_api_version": "2023-06-01",
    }
