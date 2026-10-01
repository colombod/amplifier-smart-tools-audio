"""Behavioural tests for aud.dsp.detect and aud.dsp.speech.

Every energy-domain signal (silence/transient detection) is synthesised
in-test (no committed binaries) -- this proves the numbers in the lane
report, not just that the code runs without raising.

Every openai-whisper-touching test below replays a REAL recorded
transcription from tests/fixtures/recorded/openai_whisper/ (see
tests/replay.py and RECORDING.md) instead of a hand-written fake
transcript -- the engine behind the optional `speech` extra (issue #44
replaced faster-whisper with openai-whisper; see RECORDING.md for the
re-recorded baseline and the old-vs-new comparison). Two real defects (a
degenerate word with no sample rate at all, and a sample-rate mismatch)
survived a green test suite built on hand-written fakes; the recordings
exist so those exact failure modes are what gets tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import detect
from aud.dsp import io as dsp_io
from aud.dsp import speech as dsp_speech
from aud.schemas import AudError
from tests import replay

_SR = 44100


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _SR))


def _burst(seconds: float, freq: float, rng: np.random.Generator, amplitude: float = 0.3) -> np.ndarray:
    n = int(seconds * _SR)
    t = np.arange(n) / _SR
    return amplitude * np.sin(2 * np.pi * freq * t) + 0.05 * amplitude * rng.standard_normal(n)


def _build_bursts_and_gaps(
    burst_lengths: list[float], gap_lengths: list[float], rng: np.random.Generator
) -> np.ndarray:
    """Interleave bursts and gaps: burst, gap, burst, gap, ..., burst."""
    assert len(burst_lengths) == len(gap_lengths) + 1
    pieces = [_burst(burst_lengths[0], 300.0, rng)]
    for gap_len, burst_len in zip(gap_lengths, burst_lengths[1:], strict=True):
        pieces.append(_silence(gap_len))
        pieces.append(_burst(burst_len, 300.0, rng))
    return np.concatenate(pieces)


# --- measure_noise_floor ---------------------------------------------------


def test_noise_floor_tracks_the_quiet_bed_not_the_loud_content() -> None:
    rng = np.random.default_rng(0)
    n = int(6.0 * _SR)
    noise_bed = 0.001 * rng.standard_normal(n)  # ~ -60 dBFS bed
    x = noise_bed.copy()
    loud_slice = slice(int(2.0 * _SR), int(3.0 * _SR))
    x[loud_slice] += 0.5 * np.sin(2 * np.pi * 400 * np.arange(loud_slice.stop - loud_slice.start) / _SR)

    floor = detect.measure_noise_floor(x, _SR)
    assert -66.0 < floor < -54.0  # near the ~-60 dBFS bed
    assert floor < -20.0  # nowhere near the loud content


def test_noise_floor_is_not_fooled_by_one_freak_silent_sample() -> None:
    rng = np.random.default_rng(1)
    n = int(3.0 * _SR)
    x = 0.001 * rng.standard_normal(n)
    x[12345] = 0.0  # one deliberately fake "digital silence" sample
    floor = detect.measure_noise_floor(x, _SR)
    assert -66.0 < floor < -54.0  # unaffected by the single zeroed sample


# --- detect_silence ---------------------------------------------------------


def test_silence_regions_match_planted_gaps_within_tolerance() -> None:
    rng = np.random.default_rng(2)
    burst_lengths = [1.0, 1.0, 1.0, 1.0]
    gap_lengths = [0.5, 0.6, 0.45]
    x = _build_bursts_and_gaps(burst_lengths, gap_lengths, rng)

    regions = detect.detect_silence(x, _SR, threshold_above_floor_db=6.0, min_len_ms=300.0)

    expected_starts = [1.0, 1.0 + 0.5 + 1.0, 1.0 + 0.5 + 1.0 + 0.6 + 1.0]
    expected_ends = [s + g for s, g in zip(expected_starts, gap_lengths, strict=True)]

    assert len(regions) == len(gap_lengths)
    tolerance_s = 0.02  # 20 ms, generous for a 10ms-frame detector
    for region, exp_start, exp_end in zip(regions, expected_starts, expected_ends, strict=True):
        assert abs(region["start_s"] - exp_start) <= tolerance_s, (region, exp_start)
        assert abs(region["end_s"] - exp_end) <= tolerance_s, (region, exp_end)
        assert region["peak_dbfs"] <= 0.0
        assert region["rms_dbfs"] <= region["peak_dbfs"]


def test_short_gap_below_min_len_is_not_reported() -> None:
    """Reuses the same well-conditioned recording as the tolerance test above
    (healthy ~1/3 of its duration is real silence, so the noise-floor
    percentile estimate is measuring actual background rather than being
    skewed by a lone short gap dominated by loud content) and raises
    min_len_ms above every planted gap's length: none should be reported.
    """
    rng = np.random.default_rng(2)
    x = _build_bursts_and_gaps([1.0, 1.0, 1.0, 1.0], [0.5, 0.6, 0.45], rng)
    regions = detect.detect_silence(x, _SR, threshold_above_floor_db=6.0, min_len_ms=700.0)
    assert regions == []


def test_silence_regions_are_scale_invariant() -> None:
    """The whole point of a floor-relative threshold: scaling everything by a
    fixed dB amount must not change which spans are reported."""
    rng = np.random.default_rng(4)
    x = _build_bursts_and_gaps([1.0, 1.0, 1.0], [0.5, 0.4], rng)
    quiet = x * (10 ** (-20.0 / 20.0))  # -20 dB

    loud_regions = detect.detect_silence(x, _SR, threshold_above_floor_db=6.0, min_len_ms=300.0)
    quiet_regions = detect.detect_silence(quiet, _SR, threshold_above_floor_db=6.0, min_len_ms=300.0)

    assert len(loud_regions) == len(quiet_regions) == 2
    for loud_r, quiet_r in zip(loud_regions, quiet_regions, strict=True):
        assert loud_r["start_s"] == quiet_r["start_s"]
        assert loud_r["end_s"] == quiet_r["end_s"]


def test_silence_detection_runs_on_mono_sum_of_stereo() -> None:
    rng = np.random.default_rng(5)
    mono = _build_bursts_and_gaps([1.0, 1.0], [0.5], rng)
    stereo = np.stack([mono, mono], axis=1)
    mono_regions = detect.detect_silence(mono, _SR, min_len_ms=300.0)
    stereo_regions = detect.detect_silence(stereo, _SR, min_len_ms=300.0)
    assert len(mono_regions) == len(stereo_regions) == 1


# --- detect_transients -------------------------------------------------------


def _plant_onsets(duration_s: float, onset_times: list[float], rng: np.random.Generator) -> np.ndarray:
    n = int(duration_s * _SR)
    x = 0.01 * rng.standard_normal(n)  # steady-state background
    for onset in onset_times:
        idx = int(onset * _SR)
        length = int(0.25 * _SR)
        env = np.exp(-np.arange(length) / (0.03 * _SR))
        click = np.sin(2 * np.pi * 800 * np.arange(length) / _SR) * env
        end = min(n, idx + length)
        x[idx:end] += click[: end - idx]
    return x


def test_transients_found_within_tolerance_with_no_spurious_extras() -> None:
    rng = np.random.default_rng(6)
    onset_times = [0.5, 1.2, 2.0, 3.3, 4.1]
    x = _plant_onsets(5.0, onset_times, rng)

    regions = detect.detect_transients(x, _SR, sensitivity=1.0, min_separation_ms=50.0)

    assert len(regions) == len(onset_times)
    tolerance_s = 0.01  # 10 ms
    for region, expected in zip(regions, onset_times, strict=True):
        assert region["start_s"] == region["end_s"]
        assert abs(region["start_s"] - expected) <= tolerance_s, (region, expected)
        assert 0.0 < region["strength"] <= 1.0


def test_min_separation_suppresses_a_double_trigger() -> None:
    rng = np.random.default_rng(7)
    n = int(2.0 * _SR)
    x = 0.01 * rng.standard_normal(n)
    for onset in (0.5, 0.52):  # 20 ms apart
        idx = int(onset * _SR)
        length = int(0.15 * _SR)
        env = np.exp(-np.arange(length) / (0.02 * _SR))
        click = np.sin(2 * np.pi * 900 * np.arange(length) / _SR) * env
        end = min(n, idx + length)
        x[idx:end] += click[: end - idx]

    merged = detect.detect_transients(x, _SR, sensitivity=1.0, min_separation_ms=50.0)
    separate = detect.detect_transients(x, _SR, sensitivity=1.0, min_separation_ms=5.0)

    assert len(merged) == 1
    assert len(separate) == 2


def test_transients_do_not_swamp_a_realistic_steady_state_background() -> None:
    """No spurious extras in the steady-state parts of a real signal.

    This is the property the task actually calls for: between the planted
    onsets in `test_transients_found_within_tolerance_with_no_spurious_extras`
    above (background = quiet noise, not silence), nothing extra is
    reported. That test already proves it directly.

    A separate, harder case -- a continuous, LOUD, perfectly sustained pure
    tone with no noise floor at all -- is a known, disclosed limitation of a
    lightweight spectral-flux detector: STFT analysis of a tone not aligned
    to a bin center leaks energy between bins in a way that drifts slightly
    frame-to-frame, and that drift alone can clear an adaptive threshold.
    contracts/regions.v1.md's "Not promised" section says exactly this --
    the detection algorithm is not fixed or guaranteed to be immune to every
    input -- so this test asserts the realistic bound (no explosion of
    false positives) rather than an unreachable zero, and documents why.
    """
    rng = np.random.default_rng(8)
    n = int(3.0 * _SR)
    t = np.arange(n) / _SR
    x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.005 * rng.standard_normal(n)
    regions = detect.detect_transients(x, _SR, sensitivity=1.0, min_separation_ms=50.0)
    # A handful of leakage-driven false positives on an all-tone input is
    # the documented limitation; dozens-per-second would not be.
    max_acceptable = 3
    assert len(regions) <= max_acceptable, (
        f"expected at most {max_acceptable} spurious onsets on a pure sustained tone, got {len(regions)}"
    )


# --- dsp.speech: absent-extra path ------------------------------------------


@pytest.mark.skipif(dsp_speech.is_available(), reason="requires the 'speech' extra to be ABSENT on this host")
def test_detect_fillers_without_extra_raises_speech_extra_missing() -> None:
    x = np.zeros(_SR)
    with pytest.raises(AudError) as exc_info:
        dsp_speech.detect_fillers(x, _SR)
    assert exc_info.value.code == "speech_extra_missing"
    assert "speech" in exc_info.value.message.lower()
    assert "uv tool install" in exc_info.value.remedy
    assert "aud[speech]" in exc_info.value.remedy


def test_filler_words_includes_um_and_erm() -> None:
    """Regression guard for D3: 'um' -- arguably the most common English
    filler, and one the manifest advertises by name -- and 'erm' must be in
    the one, real, built-in vocabulary.
    """
    assert "um" in dsp_speech.FILLER_WORDS
    assert "erm" in dsp_speech.FILLER_WORDS


@pytest.mark.skipif(dsp_speech.is_available(), reason="requires the 'speech' extra to be ABSENT on this host")
def test_is_available_reports_false_when_openai_whisper_is_not_installed() -> None:
    # This assertion documents the environment this lane developed and tested
    # in: openai-whisper is deliberately not installed here (installs are
    # DTU-only / the repo's own `.venv` with the `speech` extra). If this ever
    # flips to True it means the extra got installed on this box, which is a
    # different concern, not a regression.
    assert dsp_speech.is_available() is False


# --- dsp.speech: word -> region parsing, against REAL recorded words -------
#
# `replay.load_words(name)` returns the flat, ordered, real word list from
# one recorded openai-whisper run (tests/fixtures/recorded/openai_whisper/),
# each adapted through `aud.dsp.speech._WordAdapter` to expose exactly
# `.start`, `.end`, `.word`, `.probability` -- the real values openai-whisper
# produced, verbatim (openai-whisper itself returns each word as a plain
# dict; `_WordAdapter` is the one place that gets bridged, in production
# code as well as here). No hand-authored `_FakeWord` stands in for any of
# these any more: a hand-rolled fake built from the fields this module reads
# could not produce a real degenerate word (D2) or omit a field it didn't
# know to include, which is exactly how both defects survived a green suite
# (see RECORDING.md).


def test_words_to_regions_finds_filler_words_in_a_real_transcript() -> None:
    """speech_short_16000__raw.json: real transcript of "So, um, the first
    thing we need to do is, uh, check the levels. And then, erm, we can
    start recording." -- three real, correctly-timed filler words.
    """
    words = replay.load_words("speech_short_16000__raw")
    regions, degenerate = dsp_speech._words_to_regions(words, dsp_speech.FILLER_WORDS, min_pause_ms=100000.0)
    fillers = [r for r in regions if r["text"]]
    assert [(r["start_s"], r["end_s"], r["text"]) for r in fillers] == [
        (0.56, 0.7, "um"),
        (3.1, 3.22, "uh"),
        (6.5, 6.7, "erm"),
    ]
    assert degenerate == 0


def test_words_to_regions_finds_a_real_hesitation_gap() -> None:
    """Same real transcript: "... check the levels." ends at 4.24s, "And
    then ..." starts at 5.30s -- a real 1.06s pause, above the 700ms
    default threshold.
    """
    words = replay.load_words("speech_short_16000__raw")
    regions, _degenerate = dsp_speech._words_to_regions(words, dsp_speech.FILLER_WORDS, min_pause_ms=700.0)
    hesitations = [r for r in regions if not r["text"]]
    assert len(hesitations) == 1
    assert hesitations[0]["start_s"] == pytest.approx(4.24)
    assert hesitations[0]["end_s"] == pytest.approx(5.300000000000001)
    assert hesitations[0]["confidence"] == 1.0


def test_words_to_regions_ignores_a_real_gap_shorter_than_min_pause() -> None:
    """Same real transcript: "is," ends at 2.68s, "uh," starts at 3.10s --
    a real 0.42s gap, below the 700ms default threshold.
    """
    words = replay.load_words("speech_short_16000__raw")
    regions, _degenerate = dsp_speech._words_to_regions(words, dsp_speech.FILLER_WORDS, min_pause_ms=700.0)
    hesitation_starts = [r["start_s"] for r in regions if not r["text"]]
    assert 2.68 not in hesitation_starts


def test_words_to_regions_output_is_ascending_and_non_overlapping_on_a_real_transcript() -> None:
    words = replay.load_words("speech_short_16000__raw")
    regions, _degenerate = dsp_speech._words_to_regions(words, dsp_speech.FILLER_WORDS, min_pause_ms=300.0)
    starts = [r["start_s"] for r in regions]
    ends = [r["end_s"] for r in regions]
    assert len(regions) > 1  # otherwise the invariant below is checking nothing
    assert starts == sorted(starts)
    for i in range(len(regions) - 1):
        assert ends[i] <= starts[i + 1]


def test_words_to_regions_normalizes_a_real_capitalized_word_and_respects_custom_vocabulary() -> None:
    """speech_short_16000__raw.json really contains a capitalized ' So,' at
    0.0-0.22s -- openai-whisper's own capitalization/punctuation choice
    (sentence-initial), not a synthetic one. It normalizes to "so", which is
    not in the built-in FILLER_WORDS, so it is found only once a caller
    supplies "so" as a custom vocabulary word -- proving normalization and
    custom-vocabulary support together, on real output.
    """
    words = replay.load_words("speech_short_16000__raw")
    so = next(w for w in words if w.word == " So,")
    assert so.start == 0.0
    assert so.end == 0.22

    default_regions, _ = dsp_speech._words_to_regions([so], dsp_speech.FILLER_WORDS, min_pause_ms=100000.0)
    assert default_regions == []

    custom_regions, _ = dsp_speech._words_to_regions([so], ("so",), min_pause_ms=100000.0)
    assert len(custom_regions) == 1
    assert custom_regions[0]["text"] == "so"


def test_words_to_regions_drops_a_synthetic_degenerate_filler_word_and_counts_it() -> None:
    """No REAL recorded openai-whisper run re-recorded for issue #44 happened
    to produce a degenerate (`start == end`) word that is ALSO a recognised
    filler -- the re-recorded short-fixture matrix's five real degenerate
    words (speech_short_44100__raw: 4, speech_short_48000__raw: 1 more) are
    all non-filler gibberish ("man,", "emotions.", "what", "love.") produced
    by feeding whisper badly-mistimed audio, not filler words (see
    RECORDING.md's old-vs-new comparison). Re-recording a long enough real
    file to reproduce the EXACT real-degenerate-filler-word case the former
    faster-whisper fixtures had (' um,' at 24.0s) was out of this round's CPU
    budget -- disclosed here rather than silently dropped. This test instead
    combines REAL surviving filler words (speech_short_16000__raw: "um",
    "uh", "erm") with ONE deliberately-synthetic degenerate filler word,
    clearly labelled as such, to prove the exact behaviour the real case
    above used to prove: building a filler region for a zero-duration word
    would fail new_regions's `end_s > start_s` check and, because that check
    is whole-document, take every other correctly-timed word down with it
    (D2, lane report) -- so it must be dropped, and counted, while its
    neighbours survive.
    """

    class _SyntheticDegenerateFillerWord:
        start = 100.0
        end = 100.0
        word = "um,"
        probability = 1.0

    real_words = replay.load_words("speech_short_16000__raw")  # 3 real fillers: um, uh, erm
    combined = [*real_words, _SyntheticDegenerateFillerWord()]
    regions, degenerate = dsp_speech._words_to_regions(combined, dsp_speech.FILLER_WORDS, min_pause_ms=100000.0)

    assert degenerate == 1
    filler_regions = [r for r in regions if r["text"]]
    assert len(filler_regions) == 3, "the 3 real fillers must survive the synthetic degenerate word being dropped"
    assert not any(r["start_s"] == 100.0 for r in filler_regions), "the degenerate word must not appear"
    assert {r["text"] for r in filler_regions} == {"um", "uh", "erm"}


def test_words_to_regions_counts_real_non_filler_degenerate_words_as_zero() -> None:
    """`_words_to_regions` only counts a degenerate word if it is ALSO a
    recognised filler -- speech_short_44100__raw.json really produced four
    degenerate words (re-recorded for issue #44, see RECORDING.md), but
    none of them ("man,", "emotions.", "what", "love.") are fillers, so
    they contribute 0 to `degenerate_words_dropped` -- proved directly
    below against the real recorded data, not a synthetic stand-in.
    """
    non_filler_degenerate_recording = replay.load_recording("openai_whisper", "speech_short_44100__raw")
    non_filler_words = replay.load_words("speech_short_44100__raw")
    _regions, non_filler_degenerate = dsp_speech._words_to_regions(
        non_filler_words, dsp_speech.FILLER_WORDS, min_pause_ms=100000.0
    )
    assert non_filler_degenerate == 0, "degenerate non-filler words must not be counted at all"
    assert non_filler_degenerate_recording["summary"]["degenerate_word_count"] == 4  # they are real, just not fillers


def test_words_to_regions_drops_a_negative_duration_word_too() -> None:
    """`end < start` is equally degenerate and must be dropped the same way
    (the code checks `<=`, not `==`). No real recording ever produced
    `end < start` (only `end == start` -- see RECORDING.md), so this one
    boundary condition is a deliberately-constructed, clearly-disclosed
    synthetic input: it stands in for nothing faster-whisper actually
    does, it only exercises the `<=` comparison itself.
    """

    class _NegativeDurationWord:
        start = 0.5
        end = 0.4
        word = "um"
        probability = 1.0

    regions, degenerate = dsp_speech._words_to_regions(
        [_NegativeDurationWord()], dsp_speech.FILLER_WORDS, min_pause_ms=700.0
    )
    assert regions == []
    assert degenerate == 1


# --- dsp.speech: D2 -- a degenerate word must not sink the whole document -


def test_detect_fillers_full_document_survives_real_degenerate_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end through `detect_fillers`, replaying
    speech_short_48000__raw -- a real recording (re-recorded for issue #44)
    that genuinely contains five degenerate (`start == end`) words (none of
    them fillers -- see
    test_words_to_regions_counts_real_non_filler_degenerate_words_as_zero).
    `source=` is bound to the REAL shipped wav this recording came from
    (speech_short_48000.wav), so this is a fully-bound replay, not an
    UNBOUND placeholder.
    """
    wav_file = replay.wav_path("speech_short_48000.wav")
    replay.install_openai_whisper_replay(monkeypatch, "speech_short_48000__raw", source=wav_file)

    samples, sample_rate = dsp_io.read_audio(wav_file)
    regions, detection = dsp_speech.detect_fillers(samples, sample_rate)

    assert detection["degenerate_words_dropped"] == 0, "none of this recording's 5 degenerate words are fillers"
    assert regions is not None, "the document must not come back None just because some words were degenerate"


# --- dsp.speech: D1 -- sample rate was accepted and then ignored -----------


def test_detect_fillers_is_sample_rate_independent_on_two_real_recordings() -> None:
    """The literal sample-rate-independence proof: replay
    speech_short_16000__resampled_16k against the real speech_short_16000
    wav, and speech_short_48000__resampled_16k against the real
    speech_short_48000 wav -- two different native sample rates of the
    exact same spoken content (RECORDING.md's "_index.json": both
    resampled_16k paths report 2 segments / 21 words / last word end 7.72s,
    identically). Each replay is bound to its own recorded source's exact
    bytes -- `install_openai_whisper_replay`'s required `source=` parameter
    verifies the sha256 itself now -- this is deliberately NOT testing
    "similar audio produces the recorded output" (the one trap RECORDING.md
    warns against), it is testing that two *different*, each individually
    real and verified, (audio, recording) pairs agree once resampled to
    16 kHz.
    """
    results: dict[int, list[dict]] = {}
    for sr, recording_name, wav_name in (
        (16000, "speech_short_16000__resampled_16k", "speech_short_16000.wav"),
        (48000, "speech_short_48000__resampled_16k", "speech_short_48000.wav"),
    ):
        wav_file = replay.wav_path(wav_name)
        samples, sample_rate = dsp_io.read_audio(wav_file)
        assert sample_rate == sr

        with pytest.MonkeyPatch.context() as monkeypatch:
            replay.install_openai_whisper_replay(monkeypatch, recording_name, source=wav_file)
            regions, _detection = dsp_speech.detect_fillers(samples, sample_rate)
        results[sr] = regions

    # start_s/end_s/text agree exactly (whisper's own timing/text was
    # identical at both native rates once resampled to 16kHz -- see
    # RECORDING.md's _index.json). `confidence` (== whisper's per-word
    # `probability`) is NOT claimed identical: it is a real measurement,
    # and differs in the last few decimal digits because the two runs
    # resampled from different native rates, which is a real, if tiny,
    # difference in the bits whisper actually saw -- not a bug to hide
    # behind exact equality.
    assert len(results[16000]) == len(results[48000]) > 0
    for region_16k, region_48k in zip(results[16000], results[48000], strict=True):
        assert region_16k["start_s"] == region_48k["start_s"]
        assert region_16k["end_s"] == region_48k["end_s"]
        assert region_16k["text"] == region_48k["text"]
        assert region_16k["confidence"] == pytest.approx(region_48k["confidence"], abs=1e-2)


def test_resample_to_whisper_rate_is_a_no_op_at_16k() -> None:
    x = np.linspace(-1.0, 1.0, 1000)
    assert dsp_speech._resample_to_whisper_rate(x, 16000) is x


def test_resample_to_whisper_rate_preserves_real_duration() -> None:
    """Resampling to 16kHz must preserve the signal's real-world duration:
    n_out / 16000 ~= n_in / sr, for several representative native rates.
    """
    duration_s = 2.0
    for sr in (8000, 22050, 44100, 48000):
        n_in = int(duration_s * sr)
        x = np.zeros(n_in)
        resampled = dsp_speech._resample_to_whisper_rate(x, sr)
        implied_duration_s = len(resampled) / 16000.0
        assert abs(implied_duration_s - duration_s) <= 0.02, (sr, implied_duration_s)


# --- dsp.speech: the array path never spawns an ffmpeg subprocess ----------
#
# Only runs with the real `speech` extra installed (`pytest.importorskip`) --
# this is the one test in this file that calls the REAL openai-whisper, not
# a replay, because what it proves (that `whisper.audio.load_audio` -- the
# ONE function in openai-whisper that shells out to the `ffmpeg` CLI -- is
# never reached) is a property of the real call graph, not of recorded data.


def test_detect_fillers_never_spawns_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """`aud` always hands `whisper.transcribe` a numpy ndarray, never a file
    path string. `whisper.audio.log_mel_spectrogram` only calls
    `whisper.audio.load_audio` (which shells out to the `ffmpeg` CLI via
    `subprocess.run`) on the `isinstance(audio, str)` branch -- confirmed by
    reading that function's real source while resolving issue #44. This
    test does not merely read the source; it makes the real call with
    `load_audio` monkeypatched to raise, and asserts a real transcription
    still succeeds -- if `detect_fillers` ever regressed to passing a file
    path instead of an array, this would fail loudly instead of silently
    depending on an installed `ffmpeg` binary.
    """
    pytest.importorskip("whisper", reason="requires the 'speech' extra (openai-whisper) to be installed")
    import whisper.audio as whisper_audio

    def _raise_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("whisper.audio.load_audio must never be called -- aud always passes an ndarray")

    monkeypatch.setattr(whisper_audio, "load_audio", _raise_if_called)

    wav_file = replay.wav_path("speech_short_16000.wav")
    samples, sample_rate = dsp_io.read_audio(wav_file)
    regions, detection = dsp_speech.detect_fillers(samples, sample_rate)

    assert detection["engine"] == "openai-whisper"
    assert any(r["text"] for r in regions), "a real transcription of this fixture must find at least one filler"
