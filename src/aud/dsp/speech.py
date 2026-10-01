"""The openai-whisper adapter behind the optional `speech` extra.

This is the one place `detect fillers` depends on anything beyond the core
DSP stack, and it is a genuinely optional dependency: `whisper` (the
`openai-whisper` package) is imported lazily, inside `detect_fillers`, never
at module import time -- the same rule the AI-provider backends follow
(AGENTS.md #3) and for the same reason: a top-level import would make
importing `aud.dsp.speech` itself depend on the extra being installed, which
defeats the entire point of it being optional.

Previously this module wrapped `faster-whisper` (MIT). It was REPLACED (see
issue #44): `faster-whisper` hard-imports `av` (PyAV), whose PyPI wheel
bundles an FFmpeg build with `libx264`/`libx265` (GPL-2.0-or-later,
genuinely dynamically linked) and a `libmp3lame` with the GPL-only `mpglib`
decoder compiled in. `openai-whisper` is MIT (code AND model weights) and has
no such dependency; its own cost is `torch`, pinned in `pyproject.toml` to
PyPI's CPU-only wheel index for this project's own install.

A deliberate, stated exception to the `dsp/` boundary in AGENTS.md #8 ("dsp/
modules ... do not raise user-facing errors"): this module raises
`AudError(code="speech_extra_missing")` directly rather than raising a
plain exception for `aud.lib` to translate, the way `aud.dsp.engine`'s
`NotImplementedStageError` does. The reason is that `speech_extra_missing`
is a single, fully-specified failure contracts/regions.v1.md names by
code, message shape and remedy (see
contracts/regions.v1.md#producing-a-regions-document) -- there is exactly
one place in the whole tool this can happen, and duplicating that
knowledge at a second layer (a `lib.py` translation site) would only be a
second place for the two to drift apart. `aud.lib.detect_fillers` does not
catch or re-wrap this error; it is already the exact `AudError` the
contract promises.

`FILLER_WORDS` is the built-in vocabulary `detect fillers` searches for
when a caller does not supply `--words` of their own -- named here, in one
place, rather than scattered as string literals through the CLI and this
module.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Protocol

import numpy as np
from scipy import signal

from aud.dsp.channels import fold_to_mono
from aud.schemas import AudError

__all__ = ["FILLER_WORDS", "detect_fillers", "is_available"]

# The default filler vocabulary. Lowercase; matching is case-insensitive
# (see `_normalize_word`). A caller may replace this entirely via `words=`.
# This is the ONLY place this vocabulary is defined -- `cli.py`'s `--words`
# defaults to `None` precisely so this list, not a second copy of it, is
# what actually runs when a caller does not pass their own (see D3 in the
# lane report: a second hard-coded copy in the CLI is what let "um" --
# arguably the single most common English filler -- go undetectable by
# default, because the CLI's copy always won and never contained it).
FILLER_WORDS: tuple[str, ...] = ("um", "umm", "uh", "erm", "ehm", "ah", "er")

_MODEL_SIZE_DEFAULT = "base"

# whisper's ndarray input path has no sample-rate parameter: it always
# assumes the array it is handed is already 16 kHz mono PCM (see
# `whisper.audio.log_mel_spectrogram`'s own docstring: "containing the audio
# waveform in 16 kHz"). Handing it audio at any other rate does not fail --
# it silently mis-times every word, exactly the same failure mode measured
# against faster-whisper before it (D1 in the lane report: ~1.37x timestamp
# drift on a 22.05 kHz file, scaling to ~3x at 48 kHz) and re-confirmed
# against openai-whisper directly while resolving issue #44 (see
# tests/fixtures/recorded/RECORDING.md's "Versions recorded" table).
_WHISPER_SR = 16000


def _resample_to_whisper_rate(mono: np.ndarray, sr: int) -> np.ndarray:
    """Resample a mono float array to the 16 kHz rate whisper assumes.

    Pure and whisper-free -- exercised directly in tests without the real
    dependency installed. Uses the same `scipy.signal.resample_poly`
    polyphase approach the rest of the DSP stack uses for rate conversion
    (see `aud.dsp.reverb`/`aud.dsp.timepitch`), with the ratio reduced to
    small integers via `Fraction` so `resample_poly` does not choke on a
    huge up/down pair for an oddball rate.

    A no-op when `sr` is already 16000 -- returns `mono` unchanged (not a
    copy), matching `resample_poly`'s behaviour of being expensive to call
    for nothing.
    """
    if sr == _WHISPER_SR:
        return mono
    frac = Fraction(_WHISPER_SR, sr).limit_denominator(2000)
    return signal.resample_poly(mono, up=frac.numerator, down=frac.denominator)


# The exact command contracts/regions.v1.md and this module's error remedy
# point a caller at -- kept in sync with the `speech` extra declared in
# pyproject.toml's [project.optional-dependencies]. The CPU-only PyTorch
# index is REQUIRED here: `[tool.uv.sources]` in pyproject.toml only applies
# to this project's own `uv sync`/`uv run`, never to installing `aud` as a
# dependency from git -- measured directly while resolving issue #44 (see
# that PR's body). Without the explicit index, this exact command pulls the
# default (possibly CUDA, ~8 GB) torch build instead of the ~187 MB CPU one.
_INSTALL_COMMAND = (
    "uv tool install 'aud[speech] @ git+https://github.com/colombod/amplifier-smart-tools-audio' "
    "--index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
)


def is_available() -> bool:
    """Whether the `speech` extra (openai-whisper) is importable on this host."""
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _missing_extra_error() -> AudError:
    return AudError(
        code="speech_extra_missing",
        message="'detect fillers' needs word-level speech timings, and the 'speech' extra is not installed.",
        remedy=(
            f"Install aud with the speech extra: {_INSTALL_COMMAND} -- see docs/CONFIGURATION.md. "
            "The other detect verbs need nothing extra."
        ),
    )


def _normalize_word(text: str) -> str:
    return text.strip().strip(" .,!?;:\"'-").lower()


class _TimedWord(Protocol):
    """Structural shape of a word-timing object this module's internals consume.

    Only the three attributes this module reads are declared, so tests can
    hand in a plain namedtuple/dataclass instead of a real whisper result --
    see tests/test_dsp_detect.py's fake-transcript tests, which exercise
    `_words_to_regions` without openai-whisper installed. openai-whisper
    itself returns each word as a plain `dict` (keys `word`/`start`/`end`/
    `probability`), not an attribute-access object -- see `_WordAdapter`,
    which bridges that shape to this one.
    """

    start: float
    end: float
    word: str


class _WordAdapter:
    """Attribute-access wrapper around one of openai-whisper's word DICTS.

    `whisper.transcribe(..., word_timestamps=True)` returns
    `result["segments"][i]["words"][j]` as a plain dict with keys `word`,
    `start`, `end`, `probability` -- confirmed directly against a real
    installation while resolving issue #44 (unlike faster-whisper's `Word`,
    which was already an attribute-access dataclass). `_words_to_regions`
    is written once, against `_TimedWord`'s dot-access shape, for both the
    real call path and the replay harness -- this thin adapter is the only
    place that bridges openai-whisper's dict shape to it, so swapping
    engines again later only touches this one spot.
    """

    __slots__ = ("end", "probability", "start", "word")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.start = raw.get("start")
        self.end = raw.get("end")
        self.word = raw.get("word", "") or ""
        self.probability = raw.get("probability", 1.0)


def _words_to_regions(
    words: list[Any],
    vocabulary: tuple[str, ...],
    min_pause_ms: float,
) -> tuple[list[dict[str, Any]], int]:
    """Turn a sequence of word timings into filler-word and hesitation regions.

    Pure and whisper-free: `words` need only expose `.start`, `.end` and
    `.word` (see `_TimedWord`), which is what makes this testable against a
    fake transcript with the real dependency absent.

    A single left-to-right pass, so the returned list is already ascending
    and non-overlapping: a hesitation region only ever spans the gap
    strictly between the end of one word and the start of the next, and a
    filler-word region only ever spans one recognised word's own timing.

    Returns:
        `(regions, degenerate_dropped)`. The underlying engine can emit a
        word with `start == end` (observed in production against
        faster-whisper -- see D2 in the lane report; not yet observed
        against openai-whisper on the fixtures re-recorded for issue #44,
        but the handling is kept because nothing guarantees it cannot
        happen on other input). A filler region needs `end_s > start_s`
        (contracts/regions.v1.md); building one for a zero-duration word
        would fail `new_regions` validation, and because that validation
        is whole-document, would take every other correctly-timed word in
        the file down with it. This function instead drops just that
        word's region -- never fabricating a width for it -- and counts
        how many it dropped, so the caller can report the count honestly
        (`detection.degenerate_words_dropped`) rather than hiding it.
    """
    min_pause_s = min_pause_ms / 1000.0
    vocabulary_set = frozenset(vocabulary)
    regions: list[dict[str, Any]] = []
    prev_end: float | None = None
    degenerate_dropped = 0

    for w in words:
        start_s = float(w.start)
        end_s = float(w.end)
        if prev_end is not None and (start_s - prev_end) >= min_pause_s:
            regions.append({"start_s": prev_end, "end_s": start_s, "text": "", "confidence": 1.0})

        text = _normalize_word(getattr(w, "word", "") or "")
        if text in vocabulary_set:
            if end_s <= start_s:
                degenerate_dropped += 1
            else:
                confidence = float(getattr(w, "probability", 1.0))
                confidence = min(max(confidence, 0.0), 1.0)
                regions.append({"start_s": start_s, "end_s": end_s, "text": text, "confidence": confidence})

        prev_end = end_s

    return regions, degenerate_dropped


def detect_fillers(
    x: np.ndarray,
    sr: int,
    words: list[str] | None = None,
    min_pause_ms: float = 700.0,
    model_size: str = _MODEL_SIZE_DEFAULT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Locate filler words and long hesitations using word-level speech timings.

    Runs openai-whisper locally (no AI provider, no credential, no network
    call once the model is cached) to get word-level timestamps, then finds
    every word in `vocabulary` plus every gap of at least `min_pause_ms`
    between recognised words.

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_channels), at
            `sr`'s rate -- any rate is accepted; see the resampling note
            below.
        sr: The true sample rate of `x`, in Hz.
        words: The filler vocabulary to search for. Defaults to
            `FILLER_WORDS`.
        min_pause_ms: Gaps between words at least this long are reported as
            hesitations (`text=""`, `confidence=1.0`).
        model_size: The whisper model identifier to run (e.g. `"base"`,
            `"small"`, `"tiny"`).

    Returns:
        `(regions, detection)`: `regions` is a list of dicts shaped for
        `aud.core.regions.new_regions(kind="filler", ...)`; `detection` is
        that call's `detection` argument (`words`, `min_pause_ms`,
        `engine`, `model`, `degenerate_words_dropped`).

    Raises:
        AudError: code `speech_extra_missing` if openai-whisper is not
            installed. Never falls back to an energy-only guess -- see
            contracts/regions.v1.md#producing-a-regions-document.

    Resampling: whisper's ndarray input path has no sample-rate parameter of
    its own -- it always assumes 16 kHz mono PCM, so `x` is resampled to
    that rate (`_resample_to_whisper_rate`) before it is handed to the
    model. Without this, every timestamp whisper returns is wrong by the
    ratio `sr / 16000`, and those positions feed straight into `cut` -- see
    D1 in the lane report.

    No ffmpeg, ever: `x` is always a `numpy.ndarray` by the time it reaches
    `model.transcribe`, never a file path string -- whisper's own
    `load_audio` (which shells out to the `ffmpeg` CLI) is reached ONLY on
    the string-path branch of `whisper.audio.log_mel_spectrogram`, which
    this call never takes. See
    `tests/test_dsp_detect.py::test_detect_fillers_never_spawns_ffmpeg` for
    the enforced proof (monkeypatches `whisper.audio.load_audio` to raise
    and asserts a real transcription still succeeds).

    `fp16=False` is passed explicitly: on a CPU-only install (this
    project's default -- see pyproject.toml's `speech` extra), openai-whisper
    otherwise prints `UserWarning: FP16 is not supported on CPU; using FP32
    instead` on every call. Passing `fp16=False` up front gets the same FP32
    behaviour with no warning.
    """
    try:
        import whisper
    except ImportError as exc:
        raise _missing_extra_error() from exc

    vocabulary = tuple(_normalize_word(w) for w in (words or FILLER_WORDS) if w and w.strip())

    # Analysis-only mono fold -- see aud.dsp.channels.fold_to_mono's docstring.
    mono = fold_to_mono(np.asarray(x, dtype=np.float64))
    # whisper's ndarray path always assumes 16 kHz; resample to that rate
    # here so the timestamps it returns are already real seconds, no matter
    # what rate `sr` actually is (see `_resample_to_whisper_rate` and D1 in
    # the lane report).
    mono = _resample_to_whisper_rate(mono, sr)
    audio = mono.astype("float32")

    model = whisper.load_model(model_size)
    result = model.transcribe(audio, word_timestamps=True, fp16=False)

    all_words: list[Any] = []
    for segment in result.get("segments", []):
        segment_words = segment.get("words", None) or []
        all_words.extend(_WordAdapter(w) for w in segment_words)

    regions, degenerate_dropped = _words_to_regions(all_words, vocabulary, min_pause_ms)
    detection = {
        "words": list(vocabulary),
        "min_pause_ms": float(min_pause_ms),
        "engine": "openai-whisper",
        "model": model_size,
        # Optional (contracts/regions.v1.md: additive detection keys stay
        # compatible within regions_format 1) -- how many words this run
        # dropped because the engine reported them with start == end (see
        # `_words_to_regions`). Always emitted by this build so the caller
        # never has to guess whether zero means "none dropped" or "this
        # build doesn't report it".
        "degenerate_words_dropped": degenerate_dropped,
    }
    return regions, detection
