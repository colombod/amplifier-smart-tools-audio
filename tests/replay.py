"""One shared replay harness for every recorded third-party-library run.

Nothing in this module invents a value. A "replay" here always means: load
`tests/fixtures/recorded/<category>/<name>.json` (a captured real run --
see `tests/fixtures/recorded/RECORDING.md`) and hand its contents back out
through whatever surface the real library would have exposed, unmodified.

Every load function fails LOUDLY (`RecordingNotFoundError`) if the recording it
is asked for does not exist. A silent fallback to a made-up value is
exactly the defect this harness exists to remove -- see AGENTS.md /
RECORDING.md for the two real defects (a degenerate word with no sample
rate, and a single-text-block provider fake) that a hand-written fake let
through a green test suite.

This module never hand-authors the DATA a third party would have produced
(a Word's timing, a Signalsmith output length, a model's response text).
It is only "tidy" about the OBJECT SHAPE the real library wraps that data
in (attribute access, not dict access) -- and even there, RECORDING.md's
documented quirks (float32 `timeFactor` read-back, `inputLatency` etc.
being methods, not attributes) are reproduced deliberately, because a
replay that is tidier than the real library is a fake again.

`install_faster_whisper_replay`'s `source` parameter is REQUIRED, not
optional: a replay is bound to its exact recorded input bytes (a `Path`,
sha256-verified against the recording's provenance) or explicitly declared
`UNBOUND("reason")`. See that function's docstring -- this is the
structural form of the rule above "a replay asserting recorded output
against *similar* audio is testing nothing".
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

RECORDED_ROOT = Path(__file__).parent / "fixtures" / "recorded"


class RecordingNotFoundError(RuntimeError):
    """A replay was asked for a recording that was never captured.

    Never caught and papered over inside this module -- every install_*
    helper below lets this propagate so a missing/renamed recording fails
    a test loudly instead of silently substituting something invented.
    """


# ---------------------------------------------------------------------------
# Generic recording access
# ---------------------------------------------------------------------------


def load_recording(category: str, name: str) -> dict[str, Any]:
    """Load `fixtures/recorded/<category>/<name>.json` verbatim.

    Raises RecordingNotFoundError (not FileNotFoundError, so callers can catch
    one specific thing) if the file does not exist.
    """
    path = RECORDED_ROOT / category / f"{name}.json"
    if not path.exists():
        raise RecordingNotFoundError(
            f"no recording at {path} -- record it first (see tests/fixtures/recorded/RECORDING.md, "
            f"'How to re-record from scratch')"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def wav_path(filename: str) -> Path:
    """Path to one of the real source wavs under fixtures/recorded/wav/."""
    path = RECORDED_ROOT / "wav" / filename
    if not path.exists():
        raise RecordingNotFoundError(
            f"no recorded source wav at {path}. Note: the 22050/44100/48000 Hz 'long' variants were "
            f"transcribed but deliberately not shipped (RECORDING.md); only their sha256 is preserved."
        )
    return path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_matches_recorded_source(recording: dict[str, Any], wav_file: Path) -> None:
    """Fail loudly if `wav_file` is not the exact bytes this recording transcribed.

    The one rule this harness cannot relax (see RECORDING.md, "What these
    recordings prove ... and what they do not"): the same content
    resampled by a different path produced a materially different
    transcript (168 vs 106 words). A replay is only faithful when it is
    bound to its recorded source's *exact* bytes, not to "similar" audio.
    """
    expected = recording["provenance"]["source_wav_sha256"]
    actual = sha256_of(wav_file)
    if actual != expected:
        raise RecordingNotFoundError(
            f"{wav_file} does not match this recording's provenance.source_wav_sha256 "
            f"(expected {expected}, got {actual}) -- this replay would be bound to the wrong input bytes"
        )


class UnboundSource:
    """The explicit, greppable declaration that a replay install is NOT bound
    to any driving-audio source.

    Constructing this (via `UNBOUND(reason)`) is the visible admission that a
    test is replaying a recorded ANSWER regardless of what audio -- if any --
    actually drives the call. That is exactly the gap AGENTS.md SS3b and
    RECORDING.md warn about ("the same speech at 16 kHz and at
    48-kHz-resampled-to-16 kHz produced 168 vs 106 words"), so it must never
    be the silent default. `reason` is mandatory and non-empty precisely so
    `grep -n "UNBOUND(" tests/` finds every exemption together with why it
    was taken, rather than an omission looking identical to a bound replay.
    """

    def __init__(self, reason: str) -> None:
        if not reason or not reason.strip():
            raise ValueError("UNBOUND(reason) requires a non-empty reason -- an exemption with no reason is a gap")
        self.reason = reason

    def __repr__(self) -> str:
        return f"UnboundSource(reason={self.reason!r})"


def UNBOUND(reason: str) -> UnboundSource:  # noqa: N802 -- reads as a keyword at call sites, e.g. source=UNBOUND(...)
    """Declare, with a reason, that a replay install has no driving-audio source to bind to."""
    return UnboundSource(reason)


# ---------------------------------------------------------------------------
# faster-whisper replay
# ---------------------------------------------------------------------------


def _namespace_from_dict(d: dict[str, Any]) -> types.SimpleNamespace:
    """A plain attribute-access object carrying exactly the recorded fields.

    No more, no fewer -- `type_inventory` in every faster_whisper recording
    exists precisely so a replay does not need to guess this shape (see
    RECORDING.md: `Word` carries exactly `start`, `end`, `word`,
    `probability`, nothing else; `_fields` is empty because it is a
    dataclass, not a NamedTuple).
    """
    return types.SimpleNamespace(**{k: v for k, v in d.items() if k != "__type__"})


def _replay_word(d: dict[str, Any]) -> types.SimpleNamespace:
    return _namespace_from_dict(d)


def _replay_segment(d: dict[str, Any]) -> types.SimpleNamespace:
    clean = {k: v for k, v in d.items() if k != "__type__"}
    clean["words"] = [_replay_word(w) for w in clean.get("words", [])]
    return types.SimpleNamespace(**clean)


def _replay_info(d: dict[str, Any]) -> types.SimpleNamespace:
    return _namespace_from_dict(d)


def load_words(recording_name: str) -> list[types.SimpleNamespace]:
    """The flat, real, ordered word list from one faster_whisper recording.

    Each word exposes exactly `.start`, `.end`, `.word`, `.probability` --
    real values, real degenerate (`start == end`) words included, never
    filtered. For feeding straight into `aud.dsp.speech._words_to_regions`
    without going through a fake `WhisperModel` at all.
    """
    recording = load_recording("faster_whisper", recording_name)
    return [_replay_word(w) for w in recording["words_flat"]]


class FasterWhisperReplay:
    """Records which recording is being replayed and its captured version.

    Exposed so a test (or a future re-record check) can see what version
    of faster-whisper actually produced the data it is trusting -- see
    RECORDING.md's "Versions recorded" table and "What these recordings do
    not prove": a replay is only as current as its last re-record.
    """

    def __init__(self, recording: dict[str, Any], recording_name: str, source: Path | UnboundSource) -> None:
        self.recording = recording
        self.name = recording_name
        self.recorded_version = recording["provenance"]["faster_whisper_version"]
        self.source = source

    def segments(self) -> list[types.SimpleNamespace]:
        return [_replay_segment(s) for s in self.recording["segments"]]

    def info(self) -> types.SimpleNamespace:
        return _replay_info(self.recording["info"])


def install_faster_whisper_replay(
    monkeypatch: Any, recording_name: str, *, source: Path | UnboundSource
) -> FasterWhisperReplay:
    """Install a fake `faster_whisper` module that REPLAYS one recorded run.

    `WhisperModel(model_size).transcribe(audio, word_timestamps=True)`
    returns `(segments, info)` reproducing `recording_name`'s captured
    segments/words/info verbatim -- degenerate words, the hallucinated
    silence word, everything -- regardless of what `audio` it is actually
    handed. It does not (and structurally cannot) verify that the audio
    passed in matches what produced the recording: faster-whisper is never
    actually run.

    `source` is REQUIRED and is the structural fix for the trap this
    function used to allow: a replay asserting recorded output against
    *similar* (not identical) audio tests nothing (RECORDING.md: the same
    speech at 16 kHz vs 48-kHz-resampled-to-16-kHz produced 168 vs 106
    words). Pass either:

    - the `Path` to the exact wav that produced `recording_name` -- its
      sha256 is verified against `recording["provenance"]["source_wav_sha256"]`
      right here (via `assert_matches_recorded_source`), so a caller cannot
      silently drift onto the wrong bytes; or
    - `UNBOUND("reason")` -- an explicit, greppable declaration that this
      particular test drives the call with audio that is NOT the recorded
      source (e.g. the source wav was never shipped, or the test is
      deliberately proving something that does not depend on the driving
      audio), naming why in `reason`.

    Fails loudly via RecordingNotFoundError if `recording_name` was never
    captured, or if `source` is a `Path` that does not match the recording's
    provenance.
    """
    recording = load_recording("faster_whisper", recording_name)
    if isinstance(source, UnboundSource):
        pass  # explicit exemption already validated (non-empty reason) at construction
    else:
        assert_matches_recorded_source(recording, source)
    replay = FasterWhisperReplay(recording, recording_name, source)

    class _ReplayWhisperModel:
        def __init__(self, model_size: str) -> None:
            self.model_size = model_size

        def transcribe(self, audio: Any, word_timestamps: bool = True) -> tuple[list[Any], Any]:
            del audio, word_timestamps  # replayed output does not depend on the input
            return replay.segments(), replay.info()

    replay_module = types.ModuleType("faster_whisper")
    replay_module.WhisperModel = _ReplayWhisperModel  # type: ignore[attr-defined]
    replay_module.__replay_recording__ = recording_name  # type: ignore[attr-defined]
    replay_module.__replay_recorded_version__ = replay.recorded_version  # type: ignore[attr-defined]
    replay_module.__replay_source__ = source  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", replay_module)
    return replay


# ---------------------------------------------------------------------------
# python_stretch (Signalsmith Stretch) replay
# ---------------------------------------------------------------------------

_STRETCH_AUDIO_DIR = RECORDED_ROOT / "python_stretch" / "audio_pair"


def load_stretch_recording() -> dict[str, Any]:
    return load_recording("python_stretch", "python_stretch_recording")


def load_stretch_audio_pair(time_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """The exact real recorded (input, output) sample arrays for one `timeFactor`.

    `time_factor` must be one of the recorded audio-pair values (0.5, 1.0,
    2.0) -- these are the ones an actual Signalsmith Stretch call produced
    real sample data for, not lengths alone. Fails loudly if not found.
    """
    recording = load_stretch_recording()
    for entry in recording["audio_pair"]:
        if entry["time_factor_set"] == time_factor:
            input_arr = np.load(_STRETCH_AUDIO_DIR / entry["input_npy"])
            output_arr = np.load(_STRETCH_AUDIO_DIR / entry["output_npy"])
            return input_arr, output_arr
    raise RecordingNotFoundError(
        f"no recorded audio_pair entry for timeFactor={time_factor!r} in python_stretch_recording.json"
    )


def _not_an_attribute(name: str) -> Any:
    """Reproduce the real library's quirk: these four are METHODS, not attributes.

    Measured (python_stretch 0.3.1): reading `stretch.inputLatency` returns
    a bound method, not a float -- `float(stretch.inputLatency)` raises
    `TypeError("float() argument must be a string or a real number, not
    'nanobind.nb_bound_method'")`. Python's own `float()` builtin names the
    real *type* in that message, so a plain Python method here raises the
    same TypeError naming Python's own method type instead of nanobind's --
    the important, faithfully-reproduced fact is that it is a CALLABLE, not
    a float attribute, exactly like the real object. A replay that made
    these plain floats would be tidier than the real library, and this
    repo has already paid once for a replay tidier than reality.
    """

    def _method(self: Any) -> float:
        raise NotImplementedError(f"{name} is a method on the real object, not called by aud.dsp.timepitch")

    _method.__name__ = name
    return _method


class ReplaySignalsmithStretch:
    """Replays python_stretch.Signalsmith.Stretch()'s recorded behaviour.

    `timeFactor` is stored and read back through float32 (measured: 0.8 ->
    0.800000011920929 -- see RECORDING.md). `.process()` first tries an
    EXACT replay against the recorded `audio_pair` (byte-identical input
    required); failing that it falls back to the recorded timeFactor ->
    length RELATIONSHIP itself (`out_len = round(in_len / timeFactor)`,
    semitone-only transposition leaves length unchanged) -- both measured
    facts from the real recording, not invented ones, generalised only to
    whatever input length a test happens to hand in.

    Attribute/method names below deliberately mirror python_stretch's own
    mixedCase API exactly (`timeFactor`, `setTransposeSemitones`, ...) --
    this is a stand-in for that third-party surface, so faithfulness to
    its naming outranks this repo's own naming convention here.
    """

    inputLatency = _not_an_attribute("inputLatency")  # noqa: N815
    outputLatency = _not_an_attribute("outputLatency")  # noqa: N815
    blockSamples = _not_an_attribute("blockSamples")  # noqa: N815
    intervalSamples = _not_an_attribute("intervalSamples")  # noqa: N815

    def __init__(self, recording: dict[str, Any]) -> None:
        self._recording = recording
        self.sampleRate = 0.0
        self._time_factor = 1.0  # recorded default, before any assignment
        self._semitones = 0.0
        self._channels = 1
        self._input_cache: dict[int, np.ndarray] = {}
        self._pairs_by_factor = {entry["time_factor_set"]: entry for entry in recording["audio_pair"]}

    def preset(self, channels: int, sample_rate: int) -> None:
        self._channels = channels
        self.sampleRate = float(sample_rate)

    def configure(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def reset(self) -> None:
        pass

    @property
    def timeFactor(self) -> float:  # noqa: N802
        return self._time_factor

    @timeFactor.setter
    def timeFactor(self, value: float) -> None:  # noqa: N802
        # Measured float32 read-back -- see RECORDING.md.
        self._time_factor = float(np.float32(value))

    def setTimeFactor(self, value: float) -> None:  # noqa: N802
        self.timeFactor = value

    def setTransposeSemitones(self, semitones: float) -> None:  # noqa: N802
        self._semitones = float(semitones)

    def setTransposeFactor(self, factor: float) -> None:  # noqa: N802
        raise NotImplementedError("not exercised by aud.dsp.timepitch; not recorded")

    def _recorded_input(self) -> np.ndarray | None:
        input_path = _STRETCH_AUDIO_DIR / "input_mono_22050.npy"
        if not input_path.exists():
            return None
        cached = self._input_cache.get(id(input_path))
        if cached is None:
            cached = np.load(input_path)
            self._input_cache[id(input_path)] = cached
        return cached

    def process(self, audio: np.ndarray) -> np.ndarray:
        in_len = audio.shape[1]
        got = audio[0] if audio.ndim == 2 else audio

        pair = self._pairs_by_factor.get(round(self._time_factor, 6))
        if pair is not None:
            reference = self._recorded_input()
            if (
                reference is not None
                and reference.shape[0] == in_len
                and np.allclose(got.astype(np.float64), reference.astype(np.float64), atol=1e-6)
            ):
                out = np.load(_STRETCH_AUDIO_DIR / pair["output_npy"]).astype(np.float32)
                return out.reshape(1, -1) if out.ndim == 1 else out

        # Fall back to the recorded timeFactor -> length relationship
        # itself, applied to this input's own length -- see RECORDING.md:
        # "Output length is round(input / timeFactor) with no latency
        # padding at any factor tested" and "setTransposeSemitones ...
        # leaves length unchanged (out/in = 1.0000 in all five cases)".
        out_len = in_len if self._time_factor == 1.0 else max(1, round(in_len / self._time_factor))
        return np.zeros((audio.shape[0], out_len), dtype=np.float32)


def install_python_stretch_replay(monkeypatch: Any) -> dict[str, Any]:
    """Install a fake `python_stretch` module replaying the real recording.

    Also patches `aud.dsp.timepitch._signalsmith_available` to report the
    extra as present, matching `_install_fake_signalsmith`'s previous role
    -- this repo has no way to actually install the extra in this
    environment (DTU-only), so the availability check itself is the one
    piece of behaviour that must still be told, not replayed.
    """
    from aud.dsp import timepitch

    recording = load_stretch_recording()
    stretch_ns = types.SimpleNamespace(Stretch=lambda: ReplaySignalsmithStretch(recording))
    replay_module = types.SimpleNamespace(Signalsmith=stretch_ns)
    replay_module.__replay_recorded_version__ = recording["provenance"]["python_stretch_version"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "python_stretch", replay_module)
    monkeypatch.setattr(timepitch, "_signalsmith_available", lambda: True)
    return recording


# ---------------------------------------------------------------------------
# Anthropic replay -- HTTP layer (real AnthropicBackend, replayed transport)
# ---------------------------------------------------------------------------


class _ReplayHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _ReplayHTTPResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def install_anthropic_urlopen_replay(monkeypatch: Any, recording_name: str) -> dict[str, Any]:
    """Patch `urllib.request.urlopen` to replay one recorded Anthropic call.

    This exercises the REAL `AnthropicBackend.complete()` -- request
    construction, headers, and (crucially) response-content-block parsing
    -- against a REAL captured response body. Only the network transport
    is replaced, and only with what was actually received; the code that
    turned a `thinking`+`text` response into a KeyError (the exact defect
    this harness exists to catch) still runs for real.

    Asserts the request handed to it matches the recorded one in the ways
    that matter -- model name, and the presence of non-empty `system` and
    `messages` -- so a change to prompt construction that would have
    broken the real call is visible. It does not require exact text
    equality with the recorded request: a live caller's system/user text
    legitimately differs call to call.
    """
    recording = load_recording("anthropic", recording_name)
    recorded_request = recording["request"]

    def _fake_urlopen(request: Any, timeout: float | None = None) -> _ReplayHTTPResponse:
        del timeout
        body = json.loads(request.data.decode("utf-8"))
        if body.get("model") != recorded_request.get("model"):
            raise AssertionError(
                f"replay {recording_name!r}: request model {body.get('model')!r} does not match "
                f"the recorded request's model {recorded_request.get('model')!r} -- pass the recorded "
                f"model name to reuse this recording"
            )
        if not body.get("system"):
            raise AssertionError(f"replay {recording_name!r}: request has no non-empty 'system'")
        if not body.get("messages"):
            raise AssertionError(f"replay {recording_name!r}: request has no non-empty 'messages'")
        return _ReplayHTTPResponse(json.dumps(recording["response"]).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return recording


def recorded_text_block(recording: dict[str, Any]) -> str:
    """The real `text` content block from a recorded Anthropic response.

    This is what `AnthropicBackend.complete()` is supposed to return after
    correctly skipping any leading `thinking` block.
    """
    for block in recording["response"]["content"]:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            return block["text"]
    raise RecordingNotFoundError("recorded response has no 'text' content block")


class ReplayAdviceBackend:
    """An IntelligenceBackend that replays one recorded advise response's
    real extracted text -- never a hand-authored plan -- while recording
    the request it was actually called with, so a test can assert wiring
    (model, max_tokens, prompt construction) without inventing a response.
    """

    def __init__(self, recording_name: str) -> None:
        self._recording = load_recording("anthropic", recording_name)
        self.name = recording_name
        self.calls: list[dict[str, Any]] = []

    @property
    def recorded_model(self) -> str:
        return self._recording["request"]["model"]

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        self.calls.append({"system": system, "user": user, "model": model, "max_tokens": max_tokens})
        return recorded_text_block(self._recording)


# ---------------------------------------------------------------------------
# Recorded-library-version visibility (RECORDING.md's re-record contract)
# ---------------------------------------------------------------------------


def recorded_versions() -> dict[str, str]:
    """One place a caller (or a future CI check) can see every recorded
    library version at once, so a version drift is a visible diff here
    rather than a silent assumption inside a replay.
    """
    whisper_index = load_recording("faster_whisper", "_index")
    stretch = load_stretch_recording()
    anthropic_any = load_recording("anthropic", "advise-clean-haiku")
    return {
        "faster_whisper": whisper_index["faster_whisper_version"],
        "ctranslate2": whisper_index["ctranslate2_version"],
        "python_stretch": stretch["provenance"]["python_stretch_version"],
        "anthropic_api_version": anthropic_any["provenance"]["anthropic_version"],
    }
