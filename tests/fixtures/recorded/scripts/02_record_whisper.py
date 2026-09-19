#!/usr/bin/env python
"""RECORD real faster-whisper behaviour as replay fixtures.

Calls faster-whisper DIRECTLY -- not through `aud` -- in exactly the shape
`aud.dsp.speech.detect_fillers` calls it:

    model = WhisperModel(model_size)                       # model_size = "base"
    segments, info = model.transcribe(audio, word_timestamps=True)

Two input paths are recorded for every source file, because `aud` has one
of each and both matter to a replay:

  path="raw"            the float32 array at its OWN sample rate, handed
                        straight to transcribe() -- what the library does
                        when it is given non-16 kHz audio (it has no sample
                        rate parameter and assumes 16 kHz regardless).
  path="resampled_16k"  the same audio put through scipy.signal.resample_poly
                        to 16 kHz first, which is what aud.dsp.speech does
                        today (_resample_to_whisper_rate).

Everything the returned objects actually carry is serialised -- attributes
are enumerated with dir()/vars() rather than assumed, so fields our code
does not read are captured too.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import ctranslate2
import faster_whisper
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy import signal

MODEL_SIZE = "base"  # aud.dsp.speech._MODEL_SIZE_DEFAULT
WHISPER_SR = 16000  # aud.dsp.speech._WHISPER_SR

WAV_DIR = Path("/rec/wav")
OUT_DIR = Path("/rec/out/faster_whisper")

# The exact snippet that produces each recording, embedded in every file.
SNIPPET = """\
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel
x, sr = sf.read(WAV, dtype="float64", always_2d=False)
if x.ndim == 2:
    x = x.mean(axis=1)
# path == "resampled_16k" only:
#   from fractions import Fraction; from scipy import signal
#   frac = Fraction(16000, sr).limit_denominator(2000)
#   x = signal.resample_poly(x, up=frac.numerator, down=frac.denominator)
audio = x.astype("float32")
model = WhisperModel("base")
segments, info = model.transcribe(audio, word_timestamps=True)
segments = list(segments)   # generator: nothing runs until it is drained
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_PRIMITIVES = (str, int, float, bool, type(None))


def jsonable(value, depth: int = 0):
    """Convert an arbitrary library object to JSON, losing nothing knowable."""
    if isinstance(value, _PRIMITIVES):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return {"__nonfinite__": repr(value)}
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_len__": len(value)}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        if depth > 6:
            return {"__truncated_sequence__": len(value)}
        return [jsonable(v, depth + 1) for v in value]
    if depth > 6:
        return {"__truncated_object__": type(value).__name__}
    return obj_to_dict(value, depth)


def obj_to_dict(obj, depth: int = 0) -> dict:
    """Enumerate what an object REALLY carries: _asdict/vars/dir, merged."""
    out: dict = {"__type__": type(obj).__name__}
    collected: dict = {}
    asdict = getattr(obj, "_asdict", None)
    if callable(asdict):
        try:
            collected.update(asdict())
        except Exception as exc:  # pragma: no cover - diagnostic only
            out["__asdict_error__"] = repr(exc)
    try:
        collected.update(vars(obj))
    except TypeError:
        pass  # no __dict__ (slots / NamedTuple)
    for name in dir(obj):
        if name.startswith("_") or name in collected:
            continue
        try:
            attr = getattr(obj, name)
        except Exception as exc:  # pragma: no cover
            out.setdefault("__attr_errors__", {})[name] = repr(exc)
            continue
        if callable(attr):
            continue
        collected[name] = attr
    for key, val in collected.items():
        out[key] = jsonable(val, depth + 1)
    return out


def type_inventory(obj) -> dict:
    """Everything dir() reports, methods included -- the honest surface."""
    names = dir(obj)
    return {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "dir": names,
        "public_non_callable": sorted(
            n for n in names if not n.startswith("_") and not callable(getattr(obj, n, None))
        ),
        "has_dict": hasattr(obj, "__dict__"),
        "has_asdict": callable(getattr(obj, "_asdict", None)),
        "fields": list(getattr(obj, "_fields", []) or []),
    }


def resample_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    """Byte-for-byte the rule aud.dsp.speech._resample_to_whisper_rate uses."""
    if sr == WHISPER_SR:
        return mono
    frac = Fraction(WHISPER_SR, sr).limit_denominator(2000)
    return signal.resample_poly(mono, up=frac.numerator, down=frac.denominator)


def record(model: WhisperModel, wav: Path, path_mode: str) -> dict:
    raw, sr = sf.read(str(wav), dtype="float64", always_2d=False)
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    mono = raw if raw.ndim == 1 else raw.mean(axis=1)
    n_in = int(mono.shape[0])

    if path_mode == "resampled_16k":
        fed = resample_16k(mono, sr)
        fed_sr = WHISPER_SR
        resampled = sr != WHISPER_SR
    elif path_mode == "raw":
        fed = mono
        fed_sr = sr
        resampled = False
    else:
        raise ValueError(path_mode)

    audio = np.asarray(fed, dtype="float32")

    t0 = time.perf_counter()
    segments_gen, info = model.transcribe(audio, word_timestamps=True)
    segments = list(segments_gen)  # the generator is lazy; this is where work happens
    elapsed = time.perf_counter() - t0

    words = []
    for seg in segments:
        words.extend(getattr(seg, "words", None) or [])

    degenerate = [
        {"index": i, "word": getattr(w, "word", None), "start": getattr(w, "start", None), "end": getattr(w, "end", None)}
        for i, w in enumerate(words)
        if getattr(w, "start", None) is not None and w.start == w.end
    ]

    return {
        "provenance": {
            "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "faster_whisper_version": faster_whisper.__version__,
            "ctranslate2_version": ctranslate2.__version__,
            "numpy_version": np.__version__,
            "scipy_version": __import__("scipy").__version__,
            "soundfile_version": sf.__version__,
            "python_version": sys.version,
            "model": MODEL_SIZE,
            "model_constructor": f'WhisperModel("{MODEL_SIZE}")  # all other args left at library defaults',
            "transcribe_call": "model.transcribe(audio, word_timestamps=True)",
            "source_wav": wav.name,
            "source_wav_sha256": sha256(wav),
            "source_sample_rate": int(sr),
            "input_path": path_mode,
            "snippet": SNIPPET,
            "wall_seconds": round(elapsed, 3),
        },
        "input": {
            "path_mode": path_mode,
            "source_sample_rate": int(sr),
            "source_channels": int(channels),
            "source_samples": n_in,
            "source_duration_s": round(n_in / sr, 6),
            "fed_sample_rate_assumed_by_library": WHISPER_SR,
            "fed_samples": int(audio.shape[0]),
            "fed_dtype": str(audio.dtype),
            "fed_duration_s_if_16k": round(int(audio.shape[0]) / WHISPER_SR, 6),
            "true_duration_s": round(n_in / sr, 6),
            "resampled_before_transcribe": bool(resampled),
            "declared_sample_rate_of_fed_array": int(fed_sr),
        },
        "type_inventory": {
            "info": type_inventory(info),
            "segment": type_inventory(segments[0]) if segments else None,
            "word": type_inventory(words[0]) if words else None,
        },
        "summary": {
            "n_segments": len(segments),
            "n_words": len(words),
            "text": "".join(getattr(s, "text", "") for s in segments),
            "last_word_end": (getattr(words[-1], "end", None) if words else None),
            "degenerate_word_count": len(degenerate),
            "degenerate_words": degenerate,
        },
        "info": jsonable(info),
        "segments": [jsonable(s) for s in segments],
        "words_flat": [jsonable(w) for w in words],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or None

    jobs: list[tuple[str, str]] = []
    for sr in (16000, 22050, 44100, 48000):
        for length in ("short", "long"):
            for mode in ("raw", "resampled_16k"):
                jobs.append((f"speech_{length}_{sr}.wav", mode))
    jobs.append(("nospeech_silence_16000.wav", "raw"))
    jobs.append(("nospeech_tone_16000.wav", "raw"))

    print(f"loading WhisperModel({MODEL_SIZE!r}) ...", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(MODEL_SIZE)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    index = []
    for name, mode in jobs:
        if only and not any(tok in name for tok in only):
            continue
        wav = WAV_DIR / name
        if not wav.exists():
            print(f"SKIP missing {wav}", flush=True)
            continue
        out = OUT_DIR / f"{wav.stem}__{mode}.json"
        print(f"recording {name} [{mode}] ...", flush=True)
        rec = record(model, wav, mode)
        out.write_text(json.dumps(rec, indent=2, sort_keys=False))
        s = rec["summary"]
        print(
            f"  -> {out.name}: segments={s['n_segments']} words={s['n_words']} "
            f"degenerate={s['degenerate_word_count']} last_end={s['last_word_end']} "
            f"({rec['provenance']['wall_seconds']}s)",
            flush=True,
        )
        index.append(
            {
                "file": out.name,
                "source_wav": name,
                "path_mode": mode,
                "n_segments": s["n_segments"],
                "n_words": s["n_words"],
                "degenerate_word_count": s["degenerate_word_count"],
                "last_word_end": s["last_word_end"],
                "text": s["text"],
            }
        )

    idx_path = OUT_DIR / "_index.json"
    existing = []
    if idx_path.exists():
        existing = json.loads(idx_path.read_text()).get("runs", [])
    merged = {r["file"]: r for r in existing}
    for r in index:
        merged[r["file"]] = r
    idx_path.write_text(
        json.dumps(
            {
                "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "faster_whisper_version": faster_whisper.__version__,
                "ctranslate2_version": ctranslate2.__version__,
                "model": MODEL_SIZE,
                "runs": sorted(merged.values(), key=lambda r: r["file"]),
            },
            indent=2,
        )
    )
    print("wrote", idx_path, flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    raise SystemExit(main())
