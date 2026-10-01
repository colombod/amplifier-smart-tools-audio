#!/usr/bin/env python
"""RECORD real openai-whisper behaviour as replay fixtures (issue #44).

Calls openai-whisper DIRECTLY -- not through `aud` -- in exactly the shape
`aud.dsp.speech.detect_fillers` calls it:

    model = whisper.load_model(model_size)                         # model_size = "base"
    result = model.transcribe(audio, word_timestamps=True, fp16=False)

Two input paths are recorded for every source file, same convention as the
faster-whisper recordings this replaces:

  path="raw"            the float32 array at its OWN sample rate, handed
                        straight to transcribe().
  path="resampled_16k"  the same audio put through scipy.signal.resample_poly
                        to 16 kHz first (aud.dsp.speech._resample_to_whisper_rate).

Unlike the earlier recording run, this one was captured directly in this
repository's own `.venv` (not a throwaway Digital Twin Universe container) --
see RECORDING.md's "Versions recorded" table for this run's exact provenance
and the reason for the difference. No provider credential was present or
read by this script either way.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy import signal

MODEL_SIZE = "base"  # aud.dsp.speech._MODEL_SIZE_DEFAULT
WHISPER_SR = 16000  # aud.dsp.speech._WHISPER_SR

REPO_ROOT = Path(__file__).resolve().parents[4]
WAV_DIR = REPO_ROOT / "tests" / "fixtures" / "recorded" / "wav"
OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "recorded" / "openai_whisper"

SNIPPET = """\
import numpy as np, soundfile as sf
import whisper
x, sr = sf.read(WAV, dtype="float64", always_2d=False)
if x.ndim == 2:
    x = x.mean(axis=1)
# path == "resampled_16k" only:
#   from fractions import Fraction; from scipy import signal
#   frac = Fraction(16000, sr).limit_denominator(2000)
#   x = signal.resample_poly(x, up=frac.numerator, down=frac.denominator)
audio = x.astype("float32")
model = whisper.load_model("base")
result = model.transcribe(audio, word_timestamps=True, fp16=False)
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resample_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    """Byte-for-byte the rule aud.dsp.speech._resample_to_whisper_rate uses."""
    if sr == WHISPER_SR:
        return mono
    frac = Fraction(WHISPER_SR, sr).limit_denominator(2000)
    return signal.resample_poly(mono, up=frac.numerator, down=frac.denominator)


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def record(model, wav: Path, path_mode: str) -> dict:
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
    result = model.transcribe(audio, word_timestamps=True, fp16=False)
    elapsed = time.perf_counter() - t0

    segments = result["segments"]
    words = []
    for seg in segments:
        words.extend(seg.get("words", None) or [])

    degenerate = [
        {"index": i, "word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
        for i, w in enumerate(words)
        if w.get("start") is not None and w["start"] == w["end"]
    ]

    return {
        "provenance": {
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "openai_whisper_version": __import__("importlib.metadata", fromlist=["version"]).version(
                "openai-whisper"
            ),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scipy_version": __import__("scipy").__version__,
            "soundfile_version": sf.__version__,
            "python_version": sys.version,
            "model": MODEL_SIZE,
            "model_constructor": f'whisper.load_model("{MODEL_SIZE}")',
            "transcribe_call": "model.transcribe(audio, word_timestamps=True, fp16=False)",
            "source_wav": wav.name,
            "source_wav_sha256": sha256(wav),
            "source_sample_rate": int(sr),
            "input_path": path_mode,
            "snippet": SNIPPET,
            "wall_seconds": round(elapsed, 3),
            "recorded_in": "repo's own .venv (not a DTU) -- see RECORDING.md",
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
        "summary": {
            "n_segments": len(segments),
            "n_words": len(words),
            "text": result.get("text", ""),
            "last_word_end": (words[-1].get("end") if words else None),
            "degenerate_word_count": len(degenerate),
            "degenerate_words": degenerate,
        },
        "segments": [
            {
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text"),
                "words": [jsonable(w) for w in (seg.get("words") or [])],
            }
            for seg in segments
        ],
        "words_flat": [jsonable(w) for w in words],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or None

    jobs: list[tuple[str, str]] = []
    for sr in (16000, 22050, 44100, 48000):
        for mode in ("raw", "resampled_16k"):
            jobs.append((f"speech_short_{sr}.wav", mode))
    jobs.append(("nospeech_silence_16000.wav", "raw"))
    jobs.append(("nospeech_tone_16000.wav", "raw"))

    print(f"loading whisper.load_model({MODEL_SIZE!r}) ...", flush=True)
    t0 = time.perf_counter()
    model = whisper.load_model(MODEL_SIZE)
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
    idx_path.write_text(
        json.dumps(
            {
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "openai_whisper_version": __import__("importlib.metadata", fromlist=["version"]).version(
                    "openai-whisper"
                ),
                "torch_version": torch.__version__,
                "model": MODEL_SIZE,
                "runs": sorted(index, key=lambda r: r["file"]),
            },
            indent=2,
        )
    )
    print("wrote", idx_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
