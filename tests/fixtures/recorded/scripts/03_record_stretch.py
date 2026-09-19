#!/usr/bin/env python
"""RECORD real python_stretch (Signalsmith Stretch) behaviour as fixtures.

Calls python_stretch DIRECTLY -- not through `aud` -- in exactly the shape
`aud.dsp.timepitch._stretch_signalsmith` / `_pitch_signalsmith` call it:

    stretch = ps.Signalsmith.Stretch()
    stretch.preset(n_channels, sr)
    stretch.timeFactor = <value>
    stretch.setTransposeSemitones(<value>)
    out = stretch.process(audio)        # audio is (channels, samples) float32

The open question these recordings settle: which direction `timeFactor`
runs. So for every requested duration factor f, BOTH settings are recorded:
`timeFactor = f` and `timeFactor = 1/f`. The measured out/in ratio for each
is written down as-is. Nothing here corrects the library.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import python_stretch as ps
import soundfile as sf

OUT_DIR = Path("/rec/out/python_stretch")
SR = 22050

SNIPPET = """\
import numpy as np, python_stretch as ps
audio = np.ascontiguousarray(x2.T, dtype=np.float32)   # (channels, samples)
stretch = ps.Signalsmith.Stretch()
stretch.preset(n_channels, sr)
stretch.timeFactor = <value>                # only when a time factor is under test
stretch.setTransposeSemitones(<value>)      # only when a pitch shift is under test
out = stretch.process(audio)                # -> (channels, samples_out)
"""


def inventory(obj) -> dict:
    names = dir(obj)
    non_callable = []
    for n in names:
        if n.startswith("_"):
            continue
        try:
            v = getattr(obj, n)
        except Exception as exc:  # pragma: no cover
            non_callable.append({"name": n, "error": repr(exc)})
            continue
        if callable(v):
            continue
        non_callable.append({"name": n, "value": repr(v), "type": type(v).__name__})
    return {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "dir": names,
        "public_non_callable": non_callable,
        "public_callables": [n for n in names if not n.startswith("_") and callable(getattr(obj, n, None))],
    }


def make_input(n: int) -> np.ndarray:
    """A deterministic, non-trivial mono test signal: two tones + a click."""
    t = np.arange(n, dtype=np.float64) / SR
    x = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    x *= np.hanning(n) * 0.5 + 0.5  # gentle envelope so edges are not a step
    x[n // 2] += 0.5  # a click, so a replay can see time alignment
    return x.reshape(-1, 1)  # (samples, channels) -- aud's internal layout


def run_case(x2: np.ndarray, *, time_factor: float | None, semitones: float | None) -> dict:
    n_channels = x2.shape[1]
    audio = np.ascontiguousarray(x2.T, dtype=np.float32)
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(n_channels, SR)
    set_time = None
    if time_factor is not None:
        stretch.timeFactor = float(time_factor)
        set_time = float(stretch.timeFactor)  # read back what the library stored
    if semitones is not None:
        stretch.setTransposeSemitones(float(semitones))
    processed = stretch.process(audio)
    out = np.asarray(processed, dtype=np.float64).T
    n_in = int(x2.shape[0])
    n_out = int(out.shape[0])
    extra = {}
    for attr in ("inputLatency", "outputLatency", "blockSamples", "intervalSamples", "channels", "timeFactor"):
        if hasattr(stretch, attr):
            try:
                extra[attr] = float(getattr(stretch, attr))
            except Exception as exc:  # pragma: no cover
                extra[attr] = repr(exc)
    return {
        "time_factor_set": set_time,
        "semitones_set": (None if semitones is None else float(semitones)),
        "input_samples": n_in,
        "output_samples": n_out,
        "ratio_out_over_in": (n_out / n_in if n_in else None),
        "input_duration_s": round(n_in / SR, 6),
        "output_duration_s": round(n_out / SR, 6),
        "output_shape": list(np.asarray(processed).shape),
        "output_dtype": str(np.asarray(processed).dtype),
        "output_peak": float(np.max(np.abs(out))) if n_out else 0.0,
        "output_rms": float(np.sqrt(np.mean(out**2))) if n_out else 0.0,
        "stretch_state_after_process": extra,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_in = SR  # exactly 1.000 s, so ratios read directly as durations
    x2 = make_input(n_in)

    probe = ps.Signalsmith.Stretch()
    probe.preset(1, SR)
    inv = {
        "module": inventory(ps),
        "Signalsmith": inventory(ps.Signalsmith),
        "Stretch_instance_after_preset": inventory(probe),
    }

    factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    semis = [-12.0, -7.0, 0.0, 7.0, 12.0]

    direct = []  # timeFactor = f   (the literal request)
    recip = []  # timeFactor = 1/f (what aud.dsp.timepitch does today)
    for f in factors:
        d = run_case(x2, time_factor=f, semitones=None)
        d["requested_duration_factor"] = f
        d["convention_under_test"] = "timeFactor = factor"
        direct.append(d)
        print(f"timeFactor={f:<5} -> out/in={d['ratio_out_over_in']:.4f}  ({d['output_samples']}/{d['input_samples']})", flush=True)
        r = run_case(x2, time_factor=1.0 / f, semitones=None)
        r["requested_duration_factor"] = f
        r["convention_under_test"] = "timeFactor = 1/factor"
        recip.append(r)
        print(f"timeFactor=1/{f:<3} -> out/in={r['ratio_out_over_in']:.4f}  ({r['output_samples']}/{r['input_samples']})", flush=True)

    pitch = []
    for s in semis:
        p = run_case(x2, time_factor=None, semitones=s)
        pitch.append(p)
        print(f"semitones={s:<6} -> out/in={p['ratio_out_over_in']:.4f} peak={p['output_peak']:.4f}", flush=True)

    combined = []
    for f, s in ((1.2, 7.0), (0.8, -12.0), (2.0, 12.0)):
        c = run_case(x2, time_factor=1.0 / f, semitones=s)
        c["requested_duration_factor"] = f
        c["convention_under_test"] = "timeFactor = 1/factor, with transpose"
        combined.append(c)
        print(f"timeFactor=1/{f} + {s}st -> out/in={c['ratio_out_over_in']:.4f}", flush=True)

    # A real input/output AUDIO pair a replay can assert on sample-by-sample.
    pair_dir = OUT_DIR / "audio_pair"
    pair_dir.mkdir(exist_ok=True)
    np.save(pair_dir / "input_mono_22050.npy", x2[:, 0].astype(np.float32))
    sf.write(pair_dir / "input_mono_22050.wav", x2[:, 0], SR, subtype="PCM_16")
    pair_manifest = []
    for label, tf in (("timeFactor_0.5", 0.5), ("timeFactor_2.0", 2.0), ("timeFactor_1.0", 1.0)):
        audio = np.ascontiguousarray(x2.T, dtype=np.float32)
        st = ps.Signalsmith.Stretch()
        st.preset(1, SR)
        st.timeFactor = tf
        out = np.asarray(st.process(audio), dtype=np.float64).T[:, 0]
        np.save(pair_dir / f"output_{label}.npy", out.astype(np.float32))
        sf.write(pair_dir / f"output_{label}.wav", out, SR, subtype="PCM_16")
        pair_manifest.append(
            {
                "label": label,
                "time_factor_set": tf,
                "input_npy": "input_mono_22050.npy",
                "output_npy": f"output_{label}.npy",
                "input_samples": int(x2.shape[0]),
                "output_samples": int(out.shape[0]),
                "ratio_out_over_in": float(out.shape[0] / x2.shape[0]),
                "output_first16": [float(v) for v in out[:16]],
                "output_peak": float(np.max(np.abs(out))),
            }
        )
        print(f"audio pair {label}: {out.shape[0]} samples out", flush=True)

    so_files = []
    mod_dir = Path(ps.__file__).parent
    for p in sorted(mod_dir.rglob("*.so")):
        so_files.append({"path": str(p), "bytes": p.stat().st_size})

    doc = {
        "provenance": {
            "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "python_stretch_version": getattr(ps, "__version__", None),
            "python_stretch_file": ps.__file__,
            "loaded_shared_objects": so_files,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "sample_rate": SR,
            "channels": 1,
            "snippet": SNIPPET,
            "input_signal": "0.4*sin(220Hz) + 0.2*sin(1000Hz), raised-hanning envelope, +0.5 click at n/2, 1.000 s @ 22050 Hz",
        },
        "api_inventory": inv,
        "time_factor_direct": direct,
        "time_factor_reciprocal": recip,
        "transpose_only": pitch,
        "combined": combined,
        "audio_pair": pair_manifest,
    }
    (OUT_DIR / "python_stretch_recording.json").write_text(json.dumps(doc, indent=2))
    print("wrote", OUT_DIR / "python_stretch_recording.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
