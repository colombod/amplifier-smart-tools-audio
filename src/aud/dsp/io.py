"""Audio file I/O -- a thin, fail-loud wrapper over soundfile (libsndfile).

WAV, FLAC and AIFF decode with no extra dependency. Compressed formats
(mp3, m4a, ogg) need ffmpeg on the host; when libsndfile cannot decode a
file we say so plainly rather than letting a cryptic soundfile exception
propagate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ["read_audio", "write_audio"]


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file.

    Args:
        path: Path to a WAV/FLAC/AIFF/etc. file.

    Returns:
        (data, sample_rate) where data has shape (n_samples, n_channels),
        dtype float64, in the range [-1.0, 1.0] for PCM sources. Always
        2-D, even for a mono file (shape (n_samples, 1)).

    Raises:
        FileNotFoundError: The path does not exist.
        RuntimeError: libsndfile could not decode the file (e.g. an mp3
            with no ffmpeg on the host, a truncated/corrupt file, or an
            unsupported container).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")

    try:
        data, samplerate = sf.read(str(p), dtype="float64", always_2d=True)
    except Exception as exc:  # re-raised below with a clear, actionable message
        raise RuntimeError(
            f"Could not decode audio file '{p}': {exc}. "
            "If this is a compressed format (mp3, m4a, ogg), libsndfile needs "
            "ffmpeg on this host to handle it; WAV/FLAC/AIFF need nothing extra."
        ) from exc

    return np.asarray(data, dtype=np.float64), int(samplerate)


def write_audio(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """Write audio to disk.

    Args:
        path: Destination path. Parent directories are created if missing.
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        subtype: soundfile subtype, e.g. "PCM_24", "PCM_16", "FLOAT".

    Raises:
        RuntimeError: libsndfile could not encode/write the file (bad
            subtype for the container, unwritable path, disk full, etc.).
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(p), np.asarray(x), sr, subtype=subtype)
    except Exception as exc:  # re-raised below with a clear, actionable message
        raise RuntimeError(f"Could not write audio file '{p}' (subtype={subtype!r}): {exc}") from exc
