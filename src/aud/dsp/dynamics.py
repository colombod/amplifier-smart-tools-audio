"""Feed-forward (multiband) compressor -- the centre of this tool.

Topology, in order: detector (peak or RMS) -> log (dB) domain -> soft-knee
static gain computer -> gain smoothing with separate attack/release time
constants -> apply to signal -> makeup gain.

Stereo-linked detection is the default: a single detector signal, shared
across channels, drives the gain applied to every channel identically.
Independent per-channel detection would let a transient on one channel
duck only that channel, shifting the stereo image sample-by-sample -- the
opposite of what a mastering compressor should do to a finished stereo mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aud.dsp import crossover

__all__ = ["BandParams", "compress", "multiband_compress", "static_gain_reduction_db"]

_EPS = 1e-12


@dataclass
class BandParams:
    """Compressor settings for one band (or a single full-band pass)."""

    threshold_db: float = -18.0
    ratio: float = 2.0
    attack_ms: float = 10.0
    release_ms: float = 120.0
    knee_db: float = 6.0
    makeup_db: float = 0.0
    enabled: bool = True


def _db_to_lin(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (db / 20.0)


def _lin_to_db(lin: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(lin, _EPS))


def _detector_signal(x: np.ndarray, detector: str, sr: int, stereo_link: bool) -> np.ndarray:
    """Return a 1-D (or per-channel) level-detector signal, linear scale."""
    if detector not in ("peak", "rms"):
        raise ValueError(f"detector must be 'peak' or 'rms', got {detector!r}")

    if detector == "peak":
        level = np.abs(x)
        if stereo_link and x.ndim == 2 and x.shape[1] > 1:
            level = np.max(level, axis=1, keepdims=True)
        return level

    # RMS: smooth x^2 with a short (~10 ms) one-pole low-pass, then sqrt.
    power = x * x
    if stereo_link and x.ndim == 2 and x.shape[1] > 1:
        power = np.mean(power, axis=1, keepdims=True)
    rms_time_ms = 10.0
    coeff = np.exp(-1.0 / (sr * rms_time_ms / 1000.0))
    smoothed = np.empty_like(power)
    acc = np.zeros(power.shape[1]) if power.ndim == 2 else 0.0
    for n in range(power.shape[0]):
        acc = coeff * acc + (1 - coeff) * power[n]
        smoothed[n] = acc
    return np.sqrt(np.maximum(smoothed, 0.0))


def static_gain_reduction_db(level_db: np.ndarray, params: BandParams) -> np.ndarray:
    """Soft-knee gain reduction curve, in dB (<= 0), per Giannoulis et al.

    Public (not `_`-prefixed): `aud.dsp.gate.dynamic_eq` (Step 7 of the
    masking/ducking epic, issue #17) reuses this UNMODIFIED, per band, to
    build its own threshold/ratio/knee gain law from an external key's
    level -- this function already takes a `level_db` array and returns a
    gain, so it applies per band with no changes at all. Renamed from
    `_static_gain_reduction_db` (private) to this public name for that
    reuse: this repo's own convention (see `aud.dsp.collision`'s
    `masking_offset` docstring) is cross-module reuse via a PUBLIC API,
    never a private import.
    """
    t = params.threshold_db
    w = max(params.knee_db, 0.0)
    ratio = max(params.ratio, 1e-6)

    below = level_db - t < -w / 2
    above = level_db - t > w / 2
    in_knee = ~below & ~above

    y_db = np.array(level_db, dtype=np.float64, copy=True)
    y_db[above] = t + (level_db[above] - t) / ratio
    if np.any(in_knee):
        x_minus_t = level_db[in_knee] - t + w / 2
        y_db[in_knee] = (
            level_db[in_knee] + ((1.0 / ratio - 1.0) * x_minus_t**2) / (2.0 * w) if w > 0 else level_db[in_knee]
        )
    # below: y_db == level_db already (no change)
    return y_db - level_db  # <= 0 everywhere


def _smooth_gain_db(gain_db: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Attack/release smoothing of a (per-sample) gain-reduction dB signal."""
    attack_coeff = np.exp(-1.0 / (sr * max(attack_ms, 1e-3) / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * max(release_ms, 1e-3) / 1000.0))

    smoothed = np.empty_like(gain_db)
    prev = 0.0
    for n in range(gain_db.shape[0]):
        target = gain_db[n]
        coeff = attack_coeff if target < prev else release_coeff
        prev = coeff * prev + (1 - coeff) * target
        smoothed[n] = prev
    return smoothed


def compress(
    x: np.ndarray,
    sr: int,
    params: BandParams,
    detector: str = "peak",
    stereo_link: bool = True,
) -> tuple[np.ndarray, dict]:
    """Apply single-band feed-forward compression.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        params: Compressor settings.
        detector: "peak" or "rms".
        stereo_link: Share one detector signal across channels (default),
            rather than compressing each channel independently.

    Returns:
        (y, stats) where stats = {
            "max_gain_reduction_db": float,
            "avg_gain_reduction_db": float,
            "gain_envelope_db": np.ndarray,  # diagnostic, not JSON-safe
        }
    """
    x = np.asarray(x, dtype=np.float64)
    if not params.enabled:
        n = x.shape[0]
        zeros = np.zeros(n)
        return x.copy(), {
            "max_gain_reduction_db": 0.0,
            "avg_gain_reduction_db": 0.0,
            "gain_envelope_db": zeros,
        }

    level = _detector_signal(x, detector, sr, stereo_link)
    level_db = _lin_to_db(level)
    if level_db.ndim == 2:
        level_db = level_db[:, 0]

    raw_gain_db = static_gain_reduction_db(level_db, params)
    smoothed_gain_db = _smooth_gain_db(raw_gain_db, sr, params.attack_ms, params.release_ms)

    gain_lin = _db_to_lin(smoothed_gain_db)
    makeup_lin = _db_to_lin(params.makeup_db)
    y = x * gain_lin[:, None] * makeup_lin

    stats = {
        "max_gain_reduction_db": float(np.min(smoothed_gain_db)),
        "avg_gain_reduction_db": float(np.mean(smoothed_gain_db)),
        "gain_envelope_db": smoothed_gain_db,
    }
    return y, stats


def multiband_compress(
    x: np.ndarray,
    sr: int,
    crossovers: list[float],
    bands: list[BandParams],
) -> tuple[np.ndarray, dict]:
    """Split into bands at LR4 crossovers, compress each band, recombine.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        crossovers: Crossover frequencies in Hz (strictly increasing).
        bands: Per-band compressor settings; len(bands) must equal
            len(crossovers) + 1.

    Returns:
        (y, stats) where stats = {"bands": [per-band stats dict, ...]}.

    Raises:
        ValueError: len(bands) != len(crossovers) + 1.
    """
    if len(bands) != len(crossovers) + 1:
        raise ValueError(
            f"multiband_compress needs len(bands) == len(crossovers) + 1, "
            f"got {len(bands)} bands and {len(crossovers)} crossovers"
        )

    band_signals = crossover.split(x, sr, crossovers)
    processed: list[np.ndarray] = []
    band_stats: list[dict] = []
    for band_signal, band_params in zip(band_signals, bands, strict=True):
        y, stats = compress(band_signal, sr, band_params)
        processed.append(y)
        band_stats.append(stats)

    y_total = crossover.recombine(processed)
    return y_total, {"bands": band_stats}
