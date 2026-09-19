"""Onset and silence detection -- positions in, positions out.

Follows the `dsp/` boundary (AGENTS.md #8, docs/ARCHITECTURE.md #6): these
functions take arrays and parameters and return plain lists of dicts. They
never read configuration, never touch the filesystem, and never raise a
user-facing `AudError`. Turning this module's output into a regions
document -- with `source`, `sample_rate` and provenance -- is
`aud.core.regions`'s job (see `aud.lib.detect_silence` etc.).

Every detector here runs on the **mono sum** of a multi-channel signal (see
`docs/ARCHITECTURE.md`'s "One position, all channels": a per-channel search
would find a different position per channel and desynchronise them, which
is a worse defect than whatever it was trying to fix).

Which regions are found, and the exact algorithm used, is explicitly
**not promised** by contracts/regions.v1.md -- it will change between
releases. What is fixed by this module's contract with its callers is the
*shape* of what it returns: a list of dicts with the fields
`aud.core.regions.new_regions` expects for the corresponding `kind`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

__all__ = ["detect_silence", "detect_transients", "measure_noise_floor"]

_EPS = 1e-12


def _mono_sum(x: np.ndarray) -> np.ndarray:
    """Fold a (possibly multi-channel) signal down to one channel by averaging.

    Detection runs on this mono sum rather than per-channel, for the same
    reason edit-point search does (docs/ARCHITECTURE.md "One position, all
    channels"): a per-channel result would desynchronise the channels at
    the very moment it is trying to protect them.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, _EPS)))


def measure_noise_floor(x: np.ndarray, sr: int, frame_ms: float = 50.0, percentile: float = 10.0) -> float:
    """Estimate the recording's noise floor, in dBFS.

    Deliberately **not** the global minimum sample or the quietest single
    frame: a low *percentile* of short-time frame RMS across the whole
    signal. The distinction matters. The global minimum can be a single
    digitally-silent sample (or a single unluckily-quiet frame) that says
    nothing about the recording's actual background level; a percentile
    across many frames is dragged down by the *general* quiet parts of the
    file and shrugs off one outlier frame the way a median shrugs off one
    outlier sample. This is exactly what `detect_silence`'s
    `threshold_above_floor_db` needs to be relative to (see
    contracts/regions.v1.md#why-the-silence-threshold-is-relative): a floor
    that could be fooled by one freak sample would make the threshold
    meaningless for every other frame in the file.

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_channels).
        sr: Sample rate in Hz.
        frame_ms: Frame length for the short-time RMS analysis, in ms.
        percentile: Which percentile of frame RMS counts as the floor.
            10 (the default) means "quieter than 90% of the file's frames".

    Returns:
        The measured floor in dBFS (<= 0 for any input in [-1, 1]).
    """
    mono = _mono_sum(x)
    frame_len = max(1, int(frame_ms / 1000.0 * sr))
    n_frames = mono.shape[0] // frame_len
    if n_frames < 1:
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        return _db(rms)
    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    floor = float(np.percentile(frame_rms, percentile))
    return _db(floor)


def detect_silence(
    x: np.ndarray,
    sr: int,
    threshold_above_floor_db: float = 6.0,
    min_len_ms: float = 400.0,
    frame_ms: float = 10.0,
) -> list[dict[str, Any]]:
    """Find quiet spans, relative to this file's own measured noise floor.

    The threshold is expressed as dB **above the measured floor**
    (`measure_noise_floor`), never as a fixed dBFS number -- see
    contracts/regions.v1.md#why-the-silence-threshold-is-relative. This is
    what makes the same `threshold_above_floor_db` mean the same thing on a
    treated-room recording floored at -70 dBFS and a phone-in-a-kitchen
    recording floored at -38 dBFS, and it is also what makes the *same*
    signal, uniformly scaled 20 dB quieter, produce the *same* regions:
    both the floor and every frame's level drop by the same 20 dB, so the
    difference the threshold acts on is unchanged.

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_channels).
        sr: Sample rate in Hz.
        threshold_above_floor_db: A frame counts as silent while its level
            stays below `noise_floor + threshold_above_floor_db`.
        min_len_ms: Silent spans shorter than this are dropped -- they are
            not reported at all, not clamped to the minimum.
        frame_ms: Frame length for the short-time RMS envelope, in ms.
            Smaller values give finer boundary resolution.

    Returns:
        A list of dicts, each `{"start_s", "end_s", "peak_dbfs",
        "rms_dbfs"}`, ascending by `start_s` and non-overlapping -- ready
        for `aud.core.regions.new_regions(kind="silence", ...)`.
    """
    mono = _mono_sum(x)
    floor_dbfs = measure_noise_floor(mono, sr, frame_ms=50.0)
    threshold_db = floor_dbfs + threshold_above_floor_db

    frame_len = max(1, int(frame_ms / 1000.0 * sr))
    n_frames = mono.shape[0] // frame_len
    if n_frames < 1:
        return []

    frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_db = np.array([_db(v) for v in frame_rms])
    is_silent = frame_db < threshold_db

    spans: list[tuple[int, int]] = []
    start_idx: int | None = None
    for i, silent in enumerate(is_silent):
        if silent and start_idx is None:
            start_idx = i
        elif not silent and start_idx is not None:
            spans.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        spans.append((start_idx, n_frames))

    regions: list[dict[str, Any]] = []
    for start_frame, end_frame in spans:
        start_s = start_frame * frame_len / sr
        end_s = end_frame * frame_len / sr
        if (end_s - start_s) * 1000.0 < min_len_ms:
            continue
        segment = mono[start_frame * frame_len : end_frame * frame_len]
        peak = float(np.max(np.abs(segment))) if segment.size else 0.0
        rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0
        regions.append(
            {
                "start_s": float(start_s),
                "end_s": float(end_s),
                "peak_dbfs": _db(peak),
                "rms_dbfs": _db(rms),
            }
        )
    return regions


def detect_transients(
    x: np.ndarray,
    sr: int,
    sensitivity: float = 1.0,
    min_separation_ms: float = 50.0,
    frame_ms: float = 20.0,
    hop_ms: float = 5.0,
) -> list[dict[str, Any]]:
    """Find onsets -- the instants new sounds start.

    Uses spectral flux: the positive-only (half-wave-rectified) frame-to-
    frame increase in STFT magnitude, summed across frequency bins. A rise
    in flux means new energy appeared somewhere in the spectrum that was
    not there a frame ago, which is what an onset *is* -- as opposed to a
    plain level rise, which a sustained crescendo also produces without
    being a new event.

    Peaks are picked against an **adaptive** threshold: a rolling median of
    the flux curve plus a margin, rather than a fixed value. A fixed
    threshold would be right for exactly one signal level, the same
    failure mode a fixed silence threshold has; a rolling median tracks the
    local background flux (which rises during busy passages and falls
    during sparse ones) so the same `sensitivity` means the same thing
    throughout a file with varying density.

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_channels).
        sr: Sample rate in Hz.
        sensitivity: Peak-picking sensitivity; higher reports more onsets
            (the margin above the local median shrinks as sensitivity
            grows).
        min_separation_ms: Onsets closer together than this are merged,
            keeping the stronger of the two.
        frame_ms: STFT window length, in ms.
        hop_ms: STFT hop length, in ms. Onset time resolution.

    Returns:
        A list of dicts, each `{"start_s", "end_s", "strength"}` with
        `end_s == start_s`, ascending by `start_s` -- ready for
        `aud.core.regions.new_regions(kind="transient", ...)`. `strength`
        is normalised to (0, 1] within this call only (see
        contracts/regions.v1.md: comparable within one document, nowhere
        else).

    Suited to real programme material: speech, music, foley, anything with
    an actual noise floor and genuine onsets, where a real onset arrives as
    a flux spike far larger than STFT bin-leakage jitter. NOT suited to a
    continuous, perfectly sustained, unmodulated tone with no noise floor at
    all -- there, bin-leakage drift alone can clear the adaptive threshold
    and produce a handful of spurious onsets (measured: up to 3 on a 3-second
    440 Hz sine at the default sensitivity; see
    tests/test_dsp_detect.py::test_transients_do_not_swamp_a_realistic_steady_state_background).
    This is a disclosed limit of a lightweight spectral-flux detector on
    laboratory-tone input, not a promise of immunity to every input -- see
    contracts/regions.v1.md's "Not promised".
    """
    mono = _mono_sum(x)
    if sensitivity <= 0:
        sensitivity = _EPS

    frame_len = max(4, int(frame_ms / 1000.0 * sr))
    hop_len = max(1, int(hop_ms / 1000.0 * sr))
    noverlap = max(0, frame_len - hop_len)
    if mono.size < frame_len:
        return []

    _freqs, times, stft_matrix = signal.stft(mono, fs=sr, nperseg=frame_len, noverlap=noverlap, boundary=None)
    magnitude = np.abs(stft_matrix)
    if magnitude.shape[1] < 2:
        return []

    flux = np.sum(np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0.0), axis=0)
    flux = np.concatenate(([0.0], flux))

    # Rolling median over roughly a 200ms window, as the adaptive baseline.
    median_frames = max(3, int(0.2 * sr / hop_len))
    if median_frames % 2 == 0:
        median_frames += 1
    if median_frames >= flux.size:
        median_frames = flux.size if flux.size % 2 == 1 else flux.size - 1
        median_frames = max(1, median_frames)
    local_median = signal.medfilt(flux, kernel_size=median_frames) if median_frames >= 1 else np.zeros_like(flux)

    flux_std = float(np.std(flux))
    # 3-sigma margin above the local median: a standard robust-statistics
    # choice for "this is a real outlier, not background jitter". A tighter
    # margin (closer to 1 sigma) makes a continuous, loud, perfectly
    # sustained tone produce dozens of spurious onsets -- STFT analysis of a
    # tone not aligned to a bin center leaks energy between bins in a way
    # that drifts slightly frame-to-frame, and that drift alone clears a
    # low threshold. 3 sigma clears that specific noise floor while still
    # catching genuine onsets, which arrive as a flux spike far larger than
    # any single-tone leakage artifact (see tests/test_dsp_detect.py).
    margin = 3.0 * (flux_std + _EPS) / sensitivity
    threshold = local_median + margin

    min_separation_frames = max(1, int(min_separation_ms / 1000.0 * sr / hop_len))

    peak_indices: list[int] = []
    n = flux.size
    for i in range(1, n - 1):
        if flux[i] <= threshold[i]:
            continue
        if flux[i] < flux[i - 1] or flux[i] < flux[i + 1]:
            continue  # not a local maximum
        if peak_indices and (i - peak_indices[-1]) < min_separation_frames:
            if flux[i] > flux[peak_indices[-1]]:
                peak_indices[-1] = i
            continue
        peak_indices.append(i)

    max_flux = float(np.max(flux)) if flux.size else 0.0
    regions: list[dict[str, Any]] = []
    for idx in peak_indices:
        start_s = float(times[idx])
        strength = float(flux[idx] / (max_flux + _EPS))
        regions.append({"start_s": start_s, "end_s": start_s, "strength": max(strength, _EPS)})
    return regions
