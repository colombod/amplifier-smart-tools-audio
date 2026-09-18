"""Loudness measurement and gain-only normalization -- a thin pyloudnorm wrapper.

BS.1770 (ITU-R) integrated loudness is a standardized, precisely specified
algorithm; pyloudnorm (MIT-licensed) already implements it correctly, so
`integrated_lufs` wraps it directly rather than re-deriving K-weighting
filters and gating logic from scratch.

Judgment call: the pinned pyloudnorm version (<0.2.0, per this tool's
dependency bounds) does not expose a `loudness_range`/EBU Tech 3342
implementation -- that landed in pyloudnorm 0.2. Reimplementing BS.1770
K-weighting from scratch to get there would violate the "wrap, don't
re-derive" intent above, so `loudness_range` instead implements the
EBU Tech 3342 windowing/gating/percentile procedure on top of the ONE
building block 0.1.x does expose: `Meter.integrated_loudness`. Each
3-second, 100 ms-hop block's loudness is measured by calling
`integrated_loudness` on just that slice (so pyloudnorm's own accurate
K-weighting filter design does all the per-sample math); this module only
adds the windowing, the two-stage (absolute + relative) gate, and the
95th/10th percentile spread that the spec calls "loudness range". This is
a deliberate, documented approximation of Tech 3342 (which specifies an
ungated per-block loudness before the range's own gating stage; here each
block is pyloudnorm's own gated integrated loudness of that slice),
verified only against a general behavioral property (quiet -> loud
material produces a measurable finite range > 0), not against a
reference LRA implementation bit-for-bit.
"""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln

__all__ = ["integrated_lufs", "loudness_range", "normalize"]


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """BS.1770 integrated loudness, in LUFS.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.

    Returns:
        Integrated loudness in LUFS. May be -inf for silence (pyloudnorm's
        own convention); callers that need a JSON-safe value should treat
        a non-finite result as "no measurable loudness" (e.g. map to null).
    """
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))


_LRA_BLOCK_SECONDS = 3.0
_LRA_HOP_SECONDS = 0.1
_LRA_ABSOLUTE_GATE_LUFS = -70.0
_LRA_RELATIVE_GATE_OFFSET_DB = 20.0
_LRA_LOW_PERCENTILE = 10.0
_LRA_HIGH_PERCENTILE = 95.0


def loudness_range(x: np.ndarray, sr: int) -> float:
    """EBU R128 loudness range (LRA), in LU.

    See the module docstring for how this is computed on top of the
    pinned pyloudnorm version's public API.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.

    Returns:
        Loudness range in LU (the spread between the 95th and 10th
        percentile of gated block loudness). 0.0 if there are fewer than
        two gated blocks to compare (e.g. very short or very quiet input).
    """
    x = np.asarray(x, dtype=np.float64)
    meter = pyln.Meter(sr)

    block_len = int(_LRA_BLOCK_SECONDS * sr)
    hop_len = max(1, int(_LRA_HOP_SECONDS * sr))
    if x.shape[0] < block_len:
        return 0.0

    block_loudness_lufs = []
    for start in range(0, x.shape[0] - block_len + 1, hop_len):
        block = x[start : start + block_len]
        value = float(meter.integrated_loudness(block))
        if math.isfinite(value):
            block_loudness_lufs.append(value)

    absolute_gated = [v for v in block_loudness_lufs if v >= _LRA_ABSOLUTE_GATE_LUFS]
    if len(absolute_gated) < 2:
        return 0.0

    mean_power = float(np.mean([10.0 ** (v / 10.0) for v in absolute_gated]))
    relative_threshold_lufs = 10.0 * math.log10(max(mean_power, 1e-12)) - _LRA_RELATIVE_GATE_OFFSET_DB

    relative_gated = [v for v in absolute_gated if v >= relative_threshold_lufs]
    if len(relative_gated) < 2:
        return 0.0

    high = float(np.percentile(relative_gated, _LRA_HIGH_PERCENTILE))
    low = float(np.percentile(relative_gated, _LRA_LOW_PERCENTILE))
    return high - low


def normalize(x: np.ndarray, sr: int, target_lufs: float) -> tuple[np.ndarray, float]:
    """Apply a single gain to hit a target integrated loudness.

    This is gain only: it does NOT guarantee any peak or true-peak
    ceiling. A signal normalized to -14 LUFS can still clip or exceed a
    true-peak ceiling if it was already hot; enforcing a ceiling is the
    limiter's job (see `aud.dsp.limiter.brickwall`), applied after this step.

    Args:
        x: Array of shape (n_samples, n_channels).
        sr: Sample rate in Hz.
        target_lufs: Desired integrated loudness, in LUFS.

    Returns:
        (y, applied_gain_db). If the input is effectively silent
        (non-finite measured loudness), no gain is applied and
        applied_gain_db is 0.0 -- there is no meaningful gain that turns
        silence into a target loudness.
    """
    x = np.asarray(x, dtype=np.float64)
    current = integrated_lufs(x, sr)
    if not math.isfinite(current):
        return x.copy(), 0.0

    gain_db = target_lufs - current
    gain_lin = 10.0 ** (gain_db / 20.0)
    return x * gain_lin, float(gain_db)
