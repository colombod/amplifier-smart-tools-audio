"""Channel-count operations shared across the DSP stack.

`fold_to_mono` is the bare, headroom-agnostic mono fold several other `dsp/`
modules already needed for an internal ANALYSIS signal -- a measurement that
is never itself written to disk. Before this module existed, three of them
carried their own identical one-line copy: `aud.dsp.eqmatch._mono`,
`aud.dsp.speech.detect_fillers`'s inline fold before resampling to
faster-whisper's rate, and `aud.dsp.reverb`'s IR-channel downmix. All three
computed exactly `x if x.ndim == 1 else np.mean(x, axis=1)`, so they are
consolidated here rather than left as three copies of the same nine
characters. A fourth copy, `aud.dsp.resolve._mono_sum`, uses `np.sum`
instead of `np.mean` -- only the SIGN of the folded signal matters for its
one caller (zero-crossing search), so it is a different computation, not
the same one spelled differently, and is deliberately left unconsolidated;
see its own docstring.

`downmix` is the audible `downmix` plan stage (contracts/plan.v1.md):
folding a multichannel programme down to one channel a caller will actually
hear, and `aud.lib.render` will actually write to disk. That has two
obligations `fold_to_mono` does not carry:

1. It must never produce a sample outside [-1.0, 1.0] for in-range input --
   see `downmix`'s own docstring for the proof.
2. A caller must be able to tell a good fold from an accidental near-silent
   one caused by out-of-phase channels -- `downmix` always reports the
   correlation it measured, never just a quieter file with no explanation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["downmix", "fold_to_mono"]

_EPS = 1e-12

# Below this pairwise Pearson correlation, channels are close enough to
# antiphase that a mono fold will cancel most of the programme's energy.
# Ordinary stereo material (double-tracked instruments, a wide reverb
# return, a stereo synth) sits anywhere from ~0.0 to ~0.9; an accidental
# polarity flip on one channel is close to -1.0. -0.5 is comfortably past
# "decorrelated" and comfortably short of "exactly inverted", so it flags
# the case a caller actually needs to know about without tripping on an
# ordinary wide mix.
_ANTIPHASE_CORRELATION_THRESHOLD = -0.5


def fold_to_mono(x: np.ndarray) -> np.ndarray:
    """Per-sample arithmetic mean across channels. 1-D input is returned unchanged.

    This is an ANALYSIS fold: the result is a measurement input (a spectral
    profile, a speech-recognition signal, an impulse response), never
    audio written to a file, so it carries none of `downmix`'s headroom or
    antiphase-reporting obligations.
    """
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else np.mean(x, axis=1)


def _peak_dbfs(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * float(np.log10(max(peak, _EPS)))


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * float(np.log10(max(rms, _EPS)))


def _mean_pairwise_correlation(x: np.ndarray) -> float | None:
    """Average Pearson correlation over every distinct channel pair.

    `None` when it is undefined for every pair -- any channel with
    (near-)zero variance (digital silence, a pure DC offset) makes
    Pearson's denominator zero, and a fabricated 0.0 would misreport
    "uncorrelated" for a pair that is actually just quiet.
    """
    n_channels = x.shape[1]
    correlations: list[float] = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            a, b = x[:, i], x[:, j]
            if np.std(a) < _EPS or np.std(b) < _EPS:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(corr):
                correlations.append(corr)
    if not correlations:
        return None
    return sum(correlations) / len(correlations)


def downmix(x: np.ndarray, sr: int, params: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Fold a multichannel signal down to one channel, for the `downmix` plan stage.

    Args:
        x: Array of shape (n_samples, n_channels), or (n_samples,) for an
            already-mono signal.
        sr: Unused -- folding channels needs no rate information. Present
            so this function matches the `(x, sr, params)` shape every
            `dsp.engine` stage executor calls its handler with.
        params: Unused today (`downmix` takes no parameters). Present for
            the same reason as `sr`, and so a future parameter does not
            change this function's call shape.

    Returns:
        `(mono, report)`. `mono` always has shape `(n_samples, 1)` --
        matching `aud.dsp.io`'s "always 2-D" guarantee, never a bare 1-D
        squeeze. `report`:

        - `input_channels`: channel count of `x` before folding.
        - `correlation`: mean pairwise Pearson correlation across `x`'s
          channels, or `None` when undefined (mono input, or every pair
          had near-zero variance).
        - `antiphase_detected`: `True` when `correlation` is at or below
          `_ANTIPHASE_CORRELATION_THRESHOLD` -- reported, never raised as
          an error; see "Antiphase is detected, never refused" below.
        - `peak_dbfs_before` / `peak_dbfs_after`, `rms_dbfs_before` /
          `rms_dbfs_after`: sample-peak and RMS level, in dBFS, of `x` and
          of the folded mono output.

    Fold rule: the arithmetic mean across channels -- sum-and-divide,
    never a plain sum. For any input restricted to [-1.0, 1.0] (the range
    `aud.dsp.io.read_audio` guarantees for a PCM source), the mean of N
    such values can never leave that range either, by the triangle
    inequality:

        |mean(c_1, ..., c_n)| = |sum(c_i)| / n <= sum(|c_i|) / n
                              <= n * max(|c_i|) / n = max(|c_i|) <= 1.0

    A plain sum has no such guarantee: two full-scale, in-phase channels
    sum to 2.0, well outside range. `tests/test_dsp_channels.py` proves
    this with a negative control that reproduces exactly that overflow
    under a bare `np.sum`, then shows the mean rule staying at exactly the
    input's own peak on the same signal.

    Antiphase is DETECTED, never silently swallowed and never refused.
    `report["correlation"]` and `report["antiphase_detected"]` are always
    present, so a caller can tell a good fold (loud, uncorrelated-but-not-
    antiphase stereo) from an accidental near-silent one (two channels
    close to exact polarity inversion, which a naive mono fold cancels to
    near-nothing). This is a REPORT FIELD, not a raised `AudError`,
    deliberately: `downmix` on out-of-phase material is still a valid,
    requested operation -- some legitimate sources really are recorded or
    processed that way (a stereo-widened mix, a mid-side decode gone one
    step too far) -- and refusing to write the file would take away the
    one piece of evidence (the file itself, plus this report) a caller
    needs to notice and fix the actual problem upstream. A near-silent
    output that LOOKS like an ordinary successful render is the failure
    mode being designed out here; surfacing the correlation makes it
    visible instead of invisible.
    """
    del sr, params
    x = np.asarray(x, dtype=np.float64)
    x2 = x[:, None] if x.ndim == 1 else x
    n_channels = x2.shape[1]

    if n_channels == 1:
        peak = _peak_dbfs(x2)
        rms = _rms_dbfs(x2)
        return x2, {
            "input_channels": 1,
            "correlation": None,
            "antiphase_detected": False,
            "peak_dbfs_before": peak,
            "peak_dbfs_after": peak,
            "rms_dbfs_before": rms,
            "rms_dbfs_after": rms,
        }

    peak_before = _peak_dbfs(x2)
    rms_before = _rms_dbfs(x2)

    folded = np.mean(x2, axis=1)
    mono = folded.reshape(-1, 1)

    correlation = _mean_pairwise_correlation(x2)
    antiphase_detected = correlation is not None and correlation <= _ANTIPHASE_CORRELATION_THRESHOLD

    return mono, {
        "input_channels": n_channels,
        "correlation": correlation,
        "antiphase_detected": antiphase_detected,
        "peak_dbfs_before": peak_before,
        "peak_dbfs_after": _peak_dbfs(mono),
        "rms_dbfs_before": rms_before,
        "rms_dbfs_after": _rms_dbfs(mono),
    }
