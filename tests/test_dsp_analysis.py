"""Tests for `aud.dsp.analysis.analyze` -- the loudness-measurement honesty contract.

Guards the spec-adherence finding: a bare `except Exception` around the LRA
measurement used to swallow ANY exception, including a genuine bug, and
report `loudness_range_lu: null` with no indication why -- indistinguishable
from the one condition that IS a legitimate partial outcome (audio shorter
than BS.1770's gating block). These tests pin: (1) normal material measures
both fields normally, (2) genuinely-too-short audio is a documented, reasoned
partial outcome, never a crash, and (3) an unrelated bug is no longer
swallowed -- it propagates, because the precondition check replaced the
`except Exception`, not narrowed it.
"""

from __future__ import annotations

import numpy as np
import pytest

from aud.dsp import analysis

SR = 48000


def _varied_loudness_signal(seconds: float = 6.0, sr: int = SR) -> np.ndarray:
    """Quiet then loud sections, so LRA is genuinely non-zero -- a flat
    signal would pass the "no exception" assertion for the wrong reason
    (LRA of a constant-loudness signal is legitimately 0.0, same as the
    too-short case, so this would not distinguish the two)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    quiet = 0.02 * np.sin(2 * np.pi * 300 * t[: n // 2])
    loud = 0.5 * np.sin(2 * np.pi * 300 * t[n // 2 :])
    return np.stack([np.concatenate([quiet, loud])] * 2, axis=1)


def test_normal_material_measures_both_loudness_fields_with_no_reason_key():
    result = analysis.analyze(_varied_loudness_signal(), SR)
    assert isinstance(result["integrated_lufs"], float)
    assert isinstance(result["loudness_range_lu"], float)
    assert result["loudness_range_lu"] > 0.0, "a quiet-then-loud signal must show a non-zero range"
    assert "loudness_unavailable_reason" not in result


def test_audio_shorter_than_bs1770_block_is_a_documented_partial_outcome_not_a_crash():
    """0.1s is below pyloudnorm's 0.4s gating block -- a real, reproducible
    condition (any short clip), not a hypothetical."""
    short = np.zeros((int(0.1 * SR), 2))
    short[:, 0] = 0.1  # non-silent, so this isn't ALSO exercising a silence edge case
    result = analysis.analyze(short, SR)
    assert result["integrated_lufs"] is None
    assert result["loudness_range_lu"] is None
    assert "loudness_unavailable_reason" in result
    assert "0.1" in result["loudness_unavailable_reason"] or "BS.1770" in result["loudness_unavailable_reason"]
    # Every other measurement still runs normally -- a partial outcome for
    # loudness only, not the whole call failing.
    assert isinstance(result["sample_peak_dbfs"], float)
    assert isinstance(result["true_peak_dbtp"], float)


def test_a_genuine_bug_in_loudness_measurement_is_no_longer_swallowed(monkeypatch: pytest.MonkeyPatch):
    """Regression guard for the defect itself: before the fix, ANY exception
    from the loudness measurement (a real bug, not just a too-short-input
    condition) was caught by a bare `except Exception` and silently turned
    into `loudness_range_lu: null`. Proven here by injecting an unrelated
    bug (AttributeError, not the documented precondition) and asserting it
    now propagates instead of being reported as a clean partial result."""

    def _broken(*_args, **_kwargs):
        raise AttributeError("boom -- an unrelated programming bug, not a short-input condition")

    monkeypatch.setattr(analysis._loudness, "integrated_lufs", _broken)
    with pytest.raises(AttributeError, match="boom"):
        analysis.analyze(_varied_loudness_signal(), SR)
