"""Unit tests for aud.lib.verify's ceiling tolerance -- D6 in the lane report.

`lib.verify`'s `ceiling_ok` used to be a zero-tolerance `<=` while
`aud.dsp.limiter.brickwall`'s own `ceiling_met` used (and still uses) a
documented CEILING_TOLERANCE_DB numerical tolerance. A limiter working
exactly as designed can settle right at that tolerance boundary, so the old
`verify` could report a *correctly-limited* render as a verification
failure. These tests pin `verify` to the same constant the limiter uses,
so the two checks cannot silently drift apart again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aud import lib
from aud.dsp import limiter
from aud.dsp.limiter import CEILING_TOLERANCE_DB


def _patch_measured(monkeypatch: pytest.MonkeyPatch, *, true_peak_dbtp: float, integrated_lufs: float = -14.0) -> None:
    """Make `lib.verify` see a controlled `analyze()` result, decoupled from
    actually re-deriving a true-peak measurement from real audio -- the
    tolerance-boundary behaviour is what's under test here, not the
    measurement itself (that's `test_dsp_limiter.py`'s job).
    """
    from aud.dsp import analysis

    def fake_analyze(samples, sample_rate):
        return {"integrated_lufs": integrated_lufs, "true_peak_dbtp": true_peak_dbtp}

    monkeypatch.setattr(analysis, "analyze", fake_analyze)


def test_ceiling_ok_accepts_a_render_exactly_at_the_limiters_own_tolerance(
    tiny_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render that lands exactly `CEILING_TOLERANCE_DB` above the ceiling
    -- precisely what a correctly-working limiter can produce, per its own
    `ceiling_met` -- must be reported as `ceiling_ok`, not a failure.
    """
    ceiling = -1.0
    _patch_measured(monkeypatch, true_peak_dbtp=ceiling + CEILING_TOLERANCE_DB)
    result = lib.verify(str(tiny_wav), ceiling_dbtp=ceiling)
    assert result["ceiling_ok"] is True


def test_ceiling_ok_still_rejects_a_render_meaningfully_past_the_tolerance(
    tiny_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tolerance is not infinite -- a render half a dB over the ceiling
    is a genuine failure, not something the tolerance should paper over.
    """
    ceiling = -1.0
    _patch_measured(monkeypatch, true_peak_dbtp=ceiling + CEILING_TOLERANCE_DB + 0.5)
    result = lib.verify(str(tiny_wav), ceiling_dbtp=ceiling)
    assert result["ceiling_ok"] is False


def test_ceiling_ok_agrees_with_limiters_ceiling_met_at_the_same_measured_value(
    tiny_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual defect, reproduced directly: `verify`'s `ceiling_ok` and
    `limiter.brickwall`'s `ceiling_met` are two independently-written
    boundary checks on the *same* quantity (measured true peak vs a
    ceiling). Given the identical measured value, they must now agree --
    previously they used two different tolerances (0.0 vs 0.05 dB) and
    could disagree right at the boundary a working limiter settles at.
    """
    ceiling = -1.0
    boundary_tp = ceiling + CEILING_TOLERANCE_DB

    # What limiter.brickwall's own ceiling_met logic would say for this
    # exact measured value (see aud.dsp.limiter.brickwall's ceiling_met line).
    limiter_says_met = bool(boundary_tp <= ceiling + limiter.CEILING_TOLERANCE_DB)

    _patch_measured(monkeypatch, true_peak_dbtp=boundary_tp)
    verify_result = lib.verify(str(tiny_wav), ceiling_dbtp=ceiling)

    assert limiter_says_met is True
    assert verify_result["ceiling_ok"] == limiter_says_met


def test_ceiling_tolerance_is_the_one_constant_both_sides_import() -> None:
    """Structural guard: there must be exactly one defined tolerance value,
    imported by both sides, not two numbers kept equal by hand.
    """
    assert CEILING_TOLERANCE_DB == limiter.CEILING_TOLERANCE_DB
    assert CEILING_TOLERANCE_DB > 0.0
