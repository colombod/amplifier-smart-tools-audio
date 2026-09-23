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


# --- `octave_band_analysis` -- the derived, comparison-ready view ----------
#
# Guards the aud-expertise finding: `octave_band_energy_db`'s dict keys sort
# LEXICOGRAPHICALLY once JSON-serialized with `sort_keys=True` (exactly what
# `intelligence.prompts.user_prompt` does), so a naive "compare to the next
# entry" walk compares 125 Hz against 1000 Hz and 16000 Hz against 2000 Hz.
# `octave_band_analysis` is a LIST (immune to that resorting) explicitly
# ordered low-to-high by frequency, carrying two derived, already-computed
# comparisons so the reader is never asked to do the arithmetic itself.


def _pink_noise(n: int, seed: int = 0) -> np.ndarray:
    """Mono pink noise (equal energy PER OCTAVE, ~-3 dB/octave PSD slope).

    White noise has a FLAT power spectral density, but an octave band's
    absolute bandwidth grows linearly with its centre frequency (each band
    spans center/sqrt(2)..center*sqrt(2), a constant RATIO), so white noise
    measures MORE energy in higher octave bands -- a real, physical +3
    dB/octave trend, not an artefact. Pink noise (1/f power, 1/sqrt(f)
    amplitude) cancels that growth and is the correct "flat across octave
    bands" control -- the same reason aud-expertise's own fixtures used
    pink, not white, noise.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid a division by zero at DC
    pink_spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spectrum, n)
    peak = np.max(np.abs(pink))
    return pink / peak if peak > 0 else pink


def _flat_signal(seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    """Pink-noise flat-spectrum control: every octave band should sit close
    to every other one, so `rel_median_db` should read near zero
    everywhere -- the fixed prompt uses this to decide "nothing to fix"."""
    n = int(seconds * sr)
    mono = 0.2 * _pink_noise(n, seed=0)
    return np.stack([mono, mono], axis=1)


def test_octave_band_analysis_is_a_numerically_ordered_list_not_a_dict():
    """The structural fix: element order is low-to-high by centre frequency,
    which JSON's `sort_keys=True` cannot disturb (it only reorders dict
    keys, never list elements)."""
    result = analysis.analyze(_flat_signal(), SR)
    entries = result["octave_band_analysis"]
    assert isinstance(entries, list)
    hz_values = [entry["hz"] for entry in entries]
    assert hz_values == sorted(hz_values), "entries must already be ordered low-to-high by frequency"
    assert hz_values == list(analysis._OCTAVE_CENTERS)


def test_octave_band_analysis_rel_median_db_is_near_zero_on_a_flat_control():
    """A flat-ish spectrum has no band deviating from its own median --
    every `rel_median_db` should be small. This is the deterministic half
    of the "clean material must not read as a defect" guard: if the
    report itself claims a large deviation on a flat control, no prompt
    wording can be trusted to say "propose nothing" from it."""
    result = analysis.analyze(_flat_signal(), SR)
    entries = result["octave_band_analysis"]
    deviations = [abs(e["rel_median_db"]) for e in entries if e["rel_median_db"] is not None]
    assert deviations, "expected at least one measurable band"
    assert max(deviations) < 2.0, f"flat control should show near-zero relative deviation, got {deviations}"


def test_octave_band_analysis_rel_median_db_finds_a_narrow_bell_defect_correctly_signed():
    """A +6 dB bell at 500 Hz on top of flat noise must show up as a
    clearly positive `rel_median_db` at the 500 Hz entry -- correct band,
    correct sign. This is the deterministic analogue of the aud-expertise
    finding that a model, given only ten raw absolutes, inverted the sign
    of a relative comparison 3/3 on a real induced defect."""
    sr = SR
    base = _flat_signal(seconds=3.0, sr=sr)
    t = np.arange(base.shape[0]) / sr
    bell = 0.4 * np.sin(2 * np.pi * 500 * t)
    x = base + np.stack([bell, bell], axis=1)
    result = analysis.analyze(x, sr)
    by_hz = {e["hz"]: e for e in result["octave_band_analysis"]}
    assert by_hz[500.0]["rel_median_db"] > 3.0, by_hz[500.0]
    # The bands well away from the induced bell should not read as elevated.
    assert by_hz[31.5]["rel_median_db"] < 1.0, by_hz[31.5]
    assert by_hz[16000.0]["rel_median_db"] < 1.0, by_hz[16000.0]


def test_octave_band_analysis_neighbour_contrast_has_the_documented_blind_spot():
    """The secondary metric's documented weakness, reproduced deterministically:
    a defect spanning two ADJACENT bands cancels out in `neighbour_contrast_db`,
    because each depressed (or elevated) band's neighbour is the other one --
    while `rel_median_db` still shows the true deviation on both. This is
    exactly the failure mode measured on the 'thin' fixture in project
    history (neighbour-contrast read -1.51 dB against a real -3.4 dB delta).
    """
    sr = SR
    base = _flat_signal(seconds=3.0, sr=sr)
    t = np.arange(base.shape[0]) / sr
    # Boost two ADJACENT octave bands (125 Hz and 250 Hz) together.
    bump = 0.4 * np.sin(2 * np.pi * 125 * t) + 0.4 * np.sin(2 * np.pi * 250 * t)
    x = base + np.stack([bump, bump], axis=1)
    result = analysis.analyze(x, sr)
    by_hz = {e["hz"]: e for e in result["octave_band_analysis"]}
    # rel_median_db still sees both elevated bands clearly.
    assert by_hz[125.0]["rel_median_db"] > 3.0, by_hz[125.0]
    assert by_hz[250.0]["rel_median_db"] > 3.0, by_hz[250.0]
    # neighbour_contrast_db is blind to it: each band's neighbour is the
    # other elevated band, so the contrast collapses toward zero -- far
    # smaller than the true, rel_median_db-reported deviation.
    assert abs(by_hz[125.0]["neighbour_contrast_db"]) < by_hz[125.0]["rel_median_db"]
    assert abs(by_hz[250.0]["neighbour_contrast_db"]) < by_hz[250.0]["rel_median_db"]
