"""Regression tests for `octave_band_analysis`'s reference-anchored tonal signal.

Guards the fix to `advise`'s tonal diagnosis: comparing each octave band to
THIS FILE'S OWN MEDIAN (`rel_median_db`) is anchored to a quantity the very
defect being diagnosed can move, and is dominated by the material's natural
spectral shape -- not by any actual defect. Measured directly here, on six
controlled induced-defect fixtures: taking the largest `|rel_median_db|`
identifies the right band AND direction in only 1 of 6 cases (see
`test_old_statistic_scores_exactly_one_of_six`, a DELIBERATE negative
control -- it must never silently start passing more often, or `advise`'s
old, weaker anchor has quietly become the primary signal again).

`rel_reference_db` -- this band's energy relative to a clean REFERENCE
file, anchored by the MEDIAN OF THE PER-BAND DELTAS rather than either
file's own median -- gets 6 of 6 on the same fixtures.

This is deterministic DSP, so it is regression-tested here for free, with
no model call -- see `test_reference_anchored_signal_scores_six_of_six`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from aud.dsp import analysis

SR = 48000
SEED = 7
_DEFECT_PRESENCE_TOLERANCE_DB = 1.0  # generous: every measured effect below is >= ~4 dB


def _source_material(seconds: float = 3.0, sr: int = SR, seed: int = SEED) -> np.ndarray:
    """Common source material for all six fixtures below.

    Pink-ish noise (a 1st-order 200 Hz lowpass on white noise, cascaded
    three times for a steeper low-end tilt, plus 0.3x the same white noise
    so the spectrum never gets too thin at the top end) summed with
    harmonics of 110 Hz through 3520 Hz, peak-normalised to 0.45 for
    headroom before any defect is applied.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    white = rng.standard_normal(n)
    sos = signal.butter(1, 200.0, btype="lowpass", fs=sr, output="sos")
    lowpassed = white
    for _ in range(3):
        lowpassed = signal.sosfilt(sos, lowpassed)
    pink_ish = lowpassed + 0.3 * white
    t = np.arange(n) / sr
    harmonics = np.zeros(n)
    for freq in (110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0):
        harmonics += np.sin(2 * np.pi * freq * t)
    mono = pink_ish + harmonics
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.45
    return mono.astype(np.float64)


def _peak_limit(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Peak-limit ONLY if it would clip -- a blanket renormalisation here
    would rescale away the very gain difference the defect is supposed to
    introduce."""
    peak = np.max(np.abs(x))
    if peak > ceiling:
        return x / peak * ceiling
    return x


def _to_stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


def _peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ audio-eq-cookbook peaking filter -- a plain biquad, independent
    of the production `dsp.eq`/`dsp.eqmatch` stages: a fixture must not
    depend on the code under test to construct itself."""
    amp = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * amp
    b1 = -2 * cos_w0
    b2 = 1 - alpha * amp
    a0 = 1 + alpha / amp
    a1 = -2 * cos_w0
    a2 = 1 - alpha / amp
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, x)


def _low_shelf_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    """RBJ audio-eq-cookbook low-shelf filter."""
    amp = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_amp = np.sqrt(amp)
    b0 = amp * ((amp + 1) - (amp - 1) * cos_w0 + 2 * sqrt_amp * alpha)
    b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_w0)
    b2 = amp * ((amp + 1) - (amp - 1) * cos_w0 - 2 * sqrt_amp * alpha)
    a0 = (amp + 1) + (amp - 1) * cos_w0 + 2 * sqrt_amp * alpha
    a1 = -2 * ((amp - 1) + (amp + 1) * cos_w0)
    a2 = (amp + 1) + (amp - 1) * cos_w0 - 2 * sqrt_amp * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, x)


def _lowpass(x: np.ndarray, sr: int, freq: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, freq, btype="lowpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


# name -> (fix_direction, expected_top1_hz, induced_check_freq, induced signal)
#
# `fix_direction` is what a CORRECT diagnosis should recommend to UNDO the
# induced defect: "cut" when the defect ADDED energy, "boost" when it
# REMOVED energy -- matching `reference_anchored_band_deviation_db`'s own
# sign convention (positive delta => candidate for a cut).
#
# `induced_check_freq` is the actual frequency the defect was built around
# (not necessarily the expected octave-band centre) -- used only by
# `test_defects_are_actually_present...` to confirm the fixture really
# moved the spectrum before any detector is asked to find it.
def _build_defects(clean: np.ndarray) -> dict[str, tuple[str, float, float, np.ndarray]]:
    return {
        "boxy": ("cut", 500.0, 400.0, _peaking_eq(clean, SR, 400.0, 6.0, 1.2)),
        "rumbly": ("cut", 31.5, 80.0, _low_shelf_eq(clean, SR, 80.0, 9.0)),
        "dull": ("boost", 16000.0, 10000.0, _lowpass(clean, SR, 5000.0, order=4)),
        "harsh": ("cut", 8000.0, 6000.0, _peaking_eq(clean, SR, 6000.0, 6.0, 1.2)),
        "muddy": ("cut", 250.0, 250.0, _peaking_eq(clean, SR, 250.0, 6.0, 0.9)),
        "thin": ("boost", 250.0, 200.0, _peaking_eq(clean, SR, 200.0, -8.0, 0.8)),
    }


CLEAN_MONO = _source_material()
DEFECTS = _build_defects(CLEAN_MONO)


def test_defects_are_actually_present_before_anything_tries_to_find_them():
    """Confirm each induced defect really moved the spectrum (Welch PSD
    delta vs the clean reference, same technique as
    `test_dsp_crossover.py`) before any detector is asked to find it --
    never assert a detector works on a defect that has not been confirmed
    to exist."""
    freqs, psd_clean = signal.welch(CLEAN_MONO, fs=SR, nperseg=8192)
    for name, (direction, _expected_hz, check_freq, defect) in DEFECTS.items():
        _, psd_defect = signal.welch(_peak_limit(defect), fs=SR, nperseg=8192)
        ratio_db = 10 * np.log10((psd_defect + 1e-20) / (psd_clean + 1e-20))
        idx = int(np.argmin(np.abs(freqs - check_freq)))
        window = ratio_db[max(0, idx - 3) : idx + 4]
        measured = float(np.mean(window))
        print(f"\n[reference_anchored] {name}: PSD ratio near {check_freq} Hz = {measured:.2f} dB (expect {direction})")
        if direction == "cut":
            assert measured > _DEFECT_PRESENCE_TOLERANCE_DB, (
                f"{name}: expected the induced defect to ADD energy near {check_freq} Hz, measured {measured:.2f} dB"
            )
        else:
            assert measured < -_DEFECT_PRESENCE_TOLERANCE_DB, (
                f"{name}: expected the induced defect to REMOVE energy near {check_freq} Hz, measured {measured:.2f} dB"
            )


def _top1(entries: list[dict], key: str) -> tuple[float, str] | None:
    """This test's own read of 'the largest |value|, and its sign' -- the
    same rule the old, disproven approach used (take the single biggest
    deviation), applied here to whichever anchor `key` names."""
    scored = [(entry["hz"], entry[key]) for entry in entries if entry.get(key) is not None]
    if not scored:
        return None
    hz, value = max(scored, key=lambda pair: abs(pair[1]))
    return hz, ("cut" if value > 0 else "boost")


def test_reference_anchored_signal_scores_six_of_six():
    """The new, reference-anchored statistic: correct band AND direction
    on all six induced-defect fixtures."""
    reference_stereo = _to_stereo(_peak_limit(CLEAN_MONO))
    hits = 0
    for name, (direction, expected_hz, _check_freq, defect) in DEFECTS.items():
        defect_stereo = _to_stereo(_peak_limit(defect))
        result = analysis.analyze(defect_stereo, SR, reference_x=reference_stereo, reference_sr=SR)
        picked = _top1(result["octave_band_analysis"], "rel_reference_db")
        assert picked is not None, f"{name}: no rel_reference_db entries at all"
        hz, picked_direction = picked
        ok = hz == expected_hz and picked_direction == direction
        print(
            f"[reference_anchored] {name}: NEW top-1 = ({hz} Hz, {picked_direction}), "
            f"expected ({expected_hz} Hz, {direction}), ok={ok}"
        )
        hits += int(ok)
    assert hits == 6, f"expected 6 of 6 correct band+direction picks with rel_reference_db, got {hits}"


def test_old_statistic_scores_exactly_one_of_six():
    """DELIBERATE negative control -- the number this test pins is a
    measured fact about `rel_median_db` on these exact six fixtures, not
    an aspiration. Without this control, a future change could silently
    make `rel_median_db` (anchored to THIS FILE's own median, which the
    defect itself moves) the primary tonal signal again with no reference
    present, and this suite would stay green throughout -- nothing else
    here would catch that regression."""
    hits = 0
    for name, (direction, expected_hz, _check_freq, defect) in DEFECTS.items():
        defect_stereo = _to_stereo(_peak_limit(defect))
        result = analysis.analyze(defect_stereo, SR)
        picked = _top1(result["octave_band_analysis"], "rel_median_db")
        assert picked is not None, f"{name}: no rel_median_db entries at all"
        hz, picked_direction = picked
        ok = hz == expected_hz and picked_direction == direction
        print(
            f"[reference_anchored] {name}: OLD top-1 = ({hz} Hz, {picked_direction}), "
            f"expected ({expected_hz} Hz, {direction}), ok={ok}"
        )
        hits += int(ok)
    assert hits == 1, (
        f"expected the OLD (file's-own-median) statistic to score exactly 1 of 6 on these fixtures, got {hits}"
    )


def test_rel_reference_db_is_absent_without_a_reference():
    """A caller must be able to tell 'not computed' from 'computed as
    zero' -- the field itself must not appear when no reference was
    supplied, not merely read as null."""
    defect_stereo = _to_stereo(_peak_limit(DEFECTS["boxy"][3]))
    result = analysis.analyze(defect_stereo, SR)
    entries = result["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    for entry in entries:
        assert "rel_reference_db" not in entry, (
            "rel_reference_db must be ABSENT (not null) when no reference is supplied"
        )


def test_rel_reference_db_is_present_on_every_entry_with_a_reference():
    reference_stereo = _to_stereo(_peak_limit(CLEAN_MONO))
    defect_stereo = _to_stereo(_peak_limit(DEFECTS["boxy"][3]))
    result = analysis.analyze(defect_stereo, SR, reference_x=reference_stereo, reference_sr=SR)
    entries = result["octave_band_analysis"]
    assert entries, "expected at least one band entry"
    assert all("rel_reference_db" in entry for entry in entries), (
        "rel_reference_db must be present (key-wise) on every entry once a reference is supplied"
    )


def test_reference_anchored_band_deviation_db_requires_matching_lengths():
    """A misaligned pair of band-energy arrays is a programming-contract
    violation, not a user-facing error -- `dsp/` modules raise no
    user-facing errors (AGENTS.md #8)."""
    with pytest.raises(ValueError, match="same length"):
        analysis.reference_anchored_band_deviation_db([1.0, 2.0], [1.0])
