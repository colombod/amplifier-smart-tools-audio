"""Shared rendering harness for `aud.dsp.collision` tests.

Every acceptance test in `test_dsp_collision_acceptance.py` measures the
RENDERED output, not the internally-computed gain curve (see `AGENTS.md`'s
own citation of this exact anti-pattern in issue #9's acceptance text). This
module is the one place that does the STFT analyse -> per-band energy ->
`collision_gains` -> per-bin gain -> ISTFT round trip, so every acceptance
test exercises the SAME real pipeline rather than five slightly different
hand-rolled ones.

Not shipped under `src/`: this is test-only scaffolding for a step whose
own scope is explicitly "the math, given two band-energy arrays" (see
`aud/dsp/collision.py`'s module docstring) -- wiring per-band gains into a
real render pipeline with ballistics belongs to Steps 7/8, not here. This
harness applies the RAW, un-smoothed per-frame gain directly (no
attack/release), which is deliberately cruder than any real product path
would be -- exactly enough to prove the measure's own math on real audio,
no more.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from aud.dsp import stft
from aud.dsp.bands import band_edges, band_energy, bin_band_weights

N_FFT = 2048
HOP = 512
SR = 48000


def peaq_bands(n_bands: int = 32, f_min: float = 20.0, f_max: float = 20000.0) -> dict:
    return band_edges(n_bands, scale="bark_peaq", f_min=f_min, f_max=f_max, allow_extrapolation=True)


def white_noise(rng: np.random.Generator, duration_s: float, sr: int = SR) -> np.ndarray:
    return rng.standard_normal(int(duration_s * sr))


def band_limited_noise(
    rng: np.random.Generator,
    duration_s: float,
    f_lo: float,
    f_hi: float,
    sr: int = SR,
) -> np.ndarray:
    """Zero-phase Butterworth-bandpassed white noise in `[f_lo, f_hi]` Hz."""
    x = white_noise(rng, duration_s, sr=sr)
    nyq = sr / 2.0
    lo = max(f_lo / nyq, 1e-4)
    hi = min(f_hi / nyq, 0.999)
    sos = signal.butter(4, [lo, hi], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, x)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.5
    return y


def _calibration_constant(window: str, n_fft: int) -> float:
    """`aud.dsp.masking`'s calibration ("Calibration" section of that
    module's own docstring) assumes `aud.dsp.bands.band_energy`'s output is
    NORMALISED: a full-scale (amplitude 1.0) sinusoid, EXACTLY BIN-ALIGNED,
    reads a raw per-bin power of `(sum(window))**2 / 4` at its own bin --
    the standard single-sided-DFT-of-a-real-sinusoid identity for a window
    applied to `ShortTimeFFT`'s own (unnormalised) transform convention.
    Measured directly this session (a k=64, f0=1500 Hz bin-aligned
    reference tone, `n_fft=2048`, Hann window): raw bin power at k=64 was
    262144.0 to nine significant figures against the closed form below --
    not an approximation.

    Dividing every raw `band_energy` reading by this constant is what puts
    `analyze_band_energy`'s output on the same "0 dBFS == 1.0" convention
    `aud.dsp.masking`'s `playback_level_db_spl` calibration assumes. Without
    it, this session measured `collision_gains` computing masking levels
    upward of 120 dB SPL for perfectly ordinary -6 dBFS test material
    (`band_energy`'s raw, window-gain-scaled units are nowhere near a
    0-dBFS-normalised convention on their own) -- which pushed the PEAQ
    upper slope's level-dependent term far past any realistic playback
    level and produced absurd, unrealistically wide masking spread.
    """
    win = signal.get_window(window, n_fft)
    return float(np.sum(win)) ** 2 / 4.0


def analyze_band_energy(
    x: np.ndarray, bands: dict, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns `(band_power, spectrum, weights)`:
    `band_power` is `(n_bands, n_frames)` linear power, NORMALISED to the
    "0 dBFS == 1.0" convention `aud.dsp.masking` assumes (see
    `_calibration_constant`) -- this is what should be handed to
    `collision_gains`.
    `spectrum` is the RAW (un-normalised) complex STFT `(n_bins, n_frames)`
    -- `apply_band_gain` multiplies this directly; the dimensionless gain
    `collision_gains` returns does not care which scale the spectrum it is
    applied to is on.
    `weights` is `(n_bands, n_bins)` from `bin_band_weights`.
    """
    spectrum = stft.analyze(x, sr, n_fft=n_fft, hop=hop)
    weights = bin_band_weights(bands, n_fft, sr)
    power = np.abs(spectrum) ** 2
    band_power = band_energy(power, weights) / _calibration_constant(window, n_fft)
    return band_power, spectrum, weights


def apply_band_gain(spectrum: np.ndarray, gain: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Projects a `(n_bands, n_frames)` LINEAR-POWER gain back onto STFT
    bins via `weights` (the same triangular partition-of-unity `bin_band_
    weights` builds), then applies it as an AMPLITUDE multiplier -- the
    square root of the power gain -- to the complex spectrum.

    `bin_gain[k, frame] = sum_b weights[b, k] * gain[b, frame]`: since
    `weights` sums to 1 over `b` for every bin `k` (the partition-of-unity
    property `aud.dsp.bands.bin_band_weights` documents and tests), this is
    a weighted average of the (possibly few) bands overlapping that bin --
    a smooth interpolation back to bin resolution, not a hard per-band step.
    """
    bin_gain_power = weights.T @ gain  # (n_bins, n_frames)
    bin_gain_amplitude = np.sqrt(np.maximum(bin_gain_power, 0.0))
    return spectrum * bin_gain_amplitude


def resynthesize(spectrum: np.ndarray, length: int, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    return stft.resynthesize(spectrum, sr, length, n_fft=n_fft, hop=hop)


def band_level_db(band_power: np.ndarray) -> np.ndarray:
    """Mean-over-frames per-band power, in dB (10*log10), with a floor to
    keep silence finite."""
    mean_power = np.mean(band_power, axis=1)
    return 10.0 * np.log10(np.maximum(mean_power, 1e-20))
