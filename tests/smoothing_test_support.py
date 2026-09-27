"""Shared rendering harness for `aud.dsp.smoothing` tests -- mirrors
`collision_test_support.py`'s own rule (see its docstring, and `AGENTS.md`'s
citation of issue #9's acceptance text): every measurement is taken on the
RENDERED signal, never on the internally-computed gain curve alone.

Not shipped under `src/`: test-only scaffolding, same status as
`collision_test_support.py`.
"""

from __future__ import annotations

import numpy as np

from aud.dsp import stft
from aud.dsp.smoothing import duck_gain_surface, smooth_frequency_axis_db

N_FFT = 2048
HOP = 512
SR = 48000


def apply_bin_gain_db_and_render(x: np.ndarray, bin_gain_db: np.ndarray, sr: int = SR) -> np.ndarray:
    """Analyse `x`, multiply its spectrum by `10**(bin_gain_db/20)` (an
    AMPLITUDE multiplier -- `bin_gain_db` is in the same power-dB
    convention `aud.dsp.smoothing` uses throughout), resynthesise, return
    the rendered signal. `bin_gain_db` may be `(n_bins,)` (applied
    identically to every frame) or `(n_bins, n_frames)`.
    """
    spectrum = stft.analyze(x, sr, n_fft=N_FFT, hop=HOP)
    bin_gain_amplitude = 10.0 ** (np.asarray(bin_gain_db, dtype=np.float64) / 20.0)
    if bin_gain_amplitude.ndim == 1:
        bin_gain_amplitude = bin_gain_amplitude[:, None]
    y = stft.resynthesize(spectrum * bin_gain_amplitude, sr, len(x), n_fft=N_FFT, hop=HOP)
    return y


def render_with_duck_gain_surface(
    x: np.ndarray,
    band_gain_power: np.ndarray,
    bands: dict,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    **smoothing_kwargs,
) -> tuple[np.ndarray, dict]:
    """`duck_gain_surface` -> apply to `x`'s spectrum -> resynthesise.

    Returns (y, stats) -- `stats` is `duck_gain_surface`'s own return.
    """
    bin_gain_amplitude, stats = duck_gain_surface(band_gain_power, bands, n_fft, hop, sr, **smoothing_kwargs)
    spectrum = stft.analyze(x, sr, n_fft=n_fft, hop=hop)
    y = stft.resynthesize(spectrum * bin_gain_amplitude, sr, len(x), n_fft=n_fft, hop=hop)
    return y, stats


def comb_bin_gain_db(n_bins: int, period: int, depth_db: float) -> np.ndarray:
    """A deliberately frequency-DISCONTINUOUS per-bin dB gain: alternating
    blocks of `period` bins at 0 dB / `-depth_db` dB, a hard step every
    `period` bins -- the "deliberately made discontinuous across
    frequency" surface issue #18's acceptance criterion 1 asks for.
    """
    pattern = np.zeros(n_bins, dtype=np.float64)
    block = (np.arange(n_bins) // period) % 2 == 1
    pattern[block] = -depth_db
    return pattern


def frame_level_db(y: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Per-frame RMS level of `y`, in dBFS, via the same STFT framing
    (so the frame axis lines up with a gain surface's own frame axis).
    """
    spectrum = stft.analyze(y, sr, n_fft=n_fft, hop=hop)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    return 10.0 * np.log10(np.maximum(power, 1e-20))


def smoothed_comb_bin_gain_db(
    n_bins: int, period: int, depth_db: float, bands: dict, n_fft: int, sr: int
) -> np.ndarray:
    """`comb_bin_gain_db` run through `smooth_frequency_axis_db`."""
    raw = comb_bin_gain_db(n_bins, period, depth_db)
    return smooth_frequency_axis_db(raw, bands, n_fft, sr)
