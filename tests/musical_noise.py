"""Kurtosis-ratio musical-noise proxy (Saruwatari et al.), NOT Step 3's own
metric -- see `aud.dsp.smoothing`'s own module docstring, "Musical-noise
measurement," for why Step 3's proxy (issue #13) does not exist anywhere in
this repo or in `aud-mix` (checked directly, not assumed), and why this
metric is used instead, with its source verified rather than taken on
citation.

Source, fetched and read directly this session (not taken on citation
alone): Hiroshi Saruwatari, Suzumi Kanehara, Ryoichi Miyazaki, Kiyohiro
Shikano, Kazunobu Kondo, "Musical Noise Analysis for Bayesian Minimum
Mean-Square Error Speech Amplitude Estimators", Interspeech 2013,
https://www.isca-archive.org/interspeech_2013/saruwatari13_interspeech.pdf
(also cites the metric's own origin, refs [5,6,7] there).

    kurt = mu_4 / mu_2**2                                  (eq. 1)
    mu_m = integral_0^inf z**m * p(z) dz                   (eq. 2)
    kurtosis_ratio = kurt_proc / kurt_org                  (eq. 3)

where `p(z)` is the p.d.f. of a signal `z` IN THE POWER SPECTRAL DOMAIN --
i.e. `mu_m` is the RAW (non-centered) m-th moment of a collection of
power-spectrum values, not a waveform, and not a centered/"excess"
kurtosis. "This measure increases as the amount of generated musical noise
increases" (quoted directly from the paper's own text surrounding eq. 3).

`kurt_org` is the kurtosis of the OBSERVED (pre-processing) signal;
`kurt_proc` is the kurtosis of the PROCESSED (post-processing) signal --
both measured over the SAME time-frequency region of their respective
power spectrograms, per the paper's own instruction to conduct this
analysis over "a noise-only time-frequency period," here generalised to
"the region under test" since this repo's own acceptance criterion is
about a gain surface applied to a full render, not a speech/noise split.
"""

from __future__ import annotations

import numpy as np


def power_spectrogram_kurtosis(power: np.ndarray) -> float:
    """`mu_4 / mu_2**2` (eq. 1) over every value in `power` treated as one
    pooled sample of the power-spectral p.d.f. -- `power` must already be
    non-negative (a power quantity), matching `p(z)` being a p.d.f. over
    `z >= 0` in the source (eq. 2's own integration bounds, `0` to `inf`).

    Raises:
        ValueError: `power` is empty, contains a negative or non-finite
            value, or its second raw moment is (numerically) zero -- the
            ratio is undefined for a degenerate (all-zero) input.
    """
    power = np.asarray(power, dtype=np.float64).ravel()
    if power.size == 0:
        raise ValueError("power must not be empty")
    if np.any(power < 0.0) or not np.all(np.isfinite(power)):
        raise ValueError("power must be non-negative and finite (it is a power-spectral quantity)")
    mu2 = float(np.mean(power**2))
    mu4 = float(np.mean(power**4))
    if mu2 <= 0.0:
        raise ValueError("power's second raw moment is zero; kurtosis is undefined for all-zero input")
    return mu4 / mu2**2


def kurtosis_ratio(observed_power: np.ndarray, processed_power: np.ndarray) -> float:
    """`kurt(processed) / kurt(observed)` (eq. 3) -- > 1 means the
    processing INCREASED the tonal/isolated-outlier character of the power
    spectrum relative to the observed signal, i.e. generated musical noise;
    closer to 1 (from above) means less musical noise was generated.

    Measures RANDOM isolated-outlier musical noise (Saruwatari et al.'s own
    target), not adjacent-bin smoothness of a DETERMINISTIC band-discontinuous
    surface -- on such deterministic shapes it can rank a smoother,
    interpolated result WORSE than a harder-edged stepped one (measured
    directly this session on the 32-band alternating-0/-18-dB shape in
    `tests/test_dsp_smoothing_acceptance.py`: ~5.2 interpolated vs ~2.6
    stepped), so it must not be used to judge that kind of smoothness
    directly.
    """
    kurt_org = power_spectrogram_kurtosis(observed_power)
    kurt_proc = power_spectrogram_kurtosis(processed_power)
    return kurt_proc / kurt_org
