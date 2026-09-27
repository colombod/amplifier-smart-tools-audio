"""Time-aliasing measurements for `aud.dsp.smoothing`'s frequency-axis
(ERB) smoothing -- issue #18's "reason (b)" (a sharp gain edge in
frequency circularly convolves into the frame's own time-domain content).

Both measurements are this session's OWN, independently reproduced numbers
against this repo's actual STFT defaults (`n_fft=2048`, `hop=512`, Hann
analysis + Hann synthesis, WOLA -- see `aud.dsp.stft`), not a copy of the
issue text's own reference table (though they land in the same regime).
"""

from __future__ import annotations

import numpy as np

from aud.dsp import bands, stft
from aud.dsp.smoothing import smooth_frequency_axis_erb_db

N_FFT = 2048
SR = 48000


def _energy_beyond_half_nfft(gain_db: np.ndarray, m: int, n: int) -> float:
    """Fraction (in dB) of a real, non-negative gain's impulse-response
    energy that falls OUTSIDE the central `[-n/2, n/2)` sample window,
    measured at oversampled frequency resolution `m` bins (`m >> n`) so the
    kernel's TRUE (non-circularly-wrapped) decay is resolved -- see this
    module's own inline derivation in the test that calls it.
    """
    amplitude = 10.0 ** (gain_db / 20.0)
    h = np.fft.irfft(amplitude, n=m)
    h = np.fft.fftshift(h)
    total = float(np.sum(h**2))
    center = m // 2
    within = float(np.sum(h[center - n // 2 : center + n // 2] ** 2))
    beyond = total - within
    return float(10.0 * np.log10(max(beyond, 1e-30) / total))


def test_smoothed_edge_reduces_time_domain_aliasing_vs_brick_wall():
    """A brick-wall 0/-60 dB gain edge at 1 kHz, evaluated at 32x this
    repo's own bin resolution (`n_fft=2048`) to reveal its TRUE impulse
    response, is compared against the SAME edge run through this module's
    own `smooth_frequency_axis_erb_db`. Measured this session: brick-wall
    leaves ~-26 dB of energy beyond +/-N/2 samples; ERB-smoothed leaves
    ~-45 dB -- an ~19 dB improvement, reproduced here with an asserted
    conservative margin.
    """
    oversample = 32
    m = oversample * N_FFT
    bin_hz = np.arange(m // 2 + 1) * SR / m
    edge_hz = 1000.0
    raw_db = np.where(bin_hz < edge_hz, 0.0, -60.0)

    b = bands.band_edges(32, scale="bark_peaq", f_min=20.0, f_max=20000.0, allow_extrapolation=True)
    smoothed_db = smooth_frequency_axis_erb_db(raw_db, b, n_fft=m, sr=SR)

    beyond_raw = _energy_beyond_half_nfft(raw_db, m, N_FFT)
    beyond_smoothed = _energy_beyond_half_nfft(smoothed_db, m, N_FFT)

    assert beyond_raw > -35.0, f"test setup: brick-wall edge should leak substantially; got {beyond_raw:.1f} dB"
    assert beyond_smoothed < -35.0, f"ERB-smoothed edge should leak much less; got {beyond_smoothed:.1f} dB"
    assert beyond_smoothed < beyond_raw - 10.0, (
        f"expected at least a 10 dB improvement; brick_wall={beyond_raw:.1f} dB, smoothed={beyond_smoothed:.1f} dB"
    )


def test_stft_frame_step_already_smooths_over_one_to_two_hops():
    """A single bin's hard step from 0 dB to -40 dB between two adjacent
    STFT frames (this repo's real `analyze`/`resynthesize` pair, not an
    idealised filter) resynthesises as a time-domain crossfade spanning
    roughly one to two hops, not an instantaneous step -- the physical
    fact `duck_gain_surface`'s hop-resolution error is protecting a caller
    from silently missing.
    """
    hop = 512
    n_fft = N_FFT
    sr = SR
    duration_s = 1.0
    x = np.ones(int(duration_s * sr))  # a constant (DC-free-ish) probe: use a tone instead for a real bin

    # Use a bin-aligned sinusoid instead of DC so the affected bin actually carries energy.
    bin_index = 100
    freq_hz = bin_index * sr / n_fft
    t = np.arange(len(x)) / sr
    x = np.sin(2 * np.pi * freq_hz * t)

    spectrum = stft.analyze(x, sr, n_fft=n_fft, hop=hop)
    n_frames = spectrum.shape[1]
    step_frame = n_frames // 2

    gain = np.ones(n_frames)
    gain[step_frame:] = 10 ** (-40.0 / 20.0)  # step to -40 dB amplitude at step_frame
    spectrum_gained = spectrum * gain[None, :]

    y = stft.resynthesize(spectrum_gained, sr, len(x), n_fft=n_fft, hop=hop)

    # Track the affected bin's envelope directly via a matched-filter (project
    # each frame back onto the probe frequency) rather than re-running the STFT,
    # so we measure the RESYNTHESISED time-domain signal's own smooth transition.
    window = np.hanning(4 * hop)
    envelope = np.abs(np.convolve(y * np.exp(-2j * np.pi * freq_hz * t), window / window.sum(), mode="same"))
    step_sample = step_frame * hop
    # Find the 90% and 10% crossing points (relative to before/after levels) near the step.
    before_level = float(np.mean(envelope[step_sample - 4 * hop : step_sample - 2 * hop]))
    after_level = float(np.mean(envelope[step_sample + 2 * hop : step_sample + 4 * hop]))
    span = before_level - after_level
    assert span > 0, "test setup: envelope must actually drop across the step"

    search = envelope[step_sample - 3 * hop : step_sample + 3 * hop]
    threshold_90 = before_level - 0.1 * span
    threshold_10 = before_level - 0.9 * span
    idx_90 = next((i for i, v in enumerate(search) if v < threshold_90), None)
    idx_10 = next((i for i, v in enumerate(search) if v < threshold_10), None)
    assert idx_90 is not None, "crossfade's 90% crossing did not resolve within the search window"
    assert idx_10 is not None, "crossfade's 10% crossing did not resolve within the search window"

    crossfade_samples = idx_10 - idx_90
    crossfade_ms = crossfade_samples / sr * 1000.0
    hop_ms = hop / sr * 1000.0

    assert crossfade_ms > 0, f"expected a measurable (non-instantaneous) crossfade; got {crossfade_ms:.2f} ms"
    assert crossfade_ms < 3.0 * hop_ms, (
        f"crossfade ({crossfade_ms:.2f} ms) should be on the order of one to two hops "
        f"({hop_ms:.2f} ms), not much longer"
    )
