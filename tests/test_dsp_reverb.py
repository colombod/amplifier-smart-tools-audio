"""Behavioural tests for the FDN reverb and IR convolution reverb."""

from __future__ import annotations

import numpy as np

from aud.dsp import reverb

SR = 44100


def _impulse(n: int = SR * 2) -> np.ndarray:
    """A single-sample impulse followed by silence -- enough room to see a real tail."""
    x = np.zeros((n, 1))
    x[0, 0] = 1.0
    return x


def _energy_envelope(x: np.ndarray, block: int = 512) -> np.ndarray:
    """RMS energy per non-overlapping block, in dB."""
    n_blocks = x.shape[0] // block
    trimmed = x[: n_blocks * block, 0]
    blocks = trimmed.reshape(n_blocks, block)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def _decay_time_s(x: np.ndarray, sr: int, block: int = 512, drop_db: float = 20.0) -> float:
    """RT-style decay time: linear-regression slope of the (dB) energy envelope
    from its peak onward, extrapolated to `drop_db` of decay.

    A raw "first block that crossed -20 dB relative to the peak" measure is
    unstable for an FDN: the diffuse field often BUILDS for a few
    milliseconds after the direct impulse (multiple reflections arriving
    and summing) before it starts to fall, so the true peak can sit well
    into the response and a per-sample threshold crossing is dominated by
    local wiggle in that buildup rather than the actual decay rate. Fitting
    a line to the log-energy tail (ignoring the ~60 dB-down numerical noise
    floor) is the standard, much more robust way to estimate a decay time
    from a measured impulse response.
    """
    env = _energy_envelope(x, block=block)
    peak_idx = int(np.argmax(env))
    tail = env[peak_idx:]
    floor = np.max(env) - 60.0
    valid = tail > floor
    if valid.sum() < 5:
        return 0.0
    idx = np.nonzero(valid)[0]
    times = idx * block / sr
    values = tail[idx]
    slope, _intercept = np.polyfit(times, values, 1)
    slope = min(slope, -1e-9)  # guard: a real decay always has a negative slope
    return float(drop_db / abs(slope))


def test_impulse_response_decays_and_decay_time_tracks_room_size_monotonically():
    room_sizes = [0.15, 0.5, 0.9]
    decay_times = []
    for room_size in room_sizes:
        x = _impulse()
        y, stats = reverb.reverb(x, SR, room_size=room_size, damping=0.2, wet=1.0, dry=0.0, pre_delay_ms=0.0)
        env = _energy_envelope(y)
        # Roughly exponential: energy should be monotonically non-increasing on
        # average over the back half of the buffer (allow local wiggles from the
        # FDN's diffuse reflections by comparing coarse quartile means instead of
        # every sample).
        quarter = len(env) // 4
        q1 = float(np.mean(env[quarter : 2 * quarter]))
        q3 = float(np.mean(env[3 * quarter :]))
        assert q3 < q1, f"room_size={room_size}: tail did not decay (q1={q1:.2f} dB, q3={q3:.2f} dB)"

        t = _decay_time_s(y, SR)
        decay_times.append(t)
        print(
            f"\n[reverb] room_size={room_size}: -20dB decay time = {t:.4f} s (estimated={stats['estimated_decay_s']:.4f} s)"
        )

    print(f"[reverb] decay times across room sizes: {decay_times}")
    assert decay_times[0] < decay_times[1] < decay_times[2], (
        f"decay time must increase monotonically with room_size, got {decay_times} for {room_sizes}"
    )


def test_wet_zero_returns_dry_signal_unchanged():
    x = 0.3 * np.sin(2.0 * np.pi * 440.0 * np.arange(SR) / SR).reshape(-1, 1)
    y, _stats = reverb.reverb(x, SR, room_size=0.7, damping=0.5, wet=0.0, dry=1.0)
    max_diff = float(np.max(np.abs(y - x)))
    print(f"\n[reverb] wet=0 max |y - x| = {max_diff:.3e}")
    assert max_diff < 1e-9


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = float(np.sum(spec))
    if total < 1e-12:
        return 0.0
    return float(np.sum(spec * freqs) / total)


def test_damping_reduces_high_frequency_energy_in_the_tail():
    n = SR * 2
    tail_start = SR // 2  # look at the tail, well after the direct impulse

    x = _impulse(n)
    y_undamped, _ = reverb.reverb(x, SR, room_size=0.8, damping=0.02, wet=1.0, dry=0.0)
    y_damped, _ = reverb.reverb(x, SR, room_size=0.8, damping=0.9, wet=1.0, dry=0.0)

    centroid_undamped = _spectral_centroid(y_undamped[tail_start:, 0], SR)
    centroid_damped = _spectral_centroid(y_damped[tail_start:, 0], SR)

    print(f"\n[reverb] tail spectral centroid: undamped={centroid_undamped:.1f} Hz damped={centroid_damped:.1f} Hz")
    assert centroid_damped < centroid_undamped, (
        f"damping=0.9 should darken the tail relative to damping=0.02, "
        f"got damped={centroid_damped:.1f} Hz >= undamped={centroid_undamped:.1f} Hz"
    )


def test_output_never_exceeds_a_sane_level_for_normalised_input():
    rng = np.random.default_rng(7)
    x = 0.9 * rng.uniform(-1.0, 1.0, size=(SR, 2))  # normalised, hot programme material
    for room_size in (0.1, 0.5, 0.95):
        for mix in (0.1, 0.5, 0.9):
            y, _stats = reverb.reverb(x, SR, room_size=room_size, damping=0.4, wet=mix, dry=1.0 - mix)
            peak = float(np.max(np.abs(y)))
            assert peak <= 1.05, f"room_size={room_size} mix={mix}: reverb produced a peak of {peak:.4f} (clipping)"


def test_convolve_ir_produces_expected_length_and_a_measurable_tail():
    n = SR  # 1 second
    x = np.zeros((n, 1))
    x[100, 0] = 1.0  # a single impulse, early, with lots of trailing silence

    ir_len = int(0.2 * SR)
    rng = np.random.default_rng(3)
    decay_env = np.exp(-np.arange(ir_len) / (0.05 * SR))
    ir = (rng.standard_normal(ir_len) * decay_env).astype(np.float64)

    y, stats = reverb.convolve_ir(x, SR, ir, wet=1.0, dry=0.0)

    assert y.shape == x.shape
    assert stats["ir_length_samples"] == ir_len
    assert stats["resampled"] is False

    # A measurable tail well beyond where the un-convolved impulse would already
    # be silent (i.e. beyond sample 100), and before the IR's own length runs out.
    tail = y[150 : 100 + ir_len - 10, 0]
    tail_energy = float(np.sqrt(np.mean(tail * tail)))
    print(f"\n[reverb] convolve_ir tail RMS: {tail_energy:.3e}, ir_length={ir_len}")
    assert tail_energy > 1e-4, "convolution should leave a measurable tail, not act like a passthrough"
