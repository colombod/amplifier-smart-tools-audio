"""Noise gate and downward expander -- the quiet-end counterpart to compression.

`aud.dsp.dynamics` pulls loud material down. Nothing in this tool touched the
quiet end until this module: room tone between phrases, mic hiss, HVAC noise,
a breath in the wrong place. For cleanup work this is often the stage that
matters most, and it is a different device from a compressor, not a mirrored
one -- a compressor's static curve bends *toward* unity as level rises past
its threshold; a gate/expander's curve bends *away* from unity as level falls
*below* its threshold.

Two devices, one topology, sharing everything but the static curve:

    per-band level detection (optionally sidechain-highpassed)
      -> static gain-reduction curve (gate: hard step; expander: soft-knee ratio)
      -> forward lookahead (anticipate the RISE back to unity, not a dip)
      -> hold + attack/release smoothing
      -> apply per band -> recombine

`gate` is HARD: below threshold, attenuate by a fixed `range_db` regardless of
how far below. Note `range_db`, not silence -- a gate that slams to digital
black is more noticeable than the noise it removed, so depth is a parameter
and its default is a duck, not a kill. `expand` is GENTLE: below threshold,
attenuate proportionally to a `ratio`, with a soft knee -- usually the right
first tool for voice material, with `gate` reserved for harder cases.

Both are built on five properties that are easy to get wrong and are each
directly testable (see tests/test_dsp_gate.py):

1. **Hold is not optional.** Without it, material hovering right at the
   threshold makes the gate open and close many times a second (chatter),
   which is the single most recognisable way a gate sounds broken. `hold_ms`
   keeps the gate held open for that long after the raw detector last asked
   for unity, before release is allowed to start closing it.
2. **Threshold is relative to the measured noise floor**, via
   `threshold_above_floor_db` (primary control, same convention as
   `aud.dsp.detect.detect_silence`) -- a fixed dBFS number is right for
   exactly one recording. `threshold_db` is an absolute escape hatch for a
   caller that already knows one.
3. **Lookahead**, so the onset survives. Reusing `aud.dsp.limiter`'s
   machinery in spirit (not by import -- see `_lookahead_max`'s docstring):
   a forward MAXIMUM filter over the target curve, the gate's mirror image
   of the limiter's forward MINIMUM. The limiter anticipates a dip; the gate
   anticipates a rise, so the envelope has already started opening before a
   sharp attack arrives, rather than clipping its first few milliseconds.
4. **Multiband**, for the same reason `aud.dsp.dynamics` is: gating a signal
   that is loud in one band and at the noise floor in another, full-band,
   either gates the wrong thing or nothing at all. Reuses
   `aud.dsp.crossover`'s Linkwitz-Riley split exactly as `aud.dsp.deess`
   already does. `crossovers_hz=None` (or `[]`) means full-band.
5. **Sidechain highpass.** The detector runs on a highpassed copy of each
   band (`sidechain_hpf_hz`, on by default at 80 Hz -- below the fundamental
   of nearly all spoken-word material) so low-frequency rumble cannot hold
   the gate open for content that, once the rumble is filtered out, is not
   actually there.

Stats report what the stage actually DID, not the parameters handed to it:
attenuated percentage, max/average attenuation in dB, and how many times the
gate opened -- the open count is what tells a caller their hold time is
wrong, and it is the number that makes chatter visible in a render report
rather than only in a test built to provoke it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage, signal

from aud.dsp import crossover, detect

__all__ = ["expand", "gate"]

_EPS = 1e-12
_OPEN_EPS_DB = 0.01  # "no gain reduction requested" tolerance, floating-point slack only
_ATTENUATED_EPS_DB = 0.1  # "the stage actually did something to this sample" reporting threshold


def _detector_level_db(
    band: np.ndarray,
    sr: int,
    sidechain_hpf_hz: float | None,
    smoothing_ms: float = 5.0,
) -> np.ndarray:
    """Continuous, mono, short-time RMS-ish level of `band`, in dB, per sample.

    Channels are summed (mono-linked detection) for the same reason
    `aud.dsp.dynamics`'s compressor and `aud.dsp.deess`'s de-esser do: an
    independent per-channel decision would duck one channel and not the
    other, shifting the stereo image every time the gate moves.

    When `sidechain_hpf_hz` is given, the level is measured on a highpassed
    COPY of the band -- the band itself, returned to the caller, is
    untouched by this filter. This is what keeps low-frequency rumble from
    holding the gate open for content that is not actually there (property
    5 in this module's docstring).
    """
    detector_signal = band
    if sidechain_hpf_hz is not None and sidechain_hpf_hz > 0:
        sos = signal.butter(2, sidechain_hpf_hz, btype="highpass", fs=sr, output="sos")
        detector_signal = signal.sosfilt(sos, band, axis=0)

    power = detector_signal * detector_signal
    mono_power = np.mean(power, axis=1) if power.ndim == 2 else power
    coeff = np.exp(-1.0 / (sr * smoothing_ms / 1000.0))
    smoothed_power = signal.lfilter([1.0 - coeff], [1.0, -coeff], mono_power)
    return 20.0 * np.log10(np.sqrt(np.maximum(smoothed_power, _EPS)))


def _gate_curve(level_db: np.ndarray, threshold_db: float, range_db: float) -> np.ndarray:
    """Hard step: 0 dB at/above threshold, -range_db strictly below it."""
    return np.where(level_db >= threshold_db, 0.0, -range_db)


def _expander_curve(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Soft-knee downward expansion, in dB (<= 0), mirrored from the compressor's.

    Below `threshold_db`, output moves `ratio` dB for every 1 dB the input
    drops (steeper than the unprocessed slope of 1); at/above it, the signal
    is untouched. `aud.dsp.dynamics._static_gain_reduction_db` builds a
    matching soft knee for a compressor's *above*-threshold curve; this is
    its mirror image, applied *below* threshold, with a quadratic connecting
    piece derived the same way (matching value and slope at both knee
    edges). Both curves pass through the same point at the knee's center,
    `(threshold_db, threshold_db)`, which is what makes a single quadratic
    span the gap exactly.
    """
    t = threshold_db
    w = max(knee_db, 0.0)
    ratio = max(ratio, 1.0)

    if w <= 0.0:
        y_db = np.where(level_db < t, t + (level_db - t) * ratio, level_db)
        return y_db - level_db

    x1 = t - w / 2.0
    x2 = t + w / 2.0
    below = level_db <= x1
    above = level_db >= x2
    in_knee = ~below & ~above

    y_db = np.array(level_db, dtype=np.float64, copy=True)
    y_db[below] = t + (level_db[below] - t) * ratio
    if np.any(in_knee):
        x_minus_x2 = level_db[in_knee] - x2
        y_db[in_knee] = level_db[in_knee] + ((1.0 - ratio) * x_minus_x2**2) / (2.0 * w)
    # above: y_db == level_db already (no change)
    return y_db - level_db  # <= 0 everywhere


def _lookahead_max(target_db: np.ndarray, window: int) -> np.ndarray:
    """target_la[n] = max(target_db[n : n + window]) -- forward-looking.

    This is the gate's mirror image of `aud.dsp.limiter._lookahead_min` (and
    `aud.dsp.deess`'s own copy of the same idea): the limiter anticipates a
    DIP (a peak that needs gain reduction) with a forward MINIMUM, so it has
    already started reducing gain before the peak arrives. A gate needs to
    anticipate the opposite event -- a RISE back toward unity, i.e. an
    onset -- so it takes a forward MAXIMUM instead: if unity gain (0 dB, the
    least-attenuated value) occurs anywhere in the next `window` samples,
    the current sample is treated as already heading there too. Without
    this, the gate only starts opening once the onset has already arrived,
    clipping its first few milliseconds -- exactly the failure this
    property exists to prevent.

    Duplicated rather than imported for the same reason `aud.dsp.deess`
    duplicates its own copy: it is a handful of lines, the callers apply it
    to different signals for different reasons, and sharing it would couple
    two otherwise-independent dsp modules to each other's internals.
    """
    if window <= 1:
        return target_db
    origin = -(window // 2)
    return ndimage.maximum_filter1d(target_db, size=window, mode="nearest", origin=origin)


def _hold_attack_release_db(
    target_db: np.ndarray,
    sr: int,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
) -> tuple[np.ndarray, int]:
    """Hold + attack/release smoothing of a per-sample target curve (<= 0 dB).

    Property 1 in this module's docstring, made concrete: `hold_ms` is a
    countdown that resets to full every time the (lookahead-adjusted) target
    asks for unity gain (0 dB, within `_OPEN_EPS_DB`), and only once it
    expires is the gate allowed to start moving toward the target's actual
    (possibly negative) value under `release_ms`. Moving the OTHER way --
    toward unity -- always uses `attack_ms` and is never held back, so an
    onset is never delayed by a hold timer left over from the previous
    silence.

    Returns:
        (smoothed_db, open_count) where `open_count` is the number of times
        the held-open state transitioned from closed to open -- the number
        a caller reads to tell whether their hold time is too short (many
        transitions on steady material means chatter).
    """
    attack_coeff = np.exp(-1.0 / (sr * max(attack_ms, 1e-3) / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * max(release_ms, 1e-3) / 1000.0))
    hold_samples = max(0, round(hold_ms / 1000.0 * sr))

    is_open_raw = target_db > -_OPEN_EPS_DB

    smoothed = np.empty_like(target_db)
    prev = 0.0  # start fully open: no reduction assumed before the signal is seen
    hold_counter = 0
    open_count = 0
    was_open = True
    for n in range(target_db.shape[0]):
        if is_open_raw[n]:
            hold_counter = hold_samples
            held_open = True
        elif hold_counter > 0:
            hold_counter -= 1
            held_open = True
        else:
            held_open = False

        if held_open and not was_open:
            open_count += 1
        was_open = held_open

        target = 0.0 if held_open else target_db[n]
        coeff = attack_coeff if target > prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * target
        smoothed[n] = prev

    return smoothed, open_count


def _process_bands(
    x: np.ndarray,
    sr: int,
    crossovers_hz: list[float] | None,
    *,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
    lookahead_ms: float,
    sidechain_hpf_hz: float | None,
    curve: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    bands = crossover.split(x, sr, list(crossovers_hz)) if crossovers_hz else [x]
    window = max(1, round(lookahead_ms / 1000.0 * sr))

    processed: list[np.ndarray] = []
    reports: list[dict] = []
    for index, band in enumerate(bands):
        level_db = _detector_level_db(band, sr, sidechain_hpf_hz)
        raw_target_db = curve(level_db)
        target_la = _lookahead_max(raw_target_db, window)
        smoothed_db, open_count = _hold_attack_release_db(target_la, sr, attack_ms, hold_ms, release_ms)

        gain_lin = 10.0 ** (smoothed_db / 20.0)
        processed.append(band * gain_lin[:, None])

        attenuated_mask = smoothed_db < -_ATTENUATED_EPS_DB
        reports.append(
            {
                "index": index,
                "max_attenuation_db": float(np.min(smoothed_db)) if smoothed_db.size else 0.0,
                "avg_attenuation_db": float(np.mean(smoothed_db)) if smoothed_db.size else 0.0,
                "attenuated_pct": float(100.0 * np.mean(attenuated_mask)) if attenuated_mask.size else 0.0,
                "open_count": open_count,
            }
        )

    y_total = crossover.recombine(processed) if len(processed) > 1 else processed[0]
    return y_total, reports


def _aggregate_band_reports(band_reports: list[dict]) -> dict:
    """Whole-signal summary over independently-detected bands.

    `max_attenuation_db` takes the deepest cut across bands; `avg_attenuation_db`
    and `attenuated_pct` are the mean across bands (an approximation for the
    multiband case -- bands are detected and smoothed independently, so there
    is no single per-sample curve to average exactly, the same honesty
    `aud.dsp.crossover`'s own docstring applies to its reconstruction ripple);
    `open_count` is summed, since each band's detector opens independently and
    each opening is a real event in that band.
    """
    return {
        "max_attenuation_db": min(r["max_attenuation_db"] for r in band_reports),
        "avg_attenuation_db": sum(r["avg_attenuation_db"] for r in band_reports) / len(band_reports),
        "attenuated_pct": sum(r["attenuated_pct"] for r in band_reports) / len(band_reports),
        "open_count": sum(r["open_count"] for r in band_reports),
    }


def _resolve_threshold_db(
    x: np.ndarray, sr: int, threshold_db: float | None, threshold_above_floor_db: float
) -> tuple[float, float]:
    """Returns (resolved_threshold_db, measured_noise_floor_dbfs).

    The floor is always measured and reported, even when `threshold_db` (the
    absolute escape hatch) overrides it, because it is useful diagnostic
    context either way -- the same reasoning `aud.dsp.detect.detect_silence`
    reports its measured floor alongside its threshold.
    """
    noise_floor_dbfs = detect.measure_noise_floor(x, sr)
    if threshold_db is not None:
        return float(threshold_db), noise_floor_dbfs
    return float(noise_floor_dbfs + threshold_above_floor_db), noise_floor_dbfs


def gate(
    x: np.ndarray,
    sr: int,
    *,
    threshold_above_floor_db: float = 12.0,
    threshold_db: float | None = None,
    range_db: float = 20.0,
    attack_ms: float = 2.0,
    hold_ms: float = 50.0,
    release_ms: float = 150.0,
    crossovers_hz: list[float] | None = None,
    lookahead_ms: float = 3.0,
    sidechain_hpf_hz: float | None = 80.0,
) -> tuple[np.ndarray, dict]:
    """Hard noise gate: below threshold, duck by `range_db`. See module docstring.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        threshold_above_floor_db: Primary threshold control -- dB above this
            file's measured noise floor (`aud.dsp.detect.measure_noise_floor`)
            below which the gate engages. >= 0.
        threshold_db: Absolute dBFS escape hatch. When given, overrides
            `threshold_above_floor_db` entirely; the floor is still measured
            and reported for context.
        range_db: Maximum attenuation applied below threshold, in dB. >= 0.
            Note: a DUCK by this many dB, not silence -- see module docstring.
        attack_ms: Time to open (move toward 0 dB) once triggered. >= 0.
        hold_ms: Minimum time the gate stays open after last being triggered,
            before release is allowed to start closing it. >= 0. This is
            what prevents chatter (property 1).
        release_ms: Time to close (move toward -range_db) once hold expires. >= 0.
        crossovers_hz: Ascending Linkwitz-Riley crossover frequencies, Hz.
            `None` (or `[]`) is full-band.
        lookahead_ms: How far ahead the detector looks so the gate has
            already started opening before a fast onset's peak arrives. >= 0.
        sidechain_hpf_hz: Highpass corner for the LEVEL DETECTOR only (the
            signal itself is unaffected). `None` disables it.

    Returns:
        (y, stats) -- see module docstring for what `stats` reports.

    Raises:
        ValueError: `range_db` negative, any time constant negative, or
            (via `aud.dsp.crossover.split`) `crossovers_hz` not strictly
            ascending / out of range.
    """
    if range_db < 0.0:
        raise ValueError(f"range_db must be >= 0, got {range_db}")
    for name, value in (
        ("attack_ms", attack_ms),
        ("hold_ms", hold_ms),
        ("release_ms", release_ms),
        ("lookahead_ms", lookahead_ms),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    x = np.asarray(x, dtype=np.float64)
    mono_input = x.ndim == 1
    if mono_input:
        x = x[:, None]

    resolved_threshold_db, noise_floor_dbfs = _resolve_threshold_db(x, sr, threshold_db, threshold_above_floor_db)

    def curve(level_db: np.ndarray) -> np.ndarray:
        return _gate_curve(level_db, resolved_threshold_db, range_db)

    y, band_reports = _process_bands(
        x,
        sr,
        crossovers_hz,
        attack_ms=attack_ms,
        hold_ms=hold_ms,
        release_ms=release_ms,
        lookahead_ms=lookahead_ms,
        sidechain_hpf_hz=sidechain_hpf_hz,
        curve=curve,
    )
    if mono_input:
        y = y[:, 0]

    stats = _aggregate_band_reports(band_reports)
    stats.update(
        {
            "threshold_db": resolved_threshold_db,
            "threshold_above_floor_db": None if threshold_db is not None else float(threshold_above_floor_db),
            "noise_floor_dbfs": float(noise_floor_dbfs),
            "range_db": float(range_db),
            "sidechain_hpf_hz": sidechain_hpf_hz,
            "bands": band_reports,
        }
    )
    return y, stats


def expand(
    x: np.ndarray,
    sr: int,
    *,
    threshold_above_floor_db: float = 6.0,
    threshold_db: float | None = None,
    ratio: float = 2.0,
    knee_db: float = 6.0,
    attack_ms: float = 5.0,
    hold_ms: float = 50.0,
    release_ms: float = 150.0,
    crossovers_hz: list[float] | None = None,
    lookahead_ms: float = 3.0,
    sidechain_hpf_hz: float | None = 80.0,
) -> tuple[np.ndarray, dict]:
    """Soft-knee downward expander: below threshold, attenuate proportionally.

    The gentle counterpart to `gate` -- usually the right first tool for
    voice material, with `gate`'s hard step reserved for harder cases. See
    module docstring for the shared topology and the properties both share.

    Args:
        x: Array of shape (n_samples, n_channels) or (n_samples,).
        sr: Sample rate in Hz.
        threshold_above_floor_db: Primary threshold control -- dB above this
            file's measured noise floor below which expansion engages. >= 0.
        threshold_db: Absolute dBFS escape hatch; overrides
            `threshold_above_floor_db` when given.
        ratio: Downward expansion ratio, `>= 1.0`. `1.0` is no expansion;
            `2.0` means output moves 2 dB for every 1 dB the input drops
            below threshold. Below 1.0 is upward expansion -- a different
            device -- and is rejected.
        knee_db: Width of the soft knee centred on the threshold, in dB. >= 0.
        attack_ms: Time to open (move toward 0 dB) once triggered. >= 0.
        hold_ms: Minimum time held open before release may start closing. >= 0.
        release_ms: Time to move toward the target reduction once hold expires. >= 0.
        crossovers_hz: Ascending Linkwitz-Riley crossover frequencies, Hz.
            `None` (or `[]`) is full-band.
        lookahead_ms: How far ahead the detector looks so an onset survives. >= 0.
        sidechain_hpf_hz: Highpass corner for the level detector only. `None` disables it.

    Returns:
        (y, stats) -- see module docstring for what `stats` reports.

    Raises:
        ValueError: `ratio` below 1.0, `knee_db` negative, any time constant
            negative, or (via `aud.dsp.crossover.split`) `crossovers_hz` not
            strictly ascending / out of range.
    """
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1.0 (below 1.0 is upward expansion, a different device), got {ratio}")
    if knee_db < 0.0:
        raise ValueError(f"knee_db must be >= 0, got {knee_db}")
    for name, value in (
        ("attack_ms", attack_ms),
        ("hold_ms", hold_ms),
        ("release_ms", release_ms),
        ("lookahead_ms", lookahead_ms),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    x = np.asarray(x, dtype=np.float64)
    mono_input = x.ndim == 1
    if mono_input:
        x = x[:, None]

    resolved_threshold_db, noise_floor_dbfs = _resolve_threshold_db(x, sr, threshold_db, threshold_above_floor_db)

    def curve(level_db: np.ndarray) -> np.ndarray:
        return _expander_curve(level_db, resolved_threshold_db, ratio, knee_db)

    y, band_reports = _process_bands(
        x,
        sr,
        crossovers_hz,
        attack_ms=attack_ms,
        hold_ms=hold_ms,
        release_ms=release_ms,
        lookahead_ms=lookahead_ms,
        sidechain_hpf_hz=sidechain_hpf_hz,
        curve=curve,
    )
    if mono_input:
        y = y[:, 0]

    stats = _aggregate_band_reports(band_reports)
    stats.update(
        {
            "threshold_db": resolved_threshold_db,
            "threshold_above_floor_db": None if threshold_db is not None else float(threshold_above_floor_db),
            "noise_floor_dbfs": float(noise_floor_dbfs),
            "ratio": float(ratio),
            "knee_db": float(knee_db),
            "sidechain_hpf_hz": sidechain_hpf_hz,
            "bands": band_reports,
        }
    )
    return y, stats
