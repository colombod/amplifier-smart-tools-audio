"""aud.lib -- the library is the tool.

Every capability aud has lives here as a plain function. The CLI (cli.py) is
a thin wrapper: argument parsing and printing only, no logic. Anything you
can do from the shell you can do by importing this module directly.

Functions that touch samples (`analyze`, `render`, `verify`, `curve_extract`,
`curve_apply`) import `aud.dsp` lazily, inside the function body, so that
importing `aud.lib` itself never requires numpy/scipy/soundfile to have a
working DSP backend behind them -- only to be installed. If `aud.dsp` is not
available yet, these functions raise `AudError(code="not_implemented")`
rather than a bare ImportError or a silent fake result.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
import tomllib
from pathlib import Path
from typing import Any

from aud.core.manifest import load_manifest
from aud.core.skill import render_skill
from aud.plan import Plan, append
from aud.schemas import AudError, NotImplementedStageError

# ---------------------------------------------------------------------------
# manifest / check / config / skill
# ---------------------------------------------------------------------------

_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)

# name, purpose, install -- for the core DSP dependencies declared in pyproject.toml.
_CORE_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "Array math backing every DSP stage.", "https://pypi.org/project/numpy/"),
    ("scipy", "Filter design and signal processing primitives.", "https://pypi.org/project/scipy/"),
    ("soundfile", "Reads and writes WAV/FLAC/AIFF via libsndfile.", "https://pypi.org/project/soundfile/"),
    ("pyloudnorm", "ITU-R BS.1770 loudness measurement.", "https://pypi.org/project/pyloudnorm/"),
)

_CONFIG_DEFAULTS: dict[str, Any] = {
    "sample_rate_policy": "preserve",
    "default_ceiling_dbtp": -1.0,
    "default_target_lufs": -14.0,
    "oversample": 4,
    "output_subtype": "PCM_24",
}

_CONFIG_ENV_PREFIX = "AUD_"
_CONFIG_FILE_PATH = Path.home() / ".config" / "aud" / "config.toml"


def manifest() -> dict:
    """The validated SMART_TOOL.md manifest, as a plain dict."""
    return load_manifest().model_dump()


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def check() -> dict:
    """Host readiness: importable DSP deps, ffmpeg on PATH, provider credentials.

    Provider credentials are reported as name + boolean only -- the value of
    any environment variable is never read into the result.
    """
    man = load_manifest()
    manifest_requirements = {requirement.name: requirement for requirement in man.requires}

    requirements: list[dict[str, Any]] = []
    core_ready = True
    for package_name, purpose, install in _CORE_PACKAGES:
        available = _module_available(package_name)
        core_ready = core_ready and available
        requirements.append(
            {
                "name": package_name,
                "state": "satisfied" if available else "absent",
                "detail": "importable" if available else "failed to import",
                "purpose": purpose,
                "install": install,
            }
        )

    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_requirement = manifest_requirements.get("ffmpeg")
    requirements.append(
        {
            "name": "ffmpeg",
            "state": "satisfied" if ffmpeg_path else "absent",
            "detail": f"found at {ffmpeg_path}" if ffmpeg_path else "not found on PATH",
            "purpose": ffmpeg_requirement.purpose if ffmpeg_requirement else "",
            "install": ffmpeg_requirement.install if ffmpeg_requirement else "",
        }
    )

    provider_flags = {name: bool(os.environ.get(name)) for name in _PROVIDER_ENV_VARS}
    provider_requirement = manifest_requirements.get("ai-provider")
    requirements.append(
        {
            "name": "ai-provider",
            "state": "satisfied" if any(provider_flags.values()) else "absent",
            "detail": ", ".join(f"{name}={'set' if flag else 'absent'}" for name, flag in provider_flags.items()),
            "purpose": provider_requirement.purpose if provider_requirement else "",
            "install": provider_requirement.install if provider_requirement else "",
        }
    )

    return {
        "tool": man.name,
        "version": man.version,
        "python_version": platform.python_version(),
        "ready": core_ready,
        "deterministic_capabilities_available": True,
        "requirements": requirements,
    }


def _load_config_file() -> dict[str, Any]:
    if not _CONFIG_FILE_PATH.exists():
        return {}
    try:
        with _CONFIG_FILE_PATH.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _coerce_like(default: Any, raw: str) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def config(**overrides: Any) -> dict:
    """Effective settings and which tier each came from.

    Four-tier resolution, highest wins: explicit argument > config file
    (~/.config/aud/config.toml) > environment (AUD_*) > built-in default.
    Pass an override as a keyword argument set to a non-None value to make
    it win at the "argument" tier.
    """
    file_values = _load_config_file()
    result: dict[str, dict[str, Any]] = {}
    for key, default in _CONFIG_DEFAULTS.items():
        if overrides.get(key) is not None:
            result[key] = {"value": overrides[key], "source": "argument"}
            continue
        if key in file_values:
            result[key] = {"value": file_values[key], "source": "config_file"}
            continue
        env_key = _CONFIG_ENV_PREFIX + key.upper()
        if env_key in os.environ:
            result[key] = {"value": _coerce_like(default, os.environ[env_key]), "source": "environment"}
            continue
        result[key] = {"value": default, "source": "default"}
    return result


def skill() -> str:
    """The full `aud --help` skill text."""
    return render_skill()


# ---------------------------------------------------------------------------
# Stage builders -- validate parameters and append to a plan. No DSP here.
# ---------------------------------------------------------------------------

_SNAP_MODES = ("zero_crossing", "silence", "transient", "none")
_CROSSFADE_SHAPES = ("equal_power", "linear")


def _validate_edit_point_params(
    *,
    pad_out_ms: float,
    pad_in_ms: float,
    snap: str,
    snap_window_ms: float,
    fade_out_ms: float,
    fade_in_ms: float,
    crossfade_ms: float,
    crossfade_shape: str,
) -> None:
    """Shared validation for `cut` and `strip_silence`'s edit-point resolution params.

    See contracts/plan.v1.md#edit-point-resolution-shared-by-cut-and-strip_silence
    for every constraint enforced here.
    """
    for name, value in (
        ("pad_out_ms", pad_out_ms),
        ("pad_in_ms", pad_in_ms),
        ("fade_out_ms", fade_out_ms),
        ("fade_in_ms", fade_in_ms),
        ("crossfade_ms", crossfade_ms),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise AudError(
                code="bad_param",
                message=f"{name} must be a finite number >= 0, got {value!r}.",
                remedy=f"Use a non-negative number of milliseconds for {name}.",
            )
    if snap not in _SNAP_MODES:
        raise AudError(
            code="bad_param",
            message=f"snap must be one of {_SNAP_MODES}, got {snap!r}.",
            remedy="Use one of zero_crossing, silence, transient, none.",
        )
    if crossfade_shape not in _CROSSFADE_SHAPES:
        raise AudError(
            code="bad_param",
            message=f"crossfade_shape must be one of {_CROSSFADE_SHAPES}, got {crossfade_shape!r}.",
            remedy="Use equal_power or linear.",
        )
    if not isinstance(snap_window_ms, (int, float)) or not (0.0 < snap_window_ms <= 1000.0):
        raise AudError(
            code="snap_window_invalid",
            message=f"snap_window_ms must be > 0 and <= 1000.0, got {snap_window_ms!r}.",
            remedy="Use a snap_window_ms between 0 (exclusive) and 1000.0 (inclusive).",
        )


def cut(
    plan: Plan,
    regions_text: str | None,
    *,
    pad_out_ms: float = 0.0,
    pad_in_ms: float = 0.0,
    snap: str = "zero_crossing",
    snap_window_ms: float = 20.0,
    fade_out_ms: float = 0.0,
    fade_in_ms: float = 0.0,
    crossfade_ms: float = 10.0,
    crossfade_shape: str = "equal_power",
) -> Plan:
    """Editing stage: remove an explicit list of regions from the programme.

    `regions_text` is a regions document (contracts/regions.v1.md), normally
    the output of `aud detect silence` or `aud detect fillers` piped in or
    read from a file. `cut` stores plain `{"start_s", "end_s"}` positions --
    the rest of the document (peak/RMS levels, detection provenance) is not
    part of what `cut`'s stage needs to replay.

    `cut` is handed positions a caller measured and means literally, so its
    padding defaults to none -- widening someone's stated edit unasked would
    be a surprise (see contracts/plan.v1.md#padding).
    """
    _validate_edit_point_params(
        pad_out_ms=pad_out_ms,
        pad_in_ms=pad_in_ms,
        snap=snap,
        snap_window_ms=snap_window_ms,
        fade_out_ms=fade_out_ms,
        fade_in_ms=fade_in_ms,
        crossfade_ms=crossfade_ms,
        crossfade_shape=crossfade_shape,
    )
    try:
        from aud.core.regions import read_regions
    except ImportError as exc:
        raise _not_implemented("cut", exc) from exc

    doc = read_regions(regions_text)
    if doc.kind == "transient":
        raise AudError(
            code="regions_not_cuttable",
            message="'cut' was given a transients regions document; transients are zero-length and describe "
            "nothing to remove.",
            remedy="Pipe a 'silence' or 'filler' regions document into 'cut'; use transients to inform "
            "--snap transient elsewhere.",
        )

    regions = [{"start_s": float(region.start_s), "end_s": float(region.end_s)} for region in doc.regions]
    params = {
        "regions": regions,
        "pad_out_ms": pad_out_ms,
        "pad_in_ms": pad_in_ms,
        "snap": snap,
        "snap_window_ms": snap_window_ms,
        "fade_out_ms": fade_out_ms,
        "fade_in_ms": fade_in_ms,
        "crossfade_ms": crossfade_ms,
        "crossfade_shape": crossfade_shape,
    }
    return append(plan, "cut", params)


def strip_silence(
    plan: Plan,
    *,
    threshold_above_floor_db: float = 6.0,
    min_len_ms: float = 400.0,
    keep_ms: float = 150.0,
    pad_out_ms: float = 80.0,
    pad_in_ms: float = 80.0,
    snap: str = "zero_crossing",
    snap_window_ms: float = 20.0,
    fade_out_ms: float = 0.0,
    fade_in_ms: float = 0.0,
    crossfade_ms: float = 10.0,
    crossfade_shape: str = "equal_power",
) -> Plan:
    """Editing stage: remove or shorten silences, detected at render time.

    Unlike `cut`, this stage carries no positions -- it stores a rule
    (contracts/plan.v1.md#strip_silence) and detects at render time, so the
    same plan means the same thing on every file it is applied to.

    `strip_silence` finds its own boundaries from an energy threshold, whose
    bias is systematically *inside* the speech (the tail of a word crosses
    the threshold while the word is still going), so padding is on by
    default here -- unlike `cut`.
    """
    _validate_edit_point_params(
        pad_out_ms=pad_out_ms,
        pad_in_ms=pad_in_ms,
        snap=snap,
        snap_window_ms=snap_window_ms,
        fade_out_ms=fade_out_ms,
        fade_in_ms=fade_in_ms,
        crossfade_ms=crossfade_ms,
        crossfade_shape=crossfade_shape,
    )
    if not math.isfinite(threshold_above_floor_db) or threshold_above_floor_db < 0.0:
        raise AudError(
            code="bad_param",
            message=f"threshold_above_floor_db must be a finite number >= 0, got {threshold_above_floor_db!r}.",
            remedy="Use a non-negative number of dB above the measured noise floor, e.g. --threshold 6.",
        )
    if not math.isfinite(min_len_ms) or min_len_ms <= 0.0:
        raise AudError(
            code="bad_param",
            message=f"min_len_ms must be a finite number > 0, got {min_len_ms!r}.",
            remedy="Use a positive number of milliseconds, e.g. --min-len 400.",
        )
    if not math.isfinite(keep_ms) or keep_ms < 0.0:
        raise AudError(
            code="bad_param",
            message=f"keep_ms must be a finite number >= 0, got {keep_ms!r}.",
            remedy="Use a non-negative number of milliseconds, e.g. --keep 150.",
        )

    params = {
        "threshold_above_floor_db": threshold_above_floor_db,
        "min_len_ms": min_len_ms,
        "keep_ms": keep_ms,
        "pad_out_ms": pad_out_ms,
        "pad_in_ms": pad_in_ms,
        "snap": snap,
        "snap_window_ms": snap_window_ms,
        "fade_out_ms": fade_out_ms,
        "fade_in_ms": fade_in_ms,
        "crossfade_ms": crossfade_ms,
        "crossfade_shape": crossfade_shape,
    }
    return append(plan, "strip_silence", params)


def deess(plan: Plan, amount_db: float = 6.0, freq_hz: float = 6500.0) -> Plan:
    """Repair stage: tame sibilance.

    `amount_db` is the maximum gain reduction in the sibilant band, in dB.
    `freq_hz` is the centre of that band. Field names and defaults match
    contracts/plan.v1.md's `deess` stage exactly.
    """
    if not math.isfinite(amount_db) or amount_db < 0.0:
        raise AudError(
            code="bad_param",
            message=f"De-ess amount_db must be a finite number >= 0, got {amount_db!r}.",
            remedy="Use a non-negative number of dB, e.g. --amount 6.",
        )
    if not math.isfinite(freq_hz) or freq_hz <= 0.0:
        raise AudError(
            code="bad_param",
            message=f"De-ess freq_hz must be a finite number > 0, got {freq_hz!r}.",
            remedy="Use a frequency above 0 Hz, e.g. --freq 6500.",
        )
    if freq_hz > _NYQUIST_44100:
        raise AudError(
            code="bad_param",
            message=f"De-ess freq_hz {freq_hz} exceeds Nyquist at 44100 Hz ({_NYQUIST_44100} Hz).",
            remedy=f"Keep freq_hz below {_NYQUIST_44100} Hz.",
        )
    return append(plan, "deess", {"amount_db": amount_db, "freq_hz": freq_hz})


def dereverb(plan: Plan, amount_db: float = 6.0) -> Plan:
    """Repair stage: reduce room ambience.

    `amount_db` is the maximum reduction applied to the estimated
    reverberant component, in dB -- matches contracts/plan.v1.md's
    `dereverb` stage exactly.
    """
    if not math.isfinite(amount_db) or amount_db < 0.0:
        raise AudError(
            code="bad_param",
            message=f"De-reverb amount_db must be a finite number >= 0, got {amount_db!r}.",
            remedy="Use a non-negative number of dB, e.g. --amount 6.",
        )
    return append(plan, "dereverb", {"amount_db": amount_db})


def eq(
    plan: Plan,
    hpf: float | None = None,
    lpf: float | None = None,
    peaks: list[tuple[float, float, float]] | None = None,
) -> Plan:
    """Tone stage: parametric EQ.

    `hpf`/`lpf` are high-pass/low-pass corner frequencies in Hz. Each entry
    in `peaks` is a (freq_hz, gain_db, q) triple; q must be > 0.
    """
    if hpf is not None and hpf <= 0:
        raise AudError(
            code="bad_param",
            message=f"High-pass frequency must be positive, got {hpf}.",
            remedy="Use a frequency in Hz greater than 0, e.g. --hpf 40.",
        )
    if lpf is not None and lpf <= 0:
        raise AudError(
            code="bad_param",
            message=f"Low-pass frequency must be positive, got {lpf}.",
            remedy="Use a frequency in Hz greater than 0, e.g. --lpf 18000.",
        )
    if hpf is not None and lpf is not None and hpf >= lpf:
        raise AudError(
            code="bad_param",
            message=f"High-pass frequency {hpf} Hz must be below low-pass frequency {lpf} Hz.",
            remedy="Choose hpf < lpf, e.g. --hpf 40 --lpf 18000.",
        )
    validated_peaks: list[list[float]] = []
    for peak in peaks or []:
        if len(peak) != 3:
            raise AudError(
                code="bad_param",
                message=f"EQ peak {peak!r} is not a (freq, gain_db, q) triple.",
                remedy="Use three comma-separated numbers, e.g. --peak 3200,-2.5,1.4.",
            )
        peak_freq, gain_db, q = peak
        if peak_freq <= 0:
            raise AudError(
                code="bad_param",
                message=f"EQ peak frequency must be positive, got {peak_freq}.",
                remedy="Use a frequency in Hz greater than 0, e.g. --peak 3200,-2.5,1.4.",
            )
        if q <= 0:
            raise AudError(
                code="bad_param",
                message=f"EQ peak Q must be positive, got {q}.",
                remedy="Use a Q greater than 0, e.g. --peak 3200,-2.5,1.4.",
            )
        validated_peaks.append([float(peak_freq), float(gain_db), float(q)])
    return append(plan, "eq", {"hpf": hpf, "lpf": lpf, "peaks": validated_peaks})


_EQ_MATCH_MIN_POINTS = 2


def _normalize_eq_match_curve(curve: Any) -> list[list[float]]:
    """Extract and validate a bare `[[freq_hz, gain_db], ...]` curve.

    Accepts either the rich dict `curve_extract`/`spectrum_profile` produce
    (a "curve" key holding the pairs, plus measurement metadata) or a bare
    array of pairs -- the shape contracts/plan.v1.md's `eq_match.curve`
    stage field actually stores, so a hand-written curve can be pasted
    straight into `--curve` with no wrapper object needed, exactly as the
    contract promises ("the output of one can be pasted into the other").
    """
    pairs = curve.get("curve") if isinstance(curve, dict) else curve
    if not isinstance(pairs, list) or len(pairs) < _EQ_MATCH_MIN_POINTS:
        raise AudError(
            code="bad_param",
            message=f"eq_match curve must have at least {_EQ_MATCH_MIN_POINTS} [freq_hz, gain_db] pairs.",
            remedy="Pass a curve produced by 'aud curve extract' or 'aud eq-match --reference', "
            "or a hand-written array of ascending [freq_hz, gain_db] pairs.",
        )
    normalized: list[list[float]] = []
    previous_freq = float("-inf")
    for point in pairs:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
        ):
            raise AudError(
                code="bad_param",
                message=f"eq_match curve point {point!r} is not a finite [freq_hz, gain_db] pair.",
                remedy="Each curve point must be two finite numbers: frequency in Hz, then gain in dB.",
            )
        freq_hz, gain_db = float(point[0]), float(point[1])
        if freq_hz <= 0:
            raise AudError(
                code="bad_param",
                message=f"eq_match curve frequency must be positive, got {freq_hz}.",
                remedy="Use a frequency in Hz greater than 0.",
            )
        if freq_hz <= previous_freq:
            raise AudError(
                code="bad_param",
                message=f"eq_match curve frequencies must be strictly ascending, got {freq_hz} after {previous_freq}.",
                remedy="Sort curve points by ascending frequency with no repeats.",
            )
        previous_freq = freq_hz
        normalized.append([freq_hz, gain_db])
    return normalized


def eq_match(
    plan: Plan,
    curve: dict | list | None = None,
    reference_path: str | None = None,
    amount: float = 1.0,
    max_gain_db: float = 12.0,
) -> Plan:
    """Tone stage: match this file's tonal balance to a reference curve.

    Exactly one of `curve` (a curve produced by 'aud curve extract', or a
    bare array of `[freq_hz, gain_db]` pairs) or `reference_path` (a
    reference audio file, measured right now) must be given. Either way the
    plan stores plain numbers, never a file path: contracts/plan.v1.md's
    `eq_match.curve` is an array of measurements, so a saved plan replays
    identically with no access to the reference file -- the same reasoning
    `strip_silence` follows for a policy rather than positions.

    `amount` is how much of the curve to apply, 0.0-1.0 (contract default
    1.0); `max_gain_db` clamps any single point of the curve, in both
    directions, at render time (contract default 12.0) -- see
    contracts/plan.v1.md#eq_match.
    """
    if (curve is None) == (reference_path is None):
        raise AudError(
            code="bad_param",
            message="eq_match needs exactly one of 'curve' or 'reference_path'.",
            remedy="Pass --curve <curve.json> or --reference <reference.wav>, not both and not neither.",
        )
    if not (0.0 <= amount <= 1.0):
        raise AudError(
            code="bad_param",
            message=f"Amount must be between 0.0 and 1.0, got {amount}.",
            remedy="Use an amount between 0 and 1, e.g. --strength 0.7.",
        )
    if not math.isfinite(max_gain_db) or max_gain_db < 0.0:
        raise AudError(
            code="bad_param",
            message=f"max_gain_db must be a finite number >= 0, got {max_gain_db!r}.",
            remedy="Use a non-negative number of dB, e.g. --max-gain-db 12.",
        )

    if reference_path is not None:
        try:
            from aud.dsp import eqmatch, io
        except ImportError as exc:
            raise _not_implemented("eq_match", exc) from exc
        samples, sample_rate = _read_audio(io, reference_path)
        curve = eqmatch.spectrum_profile(samples, sample_rate)

    curve_points = _normalize_eq_match_curve(curve)
    return append(plan, "eq_match", {"curve": curve_points, "amount": amount, "max_gain_db": max_gain_db})


# Nyquist frequency at the most common mastering sample rate. Crossovers are
# validated against this ceiling at plan time, before the real sample rate of
# the file being rendered is known.
_NYQUIST_44100 = 44100 / 2

# Per-band defaults for every field the CLI shorthand does not expose,
# matching contracts/plan.v1.md's documented "compress" band defaults
# exactly -- a stored plan must render the same way with any aud that
# accepts plan_format 1.
_BAND_THRESHOLD_DB_DEFAULT = -24.0
_BAND_ATTACK_MS_DEFAULT = 20.0
_BAND_RELEASE_MS_DEFAULT = 180.0
_BAND_KNEE_DB_DEFAULT = 6.0
_BAND_MAKEUP_DB_DEFAULT = 0.0


def compress(plan: Plan, bands: list[float], ratio: float = 2.5) -> Plan:
    """Dynamics stage: multiband compression.

    `bands` are Linkwitz-Riley crossover frequencies in Hz, strictly
    ascending, splitting the signal into len(bands) + 1 bands. `ratio` is
    the compression ratio applied to every band.

    This is the CLI's convenience shorthand over the plan document's full
    form (contracts/plan.v1.md): the document's `compress` stage always
    carries `crossovers_hz` (the frequencies) plus a `bands` array with one
    fully-specified settings object per band -- exactly
    `len(crossovers_hz) + 1` of them. This builder expands `bands`/`ratio`
    into that full form, sharing `ratio` and the documented defaults for
    every other per-band field across all bands.
    """
    if not bands:
        raise AudError(
            code="bad_param",
            message="Compress needs at least one crossover frequency.",
            remedy="Pass one or more ascending Hz values, e.g. --bands 120,900,5500.",
        )
    if any(band <= 0 for band in bands):
        raise AudError(
            code="bad_param",
            message=f"Crossover frequencies must be positive Hz values, got {bands!r}.",
            remedy="Use positive Hz values, e.g. --bands 120,900,5500.",
        )
    if list(bands) != sorted(bands) or len(set(bands)) != len(bands):
        raise AudError(
            code="bad_param",
            message=f"Crossover frequencies must be strictly ascending, got {bands!r}.",
            remedy="Order them low to high with no repeats, e.g. --bands 120,900,5500.",
        )
    if max(bands) > _NYQUIST_44100:
        raise AudError(
            code="bad_param",
            message=f"Crossover {max(bands)} Hz exceeds Nyquist at 44100 Hz ({_NYQUIST_44100} Hz).",
            remedy=f"Keep every crossover below {_NYQUIST_44100} Hz.",
        )
    if ratio <= 0:
        raise AudError(
            code="bad_param",
            message=f"Ratio must be positive, got {ratio}.",
            remedy="Use a ratio greater than 0, e.g. --ratio 2.5.",
        )
    crossovers_hz = [float(band) for band in bands]
    band_settings = [
        {
            "threshold_db": _BAND_THRESHOLD_DB_DEFAULT,
            "ratio": ratio,
            "attack_ms": _BAND_ATTACK_MS_DEFAULT,
            "release_ms": _BAND_RELEASE_MS_DEFAULT,
            "knee_db": _BAND_KNEE_DB_DEFAULT,
            "makeup_db": _BAND_MAKEUP_DB_DEFAULT,
        }
        for _ in range(len(crossovers_hz) + 1)
    ]
    return append(plan, "compress", {"crossovers_hz": crossovers_hz, "bands": band_settings})


def saturate(plan: Plan, drive: float = 1.0, mix: float = 0.25) -> Plan:
    """Character stage: harmonic saturation. `drive` > 0, `mix` in [0, 1]."""
    if drive <= 0:
        raise AudError(
            code="bad_param",
            message=f"Drive must be positive, got {drive}.",
            remedy="Use a drive greater than 0, e.g. --drive 1.5.",
        )
    if not (0.0 <= mix <= 1.0):
        raise AudError(
            code="bad_param",
            message=f"Mix must be between 0.0 and 1.0, got {mix}.",
            remedy="Use a mix between 0 and 1, e.g. --mix 0.25.",
        )
    return append(plan, "saturate", {"drive": drive, "mix": mix})


def reverb(plan: Plan, amount: float = 0.15, decay: float = 1.2, predelay_ms: float = 0.0) -> Plan:
    """Character stage: controlled ambience.

    `amount` is the dry/wet mix, 0.0-1.0 (stored as the document's `mix`
    field); `decay` is the target decay time in seconds, > 0 (stored as
    `decay_s`); `predelay_ms` delays the onset of the reverb relative to
    the dry signal, >= 0. `amount`/`decay` are this CLI/library's argument
    names for `mix`/`decay_s` -- the same naming seam `compress`'s
    `--bands` already documents (contracts/plan.v1.md's field names are
    the contract; these argument names are not).
    """
    if not (0.0 <= amount <= 1.0):
        raise AudError(
            code="bad_param",
            message=f"Reverb amount must be between 0.0 and 1.0, got {amount}.",
            remedy="Use an amount between 0 and 1, e.g. --amount 0.15.",
        )
    if not math.isfinite(decay) or decay <= 0:
        raise AudError(
            code="bad_param",
            message=f"Decay must be a positive number of seconds, got {decay}.",
            remedy="Use a decay greater than 0, e.g. --decay 1.2.",
        )
    if not math.isfinite(predelay_ms) or predelay_ms < 0:
        raise AudError(
            code="bad_param",
            message=f"Pre-delay must be a non-negative number of milliseconds, got {predelay_ms}.",
            remedy="Use a pre-delay of 0 or more, e.g. --predelay 0.",
        )
    return append(plan, "reverb", {"mix": amount, "decay_s": decay, "predelay_ms": predelay_ms})


def stretch(plan: Plan, factor: float = 1.0) -> Plan:
    """Retime a programme without changing its pitch. `factor` > 0 (1.0 = no change).

    Stored as the document's `ratio` field (contracts/plan.v1.md); `factor`
    is this CLI/library's argument name for it.
    """
    if not math.isfinite(factor) or factor <= 0:
        raise AudError(
            code="bad_param",
            message=f"Stretch factor must be positive, got {factor}.",
            remedy="Use a factor greater than 0, e.g. --factor 1.0 for no change.",
        )
    return append(plan, "stretch", {"ratio": factor})


def pitch(plan: Plan, semitones: float = 0.0) -> Plan:
    """Re-pitch a programme without changing its timing. `semitones` in [-24, 24]."""
    if not (-24.0 <= semitones <= 24.0):
        raise AudError(
            code="bad_param",
            message=f"Pitch shift must be between -24 and 24 semitones, got {semitones}.",
            remedy="Use a shift in that range, e.g. --semitones -2.",
        )
    return append(plan, "pitch", {"semitones": semitones})


def loudness(plan: Plan, target_lufs: float = -14.0) -> Plan:
    """Set the loudness stage's target. `target_lufs` in [-60, 0]."""
    if not (-60.0 <= target_lufs <= 0.0):
        raise AudError(
            code="bad_param",
            message=f"Target loudness must be between -60 and 0 LUFS, got {target_lufs}.",
            remedy="Use a target in that range, e.g. --target -14.",
        )
    return append(plan, "loudness", {"target_lufs": target_lufs})


def limit(plan: Plan, ceiling_dbtp: float = -1.0) -> Plan:
    """Set the true-peak brickwall ceiling. `ceiling_dbtp` must not exceed 0.0."""
    if ceiling_dbtp > 0.0:
        raise AudError(
            code="bad_param",
            message=f"True-peak ceiling must not exceed 0.0 dBTP, got {ceiling_dbtp}.",
            remedy="Use a ceiling at or below 0.0 dBTP, e.g. --ceiling -1.0.",
        )
    return append(plan, "limit", {"ceiling_dbtp": ceiling_dbtp})


# ---------------------------------------------------------------------------
# Sample-touching functions -- import aud.dsp lazily, inside the function body.
# ---------------------------------------------------------------------------


def _not_implemented(feature: str, exc: Exception) -> AudError:
    return AudError(
        code="not_implemented",
        message=f"'{feature}' needs aud.dsp, which is not available in this build: {exc}",
        remedy="Install/build the aud.dsp backend, or retry once it has landed; deterministic plan-building verbs work without it.",
    )


def _read_audio(io_module: Any, path: str) -> tuple[Any, int]:
    """Read an audio file, mapping io's I/O exceptions to a user-facing AudError.

    A missing or undecodable file is wrong input, not an internal aud bug --
    see AudError's docstring. `io.read_audio` raises plain `FileNotFoundError`
    and `RuntimeError`, by design (dsp/ modules do not raise user-facing
    errors -- AGENTS.md #8); this is the boundary where those become the two
    `AudError` codes a caller can act on. Anything else is left to propagate
    and is a genuine internal_error.
    """
    try:
        return io_module.read_audio(path)
    except FileNotFoundError as exc:
        raise AudError(
            code="file_not_found",
            message=str(exc),
            remedy=f"Check that '{path}' exists and is a readable audio file.",
        ) from exc
    except RuntimeError as exc:
        raise AudError(
            code="audio_decode_error",
            message=str(exc),
            remedy=(
                "Check that the file is a valid, uncorrupted audio file in a supported format. "
                "WAV/FLAC/AIFF need nothing extra; mp3/m4a/ogg need ffmpeg on PATH -- see "
                "docs/CONFIGURATION.md."
            ),
        ) from exc


def read_text_file(path: str) -> str:
    """Read a text file, mapping I/O failures to an AudError.

    Every path a caller hands us is read through here, so a missing or
    unreadable file always names the path as the problem instead of reaching
    the CLI's catch-all and being reported as an internal bug in `aud`.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AudError(
            code="file_not_found",
            message=str(exc),
            remedy=f"Check that '{path}' exists.",
        ) from exc
    except OSError as exc:
        raise AudError(
            code="file_not_found",
            message=f"Could not read '{path}': {exc}",
            remedy=f"Check that '{path}' exists and is readable.",
        ) from exc


def load_json_file(path: str) -> Any:
    """Read and parse a JSON file, mapping I/O and parse failures to an AudError.

    Shared by `curve_apply` and the CLI's `eq-match --curve` handling, so a
    missing or unparsable file gets the same honest `file_not_found`/
    `bad_param` envelope wherever it is read, rather than surfacing as an
    `internal_error` the caller has no way to act on.
    """
    text = read_text_file(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AudError(
            code="bad_param",
            message=f"'{path}' is not valid JSON: {exc}",
            remedy="Provide a curve JSON file produced by 'aud curve extract'.",
        ) from exc


def analyze(path: str) -> dict:
    """What is actually in a file: LUFS, true peak, crest, bands, sibilance, ambience."""
    try:
        from aud.dsp import analysis, io
    except ImportError as exc:
        raise _not_implemented("analyze", exc) from exc
    samples, sample_rate = _read_audio(io, path)
    return analysis.analyze(samples, sample_rate)


def render(plan: Plan, in_path: str, out_path: str) -> dict:
    """Apply a whole plan to a file in one decode/filter/encode pass."""
    try:
        from aud.dsp import engine, io
    except ImportError as exc:
        raise _not_implemented("render", exc) from exc
    from aud.dsp.edit import CrossfadeExceedsGapError
    from aud.dsp.engine import MissingDspModuleError
    from aud.dsp.eqmatch import CurveError
    from aud.plan import ordered

    samples, sample_rate = _read_audio(io, in_path)
    try:
        rendered, report = engine.apply_plan(samples, sample_rate, ordered(plan))
    except NotImplementedStageError as exc:
        raise AudError(
            code="not_implemented",
            message=f"'{exc.stage}' is not built yet in this release of aud.",
            remedy=(
                f"Build the chain with implemented stages only ({', '.join(exc.implemented)}); "
                f"'{exc.stage}' lands in a later release."
            ),
        ) from exc
    except MissingDspModuleError as exc:
        raise AudError(
            code="not_implemented",
            message=str(exc),
            remedy=(
                f"'{exc.needs}' has not landed in this build yet; render without it "
                f"(avoid 'strip_silence' or '--snap transient' in '{exc.stage}') until it does."
            ),
        ) from exc
    except CrossfadeExceedsGapError as exc:
        raise AudError(
            code="crossfade_exceeds_gap",
            message=str(exc),
            remedy="Use a shorter crossfade_ms, or supply regions with more material around that join.",
        ) from exc
    except CurveError as exc:
        raise AudError(
            code="bad_param",
            message=f"eq_match: {exc}",
            remedy="Re-extract the curve from this render's own sample rate, or supply a curve whose "
            "frequencies fit under this file's Nyquist frequency.",
        ) from exc
    output_subtype = config()["output_subtype"]["value"]
    io.write_audio(out_path, rendered, sample_rate, subtype=output_subtype)
    return {"out_path": out_path, "report": report}


def verify(path: str, target_lufs: float | None = None, ceiling_dbtp: float | None = None) -> dict:
    """Measure a render against the loudness/ceiling it was asked for."""
    try:
        from aud.dsp import analysis, io
    except ImportError as exc:
        raise _not_implemented("verify", exc) from exc
    samples, sample_rate = _read_audio(io, path)
    measured = analysis.analyze(samples, sample_rate)
    result: dict[str, Any] = {"measured": measured}
    if target_lufs is not None:
        result["target_lufs"] = target_lufs
        measured_lufs = measured.get("integrated_lufs")
        result["lufs_ok"] = measured_lufs is not None and abs(measured_lufs - target_lufs) <= 0.5
    if ceiling_dbtp is not None:
        result["ceiling_dbtp"] = ceiling_dbtp
        measured_ceiling = measured.get("true_peak_dbtp")
        result["ceiling_ok"] = measured_ceiling is not None and measured_ceiling <= ceiling_dbtp
    return result


def detect_silence(path: str, threshold_above_floor_db: float = 6.0, min_len_ms: float = 400.0) -> Any:
    """Find quiet spans, relative to the file's own measured noise floor.

    Emits a regions document (`kind="silence"`), never a plan -- see
    contracts/regions.v1.md. Read-only: opens `path` and writes nothing.
    """
    try:
        from aud.dsp import detect, io
    except ImportError as exc:
        raise _not_implemented("detect silence", exc) from exc
    from aud.core.regions import new_regions

    samples, sample_rate = _read_audio(io, path)
    noise_floor_dbfs = detect.measure_noise_floor(samples, sample_rate)
    regions = detect.detect_silence(
        samples, sample_rate, threshold_above_floor_db=threshold_above_floor_db, min_len_ms=min_len_ms
    )
    detection = {
        "threshold_above_floor_db": threshold_above_floor_db,
        "min_len_ms": min_len_ms,
        "noise_floor_dbfs": noise_floor_dbfs,
    }
    return new_regions(kind="silence", source=path, sample_rate=sample_rate, detection=detection, regions=regions)


def detect_transients(path: str, sensitivity: float = 1.0, min_gap_ms: float = 50.0) -> Any:
    """Find onset positions, with a strength per onset.

    Emits a regions document (`kind="transient"`), never a plan. Read-only.
    """
    try:
        from aud.dsp import detect, io
    except ImportError as exc:
        raise _not_implemented("detect transients", exc) from exc
    from aud.core.regions import new_regions

    samples, sample_rate = _read_audio(io, path)
    regions = detect.detect_transients(samples, sample_rate, sensitivity=sensitivity, min_separation_ms=min_gap_ms)
    detection = {"sensitivity": sensitivity, "min_gap_ms": min_gap_ms}
    return new_regions(kind="transient", source=path, sample_rate=sample_rate, detection=detection, regions=regions)


def detect_fillers(path: str, words: list[str] | None = None, min_pause_ms: float = 700.0) -> Any:
    """Find filler words ('umm', 'uh', 'ehm') and long hesitations.

    Emits a regions document (`kind="filler"`), never a plan. Needs the
    optional `speech` extra (faster-whisper); with it absent, raises
    `AudError(code="speech_extra_missing")` -- see `aud.dsp.speech`. Never
    degrades to an energy-only guess.
    """
    try:
        from aud.dsp import io
    except ImportError as exc:
        raise _not_implemented("detect fillers", exc) from exc
    from aud.core.regions import new_regions
    from aud.dsp import speech

    samples, sample_rate = _read_audio(io, path)
    regions, detection = speech.detect_fillers(samples, sample_rate, words=words, min_pause_ms=min_pause_ms)
    return new_regions(kind="filler", source=path, sample_rate=sample_rate, detection=detection, regions=regions)


def curve_extract(path: str, out: str) -> dict:
    """Extract a spectral profile from `path` and save it as JSON at `out`."""
    try:
        from aud.dsp import eqmatch, io
    except ImportError as exc:
        raise _not_implemented("curve_extract", exc) from exc
    samples, sample_rate = _read_audio(io, path)
    curve = eqmatch.spectrum_profile(samples, sample_rate)
    Path(out).write_text(json.dumps(curve), encoding="utf-8")
    return {"curve_path": out}


def curve_apply(path: str, curve_path: str, out_path: str) -> dict:
    """Apply a saved spectral curve to `path`, writing the result to `out_path`."""
    try:
        from aud.dsp import eqmatch, io
    except ImportError as exc:
        raise _not_implemented("curve_apply", exc) from exc
    curve = load_json_file(curve_path)
    samples, sample_rate = _read_audio(io, path)
    try:
        matched = eqmatch.apply_curve(samples, sample_rate, curve)
    except eqmatch.CurveError as exc:
        # dsp/ modules raise plain exceptions (AGENTS.md #8); this is the
        # boundary where a malformed/incompatible curve becomes a
        # user-facing bad_param instead of an unhandled internal_error.
        raise AudError(
            code="bad_param",
            message=f"Could not apply curve '{curve_path}' to '{path}': {exc}",
            remedy="Provide a curve JSON with at least 2 finite, strictly ascending [freq_hz, gain_db] "
            "pairs, as produced by 'aud curve extract'.",
        ) from exc
    io.write_audio(out_path, matched, sample_rate)
    return {"out_path": out_path}
