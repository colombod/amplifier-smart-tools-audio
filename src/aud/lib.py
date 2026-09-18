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
import os
import platform
import shutil
import tomllib
from pathlib import Path
from typing import Any

from aud.core.manifest import load_manifest
from aud.core.skill import render_skill
from aud.plan import Plan, append
from aud.schemas import AudError

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


def deess(plan: Plan, amount: float = 6.0, freq: float = 6000.0) -> Plan:
    """Repair stage: tame sibilance. `amount` in dB of reduction, `freq` in Hz."""
    if not (0.0 <= amount <= 24.0):
        raise AudError(
            code="bad_param",
            message=f"De-ess amount must be between 0 and 24 dB, got {amount}.",
            remedy="Use an amount in that range, e.g. --amount 6.",
        )
    if freq <= 0:
        raise AudError(
            code="bad_param",
            message=f"De-ess frequency must be a positive number of Hz, got {freq}.",
            remedy="Use a frequency above 0 Hz, e.g. --freq 6000.",
        )
    return append(plan, "deess", {"amount": amount, "freq": freq})


def dereverb(plan: Plan, amount: float = 50.0) -> Plan:
    """Repair stage: reduce room ambience. `amount` as a percentage, 0-100."""
    if not (0.0 <= amount <= 100.0):
        raise AudError(
            code="bad_param",
            message=f"De-reverb amount must be between 0 and 100, got {amount}.",
            remedy="Use a percentage between 0 and 100, e.g. --amount 50.",
        )
    return append(plan, "dereverb", {"amount": amount})


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


def eq_match(plan: Plan, curve: dict, mix: float = 1.0) -> Plan:
    """Tone stage: match this file's tonal balance to a reference curve.

    `curve` is a spectral profile as produced by `curve_extract`. `mix` is
    how much of the match to apply, 0.0 (none) to 1.0 (full).
    """
    if not isinstance(curve, dict) or not curve:
        raise AudError(
            code="bad_param",
            message="EQ-match curve must be a non-empty object.",
            remedy="Pass a curve produced by 'aud curve extract'.",
        )
    if not (0.0 <= mix <= 1.0):
        raise AudError(
            code="bad_param",
            message=f"Mix must be between 0.0 and 1.0, got {mix}.",
            remedy="Use a mix between 0 and 1, e.g. --mix 1.0.",
        )
    return append(plan, "eq_match", {"curve": curve, "mix": mix})


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


def reverb(plan: Plan, amount: float = 0.2, decay: float = 1.5) -> Plan:
    """Character stage: controlled ambience. `amount` in [0, 1], `decay` in seconds > 0."""
    if not (0.0 <= amount <= 1.0):
        raise AudError(
            code="bad_param",
            message=f"Reverb amount must be between 0.0 and 1.0, got {amount}.",
            remedy="Use an amount between 0 and 1, e.g. --amount 0.2.",
        )
    if decay <= 0:
        raise AudError(
            code="bad_param",
            message=f"Decay must be a positive number of seconds, got {decay}.",
            remedy="Use a decay greater than 0, e.g. --decay 1.5.",
        )
    return append(plan, "reverb", {"amount": amount, "decay": decay})


def stretch(plan: Plan, factor: float = 1.0) -> Plan:
    """Retime a programme without changing its pitch. `factor` > 0 (1.0 = no change)."""
    if factor <= 0:
        raise AudError(
            code="bad_param",
            message=f"Stretch factor must be positive, got {factor}.",
            remedy="Use a factor greater than 0, e.g. --factor 1.0 for no change.",
        )
    return append(plan, "stretch", {"factor": factor})


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


def analyze(path: str) -> dict:
    """What is actually in a file: LUFS, true peak, crest, bands, sibilance, ambience."""
    try:
        from aud.dsp import analysis, io
    except ImportError as exc:
        raise _not_implemented("analyze", exc) from exc
    samples, sample_rate = io.read_audio(path)
    return analysis.analyze(samples, sample_rate)


def render(plan: Plan, in_path: str, out_path: str) -> dict:
    """Apply a whole plan to a file in one decode/filter/encode pass."""
    try:
        from aud.dsp import engine, io
    except ImportError as exc:
        raise _not_implemented("render", exc) from exc
    from aud.plan import ordered

    samples, sample_rate = io.read_audio(in_path)
    rendered, report = engine.apply_plan(samples, sample_rate, ordered(plan))
    output_subtype = config()["output_subtype"]["value"]
    io.write_audio(out_path, rendered, sample_rate, subtype=output_subtype)
    return {"out_path": out_path, "report": report}


def verify(path: str, target_lufs: float | None = None, ceiling_dbtp: float | None = None) -> dict:
    """Measure a render against the loudness/ceiling it was asked for."""
    try:
        from aud.dsp import analysis, io
    except ImportError as exc:
        raise _not_implemented("verify", exc) from exc
    samples, sample_rate = io.read_audio(path)
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


def curve_extract(path: str, out: str) -> dict:
    """Extract a spectral profile from `path` and save it as JSON at `out`."""
    try:
        from aud.dsp import eqmatch, io
    except ImportError as exc:
        raise _not_implemented("curve_extract", exc) from exc
    samples, sample_rate = io.read_audio(path)
    curve = eqmatch.spectrum_profile(samples, sample_rate)
    Path(out).write_text(json.dumps(curve), encoding="utf-8")
    return {"curve_path": out}


def curve_apply(path: str, curve_path: str, out_path: str) -> dict:
    """Apply a saved spectral curve to `path`, writing the result to `out_path`."""
    try:
        from aud.dsp import eqmatch, io
    except ImportError as exc:
        raise _not_implemented("curve_apply", exc) from exc
    curve = json.loads(Path(curve_path).read_text(encoding="utf-8"))
    samples, sample_rate = io.read_audio(path)
    matched = eqmatch.apply_curve(samples, sample_rate, curve)
    io.write_audio(out_path, matched, sample_rate)
    return {"out_path": out_path}
