"""Measurements in, a validated mastering plan out.

The model sees only `aud analyze`'s measurement report (and, optionally, a
reference file's own measurement report) -- never raw audio, never a
sample. Its response is untrusted input: parsed as JSON, then built into a
`Plan` through the exact same stage builders every other verb uses
(`aud.lib.eq`, `aud.lib.compress`, ...), so a malformed or out-of-range
proposal fails the same honest way a hand-written bad plan would -- see
docs/ARCHITECTURE.md #7.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

from aud import lib
from aud.intelligence import prompts
from aud.intelligence.interface import IntelligenceBackend
from aud.plan import Plan, new_plan
from aud.schemas import AudError

__all__ = ["ALLOWED_STAGES", "advise"]

# Exactly the stages a measurement report alone can justify. `cut` and
# `strip_silence` need detected regions, not measurements; `eq_match` needs
# a stored curve (not something advise ever holds); `stretch`/`pitch` are
# not decisions a loudness/spectral report has any basis to make. Out of
# scope for advise by design, not oversight -- see SMART_TOOL.md.
ALLOWED_STAGES: tuple[str, ...] = (
    "gate",
    "expand",
    "deess",
    "dereverb",
    "eq",
    "compress",
    "saturate",
    "reverb",
    "loudness",
    "limit",
)

# Every one of these is a same-named function on aud.lib, with the exact
# kwarg names prompts.py's stage reference documents -- so a validated
# proposal applies with a plain **params call, no translation layer between
# what the model said and what the builder wants.
_BUILDERS: dict[str, Callable[..., Plan]] = {name: getattr(lib, name) for name in ALLOWED_STAGES}

_DEFAULT_MAX_TOKENS = 2000


def _strip_code_fence(text: str) -> str:
    """Tolerate a model wrapping its JSON in ```json ... ``` anyway."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_model_json(raw: str) -> Any:
    text = _strip_code_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AudError(
            code="bad_model_output",
            message=f"Model response was not valid JSON: {exc}",
            remedy="Retry advise/master, or build the chain by hand with the deterministic stage verbs.",
        ) from exc


# --- Deterministic guard: an EQ move must not contradict its own signed
# measurement --------------------------------------------------------------
#
# Root cause this guards against (measured, not hypothetical): given
# `rel_reference_db` -- a SIGNED per-band deviation from a reference file,
# where positive means "this file carries MORE energy than the reference in
# that band" (a candidate for a CUT) and negative means "LESS energy" (a
# candidate for a BOOST) -- a real model read the number and its own prose
# correctly ("16 kHz is -36.79 dB below reference... extreme high-frequency
# rolloff... lacks presence") and then proposed a further CUT at that exact
# frequency. `prompts.py` already states the sign convention explicitly;
# rewording the prompt again is unverifiable and costs a model call to test.
# This check is free and deterministic -- it runs on parsed JSON, no model
# call, no audio.
#
# Guarded: `peaks` and `shelves`. Each carries an explicit (freq_hz,
# gain_db) pair that maps directly onto one measured band's signed
# deviation -- the exact shape the contradiction is defined over.
#
# NOT guarded: `hpf`/`lpf`. Both are broadband corner frequencies with no
# `gain_db` of their own -- the attenuation they apply varies continuously
# with frequency rather than being one number at one band, so there is no
# well-defined sign to compare against a single band's measurement the way
# there is for a peak's or a shelf's stated gain. Forcing that mapping would
# invent a semantics the model was never asked to use.
#
# Response to a contradiction: DROP the offending move (never silently flip
# its sign -- that would rewrite what the model actually proposed into
# something it never said, which is how a tool starts lying about its own
# provenance) and report it as an extra `{"stage": "eq", "reason": "GUARDED: ..."}`
# entry in the same `reasoning` list every other stage's rationale already
# travels in -- printed to stderr by the CLI (`_print_advise_reasoning`) and
# returned to every library caller (`advise()`'s `"stages"` key) exactly
# like any other stage's reason. This is chosen over failing the whole
# `advise` call: one wrong sign on one band should not discard an otherwise
# good chain (a correct `loudness`/`limit`, or a correct peak/shelf on a
# different band) -- and dropping is not a repair, it is simply declining to
# apply a move that contradicts the very measurement cited to justify it.
_EQ_GUARD_TOLERANCE_DB = 0.5  # ignore sign noise near zero; every real contradiction found so far is >= 4 dB


def _octave_reference_lookup(measurements: dict[str, Any] | None) -> dict[float, float] | None:
    """Band-centre Hz -> `rel_reference_db`, or `None` if no band carries one.

    `rel_reference_db` is ABSENT (not null) from every band entry when no
    `reference_path` was supplied to `advise` -- see `dsp.analysis.analyze`'s
    docstring. Returning `None` here (rather than an empty dict) is what
    makes the guard below a strict no-op in exactly that case: behaviour
    must be unchanged with no reference, and must never fall back to
    guarding against `rel_median_db` -- the file's-own-median anchor this
    whole signal exists to replace (1 of 6 vs. 6 of 6 correct on the same
    fixtures; see `reference_anchored_band_deviation_db`'s docstring).
    """
    entries = (measurements or {}).get("octave_band_analysis") or []
    lookup = {
        float(entry["hz"]): float(entry["rel_reference_db"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("rel_reference_db") is not None
    }
    return lookup or None


def _nearest_band_hz(freq_hz: float, band_centers: list[float]) -> float:
    """The measured octave-band centre closest to `freq_hz`, in OCTAVES
    (log2 frequency ratio) -- not linear Hz distance.

    `dsp.analysis._octave_band_energy_db` gives each band a contiguous
    half-octave span either side of its centre (`[centre/sqrt(2),
    centre*sqrt(2)]`), so adjacent bands meet exactly at their
    geometric-mean boundary: "nearest centre in log2 space" and "which
    band's half-octave span contains this frequency" are the same rule.
    Example: a move at 6000 Hz sits between the 4000 Hz and 8000 Hz bands;
    their shared boundary is sqrt(4000*8000) ~= 5657 Hz, so 6000 Hz falls
    inside (and is nearer, in octaves, to) the 8000 Hz band --
    log2(6000/4000) = 0.585 oct vs. log2(8000/6000) = 0.415 oct.
    """
    return min(band_centers, key=lambda center: abs(math.log2(freq_hz / center)))


def _eq_move_contradiction(freq_hz: float, gain_db: float, reference_lookup: dict[float, float]) -> str | None:
    """`None` if `gain_db`'s sign agrees with the nearest band's measured
    `rel_reference_db` (or there is nothing to contradict); else a
    human-readable description of the contradiction.

    Agreement: `rel_reference_db` > tolerance (this band carries MORE
    energy than the reference -- a candidate for a CUT) pairs with
    `gain_db` < 0. `rel_reference_db` < -tolerance (LESS energy -- a
    candidate for a BOOST) pairs with `gain_db` > 0. Within +/-tolerance of
    zero, or `gain_db` == 0, there is nothing to contradict.

    `freq_hz <= 0` is likewise treated as nothing to contradict (not a
    crash): a non-positive frequency has no octave-band mapping at all
    (`math.log2` of a non-positive ratio is undefined), and it is not this
    guard's job to reject it -- `eq()`'s own validation raises
    `bad_model_plan` for exactly that, unchanged, downstream.
    """
    if gain_db == 0 or freq_hz <= 0:
        return None
    band_hz = _nearest_band_hz(freq_hz, list(reference_lookup))
    deviation = reference_lookup[band_hz]
    if deviation > _EQ_GUARD_TOLERANCE_DB:
        correct_direction = "cut"
    elif deviation < -_EQ_GUARD_TOLERANCE_DB:
        correct_direction = "boost"
    else:
        return None
    proposed_direction = "cut" if gain_db < 0 else "boost"
    if proposed_direction == correct_direction:
        return None
    excess_or_deficient = (
        "excess vs reference, a candidate for a cut"
        if correct_direction == "cut"
        else ("deficient vs reference, a candidate for a boost")
    )
    return (
        f"{band_hz:g} Hz is measured {deviation:+.2f} dB rel_reference_db ({excess_or_deficient}), "
        f"but the proposed move at {freq_hz:g} Hz is a {gain_db:+.2f} dB {proposed_direction} -- "
        "contradicts the measurement."
    )


def _filter_contradictory_eq_moves(
    params: dict[str, Any], reference_lookup: dict[float, float]
) -> tuple[dict[str, Any], list[str]]:
    """Drop any `peaks`/`shelves` entry whose gain contradicts its nearest
    band's `rel_reference_db`; return the filtered params plus one
    human-readable note per dropped move.

    Defensive, not a second validator: an entry that is not a well-shaped
    sequence of numbers is left exactly as given, so `eq()`'s own
    shape/range validation still raises the same `bad_model_plan` it always
    has for a malformed proposal -- this guard only ever REMOVES a
    well-shaped, self-contradictory move; it never rejects a malformed one
    (that is `eq()`'s job, unchanged).
    """
    notes: list[str] = []

    def _keep_peak(peak: Any) -> bool:
        if not isinstance(peak, (list, tuple)) or len(peak) != 3:
            return True
        freq_hz, gain_db = peak[0], peak[1]
        if not isinstance(freq_hz, (int, float)) or not isinstance(gain_db, (int, float)):
            return True
        note = _eq_move_contradiction(float(freq_hz), float(gain_db), reference_lookup)
        if note is None:
            return True
        notes.append(f"peak: {note}")
        return False

    def _keep_shelf(shelf: Any) -> bool:
        if not isinstance(shelf, (list, tuple)) or len(shelf) != 4:
            return True
        freq_hz, gain_db = shelf[1], shelf[2]
        if not isinstance(freq_hz, (int, float)) or not isinstance(gain_db, (int, float)):
            return True
        note = _eq_move_contradiction(float(freq_hz), float(gain_db), reference_lookup)
        if note is None:
            return True
        notes.append(f"shelf: {note}")
        return False

    filtered = dict(params)
    peaks = params.get("peaks")
    if isinstance(peaks, list):
        filtered["peaks"] = [peak for peak in peaks if _keep_peak(peak)]
    shelves = params.get("shelves")
    if isinstance(shelves, list):
        filtered["shelves"] = [shelf for shelf in shelves if _keep_shelf(shelf)]
    return filtered, notes


def _validate_and_build_plan(
    proposal: Any, measurements: dict[str, Any] | None = None
) -> tuple[Plan, list[dict[str, str]]]:
    """Turn an untrusted parsed JSON value into a validated Plan + reasoning.

    Every failure here is `AudError(code="bad_model_plan")` -- a malformed
    or out-of-range proposal is reported honestly, never silently repaired
    or allowed to become a plan that would blow up later at render time.

    `measurements` (optional; the file-being-mastered's own `analyze()`
    report) feeds the EQ sign-contradiction guard above: when it carries at
    least one `rel_reference_db` (i.e. a reference file was supplied), any
    `eq` peak/shelf whose gain contradicts its band's measured deviation is
    dropped -- see `_filter_contradictory_eq_moves`. With no measurements,
    or none carrying `rel_reference_db`, this is a strict no-op.
    """
    if not isinstance(proposal, dict) or set(proposal) != {"stages"}:
        raise AudError(
            code="bad_model_plan",
            message=f"Model response must be a JSON object with exactly the key 'stages'; got {proposal!r}",
            remedy='Expected shape: {"stages": [{"stage", "params", "reason"}, ...]}.',
        )
    stages = proposal["stages"]
    if not isinstance(stages, list) or not stages:
        raise AudError(
            code="bad_model_plan",
            message=f"'stages' must be a non-empty array, got {stages!r}",
            remedy="Propose at least one stage; an empty chain is not advice.",
        )

    reference_lookup = _octave_reference_lookup(measurements)
    plan = new_plan()
    reasoning: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(stages):
        if not isinstance(entry, dict) or set(entry) != {"stage", "params", "reason"}:
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}] must have exactly 'stage', 'params' and 'reason'; got {entry!r}",
                remedy='Each stage entry is {"stage": str, "params": object, "reason": str}.',
            )
        stage_name = entry["stage"]
        params = entry["params"]
        reason = entry["reason"]

        if not isinstance(stage_name, str) or stage_name not in ALLOWED_STAGES:
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}].stage is {stage_name!r}; must be one of {ALLOWED_STAGES}.",
                remedy=f"Use one of: {', '.join(ALLOWED_STAGES)}.",
            )
        if stage_name in seen:
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}] proposes '{stage_name}' a second time; each stage may appear at most once.",
                remedy="Combine everything for one stage into a single entry.",
            )
        seen.add(stage_name)
        if not isinstance(params, dict):
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}].params must be an object, got {type(params).__name__}.",
                remedy=f"See the '{stage_name}' parameter shape the advisor was given.",
            )
        if not isinstance(reason, str) or not reason.strip():
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}].reason must be a non-empty string grounded in a measurement.",
                remedy="State which measurement justifies this stage and these parameters.",
            )

        guard_notes: list[str] = []
        if stage_name == "eq" and reference_lookup is not None:
            params, guard_notes = _filter_contradictory_eq_moves(params, reference_lookup)

        builder = _BUILDERS[stage_name]
        try:
            plan = builder(plan, **params)
        except (TypeError, AudError) as exc:
            detail = exc.message if isinstance(exc, AudError) else str(exc)
            raise AudError(
                code="bad_model_plan",
                message=f"stages[{index}] ('{stage_name}') has invalid params {params!r}: {detail}",
                remedy="The model proposed invalid parameters; retry advise/master, or build the chain by hand.",
            ) from exc

        reasoning.append({"stage": stage_name, "reason": reason.strip()})
        for note in guard_notes:
            reasoning.append({"stage": stage_name, "reason": f"GUARDED: {note}"})

    return plan, reasoning


def advise(
    measurements: dict[str, Any],
    *,
    target_lufs: float,
    ceiling_dbtp: float,
    reference_measurements: dict[str, Any] | None,
    backend: IntelligenceBackend,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> tuple[Plan, list[dict[str, str]]]:
    """Ask `backend` to choose a mastering chain from `measurements`.

    Args:
        measurements: `aud.lib.analyze()`'s report for the file to master.
        target_lufs: The integrated-loudness target to tell the model about.
        ceiling_dbtp: The true-peak ceiling to tell the model about.
        reference_measurements: Optionally, `analyze()`'s report for a
            reference file, given to the model for tonal context only.
        backend: Anything implementing `IntelligenceBackend` -- a real
            provider backend, or a test's canned fake.
        model: The model name to ask `backend` for.
        max_tokens: Forwarded to `backend.complete`.

    Returns:
        (plan, reasoning) -- a validated `Plan` ready for `aud.lib.render`,
        and a parallel list of `{"stage", "reason"}` dicts, one per stage,
        in the order the model proposed them.

    Raises:
        AudError: code "bad_model_output" if the response is not JSON; code
            "bad_model_plan" if it is JSON but not a valid, in-range chain
            built only from `ALLOWED_STAGES`.
    """
    system = prompts.system_prompt()
    user = prompts.user_prompt(
        measurements,
        target_lufs=target_lufs,
        ceiling_dbtp=ceiling_dbtp,
        reference_measurements=reference_measurements,
    )
    raw = backend.complete(system, user, model=model, max_tokens=max_tokens)
    proposal = _parse_model_json(raw)
    return _validate_and_build_plan(proposal, measurements)
