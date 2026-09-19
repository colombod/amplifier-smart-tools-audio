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


def _validate_and_build_plan(proposal: Any) -> tuple[Plan, list[dict[str, str]]]:
    """Turn an untrusted parsed JSON value into a validated Plan + reasoning.

    Every failure here is `AudError(code="bad_model_plan")` -- a malformed
    or out-of-range proposal is reported honestly, never silently repaired
    or allowed to become a plan that would blow up later at render time.
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
    return _validate_and_build_plan(proposal)
