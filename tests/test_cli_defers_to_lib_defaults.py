"""Regression guard for cli-is-thin: the CLI must DEFER to aud.lib's own
stage defaults, not merely match them with an independent copy.

Before this fix, every stage-verb parser default was a literal duplicate of
the corresponding `aud.lib` function's own default (e.g. `gate --threshold`
defaulted to 12.0 in cli.py AND threshold_above_floor_db defaulted to 12.0 in
lib.gate), and `_dispatch_stage` always forwarded `args.threshold`
unconditionally -- so changing `lib.gate`'s default would never change what
the CLI actually did. A test that just asserts "cli.py's 12.0 == lib.py's
12.0" would still pass in that broken state; it proves nothing about which
one is actually in control.

These tests prove control, not correspondence:

1. When a flag is omitted, `aud.lib`'s function is called with that keyword
   argument entirely ABSENT -- not re-supplied with a copy of today's
   number -- so a future change to the library's own default takes effect
   with no CLI change required.
2. When a flag IS given, the explicit value still wins.

See `aud.cli._defer` and its call sites in `_dispatch_stage`/`_dispatch`.
"""

from __future__ import annotations

from typing import Any

from aud import lib
from aud.cli import _build_parser, _dispatch


def test_gate_cli_omits_untouched_flags_so_the_library_default_governs(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_gate(plan: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return plan

    monkeypatch.setattr(lib, "gate", fake_gate)

    parser = _build_parser()
    args = parser.parse_args(["gate"])  # no flags at all
    _dispatch("gate", args, None)

    # Every one of gate's answer-changing parameters must be ABSENT from the
    # call -- not forwarded with a value that happens to match lib.gate's
    # current default -- proving lib.gate's own default is what will apply.
    for key in (
        "threshold_above_floor_db",
        "range_db",
        "attack_ms",
        "hold_ms",
        "release_ms",
        "lookahead_ms",
        "sidechain_hpf_hz",
    ):
        assert key not in captured, f"CLI forwarded '{key}' even though the user never passed a flag for it"


def test_gate_cli_still_forwards_a_flag_the_user_actually_passed(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_gate(plan: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return plan

    monkeypatch.setattr(lib, "gate", fake_gate)

    parser = _build_parser()
    args = parser.parse_args(["gate", "--threshold", "42"])
    _dispatch("gate", args, None)

    assert captured["threshold_above_floor_db"] == 42.0
    assert "range_db" not in captured  # untouched flags still stay absent


def test_gate_cli_effective_behaviour_follows_a_changed_library_default(monkeypatch) -> None:
    """The strongest form of the proof: change lib.gate's own default and
    show the CLI's rendered plan changes with it, with no CLI edit at all.
    """

    def fake_gate_with_new_default(plan_arg: Any, **kwargs: Any) -> Any:
        # Simulates a future aud.lib.gate whose own default changed from
        # 12.0 to 999.0 -- a value the OLD cli.py's hardcoded `default=12.0`
        # could never produce on its own.
        kwargs.setdefault("threshold_above_floor_db", 999.0)
        from aud.plan import append

        return append(plan_arg, "gate", kwargs)

    monkeypatch.setattr(lib, "gate", fake_gate_with_new_default)

    parser = _build_parser()
    args = parser.parse_args(["gate"])  # --threshold not given
    result_plan = _dispatch("gate", args, None)

    stage = result_plan.stages[-1]
    assert stage.stage == "gate"
    assert stage.params["threshold_above_floor_db"] == 999.0


def test_expand_cli_omits_untouched_flags_so_the_library_default_governs(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_expand(plan: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return plan

    monkeypatch.setattr(lib, "expand", fake_expand)

    parser = _build_parser()
    args = parser.parse_args(["expand"])
    _dispatch("expand", args, None)

    for key in ("threshold_above_floor_db", "ratio", "knee_db", "attack_ms", "hold_ms", "release_ms"):
        assert key not in captured, f"CLI forwarded '{key}' even though the user never passed a flag for it"


def test_loudness_and_limit_cli_omit_untouched_flags(monkeypatch) -> None:
    """Same contract, spot-checked on two more stages (not just gate/expand)
    -- the reviewer flagged this as a pattern across every stage verb.
    """
    captured: dict[str, dict[str, Any]] = {"loudness": {}, "limit": {}}

    def fake_loudness(plan: Any, **kwargs: Any) -> Any:
        captured["loudness"].update(kwargs)
        return plan

    def fake_limit(plan: Any, **kwargs: Any) -> Any:
        captured["limit"].update(kwargs)
        return plan

    monkeypatch.setattr(lib, "loudness", fake_loudness)
    monkeypatch.setattr(lib, "limit", fake_limit)

    parser = _build_parser()
    _dispatch("loudness", parser.parse_args(["loudness"]), None)
    _dispatch("limit", parser.parse_args(["limit"]), None)

    assert "target_lufs" not in captured["loudness"]
    assert "ceiling_dbtp" not in captured["limit"]
