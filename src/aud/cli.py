"""Thin argparse CLI over aud.lib.

All logic lives in aud.lib; this module only parses arguments, calls into
the library, and prints exactly one JSON document to stdout. Progress and
diagnostics never go to stdout.

Output contract:
  Success:  {"result": ...}
  Failure:  {"error": {"code", "message", "remedy"}}, exit 1 (AudError) or
            exit 2 (usage error).
  Exception: 'plan' and every stage verb print the plan document itself,
  raw and unwrapped, to stdout, so verbs pipe into each other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, NoReturn

from aud import lib
from aud.core.regions import write_regions
from aud.plan import read_plan, write_plan
from aud.schemas import AudError
from aud.verbdoc import VERB_DOCS

STAGE_VERB_NAMES = (
    "cut",
    "strip-silence",
    "deess",
    "dereverb",
    "eq",
    "eq-match",
    "compress",
    "saturate",
    "reverb",
    "stretch",
    "pitch",
    "loudness",
    "limit",
)
# Verbs whose stdout is the plan document itself, unwrapped.
PLAN_OUTPUT_VERBS = frozenset({"plan", *STAGE_VERB_NAMES})
# Verbs that consume an incoming plan (read from stdin when stdin is not a TTY).
# `cut` is also here even though its stdin is sometimes a REGIONS document
# instead (see `_dispatch`'s special case for it) -- either way, `cut` needs
# stdin read once, up front, exactly like every other stage verb.
PLAN_CONSUMING_VERBS = frozenset({*STAGE_VERB_NAMES, "render"})
# Verbs whose stdout is a regions document (contracts/regions.v1.md) itself,
# unwrapped -- the read-only counterpart of PLAN_OUTPUT_VERBS, and what lets
# 'aud detect silence in.wav | aud cut | aud render in.wav out.wav' pipe with
# no unwrapping in between.
REGIONS_OUTPUT_VERBS = frozenset({"detect"})
# Verbs whose capability does not exist yet in this release. Their argument
# surface is still registered (explicitly, where they take arguments) so that a
# caller writing the eventual command gets `not_implemented` -- an answer about
# the capability -- rather than a usage error about a flag that will exist.
_NOT_YET_BUILT = frozenset({"preset", "advise", "master"})


class _UsageError(Exception):
    """Raised in place of argparse's default print-to-stderr-and-exit(2)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _triple(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected freq,gain_db,q (e.g. 3200,-2.5,1.4), got {text!r}")
    try:
        freq, gain_db, q = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected three numbers, got {text!r}") from exc
    return (freq, gain_db, q)


def _float_list(text: str) -> list[float]:
    try:
        return [float(part) for part in text.split(",") if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc


SNAP_MODES = ("zero_crossing", "silence", "transient", "none")
CROSSFADE_SHAPES = ("equal_power", "linear")


def _add_edit_point_arguments(parser: argparse.ArgumentParser, *, pad_default: float) -> None:
    """Register the edit-point resolution surface shared by `cut` and `strip-silence`.

    A region boundary is a NOMINAL position; where the blade falls is resolved
    from it -- padded, then snapped within a bounded window, then joined with a
    fade or a crossfade. Both editing stages take the identical set, and only
    the padding default differs (see the call sites).

    The value spellings are the document's spellings (`zero_crossing`, not
    `zero-crossing`), so there is no translation layer between what
    contracts/plan.v1.md promises and what the shell accepts.
    """
    parser.add_argument("--pad-out", dest="pad_out", type=float, default=pad_default)
    parser.add_argument("--pad-in", dest="pad_in", type=float, default=pad_default)
    parser.add_argument("--snap", choices=SNAP_MODES, default="zero_crossing")
    parser.add_argument("--snap-window", dest="snap_window", type=float, default=20.0)
    parser.add_argument("--fade-out", dest="fade_out", type=float, default=0.0)
    parser.add_argument("--fade-in", dest="fade_in", type=float, default=0.0)
    parser.add_argument("--crossfade", type=float, default=10.0)
    parser.add_argument("--crossfade-shape", dest="crossfade_shape", choices=CROSSFADE_SHAPES, default="equal_power")


def _build_parser() -> _Parser:
    parser = _Parser(prog="aud", add_help=True)
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("manifest")
    sub.add_parser("check")

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("path")

    config_parser = sub.add_parser("config")
    config_parser.add_argument("--sample-rate-policy", dest="sample_rate_policy", default=None)
    config_parser.add_argument("--default-ceiling-dbtp", dest="default_ceiling_dbtp", type=float, default=None)
    config_parser.add_argument("--default-target-lufs", dest="default_target_lufs", type=float, default=None)
    config_parser.add_argument("--oversample", dest="oversample", type=int, default=None)
    config_parser.add_argument("--output-subtype", dest="output_subtype", default=None)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--from", dest="from_file", default=None)

    deess_parser = sub.add_parser("deess")
    deess_parser.add_argument("--amount", type=float, default=6.0)
    deess_parser.add_argument("--freq", type=float, default=6000.0)

    dereverb_parser = sub.add_parser("dereverb")
    dereverb_parser.add_argument("--amount", type=float, default=50.0)

    eq_parser = sub.add_parser("eq")
    eq_parser.add_argument("--hpf", type=float, default=None)
    eq_parser.add_argument("--lpf", type=float, default=None)
    eq_parser.add_argument("--peak", dest="peaks", type=_triple, action="append", default=None)

    eq_match_parser = sub.add_parser("eq-match")
    eq_match_parser.add_argument("--curve", required=True)
    eq_match_parser.add_argument("--mix", type=float, default=1.0)

    compress_parser = sub.add_parser("compress")
    compress_parser.add_argument("--bands", type=_float_list, required=True)
    compress_parser.add_argument("--ratio", type=float, default=2.5)

    saturate_parser = sub.add_parser("saturate")
    saturate_parser.add_argument("--drive", type=float, default=1.0)
    saturate_parser.add_argument("--mix", type=float, default=0.25)

    reverb_parser = sub.add_parser("reverb")
    reverb_parser.add_argument("--amount", type=float, default=0.2)
    reverb_parser.add_argument("--decay", type=float, default=1.5)

    stretch_parser = sub.add_parser("stretch")
    stretch_parser.add_argument("--factor", type=float, default=1.0)

    pitch_parser = sub.add_parser("pitch")
    pitch_parser.add_argument("--semitones", type=float, default=0.0)

    loudness_parser = sub.add_parser("loudness")
    loudness_parser.add_argument("--target", type=float, default=-14.0)

    limit_parser = sub.add_parser("limit")
    limit_parser.add_argument("--ceiling", type=float, default=-1.0)

    curve_parser = sub.add_parser("curve")
    curve_sub = curve_parser.add_subparsers(dest="curve_action", required=True)
    curve_extract_parser = curve_sub.add_parser("extract")
    curve_extract_parser.add_argument("path")
    curve_extract_parser.add_argument("out")
    curve_apply_parser = curve_sub.add_parser("apply")
    curve_apply_parser.add_argument("path")
    curve_apply_parser.add_argument("curve")
    curve_apply_parser.add_argument("out")

    render_parser = sub.add_parser("render")
    render_parser.add_argument("in_path")
    render_parser.add_argument("out_path")

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("path")
    verify_parser.add_argument("--target", type=float, default=None)
    verify_parser.add_argument("--ceiling", type=float, default=None)

    # Read-only detection. Emits a regions document (contracts/regions.v1.md),
    # never a plan: these verbs find things, they do not schedule work.
    detect_parser = sub.add_parser("detect")
    detect_sub = detect_parser.add_subparsers(dest="detect_kind", required=True)
    detect_transients_parser = detect_sub.add_parser("transients")
    detect_transients_parser.add_argument("path")
    detect_transients_parser.add_argument("--sensitivity", type=float, default=1.0)
    detect_transients_parser.add_argument("--min-gap", dest="min_gap", type=float, default=50.0)
    detect_silence_parser = detect_sub.add_parser("silence")
    detect_silence_parser.add_argument("path")
    # dB ABOVE the file's measured noise floor, not an absolute dBFS value.
    detect_silence_parser.add_argument("--threshold", type=float, default=6.0)
    detect_silence_parser.add_argument("--min-len", dest="min_len", type=float, default=400.0)
    detect_fillers_parser = detect_sub.add_parser("fillers")
    detect_fillers_parser.add_argument("path")
    detect_fillers_parser.add_argument("--words", type=str, default="umm,uhm,uh,ehm,er,ah")
    detect_fillers_parser.add_argument("--min-pause", dest="min_pause", type=float, default=700.0)

    # Editing stages. These DO append to the plan, at the front of canonical order.
    cut_parser = sub.add_parser("cut")
    cut_parser.add_argument("--regions", default=None)
    # `cut` is handed positions a caller measured and means literally, so its
    # padding defaults to none -- widening someone's stated edit unasked is a
    # surprise. `strip_silence` finds its own boundaries from an energy
    # threshold, whose bias is systematically INSIDE the speech, so padding is
    # on by default there. See contracts/plan.v1.md#edit-point-resolution.
    _add_edit_point_arguments(cut_parser, pad_default=0.0)

    strip_silence_parser = sub.add_parser("strip-silence")
    strip_silence_parser.add_argument("--threshold", type=float, default=6.0)
    strip_silence_parser.add_argument("--min-len", dest="min_len", type=float, default=400.0)
    strip_silence_parser.add_argument("--keep", type=float, default=150.0)
    _add_edit_point_arguments(strip_silence_parser, pad_default=80.0)

    # advise / master: model-backed, not yet built (see _NOT_YET_BUILT below), but
    # documented with the exact positional surface their own --help worked
    # examples show ('aud advise in.wav', 'aud master in.wav out.wav') -- a bare
    # registration would make a caller discover the flag doesn't parse instead of
    # getting `not_implemented`.
    advise_parser = sub.add_parser("advise")
    advise_parser.add_argument("path")

    master_parser = sub.add_parser("master")
    master_parser.add_argument("in_path")
    master_parser.add_argument("out_path")

    # preset: documented two ways -- 'aud preset --list' (verbdoc.py) and
    # 'aud preset show <name>' (docs/ARCHITECTURE.md, README.md, SMART_TOOL.md).
    preset_parser = sub.add_parser("preset")
    preset_parser.add_argument("--list", action="store_true")
    preset_sub = preset_parser.add_subparsers(dest="preset_action", required=False)
    preset_show_parser = preset_sub.add_parser("show")
    preset_show_parser.add_argument("name")

    for name in sorted(_NOT_YET_BUILT):
        if name not in sub.choices:  # the ones above registered their real arguments
            sub.add_parser(name)

    return parser


def registered_verbs() -> frozenset[str]:
    """The set of verbs the argparse CLI actually recognises.

    Single source of truth for what `_build_parser` registered, used to keep
    the CLI surface and `aud.core.skill.CAPABILITIES` in lockstep (see
    tests/test_cli_envelope.py). A capability declared in one but not the
    other is exactly the class of drift that let 'analyze' go unreachable.
    """
    parser = _build_parser()
    for action in parser._actions:  # the standard way to introspect subparsers
        if isinstance(action, argparse._SubParsersAction):
            return frozenset(action.choices)
    return frozenset()


def _dispatch_stage(verb: str, plan: Any, args: argparse.Namespace) -> Any:
    if verb == "strip-silence":
        return lib.strip_silence(
            plan,
            threshold_above_floor_db=args.threshold,
            min_len_ms=args.min_len,
            keep_ms=args.keep,
            pad_out_ms=args.pad_out,
            pad_in_ms=args.pad_in,
            snap=args.snap,
            snap_window_ms=args.snap_window,
            fade_out_ms=args.fade_out,
            fade_in_ms=args.fade_in,
            crossfade_ms=args.crossfade,
            crossfade_shape=args.crossfade_shape,
        )
    if verb == "deess":
        return lib.deess(plan, amount=args.amount, freq=args.freq)
    if verb == "dereverb":
        return lib.dereverb(plan, amount=args.amount)
    if verb == "eq":
        return lib.eq(plan, hpf=args.hpf, lpf=args.lpf, peaks=args.peaks or [])
    if verb == "eq-match":
        curve = lib.load_json_file(args.curve)
        return lib.eq_match(plan, curve=curve, mix=args.mix)
    if verb == "compress":
        return lib.compress(plan, bands=args.bands, ratio=args.ratio)
    if verb == "saturate":
        return lib.saturate(plan, drive=args.drive, mix=args.mix)
    if verb == "reverb":
        return lib.reverb(plan, amount=args.amount, decay=args.decay)
    if verb == "stretch":
        return lib.stretch(plan, factor=args.factor)
    if verb == "pitch":
        return lib.pitch(plan, semitones=args.semitones)
    if verb == "loudness":
        return lib.loudness(plan, target_lufs=args.target)
    if verb == "limit":
        return lib.limit(plan, ceiling_dbtp=args.ceiling)
    raise AssertionError(f"unreachable stage verb: {verb}")


def _dispatch(verb: str, args: argparse.Namespace, stdin_text: str | None) -> Any:
    if verb == "manifest":
        return lib.manifest()
    if verb == "check":
        return lib.check()
    if verb == "analyze":
        return lib.analyze(args.path)
    if verb == "config":
        return lib.config(
            sample_rate_policy=args.sample_rate_policy,
            default_ceiling_dbtp=args.default_ceiling_dbtp,
            default_target_lufs=args.default_target_lufs,
            oversample=args.oversample,
            output_subtype=args.output_subtype,
        )
    if verb == "plan":
        if args.from_file:
            return read_plan(lib.read_text_file(args.from_file))
        return read_plan(None)
    if verb == "cut":
        # `cut`'s regions come from --regions (a file path) when given; the
        # 'aud detect silence in.wav | aud cut | aud render' pipeline instead
        # hands them on stdin, in which case stdin is not available for an
        # upstream plan and `cut` starts a fresh one -- it is first in
        # canonical order in that usage anyway (contracts/plan.v1.md).
        if args.regions:
            regions_text = lib.read_text_file(args.regions)
            plan = read_plan(stdin_text)
        else:
            regions_text = stdin_text
            plan = read_plan(None)
        return lib.cut(
            plan,
            regions_text,
            pad_out_ms=args.pad_out,
            pad_in_ms=args.pad_in,
            snap=args.snap,
            snap_window_ms=args.snap_window,
            fade_out_ms=args.fade_out,
            fade_in_ms=args.fade_in,
            crossfade_ms=args.crossfade,
            crossfade_shape=args.crossfade_shape,
        )
    if verb in STAGE_VERB_NAMES:
        plan = read_plan(stdin_text)
        return _dispatch_stage(verb, plan, args)
    if verb == "curve":
        if args.curve_action == "extract":
            return lib.curve_extract(args.path, args.out)
        return lib.curve_apply(args.path, args.curve, args.out)
    if verb == "render":
        plan = read_plan(stdin_text)
        return lib.render(plan, args.in_path, args.out_path)
    if verb == "verify":
        return lib.verify(args.path, target_lufs=args.target, ceiling_dbtp=args.ceiling)
    if verb == "detect":
        if args.detect_kind == "transients":
            return lib.detect_transients(args.path, sensitivity=args.sensitivity, min_gap_ms=args.min_gap)
        if args.detect_kind == "silence":
            return lib.detect_silence(args.path, threshold_above_floor_db=args.threshold, min_len_ms=args.min_len)
        if args.detect_kind == "fillers":
            words = [w.strip() for w in args.words.split(",") if w.strip()] or None
            return lib.detect_fillers(args.path, words=words, min_pause_ms=args.min_pause)
        raise AssertionError(f"unreachable detect kind: {args.detect_kind}")
    if verb in _NOT_YET_BUILT:
        raise AudError(
            code="not_implemented",
            message=f"'{verb}' is not built yet in this release of aud.",
            remedy="Use the deterministic verbs directly (see 'aud --help'); this verb lands in a later release.",
        )
    raise AudError(
        code="usage_error", message=f"Unknown verb '{verb}'.", remedy="Run 'aud --help' for the list of verbs."
    )


def _print_result(result: Any) -> None:
    print(json.dumps({"result": result}, sort_keys=True, default=str))


def _print_error(exc: AudError) -> None:
    print(json.dumps({"error": exc.to_dict()}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # --debug (or AUD_DEBUG=1) is a global switch, not tied to any one verb's
    # subparser, so it is stripped out before argparse ever sees argv -- the
    # same treatment --help already gets below.
    debug = bool(os.environ.get("AUD_DEBUG")) or "--debug" in argv
    argv = [arg for arg in argv if arg != "--debug"]

    if argv[:1] == ["--help"]:
        print(lib.skill())
        return 0
    if argv[:1] and argv[0] in VERB_DOCS and "--help" in argv[1:]:
        print(VERB_DOCS[argv[0]])
        return 0

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        _print_error(
            AudError(
                code="usage_error",
                message=exc.message,
                remedy="Run 'aud --help' for the list of verbs, or 'aud <verb> --help' for one verb's full documentation.",
            )
        )
        return 2
    except SystemExit as exc:
        # Reached only via argparse's own -h/--help handling (prints usage, exit 0).
        return exc.code if isinstance(exc.code, int) else 0

    verb: str = args.verb
    stdin_text: str | None = None
    if verb in PLAN_CONSUMING_VERBS and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()

    try:
        outcome = _dispatch(verb, args, stdin_text)
    except AudError as exc:
        _print_error(exc)
        return 1
    except Exception as exc:
        # Last-resort guard: the contract promises exactly one JSON document
        # on stdout, never a raw traceback. Any exception the library did
        # not already turn into an AudError is an internal bug, not a usage
        # error -- report it honestly, and keep the real traceback available
        # on stderr behind --debug/AUD_DEBUG so it isn't lost.
        if debug:
            traceback.print_exc(file=sys.stderr)
        _print_error(
            AudError(
                code="internal_error",
                message=f"Unexpected {type(exc).__name__}: {exc}",
                remedy=(
                    "This is an internal aud bug, not something wrong with your input. "
                    "Re-run with --debug (or AUD_DEBUG=1) to see the full traceback on "
                    "stderr, and report it with that output."
                ),
            )
        )
        return 1

    if verb in PLAN_OUTPUT_VERBS:
        print(write_plan(outcome))
    elif verb in REGIONS_OUTPUT_VERBS:
        print(write_regions(outcome))
    else:
        _print_result(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
