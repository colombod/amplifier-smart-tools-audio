"""Tests for `aud.lib.config` -- settings resolution and the wrong-type-is-fatal rule.

docs/CONFIGURATION.md: "Absent, null, and wrong-type are three different
conditions... A wrong type is never coerced and never ignored." These tests
guard the two numeric settings (`oversample`: int, `default_ceiling_dbtp`/
`default_target_lufs`: float) whose environment-variable coercion used to be
a bare `int()`/`float()` call -- a bad value raised an uncaught `ValueError`
that surfaced as `internal_error` (a bug report request) rather than naming
the setting, the value found, and the type expected.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aud import lib
from aud.schemas import AudError


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aud.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


# --- lib-level: the exact ValueError-swallowing bug this guards against ----


def test_bad_int_env_value_is_a_named_bad_config_not_a_raw_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUD_OVERSAMPLE", "four")
    with pytest.raises(AudError) as excinfo:
        lib.config()
    assert excinfo.value.code == "bad_config"
    assert "AUD_OVERSAMPLE" in excinfo.value.message
    assert "four" in excinfo.value.message
    assert "oversample" in excinfo.value.message
    assert excinfo.value.remedy


def test_bad_float_env_value_is_a_named_bad_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUD_DEFAULT_CEILING_DBTP", "-1.0 dB")
    with pytest.raises(AudError) as excinfo:
        lib.config()
    assert excinfo.value.code == "bad_config"
    assert "AUD_DEFAULT_CEILING_DBTP" in excinfo.value.message
    assert "default_ceiling_dbtp" in excinfo.value.message


def test_valid_numeric_env_values_still_coerce_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUD_OVERSAMPLE", "8")
    monkeypatch.setenv("AUD_DEFAULT_TARGET_LUFS", "-18.5")
    result = lib.config()
    assert result["oversample"] == {"value": 8, "source": "environment"}
    assert result["default_target_lufs"] == {"value": -18.5, "source": "environment"}


def test_argument_tier_still_wins_over_a_bad_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad env value must not block an explicit override from resolving --
    only reaching the environment TIER for that key raises."""
    monkeypatch.setenv("AUD_OVERSAMPLE", "not-a-number")
    result = lib.config(oversample=2)
    assert result["oversample"] == {"value": 2, "source": "argument"}


# --- config-FILE tier: a TOML value already carries its own type ----------
#
# docs/CONFIGURATION.md's worked examples are config-FILE values
# (`oversample = "four"`, `output_subtype = 24`) -- distinct from the
# environment tier above, because a TOML value is never a string that needs
# parsing; it is either already the right Python type or it plainly is not.


def test_bad_type_in_config_file_is_a_named_bad_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('oversample = "four"\n', encoding="utf-8")
    monkeypatch.setattr(lib, "_CONFIG_FILE_PATH", config_file)
    with pytest.raises(AudError) as excinfo:
        lib.config()
    assert excinfo.value.code == "bad_config"
    assert "oversample" in excinfo.value.message
    assert str(config_file) in excinfo.value.message
    assert "four" in excinfo.value.message
    assert excinfo.value.remedy


def test_wrong_type_in_config_file_int_instead_of_string_is_also_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverted case from docs/CONFIGURATION.md's own example:
    `output_subtype = 24` -- a TOML integer where a string is declared."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("output_subtype = 24\n", encoding="utf-8")
    monkeypatch.setattr(lib, "_CONFIG_FILE_PATH", config_file)
    with pytest.raises(AudError) as excinfo:
        lib.config()
    assert excinfo.value.code == "bad_config"
    assert "output_subtype" in excinfo.value.message


def test_correctly_typed_config_file_value_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("oversample = 8\n", encoding="utf-8")
    monkeypatch.setattr(lib, "_CONFIG_FILE_PATH", config_file)
    result = lib.config()
    assert result["oversample"] == {"value": 8, "source": "config_file"}


# --- CLI-level: real subprocess, the error travels through main() cleanly --


def test_cli_config_with_bad_env_value_is_a_clean_bad_config_error(scrubbed_env: dict[str, str]) -> None:
    env = dict(scrubbed_env)
    env["AUD_OVERSAMPLE"] = "four"
    proc = _run(["config"], env=env)
    assert proc.returncode != 0
    assert proc.stdout == ""
    payload = json.loads(proc.stderr)
    assert payload["error"]["code"] == "bad_config"
    assert "AUD_OVERSAMPLE" in payload["error"]["message"]
    assert "Traceback" not in proc.stderr
