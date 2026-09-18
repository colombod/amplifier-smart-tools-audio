"""Shared pytest fixtures for the aud test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).parent

# The five AI-provider credentials the SMART_TOOL.md manifest says are
# optional. Deterministic verbs must work with all five absent.
PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def scrubbed_env(tmp_path: Path) -> dict[str, str]:
    """An environment with every AI-provider credential removed and PATH
    pointed at a freshly created, empty directory.

    PATH is set to an *empty* directory rather than unset, so a stray
    system binary (e.g. a host-installed ffmpeg) cannot make a "no
    provider" test pass for the wrong reason.
    """
    empty_path_dir = tmp_path / "empty-path"
    empty_path_dir.mkdir()
    env = dict(os.environ)
    for name in PROVIDER_ENV_VARS:
        env.pop(name, None)
    env["PATH"] = str(empty_path_dir)
    return env


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    """A short, real (not silent) stereo WAV fixture for DSP-touching tests.

    Two tones plus a bit of noise and a hot mid-section, so loudness, true
    peak and per-band measurements are all non-degenerate -- silence would
    let a broken measurement pass by coincidence.
    """
    sr = 44100
    seconds = 2.0
    t = np.arange(int(sr * seconds)) / sr
    x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 3000 * t)
    rng = np.random.default_rng(0)
    x = x + 0.02 * rng.standard_normal(t.size)
    hot = slice(sr // 4, sr // 4 + sr // 8)
    x[hot] *= 3.0
    path = tmp_path / "in.wav"
    sf.write(str(path), np.stack([x, x], axis=1), sr, subtype="PCM_24")
    return path
