"""The provider-agnostic seam: one Protocol, one HTTPS implementation per
provider, and credential-based provider selection.

No provider SDK is a dependency of this package or of `aud` itself -- every
backend below is plain HTTPS via the standard library (`urllib`), so the
base install (numpy/scipy/soundfile/pyloudnorm) never grows a cost for a
caller with no AI-provider credential configured. See AGENTS.md #1 and #5,
and docs/CONFIGURATION.md for the credential precedence this module reads.

Tests never call a real provider: they implement `IntelligenceBackend`
directly with a canned response (see tests/test_intelligence.py). Only
`resolve_backend` and the four backend classes below ever touch a network.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol

from aud.schemas import AudError

__all__ = [
    "DEFAULT_MODELS",
    "PROVIDER_ENV_VARS",
    "AnthropicBackend",
    "AzureOpenAIBackend",
    "GoogleBackend",
    "IntelligenceBackend",
    "OpenAIBackend",
    "resolve_backend",
]


class IntelligenceBackend(Protocol):
    """The one seam between `aud` and any model provider.

    Exactly one method: hand it a system prompt and a user prompt, get text
    back. No SDK type crosses this boundary in either direction. A test
    implements this Protocol with a canned string and passes it as `advise`'s
    `backend=` parameter -- no network, no credential, no cost.
    """

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        """Return the model's raw text response to `user`, guided by `system`."""
        ...


_HTTP_TIMEOUT_S = 60.0


def _post_json(url: str, headers: dict[str, str], body: dict, *, provider: str) -> dict:
    """POST `body` as JSON to `url`, mapping transport failures to AudError.

    A provider's HTTP/network failure is not an internal aud bug -- it is
    the same class of "external thing failed" as a missing ffmpeg binary --
    so it is reported with a code and remedy a caller can act on, never a
    raw traceback.
    """
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        # A 404, or any body naming the model, almost always means the MODEL is
        # wrong rather than the credential -- and the likeliest wrong model is
        # our own default, which is perishable and cannot be covered by a test
        # that injects a fake backend. Saying "check your credential" there
        # sends the caller to debug the one thing that is working.
        if exc.code == 404 or "model" in detail.lower():
            raise AudError(
                code="provider_model_unavailable",
                message=f"{provider} rejected the model: HTTP {exc.code}: {detail}",
                remedy=f"Pass --model with a model this account can reach (or set {_MODEL_ENV_VAR}). "
                f"The built-in default for {provider} may have been retired -- see docs/CONFIGURATION.md.",
            ) from exc
        raise AudError(
            code="provider_request_failed",
            message=f"{provider} request failed: HTTP {exc.code}: {detail}",
            remedy="Check the credential is valid and has quota; retry, or build the chain by hand with the "
            "deterministic verbs.",
        ) from exc
    except urllib.error.URLError as exc:
        raise AudError(
            code="provider_request_failed",
            message=f"{provider} request failed: {exc.reason}",
            remedy="Check network connectivity to the provider; retry, or build the chain by hand with the "
            "deterministic verbs.",
        ) from exc
    return json.loads(raw)


def _bad_shape(provider: str, data: object) -> AudError:
    return AudError(
        code="provider_request_failed",
        message=f"{provider} response did not have the expected shape: {data!r}",
        remedy="Retry, or build the chain by hand with the deterministic verbs.",
    )


class AnthropicBackend:
    """https://docs.anthropic.com/en/api/messages"""

    provider = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = _post_json("https://api.anthropic.com/v1/messages", headers, body, provider=self.provider)
        # `content` is a LIST OF BLOCKS and the text is not always first. A
        # reasoning-capable model returns a `thinking` block ahead of it, so
        # `content[0]["text"]` raises KeyError and every such model looks like
        # a malformed response. Found by calling claude-sonnet-5 for real --
        # no fake backend could have surfaced it. Take the first text block.
        try:
            blocks = data["content"]
            for block in blocks:
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    return block["text"]
        except (KeyError, TypeError) as exc:
            raise _bad_shape(self.provider, data) from exc
        raise _bad_shape(self.provider, data)


class OpenAIBackend:
    """https://platform.openai.com/docs/api-reference/chat"""

    provider = "openai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        data = _post_json("https://api.openai.com/v1/chat/completions", headers, body, provider=self.provider)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _bad_shape(self.provider, data) from exc


class GoogleBackend:
    """https://ai.google.dev/api/generate-content -- backs both GOOGLE_API_KEY and GEMINI_API_KEY."""

    provider = "google"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._api_key}"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        data = _post_json(url, {"content-type": "application/json"}, body, provider=self.provider)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _bad_shape(self.provider, data) from exc


class AzureOpenAIBackend:
    """https://learn.microsoft.com/azure/ai-services/openai/reference

    Azure OpenAI needs more than the one credential the manifest names to
    reach a deployment: a resource endpoint, and a deployment name (which
    Azure treats as the "model"). Both are read from the environment at
    construction time, never invented -- `AZURE_OPENAI_ENDPOINT` is
    required; `AZURE_OPENAI_DEPLOYMENT` falls back to whatever model name
    was resolved (see `resolve_backend`), and `AZURE_OPENAI_API_VERSION`
    falls back to a documented default.
    """

    provider = "azure_openai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise AudError(
                code="provider_config_incomplete",
                message="AZURE_OPENAI_API_KEY is set but AZURE_OPENAI_ENDPOINT is not.",
                remedy="Set AZURE_OPENAI_ENDPOINT to your resource's endpoint URL "
                "(e.g. https://<resource>.openai.azure.com) -- see docs/CONFIGURATION.md.",
            )
        self._endpoint = endpoint.rstrip("/")
        self._api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

    def complete(self, system: str, user: str, *, model: str, max_tokens: int = 2000) -> str:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)
        url = f"{self._endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self._api_version}"
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        headers = {"api-key": self._api_key, "content-type": "application/json"}
        data = _post_json(url, headers, body, provider=self.provider)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _bad_shape(self.provider, data) from exc


# Order is the documented precedence (SMART_TOOL.md, docs/CONFIGURATION.md):
# the first one present in the environment wins. GOOGLE_API_KEY and
# GEMINI_API_KEY both satisfy the same provider -- either name works.
PROVIDER_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AZURE_OPENAI_API_KEY",
)

_PROVIDER_BY_ENV: dict[str, str] = {
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
    "GOOGLE_API_KEY": "google",
    "GEMINI_API_KEY": "google",
    "AZURE_OPENAI_API_KEY": "azure_openai",
}

# One named, documented, overridable default per provider -- never a bare
# string buried in call logic (AGENTS.md #5). Overridable by --model (CLI)
# or AUD_MODEL (environment); see docs/CONFIGURATION.md.
#
# A DEFAULT MODEL NAME IS PERISHABLE and these will go stale. The first live
# call this tool ever made returned HTTP 404 "model: claude-3-5-haiku-20241022"
# against a perfectly valid key -- the default had been chosen by reading, not
# by calling, and nothing in the test suite could see it because every test
# injects a fake backend through the Protocol. That is why a 404 or an unknown
# model is reported as `provider_model_unavailable` with the remedy naming
# `--model`, rather than as a generic request failure: the likeliest cause of
# this specific error is our default, not the caller's credential.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "azure_openai": "gpt-4o-mini",
}

_MODEL_ENV_VAR = "AUD_MODEL"

_BACKEND_FACTORIES: dict[str, Callable[[str], IntelligenceBackend]] = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "google": GoogleBackend,
    "azure_openai": AzureOpenAIBackend,
}


def resolve_backend(model: str | None = None) -> tuple[IntelligenceBackend, str, str]:
    """Pick a backend from whichever provider credential is configured.

    Provider precedence is the order of `PROVIDER_ENV_VARS`: the first one
    present in the environment wins. Model precedence: the explicit `model`
    argument (the CLI's `--model`) > the `AUD_MODEL` environment variable >
    `DEFAULT_MODELS[provider]`.

    Returns:
        (backend, provider_name, model_name).

    Raises:
        AudError: code "provider_credential_missing" if none of
            `PROVIDER_ENV_VARS` is set in the environment.
    """
    for env_var in PROVIDER_ENV_VARS:
        api_key = os.environ.get(env_var)
        if not api_key:
            continue
        provider = _PROVIDER_BY_ENV[env_var]
        backend = _BACKEND_FACTORIES[provider](api_key)
        chosen_model = model or os.environ.get(_MODEL_ENV_VAR) or DEFAULT_MODELS[provider]
        return backend, provider, chosen_model

    raise AudError(
        code="provider_credential_missing",
        message=f"advise/master need a model provider credential; none of {', '.join(PROVIDER_ENV_VARS)} is set.",
        remedy=(
            "Set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY or "
            "AZURE_OPENAI_API_KEY, or use the deterministic verbs directly -- see docs/CONFIGURATION.md."
        ),
    )
