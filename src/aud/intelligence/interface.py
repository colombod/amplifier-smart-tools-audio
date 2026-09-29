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
    "ResponseSchema",
    "resolve_backend",
]


class ResponseSchema:
    """A named JSON Schema a caller wants the model's response constrained to.

    Passed to `IntelligenceBackend.complete` as the optional `response_schema`
    kwarg, this asks the backend to use whatever schema-constrained decoding
    its provider supports, so trailing content after the JSON value becomes
    STRUCTURALLY IMPOSSIBLE rather than something the caller must detect and
    repair after the fact -- see issue #35 (`aud advise` returning a valid
    plan followed by trailing prose the model kept writing after its own
    closing code fence).

    Each backend uses whatever subset of this it can act on; `complete`'s
    return type is unchanged (still `str`) either way:

    - `AnthropicBackend` forces a `tool_choice`-pinned tool call named
      `name` with `input_schema = schema`, and sets
      `disable_parallel_tool_use: true` so the API returns at most one such
      call (checked against Anthropic's current tool-use docs, 2026-09-28:
      https://docs.claude.com/en/docs/agents-and-tools/tool-use/parallel-tool-use
      -- the field lives inside `tool_choice`, not top-level). Because the
      call is forced and pinned to exactly one tool, there is no free-text
      channel left for prose to trail at all -- that structural guarantee is
      what issue #35 needed, and it holds independently of `strict`.

      **`strict: true` is deliberately NOT set** (PR #41 review round 2):
      Anthropic's strict tool use requires `additionalProperties: false` on
      EVERY object in the schema -- "there is no supported value other than
      false" (https://docs.claude.com/en/docs/agents-and-tools/tool-use/strict-tool-use,
      checked 2026-09-28) -- including `params`. `params` is deliberately
      left as a generic, propertyless object (see
      `advisor._ADVISE_RESPONSE_SCHEMA`'s comment: a full 10-way
      discriminated union over `advisor.ALLOWED_STAGES` would duplicate
      `_validate_and_build_plan`'s own checks and was explicitly deferred,
      unverified, in the original PR). Forcing `additionalProperties: false`
      onto `params` with no declared properties would make `params: {}` the
      ONLY value the API accepts for every stage -- silently making it
      impossible for the model to supply ANY stage's parameters. That is a
      real weakening of what a plan can express, not a metadata change, so
      it is refused here rather than shipped unverified. `input_schema` is
      therefore GUIDANCE the model sees, not a shape the API enforces; the
      returned string is still a fresh serialisation of the tool call's
      already-parsed `input` (never raw free text with room for prose to
      trail), and all shape/range enforcement remains
      `_validate_and_build_plan`'s job downstream, unchanged. Revisit if a
      later pass builds and live-verifies a real per-stage discriminated
      union.
    - `OpenAIBackend`/`AzureOpenAIBackend` set `response_format: {"type":
      "json_object"}` -- guarantees the ENTIRE response is one valid JSON
      value (no trailing content), but does not itself enforce `schema`'s
      shape. See this module's docstring for why full per-field
      `json_schema` strict mode is not implemented here.
    - `GoogleBackend` sets `generationConfig.responseMimeType =
      "application/json"` for the same "one valid JSON value" guarantee;
      also does not enforce `schema`'s shape.

    A backend given no `response_schema` (the default, `None`) behaves
    exactly as before this existed: free text, parsed and bounded-repaired
    by the caller (see `aud.intelligence.advisor._parse_model_json`).
    """

    def __init__(self, name: str, schema: dict, description: str = "") -> None:
        self.name = name
        self.schema = schema
        self.description = description or f"Return {name}."


class IntelligenceBackend(Protocol):
    """The one seam between `aud` and any model provider.

    Exactly one method: hand it a system prompt and a user prompt, get text
    back. No SDK type crosses this boundary in either direction. A test
    implements this Protocol with a canned string and passes it as `advise`'s
    `backend=` parameter -- no network, no credential, no cost.

    `response_schema` (added for issue #35) is optional and additive: an
    implementer that ignores it keeps working exactly as before -- every
    caller passes it as a keyword with a default of `None`, never
    positionally, and no existing behaviour changes when it is omitted.
    """

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 2000,
        response_schema: ResponseSchema | None = None,
    ) -> str:
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

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 2000,
        response_schema: ResponseSchema | None = None,
    ) -> str:
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
        if response_schema is not None:
            # A forced tool call: the API itself parses the model's answer
            # into `input_schema`-shaped JSON server-side (Anthropic's own
            # constrained decoding) and hands it back already-parsed as the
            # tool_use block's `input`. There is no free-text channel left
            # for trailing prose to occupy -- see issue #35 and
            # ResponseSchema's docstring above. `tool_choice` pins the
            # model to exactly this one tool, so it cannot choose to
            # answer in plain text instead. `disable_parallel_tool_use`
            # additionally asks the API for AT MOST ONE tool_use block
            # (PR #41 review round 2, Fix #1) -- belt-and-suspenders with
            # the refusal below, which still catches more than one block
            # if a future API version or tool_choice mode ever returns
            # several regardless.
            body["tools"] = [
                {
                    "name": response_schema.name,
                    "description": response_schema.description,
                    "input_schema": response_schema.schema,
                }
            ]
            body["tool_choice"] = {
                "type": "tool",
                "name": response_schema.name,
                "disable_parallel_tool_use": True,
            }
        data = _post_json("https://api.anthropic.com/v1/messages", headers, body, provider=self.provider)
        if response_schema is not None:
            try:
                blocks = data["content"]
                matches = [
                    block
                    for block in blocks
                    if block.get("type") == "tool_use" and block.get("name") == response_schema.name
                ]
            except (KeyError, TypeError) as exc:
                raise _bad_shape(self.provider, data) from exc
            if not matches:
                raise _bad_shape(self.provider, data)
            if len(matches) > 1:
                # More than one matching tool_use block is the same
                # ambiguity `advisor._parse_model_json` refuses for two
                # distinct JSON values -- aud cannot pick between them
                # (PR #41 review round 2, Fix #1).
                raise AudError(
                    code="bad_model_output",
                    message=(
                        f"{self.provider} returned {len(matches)} '{response_schema.name}' tool_use blocks in "
                        "one response -- aud cannot pick between multiple proposed plans."
                    ),
                    remedy="Retry advise/master -- the model proposed more than one plan and aud cannot pick "
                    "between them.",
                )
            try:
                return json.dumps(matches[0]["input"])
            except (KeyError, TypeError) as exc:
                raise _bad_shape(self.provider, data) from exc
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

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 2000,
        response_schema: ResponseSchema | None = None,
    ) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            # JSON mode (GA on Chat Completions since 2023): guarantees the
            # ENTIRE `content` string is one valid JSON value -- no trailing
            # prose, which is exactly issue #35's defect. It does NOT
            # enforce `response_schema.schema`'s shape (that is still
            # `advisor._validate_and_build_plan`'s job, unchanged).
            #
            # Full per-field Structured Outputs (`response_format:
            # {"type": "json_schema", "strict": true, ...}`, GA since
            # 2024-08-06) is NOT implemented here: strict mode requires
            # every object in the schema -- including `params`, whose shape
            # is a 10-way union across the stage types in
            # `advisor.ALLOWED_STAGES` -- to set `additionalProperties:
            # false` and list every field as `required` (optional fields
            # only via `anyOf` null-unions). That is a real, buildable
            # schema, but building it correctly needs a live OpenAI test
            # this pass did not budget for (see AGENTS.md SS3b -- an
            # unverified schema is exactly the kind of untested assumption
            # this repo does not ship). json_object mode already removes
            # the specific defect this issue reports; checked against
            # OpenAI's chat/completions API docs, 2026-09-28.
            body["response_format"] = {"type": "json_object"}
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

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 2000,
        response_schema: ResponseSchema | None = None,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._api_key}"
        generation_config: dict[str, object] = {"maxOutputTokens": max_tokens}
        if response_schema is not None:
            # Guarantees the ENTIRE response is one valid JSON value -- no
            # trailing content, which is exactly issue #35's defect (see
            # ResponseSchema's docstring). Does NOT itself enforce
            # `response_schema.schema`'s shape (`responseSchema` would, but
            # is not set here for the same reason noted in OpenAIBackend:
            # a full per-stage-params union schema is buildable but was not
            # live-verified this pass). Checked against
            # https://ai.google.dev/api/generate-content, 2026-09-28.
            generation_config["responseMimeType"] = "application/json"
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        data = _post_json(url, {"content-type": "application/json"}, body, provider=self.provider)
        # `parts` is a LIST, and with "thinking" enabled a part carries
        # `"thought": true` and precedes the actual answer part -- the SAME
        # hazard class as AnthropicBackend's `content[0]["text"]` bug
        # above, confirmed via Gemini's own docs (thought summaries are
        # returned as parts with `thought: true`, e.g.
        # https://ai.google.dev/gemini-api/docs/generate-content/thinking)
        # rather than assumed. `gemini-3.5-flash-lite` (this provider's
        # documented default -- see DEFAULT_MODELS) is minimal-thinking by
        # design, but a caller passing `--model`/`AUD_MODEL` can select a
        # model that thinks by default, so this is guarded unconditionally
        # rather than only for models known to think. Skip every thought
        # part; JOIN every remaining (non-thought) text part IN ORDER
        # (PR #41 review round 2, Fix #3) -- Google can split one JSON
        # response across multiple parts (e.g. '{"stages": [' as one part,
        # the rest as a second), and taking only the first non-thought part
        # returned a truncated fragment that failed to parse. If the join
        # happens to concatenate two complete plans, that is exactly the
        # ambiguity `advisor._parse_model_json` already refuses -- this
        # backend's job is only to reassemble the text faithfully, never to
        # judge its content.
        try:
            parts = data["candidates"][0]["content"]["parts"]
            texts = [part["text"] for part in parts if not part.get("thought") and isinstance(part.get("text"), str)]
        except (KeyError, IndexError, TypeError) as exc:
            raise _bad_shape(self.provider, data) from exc
        if texts:
            return "".join(texts)
        # Every part was a thought (or there were no parts at all) -- the
        # SAME named error as before this fix, never the thought text.
        raise _bad_shape(self.provider, data)


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

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 2000,
        response_schema: ResponseSchema | None = None,
    ) -> str:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)
        url = f"{self._endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self._api_version}"
        body: dict[str, object] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            # Same JSON-mode guarantee as OpenAIBackend (this deployment
            # shares the Chat Completions surface): the ENTIRE `content`
            # string becomes one valid JSON value, eliminating issue #35's
            # trailing-content defect, though not enforcing
            # `response_schema.schema`'s shape.
            #
            # Full Structured Outputs (`json_schema`, strict) needs API
            # version 2024-08-01-preview or later; this backend's default
            # (`self._api_version`, from AZURE_OPENAI_API_VERSION or the
            # "2024-06-01" fallback set in __init__) predates that and
            # returns HTTP 400 ("response_format value as json_schema is
            # enabled only for api versions 2024-08-01-preview and later")
            # if requested. json_object mode is supported since
            # 2023-12-01-preview, well before either version, so it is
            # safe against the default. Checked against
            # https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs,
            # 2026-09-28 -- not verified against a live Azure deployment
            # (no AZURE_OPENAI_* credential available in this pass).
            body["response_format"] = {"type": "json_object"}
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
    # Current stable lightweight text model; minimal thinking by default
    # fits this advisor's existing 2,000-token response cap. See CONFIGURATION.md.
    "google": "gemini-3.5-flash-lite",
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
