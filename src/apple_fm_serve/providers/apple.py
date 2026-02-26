import json
from collections.abc import AsyncIterator, Mapping
from typing import Protocol

from apple_fm_serve.errors import AdapterError
from apple_fm_serve.providers.base import GenerationInput

_APPLE_FM_SDK_IMPORT_ERROR: str | None
try:  # pragma: no cover - environment dependent
    import apple_fm_sdk as _apple_fm_sdk
except ImportError as import_error:  # pragma: no cover - environment dependent
    _APPLE_FM_SDK_IMPORT_ERROR = str(import_error)
    _apple_fm_sdk = None
else:  # pragma: no cover - environment dependent
    _APPLE_FM_SDK_IMPORT_ERROR = None


class _SystemLanguageModelProtocol(Protocol):
    def is_available(self) -> tuple[bool, str | None]: ...


class _LanguageModelSessionProtocol(Protocol):
    async def respond(
        self, prompt: str, *, json_schema: Mapping[str, object] | None = None
    ) -> object: ...
    def stream_response(self, prompt: str) -> AsyncIterator[str]: ...


class _FoundationModelsModuleProtocol(Protocol):
    def SystemLanguageModel(self) -> _SystemLanguageModelProtocol: ...
    def LanguageModelSession(
        self, instructions: str | None = None
    ) -> _LanguageModelSessionProtocol: ...


class AppleFoundationModelProvider:
    def __init__(self, *, model_id: str) -> None:
        self._model_id = model_id
        self._fm_module: _FoundationModelsModuleProtocol | None = _apple_fm_sdk
        self._import_error = _APPLE_FM_SDK_IMPORT_ERROR

    async def is_available(self) -> tuple[bool, str | None]:
        fm_module = self._fm_module
        if fm_module is None:
            return False, self._import_error or "apple_fm_sdk not importable"

        model = fm_module.SystemLanguageModel()
        available, reason = model.is_available()
        return bool(available), reason

    async def list_models(self) -> list[str]:
        return [self._model_id]

    async def generate(self, request: GenerationInput) -> str:
        fm_module = self._fm_module
        if fm_module is None:
            raise AdapterError(
                self._import_error or "apple_fm_sdk import failed",
                status_code=500,
                error_type="server_error",
                code="provider_import_error",
            )

        session = fm_module.LanguageModelSession(instructions=request.instructions)
        try:
            if request.response_schema is not None:
                response = await session.respond(
                    request.prompt, json_schema=request.response_schema
                )
                if hasattr(response, "to_json"):
                    return str(response.to_json())
                try:
                    return json.dumps(response)
                except TypeError:
                    return str(response)
            text_response = await session.respond(request.prompt)
            return str(text_response)
        except Exception as exc:
            raise self._map_provider_exception(exc) from exc

    async def stream_generate(self, request: GenerationInput) -> AsyncIterator[str]:
        fm_module = self._fm_module
        if fm_module is None:
            raise AdapterError(
                self._import_error or "apple_fm_sdk import failed",
                status_code=500,
                error_type="server_error",
                code="provider_import_error",
            )
        if request.response_schema is not None:
            raise AdapterError(
                "Streaming with response_format is not supported",
                status_code=400,
                param="response_format",
                code="response_format_stream_unsupported",
            )

        session = fm_module.LanguageModelSession(instructions=request.instructions)
        try:
            async for snapshot in session.stream_response(request.prompt):
                yield str(snapshot)
        except Exception as exc:
            raise self._map_provider_exception(exc) from exc

    @staticmethod
    def _map_provider_exception(exc: Exception) -> AdapterError:
        name = exc.__class__.__name__

        if name == "ExceededContextWindowSizeError":
            return AdapterError(
                "Context window exceeded",
                status_code=400,
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )
        if name == "AssetsUnavailableError":
            return AdapterError(
                "Model assets unavailable",
                status_code=503,
                error_type="server_error",
                code="assets_unavailable",
            )
        if name == "RateLimitedError":
            return AdapterError(
                "Rate limited by provider",
                status_code=429,
                error_type="rate_limit_error",
                code="rate_limited",
            )
        if name in {"GuardrailViolationError", "RefusalError"}:
            return AdapterError(
                str(exc),
                status_code=400,
                error_type="invalid_request_error",
                code="guardrail_violation",
            )

        return AdapterError(
            f"Provider error: {exc}",
            status_code=500,
            error_type="server_error",
            code="provider_error",
        )
