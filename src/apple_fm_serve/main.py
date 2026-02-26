from __future__ import annotations

import asyncio
import hmac
import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import uvicorn
from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .errors import AdapterError, AuthError, ProviderUnavailableError
from .ids import completion_id as new_completion_id
from .openai_types import (
    ChatCompletionChoice,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionsChunk,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    DeltaMessage,
    JsonObject,
    ModelObject,
    ModelsResponse,
    ResponseFormat,
    UsageObject,
)
from .prompt import build_prompt
from .providers.apple import AppleFoundationModelProvider
from .providers.base import GenerationInput, LLMProvider
from .stream import async_text_deltas_from_snapshots
from .usage import estimate_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _response_schema(response_format: ResponseFormat | None) -> JsonObject | None:
    if response_format is None or response_format.type == "text":
        return None

    if response_format.type == "json_object":
        return {"type": "object"}

    if response_format.json_schema is None:
        raise AdapterError(
            "response_format.json_schema is required for type=json_schema",
            status_code=400,
            param="response_format",
            code="invalid_response_format",
        )

    return response_format.json_schema.schema_


def _sse(payload: JsonObject | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _adapter_error_response(error: AdapterError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": asdict(error.body)})


def _usage_object(
    prompt_text: str,
    completion_text: str,
    *,
    estimate_usage_enabled: bool,
) -> UsageObject | None:
    if not estimate_usage_enabled:
        return None

    usage = estimate_usage(prompt_text, completion_text)
    return UsageObject(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


async def _ensure_provider_available(provider: LLMProvider) -> None:
    available, reason = await provider.is_available()
    if not available:
        raise ProviderUnavailableError(reason)


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    app_settings = settings or Settings()
    model_provider = provider or AppleFoundationModelProvider(model_id=app_settings.model_id)
    concurrency_limiter = asyncio.Semaphore(app_settings.max_concurrency)

    app = FastAPI(title="apple-fm-serve", version="0.1.0")

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        api_key = app_settings.api_key
        if not api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError
        token = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, api_key):
            raise AuthError

    @app.exception_handler(AdapterError)
    async def adapter_error_handler(_: Request, error: AdapterError) -> JSONResponse:
        return _adapter_error_response(error)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        await _ensure_provider_available(model_provider)
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models() -> ModelsResponse:
        await _ensure_provider_available(model_provider)
        model_ids = await model_provider.list_models()
        return ModelsResponse(data=[ModelObject(id=model_id) for model_id in model_ids])

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)], response_model=None)
    async def chat_completions(payload: ChatCompletionsRequest) -> Response:
        if payload.model != app_settings.model_id:
            raise AdapterError(
                f"Model '{payload.model}' not found",
                status_code=404,
                param="model",
                code="model_not_found",
            )

        prompt_result = build_prompt(payload.messages)
        generation_input = GenerationInput(
            instructions=prompt_result.instructions,
            prompt=prompt_result.prompt,
            response_schema=_response_schema(payload.response_format),
        )
        await _ensure_provider_available(model_provider)

        created_at_unix = int(time.time())
        completion_id = new_completion_id()

        if payload.stream:

            async def event_stream() -> AsyncIterator[str]:
                generated_text = ""
                try:
                    async with concurrency_limiter:
                        yield _sse(
                            ChatCompletionsChunk(
                                id=completion_id,
                                created=created_at_unix,
                                model=payload.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(role="assistant"),
                                        finish_reason=None,
                                    )
                                ],
                            ).model_dump(exclude_none=True)
                        )

                        async with asyncio.timeout(app_settings.request_timeout_s):
                            async for delta in async_text_deltas_from_snapshots(
                                model_provider.stream_generate(generation_input)
                            ):
                                generated_text += delta
                                yield _sse(
                                    ChatCompletionsChunk(
                                        id=completion_id,
                                        created=created_at_unix,
                                        model=payload.model,
                                        choices=[
                                            ChatCompletionChunkChoice(
                                                index=0,
                                                delta=DeltaMessage(content=delta),
                                                finish_reason=None,
                                            )
                                        ],
                                    ).model_dump(exclude_none=True)
                                )

                        usage = _usage_object(
                            prompt_result.prompt,
                            generated_text,
                            estimate_usage_enabled=app_settings.estimate_usage,
                        )
                        yield _sse(
                            ChatCompletionsChunk(
                                id=completion_id,
                                created=created_at_unix,
                                model=payload.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=DeltaMessage(),
                                        finish_reason="stop",
                                    )
                                ],
                                usage=usage,
                            ).model_dump(exclude_none=True)
                        )
                        yield _sse("[DONE]")
                except TimeoutError:
                    yield _sse({"error": {"message": "Request timed out", "type": "timeout_error"}})
                except AdapterError as exc:
                    yield _sse({"error": asdict(exc.body)})

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        async with concurrency_limiter:
            try:
                completion_text = await asyncio.wait_for(
                    model_provider.generate(generation_input),
                    timeout=app_settings.request_timeout_s,
                )
            except TimeoutError as exc:
                raise AdapterError(
                    "Request timed out",
                    status_code=504,
                    error_type="timeout_error",
                    code="request_timeout",
                ) from exc

        usage = _usage_object(
            prompt_result.prompt,
            completion_text,
            estimate_usage_enabled=app_settings.estimate_usage,
        )

        response = ChatCompletionsResponse(
            id=completion_id,
            created=created_at_unix,
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=completion_text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

        return JSONResponse(response.model_dump(exclude_none=True))

    return app


app = create_app()


def run() -> None:
    settings = Settings()
    uvicorn.run(
        "apple_fm_serve.main:app",
        host=settings.host,
        port=settings.port,
        factory=False,
    )
