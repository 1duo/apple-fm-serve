from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from apple_fm_serve.config import Settings
from apple_fm_serve.main import create_app
from apple_fm_serve.providers.mock import MockProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class SlowProvider:
    async def is_available(self) -> tuple[bool, str | None]:
        return True, None

    async def list_models(self) -> list[str]:
        return ["apple.fm.system"]

    async def generate(self, request: object) -> str:
        _ = request
        await asyncio.sleep(0.05)
        return "late"

    def stream_generate(self, request: object) -> AsyncIterator[str]:
        _ = request

        async def _iter() -> AsyncIterator[str]:
            await asyncio.sleep(0.05)
            yield "late"

        return _iter()


@pytest.mark.asyncio
async def test_models_endpoint() -> None:
    provider = MockProvider()
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "apple.fm.system"


@pytest.mark.asyncio
async def test_chat_completions_non_stream() -> None:
    provider = MockProvider(response_text="hello back")
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "hello back"
    assert payload["usage"]["total_tokens"] > 0
    assert provider.last_request is not None
    assert provider.last_request.instructions is None


@pytest.mark.asyncio
async def test_chat_completions_stream() -> None:
    provider = MockProvider(stream_snapshots=["H", "He", "Hello"])
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        client.stream("POST", "/v1/chat/completions", json=body) as response,
    ):
        text = (await response.aread()).decode()

    assert response.status_code == 200
    assert "data: [DONE]" in text
    assert '"role":"assistant"' in text
    assert '"content":"H"' in text
    assert '"content":"e"' in text
    assert '"content":"llo"' in text


@pytest.mark.asyncio
async def test_chat_completions_stream_accepts_stream_options() -> None:
    provider = MockProvider(stream_snapshots=["H", "He", "Hello"])
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        client.stream("POST", "/v1/chat/completions", json=body) as response,
    ):
        text = (await response.aread()).decode()

    assert response.status_code == 200
    assert "data: [DONE]" in text


@pytest.mark.asyncio
async def test_chat_completions_accepts_assistant_tool_calls_in_history() -> None:
    provider = MockProvider(response_text="final answer")
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "messages": [
            {"role": "user", "content": "do thing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": '{"x":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
            {"role": "user", "content": "continue"},
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "final answer"


@pytest.mark.asyncio
async def test_auth_required() -> None:
    provider = MockProvider()
    app = create_app(
        settings=Settings(model_id="apple.fm.system", api_key="secret"), provider=provider
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_response_format_json_schema_forwarded() -> None:
    provider = MockProvider(response_text=json.dumps({"x": 1}))
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "obj",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        },
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert provider.last_request is not None
    assert provider.last_request.response_schema is not None
    assert provider.last_request.response_schema["type"] == "object"


@pytest.mark.asyncio
async def test_tools_accepted_and_ignored() -> None:
    provider = MockProvider()
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                },
            }
        ],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_readyz_unavailable() -> None:
    provider = MockProvider(available=False, unavailable_reason="not ready")
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "provider_unavailable"
    assert "not ready" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_chat_completions_model_not_found() -> None:
    provider = MockProvider()
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "wrong.model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


@pytest.mark.asyncio
async def test_chat_completions_tool_choice_accepted() -> None:
    provider = MockProvider()
    app = create_app(settings=Settings(model_id="apple.fm.system"), provider=provider)

    body = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": {"type": "function", "function": {"name": "noop"}},
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_non_stream_timeout() -> None:
    provider = SlowProvider()
    app = create_app(
        settings=Settings(model_id="apple.fm.system", request_timeout_s=0.001), provider=provider
    )

    body = {
        "model": "apple.fm.system",
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


@pytest.mark.asyncio
async def test_chat_completions_stream_timeout() -> None:
    provider = SlowProvider()
    app = create_app(
        settings=Settings(model_id="apple.fm.system", request_timeout_s=0.001), provider=provider
    )

    body = {
        "model": "apple.fm.system",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }

    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        client.stream("POST", "/v1/chat/completions", json=body) as response,
    ):
        text = (await response.aread()).decode()

    assert response.status_code == 200
    assert '"type":"timeout_error"' in text
    assert "data: [DONE]" not in text
