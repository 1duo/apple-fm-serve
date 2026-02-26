from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from apple_fm_serve.openai_types import JsonObject


@dataclass(slots=True)
class GenerationInput:
    instructions: str | None
    prompt: str
    response_schema: JsonObject | None


class LLMProvider(Protocol):
    async def is_available(self) -> tuple[bool, str | None]: ...

    async def list_models(self) -> list[str]: ...

    async def generate(self, request: GenerationInput) -> str: ...

    def stream_generate(self, request: GenerationInput) -> AsyncIterator[str]: ...
