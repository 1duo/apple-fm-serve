from collections.abc import AsyncIterator

from apple_fm_serve.providers.base import GenerationInput


class MockProvider:
    def __init__(
        self,
        *,
        model_id: str = "apple.fm.system",
        available: bool = True,
        unavailable_reason: str | None = None,
        response_text: str = "ok",
        stream_snapshots: list[str] | None = None,
    ) -> None:
        self._model_id = model_id
        self._available = available
        self._unavailable_reason = unavailable_reason
        self._response_text = response_text
        self._stream_snapshots = stream_snapshots or ["o", "ok"]
        self.last_request: GenerationInput | None = None

    async def is_available(self) -> tuple[bool, str | None]:
        return self._available, self._unavailable_reason

    async def list_models(self) -> list[str]:
        return [self._model_id]

    async def generate(self, request: GenerationInput) -> str:
        self.last_request = request
        return self._response_text

    async def stream_generate(self, request: GenerationInput) -> AsyncIterator[str]:
        self.last_request = request
        for snapshot in self._stream_snapshots:
            yield snapshot
