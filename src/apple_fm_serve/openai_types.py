from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

MessagePart: TypeAlias = dict[str, JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ChatMessage(_StrictBase):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[MessagePart] | None = None
    tool_call_id: str | None = None


class JsonSchemaPayload(_StrictBase):
    name: str | None = None
    schema_: JsonObject = Field(alias="schema", serialization_alias="schema")
    strict: bool | None = None


class ResponseFormat(_StrictBase):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaPayload | None = None


class ChatCompletionsRequest(_StrictBase):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    tools: list[JsonObject] | None = None
    tool_choice: JsonValue | None = None
    response_format: ResponseFormat | None = None


class ChatCompletionMessage(_StrictBase):
    role: Literal["assistant"] = "assistant"
    content: str | None = None


class ChatCompletionChoice(_StrictBase):
    index: int
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None


class UsageObject(_StrictBase):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(_StrictBase):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageObject | None = None


class DeltaMessage(_StrictBase):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(_StrictBase):
    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None


class ChatCompletionsChunk(_StrictBase):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: UsageObject | None = None


class ModelObject(_StrictBase):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "apple"


class ModelsResponse(_StrictBase):
    object: Literal["list"] = "list"
    data: list[ModelObject]
