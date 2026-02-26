from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AdapterError
from .openai_types import ChatMessage, MessagePart


@dataclass(slots=True)
class PromptBuildResult:
    instructions: str | None
    prompt: str


def _message_content_to_text(content: str | list[MessagePart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    for content_part in content:
        part_type = content_part.get("type")
        if part_type == "image_url":
            raise AdapterError(
                "Image content is not supported by this adapter",
                status_code=400,
                param="messages",
                code="image_not_supported",
            )

        if part_type != "text":
            continue

        text_value = content_part.get("text")
        if isinstance(text_value, str) and text_value:
            text_parts.append(text_value)

    return "\n".join(text_parts)


def build_prompt(messages: Sequence[ChatMessage]) -> PromptBuildResult:
    if not messages:
        raise AdapterError(
            "messages must contain at least one item",
            status_code=400,
            param="messages",
            code="messages_required",
        )

    instructions_parts: list[str] = []
    turns: list[str] = []

    for message in messages:
        role = message.role
        text = _message_content_to_text(message.content)

        if role == "system":
            if text:
                instructions_parts.append(text)
            continue

        if role == "user":
            turns.append(f"User: {text}")
            continue

        if role == "assistant":
            turns.append(f"Assistant: {text}")
            continue

        if role == "tool":
            tool_call_id = message.tool_call_id
            prefix = "Tool"
            if isinstance(tool_call_id, str) and tool_call_id:
                prefix = f"Tool[{tool_call_id}]"
            turns.append(f"{prefix}: {text}")

    if not turns:
        raise AdapterError(
            "At least one non-system message is required",
            status_code=400,
            param="messages",
            code="no_turns",
        )

    prompt = (
        "You are answering the final user request in the following conversation.\n"
        "Return only the assistant response.\n\n" + "\n".join(turns) + "\nAssistant:"
    )

    instructions = "\n\n".join(part for part in instructions_parts if part) or None
    return PromptBuildResult(instructions=instructions, prompt=prompt)
