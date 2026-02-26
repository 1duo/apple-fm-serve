import pytest

from apple_fm_serve.errors import AdapterError
from apple_fm_serve.openai_types import ChatMessage
from apple_fm_serve.prompt import build_prompt


def test_build_prompt_with_instructions_and_turns() -> None:
    built = build_prompt(
        [
            ChatMessage(role="system", content="You are concise."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
            ChatMessage(role="user", content="Summarize this."),
        ]
    )

    assert built.instructions == "You are concise."
    assert "User: Hello" in built.prompt
    assert "Assistant: Hi" in built.prompt
    assert built.prompt.endswith("Assistant:")


def test_build_prompt_rejects_image_parts() -> None:
    with pytest.raises(AdapterError) as exc:
        build_prompt(
            [
                ChatMessage(
                    role="user",
                    content=[
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                    ],
                )
            ]
        )

    assert exc.value.body.code == "image_not_supported"
