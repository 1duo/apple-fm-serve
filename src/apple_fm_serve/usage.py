from dataclasses import dataclass


@dataclass(slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_usage(prompt: str, completion: str) -> Usage:
    return Usage(
        prompt_tokens=estimate_tokens(prompt), completion_tokens=estimate_tokens(completion)
    )
