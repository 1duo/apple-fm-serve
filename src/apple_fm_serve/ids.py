from uuid import uuid4


def completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


def model_response_id() -> str:
    return f"resp-{uuid4().hex}"
