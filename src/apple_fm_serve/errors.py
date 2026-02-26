from dataclasses import dataclass


@dataclass(slots=True)
class OpenAIErrorBody:
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class AdapterError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.body = OpenAIErrorBody(message=message, type=error_type, param=param, code=code)
        self.status_code = status_code


class AuthError(AdapterError):
    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(
            message, status_code=401, error_type="authentication_error", code="invalid_api_key"
        )


class ProviderUnavailableError(AdapterError):
    def __init__(self, reason: str | None = None) -> None:
        message = "Model provider unavailable"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(
            message,
            status_code=503,
            error_type="server_error",
            code="provider_unavailable",
        )
