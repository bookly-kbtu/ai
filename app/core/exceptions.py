class AppError(Exception):
    """Base application error."""


class AuthError(AppError):
    """Missing or invalid access token."""


class UpstreamError(AppError):
    """The Go backend rejected a request."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class LLMError(AppError):
    """The LLM provider failed or is not configured."""
