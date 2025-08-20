from typing import Optional


class GeminiError(Exception):
    """Base exception for Gemini API errors"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GeminiAPIError(GeminiError):
    """Exception for Gemini API specific errors"""

    pass


class GeminiAuthenticationError(GeminiAPIError):
    """Exception for authentication errors"""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401)


class GeminiRateLimitError(GeminiAPIError):
    """Exception for rate limit errors"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class GeminiModelNotFoundError(GeminiAPIError):
    """Exception for model not found errors"""

    def __init__(self, model: str):
        super().__init__(f"Model '{model}' not found", status_code=404)


class GeminiInvalidRequestError(GeminiAPIError):
    """Exception for invalid request errors"""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class GeminiContextLengthExceededError(GeminiAPIError):
    """Exception for context length exceeded errors"""

    def __init__(self, message: str = "Maximum context length exceeded"):
        super().__init__(message, status_code=400)


class GeminiServerError(GeminiAPIError):
    """Exception for server errors"""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, status_code=500)


class GeminiTimeoutError(GeminiAPIError):
    """Exception for timeout errors"""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, status_code=504)
