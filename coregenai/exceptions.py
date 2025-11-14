"""Exception classes for CoreGenAI."""


class CoreGenAIError(Exception):
    """Base exception class for all CoreGenAI errors."""

    pass


class BackendNotAvailableError(CoreGenAIError):
    """Raised when a requested backend is not available or not installed.

    This typically occurs when the required SDK package is not installed
    or cannot be imported.
    """

    def __init__(self, backend_name: str) -> None:
        """Initialize the exception.

        Args:
            backend_name: Name of the unavailable backend.
        """
        super().__init__(
            f"The backend '{backend_name}' is not available in the current environment. "
            f"Please ensure the required SDK is installed."
        )
        self.backend_name = backend_name


class ModelNotFoundError(CoreGenAIError):
    """Raised when a requested model is not found or not accessible."""

    def __init__(self, model_name: str) -> None:
        """Initialize the exception.

        Args:
            model_name: Name of the model that was not found.
        """
        super().__init__(
            f"The model '{model_name}' was not found. "
            f"Please verify the model name and your access permissions."
        )
        self.model_name = model_name


class AuthenticationError(CoreGenAIError):
    """Raised when there is an authentication failure.

    This typically occurs when API keys or credentials are missing,
    invalid, or expired.
    """

    def __init__(
        self, message: str = "Authentication failed. Please check your credentials."
    ) -> None:
        """Initialize the exception.

        Args:
            message: Detailed error message.
        """
        super().__init__(message)


class ConfigurationError(CoreGenAIError):
    """Raised when there is a configuration error.

    This occurs when required configuration values are missing,
    invalid, or incompatible.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception.

        Args:
            message: Detailed error message describing the configuration issue.
        """
        super().__init__(f"Configuration Error: {message}")


class APIRequestError(CoreGenAIError):
    """Raised when an API request fails.

    This includes network errors, rate limiting, server errors,
    and other API-related failures.
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize the exception.

        Args:
            status_code: HTTP status code or error code.
            message: Detailed error message.
        """
        super().__init__(f"API Request Error (Status {status_code}): {message}")
        self.status_code = status_code
