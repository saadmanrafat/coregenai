"""Facade for the unified Gemini SDK.

This module provides a high-level interface for interacting with
different AI backends through a unified API.
"""

from typing import Iterable, Literal

from coregenai.adapters import GoogleGenAIAdapter, VertexAIAdapter
from coregenai.backends import CoreGenAIBackend
from coregenai.exceptions import ConfigurationError


BackendType = Literal["genai", "vertexai"]

BACKEND_REGISTRY = {
    "genai": GoogleGenAIAdapter,
    "vertexai": VertexAIAdapter,
}


def get_backend(backend: BackendType) -> CoreGenAIBackend:
    """Get an instance of the specified backend.

    Args:
        backend: The backend type to instantiate ("genai" or "vertexai").

    Returns:
        An initialized backend instance.

    Raises:
        ConfigurationError: If the backend type is not supported.
    """
    if backend not in BACKEND_REGISTRY:
        raise ConfigurationError(
            f"Unsupported backend: '{backend}'. "
            f"Supported backends: {', '.join(BACKEND_REGISTRY.keys())}"
        )

    adapter_class = BACKEND_REGISTRY[backend]
    return adapter_class()


class CoreGenAIClient:
    """Unified client for interacting with Gemini models.

    This class provides a simple, high-level API for text generation,
    abstracting away the complexity of different backend implementations.

    Attributes:
        _backend: The underlying backend adapter instance.

    Example:
        >>> client = CoreGenAIClient(backend="vertexai")
        >>> response = client.generate_text("Hello, world!")
        >>> print(response)
    """

    def __init__(self, backend: BackendType = "vertexai") -> None:
        """Initialize the CoreGenAI client.

        Args:
            backend: The backend to use ("genai" or "vertexai").
                     Defaults to "vertexai".

        Raises:
            BackendNotAvailableError: If the backend SDK is not available.
            ConfigurationError: If the backend is not supported or
                               required configuration is missing.
            AuthenticationError: If authentication fails.
        """
        self._backend: CoreGenAIBackend = get_backend(backend)

    def generate_text(self, prompt: str) -> str:
        """Generate text response for a given prompt.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the prompt is empty.
            APIRequestError: If the API request fails.

        Example:
            >>> client = CoreGenAIClient()
            >>> response = client.generate_text("What is AI?")
            >>> print(response)
        """
        return self._backend.generate_text(prompt)

    def stream_generate_text(self, prompt: str) -> Iterable[str]:
        """Stream text generation for a given prompt.

        Args:
            prompt: The input prompt for text generation.

        Yields:
            Chunks of generated text as they become available.

        Raises:
            ValueError: If the prompt is empty.
            APIRequestError: If the API request fails.

        Example:
            >>> client = CoreGenAIClient()
            >>> for chunk in client.stream_generate_text("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        return self._backend.stream_generate_text(prompt)


