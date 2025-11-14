"""Abstract base class for CoreGenAI backends."""

from abc import ABC, abstractmethod
from typing import Iterable


class CoreGenAIBackend(ABC):
    """Abstract base class defining the common interface for CoreGenAI backends.

    This class serves as the target interface for the Adapter pattern,
    allowing different AI backends (Vertex AI, Google GenAI) to be used
    interchangeably through a common interface.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the backend client.

        Raises:
            BackendNotAvailableError: If the backend SDK is not installed.
            AuthenticationError: If authentication credentials are invalid.
            ConfigurationError: If required configuration is missing.
        """
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if the underlying SDK is installed and available.

        Returns:
            True if the backend SDK is available, False otherwise.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate a text response for a given prompt.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            The generated text response.

        Raises:
            APIRequestError: If the API request fails.
            ValueError: If the prompt is empty or invalid.
        """
        pass

    @abstractmethod
    def stream_generate_text(self, prompt: str) -> Iterable[str]:
        """Stream text generation for a given prompt.

        Args:
            prompt: The input prompt for text generation.

        Yields:
            Chunks of generated text as they become available.

        Raises:
            APIRequestError: If the API request fails.
            ValueError: If the prompt is empty or invalid.
        """
        pass
