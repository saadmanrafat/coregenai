"""Adapter implementations for different AI backends.

This module provides concrete implementations of the CoreGenAIBackend
interface for various AI services (Google GenAI and Vertex AI).
"""

from typing import Iterable

from coregenai.backends import CoreGenAIBackend
from coregenai.exceptions import (
    BackendNotAvailableError,
    AuthenticationError,
    APIRequestError,
    ConfigurationError,
)
from coregenai.auth import Settings


class GoogleGenAIAdapter(CoreGenAIBackend):
    """Adapter for Google Generative AI (GenAI) backend.

    This adapter uses the google-genai SDK to interact with Google's
    Gemini models through their public API.

    Attributes:
        client: The Google GenAI client instance.

    Raises:
        BackendNotAvailableError: If google-genai package is not installed.
        ConfigurationError: If required API key is missing.
    """

    def __init__(self) -> None:
        """Initialize the Google GenAI adapter.

        Raises:
            BackendNotAvailableError: If google-genai SDK is not available.
            ConfigurationError: If API key is not configured.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise BackendNotAvailableError("google-genai") from e

        settings = Settings()

        if not settings.google_api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY environment variable is required for GenAI backend"
            )

        try:
            self.client = genai.Client(api_key=settings.google_api_key)
        except Exception as e:
            raise AuthenticationError(
                f"Failed to initialize GenAI client: {str(e)}"
            ) from e

    def generate_text(self, prompt: str) -> str:
        """Generate text using Google Generative AI.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            Generated text response as a string.

        Raises:
            ValueError: If prompt is empty.
            APIRequestError: If the API request fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            settings = Settings()
            response = self.client.models.generate_content(
                model=settings.model_name,
                contents=prompt,
            )
            return response.text
        except ValueError as e:
            raise
        except Exception as e:
            raise APIRequestError(500, str(e)) from e

    def stream_generate_text(self, prompt: str) -> Iterable[str]:
        """Stream text generation using Google Generative AI.

        Args:
            prompt: The input prompt for text generation.

        Yields:
            Chunks of generated text as they become available.

        Raises:
            ValueError: If prompt is empty.
            APIRequestError: If the API request fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            settings = Settings()
            response_stream = self.client.models.generate_content_stream(
                model=settings.model_name,
                contents=prompt,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except ValueError as e:
            raise
        except Exception as e:
            raise APIRequestError(500, str(e)) from e

    @classmethod
    def is_available(cls) -> bool:
        """Check if google-genai SDK is available.

        Returns:
            True if the SDK can be imported, False otherwise.
        """
        try:
            import google.genai  # noqa: F401

            return True
        except ImportError:
            return False


class VertexAIAdapter(CoreGenAIBackend):
    """Adapter for Vertex AI backend.

    This adapter uses the vertexai SDK to interact with Google's
    Gemini models through Vertex AI on Google Cloud Platform.

    Attributes:
        model: The Vertex AI GenerativeModel instance.

    Raises:
        BackendNotAvailableError: If vertexai package is not installed.
        ConfigurationError: If required configuration is missing.
    """

    def __init__(self) -> None:
        """Initialize the Vertex AI adapter.

        Raises:
            BackendNotAvailableError: If vertexai SDK is not available.
            ConfigurationError: If project ID or location is not configured.
            AuthenticationError: If authentication fails.
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as e:
            raise BackendNotAvailableError("vertexai") from e

        settings = Settings()

        # Validate required configuration
        project_id = settings.google_project_id or settings.google_cloud_project
        if not project_id and not settings.google_api_key:
            raise ConfigurationError(
                "Either GOOGLE_PROJECT_ID or GOOGLE_API_KEY must be configured for Vertex AI"
            )

        try:
            # Initialize Vertex AI
            init_kwargs = {
                "location": settings.google_location,
            }

            if settings.google_api_key:
                init_kwargs["api_key"] = settings.google_api_key

            if project_id:
                init_kwargs["project"] = project_id

            vertexai.init(**init_kwargs)

            # Initialize the model
            self.model = GenerativeModel(settings.model_name)
        except Exception as e:
            raise AuthenticationError(
                f"Failed to initialize Vertex AI: {str(e)}"
            ) from e

    def generate_text(self, prompt: str) -> str:
        """Generate text using Vertex AI.

        Args:
            prompt: The input prompt for text generation.

        Returns:
            Generated text response as a string.

        Raises:
            ValueError: If prompt is empty.
            APIRequestError: If the API request fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            response = self.model.generate_content(prompt)

            # Handle multi-part responses
            if not response.candidates:
                raise APIRequestError(500, "No response candidates returned from API")

            parts = response.candidates[0].content.parts
            if not parts:
                raise APIRequestError(500, "No content parts in response")

            return "".join(part.text for part in parts if part.text)
        except ValueError as e:
            raise
        except APIRequestError as e:
            raise
        except Exception as e:
            raise APIRequestError(500, str(e)) from e

    def stream_generate_text(self, prompt: str) -> Iterable[str]:
        """Stream text generation using Vertex AI.

        Args:
            prompt: The input prompt for text generation.

        Yields:
            Chunks of generated text as they become available.

        Raises:
            ValueError: If prompt is empty.
            APIRequestError: If the API request fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            response_stream = self.model.generate_content(prompt, stream=True)

            for chunk in response_stream:
                if chunk.candidates:
                    parts = chunk.candidates[0].content.parts
                    for part in parts:
                        if part.text:
                            yield part.text
        except ValueError as e:
            raise
        except Exception as e:
            raise APIRequestError(500, str(e)) from e

    @classmethod
    def is_available(cls) -> bool:
        """Check if vertexai SDK is available.

        Returns:
            True if the SDK can be imported, False otherwise.
        """
        try:
            import vertexai  # noqa: F401

            return True
        except ImportError:
            return False


if __name__ == "__main__":
    # Example usage
    adapter = GoogleGenAIAdapter()
    response = adapter.generate_text(
        prompt="Explain the theory of relativity in simple terms."
    )
    print(response)
