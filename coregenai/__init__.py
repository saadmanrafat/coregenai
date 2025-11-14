"""CoreGenAI - Unified SDK for Google Gemini models.

This package provides a unified interface for interacting with Google's
Gemini models through different backends (Vertex AI and GenAI).
"""

from .facade import CoreGenAIClient
from .exceptions import (
    CoreGenAIError,
    BackendNotAvailableError,
    ModelNotFoundError,
    AuthenticationError,
    ConfigurationError,
    APIRequestError,
)

__version__ = "0.1.0"

__all__ = [
    "CoreGenAIClient",
    "CoreGenAIError",
    "BackendNotAvailableError",
    "ModelNotFoundError",
    "AuthenticationError",
    "ConfigurationError",
    "APIRequestError",
]
