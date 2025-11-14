"""Configuration management for CoreGenAI.

This module handles loading and validating application settings from
environment variables and .env files using Pydantic.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation.

    Settings are loaded from environment variables and .env files.
    All settings are validated using Pydantic validators.

    Example:
        >>> settings = Settings()
        >>> print(settings.model_name)
        'gemini-2.5-pro'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider Settings
    llm_provider: str = Field(
        default="vertexai", description="LLM provider (vertexai or genai)"
    )

    google_project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID for Vertex AI"
    )

    google_cloud_project: Optional[str] = Field(
        default=None, description="Alias for Google Cloud project ID"
    )

    google_location: str = Field(
        default="us-central1", description="Google Cloud region/location"
    )

    google_api_key: Optional[str] = Field(
        default=None, description="Google API key for GenAI SDK"
    )

    # Model Settings
    model_name: str = Field(
        default="gemini-2.5-pro", description="Name of the AI model to use"
    )

    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0-2.0)"
    )

    max_tokens: int = Field(
        default=2048, gt=0, description="Maximum number of output tokens"
    )

    # Retrieval Settings
    chunk_size: int = Field(
        default=300, gt=0, description="Document chunk size for retrieval"
    )

    chunk_overlap: int = Field(
        default=50, ge=0, description="Overlap between document chunks"
    )

    top_k_results: int = Field(
        default=5, gt=0, description="Number of top results to retrieve"
    )

    # Python Version
    python_version: str = Field(default="3.12", description="Target Python version")

    # Feature Flags
    enable_telemetry: bool = Field(
        default=False, description="Enable anonymous telemetry collection"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is not greater than chunk size.

        Args:
            v: The chunk overlap value.
            info: Validation context containing other field values.

        Returns:
            The validated chunk overlap value.

        Raises:
            ValueError: If chunk overlap is greater than chunk size.
        """
        chunk_size = info.data.get("chunk_size", 300)
        if v > chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than or equal to chunk_size ({chunk_size})"
            )
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.

    This function uses LRU cache to ensure settings are only loaded once.

    Returns:
        Application settings instance.
    """
    return Settings()
