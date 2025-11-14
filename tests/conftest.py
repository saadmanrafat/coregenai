"""Pytest configuration and fixtures for CoreGenAI tests."""

import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-123")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("GOOGLE_LOCATION", "us-central1")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-pro")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    monkeypatch.setenv("MAX_TOKENS", "2048")
    monkeypatch.setenv("CHUNK_SIZE", "300")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")


@pytest.fixture
def clear_settings_cache():
    """Clear the settings cache before and after tests."""
    from coregenai.auth import get_settings
    
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
