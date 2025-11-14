"""Test cases for authentication and configuration."""

import pytest
from coregenai.auth import Settings, get_settings


def test_settings_default_values():
    """Test that Settings has correct default values."""
    settings = Settings()
    
    assert settings.llm_provider == "vertexai"
    assert settings.google_location == "us-central1"
    assert settings.model_name == "gemini-2.5-pro"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 2048
    assert settings.chunk_size == 300
    assert settings.chunk_overlap == 50
    assert settings.top_k_results == 5
    assert settings.python_version == "3.12"
    assert settings.enable_telemetry is False


def test_settings_from_env(monkeypatch):
    """Test that Settings loads from environment variables."""
    monkeypatch.setenv("LLM_PROVIDER", "genai")
    monkeypatch.setenv("GOOGLE_LOCATION", "us-east1")
    monkeypatch.setenv("MODEL_NAME", "gemini-pro")
    monkeypatch.setenv("TEMPERATURE", "0.9")
    monkeypatch.setenv("MAX_TOKENS", "4096")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    
    # Clear cache to load new values
    get_settings.cache_clear()
    settings = Settings()
    
    assert settings.llm_provider == "genai"
    assert settings.google_location == "us-east1"
    assert settings.model_name == "gemini-pro"
    assert settings.temperature == 0.9
    assert settings.max_tokens == 4096
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100


def test_settings_temperature_validation():
    """Test temperature validation bounds."""
    # Valid temperatures
    settings = Settings(temperature=0.0)
    assert settings.temperature == 0.0
    
    settings = Settings(temperature=2.0)
    assert settings.temperature == 2.0
    
    settings = Settings(temperature=1.5)
    assert settings.temperature == 1.5


def test_settings_max_tokens_validation():
    """Test max_tokens must be positive."""
    settings = Settings(max_tokens=1)
    assert settings.max_tokens == 1
    
    settings = Settings(max_tokens=10000)
    assert settings.max_tokens == 10000


def test_settings_chunk_size_validation():
    """Test chunk_size must be positive."""
    settings = Settings(chunk_size=100, chunk_overlap=10)
    assert settings.chunk_size == 100
    
    settings = Settings(chunk_size=5000, chunk_overlap=100)
    assert settings.chunk_size == 5000


def test_settings_chunk_overlap_validation():
    """Test chunk_overlap validation against chunk_size."""
    # Valid: overlap less than size
    settings = Settings(chunk_size=300, chunk_overlap=50)
    assert settings.chunk_overlap == 50
    
    # Valid: overlap equal to size
    settings = Settings(chunk_size=300, chunk_overlap=300)
    assert settings.chunk_overlap == 300
    
    # Invalid: overlap greater than size
    with pytest.raises(ValueError, match="chunk_overlap.*must be less than or equal"):
        Settings(chunk_size=300, chunk_overlap=301)


def test_settings_top_k_validation():
    """Test top_k_results must be positive."""
    settings = Settings(top_k_results=1)
    assert settings.top_k_results == 1
    
    settings = Settings(top_k_results=100)
    assert settings.top_k_results == 100


def test_get_settings_caching():
    """Test that get_settings returns cached instance."""
    get_settings.cache_clear()
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    # Should be the same instance
    assert settings1 is settings2


def test_settings_case_insensitive(monkeypatch):
    """Test that settings are case insensitive."""
    monkeypatch.setenv("model_name", "gemini-pro")
    monkeypatch.setenv("TEMPERATURE", "0.8")
    
    get_settings.cache_clear()
    settings = Settings()
    
    assert settings.model_name == "gemini-pro"
    assert settings.temperature == 0.8


def test_settings_optional_fields():
    """Test that optional fields can be None."""
    settings = Settings(
        google_api_key=None,
        google_project_id=None,
        google_cloud_project=None
    )
    
    assert settings.google_api_key is None
    assert settings.google_project_id is None
    assert settings.google_cloud_project is None
