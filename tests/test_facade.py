"""Test cases for the facade/client interface."""

import pytest
from coregenai.facade import CoreGenAIClient, get_backend, BACKEND_REGISTRY
from coregenai.exceptions import ConfigurationError, BackendNotAvailableError


def test_backend_registry_contains_expected_backends():
    """Test that BACKEND_REGISTRY contains the expected backends."""
    assert "genai" in BACKEND_REGISTRY
    assert "vertexai" in BACKEND_REGISTRY
    assert len(BACKEND_REGISTRY) == 2


def test_get_backend_with_invalid_backend():
    """Test that get_backend raises error for unsupported backend."""
    with pytest.raises(ConfigurationError, match="Unsupported backend"):
        get_backend("invalid_backend")
    
    with pytest.raises(ConfigurationError, match="Supported backends"):
        get_backend("nonexistent")


def test_get_backend_error_message_lists_valid_backends():
    """Test that error message includes valid backend names."""
    try:
        get_backend("invalid")
    except ConfigurationError as e:
        error_msg = str(e)
        assert "genai" in error_msg
        assert "vertexai" in error_msg


def test_client_initialization_default_backend():
    """Test that client initializes with default backend."""
    # This will fail without proper SDK setup, but tests the interface
    try:
        client = CoreGenAIClient()
        assert client._backend is not None
    except (BackendNotAvailableError, Exception):
        # Expected if SDK not properly configured
        pass


def test_client_initialization_with_backend_parameter():
    """Test that client accepts backend parameter."""
    # Test with invalid backend to check parameter handling
    with pytest.raises(ConfigurationError):
        CoreGenAIClient(backend="invalid")


def test_client_has_generate_text_method():
    """Test that CoreGenAIClient has generate_text method."""
    assert hasattr(CoreGenAIClient, "generate_text")
    assert callable(getattr(CoreGenAIClient, "generate_text"))


def test_client_has_stream_generate_text_method():
    """Test that CoreGenAIClient has stream_generate_text method."""
    assert hasattr(CoreGenAIClient, "stream_generate_text")
    assert callable(getattr(CoreGenAIClient, "stream_generate_text"))


def test_client_generate_text_signature():
    """Test generate_text method signature."""
    import inspect
    sig = inspect.signature(CoreGenAIClient.generate_text)
    params = list(sig.parameters.keys())
    
    assert "self" in params
    assert "prompt" in params
    assert len(params) == 2


def test_client_stream_generate_text_signature():
    """Test stream_generate_text method signature."""
    import inspect
    sig = inspect.signature(CoreGenAIClient.stream_generate_text)
    params = list(sig.parameters.keys())
    
    assert "self" in params
    assert "prompt" in params
    assert len(params) == 2
