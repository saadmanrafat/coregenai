"""Test cases for exception classes."""

import pytest
from coregenai.exceptions import (
    CoreGenAIError,
    BackendNotAvailableError,
    ModelNotFoundError,
    AuthenticationError,
    ConfigurationError,
    APIRequestError,
)


def test_core_genai_error_base():
    """Test that CoreGenAIError can be raised."""
    with pytest.raises(CoreGenAIError):
        raise CoreGenAIError("Test error")


def test_backend_not_available_error():
    """Test BackendNotAvailableError with backend name."""
    error = BackendNotAvailableError("vertexai")
    
    assert "vertexai" in str(error)
    assert "not available" in str(error)
    assert error.backend_name == "vertexai"
    assert isinstance(error, CoreGenAIError)


def test_model_not_found_error():
    """Test ModelNotFoundError with model name."""
    error = ModelNotFoundError("gemini-pro")
    
    assert "gemini-pro" in str(error)
    assert "not found" in str(error)
    assert error.model_name == "gemini-pro"
    assert isinstance(error, CoreGenAIError)


def test_authentication_error_default():
    """Test AuthenticationError with default message."""
    error = AuthenticationError()
    
    assert "Authentication failed" in str(error)
    assert isinstance(error, CoreGenAIError)


def test_authentication_error_custom_message():
    """Test AuthenticationError with custom message."""
    error = AuthenticationError("Invalid API key")
    
    assert "Invalid API key" in str(error)
    assert isinstance(error, CoreGenAIError)


def test_configuration_error():
    """Test ConfigurationError with message."""
    error = ConfigurationError("Missing API key")
    
    assert "Configuration Error" in str(error)
    assert "Missing API key" in str(error)
    assert isinstance(error, CoreGenAIError)


def test_api_request_error():
    """Test APIRequestError with status code and message."""
    error = APIRequestError(500, "Internal server error")
    
    assert "500" in str(error)
    assert "Internal server error" in str(error)
    assert error.status_code == 500
    assert isinstance(error, CoreGenAIError)


def test_exception_hierarchy():
    """Test that all exceptions inherit from CoreGenAIError."""
    exceptions = [
        BackendNotAvailableError("test"),
        ModelNotFoundError("test"),
        AuthenticationError(),
        ConfigurationError("test"),
        APIRequestError(500, "test"),
    ]
    
    for exc in exceptions:
        assert isinstance(exc, CoreGenAIError)
        assert isinstance(exc, Exception)


def test_exception_can_be_caught():
    """Test that exceptions can be caught properly."""
    try:
        raise BackendNotAvailableError("test")
    except CoreGenAIError as e:
        assert isinstance(e, BackendNotAvailableError)
    
    try:
        raise APIRequestError(404, "Not found")
    except CoreGenAIError as e:
        assert isinstance(e, APIRequestError)
        assert e.status_code == 404
