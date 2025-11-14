"""Integration tests for CoreGenAI."""

import os
import pytest
from coregenai import CoreGenAIClient, CoreGenAIError
from coregenai.exceptions import BackendNotAvailableError, ConfigurationError


def _has_valid_credentials():
    """Check if valid credentials are available for testing."""
    return bool(
        os.getenv("GOOGLE_API_KEY") and
        os.getenv("GOOGLE_PROJECT_ID") and
        os.getenv("GOOGLE_API_KEY") != "test-api-key-123"
    )


class TestClientIntegration:
    """Integration tests for the complete client workflow."""
    
    def test_import_public_api(self):
        """Test that public API can be imported."""
        from coregenai import (
            CoreGenAIClient,
            CoreGenAIError,
            BackendNotAvailableError,
            ModelNotFoundError,
            AuthenticationError,
            ConfigurationError,
            APIRequestError,
        )
        
        assert CoreGenAIClient is not None
        assert CoreGenAIError is not None
        assert BackendNotAvailableError is not None
    
    def test_client_with_invalid_backend(self):
        """Test client initialization with invalid backend."""
        with pytest.raises(ConfigurationError, match="Unsupported backend"):
            CoreGenAIClient(backend="invalid")
    
    def test_exception_hierarchy(self):
        """Test that all exceptions inherit properly."""
        from coregenai import (
            CoreGenAIError,
            BackendNotAvailableError,
            APIRequestError,
        )
        
        assert issubclass(BackendNotAvailableError, CoreGenAIError)
        assert issubclass(APIRequestError, CoreGenAIError)
        assert issubclass(CoreGenAIError, Exception)
    
    def test_client_error_handling(self):
        """Test that client errors can be caught properly."""
        try:
            client = CoreGenAIClient(backend="invalid")
        except CoreGenAIError as e:
            assert isinstance(e, ConfigurationError)
    
    def test_client_initialization_with_valid_backend_names(self):
        """Test client initialization with valid backend names."""
        valid_backends = ["genai", "vertexai"]
        
        for backend in valid_backends:
            try:
                client = CoreGenAIClient(backend=backend)
                assert client is not None
            except (BackendNotAvailableError, ConfigurationError):
                # Expected if SDK not installed or not configured
                pass


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.skipif(
        not _has_valid_credentials(),
        reason="Valid credentials required for E2E tests"
    )
    def test_full_text_generation_workflow(self):
        """Test complete text generation workflow."""
        client = CoreGenAIClient(backend="vertexai")
        
        prompt = "Say hello in one word"
        response = client.generate_text(prompt)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.skipif(
        not _has_valid_credentials(),
        reason="Valid credentials required for E2E tests"
    )
    def test_full_streaming_workflow(self):
        """Test complete streaming workflow."""
        client = CoreGenAIClient(backend="vertexai")
        
        prompt = "Count to 3"
        chunks = list(client.stream_generate_text(prompt))
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_empty_prompt_validation(self):
        """Test that empty prompts are rejected."""
        try:
            client = CoreGenAIClient(backend="vertexai")
            
            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                client.generate_text("")
            
            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                client.generate_text("   ")
        except (BackendNotAvailableError, ConfigurationError):
            pytest.skip("Backend not available")
