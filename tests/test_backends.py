"""Test cases for backend abstract base class."""

import pytest
from coregenai.backends import CoreGenAIBackend


def test_backend_is_abstract():
    """Test that CoreGenAIBackend cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        CoreGenAIBackend()


def test_backend_requires_init_implementation():
    """Test that __init__ must be implemented."""
    class IncompleteBackend(CoreGenAIBackend):
        def is_available(cls):
            return True
        
        def generate_text(self, prompt):
            return "test"
        
        def stream_generate_text(self, prompt):
            yield "test"
    
    with pytest.raises(TypeError):
        IncompleteBackend()


def test_backend_requires_is_available_implementation():
    """Test that is_available must be implemented."""
    class IncompleteBackend(CoreGenAIBackend):
        def __init__(self):
            pass
        
        def generate_text(self, prompt):
            return "test"
        
        def stream_generate_text(self, prompt):
            yield "test"
    
    with pytest.raises(TypeError):
        IncompleteBackend()


def test_backend_requires_generate_text_implementation():
    """Test that generate_text must be implemented."""
    class IncompleteBackend(CoreGenAIBackend):
        def __init__(self):
            pass
        
        @classmethod
        def is_available(cls):
            return True
        
        def stream_generate_text(self, prompt):
            yield "test"
    
    with pytest.raises(TypeError):
        IncompleteBackend()


def test_backend_requires_stream_generate_text_implementation():
    """Test that stream_generate_text must be implemented."""
    class IncompleteBackend(CoreGenAIBackend):
        def __init__(self):
            pass
        
        @classmethod
        def is_available(cls):
            return True
        
        def generate_text(self, prompt):
            return "test"
    
    with pytest.raises(TypeError):
        IncompleteBackend()


def test_complete_backend_implementation():
    """Test that a complete implementation can be instantiated."""
    class CompleteBackend(CoreGenAIBackend):
        def __init__(self):
            self.initialized = True
        
        @classmethod
        def is_available(cls):
            return True
        
        def generate_text(self, prompt):
            return f"Generated: {prompt}"
        
        def stream_generate_text(self, prompt):
            for word in prompt.split():
                yield word
    
    backend = CompleteBackend()
    assert backend.initialized is True
    assert CompleteBackend.is_available() is True
    assert backend.generate_text("test") == "Generated: test"
    assert list(backend.stream_generate_text("hello world")) == ["hello", "world"]
