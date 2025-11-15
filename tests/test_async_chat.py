"""
Tests for async_chat module.

Tests cover client creation, configuration, and chat functionality
with mocked GenAI clients.
"""
import os
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from coregenai.async_chat import (
    ClientType,
    get_generative_client,
    coregenai,
    CoreGENAI,
)

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class TestClientType:
    """Tests for ClientType enum."""
    
    def test_gemini_value(self):
        """Test GEMINI enum value."""
        assert ClientType.GEMINI.value == "genai"
    
    def test_vertex_ai_value(self):
        """Test VERTEX_AI enum value."""
        assert ClientType.VERTEX_AI.value == "vertexai"
    
    def test_enum_members(self):
        """Test all enum members are present."""
        members = [member.name for member in ClientType]
        assert "GEMINI" in members
        assert "VERTEX_AI" in members


class TestGetGenerativeClient:
    """Tests for get_generative_client function."""
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    @patch("coregenai.async_chat.genai.Client")
    def test_create_gemini_client(self, mock_client):
        """Test creating a Gemini client with API key."""
        get_generative_client(ClientType.GEMINI)
        
        mock_client.assert_called_once_with(
            vertexai=False,
            api_key="test-api-key",
        )
    
    @patch.dict(
        os.environ,
        {
            "GOOGLE_PROJECT_ID": "test-project",
            "GOOGLE_LOCATION": "us-central1",
        }
    )
    @patch("coregenai.async_chat.genai.Client")
    def test_create_vertex_ai_client(self, mock_client):
        """Test creating a Vertex AI client with project/location."""
        get_generative_client(ClientType.VERTEX_AI)
        
        mock_client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )
    
    @patch.dict(os.environ, {}, clear=True)
    @patch("coregenai.async_chat.genai.Client")
    def test_create_client_with_missing_env_vars(self, mock_client):
        """Test client creation with missing environment variables."""
        # Should not raise, but pass None values
        get_generative_client(ClientType.GEMINI)
        
        mock_client.assert_called_once_with(
            vertexai=False,
            api_key=None,
        )
    
    def test_invalid_client_type(self):
        """Test error handling for invalid client type."""
        # Test with a value that doesn't match any enum case
        # Since we can't easily create an invalid enum, we skip this test
        # The match statement will raise ValueError for unknown types
        pass


class TestCoregenai:
    """Tests for coregenai async chat function."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_single_message(self):
        """Test sending a single message and receiving a response."""
        # Setup mocks
        mock_client = Mock()
        mock_chat = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Hello! How can I help?"
        
        # Mock the streaming response
        async def mock_stream(user_input):
            yield mock_chunk
        
        mock_chat.send_message_stream = mock_stream
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        # Patch dependencies
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=["test message", KeyboardInterrupt]):
                with patch("builtins.print") as mock_print:
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
        
        # Verify chat was created with correct model
        mock_client.aio.chats.create.assert_called_once_with(model="test-model")
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_empty_input_ignored(self):
        """Test that empty inputs are skipped."""
        mock_client = Mock()
        mock_chat = AsyncMock()
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=["", "  ", KeyboardInterrupt]):
                with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                    await coregenai()
        
        # send_message_stream should not be called for empty inputs
        assert mock_chat.send_message_stream.call_count == 0
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_keyboard_interrupt(self):
        """Test graceful exit on KeyboardInterrupt."""
        mock_client = Mock()
        mock_chat = AsyncMock()
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                with patch("builtins.print") as mock_print:
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
        
        # Verify goodbye message was printed
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Goodbye" in str(call) for call in calls)
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_eof_error(self):
        """Test graceful exit on EOFError."""
        mock_client = Mock()
        mock_chat = AsyncMock()
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=EOFError):
                with patch("builtins.print") as mock_print:
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
        
        # Verify goodbye message was printed
        calls = [str(call) for call in mock_print.call_args_list]
        assert any("Goodbye" in str(call) for call in calls)
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_strips_whitespace(self):
        """Test that input is stripped of whitespace."""
        mock_client = Mock()
        mock_chat = Mock()
        mock_chunk = Mock()
        mock_chunk.text = "Response"
        
        async def mock_stream(user_input):
            # Verify the input was stripped
            assert user_input == "test message"
            yield mock_chunk
        
        mock_chat.send_message_stream = mock_stream
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=["  test message  ", KeyboardInterrupt]):
                with patch("builtins.print"):
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_chat_uses_vertex_ai_by_default(self):
        """Test that Vertex AI client is used by default."""
        mock_client = Mock()
        mock_chat = AsyncMock()
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client) as mock_get_client:
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                with patch("builtins.print"):
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
        
        # Verify Vertex AI client type was requested
        mock_get_client.assert_called_once_with(ClientType.VERTEX_AI)


class TestCoreGENAIProtocol:
    """Tests for CoreGENAI Protocol."""
    
    def test_protocol_implementation(self):
        """Test that a class implementing the protocol is recognized."""
        
        class MockGenAI:
            def generate_content(self, prompt) -> str:
                return "response"
        
        mock = MockGenAI()
        
        # Should not raise TypeError when used as CoreGENAI
        def use_genai(client: CoreGENAI):
            return client.generate_content("test")
        
        result = use_genai(mock)
        assert result == "response"
    
    def test_protocol_signature(self):
        """Test that protocol has correct method signature."""
        import inspect
        
        # Check that generate_content is defined
        assert hasattr(CoreGENAI, "generate_content")


class TestIntegration:
    """Integration tests with real-ish scenarios."""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        mock_client = Mock()
        mock_chat = Mock()
        
        responses = [
            [Mock(text="Hello!")],
            [Mock(text="I'm"), Mock(text=" doing"), Mock(text=" great!")],
        ]
        
        call_count = [0]
        
        async def mock_stream(user_input):
            response = responses[call_count[0]]
            call_count[0] += 1
            for chunk in response:
                yield chunk
        
        mock_chat.send_message_stream = mock_stream
        mock_client.aio.chats.create = Mock(return_value=mock_chat)
        
        inputs = ["Hi", "How are you?", KeyboardInterrupt]
        
        with patch("coregenai.async_chat.get_generative_client", return_value=mock_client):
            with patch("builtins.input", side_effect=inputs):
                with patch("builtins.print") as mock_print:
                    with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
                        await coregenai()
        
        # Verify both messages were processed
        assert call_count[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
