"""
Async chat interface for Google GenAI (Gemini and Vertex AI).

This module provides an interactive streaming chat interface with support for
both Google AI (Gemini) and Vertex AI backends.
"""
import enum
import os
import asyncio
from typing import Protocol, TypeAlias
from google import genai
from google.genai import types, Client

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(filename=".env", raise_error_if_not_found=True, usecwd=True))


class CoreGENAI(Protocol):
    """Protocol defining the interface for generative AI clients."""
    
    def generate_content(self, prompt) -> str:
        """
        Generate content from a prompt.
        
        Args:
            prompt: The input prompt text.
            
        Returns:
            Generated text response.
        """
        ...


ConcreteClient: TypeAlias = genai.Client | genai.Client


class ClientType(enum.Enum):
    """Enumeration of supported generative AI client types."""
    
    GEMINI = "genai"
    """Google AI (Gemini) client using API key authentication."""
    
    VERTEX_AI = "vertexai"
    """Vertex AI client using project/location authentication."""


def get_generative_client(client_type: ClientType) -> Client:
    """
    Create and configure a generative AI client.
    
    Args:
        client_type: Type of client to create (GEMINI or VERTEX_AI).
        
    Returns:
        Configured Client instance.
        
    Raises:
        ValueError: If client_type is not recognized.
        
    Environment Variables:
        For GEMINI:
            - GOOGLE_API_KEY: API key for Google AI
            
        For VERTEX_AI:
            - GOOGLE_PROJECT_ID: GCP project ID
            - GOOGLE_LOCATION: GCP region (e.g., 'us-central1')
    
    Examples:
        >>> client = get_generative_client(ClientType.GEMINI)
        >>> client = get_generative_client(ClientType.VERTEX_AI)
    """
    match client_type:
        case ClientType.GEMINI:
            return genai.Client(
                vertexai=False,
                api_key=os.environ.get("GOOGLE_API_KEY"),
            )
        case ClientType.VERTEX_AI:
            return genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_PROJECT_ID"),
                location=os.environ.get("GOOGLE_LOCATION"),
            )
        case _:
            raise ValueError("ClientType Unknown")


async def coregenai(use_vertex=True):
    """
    Run an interactive streaming chat session with Google GenAI.
    
    Creates a persistent chat session that maintains conversation context
    and streams responses in real-time.
    
    Args:
        use_vertex: Whether to use Vertex AI (True) or Gemini (False).
                   Currently defaults to True but parameter is unused.
    
    Environment Variables:
        MODEL_NAME: Name of the model to use (e.g., 'gemini-2.0-flash-exp').
    
    Raises:
        KeyboardInterrupt: User pressed Ctrl+C to exit.
        EOFError: Input stream ended (e.g., pipe closed).
    
    Examples:
        >>> asyncio.run(coregenai())
        > Hello!
        Hello! How can I help you today?
        > What is Python?
        Python is a high-level programming language...
    
    Notes:
        - Empty inputs are ignored
        - Press Ctrl+C or Ctrl+D to exit
        - Conversation context is maintained throughout the session
    """
    client = get_generative_client(ClientType.VERTEX_AI)
    chat = client.aio.chats.create(
        model=os.environ.get("MODEL_NAME"),
    )
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            async for chunk in await chat.send_message_stream(user_input):
                print(chunk.text, end="")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

