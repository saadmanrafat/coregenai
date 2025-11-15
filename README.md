# coregenai 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Built with frustration from using Google's fragmented SDKs.

![](ext/coregenai.png)

### This is an experimental prototype (coregenai v0.1.0)

## Example

```python
import os

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

```

## Features

**That's it.** `coregenai` handles:
- Auto-detection of API keys, ADC, Vertex AI (`TODO`)
- Unified API across Gemini API and Vertex AI
- Type-safe responses with Pydantic
- Streaming that actually works



## Installation

```bash
cp .env.example .env
```

```bash
pip install coregenai
```


## TODO

- Add vision model support for image inputs
- Implement function calling and tool use
- Add chat session management with history
- Support for async/await operations
- Add OpenAI-compatible API endpoint
- Implement rate limiting and retry logic
- Add prompt caching for repeated queries
- Support for embeddings and semantic search
- Add batch processing capabilities
- Implement cost tracking and usage analytics

---


**Built by <a href="https://saadman.dev">Saadman</a> maintained with ❤️ by developers tired of SDK confusion.**


