# coregenai 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Built with frustration from using Google's fragmented SDKs.

<img src="ext/coregenai.png" alt="CoreGenAI">

### This is an experimental prototype (coregenai v0.1.0)

## Example

```python
from coregenai import CoreGenAIClient

client = CoreGenAIClient(backend="vertexai") # or google-genai
response = client.generate_text("Why does Google API Documentation's suck?")
print(response)

# Or stream it
for chunk in client.stream_generate_text("Please make Google docs better?"):
    print(chunk, end="", flush=True)
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


