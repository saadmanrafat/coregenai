# coregenai 
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/coregenai.svg)](https://badge.fury.io/py/coregenai)


**The production-grade wrapper for Google's Gemini SDK.**  
*Stop rewriting retry loops. Stop debugging cryptic auth errors.*

![](ext/coregenai.png)

## Why CoreGenAI?

The official `google-genai` SDK is powerful, but raw. To run it in production, you usually have to write 200+ lines of boilerplate: retry logic, exponential backoff, auth switching, and safety parsing.

**CoreGenAI** handles the infrastructure so you can focus on the prompts.

| Feature | The Raw SDK | CoreGenAI                                    |
| :--- | :--- |:---------------------------------------------|
| **Resilience** | Manual `try/except` loops | **Auto-Retries w/ Jitter** (Handles 429/503) |
| **Auth** | Complex IAM vs Key logic | **Auto-Detects** (Vertex vs Studio)          |

### This is an experimental prototype (coregenai v0.3.1)

## 🐍 Python Usage

### 1. The "Type-Safe" Generator
Stop parsing JSON with regex. Define a Pydantic model, and CoreGenAI guarantees the output matches it.

```python
import asyncio
from coregenai import CoreGenAI
from pydantic import BaseModel

# 1. Define your Schema
class SecurityAudit(BaseModel):
    risk_level: str
    vulnerabilities: list[str]
    is_critical: bool

async def main():
    # Auto-detects: 
    # - GOOGLE_APPLICATION_CREDENTIALS (Vertex) 
    # - or GOOGLE_API_KEY (AI Studio)
    core = CoreGenAI()

    try:
        # 2. Generate Type-Safe Data
        audit = await core.generate_struct(
            prompt="Scan this code snippet for SQL injection...",
            schema=SecurityAudit
        )
        
        # 3. Use it as a real object
        if audit.is_critical:
            print(f"Found {len(audit.vulnerabilities)} issues!")
            
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(main())
```

### 2. Zero-Cost Unit Testing
Run your test suite in CI/CD without an internet connection or an API bill.

```python
from coregenai import GenConfig

# 1. Enable Mock Mode
config = GenConfig(
    mock_mode=True,
    mock_response_payload='{"risk_level": "HIGH", "vulnerabilities": ["SQLi"], "is_critical": true}'
)

core = GenConfig(config=config)

# 2. Run your logic (Costs $0, takes 0.05s)
result = await core.generate_struct("...", schema=SecurityAudit)
assert result.is_critical is True
```


---

## Advanced Configuration

Control resilience and safety via `GenConfig`.

```python
from coregenai import GenConfig

config = GenConfig(
    # Backend: "auto", "vertex", or "studio"
    backend="auto", 
    
    # Resilience: How hard to try before giving up
    max_retries=5,
    request_timeout=60.0,
    
    # Safety: Models defaults
    model_name="gemini-1.5-pro-002",
    temperature=0.5
)
```

## Installation

```bash
pip install coregenai
```



