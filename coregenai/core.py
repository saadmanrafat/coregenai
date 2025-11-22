import asyncio
import json
import time

from typing import Any, TypeVar

import google.auth
from google import genai
from google.genai import types
from pydantic import BaseModel

from coregenai.interfaces import GenInterceptor
from coregenai.config import GenConfig
from coregenai.retry import with_retry

T = TypeVar("T", bound=BaseModel)


class SafetyViolationError(Exception):
    """Raised when Google Safety Filters block the response."""

    def __init__(self, reason: str, ratings: list):
        self.reason = reason
        details = ", ".join(
            [
                f"{r.category}: {r.probability}"
                for r in ratings
                if r.probability != "NEGLIGIBLE"
            ]
        )
        super().__init__(f"Safety Block [{reason}]: {details}")


class CoreGenAI:
    """
    A core client for interacting with Google's Generative AI models,
    featuring retry logic and interception hooks, structured output generation.
    """

    def __init__(
        self,
        config: GenConfig = GenConfig(),
        interceptors: list[GenInterceptor] | None = None,
    ):
        self.config = config
        self.interceptors = interceptors if interceptors is not None else []
        self.client = self._build_client()

    def _build_client(self) -> genai.Client:
        """Constructs the Google Client based on config."""
        project_id = self.config.google_project_id

        # Auto-detect project ID if using Vertex and not explicitly set
        if self.config.use_vertex and not project_id:
            try:
                _, project_id = google.auth.default()
            except Exception:
                # Fallback or let the SDK handle the error later if strictly needed
                pass

        return genai.Client(
            vertexai=self.config.use_vertex,
            project=project_id if self.config.use_vertex else None,
            location=self.config.google_location if self.config.use_vertex else None,
            api_key=self.config.google_api_key if not self.config.use_vertex else None,
        )

    @with_retry(max_retries=3)
    async def _execute_call(
        self, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Internal executor wrapped with Retry Logic. Handles Mock Mode logic here."""
        # 1. Mock Mode Check
        if self.config.mock_mode:
            await asyncio.sleep(0.1)  # Simulate slight latency
            mock_text = (
                self.config.mock_response_payload
                if self.config.mock_response_payload
                else "[MOCK] Response"
            )
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[types.Part(text=mock_text)])
                    )
                ]
            )

        return await self.client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )

    async def generate(self, prompt: str, model: str | None = None) -> str:
        """Standard Text Generation."""
        target_model = model if model is not None else self.config.model_name

        for hook in self.interceptors:
            await hook.on_start(prompt=prompt, model=target_model)

        start_t = time.time()
        try:
            response = await self._execute_call(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
            text = response.text or ""

            latency = time.time() - start_t
            for hook in self.interceptors:
                await hook.on_success(result=text, latency=latency)

            return text
        except Exception as e:
            for hook in self.interceptors:
                await hook.on_error(error=e)
            raise  # Re-raise exception while preserving the original traceback

    async def generate_struct(
        self, prompt: str, schema: type[T], model: str | None = None
    ) -> T:
        """
        Structured Output Generation. Returns an instance of the Pydantic model 'schema'.
        """
        target_model = model if model is not None else self.config.model_name

        for hook in self.interceptors:
            await hook.on_start(prompt=prompt, model=target_model)

        start_t = time.time()

        try:
            response = await self._execute_call(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,  # Google SDK now accepts Pydantic classes directly
                    temperature=0.1,  # Force low temp for structure
                ),
            )

            # Parsing Logic
            try:
                # 1. Try native parsing (SDK v1 feature)
                if hasattr(response, "parsed") and response.parsed:
                    # Sometimes the SDK returns a dict, sometimes the object
                    # We enforce the object.
                    result = schema.model_validate(response.parsed)
                else:
                    # 2. Fallback to text parsing
                    data = json.loads(response.text)
                    result = schema(**data)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse JSON from AI: {response.text}"
                ) from e

            # Notify Hooks
            latency = time.time() - start_t
            for hook in self.interceptors:
                await hook.on_success(result=result, latency=latency)

            return result

        except Exception as e:
            for hook in self.interceptors:
                await hook.on_error(error=e)
            raise e
