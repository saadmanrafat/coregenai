from abc import ABC
from typing import Any


class GenInterceptor(ABC):
    """
    Abstract base class for hooks that intercept generative AI calls.

    This class defines an interface for logging, tracing, or cost tracking.
    The base implementations are no-ops, allowing subclasses to selectively
    override methods for the events they need to handle.
    """

    async def on_start(self, *, prompt: str, model: str) -> None:
        """Asynchronously called before the generative operation begins."""
        pass

    async def on_success(self, *, result: Any, latency: float) -> None:
        """Asynchronously called after the operation completes successfully."""
        pass

    async def on_error(self, *, error: Exception, latency: float) -> None:
        """Asynchronously called if the operation fails with an exception."""
        pass
