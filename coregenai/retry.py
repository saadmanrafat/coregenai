import asyncio
import logging
import random

from typing import Tuple, Type, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryCallState,
    wait_random,
    retry_if_exception,
)


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("coregenai.retry")
# logger.setLevel(logging.INFO)


def should_retry_exception(exc: BaseException) -> bool:
    """
    Return True if we should retry, False otherwise.

    This predicate checks for a 'code' attribute on the exception,
    which is common for HTTP-related exception classes.
    """
    # Your special logic: Do not retry on 4xx client errors, except for 429.
    if hasattr(exc, "code") and isinstance(exc.code, int):
        is_client_error = 400 <= exc.code < 500
        is_rate_limit = exc.code == 429
        if is_client_error and not is_rate_limit:
            # logger.error(f"Aborting retries for non-retriable client error: {exc}")
            return False
    return True


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Async Decorator factory for Exponential Backoff + Full Jitter using tenacity.
    """
    # Use wait_random_exponential for a robust backoff strategy.
    # It waits a random amount of time between 0 and `2^attempt * multiplier`.
    # This is a best practice known as "full jitter".
    wait = wait_random_exponential(multiplier=initial_delay, max=max_delay)

    # The stop condition is the total number of attempts (initial call + retries).
    stop = stop_after_attempt(max_retries + 1)

    # The retry condition combines the base exception type check with your
    # custom logic for HTTP status codes. This composition is a great feature.
    retry_condition = retry_if_exception_type(retry_on) & retry_if_exception(
        should_retry_exception
    )

    return retry(
        wait=wait,
        stop=stop,
        retry=retry_condition,
        # before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Re-raise the last exception after retries are exhausted.
    )
