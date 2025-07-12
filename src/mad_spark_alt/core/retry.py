"""
Retry mechanisms and error handling for LLM requests.

This module provides robust retry logic with exponential backoff,
circuit breaker patterns, and comprehensive error handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar, Union

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorType(Enum):
    """Types of errors that can occur during LLM requests."""

    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_errors: tuple = (
        ErrorType.RATE_LIMIT,
        ErrorType.NETWORK,
        ErrorType.TIMEOUT,
    )


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after = retry_after


class RateLimitError(LLMError):
    """Rate limit exceeded error."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message, ErrorType.RATE_LIMIT, 429, retry_after)


class QuotaExceededError(LLMError):
    """API quota exceeded error."""

    def __init__(self, message: str):
        super().__init__(message, ErrorType.QUOTA_EXCEEDED, 429)


class AuthenticationError(LLMError):
    """Authentication error."""

    def __init__(self, message: str):
        super().__init__(message, ErrorType.AUTHENTICATION, 401)


class NetworkError(LLMError):
    """Network-related error."""

    def __init__(self, message: str):
        super().__init__(message, ErrorType.NETWORK)


class LLMTimeoutError(LLMError):
    """Request timeout error."""

    def __init__(self, message: str):
        super().__init__(message, ErrorType.TIMEOUT)


class APIError(LLMError):
    """General API error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message, ErrorType.API_ERROR, status_code)


def classify_error(exception: Exception) -> ErrorType:
    """Classify an exception into an error type."""
    if isinstance(exception, aiohttp.ClientResponseError):
        if exception.status == 429:
            return ErrorType.RATE_LIMIT
        elif exception.status == 401:
            return ErrorType.AUTHENTICATION
        elif exception.status >= 500:
            return ErrorType.API_ERROR
        else:
            return ErrorType.API_ERROR
    elif isinstance(exception, aiohttp.ClientTimeout):
        return ErrorType.TIMEOUT
    elif isinstance(exception, aiohttp.ClientError):
        return ErrorType.NETWORK
    elif isinstance(exception, LLMError):
        return exception.error_type
    else:
        return ErrorType.UNKNOWN


def should_retry(error_type: ErrorType, retry_config: RetryConfig) -> bool:
    """Determine if an error should trigger a retry."""
    return error_type in retry_config.retry_on_errors


async def calculate_delay(
    attempt: int, config: RetryConfig, retry_after: Optional[float] = None
) -> float:
    """Calculate delay before next retry attempt."""
    if retry_after:
        # Respect server's retry-after header
        return retry_after

    # Exponential backoff with jitter
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter (±25%)
        import random

        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    retry_config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for the function
        retry_config: Retry configuration
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        LLMError: If all retry attempts fail
    """
    if retry_config is None:
        retry_config = RetryConfig()

    last_exception = None

    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e
            error_type = classify_error(e)

            logger.warning(
                f"Attempt {attempt}/{retry_config.max_attempts} failed: "
                f"{error_type.value} - {str(e)}"
            )

            # Don't retry on final attempt or non-retryable errors
            if attempt == retry_config.max_attempts or not should_retry(
                error_type, retry_config
            ):
                break

            # Calculate delay and wait
            retry_after = getattr(e, "retry_after", None)
            delay = await calculate_delay(attempt, retry_config, retry_after)

            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

    # Convert final exception to LLMError if needed
    if isinstance(last_exception, LLMError):
        raise last_exception
    else:
        if last_exception is not None:
            error_type = classify_error(last_exception)
            raise LLMError(
                f"Request failed after {retry_config.max_attempts} attempts: {str(last_exception)}",
                error_type,
            ) from last_exception
        else:
            raise LLMError(
                f"Request failed after {retry_config.max_attempts} attempts",
                ErrorType.UNKNOWN,
            )


class CircuitBreaker:
    """Circuit breaker pattern for LLM requests."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker."""
        import time

        now = time.time()

        if self.state == "OPEN":
            if (
                self.last_failure_time
                and (now - self.last_failure_time) >= self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise LLMError("Circuit breaker is OPEN", ErrorType.API_ERROR)

        try:
            result = await func(*args, **kwargs)

            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = now

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(
                    f"Circuit breaker OPENED after {self.failure_count} failures"
                )

            raise


def handle_aiohttp_errors(
    response: aiohttp.ClientResponse, response_data: dict
) -> None:
    """Handle aiohttp response errors and convert to appropriate LLMError."""
    if response.status == 401:
        raise AuthenticationError("Invalid API key or authentication failed")

    elif response.status == 429:
        retry_after = None
        if "retry-after" in response.headers:
            try:
                retry_after = float(response.headers["retry-after"])
            except ValueError:
                pass

        error_message = response_data.get("error", {}).get(
            "message", "Rate limit exceeded"
        )

        if "quota" in error_message.lower():
            raise QuotaExceededError(error_message)
        else:
            raise RateLimitError(error_message, retry_after)

    elif response.status >= 500:
        error_message = response_data.get("error", {}).get(
            "message", f"Server error: {response.status}"
        )
        raise APIError(error_message, response.status)

    elif response.status >= 400:
        error_message = response_data.get("error", {}).get(
            "message", f"Client error: {response.status}"
        )
        raise APIError(error_message, response.status)


async def safe_aiohttp_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs: Any,
) -> dict:
    """
    Make a safe aiohttp request with retry logic and error handling.

    Args:
        session: aiohttp session
        method: HTTP method
        url: Request URL
        retry_config: Retry configuration
        circuit_breaker: Circuit breaker instance
        **kwargs: Additional request arguments

    Returns:
        Response JSON data

    Raises:
        LLMError: On request failure
    """

    async def _make_request() -> Any:
        try:
            timeout = aiohttp.ClientTimeout(
                total=kwargs.pop("timeout", 300)
            )  # Default 5 minutes

            async with session.request(
                method, url, timeout=timeout, **kwargs
            ) as response:
                try:
                    response_data = await response.json()
                except Exception:
                    response_data = {"error": {"message": await response.text()}}

                if response.status >= 400:
                    handle_aiohttp_errors(response, response_data)

                return response_data

        except asyncio.TimeoutError as e:
            raise LLMTimeoutError("Request timed out") from e
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {str(e)}") from e
        except Exception as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError("Request timed out") from e
            raise NetworkError(f"Network error: {str(e)}") from e

    # Apply circuit breaker if provided
    if circuit_breaker:

        async def request_func() -> Any:
            return await circuit_breaker.call(_make_request)

    else:
        request_func = _make_request

    # Apply retry logic
    result = await retry_async(request_func, retry_config=retry_config)
    return result  # type: ignore
