"""
Enhanced error recovery for evolution system.

This module provides retry logic with exponential backoff and
intelligent error handling for evolution operations.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for retryable errors."""

    pass


class NetworkError(RetryableError):
    """Network-related errors that should be retried."""

    pass


class RateLimitError(RetryableError):
    """Rate limit errors that should be retried with backoff."""

    pass


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except RetryableError as e:
            last_exception = e

            if attempt < max_retries:
                # Add jitter if enabled
                actual_delay = delay
                if jitter:
                    import random

                    actual_delay = delay * (0.5 + random.random())

                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {actual_delay:.1f}s..."
                )

                await asyncio.sleep(actual_delay)

                # Exponential backoff
                delay = min(delay * exponential_base, max_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")

        except Exception as e:
            # Non-retryable error
            logger.error(f"Non-retryable error: {str(e)}")
            raise

    # Raise the last exception if all retries failed
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed with no exception")


class RetryableEvaluator:
    """
    Fitness evaluator with automatic retry and error recovery.

    This evaluator wraps a base evaluator and adds retry logic
    for handling transient failures.
    """

    def __init__(
        self,
        base_evaluator: Optional[FitnessEvaluator] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize retryable evaluator.

        Args:
            base_evaluator: Base evaluator to wrap
            max_retries: Maximum retry attempts
            initial_delay: Initial retry delay in seconds
            max_delay: Maximum retry delay in seconds
        """
        self.base_evaluator = base_evaluator or FitnessEvaluator()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._failure_counts: Dict[str, int] = {}

    async def evaluate(
        self,
        idea: GeneratedIdea,
        config: Optional[EvolutionConfig] = None,
        context: Optional[str] = None,
    ) -> IndividualFitness:
        """
        Evaluate an idea with automatic retry on failure.

        Args:
            idea: Idea to evaluate
            config: Evolution configuration
            context: Optional evaluation context

        Returns:
            Fitness evaluation result
        """

        async def _evaluate() -> IndividualFitness:
            try:
                if hasattr(self.base_evaluator, "evaluate_individual"):
                    return await self.base_evaluator.evaluate_individual(
                        idea, config or EvolutionConfig(), context
                    )
                else:
                    # Fallback for simple evaluators
                    return await self.base_evaluator.evaluate(idea)

            except asyncio.TimeoutError as e:
                raise NetworkError(f"Evaluation timed out: {str(e)}")
            except Exception as e:
                # Check if this is a retryable error
                error_msg = str(e).lower()
                if any(
                    term in error_msg for term in ["network", "connection", "timeout"]
                ):
                    raise NetworkError(f"Network error: {str(e)}")
                elif any(
                    term in error_msg for term in ["rate limit", "quota", "too many"]
                ):
                    raise RateLimitError(f"Rate limit error: {str(e)}")
                else:
                    # Non-retryable error
                    raise

        try:
            result = await retry_with_backoff(
                _evaluate,
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
            )

            # Reset failure count on success
            idea_key = self._get_idea_key(idea)
            self._failure_counts[idea_key] = 0

            return result

        except Exception as e:
            # Track failures
            idea_key = self._get_idea_key(idea)
            self._failure_counts[idea_key] = self._failure_counts.get(idea_key, 0) + 1

            # If too many failures for this idea, return low fitness instead of crashing
            if self._failure_counts[idea_key] >= self.max_retries:
                logger.error(
                    f"Idea repeatedly failed evaluation after {self.max_retries} attempts. "
                    f"Returning minimal fitness."
                )
                return IndividualFitness(
                    idea=idea,
                    creativity_score=0.1,
                    diversity_score=0.1,
                    quality_score=0.1,
                    overall_fitness=0.1,
                    evaluation_metadata={
                        "error": str(e),
                        "failure_count": self._failure_counts[idea_key],
                    },
                )
            raise

    def _get_idea_key(self, idea: GeneratedIdea) -> str:
        """Generate a key for tracking idea failures."""
        return f"{idea.thinking_method.value}_{hash(idea.content)}"

    async def evaluate_population(
        self,
        population: List[GeneratedIdea],
        config: Optional[EvolutionConfig] = None,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """
        Evaluate a population with error recovery.

        Args:
            population: List of ideas to evaluate
            config: Evolution configuration
            context: Optional evaluation context

        Returns:
            List of fitness evaluations
        """
        results = []

        for idea in population:
            try:
                fitness = await self.evaluate(idea, config, context)
                results.append(fitness)
            except Exception as e:
                logger.error(f"Failed to evaluate idea: {str(e)}")
                # Add minimal fitness for failed evaluation
                results.append(
                    IndividualFitness(
                        idea=idea,
                        creativity_score=0.1,
                        diversity_score=0.1,
                        quality_score=0.1,
                        overall_fitness=0.1,
                        evaluation_metadata={"error": str(e)},
                    )
                )

        return results

    def get_failure_statistics(self) -> dict:
        """
        Get statistics about evaluation failures.

        Returns:
            Dictionary with failure statistics
        """
        total_failures = sum(self._failure_counts.values())
        ideas_with_failures = len([c for c in self._failure_counts.values() if c > 0])

        return {
            "total_failures": total_failures,
            "ideas_with_failures": ideas_with_failures,
            "max_failures_per_idea": (
                max(self._failure_counts.values()) if self._failure_counts else 0
            ),
            "failure_distribution": dict(self._failure_counts),
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    Opens after too many failures, preventing further attempts
    until a cooldown period has passed.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_attempts: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_attempts: Attempts in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_attempts_remaining = 0
        self._state = "closed"  # closed, open, half_open

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        import time

        # Check circuit state
        if self._state == "open":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half_open"
                self._half_open_attempts_remaining = self.half_open_attempts
            else:
                raise RuntimeError(
                    f"Circuit breaker is open. Recovery in "
                    f"{self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
                )

        try:
            result = await func(*args, **kwargs)

            # Success - update state
            if self._state == "half_open":
                self._half_open_attempts_remaining -= 1
                if self._half_open_attempts_remaining <= 0:
                    self._state = "closed"
                    self._failure_count = 0

            return result

        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if (
                self._state == "half_open"
                or self._failure_count >= self.failure_threshold
            ):
                self._state = "open"

            raise
