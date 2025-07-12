"""
Comprehensive timeout wrapper to prevent any hanging issues.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Custom timeout error with context."""
    
    def __init__(self, message: str, phase: Optional[str] = None, elapsed: Optional[float] = None):
        self.phase = phase
        self.elapsed = elapsed
        super().__init__(message)


async def with_timeout(
    coro: Callable[..., Any],
    timeout: float,
    phase: str = "unknown",
    fallback: Optional[Any] = None,
    log_progress: bool = True
) -> Any:
    """
    Execute a coroutine with timeout and progress logging.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        phase: Name of the phase for logging
        fallback: Value to return on timeout
        log_progress: Whether to log progress
        
    Returns:
        Result of coroutine or fallback value
    """
    start_time = time.time()
    
    if log_progress:
        logger.info(f"Starting {phase} (timeout: {timeout}s)")
    
    try:
        # Create a task for progress logging
        if log_progress:
            progress_task = asyncio.create_task(_log_progress(phase, timeout))
        
        # Execute with timeout
        result = await asyncio.wait_for(coro, timeout=timeout)
        
        # Cancel progress logging
        if log_progress:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        elapsed = time.time() - start_time
        if log_progress:
            logger.info(f"Completed {phase} in {elapsed:.2f}s")
        
        return result
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"Timeout in {phase} after {elapsed:.2f}s")
        
        if fallback is not None:
            logger.info(f"Using fallback for {phase}")
            return fallback
        
        raise TimeoutError(
            f"Operation '{phase}' timed out after {timeout}s",
            phase=phase,
            elapsed=elapsed
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error in {phase} after {elapsed:.2f}s: {e}")
        raise


async def _log_progress(phase: str, timeout: float):
    """Log progress periodically during long operations."""
    interval = min(10, timeout / 5)  # Log every 10s or 5 times total
    elapsed = 0
    
    try:
        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval
            remaining = timeout - elapsed
            logger.info(f"{phase} still running... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)")
    except asyncio.CancelledError:
        pass


def timeout_decorator(timeout: float, phase: Optional[str] = None, fallback: Optional[Any] = None):
    """
    Decorator to add timeout to async functions.
    
    Args:
        timeout: Timeout in seconds
        phase: Phase name (uses function name if not provided)
        fallback: Fallback value on timeout
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            phase_name = phase or func.__name__
            return await with_timeout(
                func(*args, **kwargs),
                timeout=timeout,
                phase=phase_name,
                fallback=fallback
            )
        return wrapper
    return decorator


class TimeoutManager:
    """
    Manage cascading timeouts for multi-phase operations.
    """
    
    def __init__(self, total_timeout: float):
        self.total_timeout = total_timeout
        self.start_time = time.time()
        self.phase_times = {}
        
    def get_remaining_time(self) -> float:
        """Get remaining time in the total timeout."""
        elapsed = time.time() - self.start_time
        return max(0, self.total_timeout - elapsed)
        
    def get_phase_timeout(self, phase: str, default: float = 30.0) -> float:
        """
        Get timeout for a specific phase.
        
        Returns the minimum of:
        - The default timeout for the phase
        - The remaining total timeout
        """
        remaining = self.get_remaining_time()
        if remaining <= 0:
            raise TimeoutError("Total timeout exceeded", phase="total")
        
        return min(default, remaining)
        
    def record_phase_time(self, phase: str, duration: float):
        """Record how long a phase took."""
        self.phase_times[phase] = duration
        
    def get_summary(self) -> dict:
        """Get summary of all phase times."""
        total_elapsed = time.time() - self.start_time
        return {
            "total_elapsed": total_elapsed,
            "total_timeout": self.total_timeout,
            "phase_times": self.phase_times,
            "remaining": self.get_remaining_time()
        }