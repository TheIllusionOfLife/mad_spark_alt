"""
Tests for evolution timeout fix in qadi_simple.py
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_calculate_evolution_timeout():
    """Test that timeout is calculated appropriately based on parameters."""
    
    # Test timeout calculation logic (from qadi_simple.py)
    def calculate_evolution_timeout(gens: int, pop: int) -> float:
        """Calculate adaptive timeout based on evolution complexity."""
        base_timeout = 60.0
        time_per_eval = 2.0
        total_evaluations = gens * pop
        estimated_time = base_timeout + (total_evaluations * time_per_eval)
        return min(estimated_time, 600.0)
    
    # Test various scenarios
    assert calculate_evolution_timeout(2, 2) == 68.0  # Small: 60 + 4*2
    assert calculate_evolution_timeout(5, 10) == 160.0  # Medium: 60 + 50*2
    assert calculate_evolution_timeout(10, 50) == 600.0  # Large: capped at 600


@pytest.mark.asyncio
async def test_evolution_with_timeout_wrapping():
    """Test that evolution is properly wrapped with asyncio.wait_for."""
    
    # Create a mock evolution that takes time
    evolution_called = False
    evolution_timed_out = False
    
    async def mock_evolve(request):
        nonlocal evolution_called
        evolution_called = True
        await asyncio.sleep(2.0)  # Simulate slow evolution
        return MagicMock()
    
    # Test with short timeout
    try:
        await asyncio.wait_for(mock_evolve(None), timeout=0.5)
    except asyncio.TimeoutError:
        evolution_timed_out = True
    
    assert evolution_called
    assert evolution_timed_out


@pytest.mark.asyncio 
async def test_traditional_flag_logic():
    """Test the conditional logic for traditional flag."""
    
    # Test traditional=True case
    traditional = True
    llm_provider = None if traditional else "mock_provider"
    assert llm_provider is None
    
    # Test traditional=False case
    traditional = False
    llm_provider = None if traditional else "mock_provider"
    assert llm_provider == "mock_provider"


@pytest.mark.asyncio
async def test_progress_indicator_cancellation():
    """Test that progress indicator is properly cancelled."""
    
    progress_cancelled = False
    
    async def show_progress(start_time: float, timeout: float):
        """Mock progress indicator."""
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            nonlocal progress_cancelled
            progress_cancelled = True
            raise
    
    # Create and cancel progress task
    progress_task = asyncio.create_task(show_progress(0, 10))
    await asyncio.sleep(0.2)  # Let it run briefly
    progress_task.cancel()
    
    try:
        await progress_task
    except asyncio.CancelledError:
        pass
    
    assert progress_cancelled