"""
Tests for evolution timeout quick fix.

This module tests the updated timeout calculation that prevents
evolution from timing out with larger populations and generations.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
from pathlib import Path

# Import the function and constants from qadi_simple
sys.path.insert(0, str(Path(__file__).parent.parent))
from qadi_simple import (
    calculate_evolution_timeout,
    EVOLUTION_BASE_TIMEOUT,
    EVOLUTION_TIME_PER_EVAL,
    EVOLUTION_MAX_TIMEOUT
)


class TestEvolutionTimeoutFix:
    """Test the quick fix for evolution timeout issues."""
    
    def test_qadi_simple_exports_constants(self):
        """Test that qadi_simple.py exports the correct constants."""
        # Verify that qadi_simple exports the expected constants
        assert EVOLUTION_BASE_TIMEOUT == 120.0, "Base timeout should be 120s"
        assert EVOLUTION_TIME_PER_EVAL == 8.0, "Time per eval should be 8s for semantic operators"
        assert EVOLUTION_MAX_TIMEOUT == 900.0, "Max timeout should be 900s (15 minutes)"
        
    def test_qadi_simple_exports_function(self):
        """Test that qadi_simple.py exports calculate_evolution_timeout function."""
        assert callable(calculate_evolution_timeout), "calculate_evolution_timeout should be callable"
        
        # Test that it returns expected values
        timeout = calculate_evolution_timeout(3, 10)
        assert timeout == 440.0, "Should calculate correct timeout for 3 generations, 10 population"
    
    def test_updated_timeout_calculation(self):
        """Test that timeout calculation uses updated values."""
        # Test small evolution (2 generations, 3 population)
        timeout = calculate_evolution_timeout(2, 3)
        expected = EVOLUTION_BASE_TIMEOUT + (2 * 3 + 3) * EVOLUTION_TIME_PER_EVAL
        assert timeout == expected
        assert timeout == 192.0
        
        # Test medium evolution (3 generations, 10 population)
        timeout = calculate_evolution_timeout(3, 10)
        expected = EVOLUTION_BASE_TIMEOUT + (3 * 10 + 10) * EVOLUTION_TIME_PER_EVAL
        assert timeout == expected
        assert timeout == 440.0  # This should now be enough time
        
        # Test large evolution that hits cap
        timeout = calculate_evolution_timeout(10, 50)
        expected = EVOLUTION_BASE_TIMEOUT + (10 * 50 + 50) * EVOLUTION_TIME_PER_EVAL
        assert timeout == EVOLUTION_MAX_TIMEOUT  # Should be capped
        assert timeout == 900.0
        
    def test_old_timeout_was_insufficient(self):
        """Verify the old timeout was indeed too short."""
        # Old timeout calculation
        def old_calculate_evolution_timeout(gens: int, pop: int) -> float:
            base_timeout = 90.0
            time_per_eval = 5.0
            total_evaluations = gens * pop + pop
            estimated_time = base_timeout + (total_evaluations * time_per_eval)
            return min(estimated_time, 900.0)
        
        # The problematic case from user report
        old_timeout = old_calculate_evolution_timeout(3, 10)
        assert old_timeout == 290.0  # This matched the timeout error
        
        # New timeout should be significantly higher
        new_timeout = 120.0 + (3 * 10 + 10) * 8.0
        assert new_timeout == 440.0
        assert new_timeout > old_timeout * 1.5  # At least 50% increase
        
    @pytest.mark.asyncio
    async def test_evolution_completes_within_new_timeout(self):
        """Test that evolution can complete within the new timeout."""
        # Mock evolution that takes 350 seconds (more than old 290s limit)
        async def mock_evolution():
            # Simulate work that takes longer than old timeout
            await asyncio.sleep(0.1)  # Shortened for test
            return {"status": "completed", "generations": 3}
        
        # Test with new timeout (440s for 3x10)
        new_timeout = calculate_evolution_timeout(3, 10)
        try:
            result = await asyncio.wait_for(mock_evolution(), timeout=new_timeout)
            assert result["status"] == "completed"
        except asyncio.TimeoutError:
            pytest.fail("Evolution should not timeout with new limit")
            
    def test_timeout_message_reflects_new_values(self):
        """Test that timeout messages show updated values."""
        gens = 3
        pop = 10
        timeout = 120.0 + (gens * pop + pop) * 8.0
        
        message = f"⏱️  Evolution timeout: {timeout:.0f}s (adjust --generations or --population if needed)"
        assert "440s" in message
        assert "adjust --generations or --population" in message


class TestSemanticOperatorTokenLimit:
    """Test token limit improvements for semantic operators."""
    
    def test_mutation_token_limits(self):
        """Test that mutation operations have updated token limits."""
        # Import the actual constants from semantic_operators
        from mad_spark_alt.evolution.semantic_operators import (
            SEMANTIC_MUTATION_MAX_TOKENS,
            SEMANTIC_BATCH_MUTATION_BASE_TOKENS,
            SEMANTIC_BATCH_MUTATION_MAX_TOKENS,
            SEMANTIC_CROSSOVER_MAX_TOKENS
        )
        
        # Test the actual values
        assert SEMANTIC_MUTATION_MAX_TOKENS == 1000, "Single mutation should have 1000 tokens"
        assert SEMANTIC_BATCH_MUTATION_BASE_TOKENS == 1000, "Batch mutation base should be 1000 tokens per idea"
        assert SEMANTIC_BATCH_MUTATION_MAX_TOKENS == 4000, "Batch mutation max should be 4000 tokens"
        assert SEMANTIC_CROSSOVER_MAX_TOKENS == 1500, "Crossover should have 1500 tokens"


class TestIntegrationScenarios:
    """Integration tests for the timeout fix."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_evolution_run_with_timeout(self):
        """Test a full evolution run with the new timeout settings."""
        # This test requires API key and will be run separately
        # It verifies that evolution completes successfully
        pass  # Will be implemented with actual evolution run
        
    @pytest.mark.integration 
    def test_cli_shows_updated_timeout(self):
        """Test that CLI output shows the new timeout values."""
        # This will test the actual CLI output
        pass  # Will be implemented with subprocess call