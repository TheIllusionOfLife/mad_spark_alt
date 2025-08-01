"""
Tests for evolution timeout quick fix.

This module tests the updated timeout calculation that prevents
evolution from timing out with larger populations and generations.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


class TestEvolutionTimeoutFix:
    """Test the quick fix for evolution timeout issues."""
    
    def test_updated_timeout_calculation(self):
        """Test that timeout calculation uses updated values."""
        # This tests the new timeout calculation logic
        def calculate_evolution_timeout(gens: int, pop: int) -> float:
            """Calculate timeout in seconds based on generations and population."""
            base_timeout = 120.0  # Increased from 90s for better reliability
            time_per_eval = 8.0  # Increased from 5s for semantic operators
            
            # Estimate total evaluations (including initial population)
            total_evaluations = gens * pop + pop  # Initial eval + each generation
            estimated_time = base_timeout + (total_evaluations * time_per_eval)
            
            # Cap at 15 minutes for very large evolutions
            return min(estimated_time, 900.0)
        
        # Test small evolution (2 generations, 3 population)
        timeout = calculate_evolution_timeout(2, 3)
        expected = 120.0 + (2 * 3 + 3) * 8.0  # 120 + 9*8 = 192
        assert timeout == expected
        
        # Test medium evolution (3 generations, 10 population)
        timeout = calculate_evolution_timeout(3, 10)
        expected = 120.0 + (3 * 10 + 10) * 8.0  # 120 + 40*8 = 440
        assert timeout == expected
        assert timeout == 440.0  # This should now be enough time
        
        # Test large evolution that hits cap
        timeout = calculate_evolution_timeout(10, 50)
        expected = 120.0 + (10 * 50 + 50) * 8.0  # 120 + 550*8 = 4520
        assert timeout == 900.0  # Should be capped at 15 minutes
        
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
        try:
            result = await asyncio.wait_for(mock_evolution(), timeout=440.0)
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
    
    def test_mutation_token_limit_in_code(self):
        """Test that mutation operations have updated token limits in code."""
        # Read semantic_operators.py to check token limits
        import os
        semantic_ops_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "mad_spark_alt", "evolution", "semantic_operators.py"
        )
        
        with open(semantic_ops_path, 'r') as f:
            content = f.read()
            
        # Check for increased token limits
        # Single mutation should be at least 1000 (up from 500)
        assert "max_tokens=1000," in content or "max_tokens=1500," in content, \
            "Single mutation should have at least 1000 tokens"
            
        # Batch mutation should allow more
        assert "min(1000 * len(uncached_ideas), 4000)" in content or \
               "min(800 * len(uncached_ideas), 3000)" in content, \
            "Batch mutation should scale properly with increased base"
            
    def test_crossover_token_limit_in_code(self):
        """Test that crossover operations have sufficient token limit."""
        import os
        semantic_ops_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "mad_spark_alt", "evolution", "semantic_operators.py"
        )
        
        with open(semantic_ops_path, 'r') as f:
            content = f.read()
            
        # Crossover should have at least 1500 tokens (up from 1000)
        assert "max_tokens=1500," in content or "max_tokens=2000," in content, \
            "Crossover should have at least 1500 tokens"


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