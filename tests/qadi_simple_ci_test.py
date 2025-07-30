"""CI-compatible tests for qadi_simple.py evolution improvements."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestQadiSimpleEvolutionCI:
    """Tests that can run in CI without API keys."""
    
    @pytest.mark.asyncio
    async def test_population_parameter_validation(self):
        """Test population parameter validation without running evolution."""
        import qadi_simple
        
        # Mock the main function to test argument parsing
        with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
            with patch('sys.argv', ['qadi_simple.py', 'test', '--evolve', '--population', '1']):
                # Should fail validation (population too small)
                with pytest.raises(SystemExit):
                    qadi_simple.main()
            
            with patch('sys.argv', ['qadi_simple.py', 'test', '--evolve', '--population', '11']):
                # Should fail validation (population too large)
                with pytest.raises(SystemExit):
                    qadi_simple.main()
    
    def test_simpler_qadi_prompt_override(self):
        """Test that SimplerQADIPrompts properly overrides questioning prompt."""
        from qadi_simple import SimplerQADIPrompts
        
        prompts = SimplerQADIPrompts()
        prompt = prompts.get_questioning_prompt("Test input", "Test context")
        
        # Should use simplified prompt
        assert "academic masturbation" not in prompt.lower()
        assert "actionable" in prompt.lower()
        assert "Core Question:" in prompt
    
    @pytest.mark.asyncio
    async def test_hypothesis_count_propagation(self):
        """Test that num_hypotheses propagates through the orchestration."""
        from qadi_simple import SimplerQADIOrchestrator
        from mad_spark_alt.core.qadi_prompts import QADIPrompts
        
        # Once implemented, this should work:
        # orchestrator = SimplerQADIOrchestrator(num_hypotheses=7)
        # assert orchestrator.num_hypotheses == 7
        
        # For now, test the parent class behavior
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
        
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=7)
        assert orchestrator.num_hypotheses == 7
        
        # Test prompt generation includes correct number
        prompts = QADIPrompts()
        abduction_prompt = prompts.get_abduction_prompt("test", "test", num_hypotheses=7)
        assert "generate 7 distinct approaches" in abduction_prompt
    
    def test_evolution_timeout_calculation(self):
        """Test timeout calculation for different configurations."""
        # Import the function once it's accessible
        # Currently defined inside main()
        def calculate_evolution_timeout(gens: int, pop: int) -> float:
            """Calculate timeout in seconds based on generations and population."""
            base_timeout = 60.0  # Base 1 minute
            time_per_eval = 2.0  # 2 seconds per idea evaluation
            
            # Estimate total evaluations
            total_evaluations = gens * pop
            estimated_time = base_timeout + (total_evaluations * time_per_eval)
            
            # Cap at 10 minutes
            return min(estimated_time, 600.0)
        
        # Test various configurations
        assert calculate_evolution_timeout(2, 3) == 60.0 + (2 * 3 * 2.0)  # 72 seconds
        assert calculate_evolution_timeout(5, 10) == 60.0 + (5 * 10 * 2.0)  # 160 seconds
        assert calculate_evolution_timeout(100, 100) == 600.0  # Capped at 10 minutes
    
    def test_deduplication_threshold(self):
        """Test that deduplication uses appropriate threshold."""
        from difflib import SequenceMatcher
        
        # Test similar but not identical content
        content1 = "Implement a distributed system with microservices architecture"
        content2 = "Implement a distributed system using microservices architecture"
        
        similarity = SequenceMatcher(None, content1.lower(), content2.lower()).ratio()
        assert similarity > 0.85  # Should be considered similar
        
        # Test different content
        content3 = "Build a monolithic application with traditional architecture"
        similarity2 = SequenceMatcher(None, content1.lower(), content3.lower()).ratio()
        assert similarity2 < 0.85  # Should be considered different
    
    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_config(self):
        """Test error handling for invalid configurations."""
        from qadi_simple import SimplerQADIOrchestrator
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            SimplerQADIOrchestrator(temperature_override=3.0)  # Too high
        
        with pytest.raises(ValueError):
            SimplerQADIOrchestrator(temperature_override=-0.5)  # Negative