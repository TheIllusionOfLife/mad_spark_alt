"""Tests for qadi_simple.py evolution improvements."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _create_response(num_hypotheses: int):
        hypotheses = []
        for i in range(1, num_hypotheses + 1):
            hypotheses.append(f"H{i}: Test hypothesis {i}\nThis is a detailed explanation of hypothesis {i} with sufficient content to pass validation.")
        
        return {
            "question": "Q: What is the core question?",
            "hypotheses": "\n\n".join(hypotheses),
            "answer": "Based on analysis, H2 is the best approach.",
            "synthesis": "1. Start with approach 2\n2. Consider elements from approach 1\n3. Monitor and adjust"
        }
    return _create_response


class TestSimplqerQADIOrchestrator:
    """Test SimplerQADIOrchestrator with num_hypotheses parameter."""
    
    def test_init_with_default_num_hypotheses(self):
        """Test initialization without num_hypotheses uses default."""
        from qadi_simple import SimplerQADIOrchestrator
        
        orchestrator = SimplerQADIOrchestrator()
        # Should use parent's default of 3
        assert orchestrator.num_hypotheses == 3
    
    def test_init_with_custom_num_hypotheses(self):
        """Test initialization with custom num_hypotheses."""
        from qadi_simple import SimplerQADIOrchestrator
        
        # Test various values
        orchestrator = SimplerQADIOrchestrator(num_hypotheses=5)
        assert orchestrator.num_hypotheses == 5
        
        orchestrator = SimplerQADIOrchestrator(num_hypotheses=10)
        assert orchestrator.num_hypotheses == 10
        
        # Test minimum enforcement (parent class enforces min 3)
        orchestrator = SimplerQADIOrchestrator(num_hypotheses=2)
        assert orchestrator.num_hypotheses == 3  # Should be forced to minimum
    
    def test_init_with_temperature_and_num_hypotheses(self):
        """Test initialization with both temperature and num_hypotheses."""
        from qadi_simple import SimplerQADIOrchestrator
        
        orchestrator = SimplerQADIOrchestrator(temperature_override=1.5, num_hypotheses=7)
        assert orchestrator.temperature_override == 1.5
        assert orchestrator.num_hypotheses == 7
    
    @pytest.mark.asyncio
    async def test_run_qadi_cycle_generates_correct_number_of_hypotheses(self, mock_llm_response):
        """Test that QADI cycle generates the requested number of hypotheses."""
        from qadi_simple import SimplerQADIOrchestrator
        
        # Mock the LLM provider
        with patch('mad_spark_alt.core.llm_provider.llm_manager') as mock_manager:
            # Test with 5 hypotheses
            num_hypotheses = 5
            orchestrator = SimplerQADIOrchestrator(num_hypotheses=num_hypotheses)
            
            # Create mock response
            mock_response = AsyncMock()
            responses = mock_llm_response(num_hypotheses)
            
            # Set up responses for each phase
            mock_response.generate.side_effect = [
                MagicMock(response=responses["question"], cost=0.001),  # Question phase
                MagicMock(response=responses["hypotheses"], cost=0.002),  # Abduction phase
                MagicMock(response=responses["answer"], cost=0.003),  # Deduction phase
                MagicMock(response=responses["synthesis"], cost=0.001),  # Induction phase
            ]
            mock_manager.create_request.return_value = mock_response
            
            # Run the cycle
            result = await orchestrator.run_qadi_cycle(
                user_input="Test question about AI",
                context="Test context"
            )
            
            # Verify the correct number of ideas were generated
            assert len(result.synthesized_ideas) == num_hypotheses
            
            # Verify abduction prompt was called with correct num_hypotheses
            abduction_call = mock_response.generate.call_args_list[1]
            assert f"generate {num_hypotheses} distinct approaches" in abduction_call[1]['prompt']
    
    @pytest.mark.asyncio
    async def test_evolution_with_different_populations(self, mock_llm_response):
        """Test evolution works correctly with different population sizes."""
        from qadi_simple import SimplerQADIOrchestrator
        
        test_populations = [2, 5, 8, 10]
        
        for population in test_populations:
            with patch('mad_spark_alt.core.llm_provider.llm_manager') as mock_manager:
                orchestrator = SimplerQADIOrchestrator(num_hypotheses=population)
                
                # Create mock response
                mock_response = AsyncMock()
                responses = mock_llm_response(population)
                
                # Set up responses
                mock_response.generate.side_effect = [
                    MagicMock(response=responses["question"], cost=0.001),
                    MagicMock(response=responses["hypotheses"], cost=0.002),
                    MagicMock(response=responses["answer"], cost=0.003),
                    MagicMock(response=responses["synthesis"], cost=0.001),
                ]
                mock_manager.create_request.return_value = mock_response
                
                # Run the cycle
                result = await orchestrator.run_qadi_cycle(
                    user_input=f"Test with population {population}",
                    context="Test context"
                )
                
                # For populations < 3, should still get 3 (minimum)
                expected_ideas = max(3, population)
                assert len(result.synthesized_ideas) == expected_ideas


class TestEvolutionIntegration:
    """Integration tests for evolution with various configurations."""
    
    @pytest.mark.asyncio
    async def test_evolution_uses_population_for_hypothesis_generation(self):
        """Test that evolution parameter correctly sets num_hypotheses."""
        # This will be tested with the actual implementation
        pass
    
    @pytest.mark.asyncio
    async def test_small_population_evolution(self):
        """Test evolution with small population (2-3)."""
        # This will be tested with the actual implementation
        pass
    
    @pytest.mark.asyncio 
    async def test_large_population_evolution(self):
        """Test evolution with large population (8-10)."""
        # This will be tested with the actual implementation
        pass
    
    @pytest.mark.asyncio
    async def test_population_message_accuracy(self):
        """Test that population messages are accurate."""
        # This will be tested with the actual implementation
        pass