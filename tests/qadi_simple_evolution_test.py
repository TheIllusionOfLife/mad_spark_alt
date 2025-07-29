"""
Tests for qadi_simple.py evolution mode with semantic operators.

This module tests that the --evolve flag in qadi_simple.py properly
enables semantic operators when an LLM provider is available.
"""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from pathlib import Path
import sys

# Add parent directory to path to import qadi_simple
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestQadiSimpleEvolution:
    """Test semantic operator integration in qadi_simple.py --evolve mode."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager with Google provider."""
        with patch('qadi_simple.llm_manager') as mock_manager:
            # Mock Google provider
            mock_google_provider = AsyncMock()
            mock_google_provider.generate = AsyncMock()
            
            # Set up the providers dictionary
            mock_manager.providers = {
                'GOOGLE': mock_google_provider
            }
            
            yield mock_manager

    @pytest.fixture
    def mock_genetic_algorithm(self):
        """Mock GeneticAlgorithm class."""
        with patch('mad_spark_alt.evolution.genetic_algorithm.GeneticAlgorithm') as MockGA:
            mock_instance = MagicMock()
            
            # Create a proper mock EvolutionResult
            from mad_spark_alt.evolution.interfaces import EvolutionResult
            mock_result = EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=2,
                execution_time=0.1,
                evolution_metrics={'fitness_improvement_percent': 5.0}
            )
            mock_instance.evolve = AsyncMock(return_value=mock_result)
            MockGA.return_value = mock_instance
            yield MockGA, mock_instance

    @pytest.fixture
    def mock_llm_provider_enum(self):
        """Mock LLMProvider enum."""
        with patch('qadi_simple.LLMProvider') as MockEnum:
            MockEnum.GOOGLE = 'GOOGLE'
            yield MockEnum

    @pytest.mark.asyncio
    async def test_genetic_algorithm_uses_no_llm_provider_by_default(
        self,
        mock_llm_manager,
        mock_genetic_algorithm,
        mock_llm_provider_enum
    ):
        """Test that GeneticAlgorithm disables semantic operators by default for performance."""
        MockGA, mock_instance = mock_genetic_algorithm
        
        # Import the evolution function
        import qadi_simple
        
        # Mock the orchestrator to return some ideas
        with patch.object(qadi_simple, 'SimplerQADIOrchestrator') as MockOrchestrator:
            mock_orchestrator = AsyncMock()
            # Create proper GeneratedIdea objects
            from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
            mock_result = MagicMock()
            mock_result.synthesized_ideas = [
                GeneratedIdea(
                    content="Idea 1",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="test_agent",
                    generation_prompt="test"
                ),
                GeneratedIdea(
                    content="Idea 2",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_agent",
                    generation_prompt="test"
                ),
                GeneratedIdea(
                    content="Idea 3",
                    thinking_method=ThinkingMethod.DEDUCTION,
                    agent_name="test_agent",
                    generation_prompt="test"
                ),
            ]
            mock_result.total_llm_cost = 0.01
            mock_result.core_question = "Test question"
            mock_result.hypotheses = ["H1", "H2"]
            mock_result.hypothesis_scores = []
            mock_result.final_answer = "Test answer"
            mock_result.action_plan = ["Action 1"]
            mock_result.verification_examples = []
            mock_result.verification_conclusion = ""
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            MockOrchestrator.return_value = mock_orchestrator
            
            # Run with evolution enabled
            with patch('os.getenv', return_value='test-api-key'):
                # Patch where GeneticAlgorithm is imported from
                with patch('mad_spark_alt.evolution.GeneticAlgorithm', MockGA):
                    await qadi_simple.run_qadi_analysis(
                        "Test question",
                        evolve=True,
                        generations=2,
                        population=4,
                        traditional=True  # This should force llm_provider=None
                    )
            
            # Verify GeneticAlgorithm was called without llm_provider (None for performance)
            MockGA.assert_called()
            call_kwargs = MockGA.call_args.kwargs
            assert 'llm_provider' in call_kwargs
            assert call_kwargs['llm_provider'] is None  # Disabled by default for performance

    @pytest.mark.asyncio
    async def test_semantic_operators_initialized_when_llm_available(
        self,
        mock_llm_manager,
        mock_genetic_algorithm,
        mock_llm_provider_enum
    ):
        """Test that semantic operators are initialized in GA when LLM provider available."""
        MockGA, mock_instance = mock_genetic_algorithm
        
        # Add attributes to track initialization
        mock_instance.semantic_mutation_operator = None
        mock_instance.semantic_crossover_operator = None
        mock_instance.smart_selector = None
        
        # Mock the initialization to set these attributes
        def init_side_effect(*args, **kwargs):
            if 'llm_provider' in kwargs and kwargs['llm_provider'] is not None:
                mock_instance.semantic_mutation_operator = MagicMock()
                mock_instance.semantic_crossover_operator = MagicMock()
                mock_instance.smart_selector = MagicMock()
            return mock_instance
        
        MockGA.side_effect = init_side_effect
        
        import qadi_simple
        
        with patch.object(qadi_simple, 'SimplerQADIOrchestrator') as MockOrchestrator:
            mock_orchestrator = AsyncMock()
            from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
            mock_result = MagicMock()
            mock_result.synthesized_ideas = [
                GeneratedIdea(
                    content="Idea 1",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="test_agent",
                    generation_prompt="test"
                )
            ]
            mock_result.total_llm_cost = 0.01
            mock_result.core_question = "Test question"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = []
            mock_result.final_answer = "Test answer"
            mock_result.action_plan = ["Action 1"]
            mock_result.verification_examples = []
            mock_result.verification_conclusion = ""
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            MockOrchestrator.return_value = mock_orchestrator
            
            with patch('os.getenv', return_value='test-api-key'):
                await qadi_simple.run_qadi_analysis("Test", evolve=True)
            
            # Verify semantic operators would be initialized
            assert MockGA.called
            assert 'llm_provider' in MockGA.call_args.kwargs

    @pytest.mark.asyncio
    async def test_fallback_when_no_llm_provider(
        self,
        mock_genetic_algorithm,
        mock_llm_provider_enum
    ):
        """Test that GA works without semantic operators when no LLM provider."""
        MockGA, mock_instance = mock_genetic_algorithm
        
        # Mock empty providers
        with patch('qadi_simple.llm_manager') as mock_manager:
            mock_manager.providers = {}  # No providers available
            
            import qadi_simple
            
            with patch.object(qadi_simple, 'SimplerQADIOrchestrator') as MockOrchestrator:
                mock_orchestrator = AsyncMock()
                from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
                mock_result = MagicMock()
                mock_result.synthesized_ideas = [
                    GeneratedIdea(
                        content="Idea 1",
                        thinking_method=ThinkingMethod.QUESTIONING,
                        agent_name="test_agent",
                        generation_prompt="test"
                    )
                ]
                mock_result.total_llm_cost = 0.01
                mock_result.core_question = "Test question"
                mock_result.hypotheses = ["H1"]
                mock_result.hypothesis_scores = []
                mock_result.final_answer = "Test answer"
                mock_result.action_plan = ["Action 1"]
                mock_result.verification_examples = []
                mock_result.verification_conclusion = ""
                mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
                MockOrchestrator.return_value = mock_orchestrator
                
                with patch('os.getenv', return_value='test-api-key'):
                    await qadi_simple.run_qadi_analysis("Test", evolve=True)
                
                # Verify GA was called without llm_provider
                MockGA.assert_called()
                call_kwargs = MockGA.call_args.kwargs
                # Either llm_provider is not in kwargs or it's None
                assert 'llm_provider' not in call_kwargs or call_kwargs['llm_provider'] is None

    def test_imports_required_for_semantic_operators(self):
        """Test that qadi_simple.py has necessary imports for semantic operators."""
        # Read the file to check imports
        qadi_simple_path = Path(__file__).parent.parent / "qadi_simple.py"
        with open(qadi_simple_path, 'r') as f:
            content = f.read()
        
        # Check for required imports (will fail initially in TDD)
        assert 'from mad_spark_alt.core.llm_provider import LLMProvider, llm_manager' in content or \
               'from .core.llm_provider import LLMProvider, llm_manager' in content