"""
Tests for semantic operator usage verification and tracking.

This module tests that semantic operator usage is properly tracked
and displayed to users.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig, IndividualFitness
from mad_spark_alt.core.interfaces import GeneratedIdea


class TestSemanticOperatorVerification:
    """Test semantic operator usage tracking and verification."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider
        
        provider = AsyncMock()
        mock_response = LLMResponse(
            content="Modified content", 
            provider=LLMProvider.GOOGLE,
            model="gemini-pro",
            cost=0.001
        )
        provider.generate = AsyncMock(return_value=mock_response)
        return provider

    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing."""
        return [
            GeneratedIdea(
                content="Idea 1 content",
                thinking_method="method1",
                agent_name="agent1",
                generation_prompt="prompt1"
            ),
            GeneratedIdea(
                content="Idea 2 content",
                thinking_method="method2",
                agent_name="agent2",
                generation_prompt="prompt2"
            ),
            GeneratedIdea(
                content="Idea 3 content",
                thinking_method="method3",
                agent_name="agent3",
                generation_prompt="prompt3"
            ),
        ]

    @pytest.mark.asyncio
    async def test_semantic_operator_usage_tracking(self, mock_llm_provider, sample_population):
        """Test that semantic operator usage is tracked in metrics."""
        # Create GA with semantic operators
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Mock the fitness evaluator
        with patch.object(ga.fitness_evaluator, 'evaluate_population', new_callable=AsyncMock) as mock_eval:
            with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', new_callable=AsyncMock) as mock_diversity:
                # Return list of IndividualFitness objects (awaitable)
                mock_eval.return_value = [
                    IndividualFitness(idea=idea, overall_fitness=0.7) 
                    for idea in sample_population
                ]
                mock_diversity.return_value = 0.5  # Low diversity to trigger semantic operators
                
                # Configure evolution
                config = EvolutionConfig(
                    population_size=3,
                    generations=2,
                    use_semantic_operators=True
                )
                
                request = EvolutionRequest(
                    initial_population=sample_population,
                    config=config,
                    context="Test context"
                )
                
                # With simplified logic, semantic operators are used when available and enabled
                # No need to mock SmartOperatorSelector since it was removed
                
                result = await ga.evolve(request)
                
                # Check metrics include semantic operator usage
                metrics = result.evolution_metrics
                assert 'semantic_mutations' in metrics
                assert 'semantic_crossovers' in metrics
                assert 'traditional_mutations' in metrics
                assert 'traditional_crossovers' in metrics
                
                # Should have used semantic operators
                assert metrics['semantic_mutations'] > 0 or metrics['semantic_crossovers'] > 0

    @pytest.mark.asyncio
    async def test_semantic_operator_status_in_logs(self, mock_llm_provider, caplog):
        """Test that semantic operator status is logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Create GA with semantic operators
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Check initialization logs
        assert any("Semantic operators initialized" in record.message 
                  for record in caplog.records)

    @pytest.mark.asyncio
    async def test_evolution_summary_shows_semantic_status(self, mock_llm_provider, sample_population):
        """Test that evolution summary indicates semantic operator usage."""
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        with patch.object(ga.fitness_evaluator, 'evaluate_population', new_callable=AsyncMock) as mock_eval:
            with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', new_callable=AsyncMock) as mock_diversity:
                # Return list of IndividualFitness objects (awaitable)
                mock_eval.return_value = [
                    IndividualFitness(idea=idea, overall_fitness=0.7) 
                    for idea in sample_population
                ]
                mock_diversity.return_value = 0.5  # Low diversity to trigger semantic operators
                
                config = EvolutionConfig(
                    population_size=3,
                    generations=2,
                    use_semantic_operators=True
                )
                
                request = EvolutionRequest(
                    initial_population=sample_population,
                    config=config,
                    context="Test"
                )
                
                result = await ga.evolve(request)
                
                # Check summary includes semantic operator info
                assert 'semantic_operators_enabled' in result.evolution_metrics
                assert result.evolution_metrics['semantic_operators_enabled'] is True

    def test_semantic_operator_disabled_when_no_llm(self, sample_population):
        """Test that semantic operators are disabled without LLM provider."""
        ga = GeneticAlgorithm()  # No LLM provider
        
        # Check that semantic operators are None
        assert ga.semantic_mutation_operator is None
        assert ga.semantic_crossover_operator is None
        # smart_selector removed - no longer exists

    @pytest.mark.asyncio
    async def test_cli_displays_semantic_operator_status(self):
        """Test that CLI shows whether semantic operators are active."""
        from mad_spark_alt.cli import _get_semantic_operator_status
        
        # With LLM provider
        with patch('mad_spark_alt.cli.llm_manager') as mock_manager:
            from mad_spark_alt.core.llm_provider import LLMProvider
            mock_manager.providers = {LLMProvider.GOOGLE: MagicMock()}
            status = _get_semantic_operator_status()
            assert "ENABLED" in status
            
        # Without LLM provider
        with patch('mad_spark_alt.cli.llm_manager') as mock_manager:
            mock_manager.providers = {}
            status = _get_semantic_operator_status()
            assert "DISABLED" in status

    @pytest.mark.asyncio
    async def test_semantic_llm_calls_tracked(self, mock_llm_provider, sample_population):
        """Test that LLM calls made by semantic operators are tracked."""
        from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider
        
        # Track LLM calls
        llm_call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            return LLMResponse(
                content="Modified idea content", 
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                cost=0.001
            )
        
        mock_llm_provider.generate = mock_generate
        
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        with patch.object(ga.fitness_evaluator, 'evaluate_population', new_callable=AsyncMock) as mock_eval:
            with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', new_callable=AsyncMock) as mock_diversity:
                # Return list of IndividualFitness objects (awaitable)
                mock_eval.return_value = [
                    IndividualFitness(idea=idea, overall_fitness=0.7) 
                    for idea in sample_population
                ]
                mock_diversity.return_value = 0.5  # Low diversity to trigger semantic operators
                
                config = EvolutionConfig(
                    population_size=3,
                    generations=2,
                    use_semantic_operators=True
                )
                
                request = EvolutionRequest(
                    initial_population=sample_population,
                    config=config,
                    context="Test"
                )
                
                # With simplified logic, semantic operators are automatically used when available
                result = await ga.evolve(request)
                
                # Check that LLM was called
                assert llm_call_count > 0
                
                # Check metrics track LLM calls
                metrics = result.evolution_metrics
                assert 'semantic_llm_calls' in metrics
                # The metric might differ slightly from actual calls due to caching
                # or counting methodology (e.g., crossover counts as 1 call but 2 offspring)
                assert metrics['semantic_llm_calls'] > 0
                # Ensure it's within reasonable range
                # With parallel processing, the relationship between actual calls and metrics
                # can vary more significantly due to batching and caching
                assert abs(metrics['semantic_llm_calls'] - llm_call_count) <= 5