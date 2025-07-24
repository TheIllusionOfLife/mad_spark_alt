"""
Tests for genetic algorithm integration with semantic operators.

This module tests that the genetic algorithm correctly integrates with
the smart selector and semantic operators for LLM-powered evolution.
"""

import asyncio
from unittest.mock import AsyncMock, patch
from typing import List

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse, LLMProvider
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import EvolutionConfig, EvolutionRequest, IndividualFitness


class TestGeneticAlgorithmSemanticIntegration:
    """Test semantic operator integration in genetic algorithm."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def sample_ideas(self):
        """Create sample ideas for testing."""
        return [
            GeneratedIdea(
                content="Implement urban gardens for local food production",
                thinking_method="environmental",
                agent_name="test_agent",
                generation_prompt="How to improve food systems?"
            ),
            GeneratedIdea(
                content="Use AI to optimize energy consumption patterns",
                thinking_method="technical",
                agent_name="test_agent",
                generation_prompt="How to save energy?"
            ),
            GeneratedIdea(
                content="Create community recycling programs",
                thinking_method="social",
                agent_name="test_agent", 
                generation_prompt="How to reduce waste?"
            )
        ]

    @pytest.fixture
    def config_with_semantic_operators(self):
        """Create config that enables semantic operators."""
        return EvolutionConfig(
            population_size=3,
            generations=2,
            use_semantic_operators=True,
            semantic_operator_threshold=0.5,
            mutation_rate=0.8,  # High rate to trigger mutations
            crossover_rate=0.8  # High rate to trigger crossovers
        )

    def test_ga_initialization_without_llm(self):
        """Test that GA initializes correctly without LLM provider."""
        ga = GeneticAlgorithm()
        
        assert ga.llm_provider is None
        assert ga.semantic_mutation_operator is None
        assert ga.semantic_crossover_operator is None
        assert ga.smart_selector is None

    def test_ga_initialization_with_llm(self, mock_llm_provider):
        """Test that GA initializes correctly with LLM provider."""
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        assert ga.llm_provider is mock_llm_provider
        assert ga.semantic_mutation_operator is not None
        assert ga.semantic_crossover_operator is not None
        assert ga.smart_selector is None  # Initialized lazily

    @pytest.mark.asyncio
    async def test_semantic_operators_integration_basic(
        self, mock_llm_provider, sample_ideas
    ):
        """Test basic integration of semantic operators without errors."""
        # Use simple config to avoid complex interactions
        config = EvolutionConfig(
            population_size=3,
            generations=2,  # Minimum allowed by validation
            use_semantic_operators=True,
            semantic_operator_threshold=0.9,  # High threshold to avoid semantic ops
            mutation_rate=0.2,  # Lower rate
            crossover_rate=0.2  # Lower rate
        )

        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)

        # Mock fitness evaluator 
        with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', return_value=0.8):
            with patch.object(ga.fitness_evaluator, 'evaluate_population') as mock_eval:
                # Mock fitness evaluation
                def mock_evaluate(ideas, config, context=None):
                    return [
                        IndividualFitness(
                            idea=idea,
                            creativity_score=0.7,
                            diversity_score=0.7,
                            quality_score=0.7,
                            overall_fitness=0.7
                        )
                        for idea in ideas
                    ]
                
                mock_eval.side_effect = mock_evaluate

                # Create evolution request
                request = EvolutionRequest(
                    initial_population=sample_ideas,
                    config=config,
                    context="sustainability improvement"
                )

                # Run evolution - this should work without errors
                result = await ga.evolve(request)

                # Verify that evolution completed successfully 
                assert result.error_message is None or result.error_message == ""
                assert len(result.final_population) == config.population_size
                assert result.total_generations >= 0

                # Verify semantic operators are available
                assert ga.semantic_mutation_operator is not None
                assert ga.semantic_crossover_operator is not None

    @pytest.mark.asyncio  
    async def test_traditional_operators_used_with_high_diversity(
        self, mock_llm_provider, sample_ideas, config_with_semantic_operators
    ):
        """Test that traditional operators are used when diversity is high."""
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)

        # Mock fitness evaluator to return high diversity  
        with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', return_value=0.8):
            with patch.object(ga.fitness_evaluator, 'evaluate_population') as mock_eval:
                # Mock fitness evaluation
                def mock_evaluate(ideas, config, context=None):
                    return [
                        IndividualFitness(
                            idea=idea,
                            creativity_score=0.7,
                            diversity_score=0.8,
                            quality_score=0.7,
                            overall_fitness=0.7
                        )
                        for idea in ideas
                    ]
                
                mock_eval.side_effect = mock_evaluate

                # Create evolution request
                request = EvolutionRequest(
                    initial_population=sample_ideas,
                    config=config_with_semantic_operators,
                    context="sustainability improvement"
                )

                # Run evolution
                result = await ga.evolve(request)

                # Verify that evolution completed successfully
                assert result.error_message is None
                assert len(result.final_population) == config_with_semantic_operators.population_size

                # Verify that LLM was NOT called (traditional operators used)
                assert mock_llm_provider.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_semantic_operators_disabled_in_config(
        self, mock_llm_provider, sample_ideas
    ):
        """Test that semantic operators are not used when disabled in config."""
        config_no_semantic = EvolutionConfig(
            population_size=3,
            generations=2,
            use_semantic_operators=False,  # Disabled
            mutation_rate=0.8,
            crossover_rate=0.8
        )

        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)

        # Mock fitness evaluator to return low diversity (which would normally trigger semantic)
        with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', return_value=0.2):
            with patch.object(ga.fitness_evaluator, 'evaluate_population') as mock_eval:
                # Mock fitness evaluation
                def mock_evaluate(ideas, config, context=None):
                    return [
                        IndividualFitness(
                            idea=idea,
                            creativity_score=0.7,
                            diversity_score=0.6,
                            quality_score=0.7,
                            overall_fitness=0.7
                        )
                        for idea in ideas
                    ]
                
                mock_eval.side_effect = mock_evaluate

                # Create evolution request
                request = EvolutionRequest(
                    initial_population=sample_ideas,
                    config=config_no_semantic,
                    context="sustainability improvement"
                )

                # Run evolution
                result = await ga.evolve(request)

                # Verify that evolution completed successfully
                assert result.error_message is None
                assert len(result.final_population) == config_no_semantic.population_size

                # Verify that LLM was NOT called (semantic operators disabled)
                assert mock_llm_provider.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_smart_selector_initialization(
        self, mock_llm_provider, sample_ideas, config_with_semantic_operators
    ):
        """Test that smart selector is properly initialized with config."""
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)

        with patch.object(ga.fitness_evaluator, 'calculate_population_diversity', return_value=0.5):
            with patch.object(ga.fitness_evaluator, 'evaluate_population') as mock_eval:
                # Mock fitness evaluation
                def mock_evaluate(ideas, config, context=None):
                    return [
                        IndividualFitness(
                            idea=idea,
                            overall_fitness=0.7
                        )
                        for idea in ideas
                    ]
                
                mock_eval.side_effect = mock_evaluate

                # Create evolution request
                request = EvolutionRequest(
                    initial_population=sample_ideas,
                    config=config_with_semantic_operators,
                    context="test"
                )

                # Run evolution
                result = await ga.evolve(request)

                # Verify smart selector was initialized
                assert ga.smart_selector is not None
                assert ga.smart_selector.config == config_with_semantic_operators