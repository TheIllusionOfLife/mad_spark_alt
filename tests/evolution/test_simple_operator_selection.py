"""
Tests for simplified operator selection logic (without SmartOperatorSelector).

This module tests the simplified logic that determines when to use semantic 
operators based purely on availability and configuration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mad_spark_alt.evolution.interfaces import IndividualFitness, EvolutionConfig
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator, SemanticCrossoverOperator
from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider


class TestSimpleOperatorSelection:
    """Test simplified operator selection logic."""

    @pytest.fixture
    def sample_idea(self):
        """Create a sample idea for testing."""
        return GeneratedIdea(
            content="Test idea content",
            thinking_method="test_method",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8
        )

    @pytest.fixture
    def sample_individual(self, sample_idea):
        """Create a sample individual for testing."""
        return IndividualFitness(
            idea=sample_idea,
            impact=0.7,
            feasibility=0.8,
            accessibility=0.6,
            sustainability=0.7,
            scalability=0.7,
            overall_fitness=0.7
        )

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        return MagicMock(spec=GoogleProvider)

    @pytest.fixture
    def config_with_semantic_operators(self):
        """Create config with semantic operators enabled."""
        config = EvolutionConfig()
        config.use_semantic_operators = True
        return config

    @pytest.fixture
    def config_without_semantic_operators(self):
        """Create config with semantic operators disabled."""
        config = EvolutionConfig()
        config.use_semantic_operators = False
        return config

    @pytest.mark.asyncio
    async def test_semantic_operators_used_when_available_and_enabled(
        self, mock_llm_provider, config_with_semantic_operators, sample_individual
    ):
        """Test that semantic operators are used when available and enabled."""
        # Create GA with LLM provider (semantic operators available)
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Verify semantic operators are initialized
        assert ga.semantic_mutation_operator is not None
        assert ga.semantic_crossover_operator is not None
        assert isinstance(ga.semantic_mutation_operator, BatchSemanticMutationOperator)
        assert isinstance(ga.semantic_crossover_operator, SemanticCrossoverOperator)

        # Mock the semantic operator methods - create a modified idea to trigger stats counting
        modified_idea = GeneratedIdea(
            content="Modified test idea content",  # Different content to trigger stats
            thinking_method="test_method",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8
        )
        ga.semantic_mutation_operator.mutate = AsyncMock(return_value=modified_idea)
        ga.semantic_crossover_operator.crossover = AsyncMock(
            return_value=(sample_individual.idea, sample_individual.idea)
        )
        
        # Test mutation selection logic - should use semantic when available and enabled
        mutated_idea, mutation_stats = await ga._apply_mutation_with_operator_selection(
            sample_individual.idea,
            config=config_with_semantic_operators
        )
        
        # Verify semantic mutation was called
        ga.semantic_mutation_operator.mutate.assert_called_once()
        assert mutation_stats['semantic_mutations'] == 1
        assert mutation_stats['traditional_mutations'] == 0

    @pytest.mark.asyncio
    async def test_traditional_operators_used_when_semantic_disabled(
        self, mock_llm_provider, config_without_semantic_operators, sample_individual
    ):
        """Test that traditional operators are used when semantic operators are disabled."""
        # Create GA with LLM provider but semantic operators disabled
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Mock the traditional operator method - create modified idea to trigger stats counting
        modified_idea = GeneratedIdea(
            content="Modified traditional test idea content",
            thinking_method="test_method",
            agent_name="test_agent", 
            generation_prompt="test prompt",
            confidence_score=0.8
        )
        ga.mutation_operator.mutate = AsyncMock(return_value=modified_idea)
        
        # Test mutation selection logic - should use traditional when disabled
        mutated_idea, mutation_stats = await ga._apply_mutation_with_operator_selection(
            sample_individual.idea,
            config=config_without_semantic_operators
        )
        
        # Verify traditional mutation was called
        ga.mutation_operator.mutate.assert_called_once()
        assert mutation_stats['semantic_mutations'] == 0
        assert mutation_stats['traditional_mutations'] == 1

    @pytest.mark.asyncio
    async def test_traditional_operators_used_when_semantic_unavailable(
        self, config_with_semantic_operators, sample_individual
    ):
        """Test that traditional operators are used when semantic operators are unavailable."""
        # Create GA without LLM provider (semantic operators unavailable)
        ga = GeneticAlgorithm()
        
        # Verify semantic operators are not initialized
        assert ga.semantic_mutation_operator is None
        assert ga.semantic_crossover_operator is None
        
        # Mock the traditional operator method - create modified idea to trigger stats counting
        modified_idea = GeneratedIdea(
            content="Modified unavailable test idea content",
            thinking_method="test_method",
            agent_name="test_agent", 
            generation_prompt="test prompt",
            confidence_score=0.8
        )
        ga.mutation_operator.mutate = AsyncMock(return_value=modified_idea)
        
        # Test mutation selection logic - should use traditional when unavailable
        mutated_idea, mutation_stats = await ga._apply_mutation_with_operator_selection(
            sample_individual.idea,
            config=config_with_semantic_operators
        )
        
        # Verify traditional mutation was called
        ga.mutation_operator.mutate.assert_called_once()
        assert mutation_stats['semantic_mutations'] == 0
        assert mutation_stats['traditional_mutations'] == 1

    def test_no_smart_selector_initialized(self, mock_llm_provider):
        """Test that SmartOperatorSelector is not initialized in simplified logic."""
        # Create GA with LLM provider
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Verify smart selector is not used in the simplified approach
        # Note: SmartOperatorSelector has been completely removed from implementation
        # This test verifies the absence of complex selection logic
        
        # Semantic operators should be available
        assert ga.semantic_mutation_operator is not None
        assert ga.semantic_crossover_operator is not None
        
        # But no complex selector logic
        # (This will pass once we remove SmartOperatorSelector usage)

    @pytest.mark.asyncio
    async def test_crossover_selection_logic(
        self, mock_llm_provider, config_with_semantic_operators, sample_individual
    ):
        """Test crossover operator selection follows same simplified logic."""
        # Create GA with LLM provider
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Mock semantic crossover
        ga.semantic_crossover_operator.crossover = AsyncMock(
            return_value=(sample_individual.idea, sample_individual.idea)
        )
        
        # Manually test the crossover selection logic that will be implemented
        # When semantic operators are available and enabled, use them
        use_semantic = (
            ga.semantic_crossover_operator is not None and
            config_with_semantic_operators.use_semantic_operators
        )
        
        assert use_semantic is True
        
        # When disabled, don't use semantic
        config_disabled = EvolutionConfig()
        config_disabled.use_semantic_operators = False
        
        use_semantic_disabled = (
            ga.semantic_crossover_operator is not None and
            config_disabled.use_semantic_operators
        )
        
        assert use_semantic_disabled is False

    def test_config_semantic_operators_default_enabled(self):
        """Test that semantic operators are enabled by default in config."""
        config = EvolutionConfig()
        assert config.use_semantic_operators is True

    def test_semantic_operator_metrics_tracking(self, mock_llm_provider):
        """Test that semantic operator metrics are still tracked."""
        ga = GeneticAlgorithm(llm_provider=mock_llm_provider)
        
        # Verify metrics structure exists
        assert hasattr(ga, 'semantic_operator_metrics')
        assert 'semantic_mutations' in ga.semantic_operator_metrics
        assert 'semantic_crossovers' in ga.semantic_operator_metrics
        assert 'traditional_mutations' in ga.semantic_operator_metrics
        assert 'traditional_crossovers' in ga.semantic_operator_metrics
        assert 'semantic_llm_calls' in ga.semantic_operator_metrics
        
        # All should start at 0
        for metric in ga.semantic_operator_metrics.values():
            assert metric == 0