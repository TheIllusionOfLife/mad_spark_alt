"""
Tests for the genetic algorithm implementation.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    IndividualFitness,
    SelectionStrategy,
)
from mad_spark_alt.evolution.operators import (
    CrossoverOperator,
    MutationOperator,
)


class TestGeneticAlgorithm:
    """Test suite for GeneticAlgorithm class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock fitness evaluator
        self.mock_fitness_evaluator = MagicMock(spec=FitnessEvaluator)
        self.mock_fitness_evaluator.evaluate_population = AsyncMock()
        self.mock_fitness_evaluator.calculate_population_diversity = AsyncMock(
            return_value=0.7
        )

        # Create genetic algorithm instance
        self.ga = GeneticAlgorithm(fitness_evaluator=self.mock_fitness_evaluator)

        # Create test ideas
        self.test_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}: Innovative solution for problem",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                confidence_score=0.8,
                reasoning="Test reasoning",
                metadata={"test": True, "generation": 0},
                timestamp=datetime.now().isoformat(),
            )
            for i in range(5)
        ]

    @pytest.mark.asyncio
    async def test_evolve_basic(self):
        """Test basic evolution process."""
        # Create mock fitness individuals
        mock_individuals = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.7,
                diversity_score=0.8,
                quality_score=0.6,
                overall_fitness=0.7,
            )
            for idea in self.test_ideas
        ]

        # Configure mock evaluator
        self.mock_fitness_evaluator.evaluate_population.return_value = mock_individuals

        # Create evolution request
        config = EvolutionConfig(
            population_size=5,
            generations=2,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=1,
        )

        request = EvolutionRequest(initial_population=self.test_ideas, config=config)

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify result
        assert result.success
        assert result.total_generations == 3  # 0, 1, 2 (includes initial)
        assert len(result.generation_snapshots) == 3
        assert len(result.best_ideas) <= 10
        assert result.execution_time > 0

        # Verify fitness evaluator was called
        assert self.mock_fitness_evaluator.evaluate_population.called
        assert self.mock_fitness_evaluator.calculate_population_diversity.called

    @pytest.mark.asyncio
    async def test_evolve_with_elite_preservation(self):
        """Test that elite individuals are preserved."""
        # Create individuals with varying fitness
        mock_individuals = []
        for i, idea in enumerate(self.test_ideas):
            fitness = IndividualFitness(
                idea=idea,
                creativity_score=0.7,
                diversity_score=0.8,
                quality_score=0.6,
                overall_fitness=0.9 - (i * 0.1),  # Decreasing fitness
            )
            mock_individuals.append(fitness)

        self.mock_fitness_evaluator.evaluate_population.return_value = mock_individuals

        # Configure with elite preservation
        config = EvolutionConfig(
            population_size=5,
            generations=1,
            elite_size=2,
            selection_strategy=SelectionStrategy.ELITE,
        )

        request = EvolutionRequest(initial_population=self.test_ideas, config=config)

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify elite preservation
        assert result.success
        best_fitness = result.generation_snapshots[-1].best_fitness
        assert best_fitness >= 0.8  # Best individual should be preserved

    @pytest.mark.asyncio
    async def test_evolve_with_target_metrics(self):
        """Test early termination with target metrics."""
        # Create high-fitness individuals
        mock_individuals = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.9,
                diversity_score=0.9,
                quality_score=0.9,
                overall_fitness=0.9,
            )
            for idea in self.test_ideas
        ]

        self.mock_fitness_evaluator.evaluate_population.return_value = mock_individuals

        # Configure with target metrics
        config = EvolutionConfig(
            population_size=5,
            generations=10,  # Many generations
        )

        request = EvolutionRequest(
            initial_population=self.test_ideas,
            config=config,
            target_metrics={"min_fitness": 0.85},  # Should terminate early
        )

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify early termination
        assert result.success
        assert result.total_generations < 10  # Should terminate before all generations

    @pytest.mark.asyncio
    async def test_evolve_with_adaptive_mutation(self):
        """Test adaptive mutation rate adjustment."""
        # Create individuals
        mock_individuals = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.7,
                diversity_score=0.8,
                quality_score=0.6,
                overall_fitness=0.7,
            )
            for idea in self.test_ideas
        ]

        self.mock_fitness_evaluator.evaluate_population.return_value = mock_individuals

        # Configure with adaptive mutation
        config = EvolutionConfig(
            population_size=5, generations=2, mutation_rate=0.1, adaptive_mutation=True
        )

        # Mock low diversity to trigger mutation rate increase
        self.mock_fitness_evaluator.calculate_population_diversity.return_value = 0.2

        request = EvolutionRequest(initial_population=self.test_ideas, config=config)

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify evolution completed
        assert result.success
        # Mutation rate should have been adapted (we can't easily check the internal state)

    @pytest.mark.asyncio
    async def test_evolve_invalid_request(self):
        """Test evolution with invalid request."""
        # Create invalid request (empty population)
        request = EvolutionRequest(initial_population=[], config=EvolutionConfig())

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify failure
        assert not result.success
        assert result.error_message == "Invalid evolution request"

    @pytest.mark.asyncio
    async def test_evolve_with_exception(self):
        """Test evolution handles exceptions gracefully."""
        # Configure mock to raise exception
        self.mock_fitness_evaluator.evaluate_population.side_effect = Exception(
            "Test error"
        )

        request = EvolutionRequest(
            initial_population=self.test_ideas, config=EvolutionConfig()
        )

        # Run evolution
        result = await self.ga.evolve(request)

        # Verify error handling
        assert not result.success
        assert "Test error" in result.error_message
        assert result.execution_time > 0


class TestGeneticOperators:
    """Test suite for genetic operators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.crossover_op = CrossoverOperator()
        self.mutation_op = MutationOperator()

        # Create test ideas
        self.parent1 = GeneratedIdea(
            content="First innovative idea. It uses AI. It improves efficiency.",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent1",
            generation_prompt="Test prompt 1",
            confidence_score=0.8,
            metadata={"generation": 0},
        )

        self.parent2 = GeneratedIdea(
            content="Second creative solution. It reduces cost. It scales well.",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent2",
            generation_prompt="Test prompt 2",
            confidence_score=0.7,
            metadata={"generation": 0},
        )

    @pytest.mark.asyncio
    async def test_crossover_operator(self):
        """Test crossover operation."""
        # Perform crossover
        offspring1, offspring2 = await self.crossover_op.crossover(
            self.parent1, self.parent2
        )

        # Verify offspring properties
        assert isinstance(offspring1, GeneratedIdea)
        assert isinstance(offspring2, GeneratedIdea)

        # Check that offspring are different from parents
        assert offspring1.content != self.parent1.content
        assert offspring2.content != self.parent2.content

        # Check metadata
        assert offspring1.metadata["operator"] == "crossover"
        assert offspring2.metadata["operator"] == "crossover"
        assert offspring1.metadata["generation"] == 1
        assert offspring2.metadata["generation"] == 1

        # Check parent tracking
        assert len(offspring1.parent_ideas) == 2
        assert len(offspring2.parent_ideas) == 2
        assert offspring1.parent_ideas[0] == self.parent1.content
        assert offspring1.parent_ideas[1] == self.parent2.content

    @pytest.mark.asyncio
    async def test_mutation_operator(self):
        """Test mutation operation."""
        # Test with high mutation rate to ensure mutation occurs
        # Mutation is random, so retry until content actually changes
        max_attempts = 10
        content_changed = False
        final_mutated = None
        
        for attempt in range(max_attempts):
            mutated = await self.mutation_op.mutate(
                self.parent1, mutation_rate=1.0  # Always mutate
            )
            
            if mutated.content != self.parent1.content:
                content_changed = True
                final_mutated = mutated
                break
            final_mutated = mutated  # Keep the last attempt for verification

        # Verify mutation occurred (at least once in max_attempts)
        assert content_changed, f"Content should have changed after {max_attempts} attempts"
        assert isinstance(final_mutated, GeneratedIdea)
        assert final_mutated.metadata["operator"] == "mutation"
        assert final_mutated.metadata["generation"] == 1
        assert len(final_mutated.parent_ideas) == 1
        assert final_mutated.parent_ideas[0] == self.parent1.content

    @pytest.mark.asyncio
    async def test_mutation_no_change(self):
        """Test mutation with zero rate."""
        # Test with zero mutation rate
        result = await self.mutation_op.mutate(
            self.parent1, mutation_rate=0.0  # Never mutate
        )

        # Should return a new object with same content but updated metadata
        assert result is not self.parent1  # Different object
        assert result.content == self.parent1.content  # Same content
        assert result.metadata["generation"] == 1  # Generation incremented
        assert result.metadata["mutation_type"] == "none"  # No mutation applied
        assert result.metadata["mutation_rate"] == 0.0


class TestEvolutionConfig:
    """Test suite for EvolutionConfig."""

    def test_config_validation_valid(self):
        """Test valid configuration."""
        config = EvolutionConfig(
            population_size=50,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=5,
        )
        assert config.validate()

    def test_config_validation_invalid_population(self):
        """Test invalid population size."""
        config = EvolutionConfig(population_size=1)
        assert not config.validate()

    def test_config_validation_invalid_mutation_rate(self):
        """Test invalid mutation rate."""
        config = EvolutionConfig(mutation_rate=1.5)
        assert not config.validate()

    def test_config_validation_invalid_elite_size(self):
        """Test invalid elite size."""
        config = EvolutionConfig(
            population_size=10, elite_size=15  # Larger than population
        )
        assert not config.validate()
