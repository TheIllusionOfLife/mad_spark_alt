"""
Comprehensive unit tests for the FitnessEvaluator component.

This module provides extensive test coverage for the FitnessEvaluator class,
including individual fitness evaluation, population evaluation (both parallel
and sequential modes), error handling for partial failures, score extraction
methods, edge cases with empty results, and concurrency control verification.

The test strategy focuses on:
- Individual fitness evaluation (success/failure scenarios)
- Population evaluation with different concurrency modes
- Error handling and graceful degradation
- Resource management with semaphore-based concurrency control
- Edge cases and boundary conditions
- Performance characteristics of parallel evaluation
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import (
    GeneratedIdea,
    ThinkingMethod,
)
from mad_spark_alt.core.unified_evaluator import HypothesisEvaluation, UnifiedEvaluator
from mad_spark_alt.evolution.fitness import DEFAULT_FAILURE_SCORE, FitnessEvaluator
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness


class TestFitnessEvaluator:
    """Test suite for FitnessEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock unified evaluator
        self.mock_unified_evaluator = MagicMock(spec=UnifiedEvaluator)
        self.mock_unified_evaluator.evaluate_hypothesis = AsyncMock()

        # Create fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            unified_evaluator=self.mock_unified_evaluator,
        )

        # Create test configuration
        self.config = EvolutionConfig(
            population_size=5,
            generations=3,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=1,
            fitness_weights={
                "novelty": 0.2,
                "impact": 0.3,
                "cost_efficiency": 0.2,
                "feasibility": 0.2,
                "risk": 0.1,
            },
        )

        # Create test ideas
        self.test_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}: Solution for sustainability",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                confidence_score=0.8 + i * 0.05,
                reasoning="Test reasoning",
                metadata={"test": True, "id": i},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            for i in range(3)
        ]

    @pytest.mark.asyncio
    async def test_evaluate_individual_success(self) -> None:
        """Test successful individual fitness evaluation."""
        # Mock evaluation result
        mock_evaluation = HypothesisEvaluation(
            content="Test idea 0: Solution for sustainability",
            scores={
                "novelty": 0.8,
                "impact": 0.85,
                "cost": 0.3,  # Low cost is good
                "feasibility": 0.9,
                "risks": 0.2,  # Low risk is good
            },
            overall_score=0.82,
            explanations={"reasoning": "Test evaluation reasoning"},
            metadata={"llm_cost": 0.001},
        )

        self.mock_unified_evaluator.evaluate_hypothesis.return_value = mock_evaluation

        # Evaluate individual
        idea = self.test_ideas[0]
        fitness = await self.fitness_evaluator.evaluate_individual(
            idea, self.config, context="Test context"
        )

        # Verify results
        assert isinstance(fitness, IndividualFitness)
        assert fitness.idea == idea
        # creativity_score maps to novelty
        assert fitness.creativity_score == 0.8
        # diversity_score is average of novelty and impact
        assert fitness.diversity_score == pytest.approx((0.8 + 0.85) / 2, rel=1e-3)
        # quality_score maps to feasibility
        assert fitness.quality_score == 0.9
        assert fitness.overall_fitness == pytest.approx(0.82, rel=1e-3)
        # Check metadata contains all scores
        assert "unified_scores" in fitness.evaluation_metadata
        assert fitness.evaluation_metadata["unified_scores"]["novelty"] == 0.8

        # Verify evaluation was called correctly
        self.mock_unified_evaluator.evaluate_hypothesis.assert_called_once()
        call_args = self.mock_unified_evaluator.evaluate_hypothesis.call_args
        assert call_args[1]["hypothesis"] == idea.content
        assert call_args[1]["context"] == "Test context"

    @pytest.mark.asyncio
    async def test_evaluate_individual_failure(self) -> None:
        """Test individual fitness evaluation with error."""
        # Mock evaluation failure
        self.mock_unified_evaluator.evaluate_hypothesis.side_effect = Exception(
            "Evaluation failed"
        )

        # Evaluate individual
        idea = self.test_ideas[0]
        fitness = await self.fitness_evaluator.evaluate_individual(idea, self.config)

        # Verify failure handling
        assert isinstance(fitness, IndividualFitness)
        assert fitness.idea == idea
        assert fitness.creativity_score == 0.0
        assert fitness.diversity_score == 0.0
        assert fitness.quality_score == 0.0
        assert fitness.overall_fitness == 0.0
        assert "error" in fitness.evaluation_metadata
        assert fitness.evaluation_metadata["error"] == "Evaluation failed"

    @pytest.mark.asyncio
    async def test_evaluate_population_parallel(self) -> None:
        """Test parallel population evaluation."""
        # Mock individual evaluations
        mock_fitness_results = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.8 + i * 0.05,
                diversity_score=0.7 + i * 0.05,
                quality_score=0.75 + i * 0.05,
                overall_fitness=0.75 + i * 0.05,
            )
            for i, idea in enumerate(self.test_ideas)
        ]

        # Patch evaluate_individual to return mock results
        with patch.object(
            self.fitness_evaluator,
            "evaluate_individual",
            side_effect=mock_fitness_results,
        ):
            # Evaluate population
            self.config.parallel_evaluation = True
            results = await self.fitness_evaluator.evaluate_population(
                self.test_ideas, self.config
            )

            # Verify results
            assert len(results) == len(self.test_ideas)
            for i, fitness in enumerate(results):
                assert fitness.idea == self.test_ideas[i]
                assert fitness.creativity_score == 0.8 + i * 0.05

    @pytest.mark.asyncio
    async def test_evaluate_population_sequential(self) -> None:
        """Test sequential population evaluation."""
        # Mock individual evaluations
        mock_fitness_results = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
            )
            for idea in self.test_ideas
        ]

        # Patch evaluate_individual
        with patch.object(
            self.fitness_evaluator,
            "evaluate_individual",
            side_effect=mock_fitness_results,
        ):
            # Evaluate population sequentially
            self.config.parallel_evaluation = False
            results = await self.fitness_evaluator.evaluate_population(
                self.test_ideas, self.config
            )

            # Verify results
            assert len(results) == len(self.test_ideas)
            for fitness in results:
                assert fitness.creativity_score == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_population_with_exception(self) -> None:
        """Test population evaluation with some failures."""

        # Mock mixed results (success and failure)
        async def mock_evaluate(idea, config, context=None):
            if idea.metadata.get("id", 0) == 1:
                raise Exception("Evaluation error")
            return IndividualFitness(
                idea=idea,
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
            )

        # Patch evaluate_individual
        with patch.object(
            self.fitness_evaluator, "evaluate_individual", side_effect=mock_evaluate
        ):
            # Evaluate population
            self.config.parallel_evaluation = True
            results = await self.fitness_evaluator.evaluate_population(
                self.test_ideas, self.config
            )

            # Verify results
            assert len(results) == len(self.test_ideas)
            # First and third should succeed
            assert results[0].creativity_score == 0.8
            assert results[2].creativity_score == 0.8
            # Second should fail
            assert results[1].creativity_score == 0.0
            assert results[1].overall_fitness == 0.0
            assert "error" in results[1].evaluation_metadata

    @pytest.mark.asyncio
    async def test_calculate_population_diversity(self) -> None:
        """Test population diversity calculation."""
        # Create fitness individuals with proper evaluation_metadata structure
        diverse_individuals = [
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Solar panels on every building",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
                evaluation_metadata={
                    "unified_scores": {
                        "novelty": 0.8,
                        "impact": 0.7,
                        "cost": 0.6,
                        "feasibility": 0.75,
                        "risks": 0.8
                    },
                    "unified_explanations": {
                        "novelty": "Innovative solar approach",
                        "impact": "High environmental impact",
                        "cost": "Moderate implementation cost",
                        "feasibility": "Technologically feasible",
                        "risks": "Low technical risks"
                    },
                    "llm_cost": 0.01
                }
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Urban gardens and green spaces",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
                evaluation_metadata={
                    "unified_scores": {
                        "novelty": 0.6,
                        "impact": 0.8,
                        "cost": 0.7,
                        "feasibility": 0.8,
                        "risks": 0.9
                    },
                    "unified_explanations": {
                        "novelty": "Moderate novelty in urban planning",
                        "impact": "High environmental and social impact",
                        "cost": "Reasonable implementation cost",
                        "feasibility": "Highly feasible",
                        "risks": "Very low risks"
                    },
                    "llm_cost": 0.01
                }
            ),
        ]

        # Calculate diversity
        diversity = await self.fitness_evaluator.calculate_population_diversity(
            diverse_individuals
        )

        # Verify diversity is calculated - should be 1.0 since content has no overlapping words
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
        assert diversity == 1.0  # No common words, so maximum diversity

        # Test with single individual (should return 1.0)
        single_individual = [diverse_individuals[0]]
        single_diversity = await self.fitness_evaluator.calculate_population_diversity(
            single_individual
        )
        assert single_diversity == 1.0


    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(self) -> None:
        """Test that semaphore properly limits concurrent evaluations."""
        # Create a config with limited parallelism
        limited_config = EvolutionConfig(
            population_size=10,
            generations=1,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=1,
            max_parallel_evaluations=2,  # Limit to 2 concurrent
        )

        # Track concurrent evaluations
        concurrent_count = 0
        max_concurrent = 0

        async def mock_evaluate_with_tracking(idea, config, context=None):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate evaluation time
            await asyncio.sleep(0.01)

            concurrent_count -= 1
            return IndividualFitness(
                idea=idea,
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
            )

        # Patch evaluate_individual
        with patch.object(
            self.fitness_evaluator,
            "evaluate_individual",
            side_effect=mock_evaluate_with_tracking,
        ):
            # Create many ideas to test concurrency
            many_ideas = [
                GeneratedIdea(
                    content=f"Idea {i}",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                for i in range(10)
            ]

            # Evaluate population
            limited_config.parallel_evaluation = True
            await self.fitness_evaluator.evaluate_population(many_ideas, limited_config)

            # Verify concurrency was limited
            assert max_concurrent <= limited_config.max_parallel_evaluations
