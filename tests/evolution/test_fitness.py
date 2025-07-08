"""
Unit tests for the FitnessEvaluator component.

This module tests individual fitness evaluation, population evaluation,
and diversity calculation methods.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import (
    CreativityEvaluator,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    GeneratedIdea,
    ModelOutput,
    OutputType,
    ThinkingMethod,
)
from mad_spark_alt.core.evaluator import EvaluationSummary
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness


class TestFitnessEvaluator:
    """Test suite for FitnessEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock creativity evaluator
        self.mock_creativity_evaluator = MagicMock(spec=CreativityEvaluator)
        self.mock_creativity_evaluator.evaluate = AsyncMock()

        # Create fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            creativity_evaluator=self.mock_creativity_evaluator
        )

        # Create test configuration
        self.config = EvolutionConfig(
            population_size=5,
            generations=3,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=1,
            fitness_weights={
                "creativity_score": 0.4,
                "diversity_score": 0.3,
                "quality_score": 0.3,
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
                timestamp=datetime.now().isoformat(),
            )
            for i in range(3)
        ]

    @pytest.mark.asyncio
    async def test_evaluate_individual_success(self) -> None:
        """Test successful individual fitness evaluation."""
        # Mock evaluation summary
        mock_summary = EvaluationSummary(
            request_id="test-request",
            total_outputs=1,
            total_evaluators=2,
            execution_time=0.1,
            layer_results={
                EvaluationLayer.QUANTITATIVE: [
                    EvaluationResult(
                        evaluator_name="diversity",
                        layer=EvaluationLayer.QUANTITATIVE,
                        scores={
                            "distinct_1": 0.8,
                            "distinct_2": 0.7,
                            "semantic_uniqueness": 0.9,
                            "lexical_diversity": 0.75,
                        },
                    ),
                    EvaluationResult(
                        evaluator_name="quality",
                        layer=EvaluationLayer.QUANTITATIVE,
                        scores={
                            "readability_score": 0.85,
                            "grammar_score": 0.9,
                            "coherence_score": 0.8,
                            "fluency_score": 0.88,
                        },
                    ),
                ]
            },
        )
        mock_summary.get_overall_creativity_score = MagicMock(return_value=0.82)

        self.mock_creativity_evaluator.evaluate.return_value = mock_summary

        # Evaluate individual
        idea = self.test_ideas[0]
        fitness = await self.fitness_evaluator.evaluate_individual(
            idea, self.config, context="Test context"
        )

        # Verify results
        assert isinstance(fitness, IndividualFitness)
        assert fitness.idea == idea
        assert fitness.creativity_score == 0.82
        assert fitness.diversity_score == pytest.approx(0.7875, rel=1e-3)
        assert fitness.quality_score == pytest.approx(0.8575, rel=1e-3)
        assert fitness.overall_fitness > 0

        # Verify evaluation was called correctly
        self.mock_creativity_evaluator.evaluate.assert_called_once()
        call_args = self.mock_creativity_evaluator.evaluate.call_args[0][0]
        assert isinstance(call_args, EvaluationRequest)
        assert len(call_args.outputs) == 1
        assert call_args.evaluation_config["context"] == "Test context"

    @pytest.mark.asyncio
    async def test_evaluate_individual_failure(self) -> None:
        """Test individual fitness evaluation with error."""
        # Mock evaluation failure
        self.mock_creativity_evaluator.evaluate.side_effect = Exception(
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
        # Mock the creativity evaluator to return diversity metrics
        mock_summary = EvaluationSummary(
            request_id="diversity-test",
            total_outputs=3,
            total_evaluators=1,
            execution_time=0.1,
            layer_results={
                EvaluationLayer.QUANTITATIVE: [
                    EvaluationResult(
                        evaluator_name="diversity",
                        layer=EvaluationLayer.QUANTITATIVE,
                        scores={"semantic_uniqueness": 0.8},
                    )
                ]
            },
        )
        self.mock_creativity_evaluator.evaluate.return_value = mock_summary

        # Create fitness individuals with varying content
        diverse_individuals = [
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Solar panels on every building",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                    timestamp=datetime.now().isoformat(),
                ),
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Urban gardens and green spaces",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                    timestamp=datetime.now().isoformat(),
                ),
                creativity_score=0.8,
                diversity_score=0.7,
                quality_score=0.75,
                overall_fitness=0.75,
            ),
        ]

        # Calculate diversity
        diversity = await self.fitness_evaluator.calculate_population_diversity(
            diverse_individuals
        )

        # Verify diversity is calculated
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1

        # Test with single individual (should return 1.0)
        single_individual = [diverse_individuals[0]]
        single_diversity = await self.fitness_evaluator.calculate_population_diversity(
            single_individual
        )
        assert single_diversity == 1.0

    @pytest.mark.asyncio
    async def test_extract_diversity_score(self) -> None:
        """Test diversity score extraction from evaluation results."""
        # Create layer results with diversity metrics
        layer_results = {
            EvaluationLayer.QUANTITATIVE: [
                EvaluationResult(
                    evaluator_name="diversity",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={
                        "distinct_1": 0.8,
                        "distinct_2": 0.7,
                        "semantic_uniqueness": 0.9,
                        "unrelated_metric": 0.5,
                    },
                ),
                EvaluationResult(
                    evaluator_name="other",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={"lexical_diversity": 0.75, "other_metric": 0.6},
                ),
            ]
        }

        # Extract diversity score
        score = self.fitness_evaluator._extract_diversity_score(layer_results)

        # Should average the diversity metrics: (0.8 + 0.7 + 0.9 + 0.75) / 4
        assert score == pytest.approx(0.7875, rel=1e-3)

    @pytest.mark.asyncio
    async def test_extract_quality_score(self) -> None:
        """Test quality score extraction from evaluation results."""
        # Create layer results with quality metrics
        layer_results = {
            EvaluationLayer.QUANTITATIVE: [
                EvaluationResult(
                    evaluator_name="quality",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={
                        "readability_score": 0.85,
                        "grammar_score": 0.9,
                        "unrelated_metric": 0.5,
                    },
                ),
                EvaluationResult(
                    evaluator_name="coherence",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={"coherence_score": 0.8, "fluency_score": 0.88},
                ),
            ]
        }

        # Extract quality score
        score = self.fitness_evaluator._extract_quality_score(layer_results)

        # Should average the quality metrics: (0.85 + 0.9 + 0.8 + 0.88) / 4
        assert score == pytest.approx(0.8575, rel=1e-3)

    def test_extract_scores_empty_results(self) -> None:
        """Test score extraction with empty results."""
        empty_results = {EvaluationLayer.QUANTITATIVE: []}

        # Both should return default failure score
        diversity = self.fitness_evaluator._extract_diversity_score(empty_results)
        quality = self.fitness_evaluator._extract_quality_score(empty_results)

        assert diversity == 0.1  # DEFAULT_FAILURE_SCORE
        assert quality == 0.1  # DEFAULT_FAILURE_SCORE

    def test_extract_scores_no_matching_metrics(self) -> None:
        """Test score extraction with no matching metrics."""
        results = {
            EvaluationLayer.QUANTITATIVE: [
                EvaluationResult(
                    evaluator_name="other",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={"unrelated_1": 0.8, "unrelated_2": 0.7},
                )
            ]
        }

        # Should return default failure score when no metrics match
        diversity = self.fitness_evaluator._extract_diversity_score(results)
        quality = self.fitness_evaluator._extract_quality_score(results)

        assert diversity == 0.1  # DEFAULT_FAILURE_SCORE
        assert quality == 0.1  # DEFAULT_FAILURE_SCORE

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
                    timestamp=datetime.now().isoformat(),
                )
                for i in range(10)
            ]

            # Evaluate population
            limited_config.parallel_evaluation = True
            await self.fitness_evaluator.evaluate_population(many_ideas, limited_config)

            # Verify concurrency was limited
            assert max_concurrent <= limited_config.max_parallel_evaluations
