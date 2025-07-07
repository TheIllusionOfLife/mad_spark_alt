"""
Fitness evaluation for genetic evolution.

This module integrates with the existing creativity evaluation infrastructure
to provide fitness scoring for ideas in the genetic algorithm.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from mad_spark_alt.core.evaluator import CreativityEvaluator
from mad_spark_alt.core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    GeneratedIdea,
    ModelOutput,
    OutputType,
)
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    IndividualFitness,
)

logger = logging.getLogger(__name__)

# Default score for failed evaluations or missing metrics
DEFAULT_FAILURE_SCORE = 0.1  # Penalty score rather than neutral 0.5


class FitnessEvaluator:
    """
    Evaluates fitness of ideas using the existing creativity evaluation system.

    This class bridges the genetic algorithm with the multi-layer evaluation
    infrastructure, converting creativity scores into fitness values.
    """

    def __init__(self, creativity_evaluator: Optional[CreativityEvaluator] = None):
        """
        Initialize fitness evaluator.

        Args:
            creativity_evaluator: Optional custom evaluator. If None, creates default.
        """
        self.creativity_evaluator = creativity_evaluator or CreativityEvaluator()

    async def evaluate_population(
        self,
        population: List[GeneratedIdea],
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """
        Evaluate fitness for entire population.

        Args:
            population: List of ideas to evaluate
            config: Evolution configuration with fitness weights
            context: Optional context for evaluation

        Returns:
            List of IndividualFitness objects with scores
        """
        if config.parallel_evaluation:
            return await self._evaluate_parallel(population, config, context)
        else:
            return await self._evaluate_sequential(population, config, context)

    async def _evaluate_parallel(
        self,
        population: List[GeneratedIdea],
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """Evaluate population in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(config.max_parallel_evaluations)

        async def evaluate_with_semaphore(idea: GeneratedIdea) -> IndividualFitness:
            async with semaphore:
                return await self.evaluate_individual(idea, config, context)

        tasks = [evaluate_with_semaphore(idea) for idea in population]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        fitness_results: List[IndividualFitness] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating idea {i}: {result}")
                # Create default fitness for failed evaluations
                fitness_results.append(
                    IndividualFitness(
                        idea=population[i],
                        creativity_score=0.0,
                        diversity_score=0.0,
                        quality_score=0.0,
                        overall_fitness=0.0,
                        evaluation_metadata={"error": str(result)},
                    )
                )
            elif isinstance(result, IndividualFitness):
                fitness_results.append(result)

        return fitness_results

    async def _evaluate_sequential(
        self,
        population: List[GeneratedIdea],
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """Evaluate population sequentially."""
        fitness_results = []
        for idea in population:
            try:
                fitness = await self.evaluate_individual(idea, config, context)
                fitness_results.append(fitness)
            except Exception as e:
                logger.error(f"Error evaluating idea: {e}")
                fitness_results.append(
                    IndividualFitness(
                        idea=idea,
                        creativity_score=0.0,
                        diversity_score=0.0,
                        quality_score=0.0,
                        overall_fitness=0.0,
                        evaluation_metadata={"error": str(e)},
                    )
                )
        return fitness_results

    async def evaluate_individual(
        self,
        idea: GeneratedIdea,
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> IndividualFitness:
        """
        Evaluate fitness of a single idea.

        Args:
            idea: Idea to evaluate
            config: Evolution configuration with fitness weights
            context: Optional context for evaluation

        Returns:
            IndividualFitness with calculated scores
        """
        # Convert GeneratedIdea to ModelOutput for evaluation
        model_output = ModelOutput(
            content=idea.content,
            output_type=OutputType.TEXT,
            model_name=f"{idea.agent_name}_{idea.thinking_method.value}",
            prompt=idea.generation_prompt,
            metadata={
                **idea.metadata,
                "thinking_method": idea.thinking_method.value,
                "confidence_score": idea.confidence_score,
            },
        )

        # Create evaluation request
        eval_request = EvaluationRequest(
            outputs=[model_output],
            evaluation_config={
                "context": context,
                "include_diversity": True,
                "include_quality": True,
                "include_llm_judge": False,  # Skip LLM judge for efficiency
            },
        )

        # Run evaluation
        try:
            summary = await self.creativity_evaluator.evaluate(eval_request)

            # Extract scores from evaluation summary
            creativity_score = summary.get_overall_creativity_score() or 0.0

            # Get component scores from layer results
            diversity_score = self._extract_diversity_score(summary.layer_results)
            quality_score = self._extract_quality_score(summary.layer_results)

            # Create fitness object
            fitness = IndividualFitness(
                idea=idea,
                creativity_score=creativity_score,
                diversity_score=diversity_score,
                quality_score=quality_score,
                evaluation_metadata={
                    "evaluation_time": summary.execution_time,
                    "total_evaluators": summary.total_evaluators,
                },
            )

            # Calculate overall fitness with configured weights
            fitness.calculate_overall_fitness(config.fitness_weights)

            return fitness

        except Exception as e:
            logger.error(f"Failed to evaluate idea: {e}")
            return IndividualFitness(
                idea=idea,
                creativity_score=0.0,
                diversity_score=0.0,
                quality_score=0.0,
                overall_fitness=0.0,
                evaluation_metadata={"error": str(e)},
            )

    def _extract_diversity_score(
        self, layer_results: Dict[EvaluationLayer, List[EvaluationResult]]
    ) -> float:
        """Extract diversity score from evaluation results."""
        quantitative_results = layer_results.get(EvaluationLayer.QUANTITATIVE, [])

        for result in quantitative_results:
            if "diversity" in result.evaluator_name:
                # Average different diversity metrics
                diversity_metrics = [
                    result.scores.get("distinct_1", 0.0),
                    result.scores.get("distinct_2", 0.0),
                    result.scores.get("semantic_uniqueness", 0.0),
                    result.scores.get("lexical_diversity", 0.0),
                ]
                # Include all metrics in average, even 0.0 values
                return (
                    sum(diversity_metrics) / len(diversity_metrics)
                    if diversity_metrics
                    else 0.0
                )
        return DEFAULT_FAILURE_SCORE  # Penalty for missing metrics

    def _extract_quality_score(
        self, layer_results: Dict[EvaluationLayer, List[EvaluationResult]]
    ) -> float:
        """Extract quality score from evaluation results."""
        quantitative_results = layer_results.get(EvaluationLayer.QUANTITATIVE, [])

        for result in quantitative_results:
            if "quality" in result.evaluator_name:
                # Average quality metrics
                quality_metrics = [
                    result.scores.get("fluency_score", 0.0),
                    result.scores.get("grammar_score", 0.0),
                    result.scores.get("readability_score", 0.0),
                    result.scores.get("coherence_score", 0.0),
                ]
                # Include all metrics in average, even 0.0 values
                return (
                    sum(quality_metrics) / len(quality_metrics)
                    if quality_metrics
                    else 0.0
                )
        return DEFAULT_FAILURE_SCORE  # Penalty for missing metrics

    async def calculate_population_diversity(
        self, population: List[IndividualFitness]
    ) -> float:
        """
        Calculate diversity score for entire population.

        This helps maintain genetic diversity and avoid premature convergence.

        Args:
            population: List of evaluated individuals

        Returns:
            Population diversity score (0-1)
        """
        if len(population) < 2:
            return 1.0

        # Create evaluation request for population diversity
        outputs = [
            ModelOutput(
                content=ind.idea.content,
                output_type=OutputType.TEXT,
                model_name=f"evolution_gen_{ind.idea.metadata.get('generation', 0)}",
            )
            for ind in population
        ]

        eval_request = EvaluationRequest(
            outputs=outputs,
            evaluation_config={
                "include_diversity": True,
                "include_quality": False,
                "include_llm_judge": False,
            },
        )

        try:
            summary = await self.creativity_evaluator.evaluate(eval_request)
            # Extract population-level diversity from layer results
            quantitative_results = summary.layer_results.get(
                EvaluationLayer.QUANTITATIVE, []
            )

            for result in quantitative_results:
                if "diversity" in result.evaluator_name:
                    return result.scores.get(
                        "semantic_uniqueness", DEFAULT_FAILURE_SCORE
                    )
            return DEFAULT_FAILURE_SCORE
        except Exception as e:
            logger.error(f"Failed to calculate population diversity: {e}")
            return DEFAULT_FAILURE_SCORE
