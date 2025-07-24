"""
Fitness evaluation for genetic evolution.

This module uses the unified evaluation system to provide consistent
fitness scoring for ideas using the 5-criteria system.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.unified_evaluator import HypothesisEvaluation, UnifiedEvaluator
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    IndividualFitness,
)

logger = logging.getLogger(__name__)

# Default score for failed evaluations or missing metrics
DEFAULT_FAILURE_SCORE = 0.1  # Penalty score rather than neutral 0.5


class FitnessEvaluator:
    """
    Evaluates fitness of ideas using the unified 5-criteria evaluation system.

    This class uses the same evaluation criteria as the QADI deduction phase:
    novelty, impact, cost, feasibility, and risks.
    """

    def __init__(self, unified_evaluator: Optional[UnifiedEvaluator] = None):
        """
        Initialize fitness evaluator.

        Args:
            unified_evaluator: Optional custom evaluator. If None, creates default.
        """
        self.unified_evaluator = unified_evaluator or UnifiedEvaluator()

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
            # Check if result is an exception (more robust for Python 3.13)
            is_exception = isinstance(result, BaseException)

            if is_exception:
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
        Evaluate fitness of a single idea using the unified 5-criteria system.

        Args:
            idea: Idea to evaluate
            config: Evolution configuration (weights are fixed by unified system)
            context: Optional context for evaluation

        Returns:
            IndividualFitness with calculated scores
        """
        try:
            # Use unified evaluator to get consistent scoring
            evaluation = await self.unified_evaluator.evaluate_hypothesis(
                hypothesis=idea.content,
                context=context or idea.generation_prompt,
                core_question=idea.metadata.get("core_question"),
                temperature=0.3,  # Low temperature for consistent evaluation
            )

            # Create fitness object with 5-criteria scores
            fitness = IndividualFitness(
                idea=idea,
                # Map unified scores to evolution fitness fields
                creativity_score=evaluation.scores.get("novelty", 0.0),
                diversity_score=(
                    evaluation.scores.get("novelty", 0.0)
                    + evaluation.scores.get("impact", 0.0)
                )
                / 2,  # Combined metric
                quality_score=evaluation.scores.get("feasibility", 0.0),
                overall_fitness=evaluation.overall_score,  # Already calculated by unified system
                evaluation_metadata={
                    "unified_scores": evaluation.scores,
                    "unified_explanations": evaluation.explanations,
                    "llm_cost": evaluation.metadata.get("llm_cost", 0.0),
                    "evaluation_criteria": {
                        "novelty": evaluation.scores.get("novelty", 0.0),
                        "impact": evaluation.scores.get("impact", 0.0),
                        "cost": evaluation.scores.get("cost", 0.0),
                        "feasibility": evaluation.scores.get("feasibility", 0.0),
                        "risks": evaluation.scores.get("risks", 0.0),
                    },
                },
            )

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

    async def calculate_population_diversity(
        self, population: List[IndividualFitness]
    ) -> float:
        """
        Calculate diversity score for entire population using content similarity.

        This helps maintain genetic diversity and avoid premature convergence.

        Args:
            population: List of evaluated individuals

        Returns:
            Population diversity score (0-1), where 0 = identical, 1 = completely diverse
        """
        if len(population) < 2:
            return 1.0

        try:
            # Calculate diversity using content similarity instead of novelty scores
            # This is more reliable than depending on LLM evaluation metadata
            total_similarity = 0.0
            comparisons = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    idea1 = population[i].idea.content.lower().strip()
                    idea2 = population[j].idea.content.lower().strip()
                    
                    # Simple similarity based on shared words
                    words1 = set(idea1.split())
                    words2 = set(idea2.split())
                    
                    if len(words1) == 0 and len(words2) == 0:
                        similarity = 1.0  # Both empty
                    elif len(words1) == 0 or len(words2) == 0:
                        similarity = 0.0  # One empty
                    else:
                        # Jaccard similarity
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    total_similarity += similarity
                    comparisons += 1
            
            if comparisons > 0:
                avg_similarity = total_similarity / comparisons
                # Diversity is inverse of similarity
                diversity = 1.0 - avg_similarity
                return max(0.0, min(1.0, diversity))

            return DEFAULT_FAILURE_SCORE

        except Exception as e:
            logger.error(f"Failed to calculate population diversity: {e}")
            return DEFAULT_FAILURE_SCORE
