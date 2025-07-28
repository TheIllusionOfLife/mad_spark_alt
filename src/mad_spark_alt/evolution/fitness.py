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
        """Evaluate population in parallel with batch optimization."""
        # Use batch evaluation for better performance
        batch_size = min(5, config.max_parallel_evaluations)  # Batch up to 5 at a time
        
        # Extract hypotheses from ideas
        hypotheses = [idea.content for idea in population]
        
        # Perform batch evaluation
        evaluations = await self.unified_evaluator.evaluate_multiple(
            hypotheses=hypotheses,
            context=context or population[0].generation_prompt if population else "",
            core_question=population[0].metadata.get("core_question") if population else None,
            parallel=True,
            batch_size=batch_size
        )
        
        # Convert evaluations to fitness results
        fitness_results: List[IndividualFitness] = []
        for idea, evaluation in zip(population, evaluations):
            try:
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
                    overall_fitness=evaluation.overall_score,
                    evaluation_metadata={
                        "unified_scores": evaluation.scores,
                        "unified_explanations": evaluation.explanations,
                        "llm_cost": evaluation.metadata.get("llm_cost", 0.0),
                        "batch_evaluation": evaluation.metadata.get("batch_evaluation", False),
                        "evaluation_criteria": {
                            "novelty": evaluation.scores.get("novelty", 0.0),
                            "impact": evaluation.scores.get("impact", 0.0),
                            "cost": evaluation.scores.get("cost", 0.0),
                            "feasibility": evaluation.scores.get("feasibility", 0.0),
                            "risks": evaluation.scores.get("risks", 0.0),
                        },
                    },
                )
                fitness_results.append(fitness)
            except Exception as e:
                logger.error(f"Error converting evaluation to fitness: {e}")
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

    async def _evaluate_sequential(
        self,
        population: List[GeneratedIdea],
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """Evaluate population sequentially with batch optimization."""
        # Even in sequential mode, we can still batch the LLM calls
        batch_size = 5  # Process 5 at a time for efficiency
        
        # Extract hypotheses from ideas
        hypotheses = [idea.content for idea in population]
        
        # Perform batch evaluation
        evaluations = await self.unified_evaluator.evaluate_multiple(
            hypotheses=hypotheses,
            context=context or population[0].generation_prompt if population else "",
            core_question=population[0].metadata.get("core_question") if population else None,
            parallel=False,  # Sequential processing
            batch_size=batch_size
        )
        
        # Convert evaluations to fitness results
        fitness_results: List[IndividualFitness] = []
        for idea, evaluation in zip(population, evaluations):
            try:
                fitness = IndividualFitness(
                    idea=idea,
                    creativity_score=evaluation.scores.get("novelty", 0.0),
                    diversity_score=(
                        evaluation.scores.get("novelty", 0.0)
                        + evaluation.scores.get("impact", 0.0)
                    )
                    / 2,
                    quality_score=evaluation.scores.get("feasibility", 0.0),
                    overall_fitness=evaluation.overall_score,
                    evaluation_metadata={
                        "unified_scores": evaluation.scores,
                        "unified_explanations": evaluation.explanations,
                        "llm_cost": evaluation.metadata.get("llm_cost", 0.0),
                        "batch_evaluation": evaluation.metadata.get("batch_evaluation", False),
                        "evaluation_criteria": evaluation.scores,
                    },
                )
                fitness_results.append(fitness)
            except Exception as e:
                logger.error(f"Error converting evaluation to fitness: {e}")
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
