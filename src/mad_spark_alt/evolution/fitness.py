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
from mad_spark_alt.core.llm_provider import LLMProviderInterface
from mad_spark_alt.evolution.interfaces import (
    DiversityMethod,
    EvolutionConfig,
    IndividualFitness,
)
from mad_spark_alt.evolution.diversity_calculator import DiversityCalculator
from mad_spark_alt.evolution.jaccard_diversity import JaccardDiversityCalculator
from mad_spark_alt.evolution.gemini_diversity import GeminiDiversityCalculator

logger = logging.getLogger(__name__)

# Default score for failed evaluations or missing metrics
DEFAULT_FAILURE_SCORE = 0.1  # Penalty score rather than neutral 0.5


def create_diversity_calculator(
    diversity_method: DiversityMethod,
    llm_provider: Optional[LLMProviderInterface] = None
) -> DiversityCalculator:
    """
    Create appropriate diversity calculator based on method.
    
    Args:
        diversity_method: The diversity calculation method to use
        llm_provider: LLM provider for semantic method (required for SEMANTIC)
        
    Returns:
        DiversityCalculator instance
        
    Raises:
        ValueError: If SEMANTIC method requested without LLM provider
    """
    if diversity_method == DiversityMethod.JACCARD:
        return JaccardDiversityCalculator()
    elif diversity_method == DiversityMethod.SEMANTIC:
        if llm_provider is None:
            raise ValueError("LLM provider required for semantic diversity calculation")
        return GeminiDiversityCalculator(llm_provider=llm_provider)
    else:
        raise ValueError(f"Unknown diversity method: {diversity_method}")


class FitnessEvaluator:
    """
    Evaluates fitness of ideas using the unified 5-criteria evaluation system.

    This class uses the same evaluation criteria as the QADI deduction phase:
    novelty, impact, cost, feasibility, and risks.
    """

    def __init__(
        self, 
        unified_evaluator: Optional[UnifiedEvaluator] = None,
        diversity_calculator: Optional[DiversityCalculator] = None,
        fallback_diversity_calculator: Optional[DiversityCalculator] = None
    ):
        """
        Initialize fitness evaluator.

        Args:
            unified_evaluator: Optional custom evaluator. If None, creates default.
            diversity_calculator: Optional diversity calculator. If None, uses JaccardDiversityCalculator.
            fallback_diversity_calculator: Optional fallback calculator if primary fails.
        """
        self.unified_evaluator = unified_evaluator or UnifiedEvaluator()
        self.diversity_calculator = diversity_calculator or JaccardDiversityCalculator()
        self.fallback_diversity_calculator = fallback_diversity_calculator

    def configure_diversity_method(
        self, 
        diversity_method: DiversityMethod, 
        llm_provider: Optional[LLMProviderInterface] = None
    ) -> None:
        """
        Configure diversity calculation method.
        
        Args:
            diversity_method: The diversity calculation method to use
            llm_provider: LLM provider for semantic method (required for SEMANTIC)
        """
        # Create primary calculator
        self.diversity_calculator = create_diversity_calculator(diversity_method, llm_provider)
        
        # Create fallback calculator (always Jaccard for reliability)
        if diversity_method != DiversityMethod.JACCARD:
            self.fallback_diversity_calculator = JaccardDiversityCalculator()
        else:
            self.fallback_diversity_calculator = None

    def _convert_evaluation_to_fitness(
        self, idea: GeneratedIdea, evaluation: HypothesisEvaluation
    ) -> IndividualFitness:
        """Convert unified evaluation to individual fitness.
        
        This helper method reduces code duplication between parallel and sequential evaluation.
        
        Args:
            idea: The idea being evaluated
            evaluation: The evaluation results from unified evaluator
            
        Returns:
            IndividualFitness object with scores and metadata
        """
        return IndividualFitness(
            idea=idea,
            # Use QADI scoring criteria directly
            impact=evaluation.scores.get("impact", 0.0),
            feasibility=evaluation.scores.get("feasibility", 0.0),
            accessibility=evaluation.scores.get("accessibility", 0.0),
            sustainability=evaluation.scores.get("sustainability", 0.0),
            scalability=evaluation.scores.get("scalability", 0.0),
            overall_fitness=evaluation.overall_score,
            evaluation_metadata={
                "unified_scores": evaluation.scores,
                "unified_explanations": evaluation.explanations,
                "llm_cost": evaluation.metadata.get("llm_cost", 0.0),
                "batch_evaluation": evaluation.metadata.get("batch_evaluation", False),
                "evaluation_criteria": {
                    "impact": evaluation.scores.get("impact", 0.0),
                    "feasibility": evaluation.scores.get("feasibility", 0.0),
                    "accessibility": evaluation.scores.get("accessibility", 0.0),
                    "sustainability": evaluation.scores.get("sustainability", 0.0),
                    "scalability": evaluation.scores.get("scalability", 0.0),
                },
            },
        )

    def _create_error_fitness(self, idea: GeneratedIdea, error: Exception) -> IndividualFitness:
        """Create a fitness object for failed evaluations.
        
        Args:
            idea: The idea that failed evaluation
            error: The exception that occurred
            
        Returns:
            IndividualFitness with zero scores and error metadata
        """
        logger.error(f"Error converting evaluation to fitness: {error}")
        return IndividualFitness(
            idea=idea,
            impact=0.0,
            feasibility=0.0,
            accessibility=0.0,
            sustainability=0.0,
            scalability=0.0,
            overall_fitness=0.0,
            evaluation_metadata={"error": str(error)},
        )

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
                fitness = self._convert_evaluation_to_fitness(idea, evaluation)
                fitness_results.append(fitness)
            except Exception as e:
                fitness_results.append(self._create_error_fitness(idea, e))
        
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
                fitness = self._convert_evaluation_to_fitness(idea, evaluation)
                fitness_results.append(fitness)
            except Exception as e:
                fitness_results.append(self._create_error_fitness(idea, e))
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

            # Create fitness object with QADI scoring criteria
            return self._convert_evaluation_to_fitness(idea, evaluation)

        except Exception as e:
            return self._create_error_fitness(idea, e)

    async def calculate_population_diversity(
        self, population: List[IndividualFitness]
    ) -> float:
        """
        Calculate diversity score for a population using the configured calculator.

        This helps maintain genetic diversity and avoid premature convergence.

        Args:
            population: List of evaluated individuals

        Returns:
            Population diversity score (0-1), where 0 = identical, 1 = completely diverse
        """
        try:
            # Try primary calculator
            diversity = await self.diversity_calculator.calculate_diversity(population)
            return max(0.0, min(1.0, diversity))
            
        except Exception as e:
            logger.warning(f"Primary diversity calculator failed: {e}")
            
            # Try fallback if available
            if self.fallback_diversity_calculator:
                try:
                    diversity = await self.fallback_diversity_calculator.calculate_diversity(population)
                    logger.info("Using fallback diversity calculator")
                    return max(0.0, min(1.0, diversity))
                except Exception as fallback_e:
                    logger.error(f"Fallback diversity calculator also failed: {fallback_e}")
            
            # If all fails, return default
            logger.error("All diversity calculators failed, using default score")
            return DEFAULT_FAILURE_SCORE
