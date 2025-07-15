"""
LLM-powered genetic operators for evolution system.

This module provides intelligent genetic operators that use
LLM reasoning instead of simple text manipulation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMProviderInterface
from mad_spark_alt.evolution.interfaces import (
    CrossoverInterface,
    MutationInterface,
    IndividualFitness,
)
from mad_spark_alt.evolution.operators import CrossoverOperator, MutationOperator

logger = logging.getLogger(__name__)


@dataclass
class LLMOperatorResult:
    """Result from LLM operator."""

    content: str
    reasoning: str
    metadata: Dict[str, Any]


class LLMCrossoverOperator(CrossoverInterface):
    """
    Intelligent crossover using LLM reasoning.

    Instead of simple text manipulation, this operator uses
    an LLM to intelligently combine parent ideas.
    """

    def __init__(
        self,
        llm_provider: LLMProviderInterface,
        fallback_to_traditional: bool = True,
    ):
        """
        Initialize LLM crossover operator.

        Args:
            llm_provider: LLM provider for generation
            fallback_to_traditional: Use traditional crossover on LLM failure
        """
        self.llm_provider = llm_provider
        self.fallback_to_traditional = fallback_to_traditional
        self._traditional_crossover = CrossoverOperator()
        self._cost_tracker = LLMOperatorCostTracker()

    async def crossover(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        context: Optional[str] = None,
    ) -> Tuple[GeneratedIdea, GeneratedIdea]:
        """
        Perform intelligent crossover using LLM.

        Args:
            parent1: First parent idea
            parent2: Second parent idea
            context: Optional context for crossover

        Returns:
            Tuple of two offspring ideas
        """
        prompt = f"""You are a genetic algorithm expert tasked with creating innovative offspring from two parent ideas.

Parent 1: {parent1.content}
Parent 2: {parent2.content}

Context: {context or "General innovation"}

Please create two offspring ideas that:
1. Inherit the best aspects of both parents
2. Introduce novel combinations not present in either parent  
3. Are more innovative than simple text mixing
4. Maintain practical implementability
5. Explore different combination strategies for each offspring

Return a JSON object with this structure:
{{
    "offspring1": {{
        "content": "The first offspring idea",
        "reasoning": "Brief explanation of how elements were combined"
    }},
    "offspring2": {{
        "content": "The second offspring idea", 
        "reasoning": "Brief explanation of the different combination approach"
    }}
}}"""

        try:
            # Call LLM
            from mad_spark_alt.core.llm_provider import LLMRequest
            
            request = LLMRequest(
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            response = await self.llm_provider.generate(request)
            
            # Parse JSON response
            from mad_spark_alt.core.json_utils import safe_json_parse
            result = safe_json_parse(response.content)

            # Track costs
            if hasattr(response, "usage"):
                tokens = response.usage.get("total_tokens", 0)
                cost = self._estimate_cost(tokens)
                self._cost_tracker.track_crossover(cost, tokens)

            # Create offspring ideas
            offspring1 = GeneratedIdea(
                content=result["offspring1"]["content"],
                thinking_method=parent1.thinking_method,  # Inherit from parent1
                agent_name="LLMCrossover",
                generation_prompt=f"LLM crossover of: {parent1.content[:30]}... + {parent2.content[:30]}...",
                metadata={
                    "crossover_reasoning": result["offspring1"]["reasoning"],
                    "parent1": parent1.content[:50] + "...",
                    "parent2": parent2.content[:50] + "...",
                    "llm_generated": True,
                },
            )

            offspring2 = GeneratedIdea(
                content=result["offspring2"]["content"],
                thinking_method=parent2.thinking_method,  # Inherit from parent2
                agent_name="LLMCrossover",
                generation_prompt=f"LLM crossover of: {parent1.content[:30]}... + {parent2.content[:30]}...",
                metadata={
                    "crossover_reasoning": result["offspring2"]["reasoning"],
                    "parent1": parent1.content[:50] + "...",
                    "parent2": parent2.content[:50] + "...",
                    "llm_generated": True,
                },
            )

            return offspring1, offspring2

        except Exception as e:
            logger.error(f"LLM crossover failed: {e}")

            if self.fallback_to_traditional:
                logger.info("Falling back to traditional crossover")
                offspring1, offspring2 = await self._traditional_crossover.crossover(
                    parent1, parent2
                )
                # Mark as fallback
                offspring1.metadata["fallback_used"] = True
                offspring2.metadata["fallback_used"] = True
                return offspring1, offspring2
            else:
                raise

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Rough estimate - adjust based on actual model pricing
        cost_per_1k_tokens = 0.03  # Example for GPT-4
        return (tokens / 1000) * cost_per_1k_tokens


class LLMMutationOperator(MutationInterface):
    """
    Intelligent mutation using LLM reasoning.

    Uses an LLM to apply semantically meaningful mutations
    rather than random text changes.
    """

    def __init__(
        self,
        llm_provider: LLMProviderInterface,
        fallback_to_traditional: bool = True,
    ):
        """
        Initialize LLM mutation operator.

        Args:
            llm_provider: LLM provider for generation
            fallback_to_traditional: Use traditional mutation on LLM failure
        """
        self.llm_provider = llm_provider
        self.fallback_to_traditional = fallback_to_traditional
        self._traditional_mutation = MutationOperator()
        self._cost_tracker = LLMOperatorCostTracker()

    async def mutate(
        self,
        individual: GeneratedIdea,
        mutation_rate: float = 0.1,
        context: Optional[str] = None,
    ) -> GeneratedIdea:
        """
        Perform intelligent mutation using LLM.

        Args:
            individual: Idea to mutate
            mutation_rate: Mutation intensity (0-1)

        Returns:
            Mutated idea
        """
        # Map mutation rate to mutation strategies
        if mutation_rate < 0.3:
            mutation_type = "minor refinement"
        elif mutation_rate < 0.7:
            mutation_type = "moderate transformation"
        else:
            mutation_type = "radical reimagining"

        prompt = f"""You are a genetic algorithm expert tasked with mutating an idea to create innovation.

Original Idea: {individual.content}
Mutation Type: {mutation_type}
Mutation Rate: {mutation_rate}

Please create a mutated version that:
1. Applies a {mutation_type} to the original idea
2. Introduces creative changes while maintaining core viability
3. Explores new dimensions or perspectives
4. Is different enough to expand the solution space

Mutation strategies to consider:
- Technology shift (use different tech/approach)
- Scale change (make it bigger/smaller/different scope)
- Domain transfer (apply to different field)
- Constraint addition/removal
- Perspective flip (invert assumptions)

Return a JSON object with this structure:
{{
    "mutated_idea": {{
        "content": "The mutated idea",
        "mutation_type": "Type of mutation applied",
        "reasoning": "Brief explanation of the mutation"
    }}
}}"""

        try:
            # Call LLM
            from mad_spark_alt.core.llm_provider import LLMRequest
            
            request = LLMRequest(
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            response = await self.llm_provider.generate(request)
            
            # Parse JSON response
            from mad_spark_alt.core.json_utils import safe_json_parse
            result = safe_json_parse(response.content)

            # Track costs
            if hasattr(response, "usage"):
                tokens = response.usage.get("total_tokens", 0)
                cost = self._estimate_cost(tokens)
                self._cost_tracker.track_mutation(cost, tokens)

            # Create mutated idea
            mutated = GeneratedIdea(
                content=result["mutated_idea"]["content"],
                thinking_method=individual.thinking_method,
                agent_name="LLMMutation",
                generation_prompt=f"LLM mutation ({mutation_type}) of: {individual.content[:50]}...",
                metadata={
                    "mutation_type": result["mutated_idea"]["mutation_type"],
                    "mutation_reasoning": result["mutated_idea"]["reasoning"],
                    "mutation_rate": mutation_rate,
                    "original": individual.content[:50] + "...",
                    "llm_generated": True,
                },
            )

            return mutated

        except Exception as e:
            logger.error(f"LLM mutation failed: {e}")

            if self.fallback_to_traditional:
                logger.info("Falling back to traditional mutation")
                mutated = await self._traditional_mutation.mutate(individual, mutation_rate, context)
                mutated.metadata["fallback_used"] = True
                return mutated
            else:
                raise

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage."""
        cost_per_1k_tokens = 0.03  # Example for GPT-4
        return (tokens / 1000) * cost_per_1k_tokens


class LLMSelectionAdvisor:
    """
    LLM-powered selection advisor.

    Provides intelligent selection recommendations based on
    semantic analysis of idea quality and potential.
    """

    def __init__(self, llm_provider: LLMProviderInterface):
        """
        Initialize selection advisor.

        Args:
            llm_provider: LLM provider for analysis
        """
        self.llm_provider = llm_provider
        self._cost_tracker = LLMOperatorCostTracker()

    async def advise_selection(
        self,
        population: List[IndividualFitness],
        num_parents: int,
        context: Optional[str] = None,
    ) -> Dict:
        """
        Provide selection advice using LLM analysis.

        Args:
            population: Current population with fitness scores
            num_parents: Number of parents to select
            context: Optional selection context

        Returns:
            Selection advice with recommendations
        """
        # Prepare population summary
        population_summary = []
        for i, individual in enumerate(population):
            population_summary.append(
                f"{i}. {individual.idea.content[:100]}... (fitness: {individual.overall_fitness:.3f})"
            )

        prompt = f"""You are an expert in evolutionary algorithms tasked with selecting the best parents for the next generation.

Population (showing first 100 chars of each idea):
{chr(10).join(population_summary)}

Context: {context or "General innovation"}
Number of parents to select: {num_parents}

Please analyze each idea for:
1. Innovation potential
2. Practical feasibility  
3. Uniqueness in the population
4. Ability to produce good offspring

Return a JSON object with this structure:
{{
    "selection_scores": [
        {{"index": 0, "score": 0.9, "reasoning": "Why this is a good parent"}},
        ...
    ],
    "recommended_parents": [0, 1, ...],  // Indices of top {num_parents} parents
    "diversity_consideration": "Note about maintaining population diversity"
}}"""

        try:
            # Call LLM
            from mad_spark_alt.core.llm_provider import LLMRequest
            
            request = LLMRequest(
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            response = await self.llm_provider.generate(request)
            
            # Parse JSON response
            from mad_spark_alt.core.json_utils import safe_json_parse
            result = safe_json_parse(response.content)

            # Track costs
            if hasattr(response, "usage"):
                tokens = response.usage.get("total_tokens", 0)
                cost = self._estimate_cost(tokens)
                self._cost_tracker.track_selection(cost, tokens)

            return result

        except Exception as e:
            logger.error(f"LLM selection advice failed: {e}")
            # Return fitness-based selection as fallback
            sorted_indices = sorted(
                range(len(population)),
                key=lambda i: population[i].overall_fitness,
                reverse=True,
            )
            return {
                "selection_scores": [
                    {
                        "index": i,
                        "score": population[i].overall_fitness,
                        "reasoning": "Fitness-based selection (fallback)",
                    }
                    for i in range(len(population))
                ],
                "recommended_parents": sorted_indices[:num_parents],
                "diversity_consideration": "Fallback to fitness-based selection",
            }

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token usage."""
        cost_per_1k_tokens = 0.03  # Example for GPT-4
        return (tokens / 1000) * cost_per_1k_tokens


class LLMOperatorCostTracker:
    """Track costs and usage for LLM operators."""

    def __init__(self) -> None:
        """Initialize cost tracker."""
        self._crossover_count = 0
        self._crossover_tokens = 0
        self._crossover_cost = 0.0

        self._mutation_count = 0
        self._mutation_tokens = 0
        self._mutation_cost = 0.0

        self._selection_count = 0
        self._selection_tokens = 0
        self._selection_cost = 0.0

    def track_crossover(self, cost: float, tokens: int) -> None:
        """Track crossover operation."""
        self._crossover_count += 1
        self._crossover_tokens += tokens
        self._crossover_cost += cost

    def track_mutation(self, cost: float, tokens: int) -> None:
        """Track mutation operation."""
        self._mutation_count += 1
        self._mutation_tokens += tokens
        self._mutation_cost += cost

    def track_selection(self, cost: float, tokens: int) -> None:
        """Track selection operation."""
        self._selection_count += 1
        self._selection_tokens += tokens
        self._selection_cost += cost

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_cost": self._crossover_cost
            + self._mutation_cost
            + self._selection_cost,
            "total_tokens": self._crossover_tokens
            + self._mutation_tokens
            + self._selection_tokens,
            "crossover_count": self._crossover_count,
            "crossover_tokens": self._crossover_tokens,
            "crossover_cost": self._crossover_cost,
            "mutation_count": self._mutation_count,
            "mutation_tokens": self._mutation_tokens,
            "mutation_cost": self._mutation_cost,
            "selection_count": self._selection_count,
            "selection_tokens": self._selection_tokens,
            "selection_cost": self._selection_cost,
        }
