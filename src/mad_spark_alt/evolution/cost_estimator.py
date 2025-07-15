"""
Cost estimation system for evolution.

This module provides cost prediction and tracking for LLM-based
genetic evolution operations.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from mad_spark_alt.evolution.constants import (
    AGGRESSIVE_POPULATION_SCALE_FACTOR,
    BUDGET_CRITICAL_THRESHOLD,
    BUDGET_EMERGENCY_THRESHOLD,
    BUDGET_HIGH_USAGE_THRESHOLD,
    BUDGET_MODERATE_USAGE_THRESHOLD,
    CLAUDE3_OPUS_INPUT_COST,
    CLAUDE3_OPUS_OUTPUT_COST,
    CLAUDE3_SONNET_COST_REDUCTION,
    CLAUDE3_SONNET_INPUT_COST,
    CLAUDE3_SONNET_OUTPUT_COST,
    CONFIDENCE_INTERVAL_LOWER,
    CONFIDENCE_INTERVAL_UPPER,
    CROSSOVER_OPERATION_FRACTION,
    CROSSOVER_TOKEN_ESTIMATE,
    DEFAULT_CACHE_HIT_RATE,
    DEFAULT_FALLBACK_COST,
    GEMINI_PRO_COST_REDUCTION,
    GEMINI_PRO_INPUT_COST,
    GEMINI_PRO_OUTPUT_COST,
    GPT4_INPUT_COST,
    GPT4_OUTPUT_COST,
    GPT4_TURBO_INPUT_COST,
    GPT4_TURBO_OUTPUT_COST,
    GPT35_TURBO_COST_REDUCTION,
    GPT35_TURBO_INPUT_COST,
    GPT35_TURBO_OUTPUT_COST,
    INPUT_TOKEN_RATIO,
    MIN_POPULATION_SIZE_AGGRESSIVE,
    MIN_POPULATION_SIZE_MODERATE,
    MODERATE_POPULATION_SCALE_FACTOR,
    MUTATION_TOKEN_ESTIMATE,
    OPERATOR_INPUT_RATIO,
    OPERATOR_OUTPUT_RATIO,
    OUTPUT_TOKEN_RATIO,
    SELECTION_TOKEN_ESTIMATE,
)
from mad_spark_alt.evolution.interfaces import EvolutionConfig


@dataclass
class ModelCosts:
    """Cost structure for a specific model."""

    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k_tokens
        return input_cost + output_cost


class EvolutionCostEstimator:
    """
    Estimates and tracks costs for evolution operations.

    This estimator predicts costs based on configuration and
    historical data about token usage patterns.
    """

    # Default model costs (as of 2024)
    DEFAULT_MODEL_COSTS = {
        "gpt-4": ModelCosts(
            input_cost_per_1k_tokens=GPT4_INPUT_COST,
            output_cost_per_1k_tokens=GPT4_OUTPUT_COST,
        ),
        "gpt-4-turbo": ModelCosts(
            input_cost_per_1k_tokens=GPT4_TURBO_INPUT_COST,
            output_cost_per_1k_tokens=GPT4_TURBO_OUTPUT_COST,
        ),
        "gpt-3.5-turbo": ModelCosts(
            input_cost_per_1k_tokens=GPT35_TURBO_INPUT_COST,
            output_cost_per_1k_tokens=GPT35_TURBO_OUTPUT_COST,
        ),
        "claude-3-opus": ModelCosts(
            input_cost_per_1k_tokens=CLAUDE3_OPUS_INPUT_COST,
            output_cost_per_1k_tokens=CLAUDE3_OPUS_OUTPUT_COST,
        ),
        "claude-3-sonnet": ModelCosts(
            input_cost_per_1k_tokens=CLAUDE3_SONNET_INPUT_COST,
            output_cost_per_1k_tokens=CLAUDE3_SONNET_OUTPUT_COST,
        ),
        "gemini-pro": ModelCosts(
            input_cost_per_1k_tokens=GEMINI_PRO_INPUT_COST,
            output_cost_per_1k_tokens=GEMINI_PRO_OUTPUT_COST,
        ),
    }

    def __init__(self) -> None:
        """Initialize cost estimator."""
        self._model_costs = self.DEFAULT_MODEL_COSTS.copy()
        self._historical_token_usage: List[Dict] = []

    def set_model_costs(self, model_costs: Dict[str, Dict[str, float]]) -> None:
        """
        Set custom model costs.

        Args:
            model_costs: Dictionary mapping model names to cost info
                        {'model': {'input': cost, 'output': cost}}
        """
        for model, costs in model_costs.items():
            self._model_costs[model] = ModelCosts(
                input_cost_per_1k_tokens=costs["input"],
                output_cost_per_1k_tokens=costs["output"],
            )

    def estimate_evolution_cost(
        self,
        config: EvolutionConfig,
        model: str = "gpt-4",
        avg_tokens_per_evaluation: int = 1000,
        cache_hit_rate: float = DEFAULT_CACHE_HIT_RATE,
        enable_llm_operators: bool = False,
    ) -> Dict[str, Any]:
        """
        Estimate total cost for an evolution run.

        Args:
            config: Evolution configuration
            model: LLM model to use
            avg_tokens_per_evaluation: Average tokens per fitness evaluation
            cache_hit_rate: Expected cache hit rate (0-1)
            enable_llm_operators: Whether LLM operators will be used

        Returns:
            Cost estimate with breakdown and confidence interval
        """
        if model not in self._model_costs:
            raise ValueError(f"Unknown model: {model}")

        model_cost = self._model_costs[model]

        # Calculate base evaluations
        total_evaluations = config.population_size * config.generations
        cached_evaluations = int(total_evaluations * cache_hit_rate)
        actual_evaluations = total_evaluations - cached_evaluations

        # Estimate tokens for evaluations
        eval_input_tokens = actual_evaluations * (
            avg_tokens_per_evaluation * INPUT_TOKEN_RATIO
        )
        eval_output_tokens = actual_evaluations * (
            avg_tokens_per_evaluation * OUTPUT_TOKEN_RATIO
        )

        # Calculate evaluation costs
        evaluation_cost = model_cost.calculate_cost(
            int(eval_input_tokens), int(eval_output_tokens)
        )

        # Estimate operator costs if enabled
        operator_cost = 0.0
        operator_tokens = 0

        if enable_llm_operators:
            # Crossover operations
            crossover_ops = int(
                config.generations
                * config.population_size
                * config.crossover_rate
                * CROSSOVER_OPERATION_FRACTION
            )
            crossover_tokens = crossover_ops * CROSSOVER_TOKEN_ESTIMATE

            # Mutation operations
            mutation_ops = int(
                config.generations * config.population_size * config.mutation_rate
            )
            mutation_tokens = mutation_ops * MUTATION_TOKEN_ESTIMATE

            # Selection advice
            selection_ops = config.generations
            selection_tokens = selection_ops * SELECTION_TOKEN_ESTIMATE

            operator_tokens = crossover_tokens + mutation_tokens + selection_tokens
            operator_cost = model_cost.calculate_cost(
                int(operator_tokens * OPERATOR_INPUT_RATIO),
                int(operator_tokens * OPERATOR_OUTPUT_RATIO),
            )

        # Total cost
        total_cost = evaluation_cost + operator_cost

        # Calculate confidence interval (±20%)
        confidence_interval = (
            total_cost * CONFIDENCE_INTERVAL_LOWER,
            total_cost * CONFIDENCE_INTERVAL_UPPER,
        )

        return {
            "estimated_cost": total_cost,
            "confidence_interval": confidence_interval,
            "cost_breakdown": {
                "evaluations": evaluation_cost,
                "operators": operator_cost,
            },
            "token_breakdown": {
                "evaluation_tokens": int(eval_input_tokens + eval_output_tokens),
                "operator_tokens": operator_tokens,
                "total_tokens": int(
                    eval_input_tokens + eval_output_tokens + operator_tokens
                ),
            },
            "assumptions": {
                "model": model,
                "avg_tokens_per_evaluation": avg_tokens_per_evaluation,
                "cache_hit_rate": cache_hit_rate,
                "enable_llm_operators": enable_llm_operators,
            },
            "total_evaluations": total_evaluations,
            "cached_evaluations": cached_evaluations,
            "actual_evaluations": actual_evaluations,
        }

    def track_actual_usage(
        self,
        operation_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ) -> None:
        """
        Track actual token usage for improving estimates.

        Args:
            operation_type: Type of operation (evaluation, crossover, etc.)
            model: Model used
            input_tokens: Input tokens consumed
            output_tokens: Output tokens generated
            cost: Actual cost incurred
        """
        self._historical_token_usage.append(
            {
                "operation_type": operation_type,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost,
            }
        )

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from historical usage.

        Returns:
            Usage statistics by operation type
        """
        if not self._historical_token_usage:
            return {"message": "No historical data available"}

        stats = {}
        operation_types = set(u["operation_type"] for u in self._historical_token_usage)

        for op_type in operation_types:
            op_usage = [
                u
                for u in self._historical_token_usage
                if u["operation_type"] == op_type
            ]

            total_tokens = sum(u["total_tokens"] for u in op_usage)
            total_cost = sum(u["cost"] for u in op_usage)

            stats[op_type] = {
                "count": len(op_usage),
                "total_tokens": total_tokens,
                "avg_tokens": total_tokens / len(op_usage) if op_usage else 0,
                "total_cost": total_cost,
                "avg_cost": total_cost / len(op_usage) if op_usage else 0,
            }

        return stats

    def suggest_cost_optimizations(
        self,
        current_config: EvolutionConfig,
        target_budget: float,
        model: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        """
        Suggest configuration changes to meet budget constraints.

        Args:
            current_config: Current evolution configuration
            target_budget: Target budget in dollars
            model: Model being used

        Returns:
            List of optimization suggestions
        """
        current_estimate = self.estimate_evolution_cost(current_config, model)
        current_cost = current_estimate["estimated_cost"]

        suggestions: List[Dict[str, Any]] = []

        if current_cost <= target_budget:
            suggestions.append(
                {
                    "type": "no_change_needed",
                    "message": f"Current estimated cost (${current_cost:.2f}) is within budget",
                }
            )
            return suggestions

        # Calculate required reduction
        reduction_needed = (current_cost - target_budget) / current_cost

        # Suggest cheaper model
        cheaper_models = [
            ("gpt-3.5-turbo", GPT35_TURBO_COST_REDUCTION),
            ("gemini-pro", GEMINI_PRO_COST_REDUCTION),
            ("claude-3-sonnet", CLAUDE3_SONNET_COST_REDUCTION),
        ]

        for alt_model, reduction in cheaper_models:
            if alt_model in self._model_costs:
                alt_estimate = self.estimate_evolution_cost(current_config, alt_model)
                if alt_estimate["estimated_cost"] <= target_budget:
                    suggestions.append(
                        {
                            "type": "change_model",
                            "model": alt_model,
                            "estimated_cost": alt_estimate["estimated_cost"],
                            "savings": current_cost - alt_estimate["estimated_cost"],
                        }
                    )
                    break

        # Suggest reducing generations
        reduced_gens = max(1, int(current_config.generations * (1 - reduction_needed)))
        if reduced_gens < current_config.generations:
            reduced_config = replace(current_config, generations=reduced_gens)
            reduced_estimate = self.estimate_evolution_cost(reduced_config, model)
            suggestions.append(
                {
                    "type": "reduce_generations",
                    "new_generations": reduced_gens,
                    "estimated_cost": reduced_estimate["estimated_cost"],
                    "savings": current_cost - reduced_estimate["estimated_cost"],
                }
            )

        # Suggest reducing population
        reduced_pop = max(
            5, int(current_config.population_size * (1 - reduction_needed))
        )
        if reduced_pop < current_config.population_size:
            reduced_config = replace(current_config, population_size=reduced_pop)
            reduced_estimate = self.estimate_evolution_cost(reduced_config, model)
            suggestions.append(
                {
                    "type": "reduce_population",
                    "new_population": reduced_pop,
                    "estimated_cost": reduced_estimate["estimated_cost"],
                    "savings": current_cost - reduced_estimate["estimated_cost"],
                }
            )

        # Suggest disabling LLM operators
        if current_config.enable_llm_operators:
            no_llm_estimate = self.estimate_evolution_cost(
                current_config, model, enable_llm_operators=False
            )
            suggestions.append(
                {
                    "type": "disable_llm_operators",
                    "estimated_cost": no_llm_estimate["estimated_cost"],
                    "savings": current_cost - no_llm_estimate["estimated_cost"],
                }
            )

        return suggestions

    def calculate_token_cost(
        self,
        tokens: int,
        model: str = "gpt-4",
        assume_equal_input_output: bool = True,
    ) -> float:
        """
        Calculate cost for a given number of tokens.

        This is a centralized cost calculation method that replaces
        the duplicated _estimate_cost methods in LLM operators.

        Args:
            tokens: Total number of tokens
            model: Model name (defaults to GPT-4)
            assume_equal_input_output: If True, assumes equal input/output split

        Returns:
            Estimated cost in dollars
        """
        if model not in self._model_costs:
            # Fall back to GPT-4 pricing if model not found
            model = "gpt-4"

        model_costs = self._model_costs[model]

        if assume_equal_input_output:
            # For simplicity, assume roughly equal input/output split
            input_tokens = tokens // 2
            output_tokens = tokens - input_tokens
            return model_costs.calculate_cost(input_tokens, output_tokens)
        else:
            # Use average of input/output costs
            avg_cost_per_1k = (
                model_costs.input_cost_per_1k_tokens
                + model_costs.output_cost_per_1k_tokens
            ) / 2
            return (tokens / 1000) * avg_cost_per_1k


# Global cost estimator instance for centralized access
_global_cost_estimator = EvolutionCostEstimator()


def estimate_token_cost(
    tokens: int,
    model: str = "gpt-4",
    assume_equal_input_output: bool = True,
) -> float:
    """
    Convenience function for token cost estimation.

    This function provides a simple interface to the centralized
    cost calculation logic, making it easy to replace duplicated
    _estimate_cost methods throughout the codebase.

    Args:
        tokens: Total number of tokens
        model: Model name (defaults to GPT-4)
        assume_equal_input_output: If True, assumes equal input/output split

    Returns:
        Estimated cost in dollars
    """
    return _global_cost_estimator.calculate_token_cost(
        tokens, model, assume_equal_input_output
    )


class DynamicCostManager:
    """
    Dynamic cost management for evolution with budget constraints.

    This class provides real-time cost monitoring and automatic
    parameter adjustment to stay within budget limits.
    """

    def __init__(
        self,
        budget_limit: float,
        cost_estimator: Optional[EvolutionCostEstimator] = None,
    ):
        """
        Initialize dynamic cost manager.

        Args:
            budget_limit: Maximum budget in dollars
            cost_estimator: Cost estimator instance (uses global if None)
        """
        self.budget_limit = budget_limit
        self.cost_estimator = cost_estimator or _global_cost_estimator
        self.accumulated_cost = 0.0
        self.cost_history: List[Dict[str, Any]] = []
        self.emergency_mode = False

    def track_cost(self, operation_cost: float, operation_type: str) -> None:
        """
        Track cost from an operation and update state.

        Args:
            operation_cost: Cost of the operation
            operation_type: Type of operation (e.g., 'fitness_evaluation', 'crossover')
        """
        self.accumulated_cost += operation_cost
        self.cost_history.append(
            {
                "cost": operation_cost,
                "operation_type": operation_type,
                "accumulated_cost": self.accumulated_cost,
                "budget_remaining": self.budget_limit - self.accumulated_cost,
            }
        )

        # Check if we're approaching budget limit
        budget_used_pct = self.accumulated_cost / self.budget_limit
        if budget_used_pct > BUDGET_EMERGENCY_THRESHOLD:
            self.emergency_mode = True

    def can_afford_operation(self, estimated_cost: float) -> bool:
        """
        Check if an operation can be afforded within budget.

        Args:
            estimated_cost: Estimated cost of the operation

        Returns:
            True if operation is affordable
        """
        return (self.accumulated_cost + estimated_cost) <= self.budget_limit

    def get_adaptive_config(self, base_config: EvolutionConfig) -> EvolutionConfig:
        """
        Generate an adapted configuration based on current cost state.

        Args:
            base_config: Base evolution configuration

        Returns:
            Adapted configuration optimized for remaining budget
        """
        remaining_budget = self.budget_limit - self.accumulated_cost
        budget_used_pct = self.accumulated_cost / self.budget_limit

        # Create a copy of the config to modify
        adapted_config = EvolutionConfig(
            population_size=base_config.population_size,
            generations=base_config.generations,
            mutation_rate=base_config.mutation_rate,
            crossover_rate=base_config.crossover_rate,
            elite_size=base_config.elite_size,
            selection_strategy=base_config.selection_strategy,
            parallel_evaluation=base_config.parallel_evaluation,
            max_parallel_evaluations=base_config.max_parallel_evaluations,
            fitness_weights=base_config.fitness_weights,
            random_seed=base_config.random_seed,
            timeout_seconds=base_config.timeout_seconds,
        )

        # Apply budget-based adaptations
        if budget_used_pct > BUDGET_HIGH_USAGE_THRESHOLD:
            # Reduce population size
            adapted_config.population_size = max(
                MIN_POPULATION_SIZE_AGGRESSIVE,
                int(base_config.population_size * AGGRESSIVE_POPULATION_SCALE_FACTOR),
            )
            # Reduce max parallel evaluations to save on batch costs
            adapted_config.max_parallel_evaluations = min(
                adapted_config.max_parallel_evaluations,
                adapted_config.population_size // 2,
            )
        elif budget_used_pct > BUDGET_MODERATE_USAGE_THRESHOLD:
            adapted_config.population_size = max(
                MIN_POPULATION_SIZE_MODERATE,
                int(base_config.population_size * MODERATE_POPULATION_SCALE_FACTOR),
            )
            adapted_config.max_parallel_evaluations = min(
                adapted_config.max_parallel_evaluations, adapted_config.population_size
            )

        return adapted_config

    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget status and recommendations.

        Returns:
            Dictionary with budget status information
        """
        remaining_budget = self.budget_limit - self.accumulated_cost
        budget_used_pct = self.accumulated_cost / self.budget_limit

        status: Dict[str, Any] = {
            "budget_limit": self.budget_limit,
            "accumulated_cost": self.accumulated_cost,
            "remaining_budget": remaining_budget,
            "budget_used_percentage": budget_used_pct,
            "emergency_mode": self.emergency_mode,
            "cost_history_count": len(self.cost_history),
        }

        # Add recommendations based on budget state
        if budget_used_pct > BUDGET_CRITICAL_THRESHOLD:
            status["recommendation"] = "STOP: Budget nearly exhausted"
        elif budget_used_pct > BUDGET_HIGH_USAGE_THRESHOLD:
            status["recommendation"] = (
                "REDUCE: Switch to smaller populations and fewer generations"
            )
        elif budget_used_pct > BUDGET_MODERATE_USAGE_THRESHOLD:
            status["recommendation"] = "MODERATE: Consider reducing parallelism"
        else:
            status["recommendation"] = "CONTINUE: Budget usage is healthy"

        return status

    def estimate_remaining_operations(
        self,
        config: EvolutionConfig,
        avg_cost_per_evaluation: Optional[float] = None,
    ) -> int:
        """
        Estimate how many more fitness evaluations can be performed.

        Args:
            config: Current evolution configuration
            avg_cost_per_evaluation: Average cost per evaluation (estimated if None)

        Returns:
            Estimated number of remaining evaluations possible
        """
        remaining_budget = self.budget_limit - self.accumulated_cost

        if avg_cost_per_evaluation is None:
            # Estimate based on recent cost history
            if len(self.cost_history) >= 3:
                recent_costs = [
                    entry["cost"]
                    for entry in self.cost_history[-3:]
                    if entry["operation_type"] == "fitness_evaluation"
                ]
                avg_cost_per_evaluation = (
                    sum(recent_costs) / len(recent_costs)
                    if recent_costs
                    else DEFAULT_FALLBACK_COST
                )
            else:
                # Default estimate for GPT-4 with 1000 tokens
                avg_cost_per_evaluation = estimate_token_cost(1000)

        if avg_cost_per_evaluation <= 0:
            return 0

        return int(remaining_budget / avg_cost_per_evaluation)

    def reset_budget(self, new_budget_limit: Optional[float] = None) -> None:
        """
        Reset the budget tracking state.

        Args:
            new_budget_limit: New budget limit (keeps current if None)
        """
        if new_budget_limit is not None:
            self.budget_limit = new_budget_limit
        self.accumulated_cost = 0.0
        self.cost_history.clear()
        self.emergency_mode = False
