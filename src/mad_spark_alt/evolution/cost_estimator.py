"""
Cost estimation system for evolution.

This module provides cost prediction and tracking for LLM-based
genetic evolution operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
            input_cost_per_1k_tokens=0.03, output_cost_per_1k_tokens=0.06
        ),
        "gpt-4-turbo": ModelCosts(
            input_cost_per_1k_tokens=0.01, output_cost_per_1k_tokens=0.03
        ),
        "gpt-3.5-turbo": ModelCosts(
            input_cost_per_1k_tokens=0.001, output_cost_per_1k_tokens=0.002
        ),
        "claude-3-opus": ModelCosts(
            input_cost_per_1k_tokens=0.015, output_cost_per_1k_tokens=0.075
        ),
        "claude-3-sonnet": ModelCosts(
            input_cost_per_1k_tokens=0.003, output_cost_per_1k_tokens=0.015
        ),
        "gemini-pro": ModelCosts(
            input_cost_per_1k_tokens=0.001, output_cost_per_1k_tokens=0.002
        ),
    }

    def __init__(self):
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
        cache_hit_rate: float = 0.3,
        enable_llm_operators: bool = False,
    ) -> Dict:
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
        eval_input_tokens = actual_evaluations * (avg_tokens_per_evaluation * 0.7)
        eval_output_tokens = actual_evaluations * (avg_tokens_per_evaluation * 0.3)

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
                * 0.5
            )
            crossover_tokens = crossover_ops * 500  # Estimate 500 tokens per crossover

            # Mutation operations
            mutation_ops = int(
                config.generations * config.population_size * config.mutation_rate
            )
            mutation_tokens = mutation_ops * 300  # Estimate 300 tokens per mutation

            # Selection advice
            selection_ops = config.generations
            selection_tokens = (
                selection_ops * 1000
            )  # Estimate 1000 tokens per selection

            operator_tokens = crossover_tokens + mutation_tokens + selection_tokens
            operator_cost = model_cost.calculate_cost(
                int(operator_tokens * 0.6),  # 60% input
                int(operator_tokens * 0.4),  # 40% output
            )

        # Total cost
        total_cost = evaluation_cost + operator_cost

        # Calculate confidence interval (±20%)
        confidence_interval = (total_cost * 0.8, total_cost * 1.2)

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

    def get_usage_statistics(self) -> Dict:
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
    ) -> List[Dict]:
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

        suggestions = []

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
            ("gpt-3.5-turbo", 0.95),  # 95% cost reduction
            ("gemini-pro", 0.93),  # 93% cost reduction
            ("claude-3-sonnet", 0.80),  # 80% cost reduction
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
            reduced_config = EvolutionConfig(
                **{**current_config.__dict__, "generations": reduced_gens}
            )
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
            reduced_config = EvolutionConfig(
                **{**current_config.__dict__, "population_size": reduced_pop}
            )
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
