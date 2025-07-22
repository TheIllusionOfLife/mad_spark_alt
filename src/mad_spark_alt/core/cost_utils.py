"""
Centralized cost calculation utilities for LLM operations.

This module provides a unified interface for calculating costs across
the entire codebase, eliminating duplicated cost calculation logic.
"""

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelCosts:
    """Cost structure for a specific model."""

    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


# Default model costs (as of 2024)
_MODEL_COSTS = {
    "gpt-4": ModelCosts(
        input_cost_per_1k_tokens=0.03,
        output_cost_per_1k_tokens=0.06,
    ),
    "gpt-4-turbo": ModelCosts(
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
    ),
    "gpt-3.5-turbo": ModelCosts(
        input_cost_per_1k_tokens=0.0005,
        output_cost_per_1k_tokens=0.0015,
    ),
    "gpt-4o-mini": ModelCosts(
        input_cost_per_1k_tokens=0.00015,
        output_cost_per_1k_tokens=0.0006,
    ),
    "claude-3-opus": ModelCosts(
        input_cost_per_1k_tokens=0.015,
        output_cost_per_1k_tokens=0.075,
    ),
    "claude-3-sonnet": ModelCosts(
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
    ),
    "claude-3-haiku-20240307": ModelCosts(
        input_cost_per_1k_tokens=0.00025,
        output_cost_per_1k_tokens=0.00125,
    ),
    "gemini-pro": ModelCosts(
        input_cost_per_1k_tokens=0.00025,
        output_cost_per_1k_tokens=0.001,
    ),
}

# Make the costs immutable to prevent accidental modifications
DEFAULT_MODEL_COSTS = MappingProxyType(_MODEL_COSTS)

# Re-export for convenience
__all__ = [
    "ModelCosts",
    "calculate_cost_with_usage",
    "calculate_llm_cost",
    "calculate_token_cost",
    "estimate_token_cost",
    "get_available_models",
    "get_model_costs",
]


def calculate_llm_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4",
) -> float:
    """
    Calculate cost for LLM usage given input and output tokens.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3-opus")

    Returns:
        Total cost in dollars

    Example:
        >>> cost = calculate_llm_cost(1000, 500, "gpt-4")
        >>> print(f"Cost: ${cost:.4f}")
    """
    model_costs = get_model_costs(model)
    if model_costs is None:
        # Fall back to GPT-4 costs if model not found
        logger.warning("Model '%s' not found, falling back to 'gpt-4' costs.", model)
        model_costs = DEFAULT_MODEL_COSTS["gpt-4"]

    input_cost = (input_tokens / 1000) * model_costs.input_cost_per_1k_tokens
    output_cost = (output_tokens / 1000) * model_costs.output_cost_per_1k_tokens

    return input_cost + output_cost


def calculate_token_cost(
    total_tokens: int,
    model: str = "gpt-4",
    input_output_ratio: float = 0.5,
) -> float:
    """
    Calculate cost for a total number of tokens with estimated input/output split.

    This is a convenience wrapper for cases where you only know the total tokens.

    Args:
        total_tokens: Total number of tokens
        model: Model name
        input_output_ratio: Ratio of input tokens (0.5 = equal split)

    Returns:
        Estimated cost in dollars
    """
    input_tokens = int(total_tokens * input_output_ratio)
    output_tokens = total_tokens - input_tokens
    return calculate_llm_cost(input_tokens, output_tokens, model)


def estimate_token_cost(
    tokens: int,
    model: str = "gpt-4",
    assume_equal_input_output: bool = True,
) -> float:
    """
    Estimate cost for a given number of tokens.

    Args:
        tokens: Total number of tokens
        model: Model name (defaults to GPT-4)
        assume_equal_input_output: If True, assumes equal input/output split

    Returns:
        Estimated cost in dollars
    """
    if assume_equal_input_output:
        return calculate_token_cost(tokens, model, 0.5)
    else:
        # Use a typical ratio of 30% input, 70% output for general estimates
        return calculate_token_cost(tokens, model, 0.3)


def get_model_costs(model: str) -> Optional[ModelCosts]:
    """
    Get cost structure for a specific model.

    Args:
        model: Model name

    Returns:
        ModelCosts object or None if model not found
    """
    return DEFAULT_MODEL_COSTS.get(model)


def get_available_models() -> Dict[str, ModelCosts]:
    """
    Get all available models and their cost structures.

    Returns:
        Dictionary mapping model names to ModelCosts
    """
    return dict(DEFAULT_MODEL_COSTS)


def calculate_cost_with_usage(
    usage: Dict[str, int],
    model: str = "gpt-4",
) -> Tuple[float, int, int]:
    """
    Calculate cost from a usage dictionary (OpenAI/Anthropic format).

    Args:
        usage: Dictionary with 'prompt_tokens' and 'completion_tokens'
        model: Model name

    Returns:
        Tuple of (total_cost, input_tokens, output_tokens)
    """
    # Prioritize 'prompt_tokens'/'completion_tokens', but fall back to 'input_tokens'/'output_tokens'.
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

    cost = calculate_llm_cost(input_tokens, output_tokens, model)

    return cost, input_tokens, output_tokens
