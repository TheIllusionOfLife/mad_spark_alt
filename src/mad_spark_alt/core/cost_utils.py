"""
Centralized cost calculation utilities for LLM operations.

This module provides a unified interface for calculating costs across
the entire codebase, eliminating duplicated cost calculation logic.

Model pricing is managed in model_registry.py - update costs there.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .model_registry import (
    MODELS,
    ProviderType,
    get_default_model,
    get_model_spec,
)

logger = logging.getLogger(__name__)

# Default model constant - sourced from model registry
DEFAULT_MODEL = get_default_model(ProviderType.GEMINI)


@dataclass
class ModelCosts:
    """Cost structure for a specific model."""

    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


# Re-export for convenience
__all__ = [
    "ModelCosts",
    "calculate_cost_with_usage",
    "calculate_llm_cost",
    "calculate_llm_cost_from_config",
    "calculate_token_cost",
    "estimate_token_cost",
    "get_available_models",
    "get_model_costs",
    "DEFAULT_MODEL",
]


def calculate_llm_cost_from_config(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    """
    Calculate cost for LLM usage using provided cost rates.

    This function uses cost values directly from ModelConfig objects,
    avoiding model name mismatch issues.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        input_cost_per_1k: Cost per 1k input tokens (from ModelConfig)
        output_cost_per_1k: Cost per 1k output tokens (from ModelConfig)

    Returns:
        Total cost in USD

    Example:
        >>> cost = calculate_llm_cost_from_config(1000, 500, 0.00015, 0.0006)
        >>> print(f"Cost: ${cost:.4f}")
    """
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost


def calculate_llm_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Calculate cost for LLM usage given input and output tokens.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: Model name (defaults to current default Gemini model)

    Returns:
        Total cost in USD

    Example:
        >>> cost = calculate_llm_cost(1000, 500)
        >>> print(f"Cost: ${cost:.4f}")
    """
    model_costs = get_model_costs(model)
    if model_costs is None:
        # Fall back to default model costs if model not found
        logger.warning(
            "Model '%s' not found, falling back to '%s' costs.", model, DEFAULT_MODEL
        )
        model_costs = get_model_costs(DEFAULT_MODEL)
        if model_costs is None:
            # Should never happen, but handle gracefully
            logger.error("Default model '%s' not found in registry", DEFAULT_MODEL)
            return 0.0

    input_cost = (input_tokens / 1000) * model_costs.input_cost_per_1k_tokens
    output_cost = (output_tokens / 1000) * model_costs.output_cost_per_1k_tokens

    return input_cost + output_cost


def calculate_token_cost(
    total_tokens: int,
    model: str = DEFAULT_MODEL,
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
        Estimated cost in USD
    """
    input_tokens = int(total_tokens * input_output_ratio)
    output_tokens = total_tokens - input_tokens
    return calculate_llm_cost(input_tokens, output_tokens, model)


def estimate_token_cost(
    tokens: int,
    model: str = DEFAULT_MODEL,
    assume_equal_input_output: bool = True,
) -> float:
    """
    Estimate cost for a given number of tokens.

    Args:
        tokens: Total number of tokens
        model: Model name (defaults to current default Gemini model)
        assume_equal_input_output: If True, assumes equal input/output split

    Returns:
        Estimated cost in USD
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
    spec = get_model_spec(model)
    if spec is None:
        return None
    return ModelCosts(
        input_cost_per_1k_tokens=spec.input_cost_per_1k,
        output_cost_per_1k_tokens=spec.output_cost_per_1k,
    )


def get_available_models() -> Dict[str, ModelCosts]:
    """
    Get all available models and their cost structures.

    Returns:
        Dictionary mapping model names to ModelCosts
    """
    return {
        model_id: ModelCosts(
            input_cost_per_1k_tokens=spec.input_cost_per_1k,
            output_cost_per_1k_tokens=spec.output_cost_per_1k,
        )
        for model_id, spec in MODELS.items()
    }


def calculate_cost_with_usage(
    usage: Dict[str, int],
    model: str = DEFAULT_MODEL,
) -> Tuple[float, int, int]:
    """
    Calculate cost from a usage dictionary (Google format).

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
