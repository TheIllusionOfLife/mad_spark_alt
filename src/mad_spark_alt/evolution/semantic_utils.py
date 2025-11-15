"""
Utility functions for semantic operators.

This module provides shared utilities used by semantic mutation and crossover operators,
including context preparation, cache key generation, truncation detection, and JSON schemas.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple, Union

from mad_spark_alt.core.schemas import BatchMutationResponse, CrossoverResponse
from mad_spark_alt.core.system_constants import CONSTANTS
from mad_spark_alt.evolution.interfaces import EvaluationContext

# Re-export token constants from system_constants for backward compatibility
SEMANTIC_MUTATION_MAX_TOKENS = CONSTANTS.LLM.SEMANTIC_MUTATION_MAX_TOKENS
SEMANTIC_BATCH_MUTATION_BASE_TOKENS = CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_BASE_TOKENS
SEMANTIC_BATCH_MUTATION_MAX_TOKENS = CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS
SEMANTIC_CROSSOVER_MAX_TOKENS = CONSTANTS.LLM.SEMANTIC_CROSSOVER_MAX_TOKENS

# Re-export cache constants from system_constants for backward compatibility
CACHE_MAX_SIZE = CONSTANTS.CACHE.CACHE_MAX_SIZE
SIMILARITY_KEY_LENGTH = CONSTANTS.CACHE.SIMILARITY_KEY_LENGTH
SIMILARITY_CONTENT_PREFIX_LENGTH = CONSTANTS.CACHE.SIMILARITY_CONTENT_PREFIX_LENGTH
SIMILARITY_WORDS_COUNT = CONSTANTS.CACHE.SIMILARITY_WORDS_COUNT

# Session management constants (not centralized yet)
SESSION_TTL_EXTENSION_RATE = 0.1  # Rate of TTL extension during session
MAX_SESSION_TTL_EXTENSION = 3600  # Maximum TTL extension in seconds

# Stop words for similarity matching
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'by', 'for', 'with', 'to', 'of', 'in', 'on', 'at'}

# Truncation indicator words (words that suggest incomplete text)
_TRUNCATION_INDICATOR_WORDS = {
    # Short function words that rarely end sentences
    'a', 'an', 'the', 'and', 'or', 'but', 'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'was',
    # Prepositions that suggest continuation
    'without', 'within', 'through', 'before', 'after', 'during',
    # Context-dependent words
    'previous',  # Often indicates incomplete thought
}


def _prepare_operator_contexts(
    context: Union[Optional[str], EvaluationContext],
    idea_prompt: str,
    default_context: str
) -> Tuple[str, str]:
    """
    Prepare string and evaluation contexts for semantic operators.

    Args:
        context: Either string context or EvaluationContext object
        idea_prompt: The idea's generation prompt as fallback
        default_context: Default context if all else fails

    Returns:
        Tuple of (context_str, evaluation_context_str)
    """
    if isinstance(context, EvaluationContext):
        context_str = context.original_question or idea_prompt or default_context
        evaluation_context_str = format_evaluation_context(context)
    else:
        context_str = context or idea_prompt or default_context
        evaluation_context_str = "No specific evaluation context provided."

    return context_str, evaluation_context_str


def _prepare_cache_key_with_context(
    base_key: str,
    context: Union[Optional[str], EvaluationContext]
) -> str:
    """
    Prepare cache key that includes full context for context-aware caching.

    Args:
        base_key: Base cache key (e.g., idea content or parent combination)
        context: Either string context or EvaluationContext object

    Returns:
        Cache key that includes context information if applicable
    """
    if isinstance(context, EvaluationContext):
        # Include ALL context fields in cache key for proper differentiation
        # - original_question: Different questions need different mutations
        # - target_improvements: Different improvement goals need different approaches
        # - current_best_scores: Current performance affects mutation strategy
        # - evaluation_criteria: Different criteria need different optimizations
        context_parts = [
            ("question", context.original_question),
            ("improvements", tuple(sorted(context.target_improvements))),
            ("scores", tuple(sorted(context.current_best_scores.items()))),
            ("criteria", tuple(sorted(context.evaluation_criteria)))
        ]
        # Use deterministic hashing for consistent cache keys across interpreter restarts
        normalized = repr(tuple(sorted(context_parts))).encode("utf-8")
        context_hash = hashlib.md5(normalized).hexdigest()
        return f"{base_key}||ctx:{context_hash}"
    elif isinstance(context, str):
        # String contexts also affect prompt generation, must be in cache key
        # Use deterministic hashing for consistent cache keys
        string_hash = hashlib.md5(context.encode("utf-8")).hexdigest()
        return f"{base_key}||str:{string_hash}"
    else:
        return base_key


def format_evaluation_context(context: EvaluationContext) -> str:
    """
    Format evaluation context for inclusion in prompts.

    Args:
        context: EvaluationContext with scoring information

    Returns:
        Formatted context string for prompts
    """
    context_parts = [
        f"Original Question: {context.original_question}",
        ""
    ]

    if context.current_best_scores:
        context_parts.append("Current Best Scores:")
        for criterion, score in context.current_best_scores.items():
            context_parts.append(f"  {criterion.title()}: {score:.1f}")
        context_parts.append("")

    if context.target_improvements:
        context_parts.append(f"Target Improvements: {', '.join(context.target_improvements)}")
        context_parts.append("")

    context_parts.append("FOCUS: Create variations that improve the target criteria while maintaining strengths.")

    return "\n".join(context_parts)


def is_likely_truncated(text: str) -> bool:
    """
    Detect if text appears to be truncated.

    Args:
        text: Text to check for truncation

    Returns:
        True if text appears truncated
    """
    if not text:
        return False

    # Check for common truncation indicators
    text = text.strip()

    if not text:
        return False

    # Check if ends with ellipsis
    if text.endswith('...'):
        return True

    # Check for incomplete JSON
    if text.startswith('{') and text.count('{') != text.count('}'):
        return True
    if text.startswith('[') and text.count('[') != text.count(']'):
        return True

    # Check if ends mid-sentence (no proper ending punctuation)
    if text[-1] not in '.!?"\'':
        words = text.split()
        if words:
            last_word = words[-1]
            # Check for comma or colon at end (likely truncated)
            if last_word.endswith(',') or last_word.endswith(':'):
                return True
            # Check if last word is a common truncation indicator
            if last_word.lower() in _TRUNCATION_INDICATOR_WORDS:
                return True

    return False


def get_mutation_schema() -> Dict[str, Any]:
    """Get JSON schema for structured mutation output.

    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return BatchMutationResponse.model_json_schema()


def get_crossover_schema() -> Dict[str, Any]:
    """Get JSON schema for structured crossover output.

    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return CrossoverResponse.model_json_schema()


def generate_crossover_fallback_text(
    parent1: Optional[Any],
    parent2: Optional[Any],
    is_first: bool,
    extra_detail: str = ""
) -> str:
    """
    Generate fallback text for crossover offspring (DRY utility).

    Args:
        parent1: First parent idea (GeneratedIdea or None)
        parent2: Second parent idea (GeneratedIdea or None)
        is_first: Whether this is the first offspring
        extra_detail: Additional detail for batch variants

    Returns:
        Fallback text for offspring
    """
    # Extract content snippets if parents exist
    p1_snippet = parent1.content[:50] if parent1 and hasattr(parent1, 'content') else "parent concept"
    p2_snippet = parent2.content[:50] if parent2 and hasattr(parent2, 'content') else "complementary approach"

    if is_first:
        return (
            f"[FALLBACK TEXT] Hybrid approach combining elements from both parent ideas: "
            f"This solution integrates key aspects from '{p1_snippet}...' and '{p2_snippet}...'. "
            f"The approach combines the structural framework of the first concept with the innovative "
            f"mechanisms of the second, creating a comprehensive solution that addresses the same core "
            f"problem through multiple complementary strategies. Implementation would involve adapting the "
            f"proven methodologies from both approaches while ensuring seamless integration and enhanced "
            f"effectiveness.{extra_detail}"
        )
    else:
        return (
            f"[FALLBACK TEXT] Alternative integration emphasizing synergy: This variation explores a "
            f"different combination pattern by merging the core principles from '{p1_snippet}...' with "
            f"the practical implementation strategies from '{p2_snippet}...'. The resulting solution "
            f"maintains the strengths of both parent approaches while introducing novel elements that "
            f"emerge from their interaction. This alternative demonstrates how the same foundational "
            f"concepts can yield distinctly different yet equally valuable outcomes through strategic "
            f"recombination.{extra_detail}"
        )
