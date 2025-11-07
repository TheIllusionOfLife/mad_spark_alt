"""
Utility functions for semantic operators.

This module provides shared utilities used by semantic mutation and crossover operators,
including context preparation, cache key generation, truncation detection, and JSON schemas.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple, Union

from mad_spark_alt.evolution.interfaces import EvaluationContext

# Token limits for semantic operators (optimized for performance)
SEMANTIC_MUTATION_MAX_TOKENS = 1500  # Increased to reduce truncation warnings
SEMANTIC_BATCH_MUTATION_BASE_TOKENS = 1500  # Base tokens per idea in batch
SEMANTIC_BATCH_MUTATION_MAX_TOKENS = 6000  # Maximum tokens for batch mutation
SEMANTIC_CROSSOVER_MAX_TOKENS = 2000  # Increased for better synthesis

# Cache configuration constants
CACHE_MAX_SIZE = 1000  # Increased maximum number of cache entries for better performance
SIMILARITY_KEY_LENGTH = 16  # Length of similarity hash key
SIMILARITY_CONTENT_PREFIX_LENGTH = 50  # Characters to use for similarity matching
SIMILARITY_WORDS_COUNT = 10  # Number of meaningful words for similarity key
SESSION_TTL_EXTENSION_RATE = 0.1  # Rate of TTL extension during session
MAX_SESSION_TTL_EXTENSION = 3600  # Maximum TTL extension in seconds

# Stop words for similarity matching
STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'by', 'for', 'with', 'to', 'of', 'in', 'on', 'at'}


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
    Prepare cache key that includes EvaluationContext for context-aware caching.

    Args:
        base_key: Base cache key (e.g., idea content or parent combination)
        context: Either string context or EvaluationContext object

    Returns:
        Cache key that includes context information if applicable
    """
    if isinstance(context, EvaluationContext):
        # Include target improvements and current scores in cache key
        context_hash = hash(frozenset([
            (k, v) for k, v in context.current_best_scores.items()
        ] + [tuple(context.target_improvements)]))
        return f"{base_key}||ctx:{context_hash}"
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
            # Check if last word is very short (likely a determiner or preposition)
            # Common truncation patterns: "the", "a", "an", "and", "or", "with", etc.
            if len(last_word) <= 3 and last_word.lower() in {'a', 'an', 'the', 'and', 'or',
                                                              'but', 'for', 'with', 'to', 'of',
                                                              'in', 'on', 'at', 'by', 'is', 'was'}:
                return True
            # Check for other common incomplete endings
            if last_word.lower() in {'without', 'within', 'through', 'before', 'after', 'during'}:
                return True
            # Check if it appears to be mid-word (e.g., "previous" without context)
            if len(words) >= 3 and last_word == "previous":
                return True

    return False


def get_mutation_schema() -> Dict[str, Any]:
    """Get JSON schema for structured mutation output.

    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "mutations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "content": {"type": "STRING"},
                    },
                    "required": ["id", "content"]
                }
            }
        },
        "required": ["mutations"]
    }


def get_crossover_schema() -> Dict[str, Any]:
    """Get JSON schema for structured crossover output.

    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "offspring_1": {"type": "STRING"},
            "offspring_2": {"type": "STRING"}
        },
        "required": ["offspring_1", "offspring_2"]
    }
