"""
Semantic (LLM-powered) genetic operators for evolution.

This module re-exports semantic operators from their respective modules
for backward compatibility. The implementation has been split into focused
modules for better maintainability.

**Deprecated:** Direct imports from this module are maintained for backward
compatibility but users should import from specific modules:
- operator_cache.SemanticOperatorCache
- semantic_mutation.BatchSemanticMutationOperator
- semantic_crossover.SemanticCrossoverOperator, BatchSemanticCrossoverOperator
- semantic_utils.* (utility functions)
"""

# Re-export cache
from .operator_cache import SemanticOperatorCache

# Re-export mutation operators
from .semantic_mutation import BatchSemanticMutationOperator

# Re-export crossover operators
from .semantic_crossover import (
    BatchSemanticCrossoverOperator,
    SemanticCrossoverOperator,
)

# Re-export utility functions
from .semantic_utils import (
    SEMANTIC_MUTATION_MAX_TOKENS,
    SEMANTIC_BATCH_MUTATION_BASE_TOKENS,
    SEMANTIC_BATCH_MUTATION_MAX_TOKENS,
    SEMANTIC_CROSSOVER_MAX_TOKENS,
    format_evaluation_context,
    get_crossover_schema,
    get_mutation_schema,
    is_likely_truncated,
    _prepare_cache_key_with_context,
    _prepare_operator_contexts,
)

__all__ = [
    # Cache
    "SemanticOperatorCache",
    # Mutation operators
    "BatchSemanticMutationOperator",
    # Crossover operators
    "SemanticCrossoverOperator",
    "BatchSemanticCrossoverOperator",
    # Utilities
    "format_evaluation_context",
    "get_mutation_schema",
    "get_crossover_schema",
    "is_likely_truncated",
    "_prepare_operator_contexts",
    "_prepare_cache_key_with_context",
    # Constants
    "SEMANTIC_MUTATION_MAX_TOKENS",
    "SEMANTIC_BATCH_MUTATION_BASE_TOKENS",
    "SEMANTIC_BATCH_MUTATION_MAX_TOKENS",
    "SEMANTIC_CROSSOVER_MAX_TOKENS",
]
