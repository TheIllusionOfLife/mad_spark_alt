"""System-wide constants for Mad Spark Alt.

This module centralizes all magic numbers used throughout the system.
Constants are organized by category and documented for clarity.

Usage:
    from mad_spark_alt.core.system_constants import CONSTANTS

    timeout = CONSTANTS.EVOLUTION.MIN_GENERATIONS
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionConstants:
    """Constants for genetic evolution system."""

    # Population size constraints
    MIN_POPULATION_SIZE: int = 2
    """Minimum population size for evolution. Below this, evolution cannot occur."""

    MAX_POPULATION_SIZE: int = 10
    """Maximum population size to prevent excessive API costs."""

    # Generation constraints
    MIN_GENERATIONS: int = 2
    """Minimum number of generations. Single generation is just mutation, not evolution."""

    MAX_GENERATIONS: int = 5
    """Maximum generations to balance quality improvement with cost/time."""

    # Selection parameters
    DEFAULT_ELITE_SIZE: int = 1
    """Number of top individuals preserved across generations."""

    DEFAULT_TOURNAMENT_SIZE: int = 2
    """Default tournament size for selection (must not exceed population size)."""

    # Parallel processing
    DEFAULT_MAX_PARALLEL_EVALUATIONS: int = 3
    """Maximum parallel LLM evaluations to manage API rate limits."""

    MIN_BATCH_SIZE_FOR_PARALLEL: int = 3
    """Minimum population size to enable parallel processing."""


@dataclass(frozen=True)
class TimeoutConstants:
    """Timeout configuration for various operations.

    Note: CLI and orchestrator use different timeout strategies:
    - Orchestrator: Fixed base timeout (90s) + total cap (900s)
    - CLI: Dynamic calculation based on population/generations (25s per evaluation)

    This difference is intentional - orchestrator handles individual phases,
    while CLI estimates total runtime including all phases and evolution.
    """

    # Orchestrator timeouts (for individual QADI phases)
    PHASE_TIMEOUT_BASE: int = 90
    """Base timeout in seconds for a single QADI phase."""

    PHASE_TIMEOUT_TOTAL: int = 900
    """Total timeout in seconds for complete orchestration (15 minutes)."""

    # CLI timeouts (for total process estimation)
    CLI_BASE_TIMEOUT_SECONDS: float = 300.0
    """Base timeout for CLI operations before evolution."""

    CLI_SECONDS_PER_EVALUATION: int = 60
    """Estimated seconds per evaluation for timeout calculation display."""

    CLI_EVOLUTION_TIMEOUT_PER_EVAL: int = 60
    """Actual seconds allocated per evaluation in evolution timeout calculation."""

    CLI_MAX_TIMEOUT_SECONDS: float = 900.0
    """Maximum timeout for CLI operations (15 minutes, matches PHASE_TIMEOUT_TOTAL)."""

    # Ollama provider timeouts
    OLLAMA_INFERENCE_TIMEOUT: int = 180
    """Timeout in seconds for Ollama local inference (3 minutes)."""

    OLLAMA_CONNECTION_CHECK_TIMEOUT: int = 2
    """Timeout in seconds for checking Ollama server availability."""

    # Gemini provider timeouts
    GEMINI_REQUEST_TIMEOUT: int = 300
    """Timeout in seconds for Gemini API requests (5 minutes)."""


@dataclass(frozen=True)
class LLMConstants:
    """Constants for LLM API calls and prompt engineering."""

    # Temperature settings
    DEFAULT_QADI_TEMPERATURE: float = 0.8
    """Default temperature for QADI phases (balanced creativity/coherence)."""

    REGULAR_MUTATION_TEMPERATURE: float = 0.8
    """Temperature for regular evolutionary mutations."""

    BREAKTHROUGH_TEMPERATURE: float = 0.95
    """Higher temperature for breakthrough mutations to encourage novel ideas."""

    # Top-p (nucleus sampling)
    DEFAULT_QADI_TOP_P: float = 0.95
    """Default top_p for QADI phases."""

    # Token limits
    SEMANTIC_MUTATION_MAX_TOKENS: int = 1500
    """Maximum tokens for semantic mutation operations."""

    SEMANTIC_BATCH_MUTATION_BASE_TOKENS: int = 1500
    """Base token allocation for batch mutations."""

    SEMANTIC_BATCH_MUTATION_MAX_TOKENS: int = 6000
    """Maximum tokens for batch mutation operations."""

    SEMANTIC_CROSSOVER_MAX_TOKENS: int = 2000
    """Maximum tokens for semantic crossover operations."""

    # Breakthrough mutation parameters
    BREAKTHROUGH_TOKEN_MULTIPLIER: int = 2
    """Token multiplier for breakthrough mutations (2x normal limit)."""

    BREAKTHROUGH_FITNESS_THRESHOLD: float = 0.8
    """Minimum fitness score to qualify for breakthrough mutation."""

    BREAKTHROUGH_CONFIDENCE_THRESHOLD: float = 0.85
    """Minimum confidence to qualify for breakthrough mutation."""

    BREAKTHROUGH_CONFIDENCE_MULTIPLIER: float = 1.05
    """Confidence boost multiplier for breakthrough ideas."""

    REGULAR_CONFIDENCE_MULTIPLIER: float = 0.95
    """Confidence adjustment for regular mutations."""

    # Default values
    DEFAULT_HYPOTHESIS_CONFIDENCE: float = 0.8
    """Default confidence score when LLM doesn't provide one."""

    # Ollama provider defaults
    OLLAMA_DEFAULT_BASE_URL: str = "http://localhost:11434"
    """Default base URL for Ollama local server."""

    OLLAMA_DEFAULT_MODEL: str = "gemma3:12b-it-qat"
    """Default Ollama model for QADI operations."""

    OLLAMA_DEFAULT_MAX_TOKENS: int = 8192
    """Default context window for Ollama models."""


@dataclass(frozen=True)
class SimilarityConstants:
    """Thresholds for similarity calculations in different contexts.

    Different operations require different similarity thresholds:
    - Crossover (0.7): More lenient to allow combining similar ideas
    - Cache matching (0.8): Balanced to avoid cache misses while preventing duplicates
    - Deduplication (0.85): Stricter to ensure truly unique results shown to user
    - Semantic operators (0.8): Balanced threshold for operator selection
    """

    CROSSOVER_THRESHOLD: float = 0.7
    """Similarity threshold for crossover operations (allows moderately similar parents)."""

    CACHE_THRESHOLD: float = 0.8
    """Similarity threshold for cache hit determination."""

    DEDUP_THRESHOLD: float = 0.85
    """Similarity threshold for deduplication (stricter to ensure uniqueness)."""

    SEMANTIC_OPERATOR_THRESHOLD: float = 0.8
    """Threshold for enabling semantic operators in evolution."""


@dataclass(frozen=True)
class TextProcessingConstants:
    """Constants for text processing, truncation, and display formatting."""

    # Display length limits
    MAX_IDEA_DISPLAY_LENGTH: int = 200
    """Maximum length for displaying idea content in output."""

    MAX_CONTEXT_TRUNCATION_LENGTH: int = 350
    """Maximum context length before truncation."""

    MAX_RESULT_LENGTH: int = 300
    """Maximum result text length for display."""

    MAX_EXAMPLE_LENGTH: int = 400
    """Maximum length for example text."""

    MAX_SHORT_SNIPPET: int = 50
    """Maximum length for short text snippets."""

    MAX_ACTION_LENGTH: int = 100
    """Maximum length for action plan items."""

    TRUNCATED_RESPONSE_LENGTH: int = 200
    """Length to truncate LLM responses in logs."""

    # Title and hypothesis formatting
    MIN_HYPOTHESIS_LENGTH: int = 10
    """Minimum length for a valid hypothesis."""

    MIN_MEANINGFUL_TITLE_LENGTH: int = 10
    """Minimum length for a meaningful title."""

    MAX_TITLE_LENGTH: int = 80
    """Maximum title length before truncation."""

    WORD_BOUNDARY_THRESHOLD: int = 50
    """Character threshold for finding word boundaries."""

    MIN_MEANINGFUL_LENGTH: int = 20
    """Minimum length for meaningful text content."""

    # Word boundary detection
    WORD_BOUNDARY_RATIO: float = 0.8
    """Ratio for finding clean word boundaries (80% of max length)."""

    # Display limits
    MAX_DISPLAY_IDEAS: int = 8
    """Maximum number of ideas to display in lists."""

    MAX_TOP_IDEAS_DISPLAY: int = 3
    """Maximum number of top ideas to highlight."""


@dataclass(frozen=True)
class CacheConstants:
    """Constants for caching operations."""

    CACHE_MAX_SIZE: int = 1000
    """Maximum number of entries in semantic operation cache."""

    SIMILARITY_KEY_LENGTH: int = 16
    """Length of similarity hash key for cache lookups."""

    SIMILARITY_CONTENT_PREFIX_LENGTH: int = 50
    """Length of content prefix used in similarity calculations."""

    SIMILARITY_WORDS_COUNT: int = 10
    """Number of words to use in similarity key generation."""


@dataclass(frozen=True)
class ScoringConstants:
    """Constants for scoring and weighting calculations."""

    DEFAULT_QADI_CRITERION_WEIGHT: float = 0.2
    """Default weight per QADI criterion (5 criteria × 0.2 = 1.0)."""

    # These weights are already well-defined in constants.py
    # We reference them here for completeness but they remain in constants.py
    # to avoid breaking existing imports


@dataclass(frozen=True)
class SystemConstants:
    """Master container for all system constants.

    Access constants via the singleton CONSTANTS instance:
        from mad_spark_alt.core.system_constants import CONSTANTS

        min_pop = CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE
        timeout = CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE
    """

    EVOLUTION: EvolutionConstants = EvolutionConstants()
    TIMEOUTS: TimeoutConstants = TimeoutConstants()
    LLM: LLMConstants = LLMConstants()
    SIMILARITY: SimilarityConstants = SimilarityConstants()
    TEXT: TextProcessingConstants = TextProcessingConstants()
    CACHE: CacheConstants = CacheConstants()
    SCORING: ScoringConstants = ScoringConstants()


# Singleton instance - import and use this throughout the codebase
CONSTANTS = SystemConstants()
