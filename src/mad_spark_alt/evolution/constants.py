"""
Constants for the evolution system.

This module defines named constants to replace magic numbers throughout
the evolution codebase, improving maintainability and clarity.
"""

# Budget Management Constants
BUDGET_EMERGENCY_THRESHOLD = 0.9  # 90% of budget used - emergency mode
BUDGET_HIGH_USAGE_THRESHOLD = 0.8  # 80% of budget used - aggressive scaling
BUDGET_MODERATE_USAGE_THRESHOLD = 0.6  # 60% of budget used - moderate scaling
BUDGET_CRITICAL_THRESHOLD = 0.95  # 95% of budget used - stop operations

# Population Size Scaling Factors
AGGRESSIVE_POPULATION_SCALE_FACTOR = (
    0.6  # Reduce population to 60% under high budget usage
)
MODERATE_POPULATION_SCALE_FACTOR = (
    0.8  # Reduce population to 80% under moderate budget usage
)
MIN_POPULATION_SIZE_AGGRESSIVE = 5  # Minimum population size under aggressive scaling
MIN_POPULATION_SIZE_MODERATE = 8  # Minimum population size under moderate scaling

# Cost Estimation Constants
DEFAULT_CACHE_HIT_RATE = 0.3  # 30% default cache hit rate
INPUT_TOKEN_RATIO = 0.7  # 70% of tokens are typically input
OUTPUT_TOKEN_RATIO = 0.3  # 30% of tokens are typically output
OPERATOR_INPUT_RATIO = 0.6  # 60% input tokens for operators
OPERATOR_OUTPUT_RATIO = 0.4  # 40% output tokens for operators
CONFIDENCE_INTERVAL_LOWER = 0.8  # Lower bound for cost confidence interval
CONFIDENCE_INTERVAL_UPPER = 1.2  # Upper bound for cost confidence interval

# Crossover Constants
CROSSOVER_OPERATION_FRACTION = 0.5  # Fraction of crossover operations per generation
CROSSOVER_TOKEN_ESTIMATE = 500  # Estimated tokens per crossover operation

# Mutation Constants
MUTATION_TOKEN_ESTIMATE = 300  # Estimated tokens per mutation operation
MINOR_MUTATION_THRESHOLD = 0.3  # Below this rate = minor refinement
MAJOR_MUTATION_THRESHOLD = 0.7  # Above this rate = radical reimagining

# Selection Constants
SELECTION_TOKEN_ESTIMATE = 1000  # Estimated tokens per selection operation

# LLM Temperature and Token Limits
DEFAULT_LLM_TEMPERATURE = 0.7  # Default temperature for LLM operations
DEFAULT_MAX_TOKENS = 1000  # Default max tokens for LLM operations

# Fitness Score Constants
DEFAULT_FAILURE_SCORE = 0.1  # Score for failed evaluations
DEFAULT_NEUTRAL_SCORE = 0.5  # Neutral/average score
MINIMAL_FITNESS_SCORE = 0.1  # Minimal fitness for repeatedly failed evaluations
ZERO_SCORE = 0.0  # Zero score for initialization

# Confidence Score Adjustments
MUTATION_CONFIDENCE_REDUCTION = 0.95  # Multiply confidence by this after mutation
DEFAULT_CONFIDENCE_SCORE = 0.5  # Default confidence when not specified

# Diversity and Selection Constants
LOW_DIVERSITY_THRESHOLD = 0.1  # Below this = low diversity
MODERATE_DIVERSITY_THRESHOLD = 0.3  # Below this = moderate diversity
HIGH_DIVERSITY_THRESHOLD = 0.7  # Above this = high diversity
MUTATION_RATE_INCREASE_FACTOR = 1.2  # Increase mutation rate by this factor
MUTATION_RATE_DECREASE_FACTOR = 0.8  # Decrease mutation rate by this factor
MAX_MUTATION_RATE = 0.5  # Maximum allowed mutation rate
MIN_MUTATION_RATE = 0.01  # Minimum allowed mutation rate

# Selection Pressure Constants
SELECTION_PRESSURE_ADJUSTMENT = 0.001  # Small value to avoid zero probabilities

# Similarity and Semantic Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # Minimum similarity for semantic cache hit
SIMILARITY_BOOST_FACTOR = 1.1  # 10% boost for same thinking method

# Circuit Breaker and Retry Constants
DEFAULT_MAX_RETRIES = 3  # Default maximum retry attempts
DEFAULT_INITIAL_DELAY = 1.0  # Default initial delay in seconds
DEFAULT_MAX_DELAY = 60.0  # Default maximum delay in seconds
DEFAULT_EXPONENTIAL_BASE = 2.0  # Default exponential backoff base
JITTER_MIN_FACTOR = 0.5  # Minimum jitter factor
JITTER_MAX_FACTOR = 1.0  # Maximum jitter factor (0.5 + random.random() = 0.5-1.5)

# Cache and Performance Constants
DEFAULT_CACHE_TTL_SECONDS = 3600  # Default cache TTL (1 hour)
DEFAULT_MAX_CACHE_SIZE = 1000  # Default maximum cache size
DEFAULT_MAX_CANDIDATES = 10  # Default maximum candidates for similarity matching

# Cost Model Constants (in dollars per 1k tokens)
GPT4_INPUT_COST = 0.03
GPT4_OUTPUT_COST = 0.06
GPT4_TURBO_INPUT_COST = 0.01
GPT4_TURBO_OUTPUT_COST = 0.03
GPT35_TURBO_INPUT_COST = 0.001
GPT35_TURBO_OUTPUT_COST = 0.002
CLAUDE3_OPUS_INPUT_COST = 0.015
CLAUDE3_OPUS_OUTPUT_COST = 0.075
CLAUDE3_SONNET_INPUT_COST = 0.003
CLAUDE3_SONNET_OUTPUT_COST = 0.015
GEMINI_PRO_INPUT_COST = 0.001
GEMINI_PRO_OUTPUT_COST = 0.002

# Cost Reduction Estimates for Model Switching
GPT35_TURBO_COST_REDUCTION = 0.95  # 95% cost reduction vs GPT-4
GEMINI_PRO_COST_REDUCTION = 0.93  # 93% cost reduction vs GPT-4
CLAUDE3_SONNET_COST_REDUCTION = 0.80  # 80% cost reduction vs GPT-4

# Default Token Estimates
DEFAULT_TOKENS_PER_EVALUATION = 1000  # Default tokens per fitness evaluation
DEFAULT_FALLBACK_COST = 0.045  # Default cost estimate when no history available

# Weight Constants for Fitness Calculation
DEFAULT_CREATIVITY_WEIGHT = 0.4  # Default weight for creativity score
DEFAULT_DIVERSITY_WEIGHT = 0.3  # Default weight for diversity score
DEFAULT_QUALITY_WEIGHT = 0.3  # Default weight for quality score

# Equal weight fallbacks
EQUAL_WEIGHT_CREATIVITY = 0.33  # Equal weight for creativity
EQUAL_WEIGHT_DIVERSITY = 0.33  # Equal weight for diversity
EQUAL_WEIGHT_QUALITY = 0.34  # Equal weight for quality (slightly higher to sum to 1.0)

# Evolution Configuration Defaults
DEFAULT_MUTATION_RATE = 0.1  # Default mutation rate
DEFAULT_CROSSOVER_RATE = 0.7  # Default crossover rate
DEFAULT_DIVERSITY_PRESSURE = 0.1  # Default diversity pressure

# Evaluation Layer Weights
QUANTITATIVE_LAYER_WEIGHT = 0.3  # Weight for quantitative evaluation
LLM_JUDGE_LAYER_WEIGHT = 0.5  # Weight for LLM judge evaluation
HUMAN_LAYER_WEIGHT = 0.2  # Weight for human evaluation

# Progress and Performance Constants
CPU_MEASUREMENT_INTERVAL = 0.1  # CPU measurement interval in seconds
DEFAULT_EVOLUTION_TIMEOUT = 300.0  # Default evolution timeout (5 minutes)

# Chart and Visualization Constants
CHART_ALPHA_LIGHT = 0.3  # Light alpha for chart elements
CHART_ALPHA_MEDIUM = 0.2  # Medium alpha for chart elements

# Standard QADI evaluation criteria used throughout the evolution system
EVALUATION_CRITERIA = [
    "impact", 
    "feasibility", 
    "accessibility", 
    "sustainability", 
    "scalability"
]
