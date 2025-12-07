"""Tests for system_constants module.

Tests verify:
1. Constants are immutable (frozen dataclass)
2. All expected constants exist and have correct types
3. Logical relationships between constants are valid
4. Constants can be imported and accessed correctly
"""

import pytest

from mad_spark_alt.core.system_constants import (
    CONSTANTS,
    CacheConstants,
    EvolutionConstants,
    LLMConstants,
    ScoringConstants,
    SimilarityConstants,
    SystemConstants,
    TextProcessingConstants,
    TimeoutConstants,
)


class TestEvolutionConstants:
    """Test evolution-related constants."""

    def test_population_constraints_are_valid(self):
        """Test that population size constraints are logically valid."""
        assert CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE >= 2, "Need at least 2 for evolution"
        assert CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE >= CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE
        assert CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE == 2
        assert CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE == 10

    def test_generation_constraints_are_valid(self):
        """Test that generation constraints are logically valid."""
        assert CONSTANTS.EVOLUTION.MIN_GENERATIONS >= 2, "Need at least 2 for evolution"
        assert CONSTANTS.EVOLUTION.MAX_GENERATIONS >= CONSTANTS.EVOLUTION.MIN_GENERATIONS
        assert CONSTANTS.EVOLUTION.MIN_GENERATIONS == 2
        assert CONSTANTS.EVOLUTION.MAX_GENERATIONS == 5

    def test_selection_parameters_are_valid(self):
        """Test that selection parameters are valid."""
        assert CONSTANTS.EVOLUTION.DEFAULT_ELITE_SIZE >= 1
        assert CONSTANTS.EVOLUTION.DEFAULT_TOURNAMENT_SIZE >= 2
        assert CONSTANTS.EVOLUTION.DEFAULT_ELITE_SIZE < CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE

    def test_parallel_processing_thresholds(self):
        """Test parallel processing configuration."""
        assert CONSTANTS.EVOLUTION.DEFAULT_MAX_PARALLEL_EVALUATIONS > 0
        assert CONSTANTS.EVOLUTION.MIN_BATCH_SIZE_FOR_PARALLEL >= 3


class TestTimeoutConstants:
    """Test timeout configuration constants."""

    def test_orchestrator_timeouts_are_valid(self):
        """Test that orchestrator timeout values are reasonable."""
        assert CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE > 0
        assert CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_TOTAL > CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE
        assert CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE == 90
        assert CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_TOTAL == 900

    def test_cli_timeouts_are_valid(self):
        """Test that CLI timeout values are reasonable."""
        assert CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS > 0
        assert CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS >= CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS
        assert CONSTANTS.TIMEOUTS.CLI_SECONDS_PER_EVALUATION > 0
        assert CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL > 0

    def test_timeout_calculation_constants(self):
        """Test that timeout calculation parameters make sense."""
        # CLI evolution timeout should be >= display estimate
        assert CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL >= CONSTANTS.TIMEOUTS.CLI_SECONDS_PER_EVALUATION


class TestLLMConstants:
    """Test LLM-related constants."""

    def test_temperature_ranges_are_valid(self):
        """Test that temperature values are in valid range [0.0, 2.0]."""
        assert 0.0 <= CONSTANTS.LLM.DEFAULT_QADI_TEMPERATURE <= 2.0
        assert 0.0 <= CONSTANTS.LLM.REGULAR_MUTATION_TEMPERATURE <= 2.0
        assert 0.0 <= CONSTANTS.LLM.BREAKTHROUGH_TEMPERATURE <= 2.0
        # Breakthrough should be higher than regular
        assert CONSTANTS.LLM.BREAKTHROUGH_TEMPERATURE >= CONSTANTS.LLM.REGULAR_MUTATION_TEMPERATURE

    def test_top_p_is_valid(self):
        """Test that top_p value is in valid range [0.0, 1.0]."""
        assert 0.0 <= CONSTANTS.LLM.DEFAULT_QADI_TOP_P <= 1.0

    def test_token_limits_are_positive(self):
        """Test that all token limits are positive."""
        assert CONSTANTS.LLM.SEMANTIC_MUTATION_MAX_TOKENS > 0
        assert CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_BASE_TOKENS > 0
        assert CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS > 0
        assert CONSTANTS.LLM.SEMANTIC_CROSSOVER_MAX_TOKENS > 0

    def test_batch_token_relationships(self):
        """Test that batch token limits are consistent."""
        assert (
            CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS
            >= CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_BASE_TOKENS
        )

    def test_breakthrough_thresholds_are_valid(self):
        """Test that breakthrough thresholds are in valid ranges."""
        assert 0.0 <= CONSTANTS.LLM.BREAKTHROUGH_FITNESS_THRESHOLD <= 1.0
        assert 0.0 <= CONSTANTS.LLM.BREAKTHROUGH_CONFIDENCE_THRESHOLD <= 1.0
        assert CONSTANTS.LLM.BREAKTHROUGH_TOKEN_MULTIPLIER >= 1
        assert CONSTANTS.LLM.BREAKTHROUGH_CONFIDENCE_MULTIPLIER >= 1.0
        assert CONSTANTS.LLM.REGULAR_CONFIDENCE_MULTIPLIER <= 1.0

    def test_default_hypothesis_confidence_is_valid(self):
        """Test that default confidence is in valid range."""
        assert 0.0 <= CONSTANTS.LLM.DEFAULT_HYPOTHESIS_CONFIDENCE <= 1.0


class TestSimilarityConstants:
    """Test similarity threshold constants."""

    def test_all_thresholds_in_valid_range(self):
        """Test that all similarity thresholds are in [0.0, 1.0]."""
        assert 0.0 <= CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD <= 1.0
        assert 0.0 <= CONSTANTS.SIMILARITY.CACHE_THRESHOLD <= 1.0
        assert 0.0 <= CONSTANTS.SIMILARITY.DEDUP_THRESHOLD <= 1.0
        assert 0.0 <= CONSTANTS.SIMILARITY.SEMANTIC_OPERATOR_THRESHOLD <= 1.0

    def test_threshold_relationships(self):
        """Test logical relationships between thresholds.

        Dedup should be strictest (highest) to ensure uniqueness.
        Crossover can be most lenient (lowest) to allow combining ideas.
        """
        # Dedup should be >= other thresholds (strictest)
        assert CONSTANTS.SIMILARITY.DEDUP_THRESHOLD >= CONSTANTS.SIMILARITY.CACHE_THRESHOLD
        assert CONSTANTS.SIMILARITY.DEDUP_THRESHOLD >= CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD

    def test_specific_threshold_values(self):
        """Test specific threshold values match requirements."""
        assert CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD == 0.7
        assert CONSTANTS.SIMILARITY.CACHE_THRESHOLD == 0.8
        assert CONSTANTS.SIMILARITY.DEDUP_THRESHOLD == 0.85
        assert CONSTANTS.SIMILARITY.SEMANTIC_OPERATOR_THRESHOLD == 0.8


class TestTextProcessingConstants:
    """Test text processing and display constants."""

    def test_length_limits_are_positive(self):
        """Test that all length limits are positive."""
        assert CONSTANTS.TEXT.MAX_IDEA_DISPLAY_LENGTH > 0
        assert CONSTANTS.TEXT.MAX_CONTEXT_TRUNCATION_LENGTH > 0
        assert CONSTANTS.TEXT.MAX_RESULT_LENGTH > 0
        assert CONSTANTS.TEXT.MAX_EXAMPLE_LENGTH > 0
        assert CONSTANTS.TEXT.MAX_SHORT_SNIPPET > 0
        assert CONSTANTS.TEXT.MAX_ACTION_LENGTH > 0
        assert CONSTANTS.TEXT.TRUNCATED_RESPONSE_LENGTH > 0

    def test_title_constraints_are_logical(self):
        """Test that title constraints make sense."""
        assert CONSTANTS.TEXT.MIN_HYPOTHESIS_LENGTH > 0
        assert CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH > 0
        assert CONSTANTS.TEXT.MAX_TITLE_LENGTH > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH
        assert CONSTANTS.TEXT.MIN_MEANINGFUL_LENGTH > 0

    def test_word_boundary_settings(self):
        """Test word boundary detection settings."""
        assert 0.0 < CONSTANTS.TEXT.WORD_BOUNDARY_RATIO <= 1.0
        assert CONSTANTS.TEXT.WORD_BOUNDARY_THRESHOLD > 0

    def test_display_limits(self):
        """Test display count limits."""
        assert CONSTANTS.TEXT.MAX_DISPLAY_IDEAS > 0
        assert CONSTANTS.TEXT.MAX_TOP_IDEAS_DISPLAY > 0
        assert CONSTANTS.TEXT.MAX_TOP_IDEAS_DISPLAY <= CONSTANTS.TEXT.MAX_DISPLAY_IDEAS


class TestCacheConstants:
    """Test cache-related constants."""

    def test_cache_size_limits(self):
        """Test that cache size limits are reasonable."""
        assert CONSTANTS.CACHE.CACHE_MAX_SIZE > 0
        assert CONSTANTS.CACHE.SIMILARITY_KEY_LENGTH > 0
        assert CONSTANTS.CACHE.SIMILARITY_CONTENT_PREFIX_LENGTH > 0
        assert CONSTANTS.CACHE.SIMILARITY_WORDS_COUNT > 0

    def test_specific_cache_values(self):
        """Test specific cache configuration values."""
        assert CONSTANTS.CACHE.CACHE_MAX_SIZE == 1000
        assert CONSTANTS.CACHE.SIMILARITY_KEY_LENGTH == 16
        assert CONSTANTS.CACHE.SIMILARITY_CONTENT_PREFIX_LENGTH == 50
        assert CONSTANTS.CACHE.SIMILARITY_WORDS_COUNT == 10


class TestScoringConstants:
    """Test scoring and weighting constants."""

    def test_qadi_criterion_weight(self):
        """Test that QADI criterion weight is valid."""
        assert 0.0 < CONSTANTS.SCORING.DEFAULT_QADI_CRITERION_WEIGHT <= 1.0
        # 5 criteria × 0.2 should equal 1.0
        assert abs(CONSTANTS.SCORING.DEFAULT_QADI_CRITERION_WEIGHT * 5 - 1.0) < 0.01


class TestSystemConstants:
    """Test the master SystemConstants container."""

    def test_all_subcategories_exist(self):
        """Test that all constant subcategories are accessible."""
        assert isinstance(CONSTANTS.EVOLUTION, EvolutionConstants)
        assert isinstance(CONSTANTS.TIMEOUTS, TimeoutConstants)
        assert isinstance(CONSTANTS.LLM, LLMConstants)
        assert isinstance(CONSTANTS.SIMILARITY, SimilarityConstants)
        assert isinstance(CONSTANTS.TEXT, TextProcessingConstants)
        assert isinstance(CONSTANTS.CACHE, CacheConstants)
        assert isinstance(CONSTANTS.SCORING, ScoringConstants)

    def test_constants_are_immutable(self):
        """Test that constants cannot be modified (frozen dataclass)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE = 999  # type: ignore

        with pytest.raises(Exception):
            CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE = 999  # type: ignore

        with pytest.raises(Exception):
            CONSTANTS.LLM.DEFAULT_QADI_TEMPERATURE = 999.0  # type: ignore

    def test_singleton_instance_is_accessible(self):
        """Test that the CONSTANTS singleton is properly accessible."""
        assert isinstance(CONSTANTS, SystemConstants)
        # Accessing nested constants should work
        assert CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE == 2
        assert CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE == 90
        assert CONSTANTS.LLM.DEFAULT_QADI_TEMPERATURE == 0.8


class TestConstantsIntegration:
    """Integration tests for constants usage across system."""

    def test_evolution_config_validation_compatibility(self):
        """Test that constants work with EvolutionConfig validation."""
        # These values should be compatible with EvolutionConfig
        assert CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE == 2
        assert CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE == 10
        assert CONSTANTS.EVOLUTION.MIN_GENERATIONS == 2
        assert CONSTANTS.EVOLUTION.MAX_GENERATIONS == 5

    def test_timeout_calculation_compatibility(self):
        """Test that timeout constants support expected calculations."""
        # CLI timeout calculation: min(max(base, gens × pop × per_eval), max_timeout)
        # Test with small values (should use base timeout)
        gens = 2
        pop = 2
        estimated_time = gens * pop * CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL  # 240s
        calculated_timeout = min(
            max(CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS, estimated_time),
            CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS
        )
        # With base=300, per_eval=60: max(300, 240) = 300
        assert calculated_timeout == CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS

        # Test with moderate values (should use estimated time)
        gens = 2
        pop = 5
        estimated_time = gens * pop * CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL  # 600s
        calculated_timeout = min(
            max(CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS, estimated_time),
            CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS
        )
        # max(300, 600) = 600, min(600, 900) = 600
        assert calculated_timeout == 600.0

        # Test with max values (should be capped)
        max_gens = CONSTANTS.EVOLUTION.MAX_GENERATIONS  # 5
        max_pop = CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE  # 10
        max_estimated_time = max_gens * max_pop * CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL  # 3000s
        max_calculated_timeout = min(
            max(CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS, max_estimated_time),
            CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS
        )
        # Should be capped at CLI_MAX_TIMEOUT_SECONDS
        assert max_calculated_timeout == CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS

    def test_similarity_thresholds_for_different_purposes(self):
        """Test that different similarity thresholds serve different purposes."""
        # Crossover should be most lenient (allow similar parents)
        assert CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD < CONSTANTS.SIMILARITY.DEDUP_THRESHOLD

        # Dedup should be strictest (ensure unique output)
        assert CONSTANTS.SIMILARITY.DEDUP_THRESHOLD == max(
            CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD,
            CONSTANTS.SIMILARITY.CACHE_THRESHOLD,
            CONSTANTS.SIMILARITY.DEDUP_THRESHOLD,
            CONSTANTS.SIMILARITY.SEMANTIC_OPERATOR_THRESHOLD,
        )

    def test_breakthrough_mutation_requirements(self):
        """Test that breakthrough mutation constants are consistent."""
        # Breakthrough should give more tokens
        regular_tokens = CONSTANTS.LLM.SEMANTIC_MUTATION_MAX_TOKENS
        breakthrough_tokens = regular_tokens * CONSTANTS.LLM.BREAKTHROUGH_TOKEN_MULTIPLIER
        assert breakthrough_tokens > regular_tokens

        # Breakthrough should increase confidence
        assert CONSTANTS.LLM.BREAKTHROUGH_CONFIDENCE_MULTIPLIER > 1.0
        # Regular should slightly decrease confidence
        assert CONSTANTS.LLM.REGULAR_CONFIDENCE_MULTIPLIER < 1.0
