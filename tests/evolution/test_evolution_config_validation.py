"""
Tests for EvolutionConfig validation with population and generation limits.

This module tests the new constraints:
- Population: 2-10
- Generations: 2-5
"""

import pytest
from mad_spark_alt.evolution.interfaces import EvolutionConfig


class TestEvolutionConfigValidation:
    """Test evolution configuration validation."""

    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = EvolutionConfig()
        assert config.validate() is True

    def test_population_size_minimum(self):
        """Test population size must be at least 2."""
        # Valid minimum (must also set max_parallel_evaluations to be <= population_size)
        config = EvolutionConfig(population_size=2, max_parallel_evaluations=2)
        assert config.validate() is True
        
        # Invalid: too small
        config = EvolutionConfig(population_size=1)
        assert config.validate() is False
        
        config = EvolutionConfig(population_size=0)
        assert config.validate() is False

    def test_population_size_maximum(self):
        """Test population size must not exceed 10."""
        # Valid maximum
        config = EvolutionConfig(population_size=10)
        assert config.validate() is True
        
        # Invalid: too large
        config = EvolutionConfig(population_size=11)
        assert config.validate() is False
        
        config = EvolutionConfig(population_size=20)
        assert config.validate() is False

    def test_generation_minimum(self):
        """Test generations must be at least 2."""
        # Valid minimum
        config = EvolutionConfig(generations=2)
        assert config.validate() is True
        
        # Invalid: too small
        config = EvolutionConfig(generations=1)
        assert config.validate() is False
        
        config = EvolutionConfig(generations=0)
        assert config.validate() is False

    def test_generation_maximum(self):
        """Test generations must not exceed 5."""
        # Valid maximum
        config = EvolutionConfig(generations=5)
        assert config.validate() is True
        
        # Invalid: too large
        config = EvolutionConfig(generations=6)
        assert config.validate() is False
        
        config = EvolutionConfig(generations=10)
        assert config.validate() is False

    def test_elite_size_with_small_population(self):
        """Test elite size validation with small population."""
        # Valid: elite < population
        config = EvolutionConfig(population_size=3, elite_size=2)
        assert config.validate() is True
        
        # Invalid: elite >= population
        config = EvolutionConfig(population_size=2, elite_size=2)
        assert config.validate() is False

    def test_tournament_size_with_small_population(self):
        """Test tournament size validation with small population."""
        # Valid: tournament <= population
        config = EvolutionConfig(population_size=3, tournament_size=3)
        assert config.validate() is True
        
        # Invalid: tournament > population
        config = EvolutionConfig(population_size=2, tournament_size=3)
        assert config.validate() is False

    def test_max_parallel_evaluations_validation(self):
        """Test max_parallel_evaluations must not exceed population_size."""
        # Valid: max_parallel_evaluations <= population_size
        config = EvolutionConfig(population_size=5, max_parallel_evaluations=5)
        assert config.validate() is True
        
        config = EvolutionConfig(population_size=10, max_parallel_evaluations=5)
        assert config.validate() is True
        
        # Invalid: max_parallel_evaluations > population_size
        config = EvolutionConfig(population_size=2, max_parallel_evaluations=3)
        assert config.validate() is False

    def test_semantic_operator_config(self):
        """Test new semantic operator configuration fields."""
        config = EvolutionConfig(
            population_size=5,
            generations=3,
            use_semantic_operators=True,
            semantic_operator_threshold=0.5,
            semantic_batch_size=5,
            semantic_cache_ttl=3600
        )
        assert config.validate() is True
        
        # These fields should exist and have proper values
        assert hasattr(config, 'use_semantic_operators')
        assert hasattr(config, 'semantic_operator_threshold')
        assert hasattr(config, 'semantic_batch_size')
        assert hasattr(config, 'semantic_cache_ttl')

    def test_validation_error_messages(self):
        """Test that validation provides clear error messages."""
        config = EvolutionConfig(population_size=1)
        
        # For now, validate() returns bool, but we'll enhance it later
        # to return validation errors
        assert config.validate() is False

    def test_combined_valid_config(self):
        """Test a valid configuration with all constraints."""
        config = EvolutionConfig(
            population_size=5,
            generations=3,
            elite_size=1,
            tournament_size=3,
            mutation_rate=0.3,
            crossover_rate=0.75,
            use_semantic_operators=True,
            semantic_operator_threshold=0.5
        )
        assert config.validate() is True

    def test_edge_cases(self):
        """Test edge cases for validation."""
        # Minimum valid config
        min_config = EvolutionConfig(
            population_size=2,
            generations=2,
            elite_size=0,
            tournament_size=2,
            max_parallel_evaluations=2  # Must be <= population_size
        )
        assert min_config.validate() is True
        
        # Maximum valid config
        max_config = EvolutionConfig(
            population_size=10,
            generations=5,
            elite_size=3,
            tournament_size=5
        )
        assert max_config.validate() is True