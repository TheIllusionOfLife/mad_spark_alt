"""
Tests for cached fitness evaluation in the evolution system.

This module tests the caching functionality for fitness evaluations,
ensuring efficient reuse of previously computed fitness scores.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import (
    EvaluationResult,
    GeneratedIdea,
    ThinkingMethod,
)
from mad_spark_alt.evolution.cached_fitness import (
    CachedFitnessEvaluator,
    FitnessCache,
    FitnessCacheEntry,
)
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness


class TestFitnessCache:
    """Test suite for FitnessCache."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.cache = FitnessCache(ttl_seconds=3600)

    def test_cache_key_generation(self) -> None:
        """Test cache key generation is deterministic and unique."""
        # Create test idea
        idea = GeneratedIdea(
            content="Test idea content",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        # Generate keys
        key1 = self.cache.generate_key(idea, "test_context")
        key2 = self.cache.generate_key(idea, "test_context")
        key3 = self.cache.generate_key(idea, "different_context")
        
        # Same input should produce same key
        assert key1 == key2
        # Different context should produce different key
        assert key1 != key3

    def test_cache_set_and_get(self) -> None:
        """Test basic cache set and get operations."""
        # Create test data
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        fitness = IndividualFitness(
            idea=idea,
            creativity_score=0.8,
            diversity_score=0.7,
            quality_score=0.9,
            overall_fitness=0.8,
        )
        
        # Set and get
        key = self.cache.generate_key(idea, "context")
        self.cache.set(key, fitness)
        retrieved = self.cache.get(key)
        
        assert retrieved is not None
        assert retrieved.creativity_score == fitness.creativity_score
        assert retrieved.diversity_score == fitness.diversity_score
        assert retrieved.quality_score == fitness.quality_score

    def test_cache_expiration(self) -> None:
        """Test cache entries expire after TTL."""
        # Create short-lived cache
        cache = FitnessCache(ttl_seconds=0.1)
        
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        fitness = IndividualFitness(idea=idea, overall_fitness=0.5)
        
        # Set and verify exists
        key = cache.generate_key(idea, "context")
        cache.set(key, fitness)
        assert cache.get(key) is not None
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get(key) is None

    def test_cache_stats(self) -> None:
        """Test cache statistics tracking."""
        cache = FitnessCache(ttl_seconds=3600)
        
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.INDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        fitness = IndividualFitness(idea=idea, overall_fitness=0.6)
        key = cache.generate_key(idea, "context")
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Miss
        cache.get(key)
        stats = cache.get_stats()
        assert stats["misses"] == 1
        
        # Set and hit
        cache.set(key, fitness)
        cache.get(key)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total

    def test_cache_clear(self) -> None:
        """Test cache clearing functionality."""
        # Add multiple entries
        for i in range(5):
            idea = GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            )
            fitness = IndividualFitness(idea=idea, overall_fitness=0.5 + i * 0.1)
            key = self.cache.generate_key(idea, "context")
            self.cache.set(key, fitness)
        
        # Verify entries exist
        assert self.cache.get_stats()["size"] == 5
        
        # Clear cache
        self.cache.clear()
        
        # Verify empty
        assert self.cache.get_stats()["size"] == 0


class TestCachedFitnessEvaluator:
    """Test suite for CachedFitnessEvaluator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create mock base evaluator
        self.mock_base_evaluator = MagicMock(spec=FitnessEvaluator)
        self.mock_base_evaluator.evaluate_individual = AsyncMock()
        self.mock_base_evaluator.calculate_population_diversity = AsyncMock(
            return_value=0.75
        )
        # Add creativity_evaluator attribute to mock
        self.mock_base_evaluator.creativity_evaluator = MagicMock()
        
        # Create cached evaluator
        self.cached_evaluator = CachedFitnessEvaluator(
            base_evaluator=self.mock_base_evaluator,
            cache_ttl=3600,
        )
        
        # Create test config
        self.config = EvolutionConfig(
            population_size=10,
            fitness_weights={
                "creativity_score": 0.4,
                "diversity_score": 0.3,
                "quality_score": 0.3,
            }
        )

    @pytest.mark.asyncio
    async def test_evaluate_individual_cache_miss(self) -> None:
        """Test evaluation when cache misses."""
        # Create test idea
        idea = GeneratedIdea(
            content="Novel test idea",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        # Mock base evaluator response
        expected_fitness = IndividualFitness(
            idea=idea,
            creativity_score=0.85,
            diversity_score=0.80,
            quality_score=0.90,
            overall_fitness=0.85,
        )
        self.mock_base_evaluator.evaluate_individual.return_value = expected_fitness
        
        # Evaluate
        result = await self.cached_evaluator.evaluate_individual(
            idea, self.config, "test_context"
        )
        
        # Should call base evaluator
        self.mock_base_evaluator.evaluate_individual.assert_called_once()
        assert result.creativity_score == expected_fitness.creativity_score
        
        # Check cache stats
        stats = self.cached_evaluator.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_individual_cache_hit(self) -> None:
        """Test evaluation when cache hits."""
        # Create test idea
        idea = GeneratedIdea(
            content="Cached test idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        # Mock base evaluator response
        expected_fitness = IndividualFitness(
            idea=idea,
            creativity_score=0.75,
            diversity_score=0.70,
            quality_score=0.80,
            overall_fitness=0.75,
        )
        self.mock_base_evaluator.evaluate_individual.return_value = expected_fitness
        
        # First evaluation (cache miss)
        result1 = await self.cached_evaluator.evaluate_individual(
            idea, self.config, "test_context"
        )
        
        # Second evaluation (cache hit)
        result2 = await self.cached_evaluator.evaluate_individual(
            idea, self.config, "test_context"
        )
        
        # Base evaluator should only be called once
        assert self.mock_base_evaluator.evaluate_individual.call_count == 1
        
        # Results should be identical
        assert result1.creativity_score == result2.creativity_score
        assert result1.diversity_score == result2.diversity_score
        assert result1.quality_score == result2.quality_score
        
        # Check cache stats
        stats = self.cached_evaluator.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_population_with_cache(self) -> None:
        """Test population evaluation with caching."""
        # Create test population with unique ideas first
        population = []
        for i in range(3):
            idea = GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            )
            population.append(idea)
        
        # Mock base evaluator responses for batch evaluation
        def create_fitness(idea, base_score):
            return IndividualFitness(
                idea=idea,
                creativity_score=base_score,
                diversity_score=base_score,
                quality_score=base_score,
                overall_fitness=base_score,
            )
        
        # First batch of fitness scores
        first_batch_scores = [create_fitness(population[i], 0.5 + i * 0.1) for i in range(3)]
        
        # Create new idea for second population
        new_idea = GeneratedIdea(
            content="New test idea",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        new_idea_fitness = create_fitness(new_idea, 0.9)
        
        # Mock population evaluation method since cached evaluator calls batch evaluation
        self.mock_base_evaluator.evaluate_population = AsyncMock(
            side_effect=[
                first_batch_scores,  # First batch returns all 3 scores
                [new_idea_fitness],  # Second batch returns only the new idea
            ]
        )
        
        # Evaluate first batch (should populate cache)
        first_results = await self.cached_evaluator.evaluate_population(
            population, self.config, "test_context"
        )
        
        assert len(first_results) == 3
        assert self.mock_base_evaluator.evaluate_population.call_count == 1  # Called once for batch
        
        # Now create a second population that includes some duplicates
        second_population = population[:2].copy()  # Reuse first 2 ideas (should be cache hits)
        second_population.append(new_idea)
        
        # Evaluate second batch (should have cache hits for first 2, miss for last)
        second_results = await self.cached_evaluator.evaluate_population(
            second_population, self.config, "test_context"
        )
        
        # Should call population evaluation 2 times total (first batch + second batch with only new idea)
        assert self.mock_base_evaluator.evaluate_population.call_count == 2
        assert len(second_results) == 3
        
        # Check that cached results match original results
        assert second_results[0].creativity_score == first_results[0].creativity_score
        assert second_results[1].creativity_score == first_results[1].creativity_score
        
        # Check final cache stats
        stats = self.cached_evaluator.get_cache_stats()
        assert stats["hits"] == 2  # Two cache hits for duplicates in second batch
        assert stats["misses"] == 4  # Three misses in first batch + one miss in second batch

    @pytest.mark.asyncio
    async def test_cache_disabled_mode(self) -> None:
        """Test evaluator works with caching disabled."""
        # Create evaluator with caching disabled
        evaluator = CachedFitnessEvaluator(
            base_evaluator=self.mock_base_evaluator,
            cache_ttl=0,  # Disable caching
        )
        
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.INDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        
        expected_fitness = IndividualFitness(
            idea=idea,
            overall_fitness=0.7,
        )
        self.mock_base_evaluator.evaluate_individual.return_value = expected_fitness
        
        # Evaluate twice
        result1 = await evaluator.evaluate_individual(idea, self.config, "context")
        result2 = await evaluator.evaluate_individual(idea, self.config, "context")
        
        # Should call base evaluator twice (no caching)
        assert self.mock_base_evaluator.evaluate_individual.call_count == 2

    def test_cache_key_includes_all_relevant_factors(self) -> None:
        """Test cache key generation includes all relevant factors."""
        idea = GeneratedIdea(
            content="Test idea with specific content",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            metadata={"extra": "data"},
        )
        
        # Different contexts should produce different keys
        key1 = self.cached_evaluator._generate_cache_key(idea, "context1")
        key2 = self.cached_evaluator._generate_cache_key(idea, "context2")
        assert key1 != key2
        
        # Config weights should NOT affect cache key (by design)
        # The implementation intentionally ignores config variations for consistency
        config1 = EvolutionConfig(fitness_weights={"creativity_score": 0.5})
        config2 = EvolutionConfig(fitness_weights={"creativity_score": 0.6})
        
        key3 = self.cached_evaluator._generate_cache_key(idea, "context", config1)
        key4 = self.cached_evaluator._generate_cache_key(idea, "context", config2)
        assert key3 == key4  # Same key regardless of config weights
        
        # Different ideas should produce different keys
        different_idea = GeneratedIdea(
            content="Different test idea content",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
        )
        key5 = self.cached_evaluator._generate_cache_key(different_idea, "context1")
        assert key1 != key5