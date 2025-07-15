"""
Cached fitness evaluation for genetic evolution.

This module provides caching functionality for fitness evaluations to avoid
redundant LLM calls and improve performance of the evolution system.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    IndividualFitness,
)

logger = logging.getLogger(__name__)


@dataclass
class FitnessCacheEntry:
    """Cache entry for fitness evaluation results."""
    
    fitness: IndividualFitness
    timestamp: float
    hit_count: int = 0


class FitnessCache:
    """
    In-memory cache for fitness evaluation results.
    
    This cache stores fitness scores for ideas to avoid redundant evaluations.
    Entries expire after a configurable TTL.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize fitness cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, FitnessCacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def generate_key(self, idea: GeneratedIdea, context: Optional[str] = None) -> str:
        """
        Generate a cache key for an idea and context.
        
        Args:
            idea: The idea to generate a key for
            context: Optional evaluation context
            
        Returns:
            Cache key string
        """
        # Include relevant idea attributes in key
        key_parts = [
            idea.content,
            str(idea.thinking_method.value),
            idea.agent_name,
            str(context) if context else "",
        ]
        
        # Create hash of combined parts
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"fitness_{key_hash}"
    
    def get(self, key: str) -> Optional[IndividualFitness]:
        """
        Retrieve a fitness score from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached fitness score or None if not found/expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self._cache[key]
            self._stats["evictions"] += 1
            self._stats["misses"] += 1
            return None
        
        # Update hit count and stats
        entry.hit_count += 1
        self._stats["hits"] += 1
        
        return entry.fitness
    
    def set(self, key: str, fitness: IndividualFitness) -> None:
        """
        Store a fitness score in cache.
        
        Args:
            key: Cache key
            fitness: Fitness score to cache
        """
        self._cache[key] = FitnessCacheEntry(
            fitness=fitness,
            timestamp=time.time(),
        )
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats including hit rate
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }


class CachedFitnessEvaluator:
    """
    Fitness evaluator with caching support.
    
    This evaluator wraps a base fitness evaluator and adds caching to avoid
    redundant evaluations of the same ideas.
    """
    
    def __init__(
        self,
        base_evaluator: Optional[FitnessEvaluator] = None,
        cache_ttl: int = 3600,
    ):
        """
        Initialize cached fitness evaluator.
        
        Args:
            base_evaluator: Base evaluator to wrap (creates default if None)
            cache_ttl: Cache time-to-live in seconds (0 to disable caching)
        """
        self.base_evaluator = base_evaluator or FitnessEvaluator()
        self.cache_enabled = cache_ttl > 0
        self.cache = FitnessCache(ttl_seconds=cache_ttl) if self.cache_enabled else None
    
    async def evaluate_individual(
        self,
        idea: GeneratedIdea,
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> IndividualFitness:
        """
        Evaluate fitness of a single idea with caching.
        
        Args:
            idea: Idea to evaluate
            config: Evolution configuration
            context: Optional evaluation context
            
        Returns:
            Individual fitness score
        """
        # Check cache if enabled
        if self.cache_enabled and self.cache:
            cache_key = self._generate_cache_key(idea, context, config)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for idea: {idea.content[:50]}...")
                return cached_result
        
        # Evaluate using base evaluator
        logger.debug(f"Cache miss, evaluating idea: {idea.content[:50]}...")
        fitness = await self.base_evaluator.evaluate_individual(idea, config, context)
        
        # Store in cache if enabled
        if self.cache_enabled and self.cache:
            self.cache.set(cache_key, fitness)
        
        return fitness
    
    async def evaluate_population(
        self,
        population: List[GeneratedIdea],
        config: EvolutionConfig,
        context: Optional[str] = None,
    ) -> List[IndividualFitness]:
        """
        Evaluate fitness for entire population with caching.
        
        Args:
            population: List of ideas to evaluate
            config: Evolution configuration
            context: Optional evaluation context
            
        Returns:
            List of individual fitness scores
        """
        results = []
        
        # Process each individual
        for idea in population:
            fitness = await self.evaluate_individual(idea, config, context)
            results.append(fitness)
        
        # Log cache performance
        if self.cache_enabled and self.cache:
            stats = self.cache.get_stats()
            logger.info(
                f"Cache stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                f"Hit rate: {stats['hit_rate']:.2%}"
            )
        
        return results
    
    async def calculate_population_diversity(
        self, population: List[IndividualFitness]
    ) -> float:
        """
        Calculate population diversity score.
        
        Delegates to base evaluator as diversity calculation doesn't benefit from caching.
        
        Args:
            population: List of evaluated individuals
            
        Returns:
            Diversity score between 0 and 1
        """
        return await self.base_evaluator.calculate_population_diversity(population)
    
    def _generate_cache_key(
        self,
        idea: GeneratedIdea,
        context: Optional[str] = None,
        config: Optional[EvolutionConfig] = None,
    ) -> str:
        """
        Generate a comprehensive cache key including all relevant factors.
        
        Args:
            idea: The idea to generate a key for
            context: Optional evaluation context
            config: Optional evolution configuration
            
        Returns:
            Cache key string
        """
        # Base key from idea and context
        if not self.cache:
            return "no_cache"
        base_key = self.cache.generate_key(idea, context)
        
        # Add config-specific elements if provided
        if config and config.fitness_weights:
            # Sort weights for consistent ordering
            weights_str = json.dumps(config.fitness_weights, sort_keys=True)
            weights_hash = hashlib.sha256(weights_str.encode()).hexdigest()[:8]
            base_key = f"{base_key}_{weights_hash}"
        
        return base_key
    
    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        if self.cache_enabled and self.cache:
            return self.cache.get_stats()
        return {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0,
            "size": 0,
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.cache_enabled and self.cache:
            self.cache.clear()
            logger.info("Fitness cache cleared")