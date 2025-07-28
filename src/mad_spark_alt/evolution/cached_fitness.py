"""
Cached fitness evaluation for genetic evolution.

This module provides caching functionality for fitness evaluations to avoid
redundant LLM calls and improve performance of the evolution system.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
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

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize fitness cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_size: Maximum number of entries in cache (LRU eviction when exceeded)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, FitnessCacheEntry] = {}
        self._access_order: List[str] = []  # For LRU tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "ttl_evictions": 0,
            "lru_evictions": 0,
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
        # Normalize content for better cache hits
        normalized_content = self._normalize_content(idea.content)
        
        # Include relevant idea attributes in key
        key_parts = [
            normalized_content,
            str(idea.thinking_method.value),
            idea.agent_name,
            str(context) if context else "",
        ]

        # Create hash of combined parts
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"fitness_{key_hash}"
    
    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for cache key generation to improve hit rates.
        
        Args:
            content: The idea content to normalize
            
        Returns:
            Normalized content string
        """
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Remove common punctuation variations
        normalized = normalized.replace("**", "")  # Remove markdown bold
        normalized = normalized.replace("*", "")   # Remove markdown emphasis
        
        # Sort words to handle minor rephrasing
        # Only do this for short content to avoid losing meaning
        if len(normalized.split()) <= 20:
            words = sorted(normalized.split())
            normalized = " ".join(words)
        
        return normalized

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
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats["evictions"] += 1
            self._stats["ttl_evictions"] += 1
            self._stats["misses"] += 1
            return None

        # Cache hit - update access order and stats
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
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
        # Update access order for LRU
        if key in self._cache:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Store entry
        self._cache[key] = FitnessCacheEntry(
            fitness=fitness,
            timestamp=time.time(),
        )

        # Enforce cache size limit with LRU eviction
        self._enforce_cache_limits()

    def _enforce_cache_limits(self) -> None:
        """Enforce cache size limits using LRU eviction."""
        while len(self._cache) > self.max_size:
            if not self._access_order:
                break

            # Remove least recently used item
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats["evictions"] += 1
                self._stats["lru_evictions"] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "ttl_evictions": 0,
            "lru_evictions": 0,
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
    
    def get_similar(self, idea: GeneratedIdea, context: Optional[str] = None, 
                    similarity_threshold: float = 0.8) -> Optional[IndividualFitness]:
        """
        Find a similar cached entry using semantic similarity.
        
        Args:
            idea: The idea to find similar cached entries for
            context: Optional evaluation context
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            Cached fitness for most similar idea, or None
        """
        if not self._cache:
            return None
        
        # Normalize the target idea content
        target_content = self._normalize_content(idea.content)
        target_words = set(target_content.split())
        
        if not target_words:
            return None
        
        best_match: Optional[Tuple[str, float]] = None
        best_similarity = 0.0
        
        # Search through cache for similar entries
        expired_keys = []
        for key, entry in self._cache.items():
            # Remove expired entries to prevent cache bloat
            if time.time() - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
                continue
            
            # Compare idea content similarity
            cached_idea = entry.fitness.idea
            cached_content = self._normalize_content(cached_idea.content)
            cached_words = set(cached_content.split())
            
            if not cached_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(target_words.intersection(cached_words))
            union = len(target_words.union(cached_words))
            similarity = intersection / union if union > 0 else 0.0
            
            # Check if this is the best match so far
            if similarity > best_similarity and similarity >= similarity_threshold:
                # Also check that thinking method matches
                if cached_idea.thinking_method == idea.thinking_method:
                    best_match = (key, similarity)
                    best_similarity = similarity
        
        # Clean up expired entries found during search
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._stats["evictions"] += 1
            self._stats["ttl_evictions"] += 1
        
        if best_match:
            key, similarity = best_match
            logger.debug(f"Found similar cached entry with {similarity:.2%} similarity")
            return self.get(key)
        
        return None


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
        Evaluate fitness for entire population with caching and batch optimization.

        Args:
            population: List of ideas to evaluate
            config: Evolution configuration
            context: Optional evaluation context

        Returns:
            List of individual fitness scores
        """
        if not self.cache_enabled:
            # No cache, use base evaluator directly with batch optimization
            return await self.base_evaluator.evaluate_population(population, config, context)
        
        # With cache enabled, separate cached and uncached ideas
        cached_results: List[Optional[IndividualFitness]] = []
        uncached_ideas: List[GeneratedIdea] = []
        uncached_indices: List[int] = []
        
        # Check cache for each idea
        for i, idea in enumerate(population):
            cache_key = self._generate_cache_key(idea, context, config)
            cached_result = self.cache.get(cache_key) if self.cache else None
            
            # If exact match not found, try semantic similarity
            if cached_result is None and self.cache:
                cached_result = self.cache.get_similar(idea, context, similarity_threshold=0.7)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for idea {i}: {idea.content[:50]}...")
                cached_results.append(cached_result)
            else:
                logger.debug(f"Cache miss for idea {i}: {idea.content[:50]}...")
                cached_results.append(None)
                uncached_ideas.append(idea)
                uncached_indices.append(i)
        
        # Evaluate uncached ideas in batch if there are any
        if uncached_ideas:
            # Use base evaluator's batch implementation
            uncached_fitness = await self.base_evaluator.evaluate_population(
                uncached_ideas, config, context
            )
            
            # Store results in cache and merge with cached results
            for idx, (orig_idx, fitness) in enumerate(zip(uncached_indices, uncached_fitness)):
                idea = uncached_ideas[idx]
                cache_key = self._generate_cache_key(idea, context, config)
                if self.cache:
                    self.cache.set(cache_key, fitness)
                cached_results[orig_idx] = fitness
        
        # Log cache performance
        if self.cache:
            stats = self.cache.get_stats()
            logger.info(
                f"Cache stats - Hits: {stats['hits']}, Misses: {stats['misses']}, "
                f"Hit rate: {stats['hit_rate']:.2%}"
            )
        
        # Ensure all positions have valid fitness results, maintaining order
        # IMPORTANT: Do NOT filter None values as it breaks index correspondence
        final_results: List[IndividualFitness] = []
        for i, result in enumerate(cached_results):
            if result is None:
                logger.error(f"Missing fitness result for idea at index {i}")
                # Use default fitness to maintain position
                final_results.append(IndividualFitness(
                    idea=population[i],
                    creativity_score=0.0,
                    diversity_score=0.0,
                    quality_score=0.0,
                    overall_fitness=0.0,
                    evaluation_metadata={"error": "Evaluation failed for this idea"}
                ))
            else:
                final_results.append(result)
        
        return final_results

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
        
        # Use normalized context to improve cache hits
        normalized_context = None
        if context:
            # Extract key parts of context, ignoring minor variations
            context_lower = context.lower()
            if "food waste" in context_lower:
                normalized_context = "reduce_food_waste"
            elif "climate" in context_lower:
                normalized_context = "climate_solutions"
            else:
                # For other contexts, just use first 20 chars
                normalized_context = context_lower[:20]
        
        base_key = self.cache.generate_key(idea, normalized_context)

        # Don't include config weights in key - fitness evaluation should be consistent
        # regardless of slight weight variations
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
