"""
Semantic cache for evolution system.

This module provides semantic similarity-based caching to improve
cache hit rates by recognizing similar ideas.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    np = None  # type: ignore

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.cached_fitness import FitnessCache, FitnessCacheEntry
from mad_spark_alt.evolution.interfaces import IndividualFitness

logger = logging.getLogger(__name__)


class SemanticCache(FitnessCache):
    """
    Enhanced fitness cache with semantic similarity matching.

    This cache uses text similarity to find cached results for
    semantically similar ideas, improving cache hit rates.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        max_candidates: int = 10,
    ):
        """
        Initialize semantic cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum cache size
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_candidates: Maximum candidates to check for similarity
        """
        super().__init__(ttl_seconds, max_size)
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates

        # Text vectorizer for similarity computation
        if HAS_SKLEARN:
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 2),
            )
        else:
            self._vectorizer = None

        # Store idea content for similarity matching
        self._idea_contents: Dict[str, str] = {}
        self._idea_vectors: Dict[str, np.ndarray] = {}
        self._is_fitted = False

    def generate_semantic_key(self, idea: GeneratedIdea) -> str:
        """
        Generate a semantic-aware cache key.

        First attempts to find semantically similar cached ideas,
        falling back to standard key generation.

        Args:
            idea: Idea to generate key for

        Returns:
            Cache key (existing similar or new)
        """
        # Check for semantic matches
        similar_key = self._find_similar_cached_idea(idea)
        if similar_key:
            logger.debug(f"Found semantic match with key: {similar_key}")
            return similar_key

        # Generate new key for this idea
        return self.generate_key(idea)

    def set(self, key: str, fitness: IndividualFitness) -> None:
        """
        Store fitness result with semantic indexing.

        Args:
            key: Cache key
            fitness: Fitness result to cache
        """
        super().set(key, fitness)

        # Store idea content for semantic matching
        idea_content = fitness.idea.content
        self._idea_contents[key] = idea_content

        # Update vectorizer if needed
        self._update_vectors()

    def get(self, key: str) -> Optional[IndividualFitness]:
        """
        Retrieve a fitness score from cache with vector cleanup on TTL eviction.

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

            # Clean up semantic data for expired entry
            if key in self._idea_contents:
                del self._idea_contents[key]
            if key in self._idea_vectors:
                del self._idea_vectors[key]

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

    def get_semantic(self, idea: GeneratedIdea) -> Optional[IndividualFitness]:
        """
        Retrieve fitness using semantic similarity.

        Args:
            idea: Idea to look up

        Returns:
            Cached fitness or None
        """
        key = self.generate_semantic_key(idea)
        return self.get(key)

    def compute_similarity(self, idea1: GeneratedIdea, idea2: GeneratedIdea) -> float:
        """
        Compute semantic similarity between two ideas.

        Args:
            idea1: First idea
            idea2: Second idea

        Returns:
            Similarity score (0-1)
        """
        if not self._is_fitted:
            return 0.0

        try:
            # Vectorize both ideas
            vec1 = self._vectorizer.transform([idea1.content])
            vec2 = self._vectorizer.transform([idea2.content])

            # Compute cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0, 0]

            # Boost similarity if same thinking method
            if idea1.thinking_method == idea2.thinking_method:
                similarity = similarity * 1.1  # 10% boost

            return float(min(similarity, 1.0))

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def _find_similar_cached_idea(self, idea: GeneratedIdea) -> Optional[str]:
        """
        Find the most similar cached idea above threshold.

        Args:
            idea: Idea to match

        Returns:
            Cache key of similar idea or None
        """
        if not self._idea_contents or not self._is_fitted:
            return None

        try:
            # Vectorize the query idea
            query_vector = self._vectorizer.transform([idea.content])

            # Get candidate keys (most recent first)
            candidate_keys = list(self._idea_contents.keys())[-self.max_candidates :]

            best_similarity = 0.0
            best_key = None

            for key in candidate_keys:
                # Skip if not in cache anymore
                if key not in self._idea_vectors:
                    continue

                # Compute similarity
                candidate_vector = self._idea_vectors[key].reshape(1, -1)
                similarity = cosine_similarity(query_vector, candidate_vector)[0, 0]

                # Check if this is the best match
                if (
                    similarity > best_similarity
                    and similarity >= self.similarity_threshold
                ):
                    best_similarity = similarity
                    best_key = key

            return best_key

        except Exception as e:
            logger.error(f"Error finding similar ideas: {e}")
            return None

    def _update_vectors(self) -> None:
        """Update TF-IDF vectors for all cached ideas."""
        if len(self._idea_contents) < 2:
            return

        try:
            # Get all idea contents
            contents = list(self._idea_contents.values())
            keys = list(self._idea_contents.keys())

            # Fit vectorizer if not already fitted
            if not self._is_fitted:
                self._vectorizer.fit(contents)
                self._is_fitted = True

            # Transform all contents
            vectors = self._vectorizer.transform(contents)

            # Store vectors
            for i, key in enumerate(keys):
                self._idea_vectors[key] = vectors[i].toarray().flatten()

        except Exception as e:
            logger.error(f"Error updating vectors: {e}")

    def _enforce_cache_limits(self) -> None:
        """Enforce cache size limits using LRU eviction with vector cleanup."""
        while len(self._cache) > self.max_size:
            if not self._access_order:
                break

            # Remove least recently used item
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]

                # Clean up semantic data for evicted entry
                if lru_key in self._idea_contents:
                    del self._idea_contents[lru_key]
                if lru_key in self._idea_vectors:
                    del self._idea_vectors[lru_key]

                self._stats["evictions"] += 1
                self._stats["lru_evictions"] += 1

    def clear(self) -> None:
        """Clear cache and semantic indices."""
        super().clear()
        self._idea_contents.clear()
        self._idea_vectors.clear()
        self._is_fitted = False

    def get_semantic_stats(self) -> Dict[str, Any]:
        """
        Get semantic cache statistics.

        Returns:
            Statistics including similarity distribution
        """
        base_stats = self.get_stats()

        semantic_stats = {
            **base_stats,
            "indexed_ideas": len(self._idea_contents),
            "vectorizer_fitted": self._is_fitted,
            "similarity_threshold": self.similarity_threshold,
        }

        return semantic_stats


class SemanticClusterCache:
    """
    Advanced semantic cache using idea clustering.

    Groups similar ideas into clusters for more efficient
    similarity matching and better cache utilization.
    """

    def __init__(
        self,
        n_clusters: int = 20,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
    ):
        """
        Initialize cluster-based cache.

        Args:
            n_clusters: Number of idea clusters
            ttl_seconds: Cache entry TTL
            max_size: Maximum cache size
        """
        self.n_clusters = n_clusters
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

        # Cluster assignments
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_caches: Dict[int, FitnessCache] = {}

        # Vectorizer for clustering
        self._vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
        )

        self._is_fitted = False

    def get_cluster_id(self, idea: GeneratedIdea) -> int:
        """
        Get cluster ID for an idea.

        Args:
            idea: Idea to cluster

        Returns:
            Cluster ID (0 to n_clusters-1)
        """
        if not self._is_fitted:
            return 0

        try:
            # Vectorize idea
            idea_vector = self._vectorizer.transform([idea.content])

            # Find nearest cluster
            distances = cosine_similarity(idea_vector, self._cluster_centers)[0]
            cluster_id = int(np.argmax(distances))

            return cluster_id

        except Exception as e:
            logger.error(f"Error assigning cluster: {e}")
            return 0

    def get(self, idea: GeneratedIdea) -> Optional[IndividualFitness]:
        """
        Get cached fitness from appropriate cluster.

        Args:
            idea: Idea to look up

        Returns:
            Cached fitness or None
        """
        cluster_id = self.get_cluster_id(idea)

        if cluster_id not in self._cluster_caches:
            return None

        cache = self._cluster_caches[cluster_id]
        key = cache.generate_key(idea)

        return cache.get(key)

    def set(self, idea: GeneratedIdea, fitness: IndividualFitness) -> None:
        """
        Store fitness in appropriate cluster cache.

        Args:
            idea: Idea evaluated
            fitness: Fitness result
        """
        cluster_id = self.get_cluster_id(idea)

        # Create cluster cache if needed
        if cluster_id not in self._cluster_caches:
            cluster_size = self.max_size // self.n_clusters
            self._cluster_caches[cluster_id] = FitnessCache(
                ttl_seconds=self.ttl_seconds,
                max_size=max(10, cluster_size),
            )

        cache = self._cluster_caches[cluster_id]
        key = cache.generate_key(idea)
        cache.set(key, fitness)

    def update_clusters(self, all_ideas: List[GeneratedIdea]) -> None:
        """
        Update clustering based on all ideas seen.

        Args:
            all_ideas: List of all ideas to cluster
        """
        if len(all_ideas) < self.n_clusters:
            return

        try:
            # Vectorize all ideas
            contents = [idea.content for idea in all_ideas]
            vectors = self._vectorizer.fit_transform(contents)

            # Perform clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            kmeans.fit(vectors)

            # Store cluster centers
            self._cluster_centers = kmeans.cluster_centers_
            self._is_fitted = True

            logger.info(f"Updated {self.n_clusters} idea clusters")

        except Exception as e:
            logger.error(f"Error updating clusters: {e}")

    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics for each cluster.

        Returns:
            Cluster statistics
        """
        stats = {}

        for cluster_id, cache in self._cluster_caches.items():
            cache_stats = cache.get_stats()
            stats[f"cluster_{cluster_id}"] = {
                "size": cache_stats["size"],
                "hit_rate": cache_stats["hit_rate"],
                "hits": cache_stats["hits"],
                "misses": cache_stats["misses"],
            }

        return stats
