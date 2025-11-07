"""
Caching functionality for semantic operators.

This module provides an enhanced in-memory cache for semantic operator results
with intelligent cache key clustering and session-based TTL management.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .semantic_utils import (
    CACHE_MAX_SIZE,
    SIMILARITY_KEY_LENGTH,
    SIMILARITY_CONTENT_PREFIX_LENGTH,
    SIMILARITY_WORDS_COUNT,
    SESSION_TTL_EXTENSION_RATE,
    MAX_SESSION_TTL_EXTENSION,
    STOP_WORDS,
)

logger = logging.getLogger(__name__)


class SemanticOperatorCache:
    """
    Enhanced in-memory cache for semantic operator results with session-based TTL.

    Reduces redundant LLM calls by caching mutation and crossover results with
    intelligent cache key clustering and extended session-based TTL.
    """

    def __init__(self, ttl_seconds: int = 10800):  # Extended to 3 hours for longer evolution sessions
        """
        Initialize cache with enhanced session-based time-to-live.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 3 hours)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # key -> (result_data, timestamp)
        self._similarity_index: Dict[str, List[str]] = {}  # Map similarity keys to cache keys
        self._session_start = time.time()  # Track session for extended caching

    def _get_cache_key(self, content: str, operation_type: str = "default") -> str:
        """
        Generate consistent cache key with operation type for better clustering.

        Args:
            content: Content to generate key for
            operation_type: Type of operation (mutation, crossover, etc.) for clustering
        """
        # Include operation type in key for better cache organization
        combined_content = f"{operation_type}:{content}"
        return hashlib.md5(combined_content.encode()).hexdigest()

    def _get_similarity_key(self, content: str) -> str:
        """
        Generate similarity-based key for cache clustering.
        Uses first 50 characters of normalized content for similarity matching.
        """
        # Normalize content for similarity matching
        normalized = content.lower().strip()[:SIMILARITY_CONTENT_PREFIX_LENGTH]
        # Remove common words that don't affect semantic meaning
        words = [w for w in normalized.split() if w not in STOP_WORDS]
        key_content = ' '.join(words[:SIMILARITY_WORDS_COUNT])
        return hashlib.md5(key_content.encode()).hexdigest()[:SIMILARITY_KEY_LENGTH]

    def _get_effective_ttl(self, current_time: float) -> float:
        """Calculate effective TTL with session-based extension."""
        session_duration = current_time - self._session_start
        return self.ttl_seconds + min(
            session_duration * SESSION_TTL_EXTENSION_RATE,
            MAX_SESSION_TTL_EXTENSION
        )

    def get(self, content: str, operation_type: str = "default", return_dict: bool = True) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Get cached result with enhanced lookup including similarity matching.

        Args:
            content: Original content to look up
            operation_type: Operation type for cache clustering
            return_dict: If False, return just content string for backward compatibility

        Returns:
            Cached result (dict with metadata or string based on return_dict) or None if not found/expired
        """
        # Try exact match first
        exact_key = self._get_cache_key(content, operation_type)

        if exact_key in self._cache:
            value, timestamp = self._cache[exact_key]

            # Check if expired (extended session-based TTL)
            current_time = time.time()
            effective_ttl = self._get_effective_ttl(current_time)

            if current_time - timestamp < effective_ttl:
                logger.debug(f"Cache exact hit for {operation_type} hash {exact_key[:8]}")
                # Return based on requested format
                if return_dict:
                    return value
                else:
                    # Backward compatibility - return just content string
                    if isinstance(value, dict):
                        return str(value.get("content", ""))
                    else:
                        return str(value)
            else:
                # Remove expired entry
                del self._cache[exact_key]
                logger.debug(f"Cache expired for {operation_type} hash {exact_key[:8]}")

        # Try similarity-based lookup for mutation operations
        # Store similarity keys separately for efficient lookup
        if operation_type == "mutation" and hasattr(self, '_similarity_index'):
            similarity_key = self._get_similarity_key(content)
            if similarity_key in self._similarity_index:
                # Get all cache keys with this similarity
                current_time = time.time()  # Define current_time if not already defined
                effective_ttl = self._get_effective_ttl(current_time)
                for cache_key in self._similarity_index[similarity_key]:
                    if cache_key in self._cache:
                        cached_value, timestamp = self._cache[cache_key]
                        if current_time - timestamp < effective_ttl:  # Reuse calculated TTL
                            logger.debug(f"Cache similarity hit for {operation_type} hash {cache_key[:8]}")
                            # Return based on requested format
                            if return_dict:
                                return cached_value
                            else:
                                if isinstance(cached_value, dict):
                                    return str(cached_value.get("content", ""))
                                else:
                                    return str(cached_value)

        return None

    def put(self, content: str, result: Union[str, Dict[str, Any]], operation_type: str = "default") -> None:
        """
        Store result in enhanced cache with operation type clustering.

        Args:
            content: Original content
            result: Result data (string for backward compatibility or dict with metadata)
            operation_type: Operation type for cache clustering
        """
        key = self._get_cache_key(content, operation_type)

        # Convert string to dict format for consistency
        if isinstance(result, str):
            result_data = {"content": result, "mutation_type": operation_type}
        else:
            result_data = result

        self._cache[key] = (result_data, time.time())
        logger.debug(f"Cached {operation_type} result for hash {key[:8]}")

        # Update similarity index for mutation operations
        if operation_type == "mutation":
            similarity_key = self._get_similarity_key(content)
            if similarity_key not in self._similarity_index:
                self._similarity_index[similarity_key] = []
            self._similarity_index[similarity_key].append(key)

        # Periodic cache cleanup to prevent memory growth
        if len(self._cache) > CACHE_MAX_SIZE:
            self._cleanup_expired_entries()

    def _cleanup_expired_entries(self) -> None:
        """Clean up expired cache entries to manage memory usage."""
        current_time = time.time()
        effective_ttl = self._get_effective_ttl(current_time)
        expired_keys = []

        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp >= effective_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        # Clean up similarity index
        for sim_key in list(self._similarity_index.keys()):
            self._similarity_index[sim_key] = [k for k in self._similarity_index[sim_key] if k not in expired_keys]
            if not self._similarity_index[sim_key]:
                del self._similarity_index[sim_key]

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring performance."""
        current_time = time.time()
        effective_ttl = self._get_effective_ttl(current_time)
        valid_entries = sum(1 for _, (_, timestamp) in self._cache.items()
                           if current_time - timestamp < effective_ttl)

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "session_duration_minutes": int((current_time - self._session_start) / 60),
        }
