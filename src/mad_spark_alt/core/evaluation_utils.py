"""
Shared utilities for evaluation metrics and common patterns.

This module contains reusable functions and classes to reduce code duplication
across various evaluators in the system.
"""

import asyncio
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar, TypedDict, Union, cast

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# TypedDict definitions for structured data
class GrammarMetricsDict(TypedDict):
    """Type definition for grammar metrics."""
    has_punctuation: float
    capitalization_score: float


class ReadabilityMetricsDict(TypedDict):
    """Type definition for readability metrics."""
    avg_sentence_length: float
    avg_word_length: float
    sentence_length_score: float
    word_length_score: float


class CodeStructureMetricsDict(TypedDict, total=False):
    """Type definition for code structure metrics."""
    total_lines: float
    code_lines: float
    blank_lines: float
    comment_lines: float
    indentation_ratio: float
    comment_ratio: float
    code_density: float
    code_structure_score: float  # Additional metric for quality evaluation


class TextAnalyzer:
    """Common text analysis utilities."""

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences using common punctuation marks.

        Args:
            text: Input text to split

        Returns:
            List of sentences (non-empty strings)
        """
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def calculate_distinct_n(text: str, n: int) -> float:
        """
        Calculate distinct n-gram ratio (diversity metric).

        Args:
            text: Input text
            n: N-gram size (1 for unigrams, 2 for bigrams, etc.)

        Returns:
            Ratio of unique n-grams to total n-grams
        """
        words = text.lower().split()
        if len(words) < n:
            return 1.0

        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)

        return len(unique_ngrams) / len(ngrams) if ngrams else 0.0

    @staticmethod
    def calculate_lexical_diversity(text: str) -> float:
        """
        Calculate lexical diversity (type-token ratio).

        Args:
            text: Input text

        Returns:
            Ratio of unique words to total words
        """
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    @staticmethod
    def calculate_word_frequency(text: str) -> Dict[str, int]:
        """
        Calculate word frequency distribution.

        Args:
            text: Input text

        Returns:
            Dictionary mapping words to their frequencies
        """
        words = text.lower().split()
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq

    @staticmethod
    def check_basic_grammar(text: str) -> GrammarMetricsDict:
        """
        Check basic grammar metrics.

        Args:
            text: Input text

        Returns:
            Dictionary with grammar-related scores
        """
        if not text.strip():
            return GrammarMetricsDict(has_punctuation=0.0, capitalization_score=0.0)

        # Check for basic punctuation
        has_punctuation = 1.0 if re.search(r"[.!?]", text) else 0.0

        # Check for proper capitalization
        sentences = TextAnalyzer.split_sentences(text)
        if sentences:
            uncapitalized = sum(1 for s in sentences if s and not s[0].isupper())
            capitalization_score = 1.0 - (uncapitalized / len(sentences))
        else:
            capitalization_score = 0.0

        return GrammarMetricsDict(
            has_punctuation=has_punctuation,
            capitalization_score=capitalization_score
        )

    @staticmethod
    def analyze_readability(text: str) -> ReadabilityMetricsDict:
        """
        Analyze text readability metrics.

        Args:
            text: Input text

        Returns:
            Dictionary with readability metrics
        """
        if not text.strip():
            return ReadabilityMetricsDict(
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                sentence_length_score=0.0,
                word_length_score=0.0,
            )

        sentences = TextAnalyzer.split_sentences(text)
        words = text.split()

        if not sentences or not words:
            return ReadabilityMetricsDict(
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                sentence_length_score=0.0,
                word_length_score=0.0,
            )

        # Average sentence length (words per sentence)
        avg_sentence_length = len(words) / len(sentences)

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Normalize scores (optimal ranges)
        sentence_score = 1.0 - min(abs(avg_sentence_length - 15) / 15, 1.0)
        word_score = 1.0 - min(abs(avg_word_length - 5) / 5, 1.0)

        return ReadabilityMetricsDict(
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            sentence_length_score=sentence_score,
            word_length_score=word_score,
        )


class CodeAnalyzer:
    """Common code analysis utilities."""

    @staticmethod
    def analyze_code_structure(code: str) -> CodeStructureMetricsDict:
        """
        Analyze basic code structure metrics.

        Args:
            code: Source code string

        Returns:
            Dictionary with code structure metrics
        """
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        total_lines = float(len(lines))
        code_lines = float(len(non_empty_lines))
        blank_lines = float(len(lines) - len(non_empty_lines))

        # Indentation consistency
        if non_empty_lines:
            indented_lines = [
                line for line in non_empty_lines if line.startswith((" ", "\t"))
            ]
            indentation_ratio = float(len(indented_lines) / len(non_empty_lines))
        else:
            indentation_ratio = 0.0

        # Comment analysis
        comment_lines = float(CodeAnalyzer._count_comment_lines(non_empty_lines))
        comment_ratio = (
            float(comment_lines / len(non_empty_lines)) if non_empty_lines else 0.0
        )

        # Code density
        code_density = float(len(non_empty_lines) / len(lines)) if lines else 0.0

        return CodeStructureMetricsDict(
            total_lines=total_lines,
            code_lines=code_lines,
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            indentation_ratio=indentation_ratio,
            comment_ratio=comment_ratio,
            code_density=code_density,
        )

    @staticmethod
    def _count_comment_lines(lines: List[str]) -> int:
        """Count lines that are comments."""
        comment_count = 0
        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()

            # Python-style comments
            if stripped.startswith("#"):
                comment_count += 1
                continue

            # C-style single line comments
            if stripped.startswith("//"):
                comment_count += 1
                continue

            # Multi-line comment handling (simplified)
            if '"""' in stripped or "'''" in stripped:
                comment_count += 1
                in_multiline_comment = not in_multiline_comment
            elif in_multiline_comment:
                comment_count += 1

        return comment_count


class ScoreAggregator:
    """Utilities for aggregating evaluation scores."""

    @staticmethod
    def calculate_weighted_average(
        scores: Dict[str, float], weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted average of scores.

        Args:
            scores: Dictionary of score names to values
            weights: Optional dictionary of score names to weights
                    If None, equal weights are used

        Returns:
            Weighted average score
        """
        if not scores:
            return 0.0

        if weights is None:
            return sum(scores.values()) / len(scores)

        weighted_sum = 0.0
        total_weight = 0.0

        for key, value in scores.items():
            weight = weights.get(key, 1.0)
            weighted_sum += value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def aggregate_results(
        results: List[Dict[str, float]], aggregation_methods: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Aggregate multiple result dictionaries.

        Args:
            results: List of score dictionaries
            aggregation_methods: List of methods to use (mean, max, min, std)
                                If None, uses ["mean", "max", "min"]

        Returns:
            Dictionary with aggregated scores
        """
        if not results:
            return {}

        if aggregation_methods is None:
            aggregation_methods = ["mean", "max", "min"]

        # Collect all unique keys
        all_keys: Set[str] = set()
        for result in results:
            all_keys.update(result.keys())

        aggregated = {}

        for key in all_keys:
            values = [r.get(key, 0.0) for r in results if key in r]
            if not values:
                continue

            if "mean" in aggregation_methods:
                aggregated[f"{key}_mean"] = float(np.mean(values))
            if "max" in aggregation_methods:
                aggregated[f"{key}_max"] = float(np.max(values))
            if "min" in aggregation_methods:
                aggregated[f"{key}_min"] = float(np.min(values))
            if "std" in aggregation_methods:
                aggregated[f"{key}_std"] = float(np.std(values))

        return aggregated


class AsyncBatchProcessor:
    """Utilities for async batch processing with concurrency control."""

    @staticmethod
    async def process_batch_with_semaphore(
        items: List[T],
        process_func: Callable[[T], Awaitable[R]],
        max_concurrent: int = 5,
        return_exceptions: bool = True,
    ) -> List[Union[R, BaseException]]:
        """
        Process items in batch with concurrency limit.

        Args:
            items: List of items to process
            process_func: Async function to process each item
            max_concurrent: Maximum concurrent tasks
            return_exceptions: Whether to return exceptions or raise them

        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_process(item: T) -> R:
            async with semaphore:
                return await process_func(item)

        tasks = [bounded_process(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    @staticmethod
    async def process_with_timeout(
        coroutine: Any,  # Coroutine
        timeout: float,
        default_result: Optional[Any] = None,
    ) -> Any:
        """
        Execute coroutine with timeout.

        Args:
            coroutine: Coroutine to execute
            timeout: Timeout in seconds
            default_result: Result to return on timeout

        Returns:
            Coroutine result or default_result on timeout
        """
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return default_result


class CacheKeyGenerator:
    """Utilities for generating cache keys."""

    @staticmethod
    def generate_text_key(text: str, prefix: str = "") -> str:
        """
        Generate cache key for text content.

        Args:
            text: Text content
            prefix: Optional prefix for the key

        Returns:
            Cache key string
        """
        # Use hash and length for uniqueness
        text_hash = hash(text)
        text_len = len(text)

        if prefix:
            return f"{prefix}_{text_hash}_{text_len}"
        return f"{text_hash}_{text_len}"

    @staticmethod
    def generate_composite_key(items: List[Any], prefix: str = "") -> str:
        """
        Generate cache key for multiple items.

        Args:
            items: List of items to include in key
            prefix: Optional prefix for the key

        Returns:
            Cache key string
        """
        item_hashes = [hash(str(item)) for item in items]
        composite_hash = hash(tuple(item_hashes))

        if prefix:
            return f"{prefix}_{composite_hash}"
        return str(composite_hash)


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to the range [min_val, max_val].

    Args:
        value: Value to normalize
        min_val: Minimum value of the range
        max_val: Maximum value of the range

    Returns:
        Normalized value
    """
    return max(min_val, min(max_val, value))


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change (can be negative)
    """
    if old_value == 0:
        return 100.0 if new_value > 0 else 0.0
    return ((new_value - old_value) / abs(old_value)) * 100.0
