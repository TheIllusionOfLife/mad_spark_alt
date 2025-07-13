"""
Tests for evaluation utility functions.
"""

import asyncio
from typing import Any, List

import pytest

from mad_spark_alt.core.evaluation_utils import (
    AsyncBatchProcessor,
    CacheKeyGenerator,
    CodeAnalyzer,
    ScoreAggregator,
    TextAnalyzer,
    calculate_percentage_change,
    normalize_score,
)


class TestTextAnalyzer:
    """Tests for TextAnalyzer utilities."""

    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "Hello world. How are you? I am fine! Thanks."
        sentences = TextAnalyzer.split_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "Hello world"
        assert sentences[1] == "How are you"
        assert sentences[2] == "I am fine"
        assert sentences[3] == "Thanks"

    def test_split_sentences_empty(self):
        """Test sentence splitting with empty text."""
        assert TextAnalyzer.split_sentences("") == []
        assert TextAnalyzer.split_sentences("   ") == []

    def test_calculate_distinct_n(self):
        """Test distinct n-gram calculation."""
        text = "the cat sat on the mat"
        
        # Unigrams
        assert TextAnalyzer.calculate_distinct_n(text, 1) == 5/6  # 5 unique words, 6 total
        
        # Bigrams
        bigram_ratio = TextAnalyzer.calculate_distinct_n(text, 2)
        assert 0.8 <= bigram_ratio <= 1.0  # Most bigrams are unique
        
        # Text shorter than n
        assert TextAnalyzer.calculate_distinct_n("word", 5) == 1.0

    def test_calculate_lexical_diversity(self):
        """Test lexical diversity calculation."""
        # All unique words
        assert TextAnalyzer.calculate_lexical_diversity("one two three") == 1.0
        
        # Some repetition
        assert TextAnalyzer.calculate_lexical_diversity("the cat and the dog") == 4/5
        
        # Empty text
        assert TextAnalyzer.calculate_lexical_diversity("") == 0.0

    def test_calculate_word_frequency(self):
        """Test word frequency calculation."""
        text = "the cat and the dog and the bird"
        freq = TextAnalyzer.calculate_word_frequency(text)
        
        assert freq["the"] == 3
        assert freq["and"] == 2
        assert freq["cat"] == 1
        assert freq["dog"] == 1
        assert freq["bird"] == 1

    def test_check_basic_grammar(self):
        """Test basic grammar checking."""
        # Good grammar
        good_text = "This is a sentence. Another one here!"
        scores = TextAnalyzer.check_basic_grammar(good_text)
        assert scores["has_punctuation"] == 1.0
        assert scores["capitalization_score"] == 1.0
        
        # Missing punctuation
        no_punct = "This is a sentence without punctuation"
        scores = TextAnalyzer.check_basic_grammar(no_punct)
        assert scores["has_punctuation"] == 0.0
        
        # Poor capitalization
        poor_caps = "this is bad. another bad one."
        scores = TextAnalyzer.check_basic_grammar(poor_caps)
        assert scores["capitalization_score"] < 1.0

    def test_analyze_readability(self):
        """Test readability analysis."""
        # Normal text
        text = "This is a normal sentence. It has reasonable length and complexity."
        metrics = TextAnalyzer.analyze_readability(text)
        
        assert metrics["avg_sentence_length"] > 0
        assert metrics["avg_word_length"] > 0
        assert 0 <= metrics["sentence_length_score"] <= 1.0
        assert 0 <= metrics["word_length_score"] <= 1.0
        
        # Empty text
        empty_metrics = TextAnalyzer.analyze_readability("")
        assert empty_metrics["avg_sentence_length"] == 0.0


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer utilities."""

    def test_analyze_code_structure(self):
        """Test code structure analysis."""
        code = """
def hello():
    # This is a comment
    print("Hello, world!")
    
    # Another comment
    return True

# Main execution
if __name__ == "__main__":
    hello()
"""
        metrics = CodeAnalyzer.analyze_code_structure(code)
        
        assert metrics["total_lines"] == 12  # Including blank lines
        assert metrics["code_lines"] == 8    # Non-empty lines
        assert metrics["blank_lines"] == 4
        assert metrics["comment_lines"] == 3
        assert metrics["indentation_ratio"] > 0
        assert metrics["comment_ratio"] > 0
        assert 0 <= metrics["code_density"] <= 1.0

    def test_analyze_empty_code(self):
        """Test analysis of empty code."""
        metrics = CodeAnalyzer.analyze_code_structure("")
        assert metrics["total_lines"] == 1  # Empty string is one line
        assert metrics["code_lines"] == 0
        assert metrics["comment_ratio"] == 0.0

    def test_count_comment_styles(self):
        """Test different comment style counting."""
        code = """
# Python comment
// C-style comment
/* Multi-line
   comment */
'''
Docstring style
'''
"""
        metrics = CodeAnalyzer.analyze_code_structure(code)
        assert metrics["comment_lines"] >= 3  # At least the single-line comments


class TestScoreAggregator:
    """Tests for ScoreAggregator utilities."""

    def test_calculate_weighted_average(self):
        """Test weighted average calculation."""
        scores = {"a": 0.8, "b": 0.6, "c": 1.0}
        
        # Equal weights
        assert ScoreAggregator.calculate_weighted_average(scores) == pytest.approx(0.8)
        
        # Custom weights
        weights = {"a": 2.0, "b": 1.0, "c": 0.5}
        weighted_avg = ScoreAggregator.calculate_weighted_average(scores, weights)
        assert weighted_avg == pytest.approx((0.8*2 + 0.6*1 + 1.0*0.5) / 3.5)
        
        # Empty scores
        assert ScoreAggregator.calculate_weighted_average({}) == 0.0

    def test_aggregate_results(self):
        """Test result aggregation."""
        results = [
            {"score1": 0.8, "score2": 0.6},
            {"score1": 0.9, "score2": 0.7},
            {"score1": 0.7, "score2": 0.8},
        ]
        
        aggregated = ScoreAggregator.aggregate_results(results)
        
        assert aggregated["score1_mean"] == pytest.approx(0.8)
        assert aggregated["score1_max"] == 0.9
        assert aggregated["score1_min"] == 0.7
        assert aggregated["score2_mean"] == pytest.approx(0.7)
        
        # With std aggregation
        aggregated_std = ScoreAggregator.aggregate_results(
            results, ["mean", "std"]
        )
        assert "score1_std" in aggregated_std
        assert aggregated_std["score1_std"] > 0


class TestAsyncBatchProcessor:
    """Tests for AsyncBatchProcessor utilities."""

    @pytest.mark.asyncio
    async def test_process_batch_with_semaphore(self):
        """Test batch processing with concurrency limit."""
        items = list(range(10))
        processed_items: List[int] = []
        
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.01)  # Simulate work
            processed_items.append(item)
            return item * 2
        
        results = await AsyncBatchProcessor.process_batch_with_semaphore(
            items, process_item, max_concurrent=3
        )
        
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]
        assert set(processed_items) == set(items)

    @pytest.mark.asyncio
    async def test_process_with_timeout(self):
        """Test processing with timeout."""
        async def slow_task():
            await asyncio.sleep(1.0)
            return "completed"
        
        # Should timeout
        result = await AsyncBatchProcessor.process_with_timeout(
            slow_task(), timeout=0.1, default_result="timeout"
        )
        assert result == "timeout"
        
        # Should complete
        async def fast_task():
            await asyncio.sleep(0.01)
            return "completed"
        
        result = await AsyncBatchProcessor.process_with_timeout(
            fast_task(), timeout=0.5
        )
        assert result == "completed"


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator utilities."""

    def test_generate_text_key(self):
        """Test text cache key generation."""
        text = "This is some text"
        key1 = CacheKeyGenerator.generate_text_key(text)
        key2 = CacheKeyGenerator.generate_text_key(text)
        key3 = CacheKeyGenerator.generate_text_key("Different text")
        
        # Same text should produce same key
        assert key1 == key2
        # Different text should produce different key
        assert key1 != key3
        
        # With prefix
        prefixed_key = CacheKeyGenerator.generate_text_key(text, prefix="test")
        assert prefixed_key.startswith("test_")
        assert prefixed_key != key1

    def test_generate_composite_key(self):
        """Test composite cache key generation."""
        items = [1, "text", 3.14, {"key": "value"}]
        key1 = CacheKeyGenerator.generate_composite_key(items)
        key2 = CacheKeyGenerator.generate_composite_key(items)
        key3 = CacheKeyGenerator.generate_composite_key([1, "different", 3.14])
        
        # Same items should produce same key
        assert key1 == key2
        # Different items should produce different key
        assert key1 != key3


class TestUtilityFunctions:
    """Tests for standalone utility functions."""

    def test_normalize_score(self):
        """Test score normalization."""
        assert normalize_score(0.5) == 0.5
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(1.5) == 1.0
        assert normalize_score(0.7, min_val=0.5, max_val=0.8) == 0.7
        assert normalize_score(0.3, min_val=0.5, max_val=0.8) == 0.5

    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        assert calculate_percentage_change(100, 150) == 50.0
        assert calculate_percentage_change(100, 50) == -50.0
        assert calculate_percentage_change(0, 100) == 100.0
        assert calculate_percentage_change(0, 0) == 0.0
        assert calculate_percentage_change(-50, -25) == 50.0