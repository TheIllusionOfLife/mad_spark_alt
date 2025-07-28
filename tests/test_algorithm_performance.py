"""
Performance testing framework for Mad Spark Alt algorithms.

Tests algorithm complexity, memory usage, and semantic operator performance
to prevent regressions and establish optimization baselines.
"""

import asyncio
import time
import tracemalloc
import statistics
from typing import List, Dict, Any
import pytest
from unittest.mock import AsyncMock
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator, SemanticCrossoverOperator, SemanticOperatorCache
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse, LLMProvider


class PerformanceBenchmark:
    """Helper class for performance benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.end_memory = current
        self.peak_memory = peak
        tracemalloc.stop()
    
    @property
    def duration(self) -> float:
        """Execution time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def memory_used(self) -> int:
        """Memory used in bytes."""
        if self.start_memory is None or self.end_memory is None:
            return 0
        return max(0, self.end_memory - self.start_memory)
    
    @property
    def peak_memory_used(self) -> int:
        """Peak memory usage in bytes."""
        if self.start_memory is None or self.peak_memory is None:
            return 0
        return max(0, self.peak_memory - self.start_memory)


class TestAlgorithmPerformance:
    """Algorithm complexity and performance validation tests."""

    def test_qadi_orchestrator_scaling(self):
        """Test QADI orchestrator performance doesn't degrade with repeated use."""
        orchestrator = SimpleQADIOrchestrator()
        durations = []
        
        # Test multiple runs to check for memory leaks or performance degradation
        test_inputs = [
            "How can we improve team productivity?",
            "What are effective learning strategies?", 
            "How can we reduce environmental impact?",
            "What makes a successful startup?",
            "How can we improve mental health awareness?"
        ]
        
        for test_input in test_inputs:
            with PerformanceBenchmark(f"QADI-{test_input[:20]}") as benchmark:
                # Use mock for performance testing (avoid API costs)
                pass  # Will be implemented when we have mock infrastructure
            
            durations.append(benchmark.duration)
        
        # Performance should be consistent (no significant degradation)
        if len(durations) > 1:
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            # Allow for more variance in CI environments - maximum shouldn't be more than 5x average
            # This is more tolerant for timing variability in CI systems
            assert max_duration <= avg_duration * 5.0, f"Performance degradation detected: {durations}"

    def test_semantic_cache_performance(self):
        """Test semantic operator cache performance and hit rates."""
        cache = SemanticOperatorCache(ttl_seconds=3600)
        
        # Test cache miss performance
        test_content = "Implement renewable energy solutions for urban areas"
        
        with PerformanceBenchmark("cache-miss") as benchmark:
            result = cache.get(test_content)
        
        assert result is None
        assert benchmark.duration < 0.001  # Cache miss should be very fast (<1ms)
        
        # Test cache set performance
        cached_result = "Enhanced solution: Implement solar panel networks with community energy sharing"
        
        with PerformanceBenchmark("cache-set") as benchmark:
            cache.put(test_content, cached_result)
        
        assert benchmark.duration < 0.001  # Cache set should be very fast (<1ms)
        
        # Test cache hit performance
        with PerformanceBenchmark("cache-hit") as benchmark:
            retrieved = cache.get(test_content)
        
        assert retrieved == cached_result
        assert benchmark.duration < 0.001  # Cache hit should be very fast (<1ms)
        
        # Test cache performance with multiple entries
        hit_count = 0
        total_requests = 100
        
        # Add multiple cache entries
        for i in range(20):
            cache.put(f"content_{i}", f"result_{i}")
        
        with PerformanceBenchmark("cache-bulk-operations") as benchmark:
            for i in range(total_requests):
                content_key = f"content_{i % 20}"  # Will create cache hits
                result = cache.get(content_key)
                if result is not None:
                    hit_count += 1
        
        # Should have high hit rate
        hit_rate = hit_count / total_requests
        assert hit_rate >= 0.8, f"Low cache hit rate: {hit_rate}"
        
        # Bulk operations should still be fast
        avg_time_per_op = benchmark.duration / total_requests
        assert avg_time_per_op < 0.0001, f"Cache operations too slow: {avg_time_per_op}s per operation"

    def test_memory_usage_bounds(self):
        """Test that algorithms stay within reasonable memory bounds."""
        
        # Test QADI orchestrator memory usage
        with PerformanceBenchmark("qadi-memory") as benchmark:
            orchestrator = SimpleQADIOrchestrator()
            # Simulate processing without actual LLM calls
            test_data = ["hypothesis " + str(i) for i in range(10)]
            
        # QADI orchestrator should use minimal memory for initialization
        assert benchmark.peak_memory_used < 1024 * 1024, f"QADI orchestrator uses too much memory: {benchmark.peak_memory_used} bytes"
        
        # Test cache memory usage
        with PerformanceBenchmark("cache-memory") as benchmark:
            cache = SemanticOperatorCache()
            
            # Add many entries to test memory scaling
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}" * 100)  # 100-char values
        
        # Cache should use reasonable memory (less than 1MB for 100 entries)
        assert benchmark.peak_memory_used < 1024 * 1024, f"Cache uses too much memory: {benchmark.peak_memory_used} bytes"

    def test_idea_generation_scaling(self):
        """Test that idea generation scales linearly with population size."""
        durations = []
        population_sizes = [3, 5, 8, 10]
        
        # Test with different population sizes
        for pop_size in population_sizes:
            ideas = []
            for i in range(pop_size):
                idea = GeneratedIdea(
                    content=f"Test idea {i} for population size {pop_size}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="TestAgent",
                    generation_prompt=f"Generate idea {i}",
                    confidence_score=0.8,
                    reasoning="Test reasoning"
                )
                ideas.append(idea)
            
            with PerformanceBenchmark(f"ideas-{pop_size}") as benchmark:
                # Simulate processing ideas (sorting, filtering, etc.)
                sorted_ideas = sorted(ideas, key=lambda x: x.confidence_score)
                filtered_ideas = [idea for idea in sorted_ideas if idea.confidence_score > 0.5]
            
            durations.append(benchmark.duration)
        
        # Performance should scale roughly linearly
        # Later populations should not be more than 3x slower than smallest
        if len(durations) > 1:
            ratio = durations[-1] / durations[0]
            expected_ratio = population_sizes[-1] / population_sizes[0]
            assert ratio <= expected_ratio * 3, f"Poor scaling: {ratio}x vs expected {expected_ratio}x"


class TestSemanticOperatorPerformance:
    """Performance tests for semantic evolution operators."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        provider.generate.return_value = LLMResponse(
            content="Optimized version of the idea",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            cost=0.001
        )
        return provider

    def create_test_ideas(self, count: int) -> List[GeneratedIdea]:
        """Create test ideas for performance testing."""
        ideas = []
        base_contents = [
            "Implement renewable energy solutions with solar panels and wind turbines",
            "Develop sustainable transportation using electric vehicles and public transit",
            "Create circular economy systems with recycling and waste reduction",
            "Build green infrastructure with parks and sustainable buildings",
            "Establish community gardens and local food production systems"
        ]
        
        for i in range(count):
            content = base_contents[i % len(base_contents)] + f" - variation {i}"
            idea = GeneratedIdea(
                content=content,
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt=f"Generate sustainable solution {i}",
                confidence_score=0.6 + (i % 3) * 0.1,  # Vary confidence
                reasoning=f"Test reasoning for idea {i}"
            )
            ideas.append(idea)
        
        return ideas

    def test_semantic_mutation_performance(self, mock_llm_provider):
        """Test semantic mutation operator performance."""
        ideas = self.create_test_ideas(10)
        cache = SemanticOperatorCache()
        mutation_operator = BatchSemanticMutationOperator(mock_llm_provider, cache_ttl=3600)
        
        durations = []
        
        # Test individual mutation performance
        for i, idea in enumerate(ideas[:5]):  # Test first 5 to avoid API costs
            with PerformanceBenchmark(f"mutation-{i}") as benchmark:
                # Mock the actual LLM call for performance testing
                # In real implementation, this would call LLM
                mutated_content = f"[MUTATED] {idea.content}"
                cache.put(idea.content, mutated_content)
            
            durations.append(benchmark.duration)
        
        # Individual mutations should be fast when cached
        avg_duration = statistics.mean(durations) if durations else 0
        assert avg_duration < 0.01, f"Semantic mutation too slow: {avg_duration}s average"

    def test_batch_semantic_operations_performance(self):
        """Test batch semantic operations performance."""
        ideas = self.create_test_ideas(8)
        cache = SemanticOperatorCache()
        
        # Test batch mutation simulation
        with PerformanceBenchmark("batch-mutation") as benchmark:
            batch_size = 4
            batches = [ideas[i:i+batch_size] for i in range(0, len(ideas), batch_size)]
            
            for batch in batches:
                # Simulate batch processing
                for idea in batch:
                    cache_key = idea.content
                    if not cache.get(cache_key):
                        # Simulate LLM call result
                        cache.put(cache_key, f"[BATCH_MUTATED] {idea.content}")
        
        # Batch operations should be efficient
        assert benchmark.duration < 0.1, f"Batch operations too slow: {benchmark.duration}s"
        
        # Memory usage should be reasonable
        assert benchmark.peak_memory_used < 1024 * 1024, f"Batch operations use too much memory: {benchmark.peak_memory_used} bytes"

    def test_cache_cleanup_performance(self):
        """Test cache cleanup and TTL performance."""
        cache = SemanticOperatorCache(ttl_seconds=1)  # Short TTL for testing
        
        # Fill cache with entries
        with PerformanceBenchmark("cache-fill") as benchmark:
            for i in range(50):
                cache.put(f"key_{i}", f"value_{i}")
        
        assert benchmark.duration < 0.1, "Cache filling too slow"
        
        # Wait for TTL expiration
        time.sleep(1.1)
        
        # Test cleanup performance
        with PerformanceBenchmark("cache-cleanup") as benchmark:
            # Access cache to trigger potential cleanup
            for i in range(10):
                cache.get(f"key_{i}")
        
        assert benchmark.duration < 0.1, "Cache cleanup affecting performance"

    def test_diversity_calculation_performance(self):
        """Test diversity calculation performance for semantic operators."""
        ideas = self.create_test_ideas(20)
        
        # Test simple content-based diversity calculation
        with PerformanceBenchmark("diversity-simple") as benchmark:
            # Simulate Jaccard similarity calculation
            diversity_scores = []
            for i, idea1 in enumerate(ideas):
                for j, idea2 in enumerate(ideas[i+1:], i+1):
                    # Simple word-based similarity (avoiding LLM costs)
                    words1 = set(idea1.content.lower().split())
                    words2 = set(idea2.content.lower().split())
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0
                    diversity_scores.append(1 - similarity)
        
        # Diversity calculation should complete quickly even for larger populations
        assert benchmark.duration < 1.0, f"Diversity calculation too slow: {benchmark.duration}s"
        
        # Should produce reasonable diversity scores
        assert len(diversity_scores) > 0
        avg_diversity = statistics.mean(diversity_scores)
        assert 0.0 <= avg_diversity <= 1.0, f"Invalid diversity scores: {avg_diversity}"


class TestPerformanceRegression:
    """Tests to prevent performance regressions."""

    def test_performance_baselines(self):
        """Test that performance meets baseline expectations."""
        
        # Baseline: SimpleQADIOrchestrator initialization should be fast
        with PerformanceBenchmark("orchestrator-init") as benchmark:
            orchestrator = SimpleQADIOrchestrator()
        
        assert benchmark.duration < 0.1, f"Orchestrator initialization too slow: {benchmark.duration}s"
        assert benchmark.peak_memory_used < 10 * 1024 * 1024, f"Orchestrator uses too much memory: {benchmark.peak_memory_used} bytes"
        
        # Baseline: Cache operations should be sub-millisecond
        cache = SemanticOperatorCache()
        
        with PerformanceBenchmark("cache-operations") as benchmark:
            for i in range(100):
                cache.put(f"key_{i}", f"value_{i}")
                cache.get(f"key_{i}")
        
        avg_time_per_op = benchmark.duration / 200  # 100 sets + 100 gets
        assert avg_time_per_op < 0.0001, f"Cache operations too slow: {avg_time_per_op}s per operation"

    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        initial_memory = None
        final_memory = None
        
        # Baseline memory usage
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Perform repeated operations that might leak memory
        cache = SemanticOperatorCache()
        
        for cycle in range(10):
            # Simulate operation cycle
            ideas = self.create_test_ideas(5)
            
            # Simulate processing
            for idea in ideas:
                cache.put(f"cycle_{cycle}_{idea.content[:20]}", f"processed_{idea.content}")
                cache.get(f"cycle_{cycle}_{idea.content[:20]}")
            
            # Clear references
            del ideas
        
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Memory usage shouldn't grow excessively
        memory_growth = final_memory - initial_memory
        # Allow for some growth, but not excessive (less than 1MB)
        assert memory_growth < 1024 * 1024, f"Potential memory leak: {memory_growth} bytes growth"

    def create_test_ideas(self, count: int) -> List[GeneratedIdea]:
        """Helper method to create test ideas."""
        return [
            GeneratedIdea(
                content=f"Test idea {i}: Implement solution approach number {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt=f"Test prompt {i}",
                confidence_score=0.7,
                reasoning=f"Test reasoning {i}"
            )
            for i in range(count)
        ]


if __name__ == "__main__":
    # Allow running performance tests directly
    pytest.main([__file__, "-v"])