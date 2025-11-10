"""
Baseline integration tests for semantic_operators.py BEFORE splitting.

These tests verify the current behavior of the monolithic semantic_operators.py
module to ensure that the split into multiple files maintains identical functionality.

Test Strategy:
1. Test all public APIs (mutation, crossover, caching, utilities)
2. Test with various configurations
3. Capture baseline behavior before refactoring
4. Use these same tests after split to verify no regressions
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse
from mad_spark_alt.evolution.interfaces import EvaluationContext
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
    BatchSemanticCrossoverOperator,
    SemanticOperatorCache,
    get_mutation_schema,
    get_crossover_schema,
    format_evaluation_context,
    is_likely_truncated,
    _prepare_operator_contexts,
    _prepare_cache_key_with_context,
)


class TestUtilityFunctions:
    """Test all utility functions that will move to semantic_utils.py"""

    def test_format_evaluation_context(self):
        """Test evaluation context formatting"""
        context = EvaluationContext(
            original_question="How to reduce plastic waste?",
            current_best_scores={"impact": 0.8, "feasibility": 0.7},
            target_improvements=["impact", "feasibility"]
        )

        result = format_evaluation_context(context)

        assert "How to reduce plastic waste?" in result
        assert "Impact: 0.8" in result
        assert "Feasibility: 0.7" in result
        assert "impact, feasibility" in result
        assert "FOCUS" in result

    def test_is_likely_truncated_detects_truncation(self):
        """Test truncation detection with various patterns"""
        # Test ellipsis
        assert is_likely_truncated("This text ends with...")

        # Test incomplete JSON
        assert is_likely_truncated('{"key": "value"')
        assert is_likely_truncated('["item1", "item2"')

        # Test incomplete sentences
        assert is_likely_truncated("This ends with a")
        assert is_likely_truncated("This ends with the")

        # Test proper endings (should NOT be truncated)
        assert not is_likely_truncated("This is a complete sentence.")
        assert not is_likely_truncated("Question?")
        assert not is_likely_truncated("Excitement!")
        assert not is_likely_truncated("")

    def test_prepare_operator_contexts_with_string(self):
        """Test context preparation with string context"""
        context_str, eval_context_str = _prepare_operator_contexts(
            "test context",
            "test prompt",
            "default"
        )

        assert context_str == "test context"
        assert eval_context_str == "No specific evaluation context provided."

    def test_prepare_operator_contexts_with_evaluation_context(self):
        """Test context preparation with EvaluationContext object"""
        eval_context = EvaluationContext(
            original_question="Test question",
            current_best_scores={"impact": 0.8},
            target_improvements=["impact"]
        )

        context_str, eval_context_str = _prepare_operator_contexts(
            eval_context,
            "fallback prompt",
            "default"
        )

        assert context_str == "Test question"
        assert "Test question" in eval_context_str
        assert "Impact: 0.8" in eval_context_str

    def test_prepare_cache_key_with_string_context(self):
        """Test cache key preparation with string context"""
        key = _prepare_cache_key_with_context("base_key", "string context")

        # String contexts should be included in cache key
        assert key.startswith("base_key||str:")
        assert "base_key" in key

        # Different string contexts should produce different keys
        key2 = _prepare_cache_key_with_context("base_key", "different context")
        assert key != key2

    def test_prepare_cache_key_with_evaluation_context(self):
        """Test cache key preparation with EvaluationContext"""
        eval_context = EvaluationContext(
            original_question="Test",
            current_best_scores={"impact": 0.8},
            target_improvements=["impact"]
        )

        key = _prepare_cache_key_with_context("base_key", eval_context)

        assert key.startswith("base_key||ctx:")
        assert len(key) > len("base_key")

    def test_get_mutation_schema(self):
        """Test mutation schema structure"""
        schema = get_mutation_schema()

        assert schema["type"] == "OBJECT"
        assert "mutations" in schema["properties"]
        assert schema["properties"]["mutations"]["type"] == "ARRAY"
        assert "id" in schema["properties"]["mutations"]["items"]["properties"]
        assert "content" in schema["properties"]["mutations"]["items"]["properties"]

    def test_get_crossover_schema(self):
        """Test crossover schema structure"""
        schema = get_crossover_schema()

        assert schema["type"] == "OBJECT"
        assert "offspring_1" in schema["properties"]
        assert "offspring_2" in schema["properties"]
        assert schema["properties"]["offspring_1"]["type"] == "STRING"
        assert schema["properties"]["offspring_2"]["type"] == "STRING"


class TestSemanticOperatorCache:
    """Test caching functionality that will move to semantic_cache.py"""

    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        cache = SemanticOperatorCache(ttl_seconds=3600)

        assert cache.ttl_seconds == 3600
        assert len(cache._cache) == 0
        assert len(cache._similarity_index) == 0

    def test_cache_put_and_get_string(self):
        """Test caching string values"""
        cache = SemanticOperatorCache()

        cache.put("test_content", "test_result", "mutation")
        result = cache.get("test_content", "mutation", return_dict=False)

        assert result == "test_result"

    def test_cache_put_and_get_dict(self):
        """Test caching dictionary values"""
        cache = SemanticOperatorCache()

        cache.put("test_content", {"content": "result", "type": "mutation"}, "mutation")
        result = cache.get("test_content", "mutation", return_dict=True)

        assert isinstance(result, dict)
        assert result["content"] == "result"
        assert result["type"] == "mutation"

    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = SemanticOperatorCache()

        result = cache.get("nonexistent", "mutation")

        assert result is None

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = SemanticOperatorCache()

        cache.put("key1", "value1", "mutation")
        cache.put("key2", "value2", "crossover")

        stats = cache.get_cache_stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 0
        assert "session_duration_minutes" in stats


class TestBatchSemanticMutationOperator:
    """Test mutation operator that will move to semantic_mutation.py"""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider"""
        provider = Mock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def mutation_operator(self, mock_llm_provider):
        """Create mutation operator with mock provider"""
        return BatchSemanticMutationOperator(mock_llm_provider)

    @pytest.fixture
    def sample_idea(self):
        """Create sample idea for testing"""
        return GeneratedIdea(
            content="Use AI to reduce plastic waste",
            thinking_method="ABDUCTION",
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7,
            reasoning="Test reasoning",
            parent_ideas=[],
            metadata={"generation": 0},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def test_operator_initialization(self, mutation_operator):
        """Test operator initializes correctly"""
        assert mutation_operator.llm_provider is not None
        assert mutation_operator.cache is not None
        assert len(mutation_operator.mutation_types) == 4
        assert len(mutation_operator.breakthrough_mutation_types) == 4
        assert mutation_operator.breakthrough_threshold == 0.8

    def test_is_high_scoring_idea_with_fitness(self, mutation_operator, sample_idea):
        """Test breakthrough detection with fitness score"""
        sample_idea.metadata["overall_fitness"] = 0.85

        assert mutation_operator._is_high_scoring_idea(sample_idea)

    def test_is_high_scoring_idea_with_confidence(self, mutation_operator, sample_idea):
        """Test breakthrough detection with confidence score"""
        sample_idea.confidence_score = 0.87
        sample_idea.metadata["generation"] = 2

        assert mutation_operator._is_high_scoring_idea(sample_idea)

    def test_is_not_high_scoring_idea(self, mutation_operator, sample_idea):
        """Test breakthrough detection returns False for low scores"""
        sample_idea.metadata["overall_fitness"] = 0.5
        sample_idea.confidence_score = 0.6

        assert not mutation_operator._is_high_scoring_idea(sample_idea)

    @pytest.mark.asyncio
    async def test_mutate_single(self, mutation_operator, mock_llm_provider, sample_idea):
        """Test single mutation operation"""
        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"mutated_content": "Use AI and blockchain to revolutionize plastic waste management"}',
            provider="google",
            model="gemini-pro",
            cost=0.001
        )

        result = await mutation_operator.mutate_single(sample_idea, "test context")

        assert result.content == "Use AI and blockchain to revolutionize plastic waste management"
        assert result.thinking_method == sample_idea.thinking_method
        assert result.metadata["operator"] in ["semantic_mutation", "breakthrough_semantic_mutation"]
        assert "mutation_type" in result.metadata

    @pytest.mark.asyncio
    async def test_mutate_batch(self, mutation_operator, mock_llm_provider, sample_idea):
        """Test batch mutation operation"""
        ideas = [sample_idea, sample_idea]

        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"mutations": [{"id": 1, "content": "Mutation 1"}, {"id": 2, "content": "Mutation 2"}]}',
            provider="google",
            model="gemini-pro",
            cost=0.002
        )

        results = await mutation_operator.mutate_batch(ideas, "test context")

        assert len(results) == 2
        assert results[0].content == "Mutation 1"
        assert results[1].content == "Mutation 2"


class TestSemanticCrossoverOperator:
    """Test crossover operator that will move to semantic_crossover.py"""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider"""
        provider = Mock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def crossover_operator(self, mock_llm_provider):
        """Create crossover operator with mock provider"""
        return SemanticCrossoverOperator(mock_llm_provider)

    @pytest.fixture
    def sample_parents(self):
        """Create sample parent ideas"""
        parent1 = GeneratedIdea(
            content="Use AI for waste sorting",
            thinking_method="ABDUCTION",
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7,
            reasoning="Test reasoning",
            parent_ideas=[],
            metadata={"generation": 0},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        parent2 = GeneratedIdea(
            content="Community-driven recycling programs",
            thinking_method="ABDUCTION",
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.6,
            reasoning="Test reasoning",
            parent_ideas=[],
            metadata={"generation": 0},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        return parent1, parent2

    def test_operator_initialization(self, crossover_operator):
        """Test operator initializes correctly"""
        assert crossover_operator.llm_provider is not None
        assert crossover_operator.cache is not None
        # SIMILARITY_THRESHOLD moved to CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD
        from mad_spark_alt.core.system_constants import CONSTANTS
        assert CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD == 0.7

    def test_calculate_similarity(self, crossover_operator):
        """Test similarity calculation"""
        similarity = crossover_operator._calculate_similarity(
            "This is a test",
            "This is a test"
        )
        assert similarity == 1.0

        similarity = crossover_operator._calculate_similarity(
            "Completely different text",
            "Another unrelated sentence"
        )
        assert similarity < 0.5

    @pytest.mark.asyncio
    async def test_crossover(self, crossover_operator, mock_llm_provider, sample_parents):
        """Test crossover operation"""
        parent1, parent2 = sample_parents

        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"offspring_1": "AI-powered community recycling with smart sorting", "offspring_2": "Community-driven waste management using AI analytics"}',
            provider="google",
            model="gemini-pro",
            cost=0.002
        )

        offspring1, offspring2 = await crossover_operator.crossover(parent1, parent2)

        assert offspring1.content == "AI-powered community recycling with smart sorting"
        assert offspring2.content == "Community-driven waste management using AI analytics"
        assert offspring1.metadata["operator"] == "semantic_crossover"
        assert offspring2.metadata["operator"] == "semantic_crossover"
        assert parent1.content in offspring1.parent_ideas
        assert parent2.content in offspring1.parent_ideas


class TestBatchSemanticCrossoverOperator:
    """Test batch crossover operator functionality"""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider"""
        provider = Mock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def batch_crossover_operator(self, mock_llm_provider):
        """Create batch crossover operator with mock provider"""
        return BatchSemanticCrossoverOperator(mock_llm_provider)

    @pytest.fixture
    def sample_parent_pairs(self):
        """Create sample parent pairs"""
        pair1 = (
            GeneratedIdea(
                content="Use AI for waste sorting",
                thinking_method="ABDUCTION",
                agent_name="TestAgent",
                generation_prompt="Test",
                confidence_score=0.7,
                reasoning="Test",
                parent_ideas=[],
                metadata={"generation": 0},
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            GeneratedIdea(
                content="Community recycling programs",
                thinking_method="ABDUCTION",
                agent_name="TestAgent",
                generation_prompt="Test",
                confidence_score=0.6,
                reasoning="Test",
                parent_ideas=[],
                metadata={"generation": 0},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )

        return [pair1]

    def test_operator_initialization(self, batch_crossover_operator):
        """Test operator initializes correctly"""
        assert batch_crossover_operator.llm_provider is not None
        assert batch_crossover_operator.cache is not None
        assert batch_crossover_operator._sequential_operator is not None

    @pytest.mark.asyncio
    async def test_crossover_batch(self, batch_crossover_operator, mock_llm_provider, sample_parent_pairs):
        """Test batch crossover operation"""
        # Mock LLM response
        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"crossovers": [{"pair_id": 1, "offspring1": "AI-powered community recycling", "offspring2": "Community-driven smart sorting"}]}',
            provider="google",
            model="gemini-pro",
            cost=0.002
        )

        results = await batch_crossover_operator.crossover_batch(sample_parent_pairs)

        assert len(results) == 1
        offspring1, offspring2 = results[0]
        assert offspring1.content == "AI-powered community recycling"
        assert offspring2.content == "Community-driven smart sorting"
        assert offspring1.metadata["operator"] == "semantic_batch_crossover"


class TestIntegrationScenarios:
    """Integration tests with various configurations"""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider"""
        provider = Mock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_mutation_with_evaluation_context(self, mock_llm_provider):
        """Test mutation with EvaluationContext"""
        operator = BatchSemanticMutationOperator(mock_llm_provider)

        eval_context = EvaluationContext(
            original_question="How to reduce plastic waste?",
            current_best_scores={"impact": 0.8, "feasibility": 0.7},
            target_improvements=["impact"]
        )

        idea = GeneratedIdea(
            content="Use AI to reduce plastic waste",
            thinking_method="ABDUCTION",
            agent_name="TestAgent",
            generation_prompt="Test",
            confidence_score=0.7,
            reasoning="Test",
            parent_ideas=[],
            metadata={"generation": 0},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"mutated_content": "Revolutionary AI-powered waste management system"}',
            provider="google",
            model="gemini-pro",
            cost=0.001
        )

        result = await operator.mutate_single(idea, eval_context)

        assert result.content == "Revolutionary AI-powered waste management system"

        # Verify that evaluation context was included in the prompt
        call_args = mock_llm_provider.generate.call_args
        assert call_args is not None
        request = call_args[0][0]
        assert "How to reduce plastic waste?" in request.user_prompt
        assert "Impact: 0.8" in request.user_prompt

    @pytest.mark.asyncio
    async def test_caching_prevents_redundant_calls(self, mock_llm_provider):
        """Test that caching prevents redundant LLM calls"""
        operator = BatchSemanticMutationOperator(mock_llm_provider, cache_ttl=3600)

        idea = GeneratedIdea(
            content="Test idea for caching",
            thinking_method="ABDUCTION",
            agent_name="TestAgent",
            generation_prompt="Test",
            confidence_score=0.7,
            reasoning="Test",
            parent_ideas=[],
            metadata={"generation": 0},
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"mutated_content": "Cached mutation result"}',
            provider="google",
            model="gemini-pro",
            cost=0.001
        )

        # First call - should call LLM
        result1 = await operator.mutate_single(idea, "context")
        assert mock_llm_provider.generate.call_count == 1

        # Second call with same idea and context - should use cache
        result2 = await operator.mutate_single(idea, "context")
        assert mock_llm_provider.generate.call_count == 1  # No additional call

        # Results should be identical
        assert result1.content == result2.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
