"""
Tests for semantic (LLM-powered) genetic operators.

This module tests batch semantic mutation and crossover operators that use
LLMs to create meaningful variations and combinations of ideas.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest, LLMResponse, LLMProvider
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
    SemanticOperatorCache
)


class TestSemanticOperatorCache:
    """Test the caching mechanism for semantic operators."""

    def test_cache_initialization(self):
        """Test cache initialization with TTL."""
        cache = SemanticOperatorCache(ttl_seconds=3600)
        assert cache.ttl_seconds == 3600
        assert len(cache._cache) == 0

    def test_cache_hit(self):
        """Test cache hit for identical content."""
        cache = SemanticOperatorCache(ttl_seconds=3600)
        
        # Add to cache
        cache.put("test content", "mutated content")
        
        # Should hit cache
        result = cache.get("test content", return_dict=False)
        assert result == "mutated content"

    def test_cache_miss(self):
        """Test cache miss for new content."""
        cache = SemanticOperatorCache(ttl_seconds=3600)
        
        result = cache.get("new content")
        assert result is None

    def test_cache_expiration(self, monkeypatch):
        """Test cache expiration after TTL."""
        import time
        cache = SemanticOperatorCache(ttl_seconds=1)  # 1 second TTL
        
        # Mock time.time to control expiration
        current_time = time.time()
        monkeypatch.setattr(time, 'time', lambda: current_time)
        
        # Add to cache
        cache.put("test content", "mutated content")
        
        # Should hit cache immediately
        assert cache.get("test content", return_dict=False) == "mutated content"
        
        # Advance time past TTL
        monkeypatch.setattr(time, 'time', lambda: current_time + 2.0)
        
        # Should miss cache after expiration
        assert cache.get("test content") is None

    def test_cache_key_generation(self):
        """Test consistent cache key generation."""
        cache = SemanticOperatorCache()
        
        # Same content should generate same key
        key1 = cache._get_cache_key("test content")
        key2 = cache._get_cache_key("test content")
        assert key1 == key2
        
        # Different content should generate different keys
        key3 = cache._get_cache_key("different content")
        assert key1 != key3


class TestBatchSemanticMutationOperator:
    """Test batch semantic mutation operator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def sample_ideas(self):
        """Create sample ideas for testing."""
        return [
            GeneratedIdea(
                content="Reduce plastic waste by implementing city-wide recycling",
                thinking_method="environmental",
                agent_name="test_agent",
                generation_prompt="How to reduce waste?"
            ),
            GeneratedIdea(
                content="Create urban gardens to grow local food",
                thinking_method="sustainability",
                agent_name="test_agent",
                generation_prompt="How to improve food systems?"
            ),
            GeneratedIdea(
                content="Use AI to optimize energy consumption",
                thinking_method="technical",
                agent_name="test_agent",
                generation_prompt="How to save energy?"
            )
        ]

    @pytest.mark.asyncio
    async def test_single_mutation(self, mock_llm_provider, sample_ideas):
        """Test mutating a single idea."""
        # Set up mock response
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Reduce plastic waste through community-led zero-waste initiatives",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            cost=0.001
        )
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Mutate single idea
        result = await operator.mutate_single(sample_ideas[0], "environmental context")
        
        assert result.content == "Reduce plastic waste through community-led zero-waste initiatives"
        assert result.agent_name == "BatchSemanticMutationOperator"
        assert "llm_cost" in result.metadata
        assert result.metadata["mutation_type"] in [
            "perspective_shift", "mechanism_change", "constraint_variation", "abstraction_shift"
        ]

    @pytest.mark.asyncio
    async def test_batch_mutation(self, mock_llm_provider, sample_ideas):
        """Test batch mutation of multiple ideas."""
        # Set up mock response with batch format
        batch_response = """IDEA_1_MUTATION: Implement neighborhood plastic-free zones with community enforcement
IDEA_2_MUTATION: Transform rooftops into productive food forests with native plants
IDEA_3_MUTATION: Deploy smart grid technology for real-time energy optimization"""
        
        mock_llm_provider.generate.return_value = LLMResponse(
            content=batch_response,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 200, "completion_tokens": 100},
            cost=0.002
        )
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Mutate batch
        results = await operator.mutate_batch(sample_ideas, "sustainability context")
        
        assert len(results) == 3
        assert results[0].content == "Implement neighborhood plastic-free zones with community enforcement"
        assert results[1].content == "Transform rooftops into productive food forests with native plants"
        assert results[2].content == "Deploy smart grid technology for real-time energy optimization"

    @pytest.mark.asyncio
    async def test_cache_functionality(self, mock_llm_provider, sample_ideas):
        """Test that caching reduces API calls."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Cached mutation result",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            cost=0.001
        )
        
        operator = BatchSemanticMutationOperator(mock_llm_provider, cache_ttl=3600)
        
        # First call should hit LLM
        result1 = await operator.mutate_single(sample_ideas[0], "context")
        assert mock_llm_provider.generate.call_count == 1
        
        # Second call with same idea should hit cache
        result2 = await operator.mutate_single(sample_ideas[0], "context")
        assert mock_llm_provider.generate.call_count == 1  # No additional call
        assert result2.content == result1.content

    @pytest.mark.asyncio
    async def test_batch_with_cache(self, mock_llm_provider, sample_ideas):
        """Test batch mutation with some cached ideas."""
        # Pre-cache one idea by calling mutate_single
        operator = BatchSemanticMutationOperator(mock_llm_provider, cache_ttl=3600)
        
        # Mock for single mutation to cache
        single_response = LLMResponse(
            content='{"mutated_content": "Cached mutation for idea 1"}',
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 50, "completion_tokens": 25},
            cost=0.0005
        )
        mock_llm_provider.generate.return_value = single_response
        
        # Cache the first idea
        cached_result = await operator.mutate_single(sample_ideas[0], "context")
        assert cached_result.content == "Cached mutation for idea 1"
        
        # Mock response for uncached ideas only
        batch_response = """IDEA_1_MUTATION: Transform rooftops into productive food forests
IDEA_2_MUTATION: Deploy smart grid technology"""
        
        # Reset mock for batch call
        mock_llm_provider.generate.reset_mock()
        mock_llm_provider.generate.return_value = LLMResponse(
            content=batch_response,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 150, "completion_tokens": 75},
            cost=0.0015
        )
        
        # Mutate batch
        results = await operator.mutate_batch(sample_ideas, "context")
        
        # First idea should come from cache
        assert results[0].content == "Cached mutation for idea 1"
        # Other ideas should come from LLM
        assert results[1].content == "Transform rooftops into productive food forests"
        assert results[2].content == "Deploy smart grid technology"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_llm_provider, sample_ideas):
        """Test error handling in batch mutations."""
        # Mock LLM error
        mock_llm_provider.generate.side_effect = Exception("LLM API error")
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Should handle error gracefully
        with pytest.raises(Exception) as exc_info:
            await operator.mutate_batch(sample_ideas, "context")
        
        assert "LLM API error" in str(exc_info.value)


class TestSemanticCrossoverOperator:
    """Test semantic crossover operator."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock(spec=GoogleProvider)
        provider.generate = AsyncMock()
        return provider

    @pytest.fixture
    def parent_ideas(self):
        """Create parent ideas for crossover."""
        return (
            GeneratedIdea(
                content="Create urban rooftop gardens to grow local food",
                thinking_method="environmental",
                agent_name="agent1",
                generation_prompt="How to improve food systems?"
            ),
            GeneratedIdea(
                content="Use IoT sensors to optimize water usage in agriculture",
                thinking_method="technical",
                agent_name="agent2",
                generation_prompt="How to save water?"
            )
        )

    @pytest.mark.asyncio
    async def test_semantic_crossover(self, mock_llm_provider, parent_ideas):
        """Test semantic crossover of two parent ideas."""
        # Mock LLM response with proper format
        mock_llm_provider.generate.return_value = LLMResponse(
            content="OFFSPRING_1: Develop smart rooftop gardens with IoT-controlled irrigation systems for efficient urban food production\nOFFSPRING_2: Create sensor-monitored community gardens that optimize water usage for local food",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 150, "completion_tokens": 75},
            cost=0.0015
        )
        
        operator = SemanticCrossoverOperator(mock_llm_provider)
        
        # Perform crossover
        offspring1, offspring2 = await operator.crossover(
            parent_ideas[0], 
            parent_ideas[1],
            "urban sustainability context"
        )
        
        # Check offspring 1
        assert "smart rooftop gardens" in offspring1.content
        assert offspring1.agent_name == "SemanticCrossoverOperator"
        assert offspring1.metadata["operator"] == "semantic_crossover"
        assert len(offspring1.parent_ideas) == 2
        
        # Check offspring 2 (should be different)
        assert offspring2.content == "Create sensor-monitored community gardens that optimize water usage for local food"
        assert offspring2.agent_name == "SemanticCrossoverOperator"

    @pytest.mark.asyncio 
    async def test_crossover_batch_prompt(self, mock_llm_provider, parent_ideas):
        """Test batch crossover prompt generation."""
        # Mock response with two offspring
        batch_response = """OFFSPRING_1: Smart urban farms with automated hydroponic systems and IoT monitoring
OFFSPRING_2: Community gardens integrated with water-saving sensor networks"""
        
        mock_llm_provider.generate.return_value = LLMResponse(
            content=batch_response,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 200, "completion_tokens": 100},
            cost=0.002
        )
        
        operator = SemanticCrossoverOperator(mock_llm_provider)
        
        offspring1, offspring2 = await operator.crossover(
            parent_ideas[0],
            parent_ideas[1],
            "sustainability"
        )
        
        assert offspring1.content == "Smart urban farms with automated hydroponic systems and IoT monitoring"
        assert offspring2.content == "Community gardens integrated with water-saving sensor networks"

    @pytest.mark.asyncio
    async def test_crossover_caching(self, mock_llm_provider, parent_ideas):
        """Test crossover result caching."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="OFFSPRING_1: Cached result 1\nOFFSPRING_2: Cached result 2",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            cost=0.001
        )
        
        operator = SemanticCrossoverOperator(mock_llm_provider, cache_ttl=3600)
        
        # First crossover
        offspring1a, offspring2a = await operator.crossover(
            parent_ideas[0], parent_ideas[1], "context"
        )
        assert mock_llm_provider.generate.call_count == 1
        
        # Same parents should hit cache
        offspring1b, offspring2b = await operator.crossover(
            parent_ideas[0], parent_ideas[1], "context"
        )
        assert mock_llm_provider.generate.call_count == 1  # No additional call
        assert offspring1b.content == offspring1a.content