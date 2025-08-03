"""
Tests for BatchSemanticCrossoverOperator.

This module tests the batch crossover functionality that processes
multiple parent pairs in a single LLM call for performance optimization.
"""

import asyncio
import json
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import EvaluationContext
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticCrossoverOperator,
    SemanticCrossoverOperator,
)


@pytest.fixture
def sample_parents() -> List[Tuple[GeneratedIdea, GeneratedIdea]]:
    """Create sample parent pairs for testing."""
    pairs = []
    for i in range(3):
        parent1 = GeneratedIdea(
            content=f"Parent 1 idea {i}: Innovative solution using technology",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            metadata={"fitness": 0.7 + i * 0.05}
        )
        parent2 = GeneratedIdea(
            content=f"Parent 2 idea {i}: Community-based approach",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            metadata={"fitness": 0.6 + i * 0.05}
        )
        pairs.append((parent1, parent2))
    return pairs


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def evaluation_context():
    """Create sample evaluation context."""
    return EvaluationContext(
        original_question="How can we improve urban transportation?",
        target_improvements=["impact", "feasibility"],
        current_best_scores={"impact": 7.5, "feasibility": 6.0, "sustainability": 5.0}
    )


class TestBatchSemanticCrossoverOperator:
    """Test batch semantic crossover functionality."""

    def test_operator_initialization(self, mock_llm_provider):
        """Test that batch crossover operator initializes correctly."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        assert operator.llm_provider == mock_llm_provider
        assert hasattr(operator, 'crossover_batch')
        assert hasattr(operator, 'cache')
        assert operator.structured_output_enabled is True

    @pytest.mark.asyncio
    async def test_batch_crossover_single_pair(self, mock_llm_provider, sample_parents):
        """Test batch crossover with a single parent pair."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # Mock structured output response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Hybrid tech-community solution combining digital platforms with local engagement",
                    "offspring2": "Innovation hub model merging technology infrastructure with community spaces"
                }
            ]
        })
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        
        # Test with single pair
        single_pair = [sample_parents[0]]
        results = await operator.crossover_batch(single_pair)
        
        assert len(results) == 1
        assert len(results[0]) == 2  # Two offspring per pair
        assert all(isinstance(offspring, GeneratedIdea) for offspring in results[0])
        # Parse the JSON from the mock response to get expected content
        expected_data = json.loads(mock_response.content)
        assert results[0][0].content == expected_data["crossovers"][0]["offspring1"]
        assert results[0][1].content == expected_data["crossovers"][0]["offspring2"]

    @pytest.mark.asyncio
    async def test_batch_crossover_multiple_pairs(self, mock_llm_provider, sample_parents):
        """Test batch crossover with multiple parent pairs."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # Mock structured output for 3 pairs
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Offspring 1-1: Tech-community hybrid",
                    "offspring2": "Offspring 1-2: Innovation hub"
                },
                {
                    "pair_id": 1,
                    "offspring1": "Offspring 2-1: Digital platform solution",
                    "offspring2": "Offspring 2-2: Community engagement system"
                },
                {
                    "pair_id": 2,
                    "offspring1": "Offspring 3-1: Infrastructure integration",
                    "offspring2": "Offspring 3-2: Sustainable transport network"
                }
            ]
        })
        mock_response.cost = 0.002
        mock_llm_provider.generate.return_value = mock_response
        
        results = await operator.crossover_batch(sample_parents)
        
        # Verify structure
        assert len(results) == 3
        assert all(len(pair_result) == 2 for pair_result in results)
        
        # Verify content matches
        expected_data = json.loads(mock_response.content)
        for i, (offspring1, offspring2) in enumerate(results):
            assert offspring1.content == expected_data["crossovers"][i]["offspring1"]
            assert offspring2.content == expected_data["crossovers"][i]["offspring2"]
            
        # Verify metadata preserved
        assert all(offspring.thinking_method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION] 
                  for pair in results for offspring in pair)

    @pytest.mark.asyncio
    async def test_batch_crossover_with_context(self, mock_llm_provider, sample_parents, evaluation_context):
        """Test batch crossover with evaluation context."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Context-aware solution targeting impact and feasibility",
                    "offspring2": "Sustainability-focused transport innovation"
                }
            ]
        })
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        
        results = await operator.crossover_batch([sample_parents[0]], evaluation_context)
        
        # Verify context was passed to LLM
        call_args = mock_llm_provider.generate.call_args
        assert "impact" in str(call_args)
        assert "feasibility" in str(call_args)
        assert "urban transportation" in str(call_args)

    @pytest.mark.asyncio
    async def test_batch_crossover_caching(self, mock_llm_provider, sample_parents):
        """Test that batch crossover results are cached properly."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # Set up mock to return proper batch response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Cached offspring 1",
                    "offspring2": "Cached offspring 2"
                }
            ]
        })
        mock_response.cost = 0.001
        
        # Mock both the batch call and any potential fallback
        mock_llm_provider.generate.return_value = mock_response
        
        # First call
        results1 = await operator.crossover_batch([sample_parents[0]])
        assert mock_llm_provider.generate.call_count == 1
        
        # Second call with same parents should use cache
        results2 = await operator.crossover_batch([sample_parents[0]])
        assert mock_llm_provider.generate.call_count == 1  # No additional call
        
        # Results should be identical
        assert results1[0][0].content == results2[0][0].content
        assert results1[0][1].content == results2[0][1].content

    @pytest.mark.asyncio
    async def test_batch_crossover_mixed_cache_hits(self, mock_llm_provider, sample_parents):
        """Test batch crossover with some cached and some uncached pairs."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # First call with pair 0
        mock_response1 = MagicMock()
        mock_response1.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Cached offspring 1",
                    "offspring2": "Cached offspring 2"
                }
            ]
        })
        mock_response1.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response1
        await operator.crossover_batch([sample_parents[0]])
        
        # Second call with all pairs (pair 0 cached, pairs 1&2 not)
        mock_response2 = MagicMock()
        mock_response2.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "New offspring 1-1",
                    "offspring2": "New offspring 1-2"
                },
                {
                    "pair_id": 1,
                    "offspring1": "New offspring 2-1",
                    "offspring2": "New offspring 2-2"
                }
            ]
        })
        mock_response2.cost = 0.002
        mock_llm_provider.generate.return_value = mock_response2
        
        results = await operator.crossover_batch(sample_parents)
        
        # Should have made 2 LLM calls total
        assert mock_llm_provider.generate.call_count == 2
        
        # First pair should use cached results
        assert results[0][0].content == "Cached offspring 1"
        assert results[0][1].content == "Cached offspring 2"
        
        # Other pairs should have new results
        assert results[1][0].content == "New offspring 1-1"
        assert results[2][0].content == "New offspring 2-1"

    @pytest.mark.asyncio
    async def test_batch_crossover_empty_input(self, mock_llm_provider):
        """Test batch crossover with empty input."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        results = await operator.crossover_batch([])
        
        assert results == []
        assert mock_llm_provider.generate.call_count == 0

    @pytest.mark.asyncio
    async def test_batch_crossover_error_handling(self, mock_llm_provider, sample_parents):
        """Test batch crossover error handling and fallback."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # Mock LLM error
        mock_llm_provider.generate.side_effect = Exception("LLM API error")
        
        # Should fall back to sequential processing
        with patch.object(SemanticCrossoverOperator, 'crossover', new_callable=AsyncMock) as mock_crossover:
            mock_crossover.side_effect = [
                (GeneratedIdea(content="Fallback 1-1", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test"),
                 GeneratedIdea(content="Fallback 1-2", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test")),
                (GeneratedIdea(content="Fallback 2-1", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test"),
                 GeneratedIdea(content="Fallback 2-2", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test")),
                (GeneratedIdea(content="Fallback 3-1", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test"),
                 GeneratedIdea(content="Fallback 3-2", thinking_method=ThinkingMethod.QUESTIONING,
                              agent_name="Test", generation_prompt="Test"))
            ]
            
            results = await operator.crossover_batch(sample_parents)
            
            # Should have fallen back to sequential
            assert len(results) == 3
            assert mock_crossover.call_count == 3
            assert results[0][0].content == "Fallback 1-1"

    @pytest.mark.asyncio
    async def test_batch_vs_sequential_consistency(self, mock_llm_provider, sample_parents):
        """Test that batch processing produces similar quality to sequential."""
        batch_operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        sequential_operator = SemanticCrossoverOperator(mock_llm_provider)
        
        # Mock batch response
        batch_response = MagicMock()
        batch_response.content = json.dumps({
            "crossovers": [
                {
                    "pair_id": 0,
                    "offspring1": "Batch offspring 1: Comprehensive solution",
                    "offspring2": "Batch offspring 2: Integrated approach"
                }
            ]
        })
        batch_response.cost = 0.001
        
        # Test batch
        mock_llm_provider.generate.return_value = batch_response
        batch_results = await batch_operator.crossover_batch([sample_parents[0]])
        
        # Test sequential
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offspring_1": "Sequential offspring: Similar comprehensive solution 1",
            "offspring_2": "Sequential offspring: Similar comprehensive solution 2"
        })
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        sequential_results = await sequential_operator.crossover(sample_parents[0][0], sample_parents[0][1])
        
        # Both should produce GeneratedIdea objects
        assert isinstance(batch_results[0][0], GeneratedIdea)
        assert isinstance(sequential_results[0], GeneratedIdea)
        
        # Both should have similar structure
        assert batch_results[0][0].thinking_method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION]
        assert sequential_results[0].thinking_method in [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION]

    @pytest.mark.asyncio
    async def test_batch_crossover_performance(self, mock_llm_provider):
        """Test that batch crossover is faster than sequential for multiple pairs."""
        # Create 10 parent pairs
        many_pairs = []
        for i in range(10):
            parent1 = GeneratedIdea(
                content=f"Parent 1-{i}", thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="Test", generation_prompt="Test"
            )
            parent2 = GeneratedIdea(
                content=f"Parent 2-{i}", thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="Test", generation_prompt="Test"
            )
            many_pairs.append((parent1, parent2))
        
        batch_operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        # Mock batch response for all 10 pairs
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {"pair_id": i, "offspring1": f"Batch offspring {i}-1", "offspring2": f"Batch offspring {i}-2"}
                for i in range(10)
            ]
        })
        mock_response.cost = 0.005
        mock_llm_provider.generate.return_value = mock_response
        
        # Batch should make only 1 LLM call
        await batch_operator.crossover_batch(many_pairs)
        assert mock_llm_provider.generate.call_count == 1
        
        # Sequential would make 10 calls (simulated)
        sequential_operator = SemanticCrossoverOperator(mock_llm_provider)
        mock_llm_provider.generate.reset_mock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offspring_1": "Sequential offspring 1",
            "offspring_2": "Sequential offspring 2"
        })
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        
        for pair in many_pairs:
            await sequential_operator.crossover(pair[0], pair[1])
        
        assert mock_llm_provider.generate.call_count == 10

    @pytest.mark.asyncio
    async def test_batch_crossover_prompt_structure(self, mock_llm_provider, sample_parents):
        """Test that batch crossover creates proper structured prompt."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({"crossovers": [{"pair_id": 0, "offspring1": "O1", "offspring2": "O2"}]})
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        
        await operator.crossover_batch([sample_parents[0]])
        
        # Check prompt structure
        call_args = mock_llm_provider.generate.call_args
        # The prompt is in the LLMRequest object
        if call_args[0]:
            llm_request = call_args[0][0]
            prompt = llm_request.user_prompt
        else:
            prompt = ''
        
        assert "Pair 1:" in prompt
        assert "Parent 1:" in prompt
        assert "Parent 2:" in prompt
        assert sample_parents[0][0].content in prompt
        assert sample_parents[0][1].content in prompt

    @pytest.mark.asyncio
    async def test_batch_crossover_metadata_preservation(self, mock_llm_provider, sample_parents):
        """Test that offspring metadata is properly set."""
        operator = BatchSemanticCrossoverOperator(mock_llm_provider)
        
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "crossovers": [
                {"pair_id": 0, "offspring1": "O1", "offspring2": "O2"}
            ]
        })
        mock_response.cost = 0.001
        mock_llm_provider.generate.return_value = mock_response
        
        results = await operator.crossover_batch([sample_parents[0]])
        
        offspring1, offspring2 = results[0]
        
        # Check metadata
        assert 'crossover_type' in offspring1.metadata
        assert offspring1.metadata['crossover_type'] == 'semantic_batch'
        assert 'parent_ids' in offspring1.metadata
        assert 'llm_cost' in offspring1.metadata
        assert offspring1.metadata.get('from_cache', False) is False
        
        # Check parent tracking (parent_ideas is a field on GeneratedIdea, not in metadata)
        assert len(offspring1.parent_ideas) == 2