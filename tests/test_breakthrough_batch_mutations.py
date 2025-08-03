"""
Tests for breakthrough batch mutation functionality.

This module tests the enhancement of batch mutations to properly handle
high-scoring ideas with revolutionary mutation parameters.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import EvaluationContext
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator


@pytest.fixture
def mixed_fitness_ideas() -> List[GeneratedIdea]:
    """Create ideas with mixed fitness scores for breakthrough testing."""
    ideas = []
    
    # High-scoring ideas (should get breakthrough treatment)
    for i in range(3):
        idea = GeneratedIdea(
            content=f"High-performing idea {i}: Revolutionary approach to sustainability",
            thinking_method=ThinkingMethod.INDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.85 + i * 0.05,  # 0.85, 0.90, 0.95
            metadata={"fitness": 0.85 + i * 0.05}
        )
        ideas.append(idea)
    
    # Regular ideas
    for i in range(3):
        idea = GeneratedIdea(
            content=f"Regular idea {i}: Standard improvement approach",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.5 + i * 0.1,  # 0.5, 0.6, 0.7
            metadata={"fitness": 0.5 + i * 0.1}
        )
        ideas.append(idea)
    
    return ideas


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
        original_question="How can we achieve carbon neutrality?",
        target_improvements=["impact", "scalability"],
        weak_criteria=["feasibility"],
        generation_number=3
    )


class TestBreakthroughBatchMutations:
    """Test breakthrough batch mutation functionality."""

    def test_identify_breakthrough_ideas(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that high-scoring ideas are correctly identified for breakthrough treatment."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_ideas = []
        regular_ideas = []
        
        for idea in mixed_fitness_ideas:
            if operator._is_high_scoring_idea(idea):
                breakthrough_ideas.append(idea)
            else:
                regular_ideas.append(idea)
        
        # Should have 3 breakthrough and 3 regular
        assert len(breakthrough_ideas) == 3
        assert len(regular_ideas) == 3
        
        # Verify correct classification
        assert all(idea.confidence_score >= 0.8 for idea in breakthrough_ideas)
        assert all(idea.confidence_score < 0.8 for idea in regular_ideas)

    @pytest.mark.asyncio
    async def test_separate_batch_processing(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough and regular ideas are processed in separate batches."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Mock responses for two separate batches
        breakthrough_response = {
            "mutations": [
                {"id": 0, "content": "BREAKTHROUGH: Paradigm shift in carbon capture"},
                {"id": 1, "content": "BREAKTHROUGH: System integration for net-zero"},
                {"id": 2, "content": "BREAKTHROUGH: Future-forward climate solution"}
            ]
        }
        
        regular_response = {
            "mutations": [
                {"id": 0, "content": "REGULAR: Incremental efficiency improvement"},
                {"id": 1, "content": "REGULAR: Standard optimization approach"},
                {"id": 2, "content": "REGULAR: Conventional sustainability measure"}
            ]
        }
        
        # Set up mock to return different responses
        mock_llm_provider.generate.side_effect = [breakthrough_response, regular_response]
        
        results = await operator.mutate_batch(mixed_fitness_ideas)
        
        # Should have made 2 LLM calls (one for each batch)
        assert mock_llm_provider.generate.call_count == 2
        
        # Verify results maintain order
        assert len(results) == 6
        assert all("BREAKTHROUGH" in results[i].content for i in range(3))
        assert all("REGULAR" in results[i].content for i in range(3, 6))

    @pytest.mark.asyncio
    async def test_breakthrough_temperature_parameter(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough mutations use higher temperature."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Take only high-scoring ideas
        breakthrough_ideas = [idea for idea in mixed_fitness_ideas if idea.confidence_score >= 0.8]
        
        mock_response = {
            "mutations": [
                {"id": i, "content": f"Breakthrough mutation {i}"} 
                for i in range(len(breakthrough_ideas))
            ]
        }
        mock_llm_provider.generate.return_value = mock_response
        
        await operator.mutate_batch(breakthrough_ideas)
        
        # Check that temperature 0.95 was used
        call_args = mock_llm_provider.generate.call_args
        assert call_args.kwargs.get('temperature', 0) == 0.95

    @pytest.mark.asyncio
    async def test_breakthrough_token_limits(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough mutations get double token limits."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_ideas = [idea for idea in mixed_fitness_ideas if idea.confidence_score >= 0.8]
        
        mock_response = {
            "mutations": [
                {"id": i, "content": f"Breakthrough mutation {i}"} 
                for i in range(len(breakthrough_ideas))
            ]
        }
        mock_llm_provider.generate.return_value = mock_response
        
        await operator.mutate_batch(breakthrough_ideas)
        
        # Check that max_tokens was doubled
        call_args = mock_llm_provider.generate.call_args
        max_tokens = call_args.kwargs.get('max_tokens', 0)
        assert max_tokens >= 1000  # Should be at least 2x the default

    @pytest.mark.asyncio
    async def test_breakthrough_prompt_content(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough mutations use revolutionary prompts."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_ideas = [idea for idea in mixed_fitness_ideas if idea.confidence_score >= 0.8]
        
        mock_response = {
            "mutations": [
                {"id": i, "content": f"Mutation {i}"} 
                for i in range(len(breakthrough_ideas))
            ]
        }
        mock_llm_provider.generate.return_value = mock_response
        
        await operator.mutate_batch(breakthrough_ideas)
        
        # Check prompt contains breakthrough language
        call_args = mock_llm_provider.generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args.kwargs.get('prompt', '')
        
        assert "REVOLUTIONARY" in prompt or "revolutionary" in prompt.lower()
        assert any(mutation_type in prompt.lower() for mutation_type in [
            "paradigm_shift", "system_integration", "scale_amplification", "future_forward"
        ])

    @pytest.mark.asyncio
    async def test_breakthrough_metadata_tracking(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough mutations are properly tracked in metadata."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_response = {
            "mutations": [
                {"id": 0, "content": "Paradigm shift mutation", "mutation_type": "paradigm_shift"},
                {"id": 1, "content": "System integration mutation", "mutation_type": "system_integration"},
                {"id": 2, "content": "Scale amplification mutation", "mutation_type": "scale_amplification"}
            ]
        }
        
        regular_response = {
            "mutations": [
                {"id": 0, "content": "Regular mutation 1"},
                {"id": 1, "content": "Regular mutation 2"},
                {"id": 2, "content": "Regular mutation 3"}
            ]
        }
        
        mock_llm_provider.generate.side_effect = [breakthrough_response, regular_response]
        
        results = await operator.mutate_batch(mixed_fitness_ideas)
        
        # Check breakthrough metadata
        for i in range(3):
            assert results[i].metadata.get('mutation_type') in [
                'paradigm_shift', 'system_integration', 'scale_amplification', 'future_forward'
            ]
            assert results[i].metadata.get('is_breakthrough', False) is True
        
        # Check regular metadata
        for i in range(3, 6):
            assert results[i].metadata.get('mutation_type', 'batch_mutation') == 'batch_mutation'
            assert results[i].metadata.get('is_breakthrough', False) is False

    @pytest.mark.asyncio
    async def test_mixed_batch_caching(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that breakthrough status affects cache keys."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # First call
        breakthrough_response = {
            "mutations": [{"id": i, "content": f"Breakthrough {i}"} for i in range(3)]
        }
        regular_response = {
            "mutations": [{"id": i, "content": f"Regular {i}"} for i in range(3)]
        }
        mock_llm_provider.generate.side_effect = [breakthrough_response, regular_response]
        
        results1 = await operator.mutate_batch(mixed_fitness_ideas)
        assert mock_llm_provider.generate.call_count == 2
        
        # Reset mock
        mock_llm_provider.generate.reset_mock()
        mock_llm_provider.generate.side_effect = [breakthrough_response, regular_response]
        
        # Second call should use cache
        results2 = await operator.mutate_batch(mixed_fitness_ideas)
        assert mock_llm_provider.generate.call_count == 0  # All cached
        
        # Results should match
        for i in range(6):
            assert results1[i].content == results2[i].content

    @pytest.mark.asyncio
    async def test_breakthrough_with_context(self, mock_llm_provider, mixed_fitness_ideas, evaluation_context):
        """Test breakthrough mutations with evaluation context."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_ideas = [idea for idea in mixed_fitness_ideas if idea.confidence_score >= 0.8]
        
        mock_response = {
            "mutations": [
                {"id": i, "content": f"Context-aware breakthrough {i}"} 
                for i in range(len(breakthrough_ideas))
            ]
        }
        mock_llm_provider.generate.return_value = mock_response
        
        await operator.mutate_batch(breakthrough_ideas, evaluation_context)
        
        # Check that context influences prompt
        call_args = mock_llm_provider.generate.call_args
        prompt = call_args[0][0] if call_args[0] else call_args.kwargs.get('prompt', '')
        
        assert "impact" in prompt.lower()
        assert "scalability" in prompt.lower()
        assert "carbon neutrality" in prompt

    @pytest.mark.asyncio
    async def test_empty_breakthrough_batch(self, mock_llm_provider):
        """Test handling when all ideas are regular (no breakthrough)."""
        # Create only low-scoring ideas
        regular_ideas = [
            GeneratedIdea(
                content=f"Regular idea {i}",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="Test",
                generation_prompt="Test",
                confidence_score=0.5 + i * 0.1
            ) for i in range(3)
        ]
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        regular_response = {
            "mutations": [{"id": i, "content": f"Regular mutation {i}"} for i in range(3)]
        }
        mock_llm_provider.generate.return_value = regular_response
        
        results = await operator.mutate_batch(regular_ideas)
        
        # Should make only 1 call (no breakthrough batch)
        assert mock_llm_provider.generate.call_count == 1
        assert len(results) == 3
        assert all(not idea.metadata.get('is_breakthrough', False) for idea in results)

    @pytest.mark.asyncio
    async def test_all_breakthrough_batch(self, mock_llm_provider):
        """Test handling when all ideas are breakthrough."""
        # Create only high-scoring ideas
        breakthrough_ideas = [
            GeneratedIdea(
                content=f"Breakthrough idea {i}",
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="Test",
                generation_prompt="Test",
                confidence_score=0.85 + i * 0.05
            ) for i in range(3)
        ]
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        breakthrough_response = {
            "mutations": [
                {"id": i, "content": f"Revolutionary mutation {i}", "mutation_type": "paradigm_shift"} 
                for i in range(3)
            ]
        }
        mock_llm_provider.generate.return_value = breakthrough_response
        
        results = await operator.mutate_batch(breakthrough_ideas)
        
        # Should make only 1 call (no regular batch)
        assert mock_llm_provider.generate.call_count == 1
        assert len(results) == 3
        assert all(idea.metadata.get('is_breakthrough', False) for idea in results)
        
    @pytest.mark.asyncio
    async def test_breakthrough_error_handling(self, mock_llm_provider, mixed_fitness_ideas):
        """Test error handling in breakthrough batch processing."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Mock error for breakthrough batch
        mock_llm_provider.generate.side_effect = [
            Exception("Breakthrough batch failed"),
            {"mutations": [{"id": i, "content": f"Regular {i}"} for i in range(3)]}
        ]
        
        # Should still process regular batch and use fallback for breakthrough
        results = await operator.mutate_batch(mixed_fitness_ideas)
        
        assert len(results) == 6
        # Breakthrough ideas should have fallback mutations
        assert all(results[i].content != mixed_fitness_ideas[i].content for i in range(3))

    @pytest.mark.asyncio
    async def test_breakthrough_batch_order_preservation(self, mock_llm_provider, mixed_fitness_ideas):
        """Test that original order is preserved despite batch separation."""
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Interleave high and low scoring ideas
        interleaved = []
        for i in range(3):
            interleaved.append(mixed_fitness_ideas[i])  # High
            interleaved.append(mixed_fitness_ideas[i + 3])  # Low
        
        breakthrough_response = {
            "mutations": [
                {"id": i, "content": f"BREAKTHROUGH-{i}"} for i in range(3)
            ]
        }
        regular_response = {
            "mutations": [
                {"id": i, "content": f"REGULAR-{i}"} for i in range(3)
            ]
        }
        
        mock_llm_provider.generate.side_effect = [breakthrough_response, regular_response]
        
        results = await operator.mutate_batch(interleaved)
        
        # Verify order is preserved
        assert results[0].content == "BREAKTHROUGH-0"  # First high
        assert results[1].content == "REGULAR-0"       # First low
        assert results[2].content == "BREAKTHROUGH-1"  # Second high
        assert results[3].content == "REGULAR-1"       # Second low
        assert results[4].content == "BREAKTHROUGH-2"  # Third high
        assert results[5].content == "REGULAR-2"       # Third low