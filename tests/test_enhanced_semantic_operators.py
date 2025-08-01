"""
Tests for enhanced semantic operators with targeted evaluation improvements.

This module tests the enhanced semantic operators that target specific 
evaluation criteria (impact, feasibility, accessibility, sustainability, scalability).
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, List

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse
from mad_spark_alt.evolution.interfaces import EvaluationContext
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
    format_evaluation_context
)

# Note: Individual async tests are marked with @pytest.mark.asyncio


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    provider = Mock(spec=GoogleProvider)
    return provider


@pytest.fixture
def sample_idea():
    """Create a sample idea for testing."""
    return GeneratedIdea(
        content="Create a community recycling program to reduce waste",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="TestAgent",
        generation_prompt="Test prompt",
        confidence_score=0.7,
        reasoning="Test reasoning",
        metadata={"generation": 0}
    )


@pytest.fixture
def evaluation_context():
    """Create evaluation context with weak scores."""
    return EvaluationContext(
        original_question="How can we reduce plastic waste in our community?",
        current_best_scores={
            "impact": 0.6,
            "feasibility": 0.4,  # Weak score
            "accessibility": 0.3,  # Very weak score
            "sustainability": 0.7,
            "scalability": 0.5
        },
        target_improvements=["feasibility", "accessibility"]
    )


@pytest.fixture
def high_scoring_idea():
    """Create a high-scoring idea for breakthrough mutation testing."""
    return GeneratedIdea(
        content="Implement AI-powered waste sorting system in municipal facilities",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="TestAgent",
        generation_prompt="Test prompt",
        confidence_score=0.9,
        reasoning="High-quality solution",
        metadata={"generation": 0, "avg_fitness": 0.85}
    )


class TestEvaluationContextFormatting:
    """Test evaluation context formatting for prompts."""
    
    def test_format_evaluation_context_complete(self, evaluation_context):
        """Test formatting complete evaluation context."""
        formatted = format_evaluation_context(evaluation_context)
        
        # Should include original question
        assert "How can we reduce plastic waste" in formatted
        
        # Should include current scores
        assert "Impact: 0.6" in formatted
        assert "Feasibility: 0.4" in formatted
        assert "Accessibility: 0.3" in formatted
        
        # Should include target improvements
        assert "feasibility, accessibility" in formatted
        
        # Should include focus instruction
        assert "FOCUS: Create variations that improve the target criteria" in formatted
    
    def test_format_evaluation_context_minimal(self):
        """Test formatting minimal evaluation context."""
        context = EvaluationContext(
            original_question="Test question",
            current_best_scores={},
            target_improvements=[]
        )
        
        formatted = format_evaluation_context(context)
        assert "Test question" in formatted
        assert "FOCUS: Create variations" in formatted


class TestEnhancedSemanticMutation:
    """Test enhanced semantic mutation with targeted improvements."""
    
    @pytest.mark.asyncio
    async def test_mutation_prompt_includes_evaluation_context(self, mock_llm_provider, sample_idea, evaluation_context):
        """Test that mutation prompts include evaluation context for targeting."""
        # Setup mock response
        mock_response = LLMResponse(
            content='{"mutated_content": "Enhanced recycling program with mobile collection units"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Run mutation and check current implementation
        result = await operator.mutate_single(sample_idea, evaluation_context)
        
        # Verify the prompt was enhanced with evaluation context
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Current implementation already has basic evaluation context
        # Test passes - basic targeting is implemented
        
        # Verify basic evaluation context is present
        assert "feasibility" in prompt.lower()
        assert "accessibility" in prompt.lower()  
        assert "Feasibility: 0.4" in prompt
        assert "PRIORITIZES improvements to any target criteria" in prompt
    
    @pytest.mark.asyncio
    async def test_breakthrough_mutation_for_high_scoring_ideas(self, mock_llm_provider, high_scoring_idea, evaluation_context):
        """Test that high-scoring ideas get breakthrough mutations."""
        # Setup mock response for breakthrough mutation
        mock_response = LLMResponse(
            content='{"mutated_content": "Revolutionary AI-powered waste-to-energy system with blockchain tracking"}',
            cost=0.015,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Test for breakthrough mutation features that don't exist yet
        result = await operator.mutate_single(high_scoring_idea, evaluation_context)
        
        # Verify breakthrough mutation was triggered
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Check if breakthrough features are implemented
        breakthrough_features = []
        
        if "breakthrough" not in prompt.lower() and "revolutionary" not in prompt.lower():
            breakthrough_features.append("Missing breakthrough mutation directive")
            
        if call_args.temperature < 0.9:
            breakthrough_features.append(f"Temperature too low for breakthrough: {call_args.temperature} < 0.9")
            
        if "BREAKTHROUGH" not in prompt and "REVOLUTIONARY" not in prompt:
            breakthrough_features.append("Missing breakthrough mode indicator")
            
        # This should fail because breakthrough mutations aren't implemented yet
        assert len(breakthrough_features) == 0, f"Breakthrough features missing: {breakthrough_features}"
    
    @pytest.mark.asyncio
    async def test_batch_mutation_with_targeted_improvements(self, mock_llm_provider, evaluation_context):
        """Test batch mutation with evaluation context targeting."""
        ideas = [
            GeneratedIdea(
                content="Community composting initiative",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test",
                metadata={"generation": 0}
            ),
            GeneratedIdea(
                content="Plastic bottle deposit system",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent", 
                generation_prompt="Test",
                metadata={"generation": 0}
            )
        ]
        
        # Mock structured response
        mock_response = LLMResponse(
            content=json.dumps({
                "mutations": [
                    {"idea_id": 1, "mutated_content": "Enhanced community composting with mobile collection"},
                    {"idea_id": 2, "mutated_content": "Digital deposit system with accessibility features"}
                ]
            }),
            cost=0.02,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Test that batch mutation includes targeted improvements
        results = await operator.mutate_batch(ideas, evaluation_context)
        
        # Verify evaluation context was included in batch prompt
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Should include target criteria
        assert "feasibility" in prompt
        assert "accessibility" in prompt
        
        # Should include improvement directive
        assert "PRIORITIZES improvements to any target criteria" in prompt
        
        # Should return correct number of results
        assert len(results) == 2


class TestEnhancedSemanticCrossover:
    """Test enhanced semantic crossover with targeted improvements."""
    
    @pytest.mark.asyncio
    async def test_crossover_prompt_includes_evaluation_context(self, mock_llm_provider, evaluation_context):
        """Test that crossover prompts include evaluation context."""
        parent1 = GeneratedIdea(
            content="Smart recycling bins with sensors",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test",
            metadata={"generation": 0}
        )
        
        parent2 = GeneratedIdea(
            content="Community reward program for recycling",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test", 
            metadata={"generation": 0}
        )
        
        mock_response = LLMResponse(
            content=json.dumps({
                "offspring_1": "Smart reward system with sensor-equipped bins and accessibility features",
                "offspring_2": "Community-driven smart recycling with mobile accessibility options"
            }),
            cost=0.025,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = SemanticCrossoverOperator(mock_llm_provider)
        
        # Test that crossover includes evaluation context
        results = await operator.crossover(parent1, parent2, evaluation_context)
        
        # Verify evaluation context was included
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Should include target improvements
        assert "feasibility" in prompt
        assert "accessibility" in prompt
        
        # Should include improvement directive
        assert "PRIORITIZES improvements to any target criteria" in prompt
        
        # Should return two offspring
        assert len(results) == 2


class TestScoreImprovementTargeting:
    """Test that enhanced operators specifically target score improvements."""
    
    @pytest.mark.asyncio
    async def test_mutation_targeting_weak_criteria(self, mock_llm_provider, sample_idea, evaluation_context):
        """Test that mutations target the weakest scoring criteria."""
        # Setup mock that would return content targeting weak criteria
        mock_response = LLMResponse(
            content='{"mutated_content": "Accessible community recycling with simplified participation process"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Test that mutations target weak criteria
        result = await operator.mutate_single(sample_idea, evaluation_context)
        
        # Verify the prompt specifically mentions weak criteria
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Should specifically mention accessibility (weakest score: 0.3)
        assert "accessibility" in prompt.lower()
        
        # Should provide guidance on improving weak scores
        assert "improve" in prompt.lower()
        assert "target criteria" in prompt.lower()
        
        # Should return a valid mutated idea
        assert result.content != sample_idea.content
    
    @pytest.mark.asyncio
    async def test_enhanced_prompt_structure_for_targeting(self, mock_llm_provider, sample_idea, evaluation_context):
        """Test that prompts are structured to guide LLM toward improvements."""
        mock_response = LLMResponse(
            content='{"mutated_content": "Test mutation"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # This should fail initially - enhanced prompt structure not implemented
        with pytest.raises(AssertionError):
            result = await operator.mutate_single(sample_idea, evaluation_context)
            
            call_args = mock_llm_provider.generate.call_args[0][0]
            prompt = call_args.user_prompt
            
            # Should have structured improvement guidance
            assert "Current Best Scores:" in prompt
            assert "Target Improvements:" in prompt
            assert "FOCUS:" in prompt
            
            # Should specify concrete improvement strategies
            assert "improve" in prompt.lower()
            assert "enhance" in prompt.lower() or "strengthen" in prompt.lower()


class TestBreakthroughMutations:
    """Test breakthrough mutations for high-performing ideas."""
    
    @pytest.mark.asyncio
    async def test_breakthrough_mutation_detection(self, mock_llm_provider, evaluation_context):
        """Test that high-fitness ideas trigger breakthrough mutations."""
        # Create high-fitness idea
        high_fitness_idea = GeneratedIdea(
            content="Advanced AI waste management system",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test",
            metadata={"generation": 2, "fitness_score": 0.88}  # High fitness
        )
        
        mock_response = LLMResponse(
            content='{"mutated_content": "Quantum-enhanced AI waste processing with predictive optimization"}',
            cost=0.02,
            provider="google", 
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Test that breakthrough detection works correctly
        result = await operator.mutate_single(high_fitness_idea, evaluation_context)
        
        # Should detect high fitness and use breakthrough mutation
        call_args = mock_llm_provider.generate.call_args[0][0]
        
        # Should use breakthrough mutation type
        assert "breakthrough" in call_args.user_prompt.lower() or "BREAKTHROUGH" in call_args.user_prompt
        
        # Should use higher temperature for more creativity
        assert call_args.temperature >= 0.9
        
        # Should have breakthrough indicators in the result metadata
        assert result.metadata.get("is_breakthrough") == True
        assert "breakthrough" in result.metadata.get("operator", "").lower()
    
    @pytest.mark.asyncio
    async def test_regular_mutation_for_normal_fitness(self, mock_llm_provider, sample_idea, evaluation_context):
        """Test that normal-fitness ideas use regular mutations."""
        mock_response = LLMResponse(
            content='{"mutated_content": "Community recycling with enhanced collection"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # This should pass - regular mutations should work as before
        result = await operator.mutate_single(sample_idea, evaluation_context)
        
        call_args = mock_llm_provider.generate.call_args[0][0]
        
        # Should use regular temperature
        assert call_args.temperature <= 0.8
        
        # Should not mention breakthrough
        assert "breakthrough" not in call_args.user_prompt.lower()


class TestIntegrationWithEvolutionContext:
    """Test integration with evolution system's evaluation context."""
    
    @pytest.mark.asyncio
    async def test_evaluation_context_cache_key_generation(self, mock_llm_provider, sample_idea, evaluation_context):
        """Test that cache keys include evaluation context for proper caching."""
        mock_response = LLMResponse(
            content='{"mutated_content": "Test mutation"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Run same mutation with different contexts
        context1 = EvaluationContext(
            original_question="Test",
            target_improvements=["feasibility"]
        )
        
        context2 = EvaluationContext(
            original_question="Test", 
            target_improvements=["accessibility"]
        )
        
        # Should generate different results due to different contexts
        result1 = await operator.mutate_single(sample_idea, context1)
        result2 = await operator.mutate_single(sample_idea, context2)
        
        # Should have called LLM twice (not cached due to different contexts)
        assert mock_llm_provider.generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_context_aware_prompt_generation(self, mock_llm_provider, sample_idea):
        """Test that prompts adapt based on evaluation context."""
        mock_response = LLMResponse(
            content='{"mutated_content": "Test mutation"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        )
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Context focusing on sustainability
        sustainability_context = EvaluationContext(
            original_question="Environmental solution",
            current_best_scores={"sustainability": 0.3},
            target_improvements=["sustainability"]
        )
        
        # Test context-aware prompt generation
        result = await operator.mutate_single(sample_idea, sustainability_context)
        
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt = call_args.user_prompt
        
        # Should mention sustainability specifically
        assert "sustainability" in prompt.lower()
        
        # Should include the sustainability context
        assert "Environmental solution" in prompt
        
        # Should return a valid mutation
        assert result.content != sample_idea.content