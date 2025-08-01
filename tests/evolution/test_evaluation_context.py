"""
Tests for evaluation context enhancement in evolution system.

This module tests the new evaluation context functionality that passes
original question and scoring information to semantic operators.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import asdict

from mad_spark_alt.evolution.interfaces import (
    EvolutionRequest, EvolutionConfig, IndividualFitness, EvaluationContext
)
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator, SemanticCrossoverOperator, format_evaluation_context
from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse


class TestEvaluationContext:
    """Test evaluation context functionality."""

    @pytest.fixture
    def sample_idea(self):
        """Create a sample idea for testing."""
        return GeneratedIdea(
            content="Create a community garden program",
            thinking_method="abduction",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8
        )

    @pytest.fixture
    def sample_individual(self, sample_idea):
        """Create a sample individual for testing."""
        return IndividualFitness(
            idea=sample_idea,
            impact=0.7,
            feasibility=0.5,  # Low feasibility to test targeting
            accessibility=0.8,
            sustainability=0.6,
            scalability=0.7,
            overall_fitness=0.66
        )

    @pytest.fixture
    def evaluation_context(self):
        """Create sample evaluation context."""
        return EvaluationContext(
            original_question="How can we reduce food waste in our community?",
            current_best_scores={
                "impact": 0.7,
                "feasibility": 0.5,
                "accessibility": 0.8,
                "sustainability": 0.6,
                "scalability": 0.7
            },
            target_improvements=["feasibility", "sustainability"],
            evaluation_criteria=["impact", "feasibility", "accessibility", "sustainability", "scalability"]
        )

    @pytest.fixture
    def evolution_request_with_context(self, sample_individual, evaluation_context):
        """Create evolution request with evaluation context."""
        return EvolutionRequest(
            initial_population=[sample_individual.idea],
            config=EvolutionConfig(population_size=2, generations=2),
            context="general improvement",
            evaluation_context=evaluation_context
        )

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=GoogleProvider)
        provider.generate = AsyncMock(return_value=LLMResponse(
            content='{"mutated_content": "Enhanced community garden program with improved feasibility through partnerships"}',
            cost=0.01,
            provider="google",
            model="gemini-pro"
        ))
        return provider

    def test_evaluation_context_creation(self, evaluation_context):
        """Test EvaluationContext dataclass creation and validation."""
        assert evaluation_context.original_question == "How can we reduce food waste in our community?"
        assert evaluation_context.current_best_scores["feasibility"] == 0.5
        assert "feasibility" in evaluation_context.target_improvements
        assert "sustainability" in evaluation_context.target_improvements
        assert len(evaluation_context.evaluation_criteria) == 5

    def test_evolution_request_with_context(self, evolution_request_with_context):
        """Test EvolutionRequest includes evaluation context."""
        assert evolution_request_with_context.evaluation_context is not None
        assert evolution_request_with_context.evaluation_context.original_question is not None
        assert evolution_request_with_context.evaluation_context.current_best_scores is not None

    @pytest.mark.asyncio
    async def test_semantic_mutation_receives_context(self, mock_llm_provider, evaluation_context, sample_idea):
        """Test that semantic mutation operator receives and uses evaluation context."""
        # Create semantic mutation operator
        mutation_operator = BatchSemanticMutationOperator(mock_llm_provider)
        
        # Test mutation with context
        mutated_idea = await mutation_operator.mutate_single(sample_idea, evaluation_context)
        
        # Verify LLM was called
        mock_llm_provider.generate.assert_called_once()
        
        # Check that the prompt includes context information
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt_text = call_args.user_prompt.lower()
        
        # Should include original question
        assert "food waste" in prompt_text or "community" in prompt_text
        
        # Should include context about improving feasibility
        assert "feasibility" in prompt_text or "practical" in prompt_text

    @pytest.mark.asyncio
    async def test_semantic_crossover_receives_context(
        self, mock_llm_provider, evaluation_context, sample_idea
    ):
        """Test that semantic crossover operator receives and uses evaluation context."""
        # Create second idea for crossover
        idea2 = GeneratedIdea(
            content="Start a composting initiative",
            thinking_method="deduction",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.7
        )
        
        # Mock crossover response
        mock_llm_provider.generate.return_value = LLMResponse(
            content='{"offspring_1": "Community garden with composting", "offspring_2": "Composting with garden education"}',
            cost=0.02,
            provider="google",
            model="gemini-pro"
        )
        
        # Create semantic crossover operator
        crossover_operator = SemanticCrossoverOperator(mock_llm_provider)
        
        # Test crossover with context
        offspring1, offspring2 = await crossover_operator.crossover(sample_idea, idea2, evaluation_context)
        
        # Verify LLM was called
        mock_llm_provider.generate.assert_called_once()
        
        # Check that the prompt includes context information
        call_args = mock_llm_provider.generate.call_args[0][0]
        prompt_text = call_args.user_prompt.lower()
        
        # Should include original question context
        assert "food waste" in prompt_text or "community" in prompt_text

    def test_context_prompt_enhancement_mutation(self, evaluation_context):
        """Test that mutation prompts are enhanced with context information."""
        # Test the prompt formatting that will be implemented
        original_question = evaluation_context.original_question
        current_scores = evaluation_context.current_best_scores
        target_improvements = evaluation_context.target_improvements
        
        # Expected context formatting
        expected_context_parts = [
            f"Original Question: {original_question}",
            "Current Best Scores:",
            f"Impact: {current_scores['impact']:.1f}",
            f"Feasibility: {current_scores['feasibility']:.1f}",
            "Target Improvements: feasibility, sustainability"
        ]
        
        # This tests the format we'll implement in semantic operators
        context_string = format_evaluation_context(evaluation_context)
        
        for part in expected_context_parts:
            assert part in context_string

    def test_context_prompt_enhancement_crossover(self, evaluation_context):
        """Test that crossover prompts are enhanced with context information."""
        context_string = format_evaluation_context(evaluation_context)
        
        # Should include scoring guidance
        assert "Current Best Scores" in context_string
        assert "Target Improvements" in context_string
        assert evaluation_context.original_question in context_string

    def test_context_serialization(self, evaluation_context):
        """Test that evaluation context can be serialized for logging/debugging."""
        context_dict = asdict(evaluation_context)
        
        assert "original_question" in context_dict
        assert "current_best_scores" in context_dict
        assert "target_improvements" in context_dict
        assert "evaluation_criteria" in context_dict
        
        # Should be JSON-serializable
        import json
        json_str = json.dumps(context_dict)
        parsed_back = json.loads(json_str)
        assert parsed_back["original_question"] == evaluation_context.original_question

    def test_context_validation(self):
        """Test evaluation context validation."""
        # Valid context
        valid_context = EvaluationContext(
            original_question="Test question?",
            current_best_scores={"impact": 0.5, "feasibility": 0.7},
            target_improvements=["impact"],
            evaluation_criteria=["impact", "feasibility"]
        )
        assert valid_context.original_question is not None
        assert len(valid_context.current_best_scores) > 0
        
        # Context with missing scores
        context_missing_scores = EvaluationContext(
            original_question="Test question?",
            current_best_scores={},  # Empty scores
            target_improvements=["impact"],
            evaluation_criteria=["impact"]
        )
        # Should still be valid, just less informative
        assert context_missing_scores.original_question is not None

    def test_target_improvements_identification(self):
        """Test logic for identifying which criteria need improvement."""
        scores = {
            "impact": 0.9,      # High - good
            "feasibility": 0.3, # Low - needs improvement  
            "accessibility": 0.8, # High - good
            "sustainability": 0.4, # Low - needs improvement
            "scalability": 0.7    # Medium - okay
        }
        
        # Logic to identify low scores (< 0.6) as targets for improvement
        targets = [criterion for criterion, score in scores.items() if score < 0.6]
        
        assert "feasibility" in targets
        assert "sustainability" in targets
        assert "impact" not in targets
        assert "accessibility" not in targets
        assert len(targets) == 2

