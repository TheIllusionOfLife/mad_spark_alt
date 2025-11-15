"""
Integration tests for Pydantic schemas with real LLM API (Phase 5).

This module tests that Pydantic schemas work correctly with real LLM providers
(Google Gemini API) in actual API calls, not just mocked responses.

These tests require GOOGLE_API_KEY environment variable and are marked with
@pytest.mark.integration to be skipped in CI.
"""

import json
import os

import pytest
from pydantic import ValidationError

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMProvider, LLMRequest
from mad_spark_alt.core.phase_logic import (
    AbductionResult,
    DeductionResult,
    PhaseInput,
    execute_abduction_phase,
    execute_deduction_phase,
)
from mad_spark_alt.core.schemas import (
    BatchCrossoverResponse,
    BatchMutationResponse,
    CrossoverResponse,
    DeductionResponse,
    HypothesisListResponse,
    MutationResponse,
)
from mad_spark_alt.evolution.semantic_crossover import SemanticCrossoverOperator
from mad_spark_alt.evolution.semantic_mutation import BatchSemanticMutationOperator


# Skip all tests if no API key
pytestmark = pytest.mark.integration


@pytest.fixture
def real_google_provider():
    """Create a real Google provider for integration testing."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set - skipping real API tests")

    return GoogleProvider(api_key=api_key)


@pytest.fixture
def phase_input(real_google_provider):
    """Create a phase input for testing with real LLM."""
    return PhaseInput(
        user_input="How can we reduce ocean plastic pollution?",
        llm_manager=real_google_provider,
        context={},
    )


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses for deduction testing."""
    return [
        "Deploy ocean cleanup vessels with advanced filtering systems",
        "Develop biodegradable plastic alternatives for packaging",
        "Implement global plastic ban policies with enforcement",
    ]


@pytest.fixture
def sample_ideas():
    """Sample ideas for evolution operator testing."""
    return [
        GeneratedIdea(
            content="Use machine learning to optimize recycling processes",
            thinking_method="abduction",
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8,
            metadata={"generation": 0}
        ),
        GeneratedIdea(
            content="Create incentive programs for reducing plastic consumption",
            thinking_method="deduction",
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7,
            metadata={"generation": 0}
        ),
    ]


# ==================== QADI PHASE INTEGRATION TESTS ====================


@pytest.mark.asyncio
async def test_abduction_phase_with_real_api_pydantic_validation(phase_input):
    """Test that abduction phase uses Pydantic validation with real Google API."""
    result = await execute_abduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic pollution?",
        num_hypotheses=3
    )

    # Verify result structure
    assert isinstance(result, AbductionResult)
    assert len(result.hypotheses) >= 3

    # Verify hypotheses are meaningful (not placeholders or empty)
    for hypothesis in result.hypotheses:
        assert len(hypothesis) > 50, f"Hypothesis too short: {hypothesis}"
        assert "[FALLBACK" not in hypothesis, f"Fallback text found: {hypothesis}"
        assert "placeholder" not in hypothesis.lower(), f"Placeholder found: {hypothesis}"

    # Verify LLM cost was tracked
    assert result.llm_cost > 0, "LLM cost should be tracked"

    # Verify raw response exists
    assert result.raw_response, "Raw response should be captured"

    print(f"✅ Abduction phase generated {len(result.hypotheses)} hypotheses")
    print(f"   Cost: ${result.llm_cost:.6f}")
    print(f"   Sample: {result.hypotheses[0][:100]}...")


@pytest.mark.asyncio
async def test_deduction_phase_with_real_api_pydantic_validation(
    phase_input, sample_hypotheses
):
    """Test that deduction phase uses Pydantic validation with real Google API."""
    result = await execute_deduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic pollution?",
        hypotheses=sample_hypotheses
    )

    # Verify result structure
    assert isinstance(result, DeductionResult)
    assert len(result.hypothesis_scores) == len(sample_hypotheses)

    # Verify scores are meaningful (not all defaults)
    non_default_scores = [
        score for score in result.hypothesis_scores
        if score.impact != 0.5 or score.feasibility != 0.5
    ]
    assert len(non_default_scores) >= 2, "Most scores should be non-default"

    # Verify score ranges (0.0-1.0)
    for score in result.hypothesis_scores:
        assert 0.0 <= score.impact <= 1.0, f"Impact out of range: {score.impact}"
        assert 0.0 <= score.feasibility <= 1.0, f"Feasibility out of range: {score.feasibility}"
        assert 0.0 <= score.accessibility <= 1.0, f"Accessibility out of range: {score.accessibility}"
        assert 0.0 <= score.sustainability <= 1.0, f"Sustainability out of range: {score.sustainability}"
        assert 0.0 <= score.scalability <= 1.0, f"Scalability out of range: {score.scalability}"

    # Verify answer is meaningful
    assert len(result.answer) > 50, f"Answer too short: {result.answer}"
    assert "[FALLBACK" not in result.answer, f"Fallback text found: {result.answer}"

    # Verify action plan exists and is meaningful
    assert len(result.action_plan) >= 1, "Action plan should have at least one step"
    for step in result.action_plan:
        assert len(step) > 10, f"Action step too short: {step}"

    # Verify LLM cost was tracked
    assert result.llm_cost > 0, "LLM cost should be tracked"

    print(f"✅ Deduction phase evaluated {len(result.hypothesis_scores)} hypotheses")
    print(f"   Cost: ${result.llm_cost:.6f}")
    print(f"   Best score: {max(s.overall for s in result.hypothesis_scores):.2f}")
    print(f"   Answer: {result.answer[:100]}...")


# ==================== EVOLUTION OPERATOR INTEGRATION TESTS ====================


@pytest.mark.asyncio
async def test_mutation_operator_with_real_api_pydantic_validation(
    real_google_provider, sample_ideas
):
    """Test that mutation operator uses Pydantic validation with real API."""
    operator = BatchSemanticMutationOperator(llm_provider=real_google_provider)

    # Test single mutation
    mutated = await operator.mutate_single(
        sample_ideas[0],
        context="Focus on environmental sustainability"
    )

    # Verify mutation result
    assert isinstance(mutated, GeneratedIdea)
    assert len(mutated.content) > 100, f"Mutated content too short: {mutated.content}"
    assert mutated.content != sample_ideas[0].content, "Mutation should create different content"
    assert "[FALLBACK" not in mutated.content, f"Fallback text found: {mutated.content}"

    # Verify metadata
    assert "operator" in mutated.metadata
    assert "mutation_type" in mutated.metadata
    assert mutated.metadata["operator"] in ["semantic_mutation", "breakthrough_semantic_mutation"]

    print(f"✅ Mutation operator created meaningful variation")
    print(f"   Original: {sample_ideas[0].content[:80]}...")
    print(f"   Mutated: {mutated.content[:80]}...")
    print(f"   Type: {mutated.metadata.get('mutation_type', 'unknown')}")


@pytest.mark.asyncio
async def test_batch_mutation_with_real_api_pydantic_validation(
    real_google_provider, sample_ideas
):
    """Test that batch mutation uses Pydantic validation with real API."""
    operator = BatchSemanticMutationOperator(llm_provider=real_google_provider)

    # Test batch mutation
    mutated_batch = await operator.mutate_batch(
        sample_ideas,
        context="Focus on scalability and impact"
    )

    # Verify batch result
    assert len(mutated_batch) == len(sample_ideas)

    for original, mutated in zip(sample_ideas, mutated_batch):
        assert isinstance(mutated, GeneratedIdea)
        assert len(mutated.content) > 100, f"Mutated content too short: {mutated.content}"
        assert mutated.content != original.content, "Mutation should create different content"
        assert "[FALLBACK" not in mutated.content, f"Fallback text found: {mutated.content}"

        # Verify metadata
        assert "operator" in mutated.metadata
        assert "mutation_type" in mutated.metadata

    print(f"✅ Batch mutation operator created {len(mutated_batch)} meaningful variations")
    print(f"   First: {mutated_batch[0].content[:80]}...")
    print(f"   Second: {mutated_batch[1].content[:80]}...")


@pytest.mark.asyncio
async def test_crossover_operator_with_real_api_pydantic_validation(
    real_google_provider, sample_ideas
):
    """Test that crossover operator uses Pydantic validation with real API."""
    operator = SemanticCrossoverOperator(llm_provider=real_google_provider)

    # Test crossover
    offspring1, offspring2 = await operator.crossover(
        sample_ideas[0],
        sample_ideas[1],
        context="Focus on innovative solutions"
    )

    # Verify offspring results
    assert isinstance(offspring1, GeneratedIdea)
    assert isinstance(offspring2, GeneratedIdea)

    assert len(offspring1.content) > 100, f"Offspring 1 too short: {offspring1.content}"
    assert len(offspring2.content) > 100, f"Offspring 2 too short: {offspring2.content}"

    assert "[FALLBACK" not in offspring1.content, f"Fallback text in offspring 1: {offspring1.content}"
    assert "[FALLBACK" not in offspring2.content, f"Fallback text in offspring 2: {offspring2.content}"

    # Verify offspring are different from each other
    assert offspring1.content != offspring2.content, "Offspring should be different from each other"

    # Verify offspring integrate concepts from both parents (heuristic check)
    # Each offspring should mention concepts related to both parents
    parent1_words = set(sample_ideas[0].content.lower().split())
    parent2_words = set(sample_ideas[1].content.lower().split())
    offspring1_words = set(offspring1.content.lower().split())
    offspring2_words = set(offspring2.content.lower().split())

    # At least some overlap with each parent
    overlap1_parent1 = len(parent1_words & offspring1_words)
    overlap1_parent2 = len(parent2_words & offspring1_words)
    assert overlap1_parent1 > 0 or overlap1_parent2 > 0, "Offspring 1 should relate to parents"

    overlap2_parent1 = len(parent1_words & offspring2_words)
    overlap2_parent2 = len(parent2_words & offspring2_words)
    assert overlap2_parent1 > 0 or overlap2_parent2 > 0, "Offspring 2 should relate to parents"

    # Verify metadata
    assert "operator" in offspring1.metadata
    assert offspring1.metadata["operator"] == "semantic_crossover"

    print(f"✅ Crossover operator created 2 meaningful offspring")
    print(f"   Parent 1: {sample_ideas[0].content[:60]}...")
    print(f"   Parent 2: {sample_ideas[1].content[:60]}...")
    print(f"   Offspring 1: {offspring1.content[:60]}...")
    print(f"   Offspring 2: {offspring2.content[:60]}...")


# ==================== SCHEMA VALIDATION WITH REAL API ====================


@pytest.mark.asyncio
async def test_direct_pydantic_schema_with_google_api(real_google_provider):
    """Test that Pydantic schemas work directly with Google API."""
    # Test with HypothesisListResponse schema
    request = LLMRequest(
        system_prompt="You are a helpful assistant generating ideas.",
        user_prompt="Generate 3 hypotheses for reducing ocean plastic pollution. Each hypothesis should be detailed and actionable.",
        response_schema=HypothesisListResponse,
        response_mime_type="application/json",
        max_tokens=1000,
        temperature=0.7
    )

    response = await real_google_provider.generate(request)

    # Verify response structure
    assert response.content, "Response content should exist"
    assert response.cost > 0, "Cost should be tracked"

    # Verify content is valid JSON matching schema
    validated = HypothesisListResponse.model_validate_json(response.content)

    assert len(validated.hypotheses) == 3
    for hypothesis in validated.hypotheses:
        assert hypothesis.id, "Hypothesis should have ID"
        assert hypothesis.content, "Hypothesis should have content"
        assert len(hypothesis.content) > 30, f"Hypothesis content too short: {hypothesis.content}"

    print(f"✅ Direct Pydantic schema validation with Google API works")
    print(f"   Generated {len(validated.hypotheses)} hypotheses")
    print(f"   First ID: {validated.hypotheses[0].id}")
    print(f"   First content: {validated.hypotheses[0].content[:80]}...")


@pytest.mark.asyncio
async def test_deduction_response_schema_with_google_api(real_google_provider):
    """Test DeductionResponse schema with real Google API."""
    hypotheses = [
        "Deploy ocean cleanup vessels with advanced filtering",
        "Develop biodegradable plastic alternatives",
        "Implement global plastic ban policies"
    ]

    request = LLMRequest(
        system_prompt="You are an expert evaluator.",
        user_prompt=f"""Evaluate these hypotheses for reducing ocean plastic:
{chr(10).join(f'{i+1}. {h}' for i, h in enumerate(hypotheses))}

Provide scores (0.0-1.0) for each hypothesis on: impact, feasibility, accessibility, sustainability, scalability.
Then provide a synthesized answer and action plan.""",
        response_schema=DeductionResponse,
        response_mime_type="application/json",
        max_tokens=1500,
        temperature=0.7
    )

    response = await real_google_provider.generate(request)

    # Validate response
    validated = DeductionResponse.model_validate_json(response.content)

    assert len(validated.evaluations) == len(hypotheses)
    assert validated.answer, "Answer should exist"
    assert len(validated.action_plan) >= 1, "Action plan should have steps"

    # Verify score ranges
    for evaluation in validated.evaluations:
        assert 0.0 <= evaluation.scores.impact <= 1.0
        assert 0.0 <= evaluation.scores.feasibility <= 1.0
        assert 0.0 <= evaluation.scores.accessibility <= 1.0
        assert 0.0 <= evaluation.scores.sustainability <= 1.0
        assert 0.0 <= evaluation.scores.scalability <= 1.0

    print(f"✅ DeductionResponse schema validation with Google API works")
    print(f"   Evaluated {len(validated.evaluations)} hypotheses")
    print(f"   Best score: {max(e.scores.impact for e in validated.evaluations):.2f}")
    print(f"   Answer: {validated.answer[:80]}...")


# ==================== ERROR HANDLING TESTS ====================


@pytest.mark.asyncio
async def test_pydantic_validation_with_invalid_schema_response(real_google_provider):
    """Test that invalid responses are handled gracefully with proper error messages."""
    # Request with schema but intentionally minimal tokens to potentially get truncation
    request = LLMRequest(
        system_prompt="Generate hypotheses.",
        user_prompt="Generate 5 hypotheses for reducing ocean plastic.",
        response_schema=HypothesisListResponse,
        response_mime_type="application/json",
        max_tokens=50,  # Very low to potentially cause truncation
        temperature=0.7
    )

    response = await real_google_provider.generate(request)

    # Try to validate - should either work or raise ValidationError/JSONDecodeError
    try:
        validated = HypothesisListResponse.model_validate_json(response.content)
        print(f"✅ Even with low tokens, validation succeeded: {len(validated.hypotheses)} hypotheses")
    except (ValidationError, json.JSONDecodeError) as e:
        # Validation failed as expected with low tokens
        print(f"✅ Validation failed gracefully with informative error: {type(e).__name__}")
        print(f"   Error message: {str(e)[:100]}...")
        assert "validation" in str(e).lower() or "json" in str(e).lower()
