"""
Test Pydantic validation in QADI phase parsing (Phase 3b).

This module tests that QADI phases use Pydantic validation for LLM responses,
with graceful fallback to manual parsing when validation fails.

This is Phase 3b - actual validation in parsing logic (not just schema generation).
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.core.llm_provider import LLMResponse
from mad_spark_alt.core.phase_logic import (
    AbductionResult,
    DeductionResult,
    PhaseInput,
    execute_abduction_phase,
    execute_deduction_phase,
)
from mad_spark_alt.core.schemas import DeductionResponse, HypothesisListResponse


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with configurable responses."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    mock.generate_with_retry = AsyncMock()
    return mock


@pytest.fixture
def phase_input(mock_llm_manager):
    """Create a phase input for testing with mocked LLM manager."""
    return PhaseInput(
        user_input="How can we reduce ocean plastic?",
        llm_manager=mock_llm_manager,
        context={},
    )


@pytest.fixture
def sample_hypotheses():
    """Sample hypotheses for deduction testing."""
    return [
        "Deploy ocean cleanup vessels with advanced filtering",
        "Develop biodegradable plastic alternatives",
        "Implement global plastic ban policies",
    ]


@pytest.fixture
def valid_deduction_response_json():
    """Valid Pydantic-compatible deduction response JSON."""
    return json.dumps({
        "evaluations": [
            {
                "hypothesis_id": "H1",
                "scores": {
                    "impact": 0.8,
                    "feasibility": 0.7,
                    "accessibility": 0.6,
                    "sustainability": 0.75,
                    "scalability": 0.65,
                }
            },
            {
                "hypothesis_id": "H2",
                "scores": {
                    "impact": 0.6,
                    "feasibility": 0.8,
                    "accessibility": 0.7,
                    "sustainability": 0.65,
                    "scalability": 0.75,
                }
            },
            {
                "hypothesis_id": "H3",
                "scores": {
                    "impact": 0.9,
                    "feasibility": 0.5,
                    "accessibility": 0.8,
                    "sustainability": 0.7,
                    "scalability": 0.6,
                }
            },
        ],
        "answer": "This is the final answer.",
        "action_plan": ["Step 1", "Step 2", "Step 3"],
    })


@pytest.fixture
def invalid_deduction_response_scores_out_of_range():
    """Invalid deduction response with scores > 1.0 (should fail Pydantic validation)."""
    return json.dumps({
        "evaluations": [
            {
                "hypothesis_id": "H1",
                "scores": {
                    "impact": 1.5,  # Invalid: > 1.0
                    "feasibility": 0.7,
                    "accessibility": 0.6,
                    "sustainability": 0.75,
                    "scalability": 0.65,
                }
            },
            {
                "hypothesis_id": "H2",
                "scores": {
                    "impact": 0.6,
                    "feasibility": 0.8,
                    "accessibility": 0.7,
                    "sustainability": 0.65,
                    "scalability": 0.75,
                }
            },
            {
                "hypothesis_id": "H3",
                "scores": {
                    "impact": 0.9,
                    "feasibility": 0.5,
                    "accessibility": 0.8,
                    "sustainability": 0.7,
                    "scalability": 0.6,
                }
            },
        ],
        "answer": "Answer despite invalid scores",
        "action_plan": ["Step 1"],
    })


@pytest.fixture
def valid_hypothesis_list_response_json():
    """Valid Pydantic-compatible hypothesis list response JSON."""
    return json.dumps({
        "hypotheses": [
            {"id": "H1", "content": "First hypothesis content"},
            {"id": "H2", "content": "Second hypothesis content"},
            {"id": "H3", "content": "Third hypothesis content"},
        ]
    })


# ==================== DEDUCTION PHASE TESTS ====================


@pytest.mark.asyncio
async def test_deduction_valid_pydantic_response(
    phase_input, sample_hypotheses, valid_deduction_response_json, mock_llm_manager
):
    """Test that deduction phase successfully uses Pydantic validation with valid response."""
    # Mock LLM to return valid Pydantic-compatible JSON
    mock_response = LLMResponse(
        content=valid_deduction_response_json,
        raw_response="test",
        provider="google",
        model="test-model",
        input_tokens=100,
        output_tokens=200,
    )

    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic?",
        hypotheses=sample_hypotheses,
    )

    # Verify result
    assert isinstance(result, DeductionResult)
    assert len(result.hypothesis_scores) == 3

    # Verify scores were extracted correctly via Pydantic
    assert result.hypothesis_scores[0].impact == 0.8
    assert result.hypothesis_scores[0].feasibility == 0.7
    assert result.hypothesis_scores[1].impact == 0.6
    assert result.hypothesis_scores[2].impact == 0.9

    # Verify answer and action plan extracted
    assert result.answer == "This is the final answer."
    assert len(result.action_plan) == 3


@pytest.mark.asyncio
async def test_deduction_invalid_scores_validation(
    phase_input, sample_hypotheses, invalid_deduction_response_scores_out_of_range, mock_llm_manager
):
    """Test that deduction phase falls back to manual parsing when scores are invalid."""
    # Mock LLM to return JSON with invalid scores (> 1.0)
    mock_response = LLMResponse(
        content=invalid_deduction_response_scores_out_of_range,
        raw_response="test",
        provider="google",
        model="test-model",
        input_tokens=100,
        output_tokens=200,
    )

    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic?",
        hypotheses=sample_hypotheses,
    )

    # Should fall back to manual parsing and still produce results
    # Manual parsing clamps invalid scores or uses defaults
    assert isinstance(result, DeductionResult)
    assert len(result.hypothesis_scores) == 3


@pytest.mark.asyncio
async def test_deduction_fallback_to_manual_parsing(
    phase_input, sample_hypotheses, mock_llm_manager
):
    """Test that deduction phase falls back to manual parsing for non-JSON responses."""
    # Mock LLM to return non-JSON text response
    text_response = """
    Impact: 0.8 - High impact analysis
    Feasibility: 0.7 - Moderately feasible

    Final Answer: This is a text-based answer.
    """

    mock_response = LLMResponse(
        content=text_response,
        raw_response="test",
        provider="google",
        model="test-model",
        input_tokens=100,
        output_tokens=200,
    )

    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic?",
        hypotheses=sample_hypotheses,
    )

    # Should successfully fall back to text parsing
    assert isinstance(result, DeductionResult)
    assert len(result.hypothesis_scores) >= 1


# ==================== ABDUCTION PHASE TESTS ====================


@pytest.mark.asyncio
async def test_abduction_valid_pydantic_response(
    phase_input, valid_hypothesis_list_response_json, mock_llm_manager
):
    """Test that abduction phase successfully uses Pydantic validation with valid response."""
    mock_response = LLMResponse(
        content=valid_hypothesis_list_response_json,
        raw_response="test",
        provider="google",
        model="test-model",
        input_tokens=100,
        output_tokens=200,
    )

    mock_llm_manager.generate.return_value = mock_response

    result = await execute_abduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic?",
        num_hypotheses=3
    )

    # Verify result
    assert isinstance(result, AbductionResult)
    assert len(result.hypotheses) == 3
    assert result.hypotheses[0] == "First hypothesis content"
    assert result.hypotheses[1] == "Second hypothesis content"
    assert result.hypotheses[2] == "Third hypothesis content"


@pytest.mark.asyncio
async def test_abduction_fallback_to_parser(phase_input, mock_llm_manager):
    """Test that abduction phase falls back to HypothesisParser when Pydantic validation fails."""
    # Mock LLM to return text-based hypothesis list (old format)
    text_response = """
    1. First hypothesis in text format
    2. Second hypothesis in text format
    3. Third hypothesis in text format
    """

    mock_response = LLMResponse(
        content=text_response,
        raw_response="test",
        provider="google",
        model="test-model",
        input_tokens=100,
        output_tokens=200,
    )

    mock_llm_manager.generate.return_value = mock_response

    result = await execute_abduction_phase(
        phase_input,
        core_question="How can we reduce ocean plastic?",
        num_hypotheses=3
    )

    # Should fall back to HypothesisParser and still extract hypotheses
    assert isinstance(result, AbductionResult)
    assert len(result.hypotheses) >= 1  # Parser should extract at least some


# ==================== PYDANTIC MODEL VALIDATION TESTS ====================


def test_deduction_response_schema_validation():
    """Test that DeductionResponse Pydantic model enforces validation rules."""
    from pydantic import ValidationError

    # Valid response
    valid_data = {
        "evaluations": [
            {
                "hypothesis_id": "H1",
                "scores": {
                    "impact": 0.8,
                    "feasibility": 0.7,
                    "accessibility": 0.6,
                    "sustainability": 0.75,
                    "scalability": 0.65,
                }
            }
        ],
        "answer": "Test answer",
        "action_plan": ["Step 1", "Step 2", "Step 3"],
    }
    response = DeductionResponse(**valid_data)
    assert response.evaluations[0].scores.impact == 0.8

    # Invalid: score > 1.0
    invalid_score_high = {
        "evaluations": [
            {
                "hypothesis_id": "H1",
                "scores": {
                    "impact": 1.5,  # > 1.0
                    "feasibility": 0.7,
                    "accessibility": 0.6,
                    "sustainability": 0.75,
                    "scalability": 0.65,
                }
            }
        ],
        "answer": "Test",
        "action_plan": [],
    }
    with pytest.raises(ValidationError):
        DeductionResponse(**invalid_score_high)

    # Invalid: score < 0.0
    invalid_score_low = {
        "evaluations": [
            {
                "hypothesis_id": "H1",
                "scores": {
                    "impact": -0.1,  # < 0.0
                    "feasibility": 0.7,
                    "accessibility": 0.6,
                    "sustainability": 0.75,
                    "scalability": 0.65,
                }
            }
        ],
        "answer": "Test",
        "action_plan": [],
    }
    with pytest.raises(ValidationError):
        DeductionResponse(**invalid_score_low)


def test_hypothesis_list_response_schema_validation():
    """Test that HypothesisListResponse Pydantic model enforces validation rules."""
    from pydantic import ValidationError

    # Valid response
    valid_data = {
        "hypotheses": [
            {"id": "H1", "content": "Hypothesis 1"},
            {"id": "H2", "content": "Hypothesis 2"},
        ]
    }
    response = HypothesisListResponse(**valid_data)
    assert len(response.hypotheses) == 2
    assert response.hypotheses[0].content == "Hypothesis 1"

    # Invalid: missing required field
    missing_field = {"hypotheses": [{"id": "H1"}]}  # Missing content
    with pytest.raises(ValidationError):
        HypothesisListResponse(**missing_field)
