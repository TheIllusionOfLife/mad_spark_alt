"""
Tests for phase_logic.py Pydantic Schema Integration

Tests comprehensive coverage of:
- Pydantic schema generation for QADI phases
- Automatic validation with Pydantic models
- Score range validation (0.0-1.0)
- Strict validation (extra fields rejected)
- Backward compatibility with existing parsing
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from mad_spark_alt.core.llm_provider import LLMProvider, LLMResponse
from mad_spark_alt.core.phase_logic import (
    DeductionResult,
    PhaseInput,
    execute_deduction_phase,
    get_deduction_schema,
    get_hypothesis_generation_schema,
)
from mad_spark_alt.core.schemas import DeductionResponse, HypothesisListResponse


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with configurable responses."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def phase_input(mock_llm_manager):
    """Standard phase input for testing."""
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


class TestHypothesisGenerationSchema:
    """Test hypothesis generation schema with Pydantic models."""

    def test_schema_returns_pydantic_model(self):
        """Verify get_hypothesis_generation_schema returns compatible format."""
        schema = get_hypothesis_generation_schema()

        # Should be dict (can be either old format or Pydantic-generated)
        assert isinstance(schema, dict)
        assert "properties" in schema or "type" in schema

    def test_pydantic_schema_generation(self):
        """Verify HypothesisListResponse generates valid schema."""
        schema = HypothesisListResponse.model_json_schema()

        assert schema["type"] == "object"
        assert "hypotheses" in schema["properties"]
        assert schema["properties"]["hypotheses"]["type"] == "array"

    def test_pydantic_hypothesis_validation(self):
        """Test Pydantic validation for hypothesis list."""
        # Valid hypothesis list
        valid_json = json.dumps(
            {
                "hypotheses": [
                    {"id": "H1", "content": "First hypothesis"},
                    {"id": "H2", "content": "Second hypothesis"},
                ]
            }
        )

        result = HypothesisListResponse.model_validate_json(valid_json)
        assert len(result.hypotheses) == 2
        assert result.hypotheses[0].id == "H1"

    def test_pydantic_rejects_extra_fields_in_hypotheses(self):
        """Test strict validation rejects extra fields."""
        # Extra field "priority" should be rejected
        invalid_json = json.dumps(
            {
                "hypotheses": [
                    {"id": "H1", "content": "Test", "priority": "high"}  # Extra field
                ]
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            HypothesisListResponse.model_validate_json(invalid_json)

        assert "extra" in str(exc_info.value).lower() or "unexpected" in str(
            exc_info.value
        ).lower()

    def test_pydantic_requires_all_hypothesis_fields(self):
        """Test Pydantic requires id and content fields."""
        # Missing "content" field
        invalid_json = json.dumps({"hypotheses": [{"id": "H1"}]})

        with pytest.raises(ValidationError) as exc_info:
            HypothesisListResponse.model_validate_json(invalid_json)

        assert "content" in str(exc_info.value).lower()


class TestDeductionSchema:
    """Test deduction schema with Pydantic models."""

    def test_schema_returns_dict(self):
        """Verify get_deduction_schema returns dict format."""
        schema = get_deduction_schema()

        assert isinstance(schema, dict)
        assert "properties" in schema or "type" in schema

    def test_pydantic_schema_generation(self):
        """Verify DeductionResponse generates valid schema."""
        schema = DeductionResponse.model_json_schema()

        assert schema["type"] == "object"
        assert "evaluations" in schema["properties"]
        assert "answer" in schema["properties"]
        assert "action_plan" in schema["properties"]

    def test_pydantic_property_ordering(self):
        """Verify property order: evaluations → answer → action_plan."""
        schema = DeductionResponse.model_json_schema()
        properties = list(schema["properties"].keys())

        # Pydantic v2 preserves field order
        assert properties.index("evaluations") < properties.index("answer")
        assert properties.index("answer") < properties.index("action_plan")

    def test_pydantic_score_range_validation(self):
        """Test score validation enforces 0.0-1.0 range."""
        # Valid scores
        valid_json = json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.6,
                            "accessibility": 0.9,
                            "sustainability": 0.7,
                            "scalability": 0.5,
                        },
                    }
                ],
                "answer": "Test answer",
                "action_plan": ["Step 1", "Step 2", "Step 3"],
            }
        )

        result = DeductionResponse.model_validate_json(valid_json)
        assert result.evaluations[0].scores.impact == 0.8

        # Invalid: score > 1.0
        invalid_json = json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 1.5,  # Invalid
                            "feasibility": 0.6,
                            "accessibility": 0.9,
                            "sustainability": 0.7,
                            "scalability": 0.5,
                        },
                    }
                ],
                "answer": "Test",
                "action_plan": [],
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            DeductionResponse.model_validate_json(invalid_json)

        assert "less than or equal to 1" in str(exc_info.value).lower()

        # Invalid: score < 0.0
        invalid_json2 = json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": -0.1,  # Invalid
                            "accessibility": 0.9,
                            "sustainability": 0.7,
                            "scalability": 0.5,
                        },
                    }
                ],
                "answer": "Test",
                "action_plan": [],
            }
        )

        with pytest.raises(ValidationError):
            DeductionResponse.model_validate_json(invalid_json2)

    def test_pydantic_rejects_extra_fields_in_scores(self):
        """Test strict validation rejects extra score fields."""
        invalid_json = json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.6,
                            "accessibility": 0.9,
                            "sustainability": 0.7,
                            "scalability": 0.5,
                            "novelty": 0.9,  # Extra field - should be rejected
                        },
                    }
                ],
                "answer": "Test",
                "action_plan": [],
            }
        )

        with pytest.raises(ValidationError):
            DeductionResponse.model_validate_json(invalid_json)

    def test_pydantic_requires_all_score_fields(self):
        """Test Pydantic requires all 5 QADI score fields."""
        # Missing "scalability" field
        invalid_json = json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.6,
                            "accessibility": 0.9,
                            "sustainability": 0.7,
                            # Missing scalability
                        },
                    }
                ],
                "answer": "Test",
                "action_plan": [],
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            DeductionResponse.model_validate_json(invalid_json)

        assert "scalability" in str(exc_info.value).lower()


class TestDeductionPhaseWithPydantic:
    """Test deduction phase execution with Pydantic validation."""

    @pytest.mark.asyncio
    async def test_deduction_with_valid_pydantic_response(
        self, mock_llm_manager, phase_input, sample_hypotheses
    ):
        """Test deduction phase parses valid Pydantic-compatible response."""
        # Valid structured response matching DeductionResponse schema
        mock_response = LLMResponse(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "hypothesis_id": "H1",
                            "scores": {
                                "impact": 0.8,
                                "feasibility": 0.7,
                                "accessibility": 0.9,
                                "sustainability": 0.6,
                                "scalability": 0.8,
                            },
                        },
                        {
                            "hypothesis_id": "H2",
                            "scores": {
                                "impact": 0.7,
                                "feasibility": 0.8,
                                "accessibility": 0.6,
                                "sustainability": 0.9,
                                "scalability": 0.7,
                            },
                        },
                        {
                            "hypothesis_id": "H3",
                            "scores": {
                                "impact": 0.6,
                                "feasibility": 0.6,
                                "accessibility": 0.7,
                                "sustainability": 0.8,
                                "scalability": 0.6,
                            },
                        },
                    ],
                    "answer": "Ocean cleanup vessels provide the best combination of impact and feasibility.",
                    "action_plan": [
                        "Research existing vessel technologies",
                        "Prototype advanced filtration system",
                        "Test in controlled environment",
                    ],
                }
            ),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            cost=0.005,
        )
        mock_llm_manager.generate.return_value = mock_response

        result = await execute_deduction_phase(
            phase_input, "How can we reduce ocean plastic?", sample_hypotheses
        )

        assert isinstance(result, DeductionResult)
        assert len(result.hypothesis_scores) == 3
        assert result.hypothesis_scores[0].impact == 0.8
        assert result.hypothesis_scores[1].feasibility == 0.8
        assert "Ocean cleanup" in result.answer
        assert len(result.action_plan) == 3

    @pytest.mark.asyncio
    async def test_deduction_falls_back_on_pydantic_validation_error(
        self, mock_llm_manager, phase_input, sample_hypotheses
    ):
        """Test deduction falls back to text parsing when response is not JSON."""
        # Non-JSON response triggers fallback to text parsing
        mock_response = LLMResponse(
            content="""
Evaluation Results:

Hypothesis 1: Impact: 0.8, Feasibility: 0.7, Accessibility: 0.9, Sustainability: 0.6, Scalability: 0.8

Hypothesis 2: Impact: 0.7, Feasibility: 0.8, Accessibility: 0.6, Sustainability: 0.9, Scalability: 0.7

Hypothesis 3: Impact: 0.6, Feasibility: 0.6, Accessibility: 0.7, Sustainability: 0.8, Scalability: 0.6

Answer: Ocean cleanup vessels are the most viable solution.

Action Plan:
1. Research existing technologies
2. Build prototype
3. Test at scale
            """,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            cost=0.004,
        )
        mock_llm_manager.generate.return_value = mock_response

        # Should fall back to text parsing and still work
        result = await execute_deduction_phase(
            phase_input, "How can we reduce ocean plastic?", sample_hypotheses
        )

        assert isinstance(result, DeductionResult)
        assert len(result.hypothesis_scores) == 3
        # Text parsing should extract scores
        assert result.hypothesis_scores[0].impact > 0


class TestBackwardCompatibility:
    """Test that Pydantic migration doesn't break existing behavior."""

    @pytest.mark.asyncio
    async def test_existing_json_parsing_still_works(
        self, mock_llm_manager, phase_input, sample_hypotheses
    ):
        """Verify existing JSON response format continues to work."""
        # Old-style response (what current code produces)
        mock_response = LLMResponse(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "hypothesis_id": "H1",
                            "scores": {
                                "impact": 0.8,
                                "feasibility": 0.7,
                                "accessibility": 0.9,
                                "sustainability": 0.6,
                                "scalability": 0.8,
                            },
                        }
                    ],
                    "answer": "Test answer",
                    "action_plan": ["Step 1", "Step 2"],
                }
            ),
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.003,
        )
        mock_llm_manager.generate.return_value = mock_response

        result = await execute_deduction_phase(
            phase_input, "Test question?", sample_hypotheses[:1]
        )

        # Should work exactly as before
        assert len(result.hypothesis_scores) >= 1
        assert result.answer == "Test answer"
        assert len(result.action_plan) == 2

    @pytest.mark.asyncio
    async def test_text_parsing_fallback_still_works(
        self, mock_llm_manager, phase_input, sample_hypotheses
    ):
        """Verify text parsing fallback continues to work."""
        # Non-JSON response (triggers text parsing)
        mock_response = LLMResponse(
            content="Impact: 0.8\nFeasibility: 0.7\nAccessibility: 0.9",
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.002,
        )
        mock_llm_manager.generate.return_value = mock_response

        result = await execute_deduction_phase(
            phase_input, "Test question?", sample_hypotheses[:1]
        )

        # Should fall back to text parsing
        assert len(result.hypothesis_scores) >= 1
