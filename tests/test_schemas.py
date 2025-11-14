"""
Tests for schemas.py - Universal Pydantic Schema Models

Tests comprehensive coverage of:
- JSON Schema generation and format validation
- Score range validation (0.0-1.0)
- Strict validation (extra fields rejected)
- Multi-provider compatibility
- JSON parsing with model_validate_json()
- Schema reusability and nesting
"""

import json
from typing import Any, Dict

import pytest
from pydantic import ValidationError


class TestHypothesisScores:
    """Test HypothesisScores schema and validation."""

    def test_generates_standard_json_schema(self):
        """Verify schema uses standard JSON Schema format, not Gemini-specific OpenAPI 3.0."""
        from mad_spark_alt.core.schemas import HypothesisScores

        schema = HypothesisScores.model_json_schema()

        # Standard JSON Schema uses lowercase "object", not Gemini's "OBJECT"
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "$defs" in schema or "properties" in schema  # Pydantic v2 format

        # Verify all five QADI criteria present
        properties = schema["properties"]
        assert "impact" in properties
        assert "feasibility" in properties
        assert "accessibility" in properties
        assert "sustainability" in properties
        assert "scalability" in properties

    def test_score_range_validation_enforced(self):
        """Verify scores must be between 0.0 and 1.0."""
        from mad_spark_alt.core.schemas import HypothesisScores

        # Valid: All scores in range
        valid_scores = HypothesisScores(
            impact=0.8,
            feasibility=0.6,
            accessibility=0.9,
            sustainability=0.7,
            scalability=0.5,
        )
        assert valid_scores.impact == 0.8

        # Invalid: Score > 1.0
        with pytest.raises(ValidationError) as exc_info:
            HypothesisScores(
                impact=1.5,  # Invalid
                feasibility=0.6,
                accessibility=0.9,
                sustainability=0.7,
                scalability=0.5,
            )
        assert "less than or equal to 1" in str(exc_info.value).lower()

        # Invalid: Score < 0.0
        with pytest.raises(ValidationError) as exc_info:
            HypothesisScores(
                impact=0.8,
                feasibility=-0.1,  # Invalid
                accessibility=0.9,
                sustainability=0.7,
                scalability=0.5,
            )
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_strict_validation_rejects_extra_fields(self):
        """Verify extra fields are rejected (additionalProperties: false)."""
        from mad_spark_alt.core.schemas import HypothesisScores

        with pytest.raises(ValidationError) as exc_info:
            HypothesisScores(
                impact=0.8,
                feasibility=0.6,
                accessibility=0.9,
                sustainability=0.7,
                scalability=0.5,
                extra_field="should_fail",  # Should be rejected
            )
        assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    def test_json_parsing_with_validation(self):
        """Test parsing JSON with model_validate_json()."""
        from mad_spark_alt.core.schemas import HypothesisScores

        json_str = json.dumps(
            {
                "impact": 0.8,
                "feasibility": 0.6,
                "accessibility": 0.9,
                "sustainability": 0.7,
                "scalability": 0.5,
            }
        )

        scores = HypothesisScores.model_validate_json(json_str)
        assert scores.impact == 0.8
        assert scores.scalability == 0.5

    def test_schema_has_constraints_metadata(self):
        """Verify JSON Schema includes minimum/maximum constraints."""
        from mad_spark_alt.core.schemas import HypothesisScores

        schema = HypothesisScores.model_json_schema()
        impact_schema = schema["properties"]["impact"]

        # Pydantic converts Field(ge=0.0, le=1.0) to minimum/maximum
        assert "minimum" in impact_schema or "exclusiveMinimum" in impact_schema
        assert "maximum" in impact_schema or "exclusiveMaximum" in impact_schema


class TestHypothesis:
    """Test Hypothesis schema."""

    def test_generates_standard_json_schema(self):
        """Verify Hypothesis uses standard JSON Schema format."""
        from mad_spark_alt.core.schemas import Hypothesis

        schema = Hypothesis.model_json_schema()
        assert schema["type"] == "object"
        assert "id" in schema["properties"]
        assert "content" in schema["properties"]

    def test_strict_validation_rejects_extra_fields(self):
        """Verify extra fields are rejected."""
        from mad_spark_alt.core.schemas import Hypothesis

        with pytest.raises(ValidationError):
            Hypothesis(id="H1", content="Test hypothesis", extra_field="invalid")

    def test_json_parsing(self):
        """Test JSON parsing with validation."""
        from mad_spark_alt.core.schemas import Hypothesis

        json_str = json.dumps({"id": "H1", "content": "Test hypothesis"})
        hypothesis = Hypothesis.model_validate_json(json_str)
        assert hypothesis.id == "H1"
        assert hypothesis.content == "Test hypothesis"


class TestHypothesisEvaluation:
    """Test HypothesisEvaluation schema with nested HypothesisScores."""

    def test_nested_schema_reusability(self):
        """Verify nested models create $ref or inline definitions."""
        from mad_spark_alt.core.schemas import HypothesisEvaluation

        schema = HypothesisEvaluation.model_json_schema()
        assert schema["type"] == "object"
        assert "hypothesis_id" in schema["properties"]
        assert "scores" in schema["properties"]

        # Pydantic should create a reference or inline the HypothesisScores schema
        scores_schema = schema["properties"]["scores"]
        assert "type" in scores_schema or "$ref" in scores_schema

    def test_nested_validation_works(self):
        """Verify validation works for nested scores."""
        from mad_spark_alt.core.schemas import HypothesisEvaluation, HypothesisScores

        # Valid nested structure
        evaluation = HypothesisEvaluation(
            hypothesis_id="H1",
            scores=HypothesisScores(
                impact=0.8,
                feasibility=0.6,
                accessibility=0.9,
                sustainability=0.7,
                scalability=0.5,
            ),
        )
        assert evaluation.hypothesis_id == "H1"
        assert evaluation.scores.impact == 0.8

        # Invalid: Nested score out of range
        with pytest.raises(ValidationError):
            HypothesisEvaluation(
                hypothesis_id="H1",
                scores=HypothesisScores(
                    impact=1.5,  # Invalid
                    feasibility=0.6,
                    accessibility=0.9,
                    sustainability=0.7,
                    scalability=0.5,
                ),
            )

    def test_json_parsing_nested(self):
        """Test JSON parsing with nested structures."""
        from mad_spark_alt.core.schemas import HypothesisEvaluation

        json_str = json.dumps(
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
        )

        evaluation = HypothesisEvaluation.model_validate_json(json_str)
        assert evaluation.hypothesis_id == "H1"
        assert evaluation.scores.impact == 0.8


class TestDeductionResponse:
    """Test DeductionResponse schema for Phase 2 (Deduction)."""

    def test_generates_standard_json_schema(self):
        """Verify DeductionResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import DeductionResponse

        schema = DeductionResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "evaluations" in schema["properties"]
        assert "answer" in schema["properties"]
        assert "action_plan" in schema["properties"]

    def test_property_ordering_preserved(self):
        """Verify property order matches model definition (for Gemini 2.5+)."""
        from mad_spark_alt.core.schemas import DeductionResponse

        schema = DeductionResponse.model_json_schema()
        properties_list = list(schema["properties"].keys())

        # Order should be: evaluations, answer, action_plan
        assert properties_list.index("evaluations") < properties_list.index("answer")
        assert properties_list.index("answer") < properties_list.index("action_plan")

    def test_complete_deduction_response_validation(self):
        """Test full deduction response parsing and validation."""
        from mad_spark_alt.core.schemas import DeductionResponse

        json_str = json.dumps(
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
                ],
                "answer": "Comprehensive analysis of both approaches...",
                "action_plan": [
                    "Step 1: Research existing solutions",
                    "Step 2: Prototype MVP",
                    "Step 3: Test with users",
                ],
            }
        )

        response = DeductionResponse.model_validate_json(json_str)
        assert len(response.evaluations) == 2
        assert response.evaluations[0].hypothesis_id == "H1"
        assert response.evaluations[0].scores.impact == 0.8
        assert response.answer.startswith("Comprehensive")
        assert len(response.action_plan) == 3

    def test_strict_validation_on_nested_levels(self):
        """Verify strict validation applies to all nested levels."""
        from mad_spark_alt.core.schemas import DeductionResponse

        # Extra field at top level
        with pytest.raises(ValidationError):
            json_str = json.dumps(
                {
                    "evaluations": [],
                    "answer": "Test",
                    "action_plan": [],
                    "extra_top_level": "invalid",
                }
            )
            DeductionResponse.model_validate_json(json_str)

        # Extra field in nested scores
        with pytest.raises(ValidationError):
            json_str = json.dumps(
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
                                "extra_score": 0.1,  # Invalid
                            },
                        }
                    ],
                    "answer": "Test",
                    "action_plan": [],
                }
            )
            DeductionResponse.model_validate_json(json_str)


class TestHypothesisListResponse:
    """Test HypothesisListResponse schema for Phase 1 (Abduction)."""

    def test_generates_standard_json_schema(self):
        """Verify HypothesisListResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import HypothesisListResponse

        schema = HypothesisListResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "hypotheses" in schema["properties"]

        # Array items should reference Hypothesis schema
        hypotheses_schema = schema["properties"]["hypotheses"]
        assert hypotheses_schema["type"] == "array"

    def test_hypothesis_list_validation(self):
        """Test hypothesis list parsing and validation."""
        from mad_spark_alt.core.schemas import HypothesisListResponse

        json_str = json.dumps(
            {
                "hypotheses": [
                    {"id": "H1", "content": "First hypothesis"},
                    {"id": "H2", "content": "Second hypothesis"},
                    {"id": "H3", "content": "Third hypothesis"},
                ]
            }
        )

        response = HypothesisListResponse.model_validate_json(json_str)
        assert len(response.hypotheses) == 3
        assert response.hypotheses[0].id == "H1"
        assert response.hypotheses[2].content == "Third hypothesis"

    def test_empty_list_allowed(self):
        """Verify empty hypothesis list is valid."""
        from mad_spark_alt.core.schemas import HypothesisListResponse

        json_str = json.dumps({"hypotheses": []})
        response = HypothesisListResponse.model_validate_json(json_str)
        assert len(response.hypotheses) == 0


class TestMutationResponse:
    """Test MutationResponse schema for evolution operators."""

    def test_generates_standard_json_schema(self):
        """Verify MutationResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import MutationResponse

        schema = MutationResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "mutated_idea" in schema["properties"]

    def test_mutation_response_validation(self):
        """Test mutation response parsing."""
        from mad_spark_alt.core.schemas import MutationResponse

        json_str = json.dumps({"mutated_idea": "Enhanced version of the original idea..."})

        response = MutationResponse.model_validate_json(json_str)
        assert response.mutated_idea.startswith("Enhanced")


class TestBatchMutationResponse:
    """Test BatchMutationResponse schema with mutation_type."""

    def test_generates_standard_json_schema(self):
        """Verify BatchMutationResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import BatchMutationResponse

        schema = BatchMutationResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "mutations" in schema["properties"]

    def test_batch_mutation_with_types_validation(self):
        """Test batch mutation parsing with mutation types."""
        from mad_spark_alt.core.schemas import BatchMutationResponse

        json_str = json.dumps(
            {
                "mutations": [
                    {
                        "id": 1,
                        "mutated_idea": "First mutation",
                        "mutation_type": "paradigm_shift",
                    },
                    {
                        "id": 2,
                        "mutated_idea": "Second mutation",
                        "mutation_type": "scale_amplification",
                    },
                ]
            }
        )

        response = BatchMutationResponse.model_validate_json(json_str)
        assert len(response.mutations) == 2
        assert response.mutations[0].id == 1
        assert response.mutations[0].mutation_type == "paradigm_shift"
        assert response.mutations[1].mutation_type == "scale_amplification"

    def test_mutation_type_optional(self):
        """Verify mutation_type is optional (for regular mutations)."""
        from mad_spark_alt.core.schemas import BatchMutationResponse

        json_str = json.dumps(
            {"mutations": [{"id": 1, "mutated_idea": "Regular mutation"}]}
        )

        response = BatchMutationResponse.model_validate_json(json_str)
        assert response.mutations[0].mutation_type is None


class TestCrossoverResponse:
    """Test CrossoverResponse schema for crossover operators."""

    def test_generates_standard_json_schema(self):
        """Verify CrossoverResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import CrossoverResponse

        schema = CrossoverResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "offspring1" in schema["properties"]
        assert "offspring2" in schema["properties"]

    def test_crossover_response_validation(self):
        """Test crossover response parsing."""
        from mad_spark_alt.core.schemas import CrossoverResponse

        json_str = json.dumps(
            {
                "offspring1": "First offspring combining ideas...",
                "offspring2": "Second offspring with different approach...",
            }
        )

        response = CrossoverResponse.model_validate_json(json_str)
        assert response.offspring1.startswith("First offspring")
        assert response.offspring2.startswith("Second offspring")


class TestBatchCrossoverResponse:
    """Test BatchCrossoverResponse schema with pair_id."""

    def test_generates_standard_json_schema(self):
        """Verify BatchCrossoverResponse uses standard JSON Schema."""
        from mad_spark_alt.core.schemas import BatchCrossoverResponse

        schema = BatchCrossoverResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "crossovers" in schema["properties"]

    def test_batch_crossover_with_pair_ids(self):
        """Test batch crossover parsing with pair IDs."""
        from mad_spark_alt.core.schemas import BatchCrossoverResponse

        json_str = json.dumps(
            {
                "crossovers": [
                    {
                        "pair_id": 1,
                        "offspring1": "First pair offspring 1",
                        "offspring2": "First pair offspring 2",
                    },
                    {
                        "pair_id": 2,
                        "offspring1": "Second pair offspring 1",
                        "offspring2": "Second pair offspring 2",
                    },
                ]
            }
        )

        response = BatchCrossoverResponse.model_validate_json(json_str)
        assert len(response.crossovers) == 2
        assert response.crossovers[0].pair_id == 1
        assert response.crossovers[1].pair_id == 2
        assert response.crossovers[0].offspring1.startswith("First pair")


class TestMultiProviderCompatibility:
    """Test schemas work with multiple LLM provider formats."""

    def test_schema_format_is_standard_not_gemini_specific(self):
        """Verify all schemas use standard JSON Schema, not Gemini OpenAPI 3.0."""
        from mad_spark_alt.core.schemas import (
            BatchCrossoverResponse,
            BatchMutationResponse,
            CrossoverResponse,
            DeductionResponse,
            Hypothesis,
            HypothesisEvaluation,
            HypothesisListResponse,
            HypothesisScores,
            MutationResponse,
        )

        schemas = [
            HypothesisScores,
            Hypothesis,
            HypothesisEvaluation,
            DeductionResponse,
            HypothesisListResponse,
            MutationResponse,
            BatchMutationResponse,
            CrossoverResponse,
            BatchCrossoverResponse,
        ]

        for schema_model in schemas:
            schema = schema_model.model_json_schema()

            # Standard JSON Schema uses lowercase "object", not "OBJECT"
            if "type" in schema:
                assert schema["type"] == "object", f"{schema_model.__name__} uses non-standard type"

            # Should have standard properties key
            assert "properties" in schema, f"{schema_model.__name__} missing properties"

    def test_schemas_serializable_for_api_transmission(self):
        """Verify schemas can be serialized to JSON for API calls."""
        from mad_spark_alt.core.schemas import DeductionResponse

        schema = DeductionResponse.model_json_schema()

        # Should be serializable to JSON string
        json_str = json.dumps(schema)
        assert len(json_str) > 0

        # Should be deserializable back
        parsed = json.loads(json_str)
        assert parsed["type"] == "object"
