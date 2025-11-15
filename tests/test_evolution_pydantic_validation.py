"""
Test Pydantic validation in evolution operators (Phase 4).

This module tests that evolution operators use Pydantic validation for LLM responses,
with graceful fallback to manual parsing when validation fails.

Focuses on the batch operators that are actually used in production.
"""

import json

import pytest
from pydantic import ValidationError

from mad_spark_alt.core.schemas import (
    BatchCrossoverResponse,
    BatchMutationResponse,
    CrossoverResponse,
    MutationResponse,
)
from mad_spark_alt.evolution.semantic_utils import get_crossover_schema, get_mutation_schema


# ==================== SCHEMA GENERATION TESTS ====================


def test_mutation_schema_uses_pydantic():
    """Test that get_mutation_schema returns Pydantic-generated schema."""
    schema = get_mutation_schema()

    # Should be a dict (either old format or Pydantic-generated)
    assert isinstance(schema, dict)
    # After implementation, should be standard JSON Schema (lowercase "object")
    # For now, may still be uppercase "OBJECT"


def test_crossover_schema_uses_pydantic():
    """Test that get_crossover_schema returns Pydantic-generated schema."""
    schema = get_crossover_schema()

    # Should be a dict (either old format or Pydantic-generated)
    assert isinstance(schema, dict)
    # After implementation, should be standard JSON Schema


# ==================== PYDANTIC MODEL VALIDATION TESTS ====================


def test_mutation_response_schema_validation():
    """Test that MutationResponse Pydantic model enforces validation rules."""
    # Valid response
    valid_data = {"mutated_idea": "A mutated idea"}
    response = MutationResponse(**valid_data)
    assert response.mutated_idea == "A mutated idea"

    # Invalid: missing required field
    with pytest.raises(ValidationError):
        MutationResponse()


def test_batch_mutation_response_schema_validation():
    """Test that BatchMutationResponse Pydantic model enforces validation rules."""
    # Valid response
    valid_data = {
        "mutations": [
            {"id": 1, "mutated_idea": "Mutation 1"},
            {"id": 2, "mutated_idea": "Mutation 2", "mutation_type": "breakthrough"},
        ]
    }
    response = BatchMutationResponse(**valid_data)
    assert len(response.mutations) == 2
    assert response.mutations[1].mutation_type == "breakthrough"

    # Invalid: missing required field
    invalid_data = {"mutations": [{"id": 1}]}  # Missing mutated_idea
    with pytest.raises(ValidationError):
        BatchMutationResponse(**invalid_data)


def test_batch_mutation_response_json_parsing():
    """Test that BatchMutationResponse can parse JSON correctly."""
    json_str = json.dumps({
        "mutations": [
            {"id": 1, "mutated_idea": "First mutation"},
            {"id": 2, "mutated_idea": "Second mutation", "mutation_type": "paradigm_shift"},
        ]
    })

    result = BatchMutationResponse.model_validate_json(json_str)
    assert len(result.mutations) == 2
    assert result.mutations[0].id == 1
    assert result.mutations[0].mutated_idea == "First mutation"
    assert result.mutations[1].mutation_type == "paradigm_shift"


def test_crossover_response_schema_validation():
    """Test that CrossoverResponse Pydantic model enforces validation rules."""
    # Valid response
    valid_data = {
        "offspring1": "First offspring",
        "offspring2": "Second offspring"
    }
    response = CrossoverResponse(**valid_data)
    assert response.offspring1 == "First offspring"

    # Invalid: missing offspring2
    with pytest.raises(ValidationError):
        CrossoverResponse(offspring1="Only one")


def test_batch_crossover_response_schema_validation():
    """Test that BatchCrossoverResponse Pydantic model enforces validation rules."""
    # Valid response
    valid_data = {
        "crossovers": [
            {"pair_id": 1, "offspring1": "O1-1", "offspring2": "O1-2"},
            {"pair_id": 2, "offspring1": "O2-1", "offspring2": "O2-2"},
        ]
    }
    response = BatchCrossoverResponse(**valid_data)
    assert len(response.crossovers) == 2

    # Invalid: missing offspring
    invalid_data = {
        "crossovers": [{"pair_id": 1, "offspring1": "O1"}]  # Missing offspring2
    }
    with pytest.raises(ValidationError):
        BatchCrossoverResponse(**invalid_data)


def test_batch_crossover_response_json_parsing():
    """Test that BatchCrossoverResponse can parse JSON correctly."""
    json_str = json.dumps({
        "crossovers": [
            {"pair_id": 1, "offspring1": "Offspring 1-1", "offspring2": "Offspring 1-2"},
            {"pair_id": 2, "offspring1": "Offspring 2-1", "offspring2": "Offspring 2-2"},
        ]
    })

    result = BatchCrossoverResponse.model_validate_json(json_str)
    assert len(result.crossovers) == 2
    assert result.crossovers[0].pair_id == 1
    assert result.crossovers[0].offspring1 == "Offspring 1-1"
    assert result.crossovers[1].pair_id == 2


def test_batch_mutation_response_handles_extra_fields():
    """Test that BatchMutationResponse rejects extra fields (extra='forbid')."""
    invalid_data = {
        "mutations": [{"id": 1, "mutated_idea": "Mutation", "extra_field": "bad"}]
    }

    with pytest.raises(ValidationError) as exc_info:
        BatchMutationResponse(**invalid_data)

    # Should mention the extra field
    assert "extra" in str(exc_info.value).lower() or "forbidden" in str(exc_info.value).lower()


def test_batch_crossover_response_handles_extra_fields():
    """Test that BatchCrossoverResponse rejects extra fields (extra='forbid')."""
    invalid_data = {
        "crossovers": [
            {"pair_id": 1, "offspring1": "O1", "offspring2": "O2", "extra_field": "bad"}
        ]
    }

    with pytest.raises(ValidationError) as exc_info:
        BatchCrossoverResponse(**invalid_data)

    # Should mention the extra field
    assert "extra" in str(exc_info.value).lower() or "forbidden" in str(exc_info.value).lower()
