"""
Integration tests for structured output implementation.

These tests verify that LLM responses using structured output schemas are properly defined
and that evolution operators work correctly with structured output.

Note: Tests for orchestrator phase logic have been moved to test_phase_logic.py and
test_phase_integration.py after the refactoring in PR #111.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock

from mad_spark_alt.core.llm_provider import LLMResponse
from mad_spark_alt.core.phase_logic import (
    get_hypothesis_generation_schema,
    get_deduction_schema,
)
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
    get_mutation_schema,
    get_crossover_schema,
)
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


class TestStructuredOutputSchemas:
    """Test that all schemas are properly defined and valid."""

    def test_hypothesis_generation_schema_structure(self):
        """Test hypothesis generation schema has correct structure."""
        schema = get_hypothesis_generation_schema()

        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "hypotheses" in schema["properties"]

        hypotheses_schema = schema["properties"]["hypotheses"]
        assert hypotheses_schema["type"] == "ARRAY"
        assert "items" in hypotheses_schema

        item_schema = hypotheses_schema["items"]
        assert item_schema["type"] == "OBJECT"
        assert "id" in item_schema["properties"]
        assert "content" in item_schema["properties"]

    def test_deduction_schema_structure(self):
        """Test deduction (score evaluation) schema has correct structure."""
        schema = get_deduction_schema()

        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "evaluations" in schema["properties"]

        evaluations_schema = schema["properties"]["evaluations"]
        assert evaluations_schema["type"] == "ARRAY"

        # Check that schema includes all required fields
        item_schema = evaluations_schema["items"]
        assert "hypothesis_id" in item_schema["properties"]
        assert "scores" in item_schema["properties"]

        # Check score fields are in the nested scores object
        scores_schema = item_schema["properties"]["scores"]
        score_fields = ["impact", "feasibility", "accessibility",
                       "sustainability", "scalability"]

        for field in score_fields:
            assert field in scores_schema["properties"]

    def test_mutation_schema_structure(self):
        """Test mutation schema has correct structure."""
        schema = get_mutation_schema()

        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "mutations" in schema["properties"]

        mutations_schema = schema["properties"]["mutations"]
        assert mutations_schema["type"] == "ARRAY"

        item_schema = mutations_schema["items"]
        assert "id" in item_schema["properties"]
        assert item_schema["properties"]["id"]["type"] == "INTEGER"
        assert "content" in item_schema["properties"]

    def test_crossover_schema_structure(self):
        """Test crossover schema has correct structure."""
        schema = get_crossover_schema()

        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "offspring_1" in schema["properties"]
        assert "offspring_2" in schema["properties"]


class TestStructuredOutputParsing:
    """Test parsing of structured output responses for evolution operators."""

    @pytest.mark.asyncio
    async def test_mutation_with_valid_json(self):
        """Test mutation with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "mutations": [
                    {
                        "id": 1,
                        "content": "Enhanced version of the idea"
                    }
                ]
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )

        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value=mock_response)

        operator = BatchSemanticMutationOperator(mock_llm)
        idea = GeneratedIdea(
            content="Original idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )

        # BatchSemanticMutationOperator expects a list
        mutated_list = await operator.mutate_batch([idea], "reduce waste")
        mutated = mutated_list[0]
        assert mutated.content == "Enhanced version of the idea"

    @pytest.mark.asyncio
    async def test_crossover_with_valid_json(self):
        """Test crossover with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "offspring_1": "First combined idea focusing on technical implementation with advanced algorithms",
                "offspring_2": "Alternative approach emphasizing social collaboration and community engagement"
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )

        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value=mock_response)

        operator = SemanticCrossoverOperator(mock_llm)
        parent1 = GeneratedIdea(
            content="Parent 1 idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )
        parent2 = GeneratedIdea(
            content="Parent 2 idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )

        offspring = await operator.crossover(parent1, parent2, "reduce waste")
        assert len(offspring) == 2
        assert offspring[0].content == "First combined idea focusing on technical implementation with advanced algorithms"
        assert offspring[1].content == "Alternative approach emphasizing social collaboration and community engagement"
