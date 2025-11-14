"""
Tests for LLM Provider Pydantic Schema Integration

Tests comprehensive coverage of:
- LLMRequest accepting Pydantic models as response_schema
- Automatic conversion of Pydantic models to JSON Schema
- Backward compatibility with dict schemas
- Multi-provider schema format generation
"""

import json
from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelSize,
)
from mad_spark_alt.core.schema_utils import is_pydantic_model, to_gemini_schema
from mad_spark_alt.core.schemas import DeductionResponse, HypothesisListResponse


class TestLLMRequestPydanticSupport:
    """Test LLMRequest accepts Pydantic models for response_schema."""

    def test_accepts_pydantic_model_as_response_schema(self):
        """Verify LLMRequest accepts Pydantic model classes."""
        request = LLMRequest(
            user_prompt="Test prompt",
            response_schema=DeductionResponse,  # Pydantic model class
            response_mime_type="application/json",
        )

        assert request.response_schema == DeductionResponse
        assert is_pydantic_model(request.response_schema)

    def test_accepts_dict_as_response_schema_backward_compat(self):
        """Verify backward compatibility with dict schemas."""
        dict_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

        request = LLMRequest(
            user_prompt="Test prompt",
            response_schema=dict_schema,
            response_mime_type="application/json",
        )

        assert request.response_schema == dict_schema
        assert isinstance(request.response_schema, dict)

    def test_accepts_none_for_unstructured_output(self):
        """Verify None is valid for unstructured outputs."""
        request = LLMRequest(user_prompt="Test prompt", response_schema=None)

        assert request.response_schema is None


class TestSchemaConversionInProvider:
    """Test GoogleProvider correctly converts Pydantic models to JSON Schema."""

    @pytest.mark.asyncio
    async def test_google_provider_converts_pydantic_to_json_schema(self):
        """Verify GoogleProvider converts Pydantic model to JSON Schema in API call."""
        provider = GoogleProvider(api_key="test_key")

        # Create response data
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "evaluations": [],
                                        "answer": "Test answer",
                                        "action_plan": [],
                                    }
                                )
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        # Create request with Pydantic model
        request = LLMRequest(
            user_prompt="Test prompt",
            response_schema=DeductionResponse,  # Pydantic model
            response_mime_type="application/json",
        )

        # Mock the safe_aiohttp_request to capture the payload
        with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock_request:
            mock_request.return_value = response_data
            await provider.generate(request)

            # Verify API was called
            assert mock_request.called

            # Extract the generation_config from the call
            call_args = mock_request.call_args
            payload = call_args[1]["json"]
            generation_config = payload["generationConfig"]

            # Verify responseJsonSchema contains standard JSON Schema
            assert "responseJsonSchema" in generation_config
            schema = generation_config["responseJsonSchema"]

            # Should be a dict (JSON Schema), not a Pydantic model
            assert isinstance(schema, dict)
            assert schema["type"] == "object"
            assert "evaluations" in schema["properties"]
            assert "answer" in schema["properties"]
            assert "action_plan" in schema["properties"]

    @pytest.mark.asyncio
    async def test_google_provider_passes_through_dict_schema(self):
        """Verify GoogleProvider passes through dict schemas unchanged."""
        provider = GoogleProvider(api_key="test_key")

        dict_schema = {
            "type": "object",
            "properties": {"custom": {"type": "string"}},
        }

        response_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"custom": "value"}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        request = LLMRequest(
            user_prompt="Test prompt",
            response_schema=dict_schema,
            response_mime_type="application/json",
        )

        with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock_request:
            mock_request.return_value = response_data
            await provider.generate(request)

            call_args = mock_request.call_args
            payload = call_args[1]["json"]
            generation_config = payload["generationConfig"]

            # Dict schema should be passed through unchanged
            assert generation_config["responseJsonSchema"] == dict_schema


class TestSchemaUtilityIntegration:
    """Test schema_utils functions integrate correctly."""

    def test_to_gemini_schema_produces_valid_json_schema(self):
        """Verify to_gemini_schema produces standard JSON Schema."""
        schema = to_gemini_schema(HypothesisListResponse)

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "hypotheses" in schema["properties"]

    def test_is_pydantic_model_correctly_identifies_models(self):
        """Verify is_pydantic_model helper works correctly."""
        assert is_pydantic_model(DeductionResponse) is True
        assert is_pydantic_model(HypothesisListResponse) is True
        assert is_pydantic_model({"type": "object"}) is False
        assert is_pydantic_model(None) is False
        assert is_pydantic_model("string") is False


class TestBackwardCompatibility:
    """Test that Pydantic integration doesn't break existing code."""

    @pytest.mark.asyncio
    async def test_existing_dict_schemas_still_work(self):
        """Verify all existing code using dict schemas continues to work."""
        provider = GoogleProvider(api_key="test_key")

        # Old-style dict schema (as currently used in codebase)
        old_schema = {
            "type": "OBJECT",  # Old Gemini OpenAPI 3.0 format
            "properties": {
                "hypotheses": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "content": {"type": "STRING"},
                        },
                    },
                }
            },
        }

        request = LLMRequest(
            user_prompt="Test",
            response_schema=old_schema,
            response_mime_type="application/json",
        )

        response_data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"hypotheses": []}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150,
            },
        }

        with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock_request:
            mock_request.return_value = response_data
            response = await provider.generate(request)

            # Should succeed with old schema format
            assert response.content is not None

    def test_none_schema_for_unstructured_output_works(self):
        """Verify requests without response_schema work (unstructured output)."""
        request = LLMRequest(
            user_prompt="Tell me a story", response_schema=None  # No schema
        )

        assert request.response_schema is None
        # Should not raise any errors


class TestMultiProviderSchemaFormatting:
    """Test schema formatting works for multiple providers."""

    def test_schemas_generate_standard_json_schema_not_provider_specific(self):
        """Verify schemas use standard JSON Schema, not Gemini-specific format."""
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

        all_schemas = [
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

        for schema_model in all_schemas:
            schema = to_gemini_schema(schema_model)

            # Standard JSON Schema uses lowercase "object", not "OBJECT"
            assert schema["type"] == "object", f"{schema_model.__name__} uses non-standard type"

            # Standard JSON Schema uses lowercase primitives
            for prop_name, prop_schema in schema.get("properties", {}).items():
                if "type" in prop_schema:
                    prop_type = prop_schema["type"]
                    # Should be lowercase: "string", "number", "array", "object"
                    # NOT uppercase: "STRING", "NUMBER", "ARRAY", "OBJECT"
                    assert prop_type == prop_type.lower(), (
                        f"{schema_model.__name__}.{prop_name} "
                        f"uses non-standard type '{prop_type}'"
                    )
