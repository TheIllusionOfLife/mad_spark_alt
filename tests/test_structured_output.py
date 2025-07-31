"""
Tests for structured output functionality in LLM providers.

This module tests the new structured output capabilities using
Gemini's response_mime_type and response_schema features.
"""

import asyncio
import json
import pytest
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMRequest,
    LLMResponse,
    llm_manager,
)


class TestStructuredOutputSupport:
    """Test structured output support in LLM providers."""

    @pytest.fixture
    def google_provider(self):
        """Create a Google provider instance."""
        provider = GoogleProvider(api_key="test-key")
        yield provider

    def test_llm_request_accepts_response_schema(self):
        """Test that LLMRequest can accept response_schema parameter."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "age": {"type": "INTEGER"}
            },
            "required": ["name", "age"]
        }
        
        request = LLMRequest(
            user_prompt="Generate a person",
            response_schema=schema,
            response_mime_type="application/json"
        )
        
        assert request.response_schema == schema
        assert request.response_mime_type == "application/json"

    @pytest.mark.asyncio
    async def test_google_provider_uses_structured_output(self, google_provider):
        """Test that GoogleProvider correctly uses structured output parameters."""
        provider = google_provider
        
        # Mock the HTTP request
        with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock_request:
            # Mock response data
            mock_response_data = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": json.dumps({"name": "John", "age": 30})}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30
                }
            }
            mock_request.return_value = mock_response_data
            
            # Create request with schema
            schema = {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "age": {"type": "INTEGER"}
                },
                "required": ["name", "age"]
            }
            
            request = LLMRequest(
                user_prompt="Generate a person",
                response_schema=schema,
                response_mime_type="application/json"
            )
            
            # Execute
            response = await provider.generate(request)
            
            # Verify the request was made with correct parameters
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            
            # Check that generation config includes structured output params
            payload = call_kwargs["json"]
            generation_config = payload["generationConfig"]
            assert generation_config["responseMimeType"] == "application/json"
            assert generation_config["responseSchema"] == schema
            
            # Verify response
            assert response.content == json.dumps({"name": "John", "age": 30})
            assert json.loads(response.content) == {"name": "John", "age": 30}

    @pytest.mark.asyncio
    async def test_structured_output_fallback_for_non_json(self, google_provider):
        """Test that provider works normally when no schema is provided."""
        provider = google_provider
        
        # Mock the HTTP request
        with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock_request:
            # Mock normal text response
            mock_response_data = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "This is a normal text response"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30
                }
            }
            mock_request.return_value = mock_response_data
            
            # Create request without schema
            request = LLMRequest(user_prompt="Generate text")
            
            # Execute
            response = await provider.generate(request)
            
            # Verify the request was made with correct parameters
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            
            # Check that generation config doesn't include structured output params
            payload = call_kwargs["json"]
            generation_config = payload["generationConfig"]
            assert "responseMimeType" not in generation_config
            assert "responseSchema" not in generation_config
            
            # Verify response
            assert response.content == "This is a normal text response"


class TestHypothesisGenerationSchema:
    """Test schemas for hypothesis generation."""

    def test_hypothesis_schema_structure(self):
        """Test the structure of hypothesis generation schema."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "hypotheses": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "content": {"type": "STRING"},
                        },
                        "required": ["id", "content"]
                    }
                }
            },
            "required": ["hypotheses"]
        }
        
        # Verify schema structure
        assert schema["type"] == "OBJECT"
        assert "hypotheses" in schema["properties"]
        assert schema["properties"]["hypotheses"]["type"] == "ARRAY"
        
        # Verify item structure
        item_schema = schema["properties"]["hypotheses"]["items"]
        assert item_schema["type"] == "OBJECT"
        assert "id" in item_schema["properties"]
        assert "content" in item_schema["properties"]
        assert item_schema["required"] == ["id", "content"]

    @pytest.mark.asyncio
    async def test_parse_structured_hypothesis_response(self):
        """Test parsing of structured hypothesis response."""
        # Simulate structured response
        structured_response = {
            "hypotheses": [
                {"id": "H1", "content": "First hypothesis about solving the problem"},
                {"id": "H2", "content": "Second hypothesis with different approach"},
                {"id": "H3", "content": "Third hypothesis exploring alternatives"}
            ]
        }
        
        # Parse response
        hypotheses = structured_response["hypotheses"]
        
        # Verify
        assert len(hypotheses) == 3
        assert hypotheses[0]["id"] == "H1"
        assert hypotheses[0]["content"] == "First hypothesis about solving the problem"
        assert hypotheses[1]["id"] == "H2"
        assert hypotheses[2]["id"] == "H3"


class TestScoreParsingSchema:
    """Test schemas for score parsing in deduction phase."""

    def test_deduction_schema_structure(self):
        """Test the structure of deduction/scoring schema."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "evaluations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "hypothesis_id": {"type": "STRING"},
                            "scores": {
                                "type": "OBJECT",
                                "properties": {
                                    "impact": {"type": "NUMBER"},
                                    "feasibility": {"type": "NUMBER"},
                                    "accessibility": {"type": "NUMBER"},
                                    "sustainability": {"type": "NUMBER"},
                                    "scalability": {"type": "NUMBER"}
                                },
                                "required": ["impact", "feasibility", "accessibility", 
                                           "sustainability", "scalability"]
                            }
                        },
                        "required": ["hypothesis_id", "scores"]
                    }
                },
                "answer": {"type": "STRING"},
                "action_plan": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["evaluations", "answer", "action_plan"]
        }
        
        # Verify main structure
        assert schema["type"] == "OBJECT"
        assert all(key in schema["properties"] for key in ["evaluations", "answer", "action_plan"])
        
        # Verify evaluations structure
        eval_items = schema["properties"]["evaluations"]["items"]
        assert eval_items["type"] == "OBJECT"
        assert "hypothesis_id" in eval_items["properties"]
        assert "scores" in eval_items["properties"]
        
        # Verify scores structure
        scores_schema = eval_items["properties"]["scores"]
        assert scores_schema["type"] == "OBJECT"
        score_fields = ["impact", "feasibility", "accessibility", "sustainability", "scalability"]
        assert all(field in scores_schema["properties"] for field in score_fields)
        assert all(scores_schema["properties"][field]["type"] == "NUMBER" for field in score_fields)

    @pytest.mark.asyncio
    async def test_parse_structured_deduction_response(self):
        """Test parsing of structured deduction/scoring response."""
        # Simulate structured response
        structured_response = {
            "evaluations": [
                {
                    "hypothesis_id": "H1",
                    "scores": {
                        "impact": 0.8,
                        "feasibility": 0.7,
                        "accessibility": 0.9,
                        "sustainability": 0.6,
                        "scalability": 0.7
                    }
                },
                {
                    "hypothesis_id": "H2",
                    "scores": {
                        "impact": 0.9,
                        "feasibility": 0.6,
                        "accessibility": 0.8,
                        "sustainability": 0.8,
                        "scalability": 0.9
                    }
                }
            ],
            "answer": "Based on evaluation, H2 provides the best solution...",
            "action_plan": [
                "Implement the core strategy",
                "Test in controlled environment",
                "Gather feedback and iterate"
            ]
        }
        
        # Parse response
        evaluations = structured_response["evaluations"]
        answer = structured_response["answer"]
        action_plan = structured_response["action_plan"]
        
        # Verify evaluations
        assert len(evaluations) == 2
        assert evaluations[0]["hypothesis_id"] == "H1"
        assert evaluations[0]["scores"]["impact"] == 0.8
        assert evaluations[1]["hypothesis_id"] == "H2"
        assert evaluations[1]["scores"]["impact"] == 0.9
        
        # Verify answer and action plan
        assert "H2" in answer
        assert len(action_plan) == 3
        assert action_plan[0] == "Implement the core strategy"


class TestEvolutionOperatorSchemas:
    """Test schemas for evolution operators (mutation and crossover)."""

    def test_mutation_schema_structure(self):
        """Test the structure of mutation schema."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "mutations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "idea_id": {"type": "INTEGER"},
                            "mutated_content": {"type": "STRING"}
                        },
                        "required": ["idea_id", "mutated_content"]
                    }
                }
            },
            "required": ["mutations"]
        }
        
        # Verify structure
        assert schema["type"] == "OBJECT"
        assert "mutations" in schema["properties"]
        assert schema["properties"]["mutations"]["type"] == "ARRAY"
        
        # Verify item structure
        item_schema = schema["properties"]["mutations"]["items"]
        assert item_schema["properties"]["idea_id"]["type"] == "INTEGER"
        assert item_schema["properties"]["mutated_content"]["type"] == "STRING"

    def test_crossover_schema_structure(self):
        """Test the structure of crossover schema."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "offspring_1": {"type": "STRING"},
                "offspring_2": {"type": "STRING"}
            },
            "required": ["offspring_1", "offspring_2"]
        }
        
        # Verify structure
        assert schema["type"] == "OBJECT"
        assert "offspring_1" in schema["properties"]
        assert "offspring_2" in schema["properties"]
        assert schema["properties"]["offspring_1"]["type"] == "STRING"
        assert schema["properties"]["offspring_2"]["type"] == "STRING"

    @pytest.mark.asyncio
    async def test_parse_structured_mutation_response(self):
        """Test parsing of structured mutation response."""
        # Simulate structured response
        structured_response = {
            "mutations": [
                {
                    "idea_id": 1,
                    "mutated_content": "Enhanced version focusing on sustainability aspects..."
                },
                {
                    "idea_id": 2,
                    "mutated_content": "Alternative approach using distributed systems..."
                }
            ]
        }
        
        # Parse response
        mutations = structured_response["mutations"]
        
        # Verify
        assert len(mutations) == 2
        assert mutations[0]["idea_id"] == 1
        assert "sustainability" in mutations[0]["mutated_content"]
        assert mutations[1]["idea_id"] == 2
        assert "distributed" in mutations[1]["mutated_content"]

    @pytest.mark.asyncio
    async def test_parse_structured_crossover_response(self):
        """Test parsing of structured crossover response."""
        # Simulate structured response
        structured_response = {
            "offspring_1": "Combined approach integrating elements from both parents...",
            "offspring_2": "Alternative fusion emphasizing different aspects..."
        }
        
        # Parse response
        offspring_1 = structured_response["offspring_1"]
        offspring_2 = structured_response["offspring_2"]
        
        # Verify
        assert "Combined approach" in offspring_1
        assert "Alternative fusion" in offspring_2
        assert len(offspring_1) > 50  # Ensure substantial content
        assert len(offspring_2) > 50


@pytest.mark.integration
class TestStructuredOutputIntegration:
    """Integration tests for structured output with real LLM calls."""

    @pytest.mark.asyncio
    async def test_real_structured_hypothesis_generation(self):
        """Test real LLM call with structured output for hypothesis generation."""
        # This test requires GOOGLE_API_KEY to be set
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        provider = get_google_provider()
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "hypotheses": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "STRING"},
                            "content": {"type": "STRING"},
                        },
                        "required": ["id", "content"]
                    }
                }
            },
            "required": ["hypotheses"]
        }
        
        request = LLMRequest(
            user_prompt="Generate 3 hypotheses for reducing plastic waste in oceans",
            response_schema=schema,
            response_mime_type="application/json",
            max_tokens=500
        )
        
        response = await provider.generate(request)
        
        # Parse and verify response
        data = json.loads(response.content)
        assert "hypotheses" in data
        assert len(data["hypotheses"]) >= 1  # At least one hypothesis
        assert all("id" in h and "content" in h for h in data["hypotheses"])
        assert all(len(h["content"]) > 20 for h in data["hypotheses"])  # Non-trivial content

    @pytest.mark.asyncio
    async def test_real_structured_scoring(self):
        """Test real LLM call with structured output for scoring."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        provider = get_google_provider()
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "evaluations": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "hypothesis_id": {"type": "STRING"},
                            "scores": {
                                "type": "OBJECT",
                                "properties": {
                                    "impact": {"type": "NUMBER"},
                                    "feasibility": {"type": "NUMBER"},
                                    "accessibility": {"type": "NUMBER"},
                                    "sustainability": {"type": "NUMBER"},
                                    "scalability": {"type": "NUMBER"}
                                },
                                "required": ["impact", "feasibility", "accessibility", 
                                           "sustainability", "scalability"]
                            }
                        },
                        "required": ["hypothesis_id", "scores"]
                    }
                },
                "answer": {"type": "STRING"},
                "action_plan": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["evaluations", "answer", "action_plan"]
        }
        
        request = LLMRequest(
            user_prompt="""Evaluate these hypotheses for reducing plastic waste:
H1: Ban single-use plastics globally
H2: Develop biodegradable alternatives

Score each on impact, feasibility, accessibility, sustainability, and scalability (0-1).
Provide an answer recommending the best approach and an action plan.""",
            response_schema=schema,
            response_mime_type="application/json",
            max_tokens=800
        )
        
        response = await provider.generate(request)
        
        # Parse and verify response
        data = json.loads(response.content)
        assert "evaluations" in data
        assert len(data["evaluations"]) >= 1
        assert "answer" in data
        assert "action_plan" in data
        
        # Verify scores are in correct range
        for eval in data["evaluations"]:
            scores = eval["scores"]
            for score_type in ["impact", "feasibility", "accessibility", "sustainability", "scalability"]:
                assert 0 <= scores[score_type] <= 1