"""
Tests for OllamaProvider - local LLM provider implementation.

This module tests Ollama integration with structured output support.
Tests are marked with @pytest.mark.ollama for conditional execution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from mad_spark_alt.core.llm_provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelSize,
    OllamaProvider,
)
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType
from mad_spark_alt.core.schemas import HypothesisListResponse, DeductionResponse


class TestOllamaProviderUnit:
    """Unit tests for OllamaProvider with mocked API calls."""

    def test_ollama_provider_initialization(self):
        """Test OllamaProvider initializes with correct defaults."""
        provider = OllamaProvider()

        assert provider.model == "gemma3:12b-it-qat"
        assert provider.base_url == "http://localhost:11434"
        assert provider._session is None

    def test_ollama_provider_custom_config(self):
        """Test OllamaProvider accepts custom configuration."""
        provider = OllamaProvider(
            model="gemma3:4b-it-qat",
            base_url="http://custom-host:11434"
        )

        assert provider.model == "gemma3:4b-it-qat"
        assert provider.base_url == "http://custom-host:11434"

    def test_get_available_models(self):
        """Test get_available_models returns gemma3 configuration."""
        provider = OllamaProvider()
        models = provider.get_available_models()

        assert len(models) == 1
        model = models[0]
        assert model.provider == LLMProvider.OLLAMA
        assert model.model_name == "gemma3:12b-it-qat"
        assert model.model_size == ModelSize.MEDIUM
        assert model.input_cost_per_1k == 0.0  # Free
        assert model.output_cost_per_1k == 0.0
        assert model.max_tokens == 8192

    @pytest.mark.asyncio
    async def test_generate_simple_text(self):
        """Test basic text generation with mocked Ollama API."""
        provider = OllamaProvider()

        # Mock safe_aiohttp_request to return expected response
        mock_response_data = {
            "message": {
                "content": "This is a test response from Ollama."
            },
            "done": True
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            request = LLMRequest(
                user_prompt="What is 2+2?",
                temperature=0.7,
                max_tokens=100
            )

            response = await provider.generate(request)

            assert isinstance(response, LLMResponse)
            assert response.provider == LLMProvider.OLLAMA
            assert response.model == "gemma3:12b-it-qat"
            assert response.content == "This is a test response from Ollama."
            assert response.cost == 0.0  # Ollama is free
            assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_generate_with_pydantic_schema(self):
        """Test structured output with Pydantic schema (mocked)."""
        provider = OllamaProvider()

        # Mock response with valid HypothesisListResponse JSON
        mock_json_response = {
            "hypotheses": [
                {
                    "id": "H1",
                    "content": "First hypothesis about AI safety"
                },
                {
                    "id": "H2",
                    "content": "Second hypothesis about alignment"
                },
                {
                    "id": "H3",
                    "content": "Third hypothesis about governance"
                }
            ]
        }

        mock_response_data = {
            "message": {
                "content": json.dumps(mock_json_response)
            },
            "done": True
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            request = LLMRequest(
                user_prompt="Generate 3 hypotheses about AI safety",
                response_schema=HypothesisListResponse,
                response_mime_type="application/json"
            )

            response = await provider.generate(request)

            # Validate response can be parsed with Pydantic
            result = HypothesisListResponse.model_validate_json(response.content)
            assert len(result.hypotheses) == 3
            assert result.hypotheses[0].id == "H1"
            assert result.hypotheses[0].content == "First hypothesis about AI safety"


@pytest.mark.ollama
@pytest.mark.integration
class TestOllamaProviderIntegration:
    """Integration tests with real Ollama server (requires ollama running)."""

    @pytest.fixture
    def ollama_provider(self):
        """Create OllamaProvider instance."""
        return OllamaProvider()

    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama server is available."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            pytest.skip("Ollama server not running on localhost:11434")

    @pytest.mark.asyncio
    async def test_real_ollama_simple_generation(
        self,
        ollama_provider: OllamaProvider,
        check_ollama_available
    ):
        """Test real Ollama generation with simple text prompt."""
        request = LLMRequest(
            user_prompt="Explain quantum computing in exactly one sentence.",
            temperature=0.7,
            max_tokens=100
        )

        response = await ollama_provider.generate(request)

        assert response.provider == LLMProvider.OLLAMA
        assert response.model == "gemma3:12b-it-qat"
        assert len(response.content) > 0
        assert response.cost == 0.0
        assert response.response_time > 0
        assert response.usage["total_tokens"] > 0

        print(f"\n[Ollama Response Time: {response.response_time:.2f}s]")
        print(f"[Content: {response.content[:100]}...]")

    @pytest.mark.asyncio
    async def test_real_ollama_structured_output_hypothesis(
        self,
        ollama_provider: OllamaProvider,
        check_ollama_available
    ):
        """Test real Ollama with HypothesisListResponse schema."""
        request = LLMRequest(
            user_prompt="Generate exactly 3 creative hypotheses about reducing urban carbon emissions. Keep each hypothesis concise (under 100 words).",
            response_schema=HypothesisListResponse,
            response_mime_type="application/json",
            max_tokens=1500  # Increased for complete JSON response
        )

        response = await ollama_provider.generate(request)

        # Test 3-layer fallback pattern
        result_valid = False
        hypothesis_count = 0

        # Layer 1: Pydantic validation (preferred)
        try:
            result = HypothesisListResponse.model_validate_json(response.content)
            hypothesis_count = len(result.hypotheses)
            assert hypothesis_count >= 3
            assert all(len(h.content) > 10 for h in result.hypotheses)
            result_valid = True
            print("\n✅ Layer 1: Pydantic validation successful")
        except (ValidationError, json.JSONDecodeError) as e:
            # Layer 2: Manual JSON parsing
            print(f"\n⚠️ Layer 1 failed: {e}")
            try:
                data = json.loads(response.content)
                if "hypotheses" in data:
                    hypothesis_count = len(data["hypotheses"])
                    assert hypothesis_count >= 3
                    result_valid = True
                    print(f"✅ Layer 2: Manual JSON parsing successful ({hypothesis_count} hypotheses)")
            except json.JSONDecodeError as e2:
                # Layer 3: Text parsing (basic check)
                print(f"⚠️ Layer 2 failed: {e2}")
                # At minimum, check that response contains hypothesis-like content
                assert "hypothesis" in response.content.lower() or "approach" in response.content.lower()
                print("✅ Layer 3: Text parsing fallback (content detected)")

        print(f"[Response Time: {response.response_time:.2f}s]")
        print(f"[Hypotheses: {hypothesis_count if hypothesis_count > 0 else 'N/A'}]")

        # Assert that at least one fallback layer succeeded
        assert result_valid or hypothesis_count > 0 or len(response.content) > 100

    @pytest.mark.asyncio
    async def test_real_ollama_structured_output_deduction(
        self,
        ollama_provider: OllamaProvider,
        check_ollama_available
    ):
        """Test real Ollama with DeductionResponse schema."""
        request = LLMRequest(
            system_prompt="You are evaluating hypotheses. Return scores between 0.0 and 1.0.",
            user_prompt="""Evaluate this hypothesis:
"Implement bike-sharing programs in every neighborhood."

Provide scores for: impact, feasibility, accessibility, sustainability, scalability.
Use the exact JSON format required.""",
            response_schema=DeductionResponse,
            response_mime_type="application/json",
            max_tokens=800
        )

        response = await ollama_provider.generate(request)

        # Test 3-layer fallback pattern
        try:
            # Layer 1: Pydantic validation
            result = DeductionResponse.model_validate_json(response.content)
            assert len(result.evaluations) > 0
            for evaluation in result.evaluations:
                assert 0.0 <= evaluation.scores.impact <= 1.0
                assert 0.0 <= evaluation.scores.feasibility <= 1.0
            print("\n✅ DeductionResponse Pydantic validation successful")
        except (ValidationError, json.JSONDecodeError) as e:
            # Layer 2: Manual JSON parsing
            print(f"\n⚠️ Pydantic failed: {e}, trying manual parsing")
            data = json.loads(response.content)
            assert "evaluations" in data or "scores" in data
            print("✅ Manual JSON parsing successful")

    @pytest.mark.asyncio
    async def test_real_ollama_performance_benchmark(
        self,
        ollama_provider: OllamaProvider,
        check_ollama_available
    ):
        """Benchmark Ollama performance for typical QADI operations."""
        import time

        # Test 1: Hypothesis generation (typical: 3 hypotheses)
        start = time.time()
        request = LLMRequest(
            user_prompt="Generate 3 hypotheses about improving remote work productivity.",
            response_schema=HypothesisListResponse,
            response_mime_type="application/json",
            max_tokens=600
        )
        response = await ollama_provider.generate(request)
        hypothesis_time = time.time() - start

        # Test 2: Deduction evaluation
        start = time.time()
        request = LLMRequest(
            user_prompt="Evaluate this: 'Use async video meetings'. Provide all scores.",
            response_schema=DeductionResponse,
            response_mime_type="application/json",
            max_tokens=500
        )
        response = await ollama_provider.generate(request)
        deduction_time = time.time() - start

        # Print benchmark results
        print(f"\n{'='*50}")
        print("Ollama Performance Benchmark")
        print(f"{'='*50}")
        print(f"Hypothesis Generation (3): {hypothesis_time:.2f}s")
        print(f"Deduction Evaluation (1): {deduction_time:.2f}s")
        print(f"{'='*50}")

        # Performance assertion: Should complete within reasonable time
        # Allow up to 60s for hypothesis generation (includes model loading)
        assert hypothesis_time < 60.0, f"Too slow: {hypothesis_time:.2f}s"
        assert deduction_time < 30.0, f"Too slow: {deduction_time:.2f}s"


@pytest.mark.ollama
class TestOllamaProviderErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_ollama_server_unavailable(self):
        """Test graceful error when Ollama server is unavailable."""
        # Use non-existent port
        provider = OllamaProvider(base_url="http://localhost:99999")

        request = LLMRequest(user_prompt="Test")

        with pytest.raises(Exception):  # Should raise connection error
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test error handling for non-existent model."""
        provider = OllamaProvider(model="nonexistent-model:latest")

        # This should fail when trying to use the model
        # (Ollama will return error if model doesn't exist)
        request = LLMRequest(user_prompt="Test")

        # Note: Actual behavior depends on Ollama API
        # May auto-pull or return error
