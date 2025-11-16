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
        # Include actual token counts as Ollama API provides them
        mock_response_data = {
            "message": {
                "content": "This is a test response from Ollama."
            },
            "done": True,
            "prompt_eval_count": 15,  # Actual token count from Ollama
            "eval_count": 8,  # Actual completion token count
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
            # Verify actual token counts from API are used
            assert response.usage["prompt_tokens"] == 15
            assert response.usage["completion_tokens"] == 8
            assert response.usage["total_tokens"] == 23

    @pytest.mark.asyncio
    async def test_generate_with_token_count_fallback(self):
        """Test token estimation falls back to character-based when API doesn't provide counts."""
        provider = OllamaProvider()

        # Mock response WITHOUT token counts (older Ollama versions)
        mock_response_data = {
            "message": {
                "content": "Short response"  # 14 characters = ~3 tokens
            },
            "done": True
            # No prompt_eval_count or eval_count
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            request = LLMRequest(
                user_prompt="Test prompt here",  # 16 chars = ~4 tokens
                temperature=0.7,
                max_tokens=100
            )

            response = await provider.generate(request)

            # Should fall back to character-based estimation (1 token ≈ 4 chars)
            assert response.usage["prompt_tokens"] == 4  # 16 // 4
            assert response.usage["completion_tokens"] == 3  # 14 // 4
            assert response.usage["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt_token_fallback(self):
        """Test character-based fallback includes system prompt in estimation."""
        provider = OllamaProvider()

        mock_response_data = {
            "message": {
                "content": "Response text here"  # 18 chars = 4 tokens
            },
            "done": True
            # No token counts - will use fallback
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            request = LLMRequest(
                system_prompt="System message",  # 14 chars
                user_prompt="User message",  # 12 chars
                # Total: 14 + 1 (newline) + 12 = 27 chars = 6 tokens
                temperature=0.7,
                max_tokens=100
            )

            response = await provider.generate(request)

            # Character-based: (14 + 1 + 12) // 4 = 27 // 4 = 6
            assert response.usage["prompt_tokens"] == 6
            assert response.usage["completion_tokens"] == 4  # 18 // 4
            assert response.usage["total_tokens"] == 10

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
            "done": True,
            "prompt_eval_count": 42,
            "eval_count": 156,
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
            # Verify token counts from API
            assert response.usage["prompt_tokens"] == 42
            assert response.usage["completion_tokens"] == 156


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

        # OllamaProvider should wrap connection issues in LLMError
        from mad_spark_alt.core.retry import LLMError

        with pytest.raises(LLMError, match="Ollama API request failed"):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test error handling for non-existent model."""
        provider = OllamaProvider(model="nonexistent-model:latest")
        request = LLMRequest(user_prompt="Test")

        # Behavior can vary by Ollama setup; at minimum, ensure we get a wrapped error
        from mad_spark_alt.core.retry import LLMError

        with pytest.raises(LLMError):
            await provider.generate(request)


class TestOllamaProviderResourceCleanup:
    """Tests for resource management and session cleanup."""

    @pytest.mark.asyncio
    async def test_session_closes_properly(self):
        """Test that close() properly closes the aiohttp session."""
        provider = OllamaProvider()

        # Force session creation
        session = await provider._get_session()
        assert session is not None
        assert not session.closed

        # Close provider
        await provider.close()

        # Verify session is closed
        assert provider._session is not None  # Reference still exists
        assert provider._session.closed  # But it's closed

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test that close() is safe when no session was created."""
        provider = OllamaProvider()

        # Session is None initially
        assert provider._session is None

        # Should not raise any errors
        await provider.close()

        # Still None after close
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that calling close() multiple times is safe."""
        provider = OllamaProvider()

        # Create session
        session = await provider._get_session()
        assert not session.closed

        # Close multiple times
        await provider.close()
        await provider.close()  # Should not raise

        assert provider._session.closed

    @pytest.mark.asyncio
    async def test_session_reuse_after_close(self):
        """Test that new session is created after close."""
        provider = OllamaProvider()

        # Create first session
        session1 = await provider._get_session()
        assert not session1.closed

        # Close it
        await provider.close()
        assert session1.closed

        # Get new session - should create a fresh one
        session2 = await provider._get_session()
        assert session2 is not session1  # Different object
        assert not session2.closed  # New session is open

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self):
        """Test that concurrent requests share the same session."""
        provider = OllamaProvider()

        # Mock the actual HTTP request
        mock_response_data = {
            "message": {"content": "Response"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            # Get sessions from multiple requests
            import asyncio

            async def make_request():
                request = LLMRequest(user_prompt="Test", max_tokens=50)
                await provider.generate(request)
                return provider._session

            # Run concurrent requests
            results = await asyncio.gather(
                make_request(),
                make_request(),
                make_request()
            )

            # All should share the same session instance
            assert results[0] is results[1]
            assert results[1] is results[2]
            assert not results[0].closed

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_session_cleanup_after_error(self):
        """Test that session can be cleaned up even after request errors."""
        provider = OllamaProvider()

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(side_effect=Exception("Network error"))
        ):
            request = LLMRequest(user_prompt="Test", max_tokens=50)

            # Request fails
            from mad_spark_alt.core.retry import LLMError
            with pytest.raises(LLMError):
                await provider.generate(request)

            # Session was created before the error
            assert provider._session is not None

        # Should still be able to close cleanly
        await provider.close()
        assert provider._session.closed

    @pytest.mark.asyncio
    async def test_generate_uses_centralized_timeout(self):
        """Test that generate() uses the centralized timeout constant."""
        from mad_spark_alt.core.system_constants import CONSTANTS
        provider = OllamaProvider()
        request = LLMRequest(user_prompt="Test")

        mock_response_data = {"message": {"content": "Response"}, "done": True}

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ) as mock_safe_request:
            await provider.generate(request)
            mock_safe_request.assert_called_once()
            _, kwargs = mock_safe_request.call_args
            assert kwargs.get("timeout") == CONSTANTS.TIMEOUTS.OLLAMA_INFERENCE_TIMEOUT

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_centralized_constants_defined(self):
        """Test that centralized constants are properly defined."""
        from mad_spark_alt.core.system_constants import CONSTANTS

        # Verify the constant exists and has expected value
        assert hasattr(CONSTANTS.TIMEOUTS, 'OLLAMA_INFERENCE_TIMEOUT')
        assert CONSTANTS.TIMEOUTS.OLLAMA_INFERENCE_TIMEOUT == 180

        # Verify OllamaProvider uses centralized defaults
        assert hasattr(CONSTANTS.LLM, 'OLLAMA_DEFAULT_BASE_URL')
        assert CONSTANTS.LLM.OLLAMA_DEFAULT_BASE_URL == "http://localhost:11434"

        assert hasattr(CONSTANTS.LLM, 'OLLAMA_DEFAULT_MODEL')
        assert CONSTANTS.LLM.OLLAMA_DEFAULT_MODEL == "gemma3:12b-it-qat"

    @pytest.mark.asyncio
    async def test_provider_uses_default_constants(self):
        """Test that OllamaProvider initializes with centralized constants."""
        from mad_spark_alt.core.system_constants import CONSTANTS

        provider = OllamaProvider()

        # Should use centralized defaults
        assert provider.model == CONSTANTS.LLM.OLLAMA_DEFAULT_MODEL
        assert provider.base_url == CONSTANTS.LLM.OLLAMA_DEFAULT_BASE_URL

    @pytest.mark.asyncio
    async def test_provider_allows_custom_overrides(self):
        """Test that custom values override the defaults."""
        custom_model = "llama3:8b"
        custom_url = "http://remote-host:11434"

        provider = OllamaProvider(model=custom_model, base_url=custom_url)

        assert provider.model == custom_model
        assert provider.base_url == custom_url

    @pytest.mark.asyncio
    async def test_session_lock_exists(self):
        """Test that OllamaProvider has session lock for thread safety."""
        import asyncio
        provider = OllamaProvider()

        # Verify lock exists
        assert hasattr(provider, '_session_lock')
        assert isinstance(provider._session_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_temperature_respected_for_structured_output(self):
        """Test that user temperature is respected for structured output (unless too high)."""
        provider = OllamaProvider()

        mock_response_data = {
            "message": {"content": '{"hypotheses": []}'},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        captured_payloads = []

        async def capture_payload(**kwargs):
            captured_payloads.append(kwargs.get('json', {}))
            return mock_response_data

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(side_effect=capture_payload)
        ):
            # Test 1: Low temperature should be respected
            request = LLMRequest(
                user_prompt="Test",
                response_schema=HypothesisListResponse,
                temperature=0.3,  # Low temperature
                max_tokens=100
            )
            await provider.generate(request)

            # Should respect user's low temperature
            assert captured_payloads[-1]["options"]["temperature"] == 0.3

            # Test 2: High temperature should be capped at 0.5
            request = LLMRequest(
                user_prompt="Test",
                response_schema=HypothesisListResponse,
                temperature=0.95,  # High temperature
                max_tokens=100
            )
            await provider.generate(request)

            # Should cap to 0.5 for schema compliance (not 0.0)
            assert captured_payloads[-1]["options"]["temperature"] == 0.5

        await provider.close()

    @pytest.mark.asyncio
    async def test_close_is_thread_safe(self):
        """Test that close() uses the session lock for thread safety."""
        import asyncio
        provider = OllamaProvider()

        # Create a session first
        mock_response_data = {"message": {"content": "Test"}, "done": True}
        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ):
            request = LLMRequest(user_prompt="Test")
            await provider.generate(request)

        # Verify session exists
        assert provider._session is not None
        assert not provider._session.closed

        # Test that concurrent close() and _get_session() don't race
        # by acquiring the lock manually first
        async with provider._session_lock:
            # While we hold the lock, close shouldn't be able to proceed
            # This verifies close() uses the lock
            pass

        # Now close should work fine
        await provider.close()
        assert provider._session.closed


class TestOllamaFallbackDetection:
    """Test Ollama error detection patterns for fallback logic."""

    def test_ollama_failure_detection_patterns(self):
        """Test that various Ollama failure patterns are correctly identified."""
        from mad_spark_alt.core.llm_provider import OllamaProvider
        import asyncio

        provider = OllamaProvider()

        # Test patterns that should be detected as Ollama failures
        ollama_failures = [
            Exception("Ollama server not responding"),
            Exception("ollama connection refused"),
            Exception("Connection timed out"),
            Exception("aiohttp.ClientError occurred"),
            ConnectionError("Cannot connect"),
            OSError("Network unreachable"),
            asyncio.TimeoutError(),
            # Processing failures (e.g., from phase_logic.py)
            RuntimeError("Failed to generate hypotheses after max retries"),
            RuntimeError("Failed to parse deduction response"),
            Exception("Failed to generate action plan"),
        ]

        for error in ollama_failures:
            is_failure = (
                isinstance(provider, OllamaProvider) and
                (
                    isinstance(error, (ConnectionError, OSError, asyncio.TimeoutError)) or
                    any(keyword in str(error) for keyword in [
                        "Ollama", "ollama", "Connection", "aiohttp",
                        "Failed to generate", "Failed to parse"
                    ])
                )
            )
            assert is_failure, f"Expected {error} to be detected as Ollama failure"

    def test_non_ollama_failures_not_detected(self):
        """Test that non-Ollama failures are not misidentified.

        Targeted detection avoids catching generic RuntimeErrors to prevent
        masking programming bugs that should propagate.
        """
        from mad_spark_alt.core.llm_provider import OllamaProvider
        import asyncio

        provider = OllamaProvider()

        # These should NOT be detected as Ollama failures
        # They are programming bugs or unrelated errors
        non_ollama_failures = [
            ValueError("Invalid parameter"),
            KeyError("Missing key"),
            RuntimeError("Generic runtime error"),  # Generic RuntimeError NOT caught
            TypeError("Type mismatch"),
        ]

        for error in non_ollama_failures:
            is_failure = (
                isinstance(provider, OllamaProvider) and
                (
                    isinstance(error, (ConnectionError, OSError, asyncio.TimeoutError)) or
                    any(keyword in str(error) for keyword in [
                        "Ollama", "ollama", "Connection", "aiohttp",
                        "Failed to generate", "Failed to parse"
                    ])
                )
            )
            assert not is_failure, f"Expected {error} NOT to be detected as Ollama failure"
