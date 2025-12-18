"""
Tests for LLM provider infrastructure.

This module tests the LLM provider abstraction, rate limiting,
cost tracking, and error handling functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMManager,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelSize,
    RateLimitConfig,
    RateLimiter,
    UsageStats,
)
from mad_spark_alt.core.retry import ErrorType, LLMError


class TestLLMProvider:
    """Test LLM provider base functionality."""

    def test_model_config_creation(self):
        """Test ModelConfig creation and validation."""
        config = ModelConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-3-flash-preview",
            model_size=ModelSize.LARGE,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        assert config.provider == LLMProvider.GOOGLE
        assert config.model_name == "gemini-3-flash-preview"
        assert config.model_size == ModelSize.LARGE
        assert config.input_cost_per_1k == 0.00015

    def test_llm_request_creation(self):
        """Test LLMRequest creation and validation."""
        request = LLMRequest(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
            max_tokens=100,
            temperature=0.7,
        )

        assert request.system_prompt == "You are a helpful assistant."
        assert request.user_prompt == "What is the capital of France?"
        assert request.max_tokens == 100
        assert request.temperature == 0.7


class TestUsageStats:
    """Test usage statistics tracking."""

    def test_usage_stats_initialization(self):
        """Test UsageStats initialization."""
        stats = UsageStats(LLMProvider.GOOGLE, "gemini-3-flash-preview")

        assert stats.provider == LLMProvider.GOOGLE
        assert stats.model == "gemini-3-flash-preview"
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_requests == 0
        assert stats.total_cost == 0.0

    def test_usage_stats_addition(self):
        """Test adding usage statistics."""
        stats = UsageStats(LLMProvider.GOOGLE, "gemini-3-flash-preview")

        stats.add_usage(100, 50, 0.05)

        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_requests == 1
        assert stats.total_cost == 0.05
        assert stats.last_request is not None

    def test_usage_stats_accumulation(self):
        """Test accumulating multiple usage entries."""
        stats = UsageStats(LLMProvider.GOOGLE, "gemini-3-flash-preview")

        stats.add_usage(100, 50, 0.05)
        stats.add_usage(200, 75, 0.08)

        assert stats.input_tokens == 300
        assert stats.output_tokens == 125
        assert stats.total_requests == 2
        assert stats.total_cost == 0.13


class TestLLMManager:
    """Test LLM manager functionality."""

    def test_llm_manager_initialization(self):
        """Test LLMManager initialization."""
        manager = LLMManager()

        assert len(manager.providers) == 0
        assert len(manager.usage_stats) == 0
        assert len(manager.rate_limiters) == 0

    def test_provider_registration(self):
        """Test provider registration."""
        manager = LLMManager()
        mock_provider = MagicMock()

        manager.register_provider(LLMProvider.GOOGLE, mock_provider)

        assert LLMProvider.GOOGLE in manager.providers
        assert manager.providers[LLMProvider.GOOGLE] == mock_provider
        assert LLMProvider.GOOGLE in manager.rate_limiters

    def test_default_model_setting(self):
        """Test setting default model configuration."""
        manager = LLMManager()
        config = ModelConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-3-flash-preview",
            model_size=ModelSize.LARGE,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        manager.set_default_model(LLMProvider.GOOGLE, config)

        assert manager.default_configs[LLMProvider.GOOGLE] == config

    @pytest.mark.asyncio
    async def test_generate_with_no_providers(self):
        """Test generate method with no providers registered."""
        manager = LLMManager()
        request = LLMRequest(user_prompt="Test")

        with pytest.raises(ValueError, match="No LLM providers registered"):
            await manager.generate(request)

    @pytest.mark.asyncio
    async def test_generate_with_unregistered_provider(self):
        """Test generate method with unregistered provider."""
        manager = LLMManager()
        request = LLMRequest(user_prompt="Test")

        with pytest.raises(ValueError, match="Provider .* not registered"):
            await manager.generate(request, LLMProvider.GOOGLE)


class TestGoogleProvider:
    """Test Google provider implementation."""

    def test_google_provider_initialization(self):
        """Test GoogleProvider initialization."""
        provider = GoogleProvider("test_api_key")

        assert provider.api_key == "test_api_key"
        assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"
        assert provider.retry_config is not None
        assert provider.circuit_breaker is not None

    def test_get_available_models(self):
        """Test getting available Google models."""
        provider = GoogleProvider("test_api_key")
        models = provider.get_available_models()

        assert len(models) == 1
        assert models[0].model_name == "gemini-3-flash-preview"

    def test_calculate_cost(self):
        """Test cost calculation for Google."""
        provider = GoogleProvider("test_api_key")
        config = ModelConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-3-flash-preview",
            model_size=ModelSize.LARGE,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        cost = provider.calculate_cost(1000, 500, config)
        expected_cost = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006

        assert cost == expected_cost

    @pytest.mark.asyncio
    async def test_google_provider_session_management(self):
        """Test Google provider session management."""
        provider = GoogleProvider("test_api_key")

        # Test session creation
        session1 = await provider._get_session()
        assert session1 is not None

        # Test session reuse
        session2 = await provider._get_session()
        assert session1 is session2

        # Test session cleanup
        await provider.close()


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_config(self):
        """Test RateLimitConfig creation."""
        config = RateLimitConfig(
            requests_per_minute=60, tokens_per_minute=150000, max_concurrent_requests=10
        )

        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 150000
        assert config.max_concurrent_requests == 10

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        config = RateLimitConfig(
            requests_per_minute=10, tokens_per_minute=1000, max_concurrent_requests=2
        )
        limiter = RateLimiter(config)

        assert limiter.config == config
        assert len(limiter.request_times) == 0
        assert len(limiter.token_usage) == 0
        assert limiter._semaphore._value == 2  # max_concurrent_requests

    @pytest.mark.asyncio
    async def test_rate_limiter_basic_acquire_release(self):
        """Test basic acquire and release functionality."""
        config = RateLimitConfig(
            requests_per_minute=60, tokens_per_minute=1000, max_concurrent_requests=3
        )
        limiter = RateLimiter(config)

        # Should acquire without delay
        await limiter.acquire(100)
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1
        assert limiter.token_usage[0][1] == 100  # tokens

        limiter.release()

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self):
        """Test concurrent request limiting."""
        config = RateLimitConfig(
            requests_per_minute=60, tokens_per_minute=10000, max_concurrent_requests=2
        )
        limiter = RateLimiter(config)

        # Acquire two concurrent slots
        await limiter.acquire(100)
        await limiter.acquire(100)

        # Third request should be blocked (would need to test with timeout in real scenario)
        assert limiter._semaphore._value == 0  # No more slots available

        # Release one slot
        limiter.release()
        assert limiter._semaphore._value == 1  # One slot available

        limiter.release()
        assert limiter._semaphore._value == 2  # Back to full capacity

    @pytest.mark.asyncio
    async def test_rate_limiter_cleans_old_entries(self):
        """Test that rate limiter cleans up old entries."""
        config = RateLimitConfig(
            requests_per_minute=5, tokens_per_minute=500, max_concurrent_requests=5
        )
        limiter = RateLimiter(config)

        # Manually add old entries (more than 60 seconds ago)
        import time

        old_time = time.time() - 70  # 70 seconds ago
        limiter.request_times.append(old_time)
        limiter.token_usage.append((old_time, 100))

        # Acquire should clean up old entries
        await limiter.acquire(50)

        # Old entries should be removed, only new entry should remain
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1
        assert limiter.request_times[0] > old_time
        assert limiter.token_usage[0][0] > old_time

        limiter.release()

    @pytest.mark.asyncio
    async def test_rate_limiter_loop_logic(self):
        """Test that rate limiter loop correctly evaluates conditions."""
        config = RateLimitConfig(
            requests_per_minute=60, tokens_per_minute=500, max_concurrent_requests=5
        )
        limiter = RateLimiter(config)

        # Add enough token usage to trigger the token limit
        import time

        now = time.time()
        limiter.token_usage = [(now, 400), (now, 150)]  # 550 tokens used

        # Should not be able to acquire 100 more tokens (would exceed 500 limit)
        # But we can't easily test the sleep without mocking, so just verify
        # the calculation logic by checking what would trigger sleep
        current_tokens = sum(tokens for _, tokens in limiter.token_usage)
        assert current_tokens + 100 > config.tokens_per_minute

        # Clean up for next test
        limiter.token_usage = []


class TestGoogleProviderStructuredOutput:
    """Test Google provider structured output field names."""

    @pytest.mark.asyncio
    async def test_structured_output_uses_correct_field_name(self, sample_model_config: ModelConfig):
        """Test that structured output uses responseJsonSchema, not responseSchema."""
        provider = GoogleProvider("test_api_key")

        # Use sample config fixture
        config = sample_model_config

        # Create request with structured output
        request = LLMRequest(
            user_prompt="Generate a hypothesis",
            max_tokens=1000,
            temperature=0.7,
            model_configuration=config,
            response_schema={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            response_mime_type="application/json"
        )

        # Mock the safe_aiohttp_request to capture the payload
        captured_payload = {}

        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": '{"hypotheses": ["test"]}'}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50
            }
        }

        async def mock_safe_request(**kwargs):
            nonlocal captured_payload
            if 'json' in kwargs:
                captured_payload = kwargs['json']
            return mock_response_data

        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=mock_safe_request):
            await provider.generate(request)

        # Verify the payload structure
        assert "generationConfig" in captured_payload, "Payload missing generationConfig"
        gen_config = captured_payload["generationConfig"]

        # The CRITICAL assertion: should use responseJsonSchema, not responseSchema
        assert "responseJsonSchema" in gen_config, \
            "Payload should use 'responseJsonSchema' per Gemini API specification"
        assert "responseSchema" not in gen_config, \
            "Payload should NOT use 'responseSchema' (incorrect field name)"

        # Verify the schema was included
        assert gen_config["responseJsonSchema"] == request.response_schema

        # Verify responseMimeType is also present
        assert "responseMimeType" in gen_config
        assert gen_config["responseMimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_url_context_disables_structured_output(self, sample_model_config: ModelConfig):
        """Test that URLs disable structured output (Gemini API limitation)."""
        provider = GoogleProvider("test_api_key")

        config = sample_model_config

        # Create request with BOTH URLs and response_schema
        request = LLMRequest(
            user_prompt="Summarize this page",
            urls=["https://example.com"],
            max_tokens=1000,
            temperature=0.7,
            model_configuration=config,
            response_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"}
                }
            },
            response_mime_type="application/json"
        )

        # Mock the safe_aiohttp_request to capture the payload
        captured_payload = {}

        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": 'Example Domain summary'}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50
            }
        }

        async def mock_safe_request(**kwargs):
            nonlocal captured_payload
            if 'json' in kwargs:
                captured_payload = kwargs['json']
            return mock_response_data

        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=mock_safe_request):
            await provider.generate(request)

        # Verify the payload structure
        assert "generationConfig" in captured_payload
        gen_config = captured_payload["generationConfig"]

        # CRITICAL: Structured output should be DISABLED when URLs are present
        assert "responseMimeType" not in gen_config, \
            "URL context incompatible with structured output - responseMimeType should be disabled"
        assert "responseJsonSchema" not in gen_config, \
            "URL context incompatible with structured output - responseJsonSchema should be disabled"

        # Verify URL tool is still present
        assert "tools" in captured_payload
        # Tools configuration varies, just verify structured output was disabled


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for LLM provider system."""

    async def test_mock_llm_workflow(self):
        """Test complete LLM workflow with mocked responses."""
        # Create mock response
        mock_response = LLMResponse(
            content="The capital of France is Paris.",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={"prompt_tokens": 10, "completion_tokens": 8},
            cost=0.001,
            response_time=0.5,
        )

        # Create mock provider
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = mock_response

        # Setup manager
        manager = LLMManager()
        manager.register_provider(LLMProvider.GOOGLE, mock_provider)

        # Test request
        request = LLMRequest(user_prompt="What is the capital of France?")
        response = await manager.generate(request, LLMProvider.GOOGLE)

        assert response.content == "The capital of France is Paris."
        assert response.provider == LLMProvider.GOOGLE
        assert response.cost == 0.001

        # Check usage tracking
        assert len(manager.usage_stats) == 1
        stats_key = "google:gemini-3-flash-preview"
        assert stats_key in manager.usage_stats
        assert manager.usage_stats[stats_key].total_requests == 1

    async def test_error_handling(self):
        """Test error handling in LLM workflow."""
        # Create mock provider that raises an error
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = LLMError("API Error", ErrorType.API_ERROR)

        # Setup manager
        manager = LLMManager()
        manager.register_provider(LLMProvider.GOOGLE, mock_provider)

        # Test request that should fail
        request = LLMRequest(user_prompt="Test")

        with pytest.raises(LLMError):
            await manager.generate(request, LLMProvider.GOOGLE)


# Fixtures for testing
@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        provider=LLMProvider.GOOGLE,
        model_name="gemini-3-flash-preview",
        model_size=ModelSize.LARGE,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
    )


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    return LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing."""
    return LLMResponse(
        content="The capital of France is Paris.",
        provider=LLMProvider.GOOGLE,
        model="gemini-3-flash-preview",
        usage={"prompt_tokens": 10, "completion_tokens": 8},
        cost=0.001,
        response_time=0.5,
    )


class TestMultimodalLLMRequest:
    """Test LLMRequest with multimodal inputs."""

    def test_llm_request_with_multimodal_inputs(self):
        """Test creating LLMRequest with multimodal inputs."""
        from mad_spark_alt.core.multimodal import (
            MultimodalInput,
            MultimodalInputType,
            MultimodalSourceType,
        )

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Describe this image",
            multimodal_inputs=[multimodal_input]
        )

        assert request.user_prompt == "Describe this image"
        assert request.multimodal_inputs is not None
        assert len(request.multimodal_inputs) == 1
        assert request.multimodal_inputs[0].input_type == MultimodalInputType.IMAGE

    def test_llm_request_with_urls(self):
        """Test creating LLMRequest with URLs."""
        request = LLMRequest(
            user_prompt="Summarize these sources",
            urls=["https://example.com/article1", "https://example.com/article2"]
        )

        assert request.urls is not None
        assert len(request.urls) == 2
        assert request.urls[0] == "https://example.com/article1"

    def test_llm_request_with_tools(self):
        """Test creating LLMRequest with tools."""
        request = LLMRequest(
            user_prompt="Test",
            tools=[{"url_context": {}}]
        )

        assert request.tools is not None
        assert len(request.tools) == 1
        assert "url_context" in request.tools[0]

    def test_llm_request_multimodal_defaults_to_none(self):
        """Test that multimodal fields default to None for backward compatibility."""
        request = LLMRequest(user_prompt="Test")

        assert request.multimodal_inputs is None
        assert request.urls is None
        assert request.tools is None

    def test_llm_request_validation_max_urls(self):
        """Test LLMRequest validation rejects >20 URLs."""
        from mad_spark_alt.core.llm_provider import validate_llm_request

        # Create request with 21 URLs (over limit)
        urls = [f"https://example.com/page{i}" for i in range(21)]
        request = LLMRequest(
            user_prompt="Test",
            urls=urls
        )

        with pytest.raises(ValueError, match=r"Too many URLs.*max 20"):
            validate_llm_request(request)

    def test_llm_request_validation_max_images(self):
        """Test LLMRequest validation rejects >3600 images."""
        from mad_spark_alt.core.llm_provider import validate_llm_request
        from mad_spark_alt.core.multimodal import (
            MultimodalInput,
            MultimodalInputType,
            MultimodalSourceType,
        )

        # Create request with 3601 images (over limit)
        images = [
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=f"/path/to/image{i}.png",
                mime_type="image/png"
            )
            for i in range(3601)
        ]

        request = LLMRequest(
            user_prompt="Test",
            multimodal_inputs=images
        )

        with pytest.raises(ValueError, match=r"Too many images.*max 3600"):
            validate_llm_request(request)

    def test_llm_request_validation_accepts_20_urls(self):
        """Test LLMRequest validation accepts exactly 20 URLs."""
        from mad_spark_alt.core.llm_provider import validate_llm_request

        urls = [f"https://example.com/page{i}" for i in range(20)]
        request = LLMRequest(
            user_prompt="Test",
            urls=urls
        )

        # Should not raise
        validate_llm_request(request)

    def test_llm_request_validation_accepts_3600_images(self):
        """Test LLMRequest validation accepts exactly 3600 images."""
        from mad_spark_alt.core.llm_provider import validate_llm_request
        from mad_spark_alt.core.multimodal import (
            MultimodalInput,
            MultimodalInputType,
            MultimodalSourceType,
        )

        images = [
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=f"/path/to/image{i}.png",
                mime_type="image/png"
            )
            for i in range(3600)
        ]

        request = LLMRequest(
            user_prompt="Test",
            multimodal_inputs=images
        )

        # Should not raise
        validate_llm_request(request)

    def test_llm_request_validation_calls_multimodal_validate(self):
        """Test that LLMRequest validation calls validate() on each MultimodalInput."""
        from mad_spark_alt.core.llm_provider import validate_llm_request
        from mad_spark_alt.core.multimodal import (
            MultimodalInput,
            MultimodalInputType,
            MultimodalSourceType,
        )

        # Create invalid multimodal input (invalid MIME type for image)
        invalid_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data",
            mime_type="application/pdf"  # Invalid for image
        )

        request = LLMRequest(
            user_prompt="Test",
            multimodal_inputs=[invalid_input]
        )

        with pytest.raises(ValueError, match="Unsupported image type"):
            validate_llm_request(request)

    def test_llm_request_mixed_multimodal_and_urls(self):
        """Test LLMRequest with both multimodal inputs and URLs."""
        from mad_spark_alt.core.multimodal import (
            MultimodalInput,
            MultimodalInputType,
            MultimodalSourceType,
        )

        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Analyze image and context",
            multimodal_inputs=[image],
            urls=["https://example.com/context"]
        )

        assert request.multimodal_inputs is not None
        assert len(request.multimodal_inputs) == 1
        assert request.urls is not None
        assert len(request.urls) == 1


class TestMultimodalLLMResponse:
    """Test LLMResponse with multimodal metadata."""

    def test_llm_response_with_url_context_metadata(self):
        """Test LLMResponse with URL context metadata."""
        from mad_spark_alt.core.multimodal import URLContextMetadata

        url_metadata = [
            URLContextMetadata(url="https://example.com/1", status="success"),
            URLContextMetadata(url="https://example.com/2", status="success")
        ]

        response = LLMResponse(
            content="Summary of sources",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={"prompt_tokens": 1000, "completion_tokens": 200},
            cost=0.01,
            url_context_metadata=url_metadata
        )

        assert response.url_context_metadata is not None
        assert len(response.url_context_metadata) == 2
        assert response.url_context_metadata[0].status == "success"

    def test_llm_response_with_image_count(self):
        """Test LLMResponse with total_images_processed."""
        response = LLMResponse(
            content="Analysis complete",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={"prompt_tokens": 500, "completion_tokens": 100},
            cost=0.005,
            total_images_processed=3
        )

        assert response.total_images_processed == 3

    def test_llm_response_with_pages_count(self):
        """Test LLMResponse with total_pages_processed."""
        response = LLMResponse(
            content="Document summary",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={"prompt_tokens": 10000, "completion_tokens": 500},
            cost=0.1,
            total_pages_processed=50
        )

        assert response.total_pages_processed == 50

    def test_llm_response_multimodal_defaults_to_none(self):
        """Test that multimodal fields default to None for backward compatibility."""
        response = LLMResponse(
            content="Test",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={},
            cost=0.0
        )

        assert response.url_context_metadata is None
        assert response.total_images_processed is None
        assert response.total_pages_processed is None

    def test_llm_response_with_all_multimodal_metadata(self):
        """Test LLMResponse with all multimodal metadata fields."""
        from mad_spark_alt.core.multimodal import URLContextMetadata

        url_metadata = [
            URLContextMetadata(url="https://example.com", status="success")
        ]

        response = LLMResponse(
            content="Complete analysis",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            usage={"prompt_tokens": 5000, "completion_tokens": 500},
            cost=0.05,
            url_context_metadata=url_metadata,
            total_images_processed=2,
            total_pages_processed=10
        )

        assert response.url_context_metadata is not None
        assert len(response.url_context_metadata) == 1
        assert response.total_images_processed == 2
        assert response.total_pages_processed == 10


class TestInlineSchemaDefsFunction:
    """Test inline_schema_defs() utility for Ollama compatibility."""

    def test_passthrough_schema_without_defs(self):
        """Test that schema without $defs is returned unchanged."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

        result = inline_schema_defs(schema)

        # Should be unchanged (no $defs to inline)
        assert result == schema
        assert "$defs" not in result

    def test_inline_single_ref(self):
        """Test inlining a single $ref to $defs."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        schema = {
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "owner": {"$ref": "#/$defs/Person"}
            }
        }

        result = inline_schema_defs(schema)

        # $defs should be removed
        assert "$defs" not in result
        # Reference should be inlined
        assert result["properties"]["owner"]["type"] == "object"
        assert "name" in result["properties"]["owner"]["properties"]

    def test_inline_nested_refs(self):
        """Test inlining nested $refs within arrays."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "value": {"type": "string"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"}
                }
            }
        }

        result = inline_schema_defs(schema)

        # $defs should be removed
        assert "$defs" not in result
        # Array items should have inlined definition
        items_schema = result["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "id" in items_schema["properties"]
        assert "value" in items_schema["properties"]

    def test_inline_multiple_refs_to_same_def(self):
        """Test inlining multiple references to the same definition."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "home_address": {"$ref": "#/$defs/Address"},
                "work_address": {"$ref": "#/$defs/Address"}
            }
        }

        result = inline_schema_defs(schema)

        # $defs should be removed
        assert "$defs" not in result
        # Both references should be inlined independently
        assert result["properties"]["home_address"]["type"] == "object"
        assert result["properties"]["work_address"]["type"] == "object"
        assert "street" in result["properties"]["home_address"]["properties"]
        assert "street" in result["properties"]["work_address"]["properties"]

    def test_empty_schema_passthrough(self):
        """Test that empty or None schema returns unchanged."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        assert inline_schema_defs({}) == {}
        assert inline_schema_defs(None) is None

    def test_pydantic_batch_mutation_schema(self):
        """Test with real Pydantic BatchMutationResponse schema."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs
        from mad_spark_alt.core.schemas import BatchMutationResponse

        original = BatchMutationResponse.model_json_schema()
        result = inline_schema_defs(original)

        # $defs should be removed
        assert "$defs" not in result
        # Should still have the mutations array
        assert "mutations" in result["properties"]
        # Items should be inlined (not a $ref)
        items = result["properties"]["mutations"]["items"]
        assert "$ref" not in items
        assert items["type"] == "object"
        assert "mutated_idea" in items["properties"]

    def test_pydantic_batch_crossover_schema(self):
        """Test with real Pydantic BatchCrossoverResponse schema."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs
        from mad_spark_alt.core.schemas import BatchCrossoverResponse

        original = BatchCrossoverResponse.model_json_schema()
        result = inline_schema_defs(original)

        # $defs should be removed
        assert "$defs" not in result
        # Should still have the crossovers array
        assert "crossovers" in result["properties"]
        # Items should be inlined (not a $ref)
        items = result["properties"]["crossovers"]["items"]
        assert "$ref" not in items
        assert items["type"] == "object"
        assert "pair_id" in items["properties"]
        assert "offspring1" in items["properties"]

    def test_recursive_defs_resolution(self):
        """Test resolution of definitions that reference other definitions."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs

        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"$ref": "#/$defs/Address"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "employee": {"$ref": "#/$defs/Person"}
            }
        }

        result = inline_schema_defs(schema)

        # $defs should be removed
        assert "$defs" not in result
        # Person should be inlined
        employee = result["properties"]["employee"]
        assert employee["type"] == "object"
        assert "name" in employee["properties"]
        # Address within Person should also be inlined
        address = employee["properties"]["address"]
        assert address["type"] == "object"
        assert "street" in address["properties"]
        assert "city" in address["properties"]

    def test_schema_caching_performance(self):
        """Test that schema caching works (same input returns cached result)."""
        from mad_spark_alt.core.llm_provider import inline_schema_defs, _INLINED_SCHEMA_CACHE

        # Clear cache for clean test
        _INLINED_SCHEMA_CACHE.clear()

        schema = {
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}}
            },
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}}
        }

        # First call should populate cache
        result1 = inline_schema_defs(schema)

        # Verify cache was populated
        assert len(_INLINED_SCHEMA_CACHE) == 1

        # Second call with same schema should use cache
        result2 = inline_schema_defs(schema)

        # Results should be equal but not the same object (deep copy)
        assert result1 == result2
        assert result1 is not result2


class TestOllamaOutlinesFallback:
    """Tests for Outlines integration and fallback behavior."""

    @pytest.mark.asyncio
    async def test_outlines_fallback_logs_exception_type(self):
        """Test that fallback path logs exception type for debugging."""
        from unittest.mock import patch, AsyncMock, MagicMock
        from mad_spark_alt.core.llm_provider import OllamaProvider, LLMRequest, OUTLINES_AVAILABLE
        from mad_spark_alt.core.schemas import HypothesisListResponse

        if not OUTLINES_AVAILABLE:
            pytest.skip("Outlines not installed")

        provider = OllamaProvider()

        # Mock to force Outlines failure
        with patch.object(
            provider, '_generate_with_outlines',
            side_effect=RuntimeError("Test error")
        ):
            with patch(
                'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
                new=AsyncMock(return_value={
                    "message": {"content": '{"hypotheses": []}'},
                    "prompt_eval_count": 10,
                    "eval_count": 5
                })
            ):
                request = LLMRequest(
                    user_prompt="Test",
                    response_schema=HypothesisListResponse
                )

                # Should fall back to raw API without raising
                response = await provider.generate(request)

                # Verify we got a response (fallback worked)
                assert response is not None
                assert response.provider.value == "ollama"

        await provider.close()

    @pytest.mark.asyncio
    async def test_outlines_unavailable_raises_clear_error(self):
        """Test that missing Outlines raises clear error message."""
        from unittest.mock import patch
        from mad_spark_alt.core.llm_provider import OllamaProvider, LLMRequest
        from mad_spark_alt.core.retry import LLMError
        from mad_spark_alt.core.schemas import HypothesisListResponse

        provider = OllamaProvider()

        # Simulate Outlines not being available
        with patch('mad_spark_alt.core.llm_provider.OUTLINES_AVAILABLE', False):
            request = LLMRequest(
                user_prompt="Test",
                response_schema=HypothesisListResponse
            )

            with pytest.raises(LLMError) as exc_info:
                await provider._generate_with_outlines(request, HypothesisListResponse)

            assert "outlines" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

        await provider.close()

    @pytest.mark.asyncio
    async def test_outlines_timeout_protection(self):
        """Test that Outlines call has timeout protection."""
        from unittest.mock import patch, AsyncMock, MagicMock
        from mad_spark_alt.core.llm_provider import OllamaProvider, LLMRequest, OUTLINES_AVAILABLE
        from mad_spark_alt.core.retry import LLMError, ErrorType
        from mad_spark_alt.core.schemas import HypothesisListResponse
        import asyncio

        if not OUTLINES_AVAILABLE:
            pytest.skip("Outlines not installed")

        provider = OllamaProvider()

        # Create a slow async function that will be interrupted by timeout
        async def slow_outlines_call(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow call
            return HypothesisListResponse(hypotheses=[])

        request = LLMRequest(
            user_prompt="Test timeout",
            response_schema=HypothesisListResponse
        )

        with patch('mad_spark_alt.core.llm_provider.outlines') as mock_outlines:
            # Mock the model creation
            mock_model = MagicMock()
            mock_model.return_value = slow_outlines_call()  # Returns a coroutine
            mock_outlines.models.ollama.return_value = mock_model

            # Mock asyncio.wait_for to raise TimeoutError immediately
            async def immediate_timeout(*args, **kwargs):
                raise asyncio.TimeoutError()

            with patch('asyncio.wait_for', new=immediate_timeout):
                with pytest.raises(LLMError) as exc_info:
                    await provider._generate_with_outlines(request, HypothesisListResponse)

                # Should raise timeout error
                assert exc_info.value.error_type == ErrorType.TIMEOUT
                assert "timed out" in str(exc_info.value).lower()

        await provider.close()

    @pytest.mark.asyncio
    async def test_outlines_skipped_for_multimodal_requests(self):
        """Test that Outlines is skipped when multimodal inputs are present.

        Outlines flattens the prompt to plain text and loses image context.
        The native Ollama API properly handles images via the messages format.
        """
        from unittest.mock import patch, AsyncMock, MagicMock
        from mad_spark_alt.core.llm_provider import OllamaProvider, LLMRequest, OUTLINES_AVAILABLE
        from mad_spark_alt.core.schemas import HypothesisListResponse
        from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType

        provider = OllamaProvider()

        # Create a request with multimodal inputs (image)
        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64encodedimagedata",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Describe this image",
            response_schema=HypothesisListResponse,
            multimodal_inputs=[multimodal_input]
        )

        # Mock the raw API call (should be used instead of Outlines)
        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value={
                "message": {"content": '{"hypotheses": []}'},
                "prompt_eval_count": 10,
                "eval_count": 5
            })
        ) as mock_api:
            # Mock _generate_with_outlines to track if it's called
            with patch.object(
                provider, '_generate_with_outlines',
                new=AsyncMock()
            ) as mock_outlines:
                response = await provider.generate(request)

                # Outlines should NOT have been called due to multimodal inputs
                mock_outlines.assert_not_called()

                # Raw API should have been called instead
                mock_api.assert_called_once()

                # Verify response came from raw API
                assert response is not None
                assert response.provider.value == "ollama"

        await provider.close()
