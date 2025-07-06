"""
Tests for LLM provider infrastructure.

This module tests the LLM provider abstraction, rate limiting,
cost tracking, and error handling functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.llm_provider import (
    LLMManager,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelSize,
    OpenAIProvider,
    AnthropicProvider,
    RateLimitConfig,
    RateLimiter,
    UsageStats,
)
from mad_spark_alt.core.retry import LLMError, ErrorType


class TestLLMProvider:
    """Test LLM provider base functionality."""

    def test_model_config_creation(self):
        """Test ModelConfig creation and validation."""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            model_size=ModelSize.SMALL,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4o-mini"
        assert config.model_size == ModelSize.SMALL
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
        stats = UsageStats(LLMProvider.OPENAI, "gpt-4o-mini")

        assert stats.provider == LLMProvider.OPENAI
        assert stats.model == "gpt-4o-mini"
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_requests == 0
        assert stats.total_cost == 0.0

    def test_usage_stats_addition(self):
        """Test adding usage statistics."""
        stats = UsageStats(LLMProvider.OPENAI, "gpt-4o-mini")

        stats.add_usage(100, 50, 0.05)

        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_requests == 1
        assert stats.total_cost == 0.05
        assert stats.last_request is not None

    def test_usage_stats_accumulation(self):
        """Test accumulating multiple usage entries."""
        stats = UsageStats(LLMProvider.OPENAI, "gpt-4o-mini")

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

        manager.register_provider(LLMProvider.OPENAI, mock_provider)

        assert LLMProvider.OPENAI in manager.providers
        assert manager.providers[LLMProvider.OPENAI] == mock_provider
        assert LLMProvider.OPENAI in manager.rate_limiters

    def test_default_model_setting(self):
        """Test setting default model configuration."""
        manager = LLMManager()
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            model_size=ModelSize.SMALL,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        manager.set_default_model(LLMProvider.OPENAI, config)

        assert manager.default_configs[LLMProvider.OPENAI] == config

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
            await manager.generate(request, LLMProvider.OPENAI)


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_openai_provider_initialization(self):
        """Test OpenAIProvider initialization."""
        provider = OpenAIProvider("test_api_key")

        assert provider.api_key == "test_api_key"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.retry_config is not None
        assert provider.circuit_breaker is not None

    def test_get_available_models(self):
        """Test getting available OpenAI models."""
        provider = OpenAIProvider("test_api_key")
        models = provider.get_available_models()

        assert len(models) > 0
        assert any(model.model_name == "gpt-4o-mini" for model in models)
        assert any(model.model_name == "gpt-4o" for model in models)

    def test_calculate_cost(self):
        """Test cost calculation for OpenAI."""
        provider = OpenAIProvider("test_api_key")
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            model_size=ModelSize.SMALL,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006,
        )

        cost = provider.calculate_cost(1000, 500, config)
        expected_cost = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006

        assert cost == expected_cost

    @pytest.mark.asyncio
    async def test_openai_provider_session_management(self):
        """Test OpenAI provider session management."""
        provider = OpenAIProvider("test_api_key")

        # Test session creation
        session1 = await provider._get_session()
        assert session1 is not None

        # Test session reuse
        session2 = await provider._get_session()
        assert session1 is session2

        # Test session cleanup
        await provider.close()


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_anthropic_provider_initialization(self):
        """Test AnthropicProvider initialization."""
        provider = AnthropicProvider("test_api_key")

        assert provider.api_key == "test_api_key"
        assert provider.base_url == "https://api.anthropic.com/v1"
        assert provider.retry_config is not None
        assert provider.circuit_breaker is not None

    def test_get_available_models(self):
        """Test getting available Anthropic models."""
        provider = AnthropicProvider("test_api_key")
        models = provider.get_available_models()

        assert len(models) > 0
        assert any(model.model_name == "claude-3-haiku-20240307" for model in models)
        assert any(model.model_name == "claude-3-sonnet-20240229" for model in models)

    def test_calculate_cost(self):
        """Test cost calculation for Anthropic."""
        provider = AnthropicProvider("test_api_key")
        config = ModelConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            model_size=ModelSize.SMALL,
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125,
        )

        cost = provider.calculate_cost(1000, 500, config)
        expected_cost = (1000 / 1000) * 0.00025 + (500 / 1000) * 0.00125

        assert cost == expected_cost


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
class TestIntegration:
    """Integration tests for LLM provider system."""

    async def test_mock_llm_workflow(self):
        """Test complete LLM workflow with mocked responses."""
        # Create mock response
        mock_response = LLMResponse(
            content="The capital of France is Paris.",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            usage={"prompt_tokens": 10, "completion_tokens": 8},
            cost=0.001,
            response_time=0.5,
        )

        # Create mock provider
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = mock_response

        # Setup manager
        manager = LLMManager()
        manager.register_provider(LLMProvider.OPENAI, mock_provider)

        # Test request
        request = LLMRequest(user_prompt="What is the capital of France?")
        response = await manager.generate(request, LLMProvider.OPENAI)

        assert response.content == "The capital of France is Paris."
        assert response.provider == LLMProvider.OPENAI
        assert response.cost == 0.001

        # Check usage tracking
        assert len(manager.usage_stats) == 1
        stats_key = "openai:gpt-4o-mini"
        assert stats_key in manager.usage_stats
        assert manager.usage_stats[stats_key].total_requests == 1

    async def test_error_handling(self):
        """Test error handling in LLM workflow."""
        # Create mock provider that raises an error
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = LLMError("API Error", ErrorType.API_ERROR)

        # Setup manager
        manager = LLMManager()
        manager.register_provider(LLMProvider.OPENAI, mock_provider)

        # Test request that should fail
        request = LLMRequest(user_prompt="Test")

        with pytest.raises(LLMError):
            await manager.generate(request, LLMProvider.OPENAI)


# Fixtures for testing
@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o-mini",
        model_size=ModelSize.SMALL,
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
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 8},
        cost=0.001,
        response_time=0.5,
    )
