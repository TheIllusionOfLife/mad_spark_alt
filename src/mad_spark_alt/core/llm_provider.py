"""
LLM Provider Abstraction for Multi-Agent Idea Generation.

This module provides a unified interface for working with different LLM providers
(OpenAI, Anthropic, Google) with features for cost tracking, rate limiting,
and async request handling.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from pydantic import BaseModel, Field

from .retry import (
    CircuitBreaker,
    ErrorType,
    LLMError,
    RetryConfig,
    safe_aiohttp_request,
)

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelSize(Enum):
    """Model size categories for cost estimation."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class UsageStats:
    """Track LLM usage statistics."""

    provider: LLMProvider
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    last_request: Optional[datetime] = None

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Add usage statistics."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_requests += 1
        self.total_cost += cost
        self.last_request = datetime.now(timezone.utc)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    max_concurrent_requests: int = 10


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: LLMProvider
    model_name: str
    model_size: ModelSize
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class LLMRequest(BaseModel):
    """Request structure for LLM calls."""

    system_prompt: Optional[str] = None
    user_prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None


class LLMResponse(BaseModel):
    """Response structure from LLM calls."""

    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    cost: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using the LLM."""
        pass

    @abstractmethod
    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate the cost for a request."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models."""
        pass


class RateLimiter:
    """Rate limiter for LLM requests."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []  # (timestamp, tokens)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire rate limit permission."""
        await self._semaphore.acquire()

        # Loop to wait for rate limit window to pass
        while True:
            now = time.time()
            minute_ago = now - 60

            # Clean old entries on each iteration
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [
                (t, tokens) for t, tokens in self.token_usage if t > minute_ago
            ]

            # Calculate sleep times for both limits
            request_sleep = 0.0
            if len(self.request_times) >= self.config.requests_per_minute:
                request_sleep = 60 - (now - self.request_times[0])

            token_sleep = 0.0
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.config.tokens_per_minute:
                if self.token_usage:  # Only if we have token usage data
                    token_sleep = 60 - (now - self.token_usage[0][0])

            # Sleep for the maximum required time, or break if no sleep needed
            max_sleep = max(request_sleep, token_sleep)
            if max_sleep > 0:
                await asyncio.sleep(max_sleep)
                # Continue loop to re-evaluate with fresh timestamps
            else:
                break

        # Record this request with current timestamp
        final_now = time.time()
        self.request_times.append(final_now)
        self.token_usage.append((final_now, estimated_tokens))

    def release(self) -> None:
        """Release rate limit permission."""
        self._semaphore.release()


class OpenAIProvider(LLMProviderInterface):
    """OpenAI API provider implementation."""

    def __init__(self, api_key: str, retry_config: Optional[RetryConfig] = None):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self._session: Optional[aiohttp.ClientSession] = None
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using OpenAI API."""
        session = await self._get_session()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        model_name = (
            request.model_configuration.model_name
            if request.model_configuration
            else "gpt-4o-mini"
        )

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        start_time = time.time()

        try:
            data = await safe_aiohttp_request(
                session=session,
                method="POST",
                url=f"{self.base_url}/chat/completions",
                json=payload,
                retry_config=self.retry_config,
                circuit_breaker=self.circuit_breaker,
                timeout=300,  # 5 minutes timeout for complex questions
            )

            response_time = time.time() - start_time

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            model_config = (
                request.model_configuration
                or self._get_default_model_config(model_name)
            )
            cost = self.calculate_cost(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                model_config,
            )

            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=model_name,
                usage=usage,
                cost=cost,
                response_time=response_time,
                metadata={"choices": len(data["choices"])},
            )

        except LLMError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI request: {e}")
            raise LLMError(f"OpenAI request failed: {str(e)}", ErrorType.UNKNOWN)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost for OpenAI request."""
        input_cost = (input_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_config.output_cost_per_1k
        return input_cost + output_cost

    def get_available_models(self) -> List[ModelConfig]:
        """Get available OpenAI models."""
        return [
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.005,
                output_cost_per_1k=0.015,
                max_tokens=128000,
            ),
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                model_size=ModelSize.SMALL,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                max_tokens=128000,
            ),
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.03,
                max_tokens=128000,
            ),
        ]

    def _get_default_model_config(self, model_name: str) -> ModelConfig:
        """Get default model config by name."""
        models = {model.model_name: model for model in self.get_available_models()}
        return models.get(model_name, models["gpt-4o-mini"])

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class AnthropicProvider(LLMProviderInterface):
    """Anthropic API provider implementation."""

    def __init__(self, api_key: str, retry_config: Optional[RetryConfig] = None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self._session: Optional[aiohttp.ClientSession] = None
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using Anthropic API."""
        session = await self._get_session()

        model_name = (
            request.model_configuration.model_name
            if request.model_configuration
            else "claude-3-haiku-20240307"
        )

        payload = {
            "model": model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "messages": [{"role": "user", "content": request.user_prompt}],
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        start_time = time.time()

        try:
            data = await safe_aiohttp_request(
                session=session,
                method="POST",
                url=f"{self.base_url}/messages",
                json=payload,
                retry_config=self.retry_config,
                circuit_breaker=self.circuit_breaker,
                timeout=300,  # 5 minutes timeout for complex questions
            )

            response_time = time.time() - start_time

            content = data["content"][0]["text"]
            usage = data.get("usage", {})

            model_config = (
                request.model_configuration
                or self._get_default_model_config(model_name)
            )
            cost = self.calculate_cost(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model_config,
            )

            return LLMResponse(
                content=content,
                provider=LLMProvider.ANTHROPIC,
                model=model_name,
                usage=usage,
                cost=cost,
                response_time=response_time,
                metadata={"stop_reason": data.get("stop_reason")},
            )

        except LLMError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Anthropic request: {e}")
            raise LLMError(f"Anthropic request failed: {str(e)}", ErrorType.UNKNOWN)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost for Anthropic request."""
        input_cost = (input_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_config.output_cost_per_1k
        return input_cost + output_cost

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Anthropic models."""
        return [
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-opus-20240229",
                model_size=ModelSize.XLARGE,
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                max_tokens=200000,
            ),
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                max_tokens=200000,
            ),
            ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                model_size=ModelSize.SMALL,
                input_cost_per_1k=0.00025,
                output_cost_per_1k=0.00125,
                max_tokens=200000,
            ),
        ]

    def _get_default_model_config(self, model_name: str) -> ModelConfig:
        """Get default model config by name."""
        models = {model.model_name: model for model in self.get_available_models()}
        return models.get(model_name, models["claude-3-haiku-20240307"])

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class GoogleProvider(LLMProviderInterface):
    """Google Gemini API provider implementation."""

    def __init__(self, api_key: str, retry_config: Optional[RetryConfig] = None):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._session: Optional[aiohttp.ClientSession] = None
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using Google Gemini API."""
        session = await self._get_session()

        # Get model config - use latest 2.5 model by default
        default_model = os.getenv("GEMINI_MODEL_OVERRIDE", "gemini-2.5-flash")
        model_config = request.model_configuration or self._get_default_model_config(
            default_model
        )

        # Prepare the request payload
        url = f"{self.base_url}/models/{model_config.model_name}:generateContent"

        # Build the prompt from system and user prompts
        prompt_parts = []
        if request.system_prompt:
            prompt_parts.append(f"System: {request.system_prompt}")
        if request.user_prompt:
            prompt_parts.append(f"User: {request.user_prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
        }

        params = {"key": self.api_key}

        headers = {"Content-Type": "application/json"}

        start_time = time.time()

        try:
            response_data = await safe_aiohttp_request(
                session=session,
                method="POST",
                url=url,
                json=payload,
                params=params,
                headers=headers,
                retry_config=self.retry_config,
                circuit_breaker=self.circuit_breaker,
                timeout=300,  # 5 minutes timeout for complex questions
            )
        except Exception as e:
            raise LLMError(f"Google API request failed: {str(e)}", ErrorType.API_ERROR)

        end_time = time.time()

        # Extract the generated content
        try:
            content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise LLMError(
                "Invalid response format from Google API", ErrorType.API_ERROR
            )

        # Extract usage information
        usage_metadata = response_data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        )

        # Calculate cost based on model pricing
        input_cost = (prompt_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (completion_tokens / 1000) * model_config.output_cost_per_1k
        total_cost = input_cost + output_cost

        return LLMResponse(
            content=content,
            provider=LLMProvider.GOOGLE,
            model=model_config.model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost=total_cost,
            response_time=end_time - start_time,
        )

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Google models."""
        return [
            ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-2.5-flash",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.00001,  # Very low cost for Gemini 2.5
                output_cost_per_1k=0.00002,
                max_tokens=8192,
            ),
            ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-2.0-flash",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.00001,  # Very low cost for Gemini
                output_cost_per_1k=0.00002,
                max_tokens=8192,
            ),
            ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-1.5-flash",
                model_size=ModelSize.MEDIUM,
                input_cost_per_1k=0.000075,
                output_cost_per_1k=0.0003,
                max_tokens=8192,
            ),
            ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-1.5-pro",
                model_size=ModelSize.LARGE,
                input_cost_per_1k=0.00125,
                output_cost_per_1k=0.005,
                max_tokens=8192,
            ),
        ]

    def _get_default_model_config(self, model_name: str) -> ModelConfig:
        """Get default model config by name."""
        models = {model.model_name: model for model in self.get_available_models()}
        # Use gemini-1.5-flash as fallback for better stability
        fallback = models.get("gemini-1.5-flash", list(models.values())[0])
        return models.get(model_name, fallback)

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost based on token usage and model pricing."""
        input_cost = (prompt_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (completion_tokens / 1000) * model_config.output_cost_per_1k
        return input_cost + output_cost

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class LLMManager:
    """Central manager for LLM providers with cost tracking and rate limiting."""

    def __init__(self) -> None:
        self.providers: Dict[LLMProvider, LLMProviderInterface] = {}
        self.usage_stats: Dict[str, UsageStats] = {}  # key: provider:model
        self.rate_limiters: Dict[LLMProvider, RateLimiter] = {}
        self.default_configs: Dict[LLMProvider, ModelConfig] = {}

    def register_provider(
        self,
        provider: LLMProvider,
        instance: LLMProviderInterface,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ) -> None:
        """Register an LLM provider."""
        self.providers[provider] = instance

        if rate_limit_config:
            self.rate_limiters[provider] = RateLimiter(rate_limit_config)
        else:
            # Default rate limits
            default_config = RateLimitConfig()
            self.rate_limiters[provider] = RateLimiter(default_config)

    def set_default_model(
        self, provider: LLMProvider, model_config: ModelConfig
    ) -> None:
        """Set default model configuration for a provider."""
        self.default_configs[provider] = model_config

    async def generate(
        self, request: LLMRequest, provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """Generate text using the specified or default provider."""
        if provider is None:
            # Use first available provider
            if not self.providers:
                raise ValueError("No LLM providers registered")
            provider = next(iter(self.providers.keys()))

        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not registered")

        provider_instance = self.providers[provider]
        rate_limiter = self.rate_limiters.get(provider)

        # Apply rate limiting
        if rate_limiter:
            estimated_tokens = len(request.user_prompt) // 4  # Rough estimation
            await rate_limiter.acquire(estimated_tokens)

        try:
            # Use default model config if not provided
            if not request.model_configuration and provider in self.default_configs:
                request.model_configuration = self.default_configs[provider]

            response = await provider_instance.generate(request)

            # Track usage
            stats_key = f"{provider.value}:{response.model}"
            if stats_key not in self.usage_stats:
                self.usage_stats[stats_key] = UsageStats(provider, response.model)

            self.usage_stats[stats_key].add_usage(
                response.usage.get("prompt_tokens", 0)
                or response.usage.get("input_tokens", 0),
                response.usage.get("completion_tokens", 0)
                or response.usage.get("output_tokens", 0),
                response.cost,
            )

            return response

        finally:
            if rate_limiter:
                rate_limiter.release()

    def get_usage_stats(self) -> Dict[str, UsageStats]:
        """Get usage statistics for all providers and models."""
        return self.usage_stats.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all providers and models."""
        return sum(stats.total_cost for stats in self.usage_stats.values())

    async def close(self) -> None:
        """Close all provider sessions."""
        for provider in self.providers.values():
            if hasattr(provider, "close"):
                await provider.close()


# Global LLM manager instance
llm_manager = LLMManager()


async def setup_llm_providers(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
) -> LLMManager:
    """Setup LLM providers with API keys."""
    if openai_api_key:
        openai_provider = OpenAIProvider(openai_api_key)
        llm_manager.register_provider(
            LLMProvider.OPENAI, openai_provider, rate_limit_config
        )

        # Set default model for OpenAI
        default_models = openai_provider.get_available_models()
        default_model = next(
            (m for m in default_models if m.model_name == "gpt-4o-mini"),
            default_models[0],  # Fallback to first model if gpt-4o-mini not found
        )
        llm_manager.set_default_model(LLMProvider.OPENAI, default_model)

    if anthropic_api_key:
        anthropic_provider = AnthropicProvider(anthropic_api_key)
        llm_manager.register_provider(
            LLMProvider.ANTHROPIC, anthropic_provider, rate_limit_config
        )

        # Set default model for Anthropic
        default_models = anthropic_provider.get_available_models()
        default_model = next(
            (m for m in default_models if m.model_name == "claude-3-haiku-20240307"),
            default_models[0],  # Fallback to first model if claude-3-haiku not found
        )
        llm_manager.set_default_model(LLMProvider.ANTHROPIC, default_model)

    if google_api_key:
        google_provider = GoogleProvider(google_api_key)
        llm_manager.register_provider(
            LLMProvider.GOOGLE, google_provider, rate_limit_config
        )

        # Set default model for Google - use stable 1.5 model
        default_models = google_provider.get_available_models()
        preferred_model = os.getenv("GEMINI_MODEL_OVERRIDE", "gemini-1.5-flash")
        default_model = next(
            (m for m in default_models if m.model_name == preferred_model),
            next(
                (m for m in default_models if m.model_name == "gemini-1.5-flash"),
                default_models[0],  # Fallback to first model if neither found
            )
        )
        llm_manager.set_default_model(LLMProvider.GOOGLE, default_model)

    return llm_manager
