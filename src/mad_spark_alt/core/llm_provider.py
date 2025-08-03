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
from .cost_utils import calculate_llm_cost_from_config

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

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
    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None


class LLMResponse(BaseModel):
    """Response structure from LLM calls."""

    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    cost: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request structure for embedding calls."""
    
    texts: List[str]
    model: str = "models/text-embedding-004"
    task_type: str = "SEMANTIC_SIMILARITY"
    output_dimensionality: int = 768
    title: Optional[str] = None  # Optional title for better quality


class EmbeddingResponse(BaseModel):
    """Response structure from embedding calls."""
    
    embeddings: List[List[float]]  # List of embedding vectors
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    cost: float = 0.0
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
    
    async def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Get embeddings for texts (optional - not all providers support this).
        
        Args:
            request: Embedding request with texts and configuration
            
        Returns:
            EmbeddingResponse with embedding vectors
            
        Raises:
            NotImplementedError: If provider doesn't support embeddings
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support embeddings")


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

        # Always use Gemini 2.5 Flash
        model_config = request.model_configuration or self._get_default_model_config(
            "gemini-2.5-flash"
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

        # Adjust max_tokens for Gemini 2.5-flash reasoning overhead
        max_output_tokens = request.max_tokens
        if model_config.model_name == "gemini-2.5-flash":
            # 2.5-flash uses many tokens for internal reasoning, so increase output limit
            max_output_tokens = max(
                request.max_tokens * 3, 2048
            )  # At least 3x the requested tokens

        # Build generation config
        generation_config: Dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.95,
            "topK": 40,
        }
        
        # Add structured output configuration if provided
        if request.response_schema and request.response_mime_type:
            generation_config["responseMimeType"] = request.response_mime_type
            generation_config["responseSchema"] = request.response_schema
        
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": generation_config,
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
            candidate = response_data["candidates"][0]
            content_data = candidate["content"]

            # Handle cases where parts might not exist (e.g., MAX_TOKENS with no content)
            if content_data.get(
                "parts"
            ):  # More Pythonic check for existing, non-empty list
                content = content_data["parts"][0]["text"]
            else:
                # Fallback for empty content due to finish reasons like MAX_TOKENS
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                reason_messages = {
                    "MAX_TOKENS": "[Content generation stopped due to token limit - try reducing max_tokens or prompt length]",
                    "SAFETY": "[Content blocked by safety filters]",
                    "RECITATION": "[Content blocked due to recitation concerns]",
                }
                content = reason_messages.get(
                    finish_reason,
                    f"[No content generated - finish reason: {finish_reason}]",
                )
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Invalid response format from Google API: {e}",
                ErrorType.API_ERROR,
            ) from e

        # Extract usage information
        usage_metadata = response_data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        )

        # Calculate cost based on model pricing
        total_cost = self.calculate_cost(prompt_tokens, completion_tokens, model_config)

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
                input_cost_per_1k=0.00015,  # $0.15 per million tokens
                output_cost_per_1k=0.0006,  # $0.60 per million tokens
                max_tokens=8192,
            ),
        ]

    def _get_default_model_config(self, model_name: str) -> ModelConfig:
        """Get default model config by name."""
        models = {model.model_name: model for model in self.get_available_models()}
        # Always use gemini-2.5-flash
        return models.get(model_name, models["gemini-2.5-flash"])

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """Calculate cost based on token usage and model pricing."""
        # Use centralized cost calculation with ModelConfig costs directly
        return calculate_llm_cost_from_config(
            input_tokens,
            output_tokens,
            model_config.input_cost_per_1k,
            model_config.output_cost_per_1k,
        )

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
    google_api_key: str,
    rate_limit_config: Optional[RateLimitConfig] = None,
) -> LLMManager:
    """Setup Google LLM provider with API key."""
    google_provider = GoogleProvider(google_api_key)
    llm_manager.register_provider(
        LLMProvider.GOOGLE, google_provider, rate_limit_config
    )

    # Set default model for Google - always use Gemini 2.5 Flash
    default_models = google_provider.get_available_models()
    if not default_models:
        raise RuntimeError("No available models found for the Google provider.")
    default_model = default_models[0]  # Only one model available
    llm_manager.set_default_model(LLMProvider.GOOGLE, default_model)

    return llm_manager


def get_google_provider() -> Optional[Any]:
    """
    Get Google LLM provider if available.
    
    Returns:
        Google provider instance or None if not available
    """
    if LLMProvider.GOOGLE in llm_manager.providers:
        return llm_manager.providers[LLMProvider.GOOGLE]
    return None
