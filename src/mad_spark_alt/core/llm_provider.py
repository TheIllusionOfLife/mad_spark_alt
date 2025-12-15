"""
LLM Provider Abstraction for Multi-Agent Idea Generation.

This module provides a unified interface for working with Google Gemini API
with features for cost tracking, rate limiting, and async request handling.
"""

import asyncio
import copy
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import aiohttp
from pydantic import BaseModel, Field

from .retry import (
    CircuitBreaker,
    ErrorType,
    LLMError,
    RetryConfig,
    safe_aiohttp_request,
)
from .cost_utils import calculate_llm_cost_from_config, get_model_costs
from .system_constants import CONSTANTS

if TYPE_CHECKING:
    from .multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType, URLContextMetadata

# Import multimodal types and utilities at runtime
from .multimodal import MultimodalSourceType
from ..utils.multimodal_utils import read_file_as_base64

logger = logging.getLogger(__name__)

# Model constants
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Embedding constants
# Approximate ratio of tokens to words for estimation when API doesn't provide token count
TOKEN_ESTIMATION_FACTOR = 1.3
# Cost per 1K tokens for text-embedding-004 model (as of January 2025)
EMBEDDING_COST_PER_1K_TOKENS = 0.0002


def inline_schema_defs(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Inline all $defs references in a JSON Schema for Ollama compatibility.

    Ollama has known issues with $defs/$ref JSON Schema features:
    - Issue #8444: $defs ordering bug causes incorrect grammar generation
    - Issue #8462: Limited support for arrays and complex types

    This function transforms schemas like:
        { "$defs": {"X": {...}}, "properties": {"field": {"$ref": "#/$defs/X"}} }

    Into:
        { "properties": {"field": {...}} }

    Args:
        schema: JSON Schema dict (typically from Pydantic's model_json_schema())

    Returns:
        Transformed schema with all $ref inlined and $defs removed, or None if input is None
    """
    if not schema or "$defs" not in schema:
        return schema

    # Deep copy to avoid mutating original
    result = copy.deepcopy(schema)
    defs = result.pop("$defs", {})

    def resolve_ref(obj: Any) -> Any:
        """Recursively resolve $ref references."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Handle #/$defs/Name format
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        # Recursively resolve in case the definition has $refs too
                        return resolve_ref(copy.deepcopy(defs[def_name]))
                # Return original if we can't resolve
                return obj
            else:
                # Recursively process all values
                return {k: resolve_ref(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_ref(item) for item in obj]
        else:
            return obj

    resolved: Dict[str, Any] = resolve_ref(result)
    return resolved


class LLMProvider(Enum):
    """Supported LLM providers."""

    GOOGLE = "google"
    OLLAMA = "ollama"


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
    """
    Request structure for LLM calls with multimodal support.

    Attributes:
        system_prompt: Optional system-level instructions
        user_prompt: User's question or prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling threshold
        stop_sequences: Optional stop sequences
        model_configuration: Optional model-specific configuration
        response_schema: Optional JSON schema for structured output (Dict) OR Pydantic model (Type[BaseModel])
        response_mime_type: Optional MIME type for response (e.g., "application/json")

        # Multimodal fields (Phase 1 - Foundation)
        multimodal_inputs: Optional list of multimodal inputs (images, documents)
        urls: Optional list of URLs for context retrieval (max 20 for Gemini)
        tools: Optional provider-specific tools (e.g., Gemini's url_context)
    """

    system_prompt: Optional[str] = None
    user_prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None
    model_configuration: Optional[ModelConfig] = None
    # UPDATED: Accept either dict or Pydantic model for multi-provider compatibility
    response_schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None
    response_mime_type: Optional[str] = None

    # NEW: Multimodal support (Phase 1)
    multimodal_inputs: Optional[List["MultimodalInput"]] = None
    urls: Optional[List[str]] = None  # For URL context tool
    tools: Optional[List[Dict[str, Any]]] = None  # Provider-specific tools

    def get_json_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get JSON Schema dict from response_schema.

        Converts Pydantic models to standard JSON Schema using model_json_schema().
        Dict schemas are returned as-is for backward compatibility.

        Returns:
            JSON Schema dict, or None if response_schema is None

        Example:
            >>> from mad_spark_alt.core.schemas import DeductionResponse
            >>> request = LLMRequest(
            ...     user_prompt="Test",
            ...     response_schema=DeductionResponse,
            ...     response_mime_type="application/json"
            ... )
            >>> schema = request.get_json_schema()
            >>> print(schema["type"])  # "object"
        """
        if self.response_schema is None:
            return None

        # Check if it's a Pydantic model class
        if isinstance(self.response_schema, type) and issubclass(
            self.response_schema, BaseModel
        ):
            return self.response_schema.model_json_schema()

        # Otherwise, assume it's already a dict (backward compatibility)
        # Validate it's actually a dict to fail fast with clear error message
        if not isinstance(self.response_schema, dict):
            raise TypeError(
                f"response_schema must be a Pydantic BaseModel class or dict, "
                f"got {type(self.response_schema).__name__}"
            )
        return self.response_schema


class LLMResponse(BaseModel):
    """
    Response structure from LLM calls with multimodal metadata.

    Attributes:
        content: Generated text content
        provider: LLM provider used
        model: Model name used
        usage: Token usage statistics
        cost: Estimated cost in USD
        response_time: Response time in seconds
        metadata: Additional provider-specific metadata

        # Multimodal metadata (Phase 1 - Foundation)
        url_context_metadata: Optional metadata about URL retrieval status
        total_images_processed: Optional count of images processed
        total_pages_processed: Optional count of document pages processed
    """

    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    cost: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # NEW: Multimodal metadata (Phase 1)
    url_context_metadata: Optional[List["URLContextMetadata"]] = None
    total_images_processed: Optional[int] = None
    total_pages_processed: Optional[int] = None


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


def validate_llm_request(request: LLMRequest) -> None:
    """
    Validate LLMRequest constraints including multimodal inputs.

    Validates:
    - URL count (max 20 for Gemini)
    - Image count (max 3600 for Gemini)
    - Each MultimodalInput's internal validation

    Args:
        request: LLMRequest to validate

    Raises:
        ValueError: If request violates constraints

    Example:
        request = LLMRequest(user_prompt="Test", urls=[...])
        try:
            validate_llm_request(request)
        except ValueError as e:
            print(f"Validation failed: {e}")
    """
    # Validate URL count (Gemini: max 20 URLs)
    if request.urls and len(request.urls) > 20:
        raise ValueError(
            f"Too many URLs: {len(request.urls)} (max 20). "
            f"Gemini's url_context tool supports up to 20 URLs per request."
        )

    # Validate multimodal inputs if present
    if request.multimodal_inputs:
        from .multimodal import MultimodalInputType

        image_count = 0
        for input_item in request.multimodal_inputs:
            # This will raise ValueError if the individual input is invalid
            input_item.validate()

            if input_item.input_type == MultimodalInputType.IMAGE:
                image_count += 1

        # Gemini: Max 3600 images per request
        if image_count > 3600:
            raise ValueError(
                f"Too many images: {image_count} (max 3600). "
                f"Gemini supports up to 3600 images per request."
            )


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
            DEFAULT_GEMINI_MODEL
        )

        # Prepare the request payload
        url = f"{self.base_url}/models/{model_config.model_name}:generateContent"

        # Build contents with multimodal support
        contents = self._build_contents(request)

        # Check for URL context + structured output incompatibility
        # Gemini API does not support using url_context tool with structured output
        disable_structured_output = bool(
            request.urls and request.response_schema and request.response_mime_type
        )
        if disable_structured_output:
            logger.warning(
                "URL context tool is incompatible with structured output "
                "(Gemini API limitation). Disabling structured output and using "
                "text parsing fallback for this request."
            )

        # Adjust max_tokens for Gemini 2.5-flash reasoning overhead
        max_output_tokens = request.max_tokens
        if model_config.model_name == DEFAULT_GEMINI_MODEL:
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

        # Add structured output configuration if provided and not disabled
        if request.response_schema and request.response_mime_type and not disable_structured_output:
            generation_config["responseMimeType"] = request.response_mime_type
            # Use get_json_schema() to convert Pydantic models to JSON Schema
            generation_config["responseJsonSchema"] = request.get_json_schema()

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        # Add URL context tool if URLs provided
        self._add_url_context_tool(payload, request)

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
                timeout=CONSTANTS.TIMEOUTS.GEMINI_REQUEST_TIMEOUT,
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

        # Parse URL metadata from response
        url_metadata = self._parse_url_context_metadata(response_data)

        # Count multimodal inputs
        total_images = None
        total_pages = None
        if request.multimodal_inputs:
            # Import at runtime to avoid circular dependency
            from .multimodal import MultimodalInputType as MMInputType

            total_images = sum(
                1 for item in request.multimodal_inputs
                if item.input_type == MMInputType.IMAGE
            )
            total_pages = sum(
                item.page_count or 0 for item in request.multimodal_inputs
                if item.input_type == MMInputType.DOCUMENT
            )
            # Set to None if zero (for clean API)
            total_images = total_images if total_images > 0 else None
            total_pages = total_pages if total_pages > 0 else None

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
            url_context_metadata=url_metadata,
            total_images_processed=total_images,
            total_pages_processed=total_pages,
        )

    def _build_contents(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Build Gemini contents array from multimodal inputs.

        Order (per Gemini best practices):
        1. Multimodal inputs (images/documents) - if single item, before text
        2. Text prompt (system + user combined)
        3. URLs included in text if url_context tool is used

        Args:
            request: LLMRequest with potential multimodal inputs

        Returns:
            List with single dict: {"role": "user", "parts": [...]}

        Example:
            contents = self._build_contents(request)
            # [{"role": "user", "parts": [
            #     {"inline_data": {"mime_type": "image/png", "data": "..."}},
            #     {"text": "Analyze this image"}
            # ]}]
        """
        parts = []

        # Add multimodal inputs first (Gemini best practice)
        if request.multimodal_inputs:
            for item in request.multimodal_inputs:
                part = self._create_multimodal_part(item)
                parts.append(part)

        # Build text prompt (combine system + user)
        prompt_parts = []
        if request.system_prompt:
            prompt_parts.append(f"System: {request.system_prompt}")
        if request.user_prompt:
            prompt_parts.append(f"User: {request.user_prompt}")

        full_prompt = "\n\n".join(prompt_parts)

        # If URLs are provided, include them in the text for url_context tool
        if request.urls:
            urls_text = "\n\nRelevant URLs:\n" + "\n".join(f"- {url}" for url in request.urls)
            full_prompt += urls_text

        # Add text prompt as final part
        parts.append({"text": full_prompt})

        return [{"role": "user", "parts": parts}]

    def _create_multimodal_part(self, item: "MultimodalInput") -> Dict[str, Any]:
        """
        Create Gemini part from MultimodalInput.

        Handles different source types:
        - BASE64: Use inline_data format
        - FILE_PATH: Read file, convert to base64, use inline_data
        - FILE_API: Use file_data format with file_uri
        - URL: Use inline_data with URL (Gemini fetches)

        Args:
            item: MultimodalInput to convert

        Returns:
            Gemini part dict (inline_data or file_data format)

        Raises:
            ValueError: If source_type is unsupported

        Example:
            part = self._create_multimodal_part(image_input)
            # {"inline_data": {"mime_type": "image/png", "data": "base64..."}}
        """
        if item.source_type == MultimodalSourceType.BASE64:
            return {
                "inline_data": {
                    "mime_type": item.mime_type,
                    "data": item.data
                }
            }
        elif item.source_type == MultimodalSourceType.FILE_PATH:
            # Read file and convert to base64
            base64_data, mime_type = read_file_as_base64(Path(item.data))
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data
                }
            }
        elif item.source_type == MultimodalSourceType.FILE_API:
            # Reference uploaded file via File API
            return {
                "file_data": {
                    "file_uri": item.data,
                    "mime_type": item.mime_type
                }
            }
        else:
            raise ValueError(
                f"Unsupported source type: {item.source_type}. "
                f"Note: URL resources should be passed via request.urls parameter "
                f"to use Gemini's url_context tool, not as MultimodalInput."
            )

    def _add_url_context_tool(self, payload: Dict[str, Any], request: LLMRequest) -> None:
        """
        Add url_context tool to payload if URLs are provided.

        Modifies payload in-place to add tools array with url_context.

        Args:
            payload: Gemini API payload dict (modified in-place)
            request: LLMRequest with potential URLs

        Example:
            payload = {"contents": [...], "generationConfig": {...}}
            self._add_url_context_tool(payload, request)
            # payload now has: {"tools": [{"url_context": {}}], ...}
        """
        if request.urls:
            payload["tools"] = [{"url_context": {}}]

    def _parse_url_context_metadata(
        self,
        response_data: Dict[str, Any]
    ) -> Optional[List["URLContextMetadata"]]:
        """
        Parse url_context_metadata from Gemini response.

        Args:
            response_data: Raw Gemini API response dict

        Returns:
            List of URLContextMetadata objects, or None if no metadata

        Example:
            metadata = self._parse_url_context_metadata(response)
            # [URLContextMetadata(url="...", status="success"), ...]
        """
        if "url_context_metadata" not in response_data:
            return None

        # Import at runtime to avoid circular dependency
        from .multimodal import URLContextMetadata as URLMeta

        return [
            URLMeta(
                url=meta["url"],
                status=meta["status"],
                error_message=meta.get("error_message")
            )
            for meta in response_data["url_context_metadata"]
        ]

    def get_available_models(self) -> List[ModelConfig]:
        """Get available Google models.
        
        Note: Pricing is centralized in cost_utils.py. Update pricing there.
        """
        # Get pricing from centralized cost_utils module
        model_name = DEFAULT_GEMINI_MODEL
        model_costs = get_model_costs(model_name)
        if not model_costs:
            raise ValueError(f"'{model_name}' costs not configured in cost_utils")
        
        return [
            ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name=model_name,
                model_size=ModelSize.LARGE,
                input_cost_per_1k=model_costs.input_cost_per_1k_tokens,
                output_cost_per_1k=model_costs.output_cost_per_1k_tokens,
                max_tokens=8192,
            ),
        ]

    def _get_default_model_config(self, model_name: str) -> ModelConfig:
        """Get default model config by name."""
        models = {model.model_name: model for model in self.get_available_models()}
        # Always use gemini-2.5-flash
        return models.get(model_name, models[DEFAULT_GEMINI_MODEL])

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

    async def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Get embeddings for texts using Google's embedding API.
        
        Args:
            request: Embedding request with texts and configuration
            
        Returns:
            EmbeddingResponse with embedding vectors
        """
        session = await self._get_session()
        
        # Extract model name without "models/" prefix if present
        model_name = request.model
        if model_name.startswith("models/"):
            model_name = model_name[7:]  # Remove "models/" prefix
            
        url = f"{self.base_url}/models/{model_name}:batchEmbedContents"
        
        # Prepare batch request
        batch_requests = []
        for text in request.texts:
            embed_request = {
                "model": f"models/{model_name}",
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": request.task_type,
                "outputDimensionality": request.output_dimensionality
            }
            if request.title:
                embed_request["title"] = request.title
            batch_requests.append(embed_request)
        
        payload = {
            "requests": batch_requests
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        
        start_time = time.time()
        
        # Use the retry mechanism - it returns JSON data directly
        response_json = await safe_aiohttp_request(
            session=session,
            method="POST",
            url=url,
            headers=headers,
            json=payload,
            timeout=CONSTANTS.TIMEOUTS.GEMINI_REQUEST_TIMEOUT,
            retry_config=self.retry_config,
            circuit_breaker=self.circuit_breaker,
        )
        
        response_time = time.time() - start_time
        
        # Extract embeddings
        embeddings = []
        for embedding_obj in response_json.get("embeddings", []):
            embeddings.append(embedding_obj["values"])
            
        # Get token usage from response if available
        usage_metadata = response_json.get("usageMetadata", {})
        if usage_metadata.get("totalTokenCount"):
            total_tokens = usage_metadata["totalTokenCount"]
        else:
            # Estimate tokens based on text length
            total_tokens = sum(len(text.split()) * TOKEN_ESTIMATION_FACTOR for text in request.texts)
        
        # Embedding costs are typically much lower than generation
        # Using configured cost for text-embedding-004 model
        cost = (total_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            usage={
                "total_tokens": int(total_tokens),
                "prompt_tokens": int(total_tokens),  # All tokens are input for embeddings
            },
            cost=cost,
            metadata={
                "response_time": response_time,
                "num_texts": len(request.texts),
                "task_type": request.task_type,
                "dimensions": request.output_dimensionality
            }
        )

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


class OllamaProvider(LLMProviderInterface):
    """Ollama local LLM provider implementation with structured output support."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (default from CONSTANTS.LLM.OLLAMA_DEFAULT_MODEL)
            base_url: Ollama API base URL (default from CONSTANTS.LLM.OLLAMA_DEFAULT_BASE_URL)
            retry_config: Optional retry configuration
        """
        self.model = model or CONSTANTS.LLM.OLLAMA_DEFAULT_MODEL
        self.base_url = base_url or CONSTANTS.LLM.OLLAMA_DEFAULT_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()  # Protect concurrent session creation
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session (thread-safe)."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    async def _generate_with_outlines(
        self, request: LLMRequest, pydantic_model: Type[BaseModel]
    ) -> LLMResponse:
        """
        Generate structured output using Outlines library.

        Outlines provides better structured output support for Ollama by using
        constrained grammar generation at the token level.

        Args:
            request: LLMRequest with prompt
            pydantic_model: Pydantic model class for output schema

        Returns:
            LLMResponse with generated content matching the schema
        """
        try:
            import ollama
            import outlines
        except ImportError as e:
            raise LLMError(
                f"Outlines-based generation requires 'outlines' and 'ollama' packages. "
                f"Install with: pip install 'outlines[ollama]>=1.0.0' ollama. Error: {e}",
                ErrorType.API_ERROR,
            ) from e

        start_time = time.time()

        try:
            # Create Outlines model with async Ollama client
            # Use host parameter to set base URL
            async_client = ollama.AsyncClient(host=self.base_url)
            outlines_model = outlines.from_ollama(async_client, self.model)

            # Build prompt from request
            prompt = request.user_prompt
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{prompt}"

            # Generate with Outlines - pass Pydantic model as output_type
            # Ollama expects options in 'options' dict, not as direct kwargs
            logger.debug(f"Using Outlines for structured output with model: {pydantic_model.__name__}")
            result_obj = await outlines_model(
                prompt,
                pydantic_model,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "top_p": request.top_p,
                },
            )

            # Serialize Pydantic model to JSON string (handle both Pydantic model and string)
            if hasattr(result_obj, 'model_dump_json'):
                result_json = result_obj.model_dump_json()
            else:
                # Already a string or other type
                result_json = str(result_obj)

            end_time = time.time()

            # Estimate tokens (Outlines doesn't provide token counts directly)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(result_json) // 4
            total_tokens = prompt_tokens + completion_tokens

            return LLMResponse(
                content=result_json,
                provider=LLMProvider.OLLAMA,
                model=self.model,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                cost=0.0,  # Local model, no cost
                metadata={
                    "response_time": end_time - start_time,
                    "method": "outlines",
                },
            )

        except Exception as e:
            logger.warning(f"Outlines generation failed: {e}, falling back to raw API")
            raise  # Re-raise to trigger fallback

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using Ollama API with Pydantic schema support.

        Supports:
        - Text generation
        - Image inputs (via base64 encoding)
        - Structured output via Outlines library (for Pydantic models)
        - Structured output via JSON schema format parameter (fallback)

        When a Pydantic model is provided as response_schema, this method
        attempts to use the Outlines library for better constrained generation.
        If Outlines fails, it falls back to Ollama's native JSON schema support.

        Args:
            request: LLMRequest with prompt and optional schema

        Returns:
            LLMResponse with generated content

        Raises:
            LLMError: If Ollama API request fails
        """
        # Try Outlines for Pydantic model schemas (better structured output)
        if request.response_schema is not None:
            # Check if it's a Pydantic model class
            if isinstance(request.response_schema, type) and issubclass(
                request.response_schema, BaseModel
            ):
                try:
                    return await self._generate_with_outlines(
                        request, request.response_schema
                    )
                except Exception as e:
                    logger.warning(
                        f"Outlines generation failed ({e}), falling back to raw Ollama API"
                    )
                    # Fall through to raw API

        session = await self._get_session()

        # Build messages in Ollama format
        messages = self._build_messages(request)

        # Build payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
            },
        }

        # Add structured output schema if provided
        if request.response_schema:
            schema = request.get_json_schema()
            # Inline $defs for Ollama compatibility (Issue #8444, #8462)
            schema = inline_schema_defs(schema)
            payload["format"] = schema  # Ollama's format parameter
            # User controls temperature directly - no automatic capping
            # High temperature may affect schema compliance, but user has full control
            logger.debug(
                f"Using temperature {request.temperature} for structured output"
            )
            schema_type = schema.get("type", "unknown") if schema else "unknown"
            logger.debug(f"Using structured output with schema: {schema_type}")

        # Make request to Ollama API
        url = f"{self.base_url}/api/chat"
        start_time = time.time()

        try:
            response_data = await safe_aiohttp_request(
                session=session,
                method="POST",
                url=url,
                json=payload,
                retry_config=self.retry_config,
                circuit_breaker=self.circuit_breaker,
                timeout=CONSTANTS.TIMEOUTS.OLLAMA_INFERENCE_TIMEOUT,
            )
        except Exception as e:
            raise LLMError(
                f"Ollama API request failed: {str(e)}. "
                f"Ensure Ollama is running (ollama serve) and model is available (ollama pull {self.model})",
                ErrorType.API_ERROR,
            ) from e

        end_time = time.time()

        # Extract generated content
        try:
            content = response_data["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Invalid response format from Ollama API: {e}",
                ErrorType.API_ERROR,
            ) from e

        # Extract token counts from Ollama API response
        # Ollama provides: prompt_eval_count (input) and eval_count (output)
        # Fall back to character-based estimation if not available
        if "prompt_eval_count" in response_data and "eval_count" in response_data:
            prompt_tokens = response_data["prompt_eval_count"]
            completion_tokens = response_data["eval_count"]
            logger.debug(
                f"Ollama token counts from API: prompt={prompt_tokens}, completion={completion_tokens}"
            )
        else:
            # Fallback: estimate using character count (1 token ≈ 4 chars)
            prompt_text = request.user_prompt
            if request.system_prompt:
                prompt_text = f"{request.system_prompt}\n{prompt_text}"
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(content) // 4
            logger.warning(
                "Ollama response missing token counts, using character-based estimation"
            )
        total_tokens = prompt_tokens + completion_tokens

        # Count multimodal inputs
        total_images = None
        if request.multimodal_inputs:
            from .multimodal import MultimodalInputType as MMInputType

            total_images = sum(
                1 for item in request.multimodal_inputs
                if item.input_type == MMInputType.IMAGE
            )
            total_images = total_images if total_images > 0 else None

        return LLMResponse(
            content=content,
            provider=LLMProvider.OLLAMA,
            model=self.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost=0.0,  # Ollama is free (local inference)
            response_time=end_time - start_time,
            total_images_processed=total_images,
        )

    def _build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Build Ollama messages format.

        Ollama messages format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "...", "images": ["base64..."]}
        ]

        Args:
            request: LLMRequest with prompts and optional multimodal inputs

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Build user message with optional images
        user_message: Dict[str, Any] = {
            "role": "user",
            "content": request.user_prompt,
        }

        # Handle multimodal inputs (images)
        if request.multimodal_inputs:
            images = []
            for item in request.multimodal_inputs:
                from .multimodal import MultimodalInputType, MultimodalSourceType

                if item.input_type == MultimodalInputType.IMAGE:
                    if item.source_type == MultimodalSourceType.FILE_PATH:
                        # Normalize path to absolute for Ollama compatibility
                        # Ollama requires absolute paths or "./" prefix for images
                        original_path = Path(item.data)
                        image_path = original_path.resolve()

                        # Security: Prevent path traversal attacks on relative paths
                        # Only validate relative paths - absolute paths are user's responsibility
                        if not original_path.is_absolute():
                            try:
                                image_path.relative_to(Path.cwd())
                            except ValueError:
                                raise ValueError(
                                    f"Relative path '{item.data}' resolves outside project directory. "
                                    f"Resolved to: {image_path}"
                                ) from None

                        # Read file and convert to base64
                        base64_data, _ = read_file_as_base64(image_path)
                        images.append(base64_data)
                    elif item.source_type == MultimodalSourceType.BASE64:
                        # Already base64 encoded
                        images.append(item.data)
                    else:
                        logger.warning(
                            f"Unsupported multimodal source type for Ollama: {item.source_type}. "
                            f"Ollama supports FILE_PATH and BASE64 only."
                        )

            if images:
                user_message["images"] = images

        messages.append(user_message)

        return messages

    def get_available_models(self) -> List[ModelConfig]:
        """
        Get available Ollama models.

        Returns list with current model configuration.
        Ollama is free (local inference), so costs are $0.00.
        """
        return [
            ModelConfig(
                provider=LLMProvider.OLLAMA,
                model_name=self.model,
                model_size=ModelSize.MEDIUM,
                input_cost_per_1k=0.0,  # Free (local)
                output_cost_per_1k=0.0,  # Free (local)
                max_tokens=CONSTANTS.LLM.OLLAMA_DEFAULT_MAX_TOKENS,
                temperature=0.7,
                top_p=0.9,
            )
        ]

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model_config: ModelConfig
    ) -> float:
        """
        Calculate cost for Ollama usage.

        Ollama runs locally, so cost is always $0.00.
        (Hardware costs not tracked here)

        Returns:
            0.0 (Ollama is free)
        """
        return 0.0

    async def close(self) -> None:
        """Close the session (thread-safe with respect to _get_session)."""
        async with self._session_lock:
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


# Rebuild Pydantic models after all imports to resolve forward references
# This is needed for TYPE_CHECKING imports of MultimodalInput and URLContextMetadata
try:
    from .multimodal import MultimodalInput, URLContextMetadata
    LLMRequest.model_rebuild()
    LLMResponse.model_rebuild()
except ImportError:
    # multimodal module not available (shouldn't happen, but being defensive)
    pass
