"""
Centralized model configuration registry for multi-provider support.

This module provides a single source of truth for model specifications,
pricing, and capabilities across all LLM providers. To add a new model,
simply add it to the _MODELS dictionary.

Usage:
    from .model_registry import get_model_spec, get_default_model, ProviderType

    # Get default model for a provider
    model_id = get_default_model(ProviderType.GEMINI)

    # Get model specification
    spec = get_model_spec("gemini-3-flash-preview")
    if spec:
        cost = spec.input_cost_per_1k * input_tokens / 1000
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Dict, Optional


class ProviderType(Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"
    # Future providers:
    # OPENAI = "openai"
    # ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model.

    Attributes:
        id: Model identifier (e.g., "gemini-3-flash-preview")
        provider: Provider type (GEMINI, OLLAMA, etc.)
        input_cost_per_1k: Cost per 1,000 input tokens in USD
        output_cost_per_1k: Cost per 1,000 output tokens in USD
        max_output_tokens: Maximum output tokens supported
        token_multiplier: Multiplier for reasoning models that use extra tokens
        supports_structured_output: Whether model supports JSON schema output
        supports_multimodal: Whether model supports images/documents
    """

    id: str
    provider: ProviderType
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_output_tokens: int = 8192
    token_multiplier: float = 1.0
    supports_structured_output: bool = True
    supports_multimodal: bool = False


# =============================================================================
# Model Registry
# =============================================================================
# Add new models here. The registry is immutable at runtime.

_MODELS: Dict[str, ModelSpec] = {
    # ---------------------------------------------------------------------
    # Gemini Models
    # ---------------------------------------------------------------------
    "gemini-3-flash-preview": ModelSpec(
        id="gemini-3-flash-preview",
        provider=ProviderType.GEMINI,
        input_cost_per_1k=0.0005,  # $0.50 per million tokens
        output_cost_per_1k=0.003,  # $3.00 per million tokens
        max_output_tokens=65536,  # 64k output context
        token_multiplier=3.0,  # Reasoning overhead
        supports_structured_output=True,
        supports_multimodal=True,
    ),
    # Legacy model - kept for reference and potential fallback
    "gemini-2.5-flash": ModelSpec(
        id="gemini-2.5-flash",
        provider=ProviderType.GEMINI,
        input_cost_per_1k=0.00030,  # $0.30 per million tokens
        output_cost_per_1k=0.0025,  # $2.50 per million tokens
        max_output_tokens=8192,
        token_multiplier=3.0,
        supports_structured_output=True,
        supports_multimodal=True,
    ),
    # ---------------------------------------------------------------------
    # Ollama Models (local, zero cost)
    # ---------------------------------------------------------------------
    "gemma3:12b": ModelSpec(
        id="gemma3:12b",
        provider=ProviderType.OLLAMA,
        input_cost_per_1k=0.0,  # Local inference
        output_cost_per_1k=0.0,
        max_output_tokens=8192,
        token_multiplier=1.0,
        supports_structured_output=True,
        supports_multimodal=True,
    ),
}

# Make the registry immutable to prevent accidental modifications
MODELS: MappingProxyType[str, ModelSpec] = MappingProxyType(_MODELS)

# Default models per provider - update these when changing default models
_DEFAULT_MODELS: Dict[ProviderType, str] = {
    ProviderType.GEMINI: "gemini-3-flash-preview",
    ProviderType.OLLAMA: "gemma3:12b",
}

DEFAULT_MODELS: MappingProxyType[ProviderType, str] = MappingProxyType(_DEFAULT_MODELS)


# =============================================================================
# Public API
# =============================================================================


def get_model_spec(model_id: str) -> Optional[ModelSpec]:
    """Get model specification by ID.

    Args:
        model_id: Model identifier (e.g., "gemini-3-flash-preview")

    Returns:
        ModelSpec if found, None otherwise
    """
    return MODELS.get(model_id)


def get_default_model(provider: ProviderType) -> str:
    """Get default model ID for a provider.

    Args:
        provider: Provider type

    Returns:
        Default model ID for the provider, empty string if not found
    """
    return DEFAULT_MODELS.get(provider, "")


def get_models_by_provider(provider: ProviderType) -> list[ModelSpec]:
    """Get all models for a specific provider.

    Args:
        provider: Provider type

    Returns:
        List of ModelSpec for the provider
    """
    return [spec for spec in MODELS.values() if spec.provider == provider]


# Export public API
__all__ = [
    "ModelSpec",
    "ProviderType",
    "MODELS",
    "DEFAULT_MODELS",
    "get_model_spec",
    "get_default_model",
    "get_models_by_provider",
]
