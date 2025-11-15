"""
Provider Router for Multi-LLM Support.

This module provides intelligent routing between LLM providers (Gemini, Ollama)
with automatic selection based on input type and graceful fallback mechanisms.

Key Features:
- Auto-selection: Route to appropriate provider based on input complexity
- Fallback: Ollama failures automatically retry with Gemini
- Validation: Prevent invalid provider/input combinations
- Logging: Track provider selection decisions for debugging

Design:
Provider selection follows a clear hierarchy:
1. Explicit --provider flag (user override)
2. Input requirements (documents/URLs → Gemini)
3. Default strategy (text/images → Ollama, fallback to Gemini)
"""

import logging
from enum import Enum
from typing import Optional, Tuple

from .llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMProviderInterface,
    LLMRequest,
    LLMResponse,
    OllamaProvider,
)
from .retry import ErrorType, LLMError

logger = logging.getLogger(__name__)


class ProviderSelection(Enum):
    """Provider selection strategy."""

    AUTO = "auto"  # Automatic selection based on input
    GEMINI = "gemini"  # Force Gemini API
    OLLAMA = "ollama"  # Force Ollama (local)


class ProviderRouter:
    """
    Routes requests to appropriate LLM provider with fallback support.

    The router implements intelligent provider selection:
    - Documents/URLs always use Gemini (preprocessing mode)
    - Text/images use Ollama by default (cost savings)
    - Fallback to Gemini if Ollama unavailable

    Example:
        >>> gemini = GoogleProvider(api_key="...")
        >>> ollama = OllamaProvider()
        >>> router = ProviderRouter(gemini, ollama)
        >>>
        >>> # Auto-selection for text query
        >>> provider, is_hybrid = router.select_provider(has_documents=False)
        >>> # Returns: (ollama, False)
        >>>
        >>> # Auto-selection for PDF query
        >>> provider, is_hybrid = router.select_provider(has_documents=True)
        >>> # Returns: (gemini, True)  # Hybrid mode: Gemini preprocesses
    """

    def __init__(
        self,
        gemini_provider: Optional[GoogleProvider] = None,
        ollama_provider: Optional[OllamaProvider] = None,
        default_strategy: ProviderSelection = ProviderSelection.AUTO,
    ):
        """
        Initialize provider router.

        Args:
            gemini_provider: Optional Gemini API provider
            ollama_provider: Optional Ollama local provider
            default_strategy: Default selection strategy (AUTO recommended)
        """
        self.gemini_provider = gemini_provider
        self.ollama_provider = ollama_provider
        self.default_strategy = default_strategy

        # Log available providers
        available = []
        if gemini_provider:
            available.append("Gemini")
        if ollama_provider:
            available.append("Ollama")
        logger.info(f"ProviderRouter initialized with: {', '.join(available) or 'No providers'}")

    def select_provider(
        self,
        has_documents: bool = False,
        has_urls: bool = False,
        force_provider: Optional[ProviderSelection] = None,
    ) -> Tuple[LLMProviderInterface, bool]:
        """
        Select appropriate provider based on input type and strategy.

        Args:
            has_documents: Whether request includes --document inputs
            has_urls: Whether request includes --url inputs
            force_provider: Optional explicit provider selection

        Returns:
            Tuple of (provider, is_hybrid_mode)
            - provider: Selected LLMProviderInterface instance
            - is_hybrid_mode: True if using Gemini for preprocessing only

        Raises:
            ValueError: If requested provider unavailable or invalid combination

        Example:
            >>> provider, hybrid = router.select_provider(has_documents=True)
            >>> if hybrid:
            >>>     print("Using Gemini for preprocessing, then Ollama for QADI")
        """
        # Force provider if explicitly requested
        if force_provider == ProviderSelection.GEMINI:
            if not self.gemini_provider:
                raise ValueError(
                    "Gemini provider not available. "
                    "Set GOOGLE_API_KEY environment variable."
                )
            logger.info("Using Gemini API (forced by --provider gemini)")
            return self.gemini_provider, False

        if force_provider == ProviderSelection.OLLAMA:
            if not self.ollama_provider:
                raise ValueError(
                    "Ollama provider not available. "
                    "Ensure Ollama is installed and running (ollama serve). "
                    "Install: https://ollama.ai"
                )

            # Validate: Ollama doesn't support documents/URLs directly
            if has_documents or has_urls:
                raise ValueError(
                    "Ollama doesn't support --document or --url inputs directly.\n"
                    "Options:\n"
                    "  1. Use --provider auto (Gemini preprocesses, Ollama handles QADI)\n"
                    "  2. Use --provider gemini (full Gemini pipeline)\n"
                    "  3. Remove --document/--url flags (text/image only with Ollama)"
                )

            logger.info("Using Ollama (forced by --provider ollama)")
            return self.ollama_provider, False

        # Auto-selection strategy (default)
        if has_documents or has_urls:
            # Hybrid mode: Use Gemini for input processing, Ollama for rest
            if not self.gemini_provider:
                raise ValueError(
                    "Documents/URLs require Gemini API for preprocessing.\n"
                    "Set GOOGLE_API_KEY environment variable or remove --document/--url flags."
                )

            logger.info(
                "Hybrid mode: Gemini for document/URL preprocessing → Ollama for QADI"
            )
            return self.gemini_provider, True  # Hybrid mode flag

        # Text/image only: Use Ollama for maximum cost savings
        if self.ollama_provider:
            logger.info("Using Ollama for text/image analysis (free local inference)")
            return self.ollama_provider, False

        # Fallback to Gemini if Ollama unavailable
        if self.gemini_provider:
            logger.warning(
                "Ollama not available, falling back to Gemini API. "
                "Install Ollama for free local inference: https://ollama.ai"
            )
            return self.gemini_provider, False

        # No providers available
        raise ValueError(
            "No LLM providers available.\n"
            "Set up at least one:\n"
            "  - Gemini: Set GOOGLE_API_KEY environment variable\n"
            "  - Ollama: Install and run 'ollama serve'"
        )

    async def generate_with_fallback(
        self,
        request: LLMRequest,
        primary_provider: LLMProviderInterface,
        enable_fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate with automatic fallback on failure.

        If primary provider (Ollama) fails, automatically retry with Gemini.
        This provides resilience without user intervention.

        Args:
            request: LLMRequest to process
            primary_provider: Primary provider to try first
            enable_fallback: Whether to fallback on failure (default: True)

        Returns:
            LLMResponse from primary or fallback provider

        Raises:
            LLMError: If both primary and fallback fail

        Example:
            >>> response = await router.generate_with_fallback(
            ...     request,
            ...     ollama_provider
            ... )
            >>> # If Ollama fails, automatically retries with Gemini
        """
        try:
            return await primary_provider.generate(request)
        except Exception as e:
            logger.warning(
                f"Primary provider ({primary_provider.__class__.__name__}) failed: {e}"
            )

            if not enable_fallback:
                raise

            # Try fallback provider
            if isinstance(primary_provider, OllamaProvider) and self.gemini_provider:
                logger.info("Falling back to Gemini API")
                try:
                    return await self.gemini_provider.generate(request)
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {fallback_error}")
                    raise LLMError(
                        f"Both providers failed. "
                        f"Primary ({primary_provider.__class__.__name__}): {e}. "
                        f"Fallback (Gemini): {fallback_error}",
                        ErrorType.API_ERROR,
                    ) from fallback_error

            # No fallback available
            logger.error("No fallback provider available")
            raise

    def get_provider_status(self) -> dict:
        """
        Get status of all registered providers.

        Returns:
            Dict with provider availability and configuration

        Example:
            >>> status = router.get_provider_status()
            >>> print(status)
            {
                "gemini": {"available": True, "model": "gemini-2.5-flash"},
                "ollama": {"available": True, "model": "gemma3:12b-it-qat"}
            }
        """
        status = {}

        if self.gemini_provider:
            status["gemini"] = {
                "available": True,
                "model": self.gemini_provider.get_available_models()[0].model_name
                if self.gemini_provider.get_available_models()
                else "unknown",
            }
        else:
            status["gemini"] = {"available": False}

        if self.ollama_provider:
            status["ollama"] = {
                "available": True,
                "model": self.ollama_provider.model,
                "base_url": self.ollama_provider.base_url,
            }
        else:
            status["ollama"] = {"available": False}

        return status
