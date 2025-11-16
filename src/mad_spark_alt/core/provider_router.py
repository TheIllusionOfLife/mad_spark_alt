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

import asyncio
import logging
import mimetypes
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .simple_qadi_orchestrator import SimpleQADIResult

from .llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMProviderInterface,
    LLMRequest,
    LLMResponse,
    OllamaProvider,
)
from .multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType
from .retry import ErrorType, LLMError

logger = logging.getLogger(__name__)


# Keywords indicating Ollama-specific failures (vs programming bugs)
# Used for targeted fallback detection - only fallback on provider failures,
# not on code errors that would fail with any provider
# These patterns were established in PR #147 to prevent masking programming bugs
# while still catching genuine Ollama connection/timeout issues
_OLLAMA_FAILURE_KEYWORDS = [
    "Ollama",
    "ollama",
    "Connection",
    "aiohttp",
    "Failed to generate",
    "Failed to parse",
    "Failed to extract",
    "Failed to score",
    "Max retries exceeded",
]


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
            logger.exception(
                f"Primary provider ({primary_provider.__class__.__name__}) failed"
            )

            if not enable_fallback:
                raise

            # Try fallback provider
            if isinstance(primary_provider, OllamaProvider) and self.gemini_provider:
                logger.info("Falling back to Gemini API")
                try:
                    return await self.gemini_provider.generate(request)
                except Exception as fallback_error:
                    logger.exception("Fallback provider also failed")
                    raise LLMError(
                        f"Both providers failed. "
                        f"Primary ({primary_provider.__class__.__name__}): {e}. "
                        f"Fallback (Gemini): {fallback_error}",
                        ErrorType.API_ERROR,
                    ) from fallback_error

            # No fallback available
            logger.exception("No fallback provider available")
            raise

    async def run_qadi_with_fallback(
        self,
        user_input: str,
        primary_provider: LLMProviderInterface,
        fallback_provider: Optional[LLMProviderInterface] = None,
        temperature_override: Optional[float] = None,
        num_hypotheses: int = 3,
        multimodal_inputs: Optional[List[Any]] = None,
        urls: Optional[List[str]] = None,
    ) -> Tuple["SimpleQADIResult", LLMProviderInterface, bool]:
        """
        Run QADI cycle with automatic provider fallback.

        This method centralizes fallback logic for both CLI and SDK users.
        If the primary provider (typically Ollama) fails, it automatically
        retries with the fallback provider (typically Gemini).

        Args:
            user_input: User's question or input text
            primary_provider: Primary provider to try first
            fallback_provider: Optional fallback provider (usually Gemini)
            temperature_override: Optional temperature for LLM generation
            num_hypotheses: Number of hypotheses to generate (default: 3)
            multimodal_inputs: Optional list of multimodal inputs (images, etc.)
            urls: Optional list of URLs for context

        Returns:
            Tuple of (result, active_provider, used_fallback):
            - result: SimpleQADIResult from the successful provider
            - active_provider: The provider that actually succeeded
            - used_fallback: True if fallback was used, False otherwise

        Raises:
            Exception: If both providers fail or no fallback available

        Example:
            >>> # SDK usage
            >>> router = ProviderRouter(gemini_provider, ollama_provider)
            >>> result, provider, used_fallback = await router.run_qadi_with_fallback(
            ...     user_input="How can AI improve education?",
            ...     primary_provider=ollama_provider,
            ...     fallback_provider=gemini_provider,
            ... )
            >>> if used_fallback:
            ...     print("Ollama failed, completed with Gemini")
        """
        # Import here to avoid circular dependency
        from .simple_qadi_orchestrator import SimpleQADIOrchestrator

        orchestrator = SimpleQADIOrchestrator(
            temperature_override=temperature_override,
            num_hypotheses=num_hypotheses,
            llm_provider=primary_provider,
        )

        try:
            result = await orchestrator.run_qadi_cycle(
                user_input,
                multimodal_inputs=multimodal_inputs,
                urls=urls,
            )
            return result, primary_provider, False
        except Exception as primary_error:
            # Targeted Ollama failure detection
            # Only catch connection/timeout errors and specific Ollama failures
            # Avoid catching all RuntimeError to prevent masking programming bugs
            is_ollama_failure = isinstance(
                primary_provider, OllamaProvider
            ) and (
                isinstance(
                    primary_error, (ConnectionError, OSError, asyncio.TimeoutError)
                )
                or any(
                    keyword in str(primary_error)
                    for keyword in _OLLAMA_FAILURE_KEYWORDS
                )
            )

            if is_ollama_failure and fallback_provider is not None:
                logger.warning(
                    f"Primary provider ({primary_provider.__class__.__name__}) failed, "
                    f"falling back to {fallback_provider.__class__.__name__}: {primary_error}"
                )

                fallback_orchestrator = SimpleQADIOrchestrator(
                    temperature_override=temperature_override,
                    num_hypotheses=num_hypotheses,
                    llm_provider=fallback_provider,
                )
                result = await fallback_orchestrator.run_qadi_cycle(
                    user_input,
                    multimodal_inputs=multimodal_inputs,
                    urls=urls,
                )
                return result, fallback_provider, True

            # Not an Ollama failure or no fallback available, re-raise
            raise

    async def extract_document_content(
        self,
        document_paths: Optional[tuple] = None,
        urls: Optional[tuple] = None,
    ) -> Tuple[str, float]:
        """
        Use Gemini to extract text content from documents and URLs.

        This enables hybrid routing: Gemini extracts content that Ollama
        (gemma3) cannot process directly (PDFs, CSVs, web pages), then
        passes the extracted text to Ollama for QADI reasoning.

        Args:
            document_paths: Tuple of document file paths (from --document flag)
            urls: Tuple of URL strings (from --url flag)

        Returns:
            Tuple of (extracted_text, extraction_cost):
            - extracted_text: Text content extracted from documents/URLs
            - extraction_cost: LLM cost for the extraction operation

        Raises:
            ValueError: If Gemini provider is not available

        Example:
            >>> text, cost = await router.extract_document_content(
            ...     document_paths=("report.pdf", "data.csv"),
            ...     urls=("https://example.com/article",)
            ... )
            >>> # text contains extracted content ready for Ollama
        """
        if not self.gemini_provider:
            raise ValueError(
                "Gemini provider required for document/URL content extraction.\n"
                "Set GOOGLE_API_KEY environment variable."
            )

        # Normalize parameters for SDK callers
        document_paths = document_paths or ()
        urls = urls or ()

        # Build extraction prompt
        extraction_prompt = (
            "Extract and summarize the key content from the provided documents and/or URLs.\n"
            "Focus on:\n"
            "- Main facts, data, and statistics\n"
            "- Key arguments and conclusions\n"
            "- Relevant context for analysis\n\n"
            "Return the content as plain text, organized clearly.\n"
            "Do not add your own analysis - just extract and organize the information."
        )

        # Build multimodal inputs for documents
        multimodal_inputs = []
        for doc_path in document_paths:
            mime_type, _ = mimetypes.guess_type(doc_path)
            if mime_type != "application/pdf":
                if not doc_path.lower().endswith('.pdf'):
                    # Skip non-PDF documents for now
                    logger.warning(f"Skipping non-PDF document: {doc_path}")
                    continue
                mime_type = "application/pdf"

            doc_size = Path(doc_path).stat().st_size
            multimodal_inputs.append(
                MultimodalInput(
                    input_type=MultimodalInputType.DOCUMENT,
                    source_type=MultimodalSourceType.FILE_PATH,
                    data=str(Path(doc_path).absolute()),
                    mime_type=mime_type,
                    file_size=doc_size,
                )
            )

        # Create extraction request
        request = LLMRequest(
            user_prompt=extraction_prompt,
            multimodal_inputs=multimodal_inputs if multimodal_inputs else None,
            urls=list(urls) if urls else None,
            max_tokens=4000,  # Allow detailed extraction
            temperature=0.3,  # Low temperature for factual extraction
        )

        logger.info(
            f"Extracting content from {len(document_paths)} documents and {len(urls)} URLs"
        )

        response = await self.gemini_provider.generate(request)
        extraction_cost = response.cost

        logger.info(
            f"Extracted {len(response.content)} characters (cost: ${extraction_cost:.6f})"
        )

        return response.content, extraction_cost

    async def run_hybrid_qadi(
        self,
        user_input: str,
        document_paths: Optional[tuple] = None,
        urls: Optional[tuple] = None,
        image_paths: Optional[tuple] = None,
        temperature_override: Optional[float] = None,
        num_hypotheses: int = 3,
    ) -> Tuple["SimpleQADIResult", LLMProviderInterface, bool, Dict[str, Any]]:
        """
        Run hybrid QADI: Gemini extracts document/URL content, Ollama reasons.

        This provides cost optimization for document-heavy workflows:
        - Gemini: Process documents/URLs once (extraction)
        - Ollama: Run all 4 QADI phases locally (free)

        Falls back to Gemini-only if Ollama is unavailable.

        Args:
            user_input: User's question or input text
            document_paths: Tuple of document file paths (from --document)
            urls: Tuple of URL strings (from --url)
            image_paths: Tuple of image file paths (from --image)
            temperature_override: Optional temperature for QADI phases
            num_hypotheses: Number of hypotheses to generate

        Returns:
            Tuple of (result, provider, used_fallback, metadata):
            - result: SimpleQADIResult from QADI cycle
            - provider: The provider that executed QADI (Ollama or Gemini)
            - used_fallback: True if fell back to Gemini-only
            - metadata: Dict with preprocessing info (cost, char count, etc.)

        Example:
            >>> result, provider, fallback, meta = await router.run_hybrid_qadi(
            ...     user_input="Analyze the market trends",
            ...     document_paths=("report.pdf",),
            ...     urls=("https://news.com/article",),
            ...     image_paths=(),
            ... )
            >>> print(f"Preprocessing cost: ${meta['preprocessing_cost']:.4f}")
            >>> print(f"QADI cost: ${result.total_cost:.4f}")
        """
        if not self.gemini_provider:
            raise ValueError(
                "Gemini provider required for hybrid routing (document/URL preprocessing)."
            )

        # Import here to avoid circular dependency
        from .simple_qadi_orchestrator import SimpleQADIOrchestrator

        # Normalize parameters for SDK callers
        document_paths = document_paths or ()
        urls = urls or ()
        image_paths = image_paths or ()

        metadata = {
            "preprocessing_cost": 0.0,
            "extracted_content_length": 0,
            "documents_processed": len(document_paths),
            "urls_processed": len(urls),
            "images_passed_to_ollama": len(image_paths),
            "hybrid_mode": True,
        }

        # Step 1: Extract content from documents/URLs using Gemini
        extracted_context, preprocessing_cost = await self.extract_document_content(
            document_paths=document_paths,
            urls=urls,
        )
        metadata["preprocessing_cost"] = preprocessing_cost
        metadata["extracted_content_length"] = len(extracted_context)

        # Step 2: Build image-only multimodal inputs for Ollama
        # gemma3 can handle images natively, so pass them directly
        image_inputs = []
        for img_path in image_paths:
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"

            img_size = Path(img_path).stat().st_size
            image_inputs.append(
                MultimodalInput(
                    input_type=MultimodalInputType.IMAGE,
                    source_type=MultimodalSourceType.FILE_PATH,
                    data=str(Path(img_path).absolute()),
                    mime_type=mime_type,
                    file_size=img_size,
                )
            )

        # Step 3: Prepare enhanced input with extracted context
        enhanced_input = (
            f"Context from documents/URLs:\n"
            f"---\n"
            f"{extracted_context}\n"
            f"---\n\n"
            f"Question: {user_input}"
        )

        # Step 4: Run QADI with Ollama (or fallback to Gemini)
        if self.ollama_provider:
            logger.info("Running QADI with Ollama using extracted context")
            try:
                orchestrator = SimpleQADIOrchestrator(
                    temperature_override=temperature_override,
                    num_hypotheses=num_hypotheses,
                    llm_provider=self.ollama_provider,
                )
                result = await orchestrator.run_qadi_cycle(
                    user_input=enhanced_input,
                    multimodal_inputs=image_inputs if image_inputs else None,
                    urls=None,  # Already processed by Gemini
                )
                return result, self.ollama_provider, False, metadata
            except Exception as ollama_error:
                # Check if this is an Ollama-specific failure
                is_ollama_failure = isinstance(
                    ollama_error, (ConnectionError, OSError, asyncio.TimeoutError)
                ) or any(
                    keyword in str(ollama_error)
                    for keyword in _OLLAMA_FAILURE_KEYWORDS
                )

                if is_ollama_failure:
                    logger.warning(
                        f"Ollama failed during hybrid QADI, falling back to Gemini-only: {ollama_error}"
                    )
                    # Fall through to Gemini fallback
                else:
                    # Not an Ollama failure, re-raise
                    raise

        # Fallback: Use Gemini for QADI (already have the context extracted)
        logger.info("Falling back to Gemini-only mode for QADI")
        metadata["hybrid_mode"] = False  # Not truly hybrid anymore

        # For Gemini fallback, we can either:
        # 1. Pass the original documents/URLs again (redundant but consistent)
        # 2. Use the extracted context (efficient but different behavior)
        # We choose option 2 for consistency with what Ollama would see
        fallback_orchestrator = SimpleQADIOrchestrator(
            temperature_override=temperature_override,
            num_hypotheses=num_hypotheses,
            llm_provider=self.gemini_provider,
        )

        # Use extracted context + images (same as what Ollama would receive)
        result = await fallback_orchestrator.run_qadi_cycle(
            user_input=enhanced_input,
            multimodal_inputs=image_inputs if image_inputs else None,
            urls=None,  # Already extracted
        )

        return result, self.gemini_provider, True, metadata

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
            try:
                models = self.gemini_provider.get_available_models()
                model_name = models[0].model_name if models else "unknown"
            except Exception:
                model_name = "unknown"
            status["gemini"] = {"available": True, "model": model_name}
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
