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
import ipaddress
import json
import logging
import mimetypes
import socket
from enum import Enum
from hashlib import sha256
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

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

# Content extraction limits
DEFAULT_EXTRACTION_MAX_TOKENS = 4000
OLLAMA_CONTEXT_WARNING_THRESHOLD = 6000  # Leave room for QADI prompts
MAX_CSV_ROWS = 100  # Truncate large CSVs

# Supported document extensions for text-based processing
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".md", ".markdown"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf"} | SUPPORTED_TEXT_EXTENSIONS

# SSRF prevention: blocked patterns
_BLOCKED_METADATA_PATTERNS = [
    "169.254.169.254",
    "metadata.google",
    "metadata.azure",
]


class ContentCache:
    """
    In-memory cache for extracted document content with mtime tracking.

    Caches both text documents and expensive API extractions (PDFs, URLs).
    Uses mtime+size for fast lookups without re-hashing.
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 100):  # 1 hour default
        """
        Initialize content cache.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_entries: Maximum number of entries to prevent unbounded memory growth
        """
        # Cache structure: key -> (content, cost, timestamp, mtime_or_hash, size)
        # mtime_or_hash is float (mtime) for individual files, int (hash) for extractions
        self._cache: Dict[str, Tuple[str, float, float, Union[float, int], int]] = {}
        self.ttl = ttl_seconds
        self.max_entries = max_entries

    def _make_file_key(self, file_path: Path) -> Optional[Tuple[str, float, int]]:
        """
        Create cache key from file path with mtime+size metadata.

        Returns:
            Tuple of (key, mtime, size) or None if file inaccessible
        """
        try:
            stat = file_path.stat()
            # Key is just the absolute path string for files
            # We use mtime+size for invalidation instead of content hash
            return (str(file_path.absolute()), stat.st_mtime, stat.st_size)
        except (OSError, IOError):
            return None

    def _make_extraction_key(self, doc_paths: tuple, urls: tuple) -> str:
        """
        Create composite cache key for PDF/URL extractions.

        Combines sorted file paths and URLs to create a stable key.
        """
        parts = []
        if doc_paths:
            parts.extend(sorted(str(p) for p in doc_paths))
        if urls:
            parts.extend(sorted(urls))
        # Hash the composite to keep keys reasonable length
        composite = "|".join(parts)
        return "extraction:" + sha256(composite.encode()).hexdigest()[:16]

    def get(self, file_path: Path) -> Optional[Tuple[str, float]]:
        """
        Get cached content if exists and not expired.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (content, cost) if cached and valid, None otherwise
        """
        key_data = self._make_file_key(file_path)
        if not key_data:
            return None

        key, current_mtime, current_size = key_data

        if key in self._cache:
            content, cost, timestamp, cached_mtime, cached_size = self._cache[key]
            # Check TTL and file modification
            if time() - timestamp < self.ttl and current_mtime == cached_mtime and current_size == cached_size:
                logger.debug(f"Cache hit for {file_path.name}")
                return content, cost
            else:
                # Expired or modified
                del self._cache[key]

        return None

    def set(self, file_path: Path, content: str, cost: float) -> None:
        """
        Cache extracted content.

        Args:
            file_path: Path to the file
            content: Extracted content
            cost: Extraction cost
        """
        key_data = self._make_file_key(file_path)
        if not key_data:
            return

        key, mtime, size = key_data

        # Evict oldest entry if cache is full (LRU-style eviction)
        if len(self._cache) >= self.max_entries and key not in self._cache:
            self._evict_oldest()

        self._cache[key] = (content, cost, time(), mtime, size)
        logger.debug(f"Cached content for {file_path.name}")

    def get_extraction(self, doc_paths: tuple, urls: tuple) -> Optional[Tuple[str, float]]:
        """
        Get cached extraction result for composite document+URL inputs.

        Checks file mtimes to detect modifications.

        Args:
            doc_paths: Tuple of document paths
            urls: Tuple of URLs

        Returns:
            Tuple of (content, cost) if cached and valid, None otherwise
        """
        key = self._make_extraction_key(doc_paths, urls)

        if key in self._cache:
            content, cost, timestamp, cached_mtime_hash, _ = self._cache[key]

            # Check TTL
            if time() - timestamp >= self.ttl:
                del self._cache[key]
                return None

            # For extractions with files, verify files haven't changed
            if doc_paths:
                # Compute current mtime hash (sort paths for consistency with cache key)
                current_mtimes = []
                for doc_path in sorted(doc_paths):  # Sort to match _make_extraction_key
                    try:
                        stat = Path(doc_path).stat()
                        current_mtimes.append(f"{stat.st_mtime}:{stat.st_size}")
                    except (OSError, IOError):
                        # File disappeared, cache invalid
                        del self._cache[key]
                        return None

                current_mtime_hash = hash(tuple(current_mtimes))
                if current_mtime_hash != cached_mtime_hash:
                    # Files modified, cache invalid
                    del self._cache[key]
                    return None

            logger.debug(f"Cache hit for extraction ({len(doc_paths)} docs, {len(urls)} URLs)")
            return content, cost

        return None

    def set_extraction(self, doc_paths: tuple, urls: tuple, content: str, cost: float) -> None:
        """
        Cache extraction result for composite document+URL inputs.

        Stores mtime hash for file-based extractions to detect modifications.

        Args:
            doc_paths: Tuple of document paths
            urls: Tuple of URLs
            content: Extracted content
            cost: Extraction cost
        """
        key = self._make_extraction_key(doc_paths, urls)

        # Evict oldest entry if cache is full
        if len(self._cache) >= self.max_entries and key not in self._cache:
            self._evict_oldest()

        # Compute mtime hash for files (sort paths for consistency with cache key)
        mtime_hash = 0
        if doc_paths:
            mtimes = []
            for doc_path in sorted(doc_paths):  # Sort to match _make_extraction_key
                try:
                    stat = Path(doc_path).stat()
                    mtimes.append(f"{stat.st_mtime}:{stat.st_size}")
                except (OSError, IOError):
                    pass
            mtime_hash = hash(tuple(mtimes)) if mtimes else 0

        # Store with mtime hash in the 4th position (where individual files store mtime)
        # Note: Keep mtime_hash as int to match hash() return type for reliable comparison
        self._cache[key] = (content, cost, time(), mtime_hash, 0)
        logger.debug(f"Cached extraction ({len(doc_paths)} docs, {len(urls)} URLs)")

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry to make room for new content."""
        if not self._cache:
            return
        # Find entry with oldest timestamp
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
        del self._cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry to stay within {self.max_entries} limit")

    def clear(self) -> None:
        """Clear all cached content."""
        self._cache.clear()
        logger.debug("Content cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        This prevents memory leaks in long-running processes by proactively
        cleaning up stale entries rather than waiting for access.

        Returns:
            Number of entries removed
        """
        current_time = time()
        expired_keys = [
            k for k, (_, _, timestamp, _, _) in self._cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)


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
        self._content_cache = ContentCache()

        # Log available providers
        available = []
        if gemini_provider:
            available.append("Gemini")
        if ollama_provider:
            available.append("Ollama")
        logger.info(f"ProviderRouter initialized with: {', '.join(available) or 'No providers'}")

    def _is_ollama_connection_error(self, error: Exception) -> bool:
        """
        Check if an exception is a common Ollama connection/runtime error.

        This helper provides consistent fallback detection across methods,
        avoiding duplication and ensuring all error types are handled uniformly.

        Args:
            error: The exception to check

        Returns:
            True if error indicates Ollama connection/timeout failure,
            False otherwise (likely a programming bug that should be raised)
        """
        return isinstance(
            error, (ConnectionError, OSError, asyncio.TimeoutError)
        ) or any(
            keyword in str(error)
            for keyword in _OLLAMA_FAILURE_KEYWORDS
        )

    def _validate_url_security(self, url: str) -> None:
        """
        Validate URL is safe to fetch (prevent SSRF attacks).

        This blocks:
        - Non-HTTP(S) schemes (file://, ftp://, etc.)
        - Internal/private IPs (localhost, 127.0.0.1, 10.x.x.x, etc.)
        - Cloud metadata endpoints (AWS, GCP, Azure)

        This validation performs DNS resolution to detect:
        - DNS rebinding attacks (hostnames resolving to private IPs)
        - Internal hostnames that resolve to internal IPs

        Args:
            url: URL string to validate

        Raises:
            ValueError: If URL is not safe for external fetching
        """
        parsed = urlparse(url)

        # 1. Scheme whitelist - only HTTP(S) allowed
        if parsed.scheme not in ("https", "http"):
            raise ValueError(
                f"Unsupported URL scheme: {parsed.scheme}. Only http/https allowed."
            )

        # 2. Block internal/private IPs
        hostname = parsed.hostname
        if hostname:
            # Decode percent-encoding and normalize to prevent bypasses like http://127.0.0.1%2e/
            hostname_decoded = unquote(hostname).lower().rstrip('.')

            # Block localhost variants (check decoded value)
            if hostname_decoded in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                raise ValueError(f"Internal URLs not allowed: {hostname}")

            # Block private IP ranges
            try:
                ip = ipaddress.ip_address(hostname_decoded)
            except ValueError:
                pass  # Not an IP address, hostname is fine
            else:
                # Successfully parsed as IP, check if it's private
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"Private/internal IP not allowed: {hostname}")

            # 3. Block cloud metadata endpoints (common SSRF targets)
            # Check hostname only (not entire URL) to avoid false positives on paths/queries
            # Use lowercase comparison to prevent case-sensitivity bypass (e.g., METADATA.GOOGLE.INTERNAL)
            if any(pattern in hostname_decoded for pattern in _BLOCKED_METADATA_PATTERNS):
                raise ValueError(f"Cloud metadata endpoints not allowed: {url}")

            # 4. Resolve hostname to check for private IPs (DNS rebinding protection)
            try:
                # Use getaddrinfo to support both IPv4 and IPv6
                # This blocks, but ensures we check the resolved IP.
                # Since this is a CLI tool, brief blocking for security check is acceptable.
                addr_info = socket.getaddrinfo(hostname_decoded, None)
                for _, _, _, _, sockaddr in addr_info:
                    ip_str = sockaddr[0]
                    # Ensure we have a string (AF_INET/AF_INET6), skip if not (e.g. AF_UNIX)
                    if not isinstance(ip_str, str):
                        continue

                    try:
                        # Remove scope ID from IPv6 address if present (e.g. fe80::1%lo0)
                        if "%" in ip_str:
                            ip_str = ip_str.split("%")[0]

                        ip = ipaddress.ip_address(ip_str)
                    except ValueError:
                        continue

                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise ValueError(
                            f"Hostname '{hostname}' resolves to private/internal IP {ip_str}"
                        )
            except socket.gaierror:
                # DNS resolution failed - could be invalid hostname or network issue.
                # We proceed with caution (allow) since we can't verify, and fetching
                # will likely fail anyway if DNS is broken.
                pass

    async def _read_text_document(self, file_path: Path) -> str:
        """
        Read text-based documents directly.

        Handles different file formats:
        - .csv: Format as readable table with row truncation
        - .json: Pretty-print JSON data
        - .txt, .md, .markdown: Read as plain text

        Args:
            file_path: Path to text-based document

        Returns:
            Formatted document content as string

        Note:
            Uses asyncio.to_thread to avoid blocking the event loop during file I/O.
        """
        suffix = file_path.suffix.lower()

        # Offload blocking file I/O to thread pool to avoid blocking event loop
        def _blocking_read() -> str:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _blocking_read)

        if suffix == ".csv":
            # Format CSV as readable table with truncation for large files
            lines = content.strip().split("\n")
            if len(lines) > MAX_CSV_ROWS:
                truncated_content = "\n".join(lines[:MAX_CSV_ROWS])
                return f"CSV Data:\n{truncated_content}\n... ({len(lines) - MAX_CSV_ROWS} more rows)"
            return f"CSV Data:\n{content}"

        elif suffix == ".json":
            # Pretty-print JSON for readability
            try:
                data = json.loads(content)
                # Use ensure_ascii=False to preserve non-ASCII characters (accents, scripts)
                return f"JSON Data:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
            except json.JSONDecodeError:
                return f"JSON File (invalid format):\n{content}"

        else:  # .txt, .md, .markdown
            return content

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
        # Validate all URLs for SSRF prevention BEFORE processing
        # This ensures security checks run regardless of hybrid vs non-hybrid path
        if urls:
            for url in urls:
                self._validate_url_security(url)

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
            ) and self._is_ollama_connection_error(primary_error)

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
        max_tokens: int = DEFAULT_EXTRACTION_MAX_TOKENS,
        warn_on_large: bool = True,
    ) -> Tuple[str, float]:
        """
        Use Gemini to extract text content from documents and URLs.

        This enables hybrid routing: Gemini extracts content that Ollama
        (gemma3) cannot process directly (e.g., PDFs, web pages), then
        passes the extracted text to Ollama for QADI reasoning.

        Supports multiple document formats:
        - PDF: Processed via Gemini multimodal API
        - TXT, MD: Read as plain text
        - CSV: Formatted as table with row truncation
        - JSON: Pretty-printed for readability

        Args:
            document_paths: Tuple of document file paths (from --document flag)
            urls: Tuple of URL strings (from --url flag)
            max_tokens: Maximum tokens for extraction (default: 4000)
            warn_on_large: Whether to warn about large content (default: True)

        Returns:
            Tuple of (extracted_text, extraction_cost):
            - extracted_text: Text content extracted from documents/URLs
            - extraction_cost: LLM cost for the extraction operation

        Raises:
            ValueError: If Gemini provider is not available or URLs are unsafe

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

        # Validate URLs for security (SSRF prevention)
        for url in urls:
            self._validate_url_security(url)

        # Check cache for full extraction (PDFs + URLs composite)
        if document_paths or urls:
            cached_extraction = self._content_cache.get_extraction(document_paths, urls)
            if cached_extraction:
                content, cost = cached_extraction
                logger.info(f"Using cached extraction (saved ${cost:.6f})")
                return content, cost

        # Build multimodal inputs for PDFs and text contexts for other files
        multimodal_inputs = []
        text_contexts = []
        total_cost = 0.0

        for doc_path in document_paths:
            doc_path_obj = Path(doc_path)
            if not doc_path_obj.exists():
                logger.warning(f"Document not found, skipping: {doc_path}")
                continue

            file_ext = doc_path_obj.suffix.lower()

            # Check if extension is supported
            if file_ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
                logger.warning(
                    f"Skipping unsupported document type: {doc_path} "
                    f"(supported: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))})"
                )
                continue

            # Check cache for this document
            cached = self._content_cache.get(doc_path_obj)
            if cached:
                content, cost = cached
                text_contexts.append(f"=== {doc_path_obj.name} ===\n{content}")
                total_cost += cost
                continue

            if file_ext == ".pdf":
                # PDF: Process via Gemini multimodal API
                mime_type = "application/pdf"
                doc_size = doc_path_obj.stat().st_size
                multimodal_inputs.append(
                    MultimodalInput(
                        input_type=MultimodalInputType.DOCUMENT,
                        source_type=MultimodalSourceType.FILE_PATH,
                        data=str(doc_path_obj.absolute()),
                        mime_type=mime_type,
                        file_size=doc_size,
                    )
                )
            else:
                # Text-based documents: read directly
                try:
                    text_content = await self._read_text_document(doc_path_obj)
                    text_contexts.append(f"=== {doc_path_obj.name} ===\n{text_content}")
                    # Cache the text content (cost is 0 since no API call)
                    self._content_cache.set(doc_path_obj, text_content, 0.0)
                except (OSError, IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to read {doc_path}: {e}")
                    continue

        # Check if we have anything to extract
        if not multimodal_inputs and not urls and not text_contexts:
            raise ValueError(
                "No valid documents or URLs to extract content from.\n"
                f"All provided documents were either not found or not supported.\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}\n"
                "Please check your file paths and ensure documents exist."
            )

        # If we only have text contexts (no PDFs or URLs), return them directly
        if not multimodal_inputs and not urls:
            combined_text = "\n\n".join(text_contexts)
            # Check size limits
            combined_text = self._apply_size_limits(combined_text, max_tokens, warn_on_large)
            return combined_text, total_cost

        # Build extraction prompt including any text contexts
        base_prompt = (
            "Extract and summarize the key content from the provided documents and/or URLs.\n"
            "Focus on:\n"
            "- Main facts, data, and statistics\n"
            "- Key arguments and conclusions\n"
            "- Relevant context for analysis\n\n"
            "Return the content as plain text, organized clearly.\n"
            "Do not add your own analysis - just extract and organize the information."
        )

        if text_contexts:
            # Truncate text contexts BEFORE embedding in prompt to prevent overflow
            combined_text_contexts = "".join(text_contexts)
            # Use 50% of max_tokens for text contexts, rest for PDF/URL extraction
            combined_text_contexts = self._apply_size_limits(
                combined_text_contexts, max_tokens // 2, warn_on_large=False
            )
            extraction_prompt = (
                f"Additional document content:\n"
                f"{combined_text_contexts}\n\n"
                f"{base_prompt}"
            )
        else:
            extraction_prompt = base_prompt

        # Create extraction request
        request = LLMRequest(
            user_prompt=extraction_prompt,
            multimodal_inputs=multimodal_inputs if multimodal_inputs else None,
            urls=list(urls) if urls else None,
            max_tokens=max_tokens,
            temperature=0.3,  # Low temperature for factual extraction
        )

        logger.info(
            f"Extracting content from {len(document_paths)} documents and {len(urls)} URLs"
        )

        response = await self.gemini_provider.generate(request)
        extraction_cost = response.cost + total_cost

        extracted_content = response.content

        # Apply size limits and warnings
        extracted_content = self._apply_size_limits(extracted_content, max_tokens, warn_on_large)

        logger.info(
            f"Extracted {len(extracted_content)} characters (cost: ${extraction_cost:.6f})"
        )

        # Cache the extraction result (PDFs + URLs composite)
        if multimodal_inputs or urls:
            self._content_cache.set_extraction(document_paths, urls, extracted_content, extraction_cost)

        return extracted_content, extraction_cost

    def _apply_size_limits(
        self, content: str, max_tokens: int, warn_on_large: bool
    ) -> str:
        """
        Apply size limits and warnings to extracted content.

        Args:
            content: Extracted content string
            max_tokens: Maximum allowed tokens
            warn_on_large: Whether to emit warnings for large content

        Returns:
            Content, potentially truncated with notice
        """
        # Estimate token count (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(content) // 4

        if warn_on_large and estimated_tokens > OLLAMA_CONTEXT_WARNING_THRESHOLD:
            logger.warning(
                f"Extracted content is large (~{estimated_tokens} tokens). "
                f"May exceed Ollama context limits. Consider using --provider gemini for large documents."
            )

        # Truncate if significantly over limit (50% over)
        if estimated_tokens > max_tokens * 1.5:
            truncate_chars = int(max_tokens * 4)
            original_length = len(content)
            content = content[:truncate_chars]
            content += f"\n\n[Content truncated from ~{original_length // 4} to ~{max_tokens} tokens]"
            logger.info(f"Content truncated from ~{original_length // 4} to ~{max_tokens} tokens")

        return content

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

        # Warn if extraction returned empty content (document might be unreadable)
        if not extracted_context.strip():
            logger.warning(
                "Gemini returned empty content from document/URL extraction. "
                "Documents may be empty, corrupted, or unreadable. "
                "QADI will proceed with minimal context."
            )

        # Step 2: Build image-only multimodal inputs for Ollama
        # gemma3 can handle images natively, so pass them directly
        image_inputs = []
        for img_path in image_paths:
            img_path_obj = Path(img_path)
            if not img_path_obj.exists():
                logger.warning(f"Image not found, skipping: {img_path}")
                continue

            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"

            img_size = img_path_obj.stat().st_size
            image_inputs.append(
                MultimodalInput(
                    input_type=MultimodalInputType.IMAGE,
                    source_type=MultimodalSourceType.FILE_PATH,
                    data=str(img_path_obj.absolute()),
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
                is_ollama_failure = self._is_ollama_connection_error(ollama_error)

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
                "gemini": {"available": True, "model": "gemini-3-flash-preview"},
                "ollama": {"available": True, "model": "gemma3:12b"}
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
