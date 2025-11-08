"""
Multimodal Input Abstractions for Mad Spark Alt.

This module provides provider-agnostic data structures for handling multimodal
inputs (images, documents, URLs, audio, video) in the QADI system.

Key Components:
- MultimodalInputType: Enum for input types (IMAGE, DOCUMENT, AUDIO, VIDEO)
- MultimodalSourceType: Enum for source types (FILE_PATH, URL, BASE64, FILE_API)
- MultimodalInput: Dataclass representing a multimodal input with validation
- URLContextMetadata: Dataclass for tracking URL retrieval status

Design Principles:
- Provider-agnostic: Abstractions work across different LLM providers
- Validation-first: Comprehensive validation of constraints (size, format, count)
- Type-safe: Full type hints and dataclass structure
- Extensible: Easy to add new input types (audio, video) in the future

Example Usage:
    # Create an image input from a file
    image = MultimodalInput(
        input_type=MultimodalInputType.IMAGE,
        source_type=MultimodalSourceType.FILE_PATH,
        data="/path/to/diagram.png",
        mime_type="image/png",
        description="System architecture diagram"
    )

    # Validate constraints
    image.validate()  # Raises ValueError if invalid

    # Create a document input
    document = MultimodalInput(
        input_type=MultimodalInputType.DOCUMENT,
        source_type=MultimodalSourceType.FILE_PATH,
        data="/path/to/paper.pdf",
        mime_type="application/pdf",
        page_count=50
    )

    # Track URL retrieval status
    url_meta = URLContextMetadata(
        url="https://example.com/article",
        status="success"
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class MultimodalSourceType(Enum):
    """
    Source types for multimodal inputs.

    Attributes:
        FILE_PATH: Local file system path
        URL: Remote URL (HTTP/HTTPS)
        BASE64: Inline base64-encoded data
        FILE_API: Pre-uploaded to provider's File API (e.g., Gemini File API)
    """

    FILE_PATH = "file_path"
    URL = "url"
    BASE64 = "base64"
    FILE_API = "file_api"


class MultimodalInputType(Enum):
    """
    Types of multimodal inputs.

    Attributes:
        IMAGE: Image files (PNG, JPEG, WebP, HEIC, HEIF)
        DOCUMENT: Document files (PDF, TXT, Markdown, HTML)
        AUDIO: Audio files (for future support)
        VIDEO: Video files (for future support)
    """

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class URLContextMetadata:
    """
    Metadata about URL retrieval from url_context tool.

    Tracks the success/failure status of URL fetching operations,
    typically returned by LLM providers (e.g., Gemini's url_context tool).

    Attributes:
        url: The URL that was fetched
        status: Retrieval status - "success", "failed", or "blocked"
        error_message: Optional error description if status is not "success"

    Example:
        # Successful retrieval
        meta = URLContextMetadata(url="https://example.com", status="success")

        # Failed retrieval
        meta = URLContextMetadata(
            url="https://example.com",
            status="failed",
            error_message="Connection timeout"
        )

        # Blocked by safety filter
        meta = URLContextMetadata(
            url="https://example.com",
            status="blocked",
            error_message="Safety filter triggered"
        )
    """

    url: str
    status: Literal["success", "failed", "blocked"]
    error_message: Optional[str] = None


@dataclass
class MultimodalInput:
    """
    Unified multimodal input representation.

    Provider-agnostic structure that gets translated to provider-specific
    format by LLMProvider implementations (e.g., GoogleProvider).

    Attributes:
        input_type: Type of input (IMAGE, DOCUMENT, AUDIO, VIDEO)
        source_type: Source of the data (FILE_PATH, URL, BASE64, FILE_API)
        data: The actual data - file path, URL, base64 string, or File API ID
        mime_type: MIME type (e.g., "image/jpeg", "application/pdf")
        description: Optional user-provided description of the content
        file_size: Optional file size in bytes (for validation)
        page_count: Optional page count (for documents)

    Constraints (Gemini-based):
        - Images: Max 20MB for inline (BASE64), unlimited via File API
        - Documents: Max 1000 pages per request
        - Valid image MIME types: image/jpeg, image/png, image/webp, image/heic, image/heif
        - Valid document MIME types: application/pdf, text/plain, text/markdown, text/html

    Example:
        # Image from local file
        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/Users/name/Downloads/diagram.png",
            mime_type="image/png",
            file_size=1024 * 500  # 500KB
        )

        # Validate before use
        image.validate()  # Raises ValueError if constraints violated

        # Document from File API
        doc = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_API,
            data="files/abc123xyz",
            mime_type="application/pdf",
            page_count=50
        )
    """

    input_type: MultimodalInputType
    source_type: MultimodalSourceType
    data: str  # File path, URL, base64 string, or File API ID
    mime_type: str  # e.g., "image/jpeg", "application/pdf"

    # Optional metadata
    description: Optional[str] = None  # User-provided description
    file_size: Optional[int] = None  # Size in bytes (for validation)
    page_count: Optional[int] = None  # For documents

    def validate(self) -> None:
        """
        Validate input constraints based on type.

        Raises:
            ValueError: If input violates constraints (size, format, count)

        Example:
            image = MultimodalInput(...)
            try:
                image.validate()
            except ValueError as e:
                print(f"Validation failed: {e}")
        """
        if self.input_type == MultimodalInputType.IMAGE:
            self._validate_image()
        elif self.input_type == MultimodalInputType.DOCUMENT:
            self._validate_document()
        # AUDIO and VIDEO types have no validation yet (future feature)

    def _validate_image(self) -> None:
        """
        Validate image-specific constraints.

        Checks:
        - File size for inline encoding (max 20MB for BASE64)
        - MIME type is valid for images

        Raises:
            ValueError: If image constraints are violated
        """
        # Gemini: 20MB limit for inline encoding
        if self.source_type == MultimodalSourceType.BASE64:
            if self.file_size and self.file_size > 20 * 1024 * 1024:
                raise ValueError(
                    f"Image too large for inline: {self.file_size} bytes (max 20MB). "
                    f"Consider using File API for larger files."
                )

        # Validate MIME type
        valid_image_types = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        ]
        if self.mime_type not in valid_image_types:
            raise ValueError(
                f"Unsupported image type: {self.mime_type}. "
                f"Supported: {', '.join(valid_image_types)}"
            )

    def _validate_document(self) -> None:
        """
        Validate document-specific constraints.

        Checks:
        - Page count for PDFs (max 1000 pages for Gemini)
        - MIME type is valid for documents

        Raises:
            ValueError: If document constraints are violated
        """
        # Gemini: Max 1000 pages per request
        if self.page_count and self.page_count > 1000:
            raise ValueError(
                f"Document too long: {self.page_count} pages (max 1000). "
                f"Consider splitting into multiple documents."
            )

        # Gemini primarily supports PDF for vision understanding
        # Other formats are text-extracted only
        valid_document_types = [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/html",
        ]
        if self.mime_type not in valid_document_types:
            raise ValueError(
                f"Unsupported document type: {self.mime_type}. "
                f"Supported: {', '.join(valid_document_types)}"
            )
