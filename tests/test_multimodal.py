"""
Tests for multimodal data structures and validation.

This module tests the core multimodal abstractions including:
- MultimodalInput dataclass
- MultimodalSourceType and MultimodalInputType enums
- Validation logic for images, documents, and URLs
- URLContextMetadata dataclass
"""

import pytest
from pathlib import Path

from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
    URLContextMetadata,
)


class TestMultimodalEnums:
    """Test multimodal enum types."""

    def test_multimodal_source_type_enum(self):
        """Test MultimodalSourceType enum has all required values."""
        assert MultimodalSourceType.FILE_PATH.value == "file_path"
        assert MultimodalSourceType.URL.value == "url"
        assert MultimodalSourceType.BASE64.value == "base64"
        assert MultimodalSourceType.FILE_API.value == "file_api"

    def test_multimodal_input_type_enum(self):
        """Test MultimodalInputType enum has all required values."""
        assert MultimodalInputType.IMAGE.value == "image"
        assert MultimodalInputType.DOCUMENT.value == "document"
        assert MultimodalInputType.AUDIO.value == "audio"
        assert MultimodalInputType.VIDEO.value == "video"


class TestURLContextMetadata:
    """Test URLContextMetadata dataclass."""

    def test_url_context_metadata_creation_success(self):
        """Test creating URLContextMetadata for successful retrieval."""
        metadata = URLContextMetadata(
            url="https://example.com",
            status="success"
        )

        assert metadata.url == "https://example.com"
        assert metadata.status == "success"
        assert metadata.error_message is None

    def test_url_context_metadata_creation_failed(self):
        """Test creating URLContextMetadata for failed retrieval."""
        metadata = URLContextMetadata(
            url="https://example.com",
            status="failed",
            error_message="Connection timeout"
        )

        assert metadata.url == "https://example.com"
        assert metadata.status == "failed"
        assert metadata.error_message == "Connection timeout"

    def test_url_context_metadata_creation_blocked(self):
        """Test creating URLContextMetadata for blocked retrieval."""
        metadata = URLContextMetadata(
            url="https://example.com",
            status="blocked",
            error_message="Safety filter triggered"
        )

        assert metadata.status == "blocked"
        assert metadata.error_message == "Safety filter triggered"


class TestMultimodalInputCreation:
    """Test MultimodalInput creation for different types."""

    def test_create_image_from_file_path(self):
        """Test creating image input from file path."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png"
        )

        assert input_item.input_type == MultimodalInputType.IMAGE
        assert input_item.source_type == MultimodalSourceType.FILE_PATH
        assert input_item.data == "/path/to/image.png"
        assert input_item.mime_type == "image/png"
        assert input_item.description is None
        assert input_item.file_size is None
        assert input_item.page_count is None

    def test_create_image_from_base64(self):
        """Test creating image input from base64 data."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64encodeddata",
            mime_type="image/jpeg",
            file_size=1024 * 500  # 500KB
        )

        assert input_item.source_type == MultimodalSourceType.BASE64
        assert input_item.data == "base64encodeddata"
        assert input_item.file_size == 1024 * 500

    def test_create_image_from_url(self):
        """Test creating image input from URL."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.URL,
            data="https://example.com/image.jpg",
            mime_type="image/jpeg"
        )

        assert input_item.source_type == MultimodalSourceType.URL
        assert input_item.data == "https://example.com/image.jpg"

    def test_create_image_from_file_api(self):
        """Test creating image input from File API."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_API,
            data="files/abc123xyz",
            mime_type="image/png"
        )

        assert input_item.source_type == MultimodalSourceType.FILE_API
        assert input_item.data == "files/abc123xyz"

    def test_create_document_from_file_path(self):
        """Test creating document input from file path."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/document.pdf",
            mime_type="application/pdf",
            file_size=1024 * 1024 * 10,  # 10MB
            page_count=50
        )

        assert input_item.input_type == MultimodalInputType.DOCUMENT
        assert input_item.mime_type == "application/pdf"
        assert input_item.page_count == 50

    def test_create_with_description(self):
        """Test creating input with optional description."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
            description="Architecture diagram showing microservices"
        )

        assert input_item.description == "Architecture diagram showing microservices"


class TestMultimodalInputValidation:
    """Test MultimodalInput validation logic."""

    def test_validate_image_with_valid_mime_type(self):
        """Test validation accepts valid image MIME types."""
        valid_types = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif"
        ]

        for mime_type in valid_types:
            input_item = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.BASE64,
                data="base64data",
                mime_type=mime_type
            )
            # Should not raise
            input_item.validate()

    def test_validate_image_rejects_invalid_mime_type(self):
        """Test validation rejects invalid image MIME types."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data",
            mime_type="application/pdf"  # Invalid for image
        )

        with pytest.raises(ValueError, match="Unsupported image type"):
            input_item.validate()

    def test_validate_image_rejects_oversized_inline(self):
        """Test validation rejects images >20MB for inline encoding."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data",
            mime_type="image/png",
            file_size=21 * 1024 * 1024  # 21MB (over limit)
        )

        with pytest.raises(ValueError, match="Image too large for inline"):
            input_item.validate()

    def test_validate_image_accepts_under_20mb_inline(self):
        """Test validation accepts images ≤20MB for inline encoding."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data",
            mime_type="image/png",
            file_size=20 * 1024 * 1024  # Exactly 20MB
        )

        # Should not raise
        input_item.validate()

    def test_validate_document_with_valid_mime_types(self):
        """Test validation accepts valid document MIME types."""
        valid_types = [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/html"
        ]

        for mime_type in valid_types:
            input_item = MultimodalInput(
                input_type=MultimodalInputType.DOCUMENT,
                source_type=MultimodalSourceType.FILE_PATH,
                data="/path/to/doc",
                mime_type=mime_type
            )
            # Should not raise
            input_item.validate()

    def test_validate_document_rejects_invalid_mime_type(self):
        """Test validation rejects invalid document MIME types."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc",
            mime_type="image/png"  # Invalid for document
        )

        with pytest.raises(ValueError, match="Unsupported document type"):
            input_item.validate()

    def test_validate_document_rejects_too_many_pages(self):
        """Test validation rejects documents >1000 pages."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.pdf",
            mime_type="application/pdf",
            page_count=1001  # Over limit
        )

        with pytest.raises(ValueError, match="Document too long.*1001 pages"):
            input_item.validate()

    def test_validate_document_accepts_1000_pages(self):
        """Test validation accepts documents with exactly 1000 pages."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.pdf",
            mime_type="application/pdf",
            page_count=1000  # Exactly at limit
        )

        # Should not raise
        input_item.validate()

    def test_validate_accepts_none_file_size(self):
        """Test validation accepts None file_size (not yet known)."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
            file_size=None  # Not yet determined
        )

        # Should not raise
        input_item.validate()

    def test_validate_accepts_none_page_count(self):
        """Test validation accepts None page_count (not yet known)."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.pdf",
            mime_type="application/pdf",
            page_count=None  # Not yet determined
        )

        # Should not raise
        input_item.validate()


class TestMultimodalInputEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_string(self):
        """Test handling of empty data string."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="",  # Empty data
            mime_type="image/png"
        )

        # Should create successfully (validation happens on validate() call)
        assert input_item.data == ""

    def test_audio_and_video_types_exist(self):
        """Test that AUDIO and VIDEO types are defined for future use."""
        # These types exist but don't have validation yet (future feature)
        audio_input = MultimodalInput(
            input_type=MultimodalInputType.AUDIO,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/audio.mp3",
            mime_type="audio/mpeg"
        )

        video_input = MultimodalInput(
            input_type=MultimodalInputType.VIDEO,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/video.mp4",
            mime_type="video/mp4"
        )

        assert audio_input.input_type == MultimodalInputType.AUDIO
        assert video_input.input_type == MultimodalInputType.VIDEO
        # No validation for these types yet (no-op)
        audio_input.validate()
        video_input.validate()
