"""
Tests for multimodal input validation functions.

Tests verify that validation catches invalid multimodal inputs
before they are passed to LLM providers.
"""

import pytest
from unittest.mock import Mock

from mad_spark_alt.core.phase_logic import PhaseInput, _validate_multimodal_inputs
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
)


class TestMultimodalInputValidation:
    """Test validation of multimodal inputs."""

    def test_validate_valid_image_input(self):
        """Valid image input should pass validation."""
        # Arrange
        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
            file_size=1024 * 1024,  # 1MB
        )

        # Act & Assert - should not raise
        _validate_multimodal_inputs([image], None)

    def test_validate_valid_document_input(self):
        """Valid document input should pass validation."""
        # Arrange
        document = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.pdf",
            mime_type="application/pdf",
            file_size=5 * 1024 * 1024,  # 5MB
            page_count=50,
        )

        # Act & Assert - should not raise
        _validate_multimodal_inputs([document], None)

    def test_validate_valid_urls(self):
        """Valid URLs should pass validation."""
        # Arrange
        urls = ["https://example.com/article1", "https://example.com/article2"]

        # Act & Assert - should not raise
        _validate_multimodal_inputs(None, urls)

    def test_validate_image_too_large_for_inline(self):
        """Image larger than 20MB should fail validation for inline source."""
        # Arrange
        large_image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64_data_here",
            mime_type="image/png",
            file_size=25 * 1024 * 1024,  # 25MB
        )

        # Act & Assert
        with pytest.raises(ValueError, match="too large for inline"):
            _validate_multimodal_inputs([large_image], None)

    def test_validate_document_too_many_pages(self):
        """Document with more than 1000 pages should fail validation."""
        # Arrange
        large_doc = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/large.pdf",
            mime_type="application/pdf",
            page_count=1500,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="too long"):
            _validate_multimodal_inputs([large_doc], None)

    def test_validate_unsupported_image_mime_type(self):
        """Unsupported image MIME type should fail validation."""
        # Arrange
        bad_image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.bmp",
            mime_type="image/bmp",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported image type"):
            _validate_multimodal_inputs([bad_image], None)

    def test_validate_unsupported_document_mime_type(self):
        """Unsupported document MIME type should fail validation."""
        # Arrange
        bad_doc = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported document type"):
            _validate_multimodal_inputs([bad_doc], None)

    def test_validate_too_many_urls(self):
        """More than 20 URLs should fail validation."""
        # Arrange
        urls = [f"https://example.com/article{i}" for i in range(25)]

        # Act & Assert
        with pytest.raises(ValueError, match="Too many URLs"):
            _validate_multimodal_inputs(None, urls)

    def test_validate_invalid_url_format(self):
        """Invalid URL format should fail validation."""
        # Arrange
        invalid_urls = ["not-a-valid-url", "ftp://example.com"]

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid URL"):
            _validate_multimodal_inputs(None, invalid_urls)

    def test_validate_none_inputs_passes(self):
        """None inputs should pass validation."""
        # Act & Assert - should not raise
        _validate_multimodal_inputs(None, None)

    def test_validate_empty_lists_passes(self):
        """Empty lists should pass validation."""
        # Act & Assert - should not raise
        _validate_multimodal_inputs([], [])

    def test_validate_mixed_valid_inputs(self):
        """Mix of valid images, documents, and URLs should pass."""
        # Arrange
        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
        )
        document = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/doc.pdf",
            mime_type="application/pdf",
        )
        urls = ["https://example.com/article"]

        # Act & Assert - should not raise
        _validate_multimodal_inputs([image, document], urls)
