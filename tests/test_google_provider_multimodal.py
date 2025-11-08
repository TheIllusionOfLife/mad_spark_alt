"""
Tests for GoogleProvider multimodal support (Phase 2).

This module tests the Gemini provider's ability to handle:
- Images (inline base64, file paths, URLs)
- Documents (PDFs with vision understanding)
- URL context (web content fetching)
- Mixed multimodal inputs
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
    URLContextMetadata,
)


class TestGoogleProviderMultimodal:
    """Test GoogleProvider multimodal functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.provider = GoogleProvider(api_key="test-api-key")

    def teardown_method(self):
        """Clean up after tests."""
        # Close session if it exists
        import asyncio
        try:
            if self.provider._session:
                asyncio.get_event_loop().run_until_complete(self.provider.close())
        except:
            pass

    # ============================================================================
    # Tests for _build_contents()
    # ============================================================================

    def test_build_contents_text_only(self):
        """Test _build_contents with text-only request (backward compatibility)."""
        request = LLMRequest(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is AI?"
        )

        contents = self.provider._build_contents(request)

        assert isinstance(contents, list)
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert len(contents[0]["parts"]) == 1
        assert "text" in contents[0]["parts"][0]
        assert "System: You are a helpful assistant." in contents[0]["parts"][0]["text"]
        assert "User: What is AI?" in contents[0]["parts"][0]["text"]

    def test_build_contents_with_single_image(self):
        """Test _build_contents with single image input."""
        image_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64encodeddata",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Describe this image",
            multimodal_inputs=[image_input]
        )

        contents = self.provider._build_contents(request)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        parts = contents[0]["parts"]
        assert len(parts) == 2  # Image part + text part
        # Image comes first (Gemini best practice for single image)
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/png"
        # Text comes second
        assert "text" in parts[1]

    def test_build_contents_with_multiple_images(self):
        """Test _build_contents with multiple images."""
        image1 = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data1",
            mime_type="image/png"
        )
        image2 = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64data2",
            mime_type="image/jpeg"
        )

        request = LLMRequest(
            user_prompt="Compare these images",
            multimodal_inputs=[image1, image2]
        )

        contents = self.provider._build_contents(request)

        parts = contents[0]["parts"]
        assert len(parts) == 3  # 2 images + 1 text
        assert "inline_data" in parts[0]
        assert "inline_data" in parts[1]
        assert "text" in parts[2]

    def test_build_contents_with_document(self):
        """Test _build_contents with PDF document."""
        doc_input = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.BASE64,
            data="base64pdfdata",
            mime_type="application/pdf"
        )

        request = LLMRequest(
            user_prompt="Summarize this document",
            multimodal_inputs=[doc_input]
        )

        contents = self.provider._build_contents(request)

        parts = contents[0]["parts"]
        assert len(parts) == 2  # Document + text
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "application/pdf"

    def test_build_contents_with_urls(self):
        """Test _build_contents with URL context."""
        request = LLMRequest(
            user_prompt="Summarize these articles",
            urls=["https://example.com/article1", "https://example.com/article2"]
        )

        contents = self.provider._build_contents(request)

        parts = contents[0]["parts"]
        assert len(parts) == 1  # Just text (URLs are embedded in text)
        text_content = parts[0]["text"]
        assert "https://example.com/article1" in text_content
        assert "https://example.com/article2" in text_content

    def test_build_contents_mixed_inputs(self):
        """Test _build_contents with images + documents + URLs."""
        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="imagedata",
            mime_type="image/png"
        )
        doc = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.BASE64,
            data="pdfdata",
            mime_type="application/pdf"
        )

        request = LLMRequest(
            user_prompt="Analyze all sources",
            multimodal_inputs=[image, doc],
            urls=["https://example.com/context"]
        )

        contents = self.provider._build_contents(request)

        parts = contents[0]["parts"]
        assert len(parts) == 3  # Image + document + text (with URL)
        assert "inline_data" in parts[0]  # Image
        assert "inline_data" in parts[1]  # Document
        assert "text" in parts[2]
        assert "https://example.com/context" in parts[2]["text"]

    # ============================================================================
    # Tests for _create_multimodal_part()
    # ============================================================================

    def test_create_multimodal_part_base64_source(self):
        """Test _create_multimodal_part with BASE64 source type."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64encodedimagedata",
            mime_type="image/jpeg"
        )

        part = self.provider._create_multimodal_part(input_item)

        assert "inline_data" in part
        assert part["inline_data"]["mime_type"] == "image/jpeg"
        assert part["inline_data"]["data"] == "base64encodedimagedata"

    def test_create_multimodal_part_file_path_source(self):
        """Test _create_multimodal_part with FILE_PATH source type."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png"
        )

        # Mock the read_file_as_base64 function - patch where it's imported
        with patch('mad_spark_alt.utils.multimodal_utils.read_file_as_base64') as mock_read:
            mock_read.return_value = ("mockedbase64data", "image/png")

            part = self.provider._create_multimodal_part(input_item)

            mock_read.assert_called_once_with("/path/to/image.png")
            assert "inline_data" in part
            assert part["inline_data"]["mime_type"] == "image/png"
            assert part["inline_data"]["data"] == "mockedbase64data"

    def test_create_multimodal_part_file_api_source(self):
        """Test _create_multimodal_part with FILE_API source type."""
        input_item = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_API,
            data="gs://bucket/file-id",
            mime_type="application/pdf"
        )

        part = self.provider._create_multimodal_part(input_item)

        assert "file_data" in part
        assert part["file_data"]["file_uri"] == "gs://bucket/file-id"
        assert part["file_data"]["mime_type"] == "application/pdf"

    # ============================================================================
    # Tests for _add_url_context_tool()
    # ============================================================================

    def test_add_url_context_tool_no_urls(self):
        """Test _add_url_context_tool doesn't add tool when no URLs."""
        payload = {
            "contents": [],
            "generationConfig": {}
        }
        request = LLMRequest(user_prompt="Test")

        self.provider._add_url_context_tool(payload, request)

        assert "tools" not in payload

    def test_add_url_context_tool_with_urls(self):
        """Test _add_url_context_tool adds tool when URLs present."""
        payload = {
            "contents": [],
            "generationConfig": {}
        }
        request = LLMRequest(
            user_prompt="Test",
            urls=["https://example.com"]
        )

        self.provider._add_url_context_tool(payload, request)

        assert "tools" in payload
        assert payload["tools"] == [{"url_context": {}}]

    # ============================================================================
    # Tests for _parse_url_context_metadata()
    # ============================================================================

    def test_parse_url_context_metadata_none(self):
        """Test _parse_url_context_metadata returns None when no metadata."""
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "Response"}]}}]
        }

        result = self.provider._parse_url_context_metadata(response_data)

        assert result is None

    def test_parse_url_context_metadata_success(self):
        """Test _parse_url_context_metadata with successful retrieval."""
        response_data = {
            "candidates": [{"content": {"parts": [{"text": "Response"}]}}],
            "url_context_metadata": [
                {
                    "url": "https://example.com/article",
                    "status": "success"
                }
            ]
        }

        result = self.provider._parse_url_context_metadata(response_data)

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], URLContextMetadata)
        assert result[0].url == "https://example.com/article"
        assert result[0].status == "success"
        assert result[0].error_message is None

    def test_parse_url_context_metadata_failed(self):
        """Test _parse_url_context_metadata with failed retrieval."""
        response_data = {
            "url_context_metadata": [
                {
                    "url": "https://invalid-url.com",
                    "status": "failed",
                    "error_message": "Connection timeout"
                }
            ]
        }

        result = self.provider._parse_url_context_metadata(response_data)

        assert len(result) == 1
        assert result[0].status == "failed"
        assert result[0].error_message == "Connection timeout"

    def test_parse_url_context_metadata_blocked(self):
        """Test _parse_url_context_metadata with blocked retrieval."""
        response_data = {
            "url_context_metadata": [
                {
                    "url": "https://blocked.com",
                    "status": "blocked",
                    "error_message": "Safety filter"
                }
            ]
        }

        result = self.provider._parse_url_context_metadata(response_data)

        assert len(result) == 1
        assert result[0].status == "blocked"

    # ============================================================================
    # End-to-end tests with mocked API
    # ============================================================================

    @pytest.mark.asyncio
    async def test_generate_with_image_mocked(self):
        """Test end-to-end generation with image input (mocked API)."""
        image_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.BASE64,
            data="base64imagedata",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Describe this image",
            multimodal_inputs=[image_input]
        )

        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "This image shows a diagram."}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 500,
                "candidatesTokenCount": 20,
                "totalTokenCount": 520
            }
        })

        with patch('aiohttp.ClientSession.post', new=AsyncMock(return_value=mock_response)):
            response = await self.provider.generate(request)

            assert response.content == "This image shows a diagram."
            assert response.provider == LLMProvider.GOOGLE
            assert response.total_images_processed == 1
            assert response.total_pages_processed is None
            assert response.url_context_metadata is None

    @pytest.mark.asyncio
    async def test_generate_with_document_mocked(self):
        """Test end-to-end generation with document input (mocked API)."""
        doc_input = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.BASE64,
            data="base64pdfdata",
            mime_type="application/pdf",
            page_count=10
        )

        request = LLMRequest(
            user_prompt="Summarize this document",
            multimodal_inputs=[doc_input]
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Document summary: Key findings..."}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 3000,
                "candidatesTokenCount": 100,
                "totalTokenCount": 3100
            }
        })

        with patch('aiohttp.ClientSession.post', new=AsyncMock(return_value=mock_response)):
            response = await self.provider.generate(request)

            assert response.content == "Document summary: Key findings..."
            assert response.total_images_processed is None
            assert response.total_pages_processed == 10

    @pytest.mark.asyncio
    async def test_generate_with_urls_mocked(self):
        """Test end-to-end generation with URL context (mocked API)."""
        request = LLMRequest(
            user_prompt="Summarize these articles",
            urls=["https://example.com/article1", "https://example.com/article2"]
        )

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Articles discuss AI trends..."}]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 5000,
                "candidatesTokenCount": 150,
                "totalTokenCount": 5150
            },
            "url_context_metadata": [
                {"url": "https://example.com/article1", "status": "success"},
                {"url": "https://example.com/article2", "status": "success"}
            ]
        })

        with patch('aiohttp.ClientSession.post', new=AsyncMock(return_value=mock_response)):
            response = await self.provider.generate(request)

            assert response.content == "Articles discuss AI trends..."
            assert response.url_context_metadata is not None
            assert len(response.url_context_metadata) == 2
            assert response.url_context_metadata[0].status == "success"
