"""
Integration tests for GoogleProvider multimodal support with real Gemini API.

These tests require GOOGLE_API_KEY environment variable to be set.
They make real API calls and incur small costs (~$0.01 total).

Run with: pytest tests/test_google_provider_multimodal_integration.py -v
"""

import os
from pathlib import Path

import pytest

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMRequest,
)
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
)


# Skip all tests if GOOGLE_API_KEY not set
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY environment variable not set"
)


@pytest.fixture
def google_provider():
    """Create GoogleProvider with real API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleProvider(api_key=api_key)


@pytest.fixture
def test_image_path():
    """Path to test image fixture."""
    return str(Path(__file__).parent / "fixtures" / "test_image.png")


@pytest.fixture
def test_document_path():
    """Path to test document fixture."""
    return str(Path(__file__).parent / "fixtures" / "test_document.pdf")


class TestGoogleProviderMultimodalIntegration:
    """Integration tests with real Gemini API."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_image_analysis(self, google_provider, test_image_path):
        """
        Test real image analysis with Gemini API.

        Cost: ~$0.002
        """
        # Create multimodal input from test image
        image_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=test_image_path,
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="Describe what you see in this image. What shapes and colors are present?",
            multimodal_inputs=[image_input],
            max_tokens=500
        )

        # Make real API call
        response = await google_provider.generate(request)

        # Verify response structure
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == LLMProvider.GOOGLE
        assert response.model == "gemini-2.5-flash"

        # Verify multimodal metadata
        assert response.total_images_processed == 1
        assert response.total_pages_processed is None
        assert response.url_context_metadata is None

        # Verify cost tracking
        assert response.cost > 0
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0

        # Verify content quality (should mention shapes/colors)
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["blue", "red", "square", "circle", "shape", "color"])

        print(f"\n✓ Image Analysis Response (cost: ${response.cost:.4f}):")
        print(f"  {response.content[:200]}...")
        print(f"  Tokens: {response.usage['prompt_tokens']} input, {response.usage['completion_tokens']} output")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_document_processing(self, google_provider, test_document_path):
        """
        Test real PDF processing with Gemini API.

        Cost: ~$0.003
        """
        # Create multimodal input from test PDF
        doc_input = MultimodalInput(
            input_type=MultimodalInputType.DOCUMENT,
            source_type=MultimodalSourceType.FILE_PATH,
            data=test_document_path,
            mime_type="application/pdf",
            page_count=3
        )

        request = LLMRequest(
            user_prompt="Summarize this document. How many pages does it have and what is the main topic?",
            multimodal_inputs=[doc_input],
            max_tokens=500
        )

        # Make real API call
        response = await google_provider.generate(request)

        # Verify response structure
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Verify multimodal metadata
        assert response.total_images_processed is None
        assert response.total_pages_processed == 3
        assert response.url_context_metadata is None

        # Verify cost tracking (PDF should use more tokens)
        assert response.cost > 0
        assert response.usage["prompt_tokens"] > 500  # PDF pages add tokens

        # Verify content quality (should mention 3 pages and testing)
        content_lower = response.content.lower()
        assert any(word in content_lower for word in ["three", "3", "page"])
        assert any(word in content_lower for word in ["test", "document", "sample"])

        print(f"\n✓ Document Processing Response (cost: ${response.cost:.4f}):")
        print(f"  {response.content[:200]}...")
        print(f"  Tokens: {response.usage['prompt_tokens']} input, {response.usage['completion_tokens']} output")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_url_context(self, google_provider):
        """
        Test real URL context retrieval with Gemini API.

        Cost: ~$0.003

        Note: Using a reliable public URL that's unlikely to change.
        """
        # Use a stable public URL for testing
        test_url = "https://www.example.com"

        request = LLMRequest(
            user_prompt="What is the main purpose of this website based on its content?",
            urls=[test_url],
            max_tokens=300
        )

        # Make real API call
        response = await google_provider.generate(request)

        # Verify response structure
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Verify multimodal metadata
        assert response.total_images_processed is None
        assert response.total_pages_processed is None

        # Note: url_context_metadata may be None even when URL content is successfully used
        # Gemini doesn't always return metadata, but the content shows it used the URL
        content_lower = response.content.lower()
        url_used = any(word in content_lower for word in ["example", "domain", "illustrative", "documentation"])

        # Either metadata exists OR content shows URL was used
        has_url_evidence = (response.url_context_metadata is not None) or url_used
        assert has_url_evidence, "No evidence that URL was used (no metadata and no URL content in response)"

        print(f"\n✓ URL Context Response (cost: ${response.cost:.4f}):")
        if response.url_context_metadata:
            print(f"  URL Status: {response.url_context_metadata[0].status}")
        else:
            print(f"  URL metadata not returned (but content shows URL was used)")
        print(f"  {response.content[:200]}...")
        print(f"  Tokens: {response.usage['prompt_tokens']} input, {response.usage['completion_tokens']} output")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_mixed_multimodal(self, google_provider, test_image_path):
        """
        Test real API with mixed multimodal inputs (image + URL).

        Cost: ~$0.005
        """
        # Create image input
        image_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=test_image_path,
            mime_type="image/png"
        )

        # Use stable public URL
        test_url = "https://www.example.com"

        request = LLMRequest(
            user_prompt="I've provided an image and a URL. Briefly describe what you see in the image and what the URL contains.",
            multimodal_inputs=[image_input],
            urls=[test_url],
            max_tokens=500
        )

        # Make real API call
        response = await google_provider.generate(request)

        # Verify response structure
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 50  # Should be substantive

        # Verify multimodal metadata
        assert response.total_images_processed == 1
        assert response.total_pages_processed is None

        # Verify cost tracking (mixed inputs use more tokens)
        assert response.cost > 0

        # Verify content mentions both inputs
        content_lower = response.content.lower()
        # Should mention something about the image
        has_image_content = any(word in content_lower for word in ["image", "shape", "color", "blue", "red", "square", "circle"])
        # Should mention something about the URL
        has_url_content = any(word in content_lower for word in ["example", "website", "page", "domain"])

        assert has_image_content and has_url_content, "Response should mention both image and URL content"

        print(f"\n✓ Mixed Multimodal Response (cost: ${response.cost:.4f}):")
        print(f"  Image: {response.total_images_processed} processed")
        if response.url_context_metadata:
            print(f"  URL Status: {response.url_context_metadata[0].status}")
        else:
            print(f"  URL metadata not returned (but content shows URL was used)")
        print(f"  {response.content[:300]}...")
        print(f"  Tokens: {response.usage['prompt_tokens']} input, {response.usage['completion_tokens']} output")


class TestMultimodalBackwardCompatibility:
    """Test that text-only requests still work (backward compatibility)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_text_only_still_works(self, google_provider):
        """
        Verify text-only requests work after multimodal changes.

        Cost: ~$0.001
        """
        request = LLMRequest(
            user_prompt="What is 2 + 2? Answer in one sentence.",
            max_tokens=50
        )

        response = await google_provider.generate(request)

        # Verify basic functionality
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Verify multimodal fields are None for text-only
        assert response.total_images_processed is None
        assert response.total_pages_processed is None
        assert response.url_context_metadata is None

        # Verify answer is correct
        assert "4" in response.content

        print(f"\n✓ Text-Only Backward Compatibility (cost: ${response.cost:.4f}):")
        print(f"  {response.content}")
