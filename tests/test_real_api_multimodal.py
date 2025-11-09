"""
Integration tests for multimodal support with real Gemini API.

These tests require GOOGLE_API_KEY to be set and are marked with @pytest.mark.integration.
Run with: GOOGLE_API_KEY=xxx pytest tests/test_real_api_multimodal.py -v
"""

import os
import pytest
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType


pytestmark = pytest.mark.integration


@pytest.fixture
def test_image_path():
    """Create a simple test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    # Create a simple chart image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)

    # Draw simple chart
    draw.rectangle([50, 50, 350, 250], outline='black', width=2)
    draw.text((70, 70), "Financial Chart", fill='black')
    draw.text((70, 100), "Revenue: $100K", fill='blue')
    draw.text((70, 130), "Costs: $60K", fill='red')
    draw.text((70, 160), "Profit: $40K", fill='green')

    # Draw bars
    draw.rectangle([70, 190, 170, 230], fill='blue')
    draw.rectangle([180, 200, 250, 230], fill='red')
    draw.rectangle([260, 180, 330, 230], fill='green')

    img.save(tmp_path)

    yield tmp_path

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if GOOGLE_API_KEY is not set."""
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not available")


class TestRealAPIImageProcessing:
    """Test image processing with real Gemini API."""

    @pytest.mark.asyncio
    async def test_simple_qadi_with_image(self, test_image_path, skip_if_no_api_key):
        """Test SimpleQADI with real image input."""
        orchestrator = SimpleQADIOrchestrator()

        # Create multimodal input
        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=str(Path(test_image_path).absolute()),
            mime_type="image/png",
            file_size=Path(test_image_path).stat().st_size,
        )

        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            user_input="Analyze the financial data in this chart and suggest improvements",
            multimodal_inputs=[multimodal_input],
        )

        # Verify result structure
        assert result is not None
        assert result.core_question is not None
        assert len(result.hypotheses) > 0
        assert result.final_answer is not None

        # Verify multimodal metadata
        assert result.total_images_processed == 1
        assert result.total_pages_processed == 0
        assert result.total_urls_processed == 0

        # Verify LLM cost tracking
        assert result.total_llm_cost > 0

        # Verify the answer references the chart data
        answer_lower = result.final_answer.lower()
        assert any(keyword in answer_lower for keyword in ['revenue', 'cost', 'profit', 'financial', 'chart'])

    @pytest.mark.asyncio
    async def test_simple_qadi_with_multiple_images(self, skip_if_no_api_key):
        """Test SimpleQADI with multiple images."""
        orchestrator = SimpleQADIOrchestrator()

        # Create two test images
        images = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            img = Image.new('RGB', (300, 200), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), f"Chart {i+1}", fill='black')
            draw.text((50, 80), f"Data Point: {(i+1)*100}", fill='blue')
            img.save(tmp_path)
            images.append(tmp_path)

        try:
            # Create multimodal inputs
            multimodal_inputs = [
                MultimodalInput(
                    input_type=MultimodalInputType.IMAGE,
                    source_type=MultimodalSourceType.FILE_PATH,
                    data=str(Path(img_path).absolute()),
                    mime_type="image/png",
                    file_size=Path(img_path).stat().st_size,
                )
                for img_path in images
            ]

            # Run QADI cycle
            result = await orchestrator.run_qadi_cycle(
                user_input="Compare the data in these two charts",
                multimodal_inputs=multimodal_inputs,
            )

            # Verify result
            assert result is not None
            assert result.total_images_processed == 2
            assert result.final_answer is not None

        finally:
            for img_path in images:
                Path(img_path).unlink(missing_ok=True)


class TestRealAPIURLProcessing:
    """Test URL processing with real Gemini API."""

    @pytest.mark.asyncio
    async def test_simple_qadi_with_url(self, skip_if_no_api_key):
        """Test SimpleQADI with URL input."""
        orchestrator = SimpleQADIOrchestrator()

        # Use a stable, simple URL for testing
        test_url = "https://www.example.com"

        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            user_input="What is the purpose of this website?",
            urls=[test_url],
        )

        # Verify result structure
        assert result is not None
        assert result.core_question is not None
        assert len(result.hypotheses) > 0
        assert result.final_answer is not None

        # Verify multimodal metadata
        assert result.total_urls_processed == 1
        assert result.total_images_processed == 0
        assert result.total_pages_processed == 0

        # Verify LLM cost tracking
        assert result.total_llm_cost > 0


class TestRealAPICombinedMultimodal:
    """Test combined multimodal inputs with real API."""

    @pytest.mark.asyncio
    async def test_simple_qadi_with_image_and_url(self, test_image_path, skip_if_no_api_key):
        """Test SimpleQADI with both image and URL inputs."""
        orchestrator = SimpleQADIOrchestrator()

        # Create multimodal input
        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=str(Path(test_image_path).absolute()),
            mime_type="image/png",
            file_size=Path(test_image_path).stat().st_size,
        )

        test_url = "https://www.example.com"

        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            user_input="Analyze the financial chart and provide context from the URL",
            multimodal_inputs=[multimodal_input],
            urls=[test_url],
        )

        # Verify result structure
        assert result is not None
        assert result.final_answer is not None

        # Verify all modalities were processed
        assert result.total_images_processed == 1
        assert result.total_urls_processed == 1

        # Verify LLM cost tracking
        assert result.total_llm_cost > 0


class TestRealAPIMetadataTracking:
    """Test metadata tracking across QADI phases."""

    @pytest.mark.asyncio
    async def test_metadata_aggregation_across_phases(self, test_image_path, skip_if_no_api_key):
        """Test that metadata is properly aggregated from all QADI phases."""
        orchestrator = SimpleQADIOrchestrator()

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=str(Path(test_image_path).absolute()),
            mime_type="image/png",
            file_size=Path(test_image_path).stat().st_size,
        )

        result = await orchestrator.run_qadi_cycle(
            user_input="Analyze this financial chart",
            multimodal_inputs=[multimodal_input],
        )

        # Verify metadata structure
        assert hasattr(result, 'multimodal_metadata')
        assert isinstance(result.multimodal_metadata, dict)

        # Verify phase-specific metadata exists
        assert 'questioning' in result.multimodal_metadata
        assert 'abduction' in result.multimodal_metadata
        assert 'deduction' in result.multimodal_metadata
        assert 'induction' in result.multimodal_metadata

        # Verify total counts
        assert result.total_images_processed == 1


class TestRealAPIErrorHandling:
    """Test error handling with real API."""

    @pytest.mark.asyncio
    async def test_invalid_url_handling(self, skip_if_no_api_key):
        """Test that invalid URLs are properly rejected."""
        orchestrator = SimpleQADIOrchestrator()

        with pytest.raises(RuntimeError, match="Invalid URL"):
            await orchestrator.run_qadi_cycle(
                user_input="Test",
                urls=["not-a-valid-url"],
            )

    @pytest.mark.asyncio
    async def test_too_many_urls_handling(self, skip_if_no_api_key):
        """Test that too many URLs are rejected."""
        orchestrator = SimpleQADIOrchestrator()

        too_many_urls = [f"https://example{i}.com" for i in range(25)]

        with pytest.raises(RuntimeError, match="Too many URLs"):
            await orchestrator.run_qadi_cycle(
                user_input="Test",
                urls=too_many_urls,
            )

    @pytest.mark.asyncio
    async def test_nonexistent_image_path(self, skip_if_no_api_key):
        """Test handling of nonexistent image file."""
        orchestrator = SimpleQADIOrchestrator()

        bad_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/nonexistent/image.png",
            mime_type="image/png",
        )

        # The validation should happen during phase execution
        # This might raise an error or handle gracefully depending on implementation
        try:
            result = await orchestrator.run_qadi_cycle(
                user_input="Analyze this image",
                multimodal_inputs=[bad_input],
            )
            # If it succeeds, it should at least complete without crashing
            assert result is not None
        except (FileNotFoundError, RuntimeError):
            # Expected behavior - file doesn't exist
            pass
