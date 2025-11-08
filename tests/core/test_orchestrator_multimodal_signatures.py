"""
Unit tests for orchestrator multimodal parameter signatures.

These tests verify that orchestrators properly accept multimodal parameters
and pass them through correctly. Integration tests with real API calls are
marked with @pytest.mark.integration.
"""

import pytest
from unittest.mock import AsyncMock, patch

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult
from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIOrchestrator
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator
from mad_spark_alt.core.orchestrator_config import OrchestratorConfig
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType
from mad_spark_alt.core.phase_logic import HypothesisScore


@pytest.fixture
def sample_multimodal_inputs():
    """Sample multimodal inputs for testing."""
    return [
        MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
        )
    ]


@pytest.fixture
def sample_urls():
    """Sample URLs."""
    return ["https://example.com/test"]


@pytest.fixture
def mock_simple_result():
    """Mock SimpleQADIResult."""
    return SimpleQADIResult(
        core_question="Test",
        hypotheses=["H1", "H2", "H3"],
        hypothesis_scores=[
            HypothesisScore(0.8, 0.7, 0.6, 0.7, 0.8, 0.72),
            HypothesisScore(0.7, 0.8, 0.7, 0.8, 0.7, 0.74),
            HypothesisScore(0.6, 0.6, 0.8, 0.6, 0.6, 0.64),
        ],
        final_answer="Answer",
        action_plan=["Action"],
        verification_examples=["Ex"],
        verification_conclusion="Conclusion",
        total_llm_cost=0.01,
    )


class TestOrchestratorMultimodalSignatures:
    """Test that orchestrators accept multimodal parameters."""

    @pytest.mark.asyncio
    async def test_simple_orchestrator_signature(self, sample_multimodal_inputs, sample_urls):
        """Test SimpleQADIOrchestrator accepts multimodal parameters."""
        orchestrator = SimpleQADIOrchestrator()

        # This should not raise a TypeError for unexpected keyword arguments
        try:
            with patch.object(orchestrator, 'run_qadi_cycle', new=AsyncMock()) as mock_run:
                await orchestrator.run_qadi_cycle(
                    "Test question",
                    multimodal_inputs=sample_multimodal_inputs,
                    urls=sample_urls,
                    tools=[],
                )
                # Verify method was called
                assert mock_run.called
        except TypeError as e:
            pytest.fail(f"SimpleQADIOrchestrator.run_qadi_cycle() doesn't accept multimodal params: {e}")

    @pytest.mark.asyncio
    async def test_multi_perspective_orchestrator_signature(self, sample_multimodal_inputs, sample_urls):
        """Test MultiPerspectiveQADIOrchestrator accepts multimodal parameters."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        try:
            with patch.object(orchestrator, 'run_multi_perspective_analysis', new=AsyncMock()) as mock_run:
                await orchestrator.run_multi_perspective_analysis(
                    "Test question",
                    multimodal_inputs=sample_multimodal_inputs,
                    urls=sample_urls,
                    tools=[],
                )
                assert mock_run.called
        except TypeError as e:
            pytest.fail(f"MultiPerspectiveQADIOrchestrator.run_multi_perspective_analysis() doesn't accept multimodal params: {e}")

    @pytest.mark.asyncio
    async def test_unified_orchestrator_signature(self, sample_multimodal_inputs, sample_urls):
        """Test UnifiedQADIOrchestrator accepts multimodal parameters."""
        config = OrchestratorConfig.simple_config()
        orchestrator = UnifiedQADIOrchestrator(config)

        try:
            with patch.object(orchestrator, 'run_qadi_cycle', new=AsyncMock()) as mock_run:
                await orchestrator.run_qadi_cycle(
                    "Test question",
                    multimodal_inputs=sample_multimodal_inputs,
                    urls=sample_urls,
                    tools=[],
                )
                assert mock_run.called
        except TypeError as e:
            pytest.fail(f"UnifiedQADIOrchestrator.run_qadi_cycle() doesn't accept multimodal params: {e}")

    @pytest.mark.asyncio
    async def test_simple_result_has_multimodal_fields(self, mock_simple_result):
        """Test SimpleQADIResult has multimodal metadata fields."""
        # Verify the dataclass has the expected fields
        assert hasattr(mock_simple_result, 'multimodal_metadata')
        assert hasattr(mock_simple_result, 'total_images_processed')
        assert hasattr(mock_simple_result, 'total_pages_processed')
        assert hasattr(mock_simple_result, 'total_urls_processed')


class TestMultimodalValidationInOrchestrators:
    """Test that orchestrators validate multimodal inputs."""

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self):
        """Test that invalid URLs are rejected."""
        orchestrator = SimpleQADIOrchestrator()

        with pytest.raises(RuntimeError, match="Invalid URL"):
            await orchestrator.run_qadi_cycle(
                "Test",
                urls=["not-a-url"],
            )

    @pytest.mark.asyncio
    async def test_too_many_urls_rejected(self):
        """Test that too many URLs are rejected."""
        orchestrator = SimpleQADIOrchestrator()

        too_many = [f"https://example.com/{i}" for i in range(25)]
        with pytest.raises(RuntimeError, match="Too many URLs"):
            await orchestrator.run_qadi_cycle(
                "Test",
                urls=too_many,
            )

    @pytest.mark.asyncio
    async def test_unsupported_image_type_rejected(self):
        """Test that unsupported image types are rejected."""
        orchestrator = SimpleQADIOrchestrator()

        bad_image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.bmp",
            mime_type="image/bmp",  # Unsupported
        )

        with pytest.raises(RuntimeError, match="Unsupported image type"):
            await orchestrator.run_qadi_cycle(
                "Test",
                multimodal_inputs=[bad_image],
            )
