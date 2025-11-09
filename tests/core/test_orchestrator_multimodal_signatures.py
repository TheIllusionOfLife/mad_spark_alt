"""
Unit tests for orchestrator multimodal parameter signatures.

These tests verify that orchestrators properly accept multimodal parameters
by inspecting actual method signatures. This ensures the methods truly
accept multimodal parameters rather than just mocking them.
"""

import inspect
import pytest

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIOrchestrator
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator
from mad_spark_alt.core.orchestrator_config import OrchestratorConfig
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType


class TestOrchestratorMultimodalSignatures:
    """Test that orchestrators accept multimodal parameters by inspecting signatures."""

    def test_simple_orchestrator_signature(self):
        """Test SimpleQADIOrchestrator.run_qadi_cycle() has multimodal parameters."""
        sig = inspect.signature(SimpleQADIOrchestrator.run_qadi_cycle)

        # Verify multimodal parameters are in the signature
        expected_params = ["multimodal_inputs", "urls", "tools"]
        for param_name in expected_params:
            assert param_name in sig.parameters, (
                f"Parameter '{param_name}' not found in SimpleQADIOrchestrator.run_qadi_cycle() signature. "
                f"Available parameters: {list(sig.parameters.keys())}"
            )

    def test_multi_perspective_orchestrator_signature(self):
        """Test MultiPerspectiveQADIOrchestrator.run_multi_perspective_analysis() has multimodal parameters."""
        sig = inspect.signature(MultiPerspectiveQADIOrchestrator.run_multi_perspective_analysis)

        # Verify multimodal parameters are in the signature
        expected_params = ["multimodal_inputs", "urls", "tools"]
        for param_name in expected_params:
            assert param_name in sig.parameters, (
                f"Parameter '{param_name}' not found in MultiPerspectiveQADIOrchestrator.run_multi_perspective_analysis() signature. "
                f"Available parameters: {list(sig.parameters.keys())}"
            )

    def test_unified_orchestrator_signature(self):
        """Test UnifiedQADIOrchestrator.run_qadi_cycle() has multimodal parameters."""
        sig = inspect.signature(UnifiedQADIOrchestrator.run_qadi_cycle)

        # Verify multimodal parameters are in the signature
        expected_params = ["multimodal_inputs", "urls", "tools"]
        for param_name in expected_params:
            assert param_name in sig.parameters, (
                f"Parameter '{param_name}' not found in UnifiedQADIOrchestrator.run_qadi_cycle() signature. "
                f"Available parameters: {list(sig.parameters.keys())}"
            )


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
