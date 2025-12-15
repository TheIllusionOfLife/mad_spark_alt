"""
Unit tests for PhaseInput multimodal support.

Tests verify that PhaseInput correctly accepts and validates multimodal inputs
(images, documents, URLs) and passes them through the QADI pipeline.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import Mock

from mad_spark_alt.core.phase_logic import PhaseInput
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
)


class TestPhaseInputMultimodalFields:
    """Test that PhaseInput accepts multimodal fields."""

    def test_phase_input_with_multimodal_inputs(self):
        """PhaseInput should accept multimodal_inputs parameter."""
        # Arrange
        mock_llm_manager = Mock()
        image_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
        )

        # Act
        phase_input = PhaseInput(
            user_input="Test question",
            llm_manager=mock_llm_manager,
            multimodal_inputs=[image_input],
        )

        # Assert
        assert phase_input.multimodal_inputs is not None
        assert len(phase_input.multimodal_inputs) == 1
        assert phase_input.multimodal_inputs[0] == image_input

    def test_phase_input_with_urls(self):
        """PhaseInput should accept urls parameter."""
        # Arrange
        mock_llm_manager = Mock()
        urls = ["https://example.com/article1", "https://example.com/article2"]

        # Act
        phase_input = PhaseInput(
            user_input="Test question", llm_manager=mock_llm_manager, urls=urls
        )

        # Assert
        assert phase_input.urls is not None
        assert len(phase_input.urls) == 2
        assert phase_input.urls == urls

    def test_phase_input_with_tools(self):
        """PhaseInput should accept tools parameter."""
        # Arrange
        mock_llm_manager = Mock()
        tools = [{"url_context": {}}]

        # Act
        phase_input = PhaseInput(
            user_input="Test question", llm_manager=mock_llm_manager, tools=tools
        )

        # Assert
        assert phase_input.tools is not None
        assert len(phase_input.tools) == 1
        assert phase_input.tools[0] == {"url_context": {}}

    def test_phase_input_with_all_multimodal_fields(self):
        """PhaseInput should accept all multimodal fields together."""
        # Arrange
        mock_llm_manager = Mock()
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
        tools = [{"url_context": {}}]

        # Act
        phase_input = PhaseInput(
            user_input="Test question",
            llm_manager=mock_llm_manager,
            multimodal_inputs=[image, document],
            urls=urls,
            tools=tools,
        )

        # Assert
        assert phase_input.multimodal_inputs is not None
        assert len(phase_input.multimodal_inputs) == 2
        assert phase_input.urls == urls
        assert phase_input.tools == tools

    def test_phase_input_multimodal_fields_default_to_none(self):
        """Multimodal fields should default to None if not provided."""
        # Arrange
        mock_llm_manager = Mock()

        # Act
        phase_input = PhaseInput(
            user_input="Test question", llm_manager=mock_llm_manager
        )

        # Assert
        assert phase_input.multimodal_inputs is None
        assert phase_input.urls is None
        assert phase_input.tools is None


class TestPhaseResultMultimodalMetadata:
    """Test that phase result dataclasses track multimodal metadata."""

    def test_questioning_result_has_multimodal_metadata(self):
        """QuestioningResult should have multimodal_metadata field."""
        from mad_spark_alt.core.phase_logic import QuestioningResult

        # Act
        result = QuestioningResult(
            core_question="What is X?",
            llm_cost=0.001,
            raw_response="Response text",
            multimodal_metadata={
                "images_processed": 1,
                "pages_processed": 0,
                "urls_processed": 0,
            },
        )

        # Assert
        assert result.multimodal_metadata is not None
        assert result.multimodal_metadata["images_processed"] == 1
        assert result.multimodal_metadata["pages_processed"] == 0

    def test_abduction_result_has_multimodal_metadata(self):
        """AbductionResult should have multimodal_metadata field."""
        from mad_spark_alt.core.phase_logic import AbductionResult

        # Act
        result = AbductionResult(
            hypotheses=["H1", "H2", "H3"],
            llm_cost=0.002,
            raw_response="Response text",
            num_requested=3,
            num_generated=3,
            multimodal_metadata={
                "images_processed": 2,
                "pages_processed": 10,
                "urls_processed": 1,
            },
        )

        # Assert
        assert result.multimodal_metadata is not None
        assert result.multimodal_metadata["images_processed"] == 2
        assert result.multimodal_metadata["pages_processed"] == 10

    def test_deduction_result_has_multimodal_metadata(self):
        """DeductionResult should have multimodal_metadata field."""
        from mad_spark_alt.core.phase_logic import DeductionResult

        # Act
        result = DeductionResult(
            hypothesis_scores=[],
            answer="Answer text",
            action_plan=["Step 1", "Step 2"],
            llm_cost=0.003,
            raw_response="Response text",
            used_parallel=False,
            multimodal_metadata={
                "images_processed": 0,
                "pages_processed": 5,
                "urls_processed": 2,
            },
        )

        # Assert
        assert result.multimodal_metadata is not None
        assert result.multimodal_metadata["urls_processed"] == 2

    def test_induction_result_has_multimodal_metadata(self):
        """InductionResult should have multimodal_metadata field."""
        from mad_spark_alt.core.phase_logic import InductionResult

        # Act - synthesis is now the primary field
        result = InductionResult(
            synthesis="Synthesis text that ties together all findings",
            llm_cost=0.001,
            raw_response="Response text",
            multimodal_metadata={
                "images_processed": 1,
                "pages_processed": 0,
                "urls_processed": 1,
            },
        )

        # Assert
        assert result.multimodal_metadata is not None
        assert result.multimodal_metadata["images_processed"] == 1
        # Backward compat fields
        assert result.examples == []
        assert result.conclusion == ""  # Default empty, not set from synthesis automatically

    def test_phase_results_multimodal_metadata_defaults_to_empty_dict(self):
        """Multimodal metadata should default to empty dict if not provided."""
        from mad_spark_alt.core.phase_logic import QuestioningResult

        # Act
        result = QuestioningResult(
            core_question="What is X?", llm_cost=0.001, raw_response="Response text"
        )

        # Assert
        assert result.multimodal_metadata == {}


class TestPhaseInputContextPreservation:
    """Test that multimodal data is preserved through phase chain."""

    def test_multimodal_inputs_preserved_in_context(self):
        """Multimodal inputs should be preserved when creating new PhaseInput."""
        # Arrange
        mock_llm_manager = Mock()
        image = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="/path/to/image.png",
            mime_type="image/png",
        )

        phase_input_1 = PhaseInput(
            user_input="Question 1",
            llm_manager=mock_llm_manager,
            multimodal_inputs=[image],
            urls=["https://example.com"],
        )

        # Act - Create new PhaseInput with same multimodal data
        phase_input_2 = PhaseInput(
            user_input="Question 2",
            llm_manager=mock_llm_manager,
            multimodal_inputs=phase_input_1.multimodal_inputs,
            urls=phase_input_1.urls,
            context={"previous_result": "some data"},
        )

        # Assert
        assert phase_input_2.multimodal_inputs == phase_input_1.multimodal_inputs
        assert phase_input_2.urls == phase_input_1.urls
        assert phase_input_2.multimodal_inputs[0] == image

    def test_empty_multimodal_data_preserved(self):
        """Empty multimodal data (None) should be preserved."""
        # Arrange
        mock_llm_manager = Mock()

        phase_input_1 = PhaseInput(
            user_input="Question 1", llm_manager=mock_llm_manager
        )

        # Act
        phase_input_2 = PhaseInput(
            user_input="Question 2",
            llm_manager=mock_llm_manager,
            multimodal_inputs=phase_input_1.multimodal_inputs,
            urls=phase_input_1.urls,
        )

        # Assert
        assert phase_input_2.multimodal_inputs is None
        assert phase_input_2.urls is None
