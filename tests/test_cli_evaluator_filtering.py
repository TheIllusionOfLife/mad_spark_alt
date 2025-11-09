"""Tests for CLI --evaluators flag functionality.

This module tests the ability to filter evaluators via the CLI --evaluators flag.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from click.testing import CliRunner

from mad_spark_alt.unified_cli import main
from mad_spark_alt.core.interfaces import (
    EvaluationRequest,
    ModelOutput,
    OutputType,
    EvaluationLayer,
)
from mad_spark_alt.core.evaluator import EvaluationSummary
from mad_spark_alt.core.registry import registry


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_evaluator_registry():
    """Mock evaluator registry with test evaluators."""
    # Save original state
    original_evaluators = registry._evaluators.copy()
    original_instances = registry._instances.copy()
    original_layer_index = {k: v.copy() for k, v in registry._layer_index.items()}
    original_output_type_index = {k: v.copy() for k, v in registry._output_type_index.items()}

    # Clear registry before each test
    registry._evaluators.clear()
    registry._instances.clear()
    registry._layer_index.clear()
    registry._output_type_index.clear()

    # Initialize indices
    for layer in EvaluationLayer:
        registry._layer_index[layer] = set()
    for output_type in OutputType:
        registry._output_type_index[output_type] = set()

    # Create mock evaluator instances
    novelty_eval = MagicMock()
    novelty_eval.name = "novelty_evaluator"
    novelty_eval.layer = EvaluationLayer.QUANTITATIVE
    novelty_eval.supported_output_types = [OutputType.TEXT]

    diversity_eval = MagicMock()
    diversity_eval.name = "diversity_evaluator"
    diversity_eval.layer = EvaluationLayer.QUANTITATIVE
    diversity_eval.supported_output_types = [OutputType.TEXT]

    quality_eval = MagicMock()
    quality_eval.name = "quality_evaluator"
    quality_eval.layer = EvaluationLayer.LLM_JUDGE
    quality_eval.supported_output_types = [OutputType.TEXT]

    # Create mock classes that return instances
    novelty_class = MagicMock(return_value=novelty_eval)
    diversity_class = MagicMock(return_value=diversity_eval)
    quality_class = MagicMock(return_value=quality_eval)

    # Manually add to registry
    registry._evaluators["novelty_evaluator"] = novelty_class
    registry._evaluators["diversity_evaluator"] = diversity_class
    registry._evaluators["quality_evaluator"] = quality_class

    # Update indices
    registry._layer_index[EvaluationLayer.QUANTITATIVE].add("novelty_evaluator")
    registry._layer_index[EvaluationLayer.QUANTITATIVE].add("diversity_evaluator")
    registry._layer_index[EvaluationLayer.LLM_JUDGE].add("quality_evaluator")

    registry._output_type_index[OutputType.TEXT].add("novelty_evaluator")
    registry._output_type_index[OutputType.TEXT].add("diversity_evaluator")
    registry._output_type_index[OutputType.TEXT].add("quality_evaluator")

    yield registry

    # Restore original state
    registry._evaluators.clear()
    registry._evaluators.update(original_evaluators)
    registry._instances.clear()
    registry._instances.update(original_instances)
    registry._layer_index.clear()
    registry._layer_index.update(original_layer_index)
    registry._output_type_index.clear()
    registry._output_type_index.update(original_output_type_index)


@pytest.fixture
def mock_evaluation_result():
    """Mock evaluation result."""
    from mad_spark_alt.core.interfaces import EvaluationResult

    return EvaluationSummary(
        request_id="test-request",
        total_outputs=1,
        total_evaluators=1,
        execution_time=0.1,
        layer_results={
            EvaluationLayer.QUANTITATIVE: [
                EvaluationResult(
                    evaluator_name="novelty_evaluator",
                    layer=EvaluationLayer.QUANTITATIVE,
                    scores={"novelty": 0.8},
                    explanations={"novelty": "Novel idea"},
                    metadata={}
                )
            ]
        },
        aggregate_scores={"overall": 0.8},
    )


class TestEvaluatorFiltering:
    """Test evaluator filtering via --evaluators flag."""

    def test_evaluate_with_single_evaluator(
        self, cli_runner, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test that single evaluator can be selected via --evaluators flag."""
        with patch("mad_spark_alt.unified_cli.CreativityEvaluator") as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate.return_value = mock_evaluation_result
            mock_evaluator_class.return_value = mock_evaluator

            result = cli_runner.invoke(
                main,
                ["evaluate", "--evaluators", "novelty_evaluator", "test text"],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Verify EvaluationRequest was created with evaluator_names
            call_args = mock_evaluator.evaluate.call_args
            request = call_args[0][0] if call_args else None
            assert request is not None, "EvaluationRequest not created"
            assert hasattr(request, "evaluator_names"), "evaluator_names field missing"
            assert request.evaluator_names == ["novelty_evaluator"]

    def test_evaluate_with_multiple_evaluators(
        self, cli_runner, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test that multiple evaluators can be selected via --evaluators flag."""
        with patch("mad_spark_alt.unified_cli.CreativityEvaluator") as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate.return_value = mock_evaluation_result
            mock_evaluator_class.return_value = mock_evaluator

            result = cli_runner.invoke(
                main,
                [
                    "evaluate",
                    "--evaluators",
                    "novelty_evaluator,diversity_evaluator",
                    "test text",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Verify EvaluationRequest was created with multiple evaluators
            call_args = mock_evaluator.evaluate.call_args
            request = call_args[0][0] if call_args else None
            assert request is not None
            assert request.evaluator_names == [
                "novelty_evaluator",
                "diversity_evaluator",
            ]

    def test_evaluate_with_invalid_evaluator(self, cli_runner, mock_evaluator_registry):
        """Test that invalid evaluator names produce helpful error messages."""
        result = cli_runner.invoke(
            main,
            ["evaluate", "--evaluators", "fake_evaluator", "test text"],
        )

        assert result.exit_code != 0, "Should fail with invalid evaluator"
        assert "fake_evaluator" in result.output.lower() or "unknown" in result.output.lower()
        # Should show available evaluators
        assert "novelty_evaluator" in result.output or "available" in result.output.lower()

    def test_evaluate_without_evaluators_flag(
        self, cli_runner, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test backward compatibility: no flag means use all evaluators."""
        with patch("mad_spark_alt.unified_cli.CreativityEvaluator") as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate.return_value = mock_evaluation_result
            mock_evaluator_class.return_value = mock_evaluator

            result = cli_runner.invoke(main, ["evaluate", "test text"])

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Verify EvaluationRequest was created without evaluator_names (None)
            call_args = mock_evaluator.evaluate.call_args
            request = call_args[0][0] if call_args else None
            assert request is not None
            # evaluator_names should be None for backward compatibility
            assert request.evaluator_names is None

    def test_evaluator_filtering_with_whitespace(
        self, cli_runner, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test that evaluator names are trimmed of whitespace."""
        with patch("mad_spark_alt.unified_cli.CreativityEvaluator") as mock_evaluator_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate.return_value = mock_evaluation_result
            mock_evaluator_class.return_value = mock_evaluator

            result = cli_runner.invoke(
                main,
                [
                    "evaluate",
                    "test text",
                    "--evaluators",
                    "  novelty_evaluator  ,  diversity_evaluator  ",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Verify evaluator names are trimmed
            call_args = mock_evaluator.evaluate.call_args
            request = call_args[0][0] if call_args else None
            assert request.evaluator_names == [
                "novelty_evaluator",
                "diversity_evaluator",
            ]


class TestEvaluatorSelectionLogic:
    """Test evaluator selection logic in CreativityEvaluator."""

    @pytest.mark.asyncio
    async def test_selection_respects_evaluator_names(self, mock_evaluator_registry):
        """Test that evaluator selection respects evaluator_names in request."""
        from mad_spark_alt.core.evaluator import CreativityEvaluator

        evaluator = CreativityEvaluator()

        # Create request with specific evaluators
        request = EvaluationRequest(
            outputs=[ModelOutput(content="test", output_type=OutputType.TEXT, model_name="test")],
            target_layers=[EvaluationLayer.QUANTITATIVE],
            evaluator_names=["novelty_evaluator"],  # Only novelty
        )

        # Get evaluators for request
        evaluators_by_layer = evaluator._get_evaluators_for_request(
            request, [EvaluationLayer.QUANTITATIVE]
        )

        # Should only include novelty_evaluator
        quant_evaluators = evaluators_by_layer.get(EvaluationLayer.QUANTITATIVE, [])
        evaluator_names = [e.name for e in quant_evaluators]

        assert "novelty_evaluator" in evaluator_names
        assert "diversity_evaluator" not in evaluator_names
        assert len(evaluator_names) == 1

    @pytest.mark.asyncio
    async def test_selection_without_evaluator_names(self, mock_evaluator_registry):
        """Test that selection uses all evaluators when evaluator_names is None."""
        from mad_spark_alt.core.evaluator import CreativityEvaluator

        evaluator = CreativityEvaluator()

        # Create request without specific evaluators
        request = EvaluationRequest(
            outputs=[ModelOutput(content="test", output_type=OutputType.TEXT, model_name="test")],
            target_layers=[EvaluationLayer.QUANTITATIVE],
            evaluator_names=None,  # Use all
        )

        # Get evaluators for request
        evaluators_by_layer = evaluator._get_evaluators_for_request(
            request, [EvaluationLayer.QUANTITATIVE]
        )

        # Should include all quantitative evaluators
        quant_evaluators = evaluators_by_layer.get(EvaluationLayer.QUANTITATIVE, [])
        evaluator_names = [e.name for e in quant_evaluators]

        assert "novelty_evaluator" in evaluator_names
        assert "diversity_evaluator" in evaluator_names
        assert len(evaluator_names) == 2  # Both quantitative evaluators

    @pytest.mark.asyncio
    async def test_selection_respects_layer_constraints(self, mock_evaluator_registry):
        """Test that evaluator filtering respects layer constraints."""
        from mad_spark_alt.core.evaluator import CreativityEvaluator

        evaluator = CreativityEvaluator()

        # Try to select LLM_JUDGE evaluator for QUANTITATIVE layer
        request = EvaluationRequest(
            outputs=[ModelOutput(content="test", output_type=OutputType.TEXT, model_name="test")],
            target_layers=[EvaluationLayer.QUANTITATIVE],
            evaluator_names=["quality_evaluator"],  # LLM_JUDGE layer
        )

        # Get evaluators for request
        evaluators_by_layer = evaluator._get_evaluators_for_request(
            request, [EvaluationLayer.QUANTITATIVE]
        )

        # Should not include quality_evaluator (wrong layer)
        quant_evaluators = evaluators_by_layer.get(EvaluationLayer.QUANTITATIVE, [])
        evaluator_names = [e.name for e in quant_evaluators]

        assert "quality_evaluator" not in evaluator_names
        assert len(evaluator_names) == 0  # No evaluators match


class TestEvaluateSingleOutputMethod:
    """Test evaluate_single_output method passes evaluator_names correctly."""

    @pytest.mark.asyncio
    async def test_evaluate_single_output_with_evaluator_names(
        self, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test that evaluate_single_output passes evaluator_names to request."""
        from mad_spark_alt.core.evaluator import CreativityEvaluator

        with patch.object(
            CreativityEvaluator, "evaluate", new=AsyncMock(return_value=mock_evaluation_result)
        ) as mock_evaluate:
            evaluator = CreativityEvaluator()

            output = ModelOutput(content="test", output_type=OutputType.TEXT, model_name="test")
            await evaluator.evaluate_single_output(
                output, evaluator_names=["novelty_evaluator"]
            )

            # Verify evaluate was called with request containing evaluator_names
            call_args = mock_evaluate.call_args
            request = call_args[0][0]
            assert request.evaluator_names == ["novelty_evaluator"]

    @pytest.mark.asyncio
    async def test_evaluate_single_output_without_evaluator_names(
        self, mock_evaluator_registry, mock_evaluation_result
    ):
        """Test that evaluate_single_output works without evaluator_names."""
        from mad_spark_alt.core.evaluator import CreativityEvaluator

        with patch.object(
            CreativityEvaluator, "evaluate", new=AsyncMock(return_value=mock_evaluation_result)
        ) as mock_evaluate:
            evaluator = CreativityEvaluator()

            output = ModelOutput(content="test", output_type=OutputType.TEXT, model_name="test")
            await evaluator.evaluate_single_output(output)

            # Verify evaluate was called with request without evaluator_names
            call_args = mock_evaluate.call_args
            request = call_args[0][0]
            assert request.evaluator_names is None
