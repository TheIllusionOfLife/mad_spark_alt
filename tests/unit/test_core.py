"""Test core functionality."""

import pytest
import asyncio
from src.mad_spark_alt.core import (
    CreativityEvaluator,
    EvaluationRequest,
    ModelOutput,
    OutputType,
    EvaluationLayer,
    registry,
)
from src.mad_spark_alt.layers.quantitative import DiversityEvaluator, QualityEvaluator


class TestEvaluatorRegistry:
    """Test the evaluator registry."""

    def test_register_evaluator(self):
        """Test registering an evaluator."""
        # Clear registry for clean test
        registry._evaluators.clear()
        registry._instances.clear()
        registry._layer_index = {layer: set() for layer in EvaluationLayer}
        registry._output_type_index = {output_type: set() for output_type in OutputType}

        # Register evaluator
        registry.register(DiversityEvaluator)

        # Check it's registered
        assert "diversity_evaluator" in registry._evaluators

        # Check indices are updated
        assert (
            "diversity_evaluator" in registry._layer_index[EvaluationLayer.QUANTITATIVE]
        )
        assert "diversity_evaluator" in registry._output_type_index[OutputType.TEXT]
        assert "diversity_evaluator" in registry._output_type_index[OutputType.CODE]

    def test_get_evaluator(self):
        """Test getting an evaluator instance."""
        registry.register(DiversityEvaluator)

        evaluator = registry.get_evaluator("diversity_evaluator")
        assert evaluator is not None
        assert evaluator.name == "diversity_evaluator"

        # Test singleton behavior
        evaluator2 = registry.get_evaluator("diversity_evaluator")
        assert evaluator is evaluator2

    def test_get_evaluators_by_layer(self):
        """Test getting evaluators by layer."""
        registry.register(DiversityEvaluator)
        registry.register(QualityEvaluator)

        evaluators = registry.get_evaluators_by_layer(EvaluationLayer.QUANTITATIVE)
        assert len(evaluators) == 2

        names = {e.name for e in evaluators}
        assert "diversity_evaluator" in names
        assert "quality_evaluator" in names


class TestModelOutput:
    """Test ModelOutput class."""

    def test_create_model_output(self):
        """Test creating a model output."""
        output = ModelOutput(
            content="Test content",
            output_type=OutputType.TEXT,
            model_name="test-model",
            prompt="Test prompt",
        )

        assert output.content == "Test content"
        assert output.output_type == OutputType.TEXT
        assert output.model_name == "test-model"
        assert output.prompt == "Test prompt"


class TestCreativityEvaluator:
    """Test the main creativity evaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_single_output(self):
        """Test evaluating a single output."""
        # Ensure evaluators are registered
        registry.register(DiversityEvaluator)
        registry.register(QualityEvaluator)

        output = ModelOutput(
            content="This is a test sentence for evaluation.",
            output_type=OutputType.TEXT,
            model_name="test-model",
        )

        evaluator = CreativityEvaluator()
        summary = await evaluator.evaluate_single_output(output)

        assert summary.total_outputs == 1
        assert summary.total_evaluators > 0
        assert summary.execution_time > 0
        assert len(summary.layer_results) > 0

        # Check that we have quantitative results
        assert EvaluationLayer.QUANTITATIVE in summary.layer_results
        quantitative_results = summary.layer_results[EvaluationLayer.QUANTITATIVE]
        assert len(quantitative_results) > 0

        # Check diversity evaluator results
        diversity_results = [
            r for r in quantitative_results if r.evaluator_name == "diversity_evaluator"
        ]
        assert len(diversity_results) > 0

        diversity_result = diversity_results[0]
        assert "distinct_1" in diversity_result.scores
        assert "lexical_diversity" in diversity_result.scores

    @pytest.mark.asyncio
    async def test_evaluate_multiple_outputs(self):
        """Test evaluating multiple outputs for diversity comparison."""
        registry.register(DiversityEvaluator)

        outputs = [
            ModelOutput(
                content="AI will transform healthcare.",
                output_type=OutputType.TEXT,
                model_name="model-a",
            ),
            ModelOutput(
                content="The purple elephant danced.",
                output_type=OutputType.TEXT,
                model_name="model-b",
            ),
        ]

        request = EvaluationRequest(outputs=outputs)
        evaluator = CreativityEvaluator()
        summary = await evaluator.evaluate(request)

        assert summary.total_outputs == 2

        # Check that diversity metrics include semantic uniqueness
        quantitative_results = summary.layer_results[EvaluationLayer.QUANTITATIVE]
        diversity_results = [
            r for r in quantitative_results if r.evaluator_name == "diversity_evaluator"
        ]

        # Should have results for both outputs
        assert len(diversity_results) == 2

        # Both should have semantic uniqueness scores
        for result in diversity_results:
            assert "semantic_uniqueness" in result.scores
            assert "novelty_score" in result.scores
