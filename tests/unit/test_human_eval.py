"""
Tests for the human evaluation layer.
"""

import pytest

from mad_spark_alt.core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    ModelOutput,
    OutputType,
)
from mad_spark_alt.layers.human_eval import HumanCreativityEvaluator, ABTestEvaluator


class TestHumanEvaluation:
    """Test human evaluation functionality."""

    @pytest.mark.asyncio
    async def test_human_evaluator_batch_mode(self):
        """Test human evaluator in batch mode."""
        evaluator = HumanCreativityEvaluator({
            "mode": "batch",
            "output_file": "/tmp/test_human_eval.json"
        })
        
        output = ModelOutput(
            content="A creative solution to urban planning",
            output_type=OutputType.TEXT,
            model_name="test-model",
        )
        
        request = EvaluationRequest(
            outputs=[output],
            task_context="Test human evaluation"
        )
        
        results = await evaluator.evaluate(request)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.evaluator_name == "human_creativity_evaluator_batch"
        assert result.layer == EvaluationLayer.HUMAN
        assert "info" in result.explanations

    @pytest.mark.asyncio 
    async def test_ab_test_evaluator_insufficient_outputs(self):
        """Test A/B test evaluator with insufficient outputs."""
        evaluator = ABTestEvaluator({"mode": "pairwise"})
        
        output = ModelOutput(
            content="Single output",
            output_type=OutputType.TEXT,
            model_name="test-model",
        )
        
        request = EvaluationRequest(
            outputs=[output],  # Only one output - insufficient for A/B testing
            task_context="Test A/B evaluation"
        )
        
        results = await evaluator.evaluate(request)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.layer == EvaluationLayer.HUMAN
        assert "error" in result.explanations
        assert "Need at least 2 outputs" in result.explanations["error"]

    def test_human_evaluator_properties(self):
        """Test human evaluator properties."""
        evaluator = HumanCreativityEvaluator({"mode": "interactive"})
        
        assert evaluator.name == "human_creativity_evaluator_interactive"
        assert evaluator.layer == EvaluationLayer.HUMAN
        assert OutputType.TEXT in evaluator.supported_output_types

    def test_ab_test_evaluator_properties(self):
        """Test A/B test evaluator properties."""
        evaluator = ABTestEvaluator({"mode": "pairwise"})
        
        assert evaluator.name == "ab_test_evaluator_pairwise"
        assert evaluator.layer == EvaluationLayer.HUMAN
        assert OutputType.TEXT in evaluator.supported_output_types

    def test_config_validation(self):
        """Test configuration validation."""
        evaluator = HumanCreativityEvaluator()
        
        valid_config = {"mode": "batch", "output_file": "test.json"}
        assert evaluator.validate_config(valid_config)
        
        invalid_config = {"mode": "invalid_mode"}
        assert not evaluator.validate_config(invalid_config)

        ab_evaluator = ABTestEvaluator()
        
        valid_ab_config = {"mode": "pairwise", "randomize_order": True}
        assert ab_evaluator.validate_config(valid_ab_config)
        
        invalid_ab_config = {"mode": "invalid_mode"}
        assert not ab_evaluator.validate_config(invalid_ab_config)