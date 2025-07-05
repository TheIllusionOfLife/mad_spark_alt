"""
Tests for the LLM judge evaluation layer.
"""

import asyncio
import pytest

from mad_spark_alt.core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    ModelOutput,
    OutputType,
)
from mad_spark_alt.layers.llm_judges import CreativityLLMJudge, CreativityJury


class TestLLMJudges:
    """Test LLM judge evaluation functionality."""

    @pytest.mark.asyncio
    async def test_llm_judge_basic_evaluation(self):
        """Test basic LLM judge evaluation."""
        judge = CreativityLLMJudge("mock-model")
        
        output = ModelOutput(
            content="A revolutionary idea that changes everything",
            output_type=OutputType.TEXT,
            model_name="test-model",
        )
        
        request = EvaluationRequest(
            outputs=[output],
            task_context="Test evaluation"
        )
        
        results = await judge.evaluate(request)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.evaluator_name == "creativity_llm_judge_mock_model"
        assert result.layer == EvaluationLayer.LLM_JUDGE
        assert "novelty" in result.scores
        assert "usefulness" in result.scores
        assert "overall_creativity" in result.scores
        
        # Scores should be between 0 and 1
        for score in result.scores.values():
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_llm_jury_consensus(self):
        """Test LLM jury consensus mechanism."""
        jury = CreativityJury(["mock-model-1", "mock-model-2"])
        
        output = ModelOutput(
            content="An innovative solution to climate change",
            output_type=OutputType.TEXT,
            model_name="test-model",
        )
        
        request = EvaluationRequest(
            outputs=[output],
            task_context="Jury test evaluation"
        )
        
        results = await jury.evaluate(request)
        
        assert len(results) == 1
        result = results[0]
        
        assert result.layer == EvaluationLayer.LLM_JUDGE
        assert "novelty" in result.scores
        assert result.metadata["jury_size"] == 2
        assert "consensus_method" in result.metadata

    def test_llm_judge_properties(self):
        """Test LLM judge properties."""
        judge = CreativityLLMJudge("gpt-4")
        
        assert judge.name == "creativity_llm_judge_gpt_4"
        assert judge.layer == EvaluationLayer.LLM_JUDGE
        assert OutputType.TEXT in judge.supported_output_types
        assert OutputType.CODE in judge.supported_output_types

    def test_llm_jury_properties(self):
        """Test LLM jury properties."""
        jury = CreativityJury(["gpt-4", "claude-3"])
        
        assert "creativity_jury" in jury.name
        assert jury.layer == EvaluationLayer.LLM_JUDGE
        assert OutputType.TEXT in jury.supported_output_types

    def test_config_validation(self):
        """Test configuration validation."""
        judge = CreativityLLMJudge("mock-model")
        
        valid_config = {"temperature": 0.7, "max_tokens": 1500}
        assert judge.validate_config(valid_config)
        
        invalid_config = {"invalid_key": "value"}
        assert not judge.validate_config(invalid_config)