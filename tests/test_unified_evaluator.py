"""Tests for UnifiedEvaluator - the consistent 5-criteria evaluation system."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import UnifiedEvaluator, HypothesisEvaluation
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM manager."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def evaluator():
    """Create a UnifiedEvaluator instance."""
    return UnifiedEvaluator()


class TestUnifiedEvaluator:
    """Test cases for UnifiedEvaluator."""

    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.criteria is not None
        assert len(evaluator.criteria) == 5
        assert "novelty" in evaluator.criteria
        assert "impact" in evaluator.criteria
        assert "cost" in evaluator.criteria
        assert "feasibility" in evaluator.criteria
        assert "risks" in evaluator.criteria

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_success(self, evaluator, mock_llm_manager):
        """Test successful hypothesis evaluation."""
        # Mock LLM response
        mock_response = LLMResponse(
            content="""Novelty: 0.8 - This is a highly innovative approach
Impact: 0.9 - Significant positive change expected
Cost: 0.3 - Relatively expensive to implement
Feasibility: 0.7 - Technically achievable with effort
Risks: 0.6 - Moderate risks but manageable""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.005,
            response_time=1.0,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            result = await evaluator.evaluate_hypothesis(
                hypothesis="Implement quantum computing for optimization",
                context="Improve computational efficiency",
                core_question="How to speed up calculations?",
            )

        assert isinstance(result, HypothesisEvaluation)
        assert result.content == "Implement quantum computing for optimization"
        assert result.scores["novelty"] == 0.8
        assert result.scores["impact"] == 0.9
        assert result.scores["cost"] == 0.3
        assert result.scores["feasibility"] == 0.7
        assert result.scores["risks"] == 0.6
        assert 0.5 < result.overall_score < 0.8  # Weighted score
        assert result.explanations["novelty"] == "This is a highly innovative approach"
        assert result.metadata["llm_cost"] == 0.005
        assert result.metadata["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_with_edge_scores(self, evaluator, mock_llm_manager):
        """Test evaluation with edge case scores (0.0 and 1.0)."""
        mock_response = LLMResponse(
            content="""Novelty: 0.0 - Completely standard approach
Impact: 1.0 - Revolutionary impact
Cost: 0.0 - Extremely expensive
Feasibility: 1.0 - Trivially easy
Risks: 0.0 - High risk""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.005,
            response_time=1.0,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            result = await evaluator.evaluate_hypothesis(
                hypothesis="Test hypothesis",
                context="Test context",
            )

        assert result.scores["novelty"] == 0.0
        assert result.scores["impact"] == 1.0
        assert result.scores["cost"] == 0.0
        assert result.scores["feasibility"] == 1.0
        assert result.scores["risks"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_with_out_of_range_scores(self, evaluator, mock_llm_manager):
        """Test that out-of-range scores are clamped to [0, 1]."""
        mock_response = LLMResponse(
            content="""Novelty: 1.5 - Beyond innovative
Impact: -0.3 - Negative impact
Cost: 0.5 - Moderate cost
Feasibility: 0.8 - Feasible
Risks: 2.0 - Extreme risk""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.005,
            response_time=1.0,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            result = await evaluator.evaluate_hypothesis(
                hypothesis="Test hypothesis",
                context="Test context",
            )

        # Scores should be clamped to valid range
        assert result.scores["novelty"] == 1.0  # Clamped from 1.5
        assert result.scores["impact"] == 0.0  # Clamped from -0.3 
        assert result.scores["cost"] == 0.5
        assert result.scores["feasibility"] == 0.8
        assert result.scores["risks"] == 1.0  # Clamped from 2.0

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_parse_failure(self, evaluator, mock_llm_manager):
        """Test evaluation when LLM response parsing fails."""
        mock_response = LLMResponse(
            content="This is an invalid response without proper formatting",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 20},
            cost=0.002,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            result = await evaluator.evaluate_hypothesis(
                hypothesis="Test hypothesis",
                context="Test context",
            )

        # Should return default scores on parse failure
        assert all(score == 0.5 for score in result.scores.values())
        assert pytest.approx(result.overall_score, rel=1e-5) == 0.5
        assert all(explanation == "Not evaluated" for explanation in result.explanations.values())

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_with_exception(self, evaluator, mock_llm_manager):
        """Test evaluation when LLM call raises exception."""
        mock_llm_manager.generate.side_effect = Exception("Network error")

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            result = await evaluator.evaluate_hypothesis(
                hypothesis="Test hypothesis",
                context="Test context",
            )

        # Should return default evaluation on exception
        assert all(score == 0.5 for score in result.scores.values())
        assert result.overall_score == 0.5
        assert all("evaluation failed" in explanation.lower() or "using default" in explanation.lower() for explanation in result.explanations.values())
        assert result.metadata["error"] == "Network error"

    @pytest.mark.asyncio
    async def test_evaluate_multiple_parallel(self, evaluator, mock_llm_manager):
        """Test parallel evaluation of multiple hypotheses."""
        # Mock different responses for each hypothesis
        responses = [
            LLMResponse(
                content=f"""Novelty: 0.{i+5} - Description
Impact: 0.{i+6} - Description
Cost: 0.{i+4} - Description
Feasibility: 0.{i+7} - Description
Risks: 0.{i+3} - Description""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 50, "completion_tokens": 100},
                cost=0.005,
                response_time=1.0,
            )
            for i in range(3)
        ]
        mock_llm_manager.generate.side_effect = responses

        hypotheses = ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"]

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            results = await evaluator.evaluate_multiple(
                hypotheses=hypotheses,
                context="Test context",
                core_question="Test question?",
                parallel=True,
            )

        assert len(results) == 3
        assert all(isinstance(r, HypothesisEvaluation) for r in results)
        assert results[0].scores["novelty"] == 0.5
        assert results[1].scores["novelty"] == 0.6
        assert results[2].scores["novelty"] == 0.7

    @pytest.mark.asyncio
    async def test_evaluate_multiple_sequential(self, evaluator, mock_llm_manager):
        """Test sequential evaluation of multiple hypotheses."""
        responses = [
            LLMResponse(
                content=f"""Novelty: 0.{i+5} - Description
Impact: 0.{i+6} - Description
Cost: 0.{i+4} - Description
Feasibility: 0.{i+7} - Description
Risks: 0.{i+3} - Description""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 50, "completion_tokens": 100},
                cost=0.005,
                response_time=1.0,
            )
            for i in range(2)
        ]
        mock_llm_manager.generate.side_effect = responses

        hypotheses = ["Hypothesis 1", "Hypothesis 2"]

        with patch("mad_spark_alt.core.unified_evaluator.llm_manager", mock_llm_manager):
            results = await evaluator.evaluate_multiple(
                hypotheses=hypotheses,
                context="Test context",
                parallel=False,
            )

        assert len(results) == 2
        assert all(isinstance(r, HypothesisEvaluation) for r in results)

    def test_calculate_fitness_from_evaluation(self, evaluator):
        """Test fitness calculation from evaluation."""
        evaluation = HypothesisEvaluation(
            content="Test hypothesis",
            scores={"novelty": 0.8, "impact": 0.7, "cost": 0.6, "feasibility": 0.9, "risks": 0.5},
            overall_score=0.73,
            explanations={},
            metadata={},
        )

        fitness = evaluator.calculate_fitness_from_evaluation(evaluation)
        assert fitness == 0.73

    def test_get_best_hypothesis(self, evaluator):
        """Test getting the best hypothesis from evaluations."""
        evaluations = [
            HypothesisEvaluation(
                content="Hypothesis 1",
                scores={},
                overall_score=0.65,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="Hypothesis 2",
                scores={},
                overall_score=0.82,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="Hypothesis 3",
                scores={},
                overall_score=0.71,
                explanations={},
                metadata={},
            ),
        ]

        best = evaluator.get_best_hypothesis(evaluations)
        assert best.content == "Hypothesis 2"
        assert best.overall_score == 0.82

    def test_rank_hypotheses(self, evaluator):
        """Test ranking hypotheses by score."""
        evaluations = [
            HypothesisEvaluation(
                content="Low score",
                scores={},
                overall_score=0.45,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="High score",
                scores={},
                overall_score=0.85,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="Medium score",
                scores={},
                overall_score=0.65,
                explanations={},
                metadata={},
            ),
        ]

        ranked = evaluator.rank_hypotheses(evaluations)
        assert ranked[0].overall_score == 0.85
        assert ranked[1].overall_score == 0.65
        assert ranked[2].overall_score == 0.45
        assert ranked[0].content == "High score"

    def test_get_score_summary(self, evaluator):
        """Test getting summary statistics for evaluations."""
        evaluations = [
            HypothesisEvaluation(
                content="H1",
                scores={"novelty": 0.8, "impact": 0.7, "cost": 0.5, "feasibility": 0.9, "risks": 0.6},
                overall_score=0.7,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="H2",
                scores={"novelty": 0.6, "impact": 0.8, "cost": 0.7, "feasibility": 0.8, "risks": 0.7},
                overall_score=0.72,
                explanations={},
                metadata={},
            ),
            HypothesisEvaluation(
                content="H3",
                scores={"novelty": 0.7, "impact": 0.6, "cost": 0.6, "feasibility": 0.7, "risks": 0.8},
                overall_score=0.68,
                explanations={},
                metadata={},
            ),
        ]

        summary = evaluator.get_score_summary(evaluations)

        # Check novelty stats
        assert summary["novelty"]["min"] == 0.6
        assert summary["novelty"]["max"] == 0.8
        assert summary["novelty"]["avg"] == pytest.approx(0.7, rel=1e-3)

        # Check overall stats
        assert summary["overall"]["min"] == 0.68
        assert summary["overall"]["max"] == 0.72
        assert summary["overall"]["avg"] == pytest.approx(0.7, rel=1e-3)

    def test_get_score_summary_empty(self, evaluator):
        """Test score summary with empty evaluations."""
        summary = evaluator.get_score_summary([])
        assert summary == {}

    def test_build_evaluation_prompt(self, evaluator):
        """Test evaluation prompt building."""
        prompt = evaluator._build_evaluation_prompt(
            hypothesis="Use AI for prediction",
            context="Improve forecasting",
            core_question="How to predict better?",
        )

        assert "Use AI for prediction" in prompt
        assert "Improve forecasting" in prompt
        assert "How to predict better?" in prompt
        assert "Novelty:" in prompt
        assert "Impact:" in prompt
        assert "Cost:" in prompt
        assert "Feasibility:" in prompt
        assert "Risks:" in prompt

    def test_parse_evaluation_response_various_formats(self, evaluator):
        """Test parsing various response formats."""
        # Test with different formatting styles
        responses = [
            # Standard format
            """Novelty: 0.8 - Very innovative
Impact: 0.7 - High impact
Cost: 0.6 - Moderate cost
Feasibility: 0.9 - Very feasible
Risks: 0.5 - Low risk""",
            # With extra whitespace
            """
            Novelty:    0.8    -    Very innovative
            Impact:     0.7    -    High impact  
            Cost:       0.6    -    Moderate cost
            Feasibility: 0.9   -    Very feasible
            Risks:      0.5    -    Low risk
            """,
            # Mixed case
            """novelty: 0.8 - Very innovative
IMPACT: 0.7 - High impact
Cost: 0.6 - Moderate cost
FeasiBility: 0.9 - Very feasible
RISKS: 0.5 - Low risk""",
        ]

        for response in responses:
            scores, explanations = evaluator._parse_evaluation_response(response)
            assert scores["novelty"] == 0.8
            assert scores["impact"] == 0.7
            assert scores["cost"] == 0.6
            assert scores["feasibility"] == 0.9
            assert scores["risks"] == 0.5
            assert "innovative" in explanations["novelty"]

    def test_parse_evaluation_response_missing_criteria(self, evaluator):
        """Test parsing when some criteria are missing."""
        response = """Novelty: 0.8 - Very innovative
Impact: 0.7 - High impact
Feasibility: 0.9 - Very feasible"""

        scores, explanations = evaluator._parse_evaluation_response(response)
        
        # Provided criteria
        assert scores["novelty"] == 0.8
        assert scores["impact"] == 0.7
        assert scores["feasibility"] == 0.9
        
        # Missing criteria should have defaults
        assert scores["cost"] == 0.5
        assert scores["risks"] == 0.5
        assert explanations["cost"] == "Not evaluated"
        assert explanations["risks"] == "Not evaluated"