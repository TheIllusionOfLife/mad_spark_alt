"""
Baseline tests for MultiPerspectiveQADIOrchestrator.

These tests verify the current implementation behavior before refactoring.
All tests use mocks to avoid API costs.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from mad_spark_alt.core.multi_perspective_orchestrator import (
    MultiPerspectiveQADIOrchestrator,
    MultiPerspectiveQADIResult,
    PerspectiveResult,
)
from mad_spark_alt.core.intent_detector import QuestionIntent, IntentResult
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
from mad_spark_alt.core.llm_provider import LLMResponse


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    def _create_response(content: str, cost: float = 0.01):
        return LLMResponse(
            content=content,
            cost=cost,
            provider="mock",
            model="mock-model",
            usage={"input_tokens": 100, "output_tokens": 100},
        )
    return _create_response


@pytest.fixture
def sample_questioning_response():
    """Sample response for questioning phase."""
    return """Q: How can we reduce ocean plastic pollution effectively?

Think about:
- Scale of the problem
- Technological solutions
- Policy interventions"""


@pytest.fixture
def sample_abduction_response():
    """Sample response for abduction phase."""
    return """H1: Develop biodegradable plastic alternatives that break down safely in ocean environments within months.

H2: Implement AI-powered ocean cleanup systems using autonomous vessels and smart nets.

H3: Create global plastic tax system with revenue funding cleanup and research."""


@pytest.fixture
def sample_deduction_response():
    """Sample response for deduction phase."""
    return """H1 Evaluation:
Impact: 0.9 - Revolutionary change to plastic lifecycle
Feasibility: 0.6 - Requires significant R&D
Accessibility: 0.7 - Manufacturing infrastructure exists
Sustainability: 0.9 - Addresses root cause
Scalability: 0.8 - Can scale globally once developed

H2 Evaluation:
Impact: 0.7 - Cleans existing pollution
Feasibility: 0.8 - Technology largely exists
Accessibility: 0.5 - Expensive to deploy
Sustainability: 0.6 - Ongoing operational costs
Scalability: 0.7 - Can expand to multiple oceans

H3 Evaluation:
Impact: 0.8 - Creates funding mechanism
Feasibility: 0.4 - Requires international cooperation
Accessibility: 0.9 - Easy to implement policy
Sustainability: 0.8 - Self-sustaining funding
Scalability: 0.9 - Can scale globally

ANSWER: A multi-pronged approach combining biodegradable alternatives (H1) with cleanup systems (H2) and sustainable funding (H3) offers the most comprehensive solution.

Action Plan:
1. Accelerate R&D into biodegradable ocean-safe plastics
2. Deploy pilot AI cleanup systems in high-pollution zones
3. Advocate for international plastic taxation framework"""


@pytest.fixture
def sample_induction_response():
    """Sample response for induction phase."""
    return """1. Costa Rica successfully reduced single-use plastics by 90% through similar policy measures

2. Ocean Cleanup project demonstrated feasibility of autonomous cleanup systems

3. Biodegradable plastic research at Stanford showed promising 60-day breakdown rates

Conclusion: The integrated approach is validated by existing successes in each component area."""


class TestMultiPerspectiveBaseline:
    """Test current MultiPerspective implementation."""

    @pytest.mark.asyncio
    async def test_single_perspective_analysis(
        self,
        mock_llm_response,
        sample_questioning_response,
        sample_abduction_response,
        sample_deduction_response,
        sample_induction_response,
    ):
        """Test running analysis with a single forced perspective."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Mock LLM responses
        with patch.object(
            orchestrator, "_run_llm_phase", new=AsyncMock()
        ) as mock_llm:
            mock_llm.side_effect = [
                # Environmental perspective - 4 phases
                (sample_questioning_response, 0.01),
                (sample_abduction_response, 0.02),
                (sample_deduction_response, 0.03),
                (sample_induction_response, 0.01),
                # Synthesis
                ("SYNTHESIS: Use all three approaches\n\nINTEGRATED ACTION PLAN:\n1. Start R&D\n2. Deploy pilots\n3. Advocate policy", 0.01),
            ]

            result = await orchestrator.run_multi_perspective_analysis(
                "How can we reduce ocean plastic?",
                force_perspectives=[QuestionIntent.ENVIRONMENTAL],
            )

        # Verify result structure
        assert isinstance(result, MultiPerspectiveQADIResult)
        assert result.primary_intent == QuestionIntent.ENVIRONMENTAL
        assert len(result.perspective_results) == 1
        assert result.perspective_results[0].perspective == QuestionIntent.ENVIRONMENTAL
        assert result.synthesized_answer
        assert len(result.synthesized_action_plan) > 0
        assert result.total_llm_cost > 0

    @pytest.mark.asyncio
    async def test_multiple_perspectives_parallel(
        self,
        mock_llm_response,
        sample_questioning_response,
        sample_abduction_response,
        sample_deduction_response,
        sample_induction_response,
    ):
        """Test running analysis with multiple perspectives in parallel."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Mock LLM responses - 3 perspectives × 4 phases + 1 synthesis
        responses = []
        for _ in range(3):  # 3 perspectives
            responses.extend([
                (sample_questioning_response, 0.01),
                (sample_abduction_response, 0.02),
                (sample_deduction_response, 0.03),
                (sample_induction_response, 0.01),
            ])
        # Add synthesis
        responses.append(
            ("SYNTHESIS: Comprehensive solution\n\nINTEGRATED ACTION PLAN:\n1. Action 1\n2. Action 2\n3. Action 3", 0.01)
        )

        with patch.object(
            orchestrator, "_run_llm_phase", new=AsyncMock()
        ) as mock_llm:
            mock_llm.side_effect = responses

            result = await orchestrator.run_multi_perspective_analysis(
                "How should we approach climate change?",
                force_perspectives=[
                    QuestionIntent.ENVIRONMENTAL,
                    QuestionIntent.TECHNICAL,
                    QuestionIntent.BUSINESS,
                ],
            )

        # Verify multiple perspectives
        assert len(result.perspective_results) == 3
        assert result.perspective_results[0].perspective == QuestionIntent.ENVIRONMENTAL
        assert result.perspective_results[1].perspective == QuestionIntent.TECHNICAL
        assert result.perspective_results[2].perspective == QuestionIntent.BUSINESS

        # Verify relevance scoring
        assert result.perspective_results[0].relevance_score == 1.0  # Primary
        assert result.perspective_results[1].relevance_score == pytest.approx(0.7, abs=0.01)  # 0.8 - 0.1
        assert result.perspective_results[2].relevance_score == pytest.approx(0.6, abs=0.01)  # 0.8 - 0.2

    @pytest.mark.asyncio
    async def test_intent_detection_and_auto_perspectives(self):
        """Test automatic intent detection and perspective selection."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        with patch.object(
            orchestrator.intent_detector, "detect_intent"
        ) as mock_detect, patch.object(
            orchestrator.intent_detector, "get_recommended_perspectives"
        ) as mock_perspectives, patch.object(
            orchestrator, "_run_perspective_analysis", new=AsyncMock()
        ) as mock_run, patch.object(
            orchestrator, "_synthesize_results", new=AsyncMock()
        ) as mock_synth:

            # Mock intent detection
            mock_detect.return_value = IntentResult(
                primary_intent=QuestionIntent.TECHNICAL,
                confidence=0.85,
                secondary_intents=[QuestionIntent.BUSINESS],
                keywords_matched=["algorithm", "optimize"],
            )

            # Mock perspective recommendations
            mock_perspectives.return_value = [
                QuestionIntent.TECHNICAL,
                QuestionIntent.BUSINESS,
            ]

            # Mock perspective analysis
            mock_result = SimpleQADIResult(
                core_question="Test question",
                hypotheses=["H1", "H2", "H3"],
                hypothesis_scores=[
                    HypothesisScore(0.8, 0.7, 0.6, 0.7, 0.8, 0.72),
                    HypothesisScore(0.7, 0.8, 0.7, 0.8, 0.7, 0.74),
                    HypothesisScore(0.6, 0.6, 0.8, 0.6, 0.6, 0.64),
                ],
                final_answer="Test answer",
                action_plan=["Action 1", "Action 2"],
                verification_examples=["Example 1"],
                verification_conclusion="Conclusion",
                total_llm_cost=0.05,
            )
            mock_run.return_value = mock_result

            # Mock synthesis
            mock_synth.return_value = {
                "answer": "Synthesized answer",
                "action_plan": ["Integrated 1", "Integrated 2"],
                "best_hypothesis": ("H2", QuestionIntent.TECHNICAL),
                "synthesis_cost": 0.01,
            }

            result = await orchestrator.run_multi_perspective_analysis(
                "How can we optimize the sorting algorithm?",
                max_perspectives=2,
            )

        # Verify intent detection was called
        mock_detect.assert_called_once()
        mock_perspectives.assert_called_once()

        # Verify correct perspectives used
        assert result.primary_intent == QuestionIntent.TECHNICAL
        assert result.intent_confidence == 0.85
        assert len(result.perspectives_used) == 2

    @pytest.mark.asyncio
    async def test_perspective_failure_handling(
        self,
        sample_questioning_response,
        sample_abduction_response,
        sample_deduction_response,
        sample_induction_response,
    ):
        """Test handling of perspective analysis failures."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Mock one successful and one failed perspective
        async def mock_analysis(user_input, perspective):
            if perspective == QuestionIntent.TECHNICAL:
                # Success
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
                    total_llm_cost=0.05,
                )
            else:
                # Failure
                return None

        with patch.object(
            orchestrator, "_run_perspective_analysis", new=mock_analysis
        ), patch.object(
            orchestrator, "_synthesize_results", new=AsyncMock()
        ) as mock_synth:

            mock_synth.return_value = {
                "answer": "Answer",
                "action_plan": ["Action"],
                "best_hypothesis": ("H1", QuestionIntent.TECHNICAL),
                "synthesis_cost": 0.01,
            }

            result = await orchestrator.run_multi_perspective_analysis(
                "Test question",
                force_perspectives=[QuestionIntent.TECHNICAL, QuestionIntent.BUSINESS],
            )

        # Should only include successful perspective
        assert len(result.perspective_results) == 1
        assert result.perspective_results[0].perspective == QuestionIntent.TECHNICAL

    @pytest.mark.asyncio
    async def test_hypothesis_extraction(self):
        """Test hypothesis extraction from various formats."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Test H1: H2: H3: format
        text1 = """H1: First hypothesis about X
H2: Second hypothesis about Y
H3: Third hypothesis about Z"""

        hypotheses = orchestrator._extract_hypotheses(text1)
        assert len(hypotheses) == 3
        assert "First hypothesis about X" in hypotheses[0]
        assert "Second hypothesis about Y" in hypotheses[1]
        assert "Third hypothesis about Z" in hypotheses[2]

        # Test numbered format fallback
        text2 = """1. First idea
2. Second idea
3. Third idea"""

        hypotheses = orchestrator._extract_hypotheses(text2)
        assert len(hypotheses) == 3
        assert "First idea" in hypotheses[0]

    @pytest.mark.asyncio
    async def test_score_extraction(self):
        """Test score extraction from deduction response."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Use format that ScoreParser expects (H1:, H2: headers)
        deduction_text = """H1: Evaluation
Impact: 0.9 - High impact
Feasibility: 0.7 - Moderately feasible
Accessibility: 0.8 - Accessible
Sustainability: 0.6 - Sustainable
Scalability: 0.7 - Scalable

H2: Evaluation
Impact: 0.8
Feasibility: 0.6
Accessibility: 0.7
Sustainability: 0.8
Scalability: 0.9"""

        result = orchestrator._parse_deduction_results(deduction_text, 2)

        assert len(result["scores"]) == 2

        # Check H1 scores
        assert result["scores"][0].impact == 0.9
        assert result["scores"][0].feasibility == 0.7
        assert result["scores"][0].accessibility == 0.8

        # Check H2 scores
        assert result["scores"][1].impact == 0.8
        assert result["scores"][1].feasibility == 0.6

    @pytest.mark.asyncio
    async def test_synthesis_with_no_perspectives(self):
        """Test synthesis handles empty perspective list."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        result = await orchestrator._synthesize_results("Test question", [])

        assert result["answer"] == "Unable to generate analysis."
        assert result["action_plan"] == []
        assert result["synthesis_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_best_hypothesis_selection(
        self,
        sample_questioning_response,
        sample_abduction_response,
        sample_deduction_response,
        sample_induction_response,
    ):
        """Test that best hypothesis is selected across perspectives."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        # Create perspective results with different scores
        pr1 = PerspectiveResult(
            perspective=QuestionIntent.TECHNICAL,
            result=SimpleQADIResult(
                core_question="Q",
                hypotheses=["Tech H1", "Tech H2", "Tech H3"],
                hypothesis_scores=[
                    HypothesisScore(0.6, 0.6, 0.6, 0.6, 0.6, 0.6),  # 0.6 overall
                    HypothesisScore(0.7, 0.7, 0.7, 0.7, 0.7, 0.7),  # 0.7 overall
                    HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),  # 0.5 overall
                ],
                final_answer="A",
                action_plan=[],
                verification_examples=[],
                verification_conclusion="",
                total_llm_cost=0.05,
            ),
            relevance_score=1.0,  # Primary perspective
        )

        pr2 = PerspectiveResult(
            perspective=QuestionIntent.BUSINESS,
            result=SimpleQADIResult(
                core_question="Q",
                hypotheses=["Biz H1", "Biz H2", "Biz H3"],
                hypothesis_scores=[
                    HypothesisScore(0.9, 0.9, 0.9, 0.9, 0.9, 0.9),  # 0.9 overall
                    HypothesisScore(0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
                    HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                ],
                final_answer="A",
                action_plan=[],
                verification_examples=[],
                verification_conclusion="",
                total_llm_cost=0.05,
            ),
            relevance_score=0.7,  # Secondary perspective
        )

        with patch.object(
            orchestrator, "_run_llm_phase", new=AsyncMock()
        ) as mock_llm:
            mock_llm.return_value = (
                "SYNTHESIS: Test\n\nINTEGRATED ACTION PLAN:\n1. A\n2. B",
                0.01,
            )

            result = await orchestrator._synthesize_results("Test", [pr1, pr2])

        # Best should be "Tech H2" with weighted score 0.7 * 1.0 = 0.7
        # (not "Biz H1" with weighted score 0.9 * 0.7 = 0.63)
        assert result["best_hypothesis"][0] == "Tech H2"
        assert result["best_hypothesis"][1] == QuestionIntent.TECHNICAL

    @pytest.mark.asyncio
    async def test_idea_collection(self):
        """Test collecting all ideas from all perspectives."""
        orchestrator = MultiPerspectiveQADIOrchestrator()

        pr1 = PerspectiveResult(
            perspective=QuestionIntent.TECHNICAL,
            result=SimpleQADIResult(
                core_question="Q",
                hypotheses=["H1", "H2"],
                hypothesis_scores=[
                    HypothesisScore(0.8, 0.7, 0.6, 0.7, 0.8, 0.72),
                    HypothesisScore(0.7, 0.8, 0.7, 0.8, 0.7, 0.74),
                ],
                final_answer="A",
                action_plan=[],
                verification_examples=[],
                verification_conclusion="",
                total_llm_cost=0.05,
            ),
            relevance_score=1.0,
        )

        pr2 = PerspectiveResult(
            perspective=QuestionIntent.BUSINESS,
            result=SimpleQADIResult(
                core_question="Q",
                hypotheses=["H3"],
                hypothesis_scores=[
                    HypothesisScore(0.6, 0.6, 0.8, 0.6, 0.6, 0.64),
                ],
                final_answer="A",
                action_plan=[],
                verification_examples=[],
                verification_conclusion="",
                total_llm_cost=0.05,
            ),
            relevance_score=0.7,
        )

        ideas = orchestrator._collect_all_ideas([pr1, pr2])

        # Should have 3 ideas total
        assert len(ideas) == 3
        assert ideas[0].content == "H1"
        assert ideas[1].content == "H2"
        assert ideas[2].content == "H3"

        # Check metadata (QuestionIntent.value is lowercase)
        assert ideas[0].metadata["perspective"] == "technical"
        assert ideas[2].metadata["perspective"] == "business"
        assert ideas[0].confidence_score == 0.72
        assert ideas[0].metadata["relevance_score"] == 1.0
        assert ideas[2].metadata["relevance_score"] == 0.7
