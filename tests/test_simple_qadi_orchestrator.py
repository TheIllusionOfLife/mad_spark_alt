"""Tests for SimpleQADIOrchestrator - the true QADI hypothesis-driven implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import SimpleQADIOrchestrator, SimpleQADIResult
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


@pytest.fixture
def mock_llm_manager():
    """Create a mock LLM manager."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def orchestrator():
    """Create a SimpleQADIOrchestrator instance."""
    return SimpleQADIOrchestrator()


@pytest.fixture
def orchestrator_with_temp():
    """Create a SimpleQADIOrchestrator with custom temperature."""
    return SimpleQADIOrchestrator(temperature_override=1.2)


class TestSimpleQADIOrchestrator:
    """Test cases for SimpleQADIOrchestrator.

    Note: Individual phase logic tests have been moved to test_phase_logic.py
    and test_phase_integration.py. This file now focuses on orchestrator-specific
    functionality like initialization and the complete QADI cycle.
    """

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.temperature_override is None
        assert orchestrator.prompts is not None

    def test_initialization_with_temperature(self, orchestrator_with_temp):
        """Test orchestrator initialization with temperature override."""
        assert orchestrator_with_temp.temperature_override == 1.2

    def test_invalid_temperature(self):
        """Test initialization with invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            SimpleQADIOrchestrator(temperature_override=2.5)

    @pytest.mark.asyncio
    async def test_complete_qadi_cycle(self, orchestrator, mock_llm_manager):
        """Test a complete QADI cycle from start to finish."""
        # Set up mock responses for each phase
        mock_responses = [
            # Questioning phase
            LLMResponse(
                content="Q: How can we reduce carbon emissions in urban transportation?",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                cost=0.001,
                response_time=0.5,
            ),
            # Abduction phase
            LLMResponse(
                content="""H1: Expand electric vehicle charging infrastructure throughout the city
H2: Implement congestion pricing to discourage private car usage
H3: Develop integrated multimodal transport apps for seamless public transit""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 30, "completion_tokens": 60},
                cost=0.003,
                response_time=0.8,
            ),
            # Deduction phase
            LLMResponse(
                content="""Analysis:
- H1:
  * Impact: 0.8 - significant reduction in emissions from widespread EV adoption
  * Feasibility: 0.6 - moderate implementation challenges
  * Accessibility: 0.4 - requires significant infrastructure investment
  * Sustainability: 0.7 - long-term environmental benefits
  * Scalability: 0.5 - infrastructure scaling is challenging
  * Overall: 0.58

- H2:
  * Impact: 0.9 - proven high impact on traffic and emissions reduction
  * Feasibility: 0.5 - challenging due to political and public resistance
  * Accessibility: 0.7 - affects all road users equally
  * Sustainability: 0.6 - requires ongoing public support
  * Scalability: 0.8 - can be implemented in various city contexts
  * Overall: 0.66

- H3:
  * Impact: 0.7 - good impact on efficiency and mode switching
  * Feasibility: 0.9 - highly practical with existing technology
  * Accessibility: 0.9 - easy access through smartphones
  * Sustainability: 0.8 - minimal ongoing resource requirements
  * Scalability: 0.9 - software solutions scale easily
  * Overall: 0.78

ANSWER: Congestion pricing emerges as the most impactful solution, reducing emissions by 20-30% in cities like London and Singapore.

Action Plan:
1. Conduct traffic flow analysis to identify optimal pricing zones
2. Implement gradual pricing with public consultation
3. Invest revenue in public transport improvements""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 50, "completion_tokens": 100},
                cost=0.005,
                response_time=1.0,
            ),
            # Induction phase (now uses structured output with synthesis)
            LLMResponse(
                content=json.dumps({
                    "synthesis": (
                        "Real-world implementations consistently demonstrate congestion pricing's "
                        "effectiveness in reducing urban transport emissions. London's congestion charge "
                        "reduced traffic by 30% and emissions by 20%. Singapore's Electronic Road Pricing "
                        "cut peak hour traffic by 24%. Stockholm saw 22% emission reduction and generated "
                        "€100M annually for transport improvements."
                    )
                }),
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 40, "completion_tokens": 80},
                cost=0.004,
                response_time=0.9,
            ),
        ]

        mock_llm_manager.generate.side_effect = mock_responses

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            result = await orchestrator.run_qadi_cycle(
                "Help me find solutions for urban pollution",
                context="Focus on transportation sector",
            )

        # Verify result structure
        assert isinstance(result, SimpleQADIResult)
        assert result.core_question == "How can we reduce carbon emissions in urban transportation?"
        assert len(result.hypotheses) == 3
        assert len(result.hypothesis_scores) == 3
        assert "congestion pricing" in result.final_answer.lower()
        assert len(result.action_plan) == 3
        # New induction returns synthesis, not examples
        assert result.verification_examples == []  # Empty by design
        assert "Real-world implementations" in result.verification_conclusion
        assert pytest.approx(result.total_llm_cost, rel=1e-5) == 0.013  # Sum of all costs

        # Verify synthesized ideas for evolution compatibility
        assert len(result.synthesized_ideas) == 3
        assert all(isinstance(idea, GeneratedIdea) for idea in result.synthesized_ideas)
        assert all(idea.thinking_method == ThinkingMethod.ABDUCTION for idea in result.synthesized_ideas)
