"""Tests for SimpleQADIOrchestrator - the true QADI hypothesis-driven implementation."""

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
    """Test cases for SimpleQADIOrchestrator."""

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
    async def test_questioning_phase(self, orchestrator, mock_llm_manager):
        """Test the questioning phase extracts core question correctly."""
        # Mock LLM response
        mock_response = LLMResponse(
            content="Q: How can we improve employee engagement in remote teams?",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            cost=0.001,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            question, cost = await orchestrator._run_questioning_phase(
                "I need help with remote team management", max_retries=1
            )

        assert question == "How can we improve employee engagement in remote teams?"
        assert cost == 0.001
        mock_llm_manager.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_questioning_phase_fallback(self, orchestrator, mock_llm_manager):
        """Test questioning phase fallback when no Q: prefix found."""
        # Mock LLM response without Q: prefix
        mock_response = LLMResponse(
            content="What strategies can enhance remote collaboration?",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            cost=0.001,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            question, cost = await orchestrator._run_questioning_phase(
                "Remote work challenges", max_retries=1
            )

        assert question == "What strategies can enhance remote collaboration?"
        assert cost == 0.001

    @pytest.mark.asyncio
    async def test_abduction_phase(self, orchestrator, mock_llm_manager):
        """Test the abduction phase generates hypotheses correctly."""
        # Mock LLM response with hypotheses
        mock_response = LLMResponse(
            content="""H1: Implement regular virtual coffee breaks to foster casual interactions
H2: Create structured asynchronous communication protocols for different time zones
H3: Establish clear performance metrics based on outcomes rather than hours""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 30, "completion_tokens": 60},
            cost=0.003,
            response_time=0.8,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "Remote work challenges",
                "How can we improve remote team engagement?",
                max_retries=1,
            )

        assert len(hypotheses) == 3
        assert "virtual coffee breaks" in hypotheses[0]
        assert "asynchronous communication" in hypotheses[1]
        assert "performance metrics" in hypotheses[2]
        assert cost == 0.003

    @pytest.mark.asyncio
    async def test_deduction_phase(self, orchestrator, mock_llm_manager):
        """Test the deduction phase evaluates hypotheses and provides answer."""
        # Mock LLM response with evaluation (realistic format using new 5-criteria system)
        mock_response = LLMResponse(
            content="""Analysis:
- H1: 
  * Impact: 0.7 - significant boost to team morale and informal communication
  * Feasibility: 0.8 - highly practical, just needs coordination
  * Accessibility: 0.9 - very easy for all team members to participate
  * Sustainability: 0.7 - can be maintained long-term with minimal effort
  * Scalability: 0.6 - works well for small to medium teams
  * Overall: 0.68

- H2: 
  * Impact: 0.8 - addresses core remote work collaboration challenges
  * Feasibility: 0.7 - practical but needs organizational commitment
  * Accessibility: 0.8 - requires some training but manageable
  * Sustainability: 0.8 - sustainable with proper process documentation
  * Scalability: 0.9 - scales well across different team sizes
  * Overall: 0.74

ANSWER: The most effective approach is to implement structured asynchronous communication protocols, as this addresses the core challenge of coordinating across time zones while maintaining team cohesion.

Action Plan:
1. Document communication expectations and response times
2. Set up dedicated channels for different types of communication
3. Train team members on async best practices""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.005,
            response_time=1.0,
        )
        mock_llm_manager.generate.return_value = mock_response

        hypotheses = ["Virtual coffee breaks", "Async communication protocols"]

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            result = await orchestrator._run_deduction_phase(
                "Remote work",
                "How to improve engagement?",
                hypotheses,
                max_retries=1,
            )

        assert len(result["scores"]) == 2
        assert result["scores"][0].impact == 0.7
        assert result["scores"][0].feasibility == 0.8
        assert result["scores"][0].accessibility == 0.9
        assert result["scores"][1].impact == 0.8
        assert result["scores"][1].feasibility == 0.7
        assert result["scores"][1].scalability == 0.9
        assert "asynchronous communication protocols" in result["answer"]
        assert len(result["action_plan"]) == 3
        assert result["cost"] == 0.005

    @pytest.mark.asyncio
    async def test_induction_phase(self, orchestrator, mock_llm_manager):
        """Test the induction phase verifies answer with examples."""
        # Mock LLM response with verification
        mock_response = LLMResponse(
            content="""Verification Examples:

1. GitLab successfully implemented async communication protocols and saw 40% improvement in cross-timezone collaboration efficiency.

2. Automattic (WordPress) uses P2 blogs for async updates, resulting in better documentation and knowledge sharing across their fully remote team.

3. Doist created detailed async communication guidelines that reduced meeting time by 60% while maintaining project velocity.

Conclusion: The evidence strongly supports that structured async communication is key to successful remote team engagement.""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 40, "completion_tokens": 80},
            cost=0.004,
            response_time=0.9,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            result = await orchestrator._run_induction_phase(
                "Remote work",
                "How to improve engagement?",
                "Implement async communication protocols",
                ["Virtual coffee breaks", "Async communication protocols", "Performance metrics"],
                max_retries=1,
            )

        # In the parsing logic, all examples should be parsed correctly
        assert len(result["examples"]) >= 3  # Should have at least 3 examples
        assert "GitLab" in result["examples"][0]
        assert "Automattic" in result["examples"][1]
        assert "Doist" in result["examples"][2]
        assert "evidence strongly supports" in result["conclusion"]
        assert result["cost"] == 0.004

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
            # Induction phase
            LLMResponse(
                content="""1. London's congestion charge reduced traffic by 30% and emissions by 20% in the charging zone.

2. Singapore's Electronic Road Pricing cut peak hour traffic by 24% while improving air quality.

3. Stockholm saw 22% emission reduction and generated €100M annually for transport improvements.

Conclusion: Real-world implementations consistently demonstrate congestion pricing's effectiveness in reducing urban transport emissions.""",
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
        assert len(result.verification_examples) >= 3  # Should have at least 3 examples
        assert "Real-world implementations" in result.verification_conclusion
        assert pytest.approx(result.total_llm_cost, rel=1e-5) == 0.013  # Sum of all costs
        
        # Verify synthesized ideas for evolution compatibility
        assert len(result.synthesized_ideas) == 3
        assert all(isinstance(idea, GeneratedIdea) for idea in result.synthesized_ideas)
        assert all(idea.thinking_method == ThinkingMethod.ABDUCTION for idea in result.synthesized_ideas)

    @pytest.mark.asyncio
    async def test_temperature_override_in_abduction(self, orchestrator_with_temp, mock_llm_manager):
        """Test that temperature override is applied in abduction phase."""
        mock_response = LLMResponse(
            content="H1: Test hypothesis for solution one\nH2: Another hypothesis for approach two\nH3: Third hypothesis for alternative method",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            cost=0.001,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            await orchestrator_with_temp._run_abduction_phase(
                "Test input", "Test question", max_retries=1
            )

        # Verify the temperature was overridden
        call_args = mock_llm_manager.generate.call_args[0][0]
        assert call_args.temperature == 1.2

    def test_hypothesis_score_parsing(self, orchestrator):
        """Test parsing of hypothesis scores from deduction content."""
        content = """Analysis:
- H1: 
  * Impact: 0.7 - significant positive change expected
  * Feasibility: 0.9 - very practical to implement
  * Accessibility: 0.8 - easily accessible to target users
  * Sustainability: 0.6 - moderately sustainable approach
  * Scalability: 0.8 - scales well across contexts
  * Overall: 0.74

- H2: 
  * Impact: 0.8 - substantial positive outcomes
  * Feasibility: 0.6 - moderate implementation complexity
  * Accessibility: 0.7 - reasonable accessibility requirements
  * Sustainability: 0.9 - highly sustainable long-term
  * Scalability: 0.5 - limited scalability potential
  * Overall: 0.68

ANSWER: The answer based on evaluation"""

        score1 = orchestrator._parse_hypothesis_scores(content, 1)
        assert score1.impact == 0.7
        assert score1.feasibility == 0.9
        assert score1.accessibility == 0.8
        assert score1.sustainability == 0.6
        assert score1.scalability == 0.8
        assert 0.6 < score1.overall < 0.8  # Weighted calculation

        score2 = orchestrator._parse_hypothesis_scores(content, 2)
        assert score2.impact == 0.8
        assert score2.feasibility == 0.6

    def test_hypothesis_score_parsing_with_defaults(self, orchestrator):
        """Test hypothesis score parsing returns defaults on failure."""
        content = "Invalid content without proper formatting"
        
        score = orchestrator._parse_hypothesis_scores(content, 1)
        assert score.impact == 0.5
        assert score.feasibility == 0.5
        assert score.accessibility == 0.5
        assert score.sustainability == 0.5
        assert score.scalability == 0.5
        assert score.overall == 0.5

    def test_hypothesis_score_parsing_fractional(self, orchestrator):
        """Test parsing of fractional scores (e.g., 8/10)."""
        content = """Analysis:
- H1: 
  * Impact: 9/10 - significant improvement
  * Feasibility: 7/10 - quite achievable
  * Accessibility: 8/10 - highly accessible
  * Sustainability: 6/10 - moderately sustainable
  * Scalability: 5/10 - limited scalability
"""
        
        score = orchestrator._parse_hypothesis_scores(content, 1)
        assert score.impact == 0.9
        assert score.feasibility == 0.7
        assert score.accessibility == 0.8
        assert score.sustainability == 0.6
        assert score.scalability == 0.5
        assert 0.5 < score.overall < 0.8  # Weighted calculation

    @pytest.mark.asyncio
    async def test_error_handling_in_phases(self, orchestrator, mock_llm_manager):
        """Test error handling and retries in different phases."""
        # Mock to fail first attempt, succeed on second
        mock_llm_manager.generate.side_effect = [
            Exception("Network error"),
            LLMResponse(
                content="Q: Test question",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                cost=0.001,
                response_time=0.5,
            ),
        ]

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            question, cost = await orchestrator._run_questioning_phase(
                "Test input", max_retries=2
            )

        assert question == "Test question"
        assert cost == 0.001
        assert mock_llm_manager.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, orchestrator, mock_llm_manager):
        """Test behavior when max retries are exceeded."""
        mock_llm_manager.generate.side_effect = Exception("Persistent error")

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            with pytest.raises(RuntimeError, match="Failed to extract core question"):
                await orchestrator._run_questioning_phase("Test input", max_retries=1)

        assert mock_llm_manager.generate.call_count == 2  # Initial + 1 retry