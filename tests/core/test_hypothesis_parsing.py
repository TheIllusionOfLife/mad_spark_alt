"""
Unit tests for hypothesis parsing logic in SimpleQADIOrchestrator.

Tests various LLM response formats to ensure robust parsing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


class TestHypothesisParsing:
    """Test hypothesis parsing logic with various LLM response formats."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SimpleQADIOrchestrator(num_hypotheses=3)

    @pytest.fixture
    def mock_llm_manager(self, monkeypatch):
        """Mock the LLM manager for controlled testing."""
        mock_manager = MagicMock()
        mock_manager.generate = AsyncMock()
        monkeypatch.setattr("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_manager)
        return mock_manager

    @pytest.mark.asyncio
    async def test_parse_standard_h_format(self, orchestrator, mock_llm_manager):
        """Test parsing standard H1:, H2:, H3: format."""
        # Mock LLM response with standard format
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
H1: Implement city-wide composting programs
H2: Create urban rooftop gardens
H3: Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_parse_markdown_format(self, orchestrator, mock_llm_manager):
        """Test parsing markdown-formatted hypotheses."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
**H1:** Implement city-wide composting programs to reduce organic waste
**H2:** Create urban rooftop gardens for local food production
**H3:** Develop food waste tracking apps with gamification
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs to reduce organic waste"
        assert hypotheses[1] == "Create urban rooftop gardens for local food production"
        assert hypotheses[2] == "Develop food waste tracking apps with gamification"

    @pytest.mark.asyncio
    async def test_parse_multiline_format(self, orchestrator, mock_llm_manager):
        """Test parsing multi-line hypothesis content."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
**H1:**
Implement city-wide composting programs that:
- Provide composting bins to all households
- Create neighborhood collection points
- Partner with local farms for compost use

**H2:**
Create urban rooftop gardens by:
- Converting unused rooftop spaces
- Establishing community garden cooperatives
- Growing food locally to reduce transport waste

**H3:**
Develop food waste tracking apps that:
- Monitor household food waste patterns
- Provide tips for reducing waste
- Connect surplus food with local food banks
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert "Implement city-wide composting programs that:" in hypotheses[0]
        assert "Provide composting bins to all households" in hypotheses[0]
        assert "Create urban rooftop gardens by:" in hypotheses[1]
        assert "Develop food waste tracking apps that:" in hypotheses[2]

    @pytest.mark.asyncio
    async def test_parse_approach_format(self, orchestrator, mock_llm_manager):
        """Test parsing 'Approach' prefix format."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
Approach 1: Implement city-wide composting programs
Approach 2: Create urban rooftop gardens
Approach 3: Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_parse_hypothesis_format(self, orchestrator, mock_llm_manager):
        """Test parsing 'Hypothesis' prefix format."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
Hypothesis 1: Implement city-wide composting programs
Hypothesis 2: Create urban rooftop gardens
Hypothesis 3: Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_fallback_parsing_numbered_list(self, orchestrator, mock_llm_manager):
        """Test fallback parsing for numbered list format."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
Here are three approaches to reduce food waste:

1. Implement city-wide composting programs
2. Create urban rooftop gardens  
3. Develop food waste tracking apps

These approaches address different aspects of the problem.
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_fallback_parsing_bullet_points(self, orchestrator, mock_llm_manager):
        """Test fallback parsing for bullet point format."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
Here are three approaches:

• Implement city-wide composting programs
• Create urban rooftop gardens
• Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_min_hypothesis_length_filtering(self, orchestrator, mock_llm_manager):
        """Test that very short hypotheses are filtered out."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
H1: Yes
H2: Implement comprehensive city-wide composting programs
H3: No
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        # Set num_hypotheses to 1 to test filtering
        orchestrator.num_hypotheses = 1
        
        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 1
        assert hypotheses[0] == "Implement comprehensive city-wide composting programs"

    @pytest.mark.asyncio
    async def test_empty_hypothesis_content_handling(self, orchestrator, mock_llm_manager):
        """Test handling of empty hypothesis content after title."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
**H1:**

**H2:** Create urban rooftop gardens

**H3:** Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        orchestrator.num_hypotheses = 2  # Only expect 2 valid hypotheses

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 2
        assert hypotheses[0] == "Create urban rooftop gardens"
        assert hypotheses[1] == "Develop food waste tracking apps"

    @pytest.mark.asyncio
    async def test_mixed_format_parsing(self, orchestrator, mock_llm_manager):
        """Test parsing mixed formats in same response."""
        mock_llm_manager.generate.return_value = LLMResponse(
            content="""
H1: Implement city-wide composting programs

**Hypothesis 2:** Create urban rooftop gardens

Approach 3: Develop food waste tracking apps
""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={},
            cost=0.0
        )

        hypotheses, _ = await orchestrator._run_abduction_phase(
            "How to reduce food waste?",
            "How can we reduce food waste?",
            max_retries=0
        )

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Implement city-wide composting programs"
        assert hypotheses[1] == "Create urban rooftop gardens"
        assert hypotheses[2] == "Develop food waste tracking apps"