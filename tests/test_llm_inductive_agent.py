"""
Tests for LLM-powered Inductive Agent.

This module tests the LLM-based inductive reasoning functionality,
pattern synthesis methods, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.agents.induction.llm_agent import LLMInductiveAgent
from mad_spark_alt.core.interfaces import (
    IdeaGenerationRequest,
    OutputType,
    ThinkingMethod,
)
from mad_spark_alt.core.llm_provider import LLMManager, LLMProvider, LLMResponse


class TestLLMInductiveAgent:
    """Test LLM Inductive Agent functionality."""

    def test_agent_initialization(self):
        """Test LLMInductiveAgent initialization."""
        agent = LLMInductiveAgent()

        assert agent.name == "LLMInductiveAgent"
        assert agent.thinking_method == ThinkingMethod.INDUCTION
        assert OutputType.TEXT in agent.supported_output_types
        assert OutputType.STRUCTURED in agent.supported_output_types

    def test_custom_initialization(self):
        """Test LLMInductiveAgent with custom parameters."""
        mock_llm_manager = MagicMock()
        agent = LLMInductiveAgent(
            name="CustomInductiveAgent",
            llm_manager=mock_llm_manager,
            preferred_provider=LLMProvider.GOOGLE,
        )

        assert agent.name == "CustomInductiveAgent"
        assert agent.llm_manager == mock_llm_manager
        assert agent.preferred_provider == LLMProvider.GOOGLE

    def test_validate_config(self):
        """Test configuration validation."""
        agent = LLMInductiveAgent()

        # Valid config
        valid_config = {
            "inductive_method": "pattern_synthesis",
            "pattern_depth": "deep",
            "synthesis_scope": "broad",
            "creative_synthesis": True,
        }
        assert agent.validate_config(valid_config) is True

        # Invalid config with unknown keys
        invalid_config = {
            "invalid_key": "value",
            "another_invalid": True,
        }
        assert agent.validate_config(invalid_config) is False

        # Empty config should be valid
        assert agent.validate_config({}) is True

    def test_inductive_methods_loading(self):
        """Test that inductive methods are properly loaded."""
        agent = LLMInductiveAgent()
        methods = agent._inductive_methods

        # Check that key methods exist
        expected_methods = [
            "pattern_synthesis",
            "principle_extraction",
            "meta_recognition",
            "creative_synthesis",
            "trend_analysis",
            "analogical_extension",
            "emergent_insight",
        ]

        for method in expected_methods:
            assert method in methods
            assert "name" in methods[method]
            assert "description" in methods[method]
            assert "focus" in methods[method]
            assert "cognitive_approach" in methods[method]

    def test_method_selection(self):
        """Test inductive method selection logic."""
        agent = LLMInductiveAgent()

        # Test with rich data and apparent patterns
        rich_context = {
            "data_richness": "very_rich",
            "pattern_visibility": "apparent",
            "synthesis_complexity": "complex",
            "generalization_potential": "high",
        }
        methods = agent._select_inductive_methods(rich_context, {})

        assert len(methods) <= 4  # Default max methods
        assert all("name" in m for m in methods)

        # Should prioritize synthesis and meta-recognition for rich, visible patterns
        method_names = [m["name"] for m in methods]
        assert any(
            m in method_names
            for m in ["pattern_synthesis", "meta_recognition", "principle_extraction"]
        )

        # Test with high generalization potential
        generalization_context = {
            "data_richness": "moderate",
            "pattern_visibility": "subtle",
            "generalization_potential": "very_high",
            "synthesis_complexity": "moderate",
        }
        methods = agent._select_inductive_methods(
            generalization_context, {"max_methods": 3}
        )

        assert len(methods) <= 3
        # Should prioritize principles and analogies for high generalization
        method_names = [m["name"] for m in methods]
        assert any(
            m in method_names for m in ["principle_extraction", "analogical_extension"]
        )

    @pytest.mark.asyncio
    async def test_synthesis_context_analysis_fallback(self):
        """Test synthesis context analysis with JSON parsing failure."""
        agent = LLMInductiveAgent()

        # Mock LLM manager to return invalid JSON
        mock_llm_manager = AsyncMock()
        mock_response = LLMResponse(
            content="Invalid JSON content",
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            cost=0.001,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        result = await agent._analyze_synthesis_context("Test problem", "Test context")

        # Should return fallback analysis
        assert result["data_richness"] == "moderate"
        assert result["pattern_visibility"] == "subtle"
        assert "insight_opportunities" in result

    @pytest.mark.asyncio
    async def test_successful_synthesis_analysis(self):
        """Test successful synthesis context analysis."""
        agent = LLMInductiveAgent()

        # Mock LLM manager to return valid JSON
        mock_llm_manager = AsyncMock()
        valid_analysis = {
            "data_richness": "rich",
            "pattern_visibility": "apparent",
            "synthesis_complexity": "complex",
            "generalization_potential": "high",
            "observable_patterns": ["recurring themes", "cyclical behavior"],
            "insight_opportunities": ["pattern_synthesis", "meta_recognition"],
        }

        import json

        mock_response = LLMResponse(
            content=json.dumps(valid_analysis),  # Proper JSON conversion
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            cost=0.002,
            response_time=0.7,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        result = await agent._analyze_synthesis_context(
            "How can we identify patterns in user behavior?", "Data analytics context"
        )

        assert result["data_richness"] == "rich"
        assert result["generalization_potential"] == "high"
        assert "llm_cost" in result

    @pytest.mark.asyncio
    async def test_insight_generation_with_method(self):
        """Test insight generation with specific method."""
        agent = LLMInductiveAgent()

        # Mock LLM manager
        mock_llm_manager = AsyncMock()
        insights_data = [
            {
                "insight": "Pattern synthesis reveals recurring optimization cycles in system behavior",
                "synthesis_process": "Observed repeated patterns across multiple system iterations and contexts",
                "supporting_patterns": "Cyclical performance improvements, resource optimization peaks",
                "generalization_scope": "Broadly applicable to adaptive systems and optimization processes",
                "practical_implications": "Design systems with explicit optimization cycles and feedback loops",
                "confidence_level": "high",
            },
            {
                "insight": "Meta-patterns suggest emergent properties arise from component interactions",
                "synthesis_process": "Identified higher-order patterns emerging from component-level behaviors",
                "supporting_patterns": "Component synchronization, emergent coordination behaviors",
                "generalization_scope": "Applicable to complex systems with autonomous components",
                "practical_implications": "Focus on interaction design rather than individual component optimization",
                "confidence_level": "medium",
            },
        ]

        import json

        mock_response = LLMResponse(
            content=json.dumps(insights_data),  # Proper JSON conversion
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            cost=0.005,
            response_time=1.2,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        method = agent._inductive_methods["pattern_synthesis"]
        synthesis_context = {"data_richness": "rich", "pattern_visibility": "apparent"}

        insights = await agent._apply_inductive_method(
            "System optimization patterns",
            "Complex adaptive systems",
            method,
            synthesis_context,
            {},
        )

        assert len(insights) == 2
        assert all(i.thinking_method == ThinkingMethod.INDUCTION for i in insights)
        assert all(i.agent_name == agent.name for i in insights)
        assert all("method" in i.metadata for i in insights)
        assert all("llm_cost" in i.metadata for i in insights)

        # Check confidence score mapping
        assert insights[0].confidence_score == 0.85  # high -> 0.85
        assert insights[1].confidence_score == 0.7  # medium -> 0.7

    @pytest.mark.asyncio
    async def test_generate_ideas_mock_workflow(self):
        """Test complete idea generation workflow with mocked LLM."""
        agent = LLMInductiveAgent()

        # Mock LLM manager for synthesis analysis
        mock_llm_manager = AsyncMock()

        import json

        async def mock_generate(request, provider=None):
            # Synthesis context analysis call
            if "pattern analyst" in request.system_prompt:
                analysis = {
                    "data_richness": "rich",
                    "pattern_visibility": "apparent",
                    "synthesis_complexity": "complex",
                    "generalization_potential": "high",
                    "insight_opportunities": ["pattern_synthesis", "meta_recognition"],
                }
                mock_response = LLMResponse(
                    content=json.dumps(analysis),
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.002,
                    response_time=0.5,
                )
                return mock_response
            # Insight generation call
            elif "insight synthesizer" in request.system_prompt:
                insights = [
                    {
                        "insight": "Test insight from pattern synthesis method",
                        "synthesis_process": "Systematic pattern recognition and abstraction process",
                        "supporting_patterns": "Observable recurring structures",
                        "generalization_scope": "Broadly applicable across similar domains",
                        "practical_implications": "Actionable framework for pattern recognition",
                        "confidence_level": "high",
                    }
                ]
                mock_response = LLMResponse(
                    content=json.dumps(insights),
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.003,
                    response_time=0.7,
                )
                return mock_response
            # Ranking call
            else:
                mock_response = LLMResponse(
                    content="[0]",  # First insight ranked best
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.001,
                    response_time=0.3,
                )
                return mock_response

        mock_llm_manager.generate = mock_generate
        agent.llm_manager = mock_llm_manager

        request = IdeaGenerationRequest(
            problem_statement="How can we identify patterns in complex data systems?",
            context="Data science and pattern recognition",
            max_ideas_per_method=3,
        )

        result = await agent.generate_ideas(request)

        assert result.agent_name == agent.name
        assert result.thinking_method == ThinkingMethod.INDUCTION
        assert len(result.generated_ideas) >= 1
        assert result.execution_time > 0
        assert "synthesis_context" in result.generation_metadata
        assert "methods_used" in result.generation_metadata

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during idea generation."""
        agent = LLMInductiveAgent()

        # Mock LLM manager to raise an exception
        mock_llm_manager = AsyncMock()
        mock_llm_manager.generate.side_effect = Exception("LLM service unavailable")
        agent.llm_manager = mock_llm_manager

        request = IdeaGenerationRequest(
            problem_statement="Test problem",
            max_ideas_per_method=5,
        )

        result = await agent.generate_ideas(request)

        assert result.agent_name == agent.name
        assert result.thinking_method == ThinkingMethod.INDUCTION
        assert len(result.generated_ideas) == 0
        assert result.error_message is not None
        assert "LLM service unavailable" in result.error_message

    @pytest.mark.asyncio
    async def test_ranking_fallback(self):
        """Test insight ranking with fallback mechanism."""
        agent = LLMInductiveAgent()

        # Create test insights with different methods
        insights = []
        for i, method in enumerate(
            ["pattern_synthesis", "principle_extraction", "creative_synthesis"]
        ):
            from datetime import datetime

            from mad_spark_alt.core.interfaces import GeneratedIdea

            idea = GeneratedIdea(
                content=f"Test insight {i+1}",
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name=agent.name,
                generation_prompt="Test prompt",
                confidence_score=0.8,
                reasoning="Test reasoning",
                metadata={"method": method},
                timestamp=datetime.now().isoformat(),
            )
            insights.append(idea)

        # Mock failed ranking (should trigger fallback)
        mock_llm_manager = AsyncMock()
        mock_llm_manager.generate.side_effect = Exception("Ranking failed")
        agent.llm_manager = mock_llm_manager

        selected = await agent._rank_and_select_insights(
            insights, 2, "test problem", {"data_richness": "moderate"}
        )

        # Should return 2 insights using fallback method diversity
        assert len(selected) == 2
        methods = [i.metadata["method"] for i in selected]
        assert len(set(methods)) == 2  # Different methods


# Fixtures for testing
@pytest.fixture
def sample_inductive_agent():
    """Sample LLM inductive agent for testing."""
    return LLMInductiveAgent(name="TestInductiveAgent")


@pytest.fixture
def sample_synthesis_context():
    """Sample synthesis context for testing."""
    return {
        "data_richness": "rich",
        "pattern_visibility": "apparent",
        "synthesis_complexity": "complex",
        "generalization_potential": "high",
        "observable_patterns": ["recurring themes", "cyclical behavior"],
        "insight_opportunities": ["pattern_synthesis", "meta_recognition"],
    }
