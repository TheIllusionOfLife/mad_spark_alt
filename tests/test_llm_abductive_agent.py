"""
Tests for LLM-powered Abductive Agent.

This module tests the LLM-based abductive reasoning functionality,
hypothesis generation strategies, and error handling.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.agents.abduction.llm_agent import LLMAbductiveAgent
from mad_spark_alt.core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    OutputType,
    ThinkingMethod,
)
from mad_spark_alt.core.llm_provider import LLMManager, LLMProvider, LLMResponse


class TestLLMAbductiveAgent:
    """Test LLM Abductive Agent functionality."""

    def test_agent_initialization(self):
        """Test LLMAbductiveAgent initialization."""
        agent = LLMAbductiveAgent()

        assert agent.name == "LLMAbductiveAgent"
        assert agent.thinking_method == ThinkingMethod.ABDUCTION
        assert OutputType.TEXT in agent.supported_output_types
        assert OutputType.STRUCTURED in agent.supported_output_types

    def test_custom_initialization(self):
        """Test LLMAbductiveAgent with custom parameters."""
        mock_llm_manager = MagicMock()
        agent = LLMAbductiveAgent(
            name="CustomAbductiveAgent",
            llm_manager=mock_llm_manager,
            preferred_provider=LLMProvider.GOOGLE,
        )

        assert agent.name == "CustomAbductiveAgent"
        assert agent.llm_manager == mock_llm_manager
        assert agent.preferred_provider == LLMProvider.GOOGLE

    def test_validate_config(self):
        """Test configuration validation."""
        agent = LLMAbductiveAgent()

        # Valid config with supported key
        valid_config = {
            "max_strategies": 3,
        }
        assert agent.validate_config(valid_config) is True

        # Invalid config with unknown keys
        invalid_config = {
            "invalid_key": "value",
            "another_invalid": True,
        }
        assert agent.validate_config(invalid_config) is False

        # Invalid max_strategies value (too high)
        invalid_max_config = {
            "max_strategies": 10,
        }
        assert agent.validate_config(invalid_max_config) is False

        # Invalid max_strategies value (negative)
        invalid_negative_config = {
            "max_strategies": -1,
        }
        assert agent.validate_config(invalid_negative_config) is False

        # Empty config should be valid
        assert agent.validate_config({}) is True

    def test_abductive_strategies_loading(self):
        """Test that abductive strategies are properly loaded."""
        agent = LLMAbductiveAgent()
        strategies = agent._abductive_strategies

        # Check that key strategies exist
        expected_strategies = [
            "causal_inference",
            "analogical_reasoning",
            "pattern_recognition",
            "counter_intuitive",
            "what_if_scenarios",
            "systems_perspective",
            "temporal_reasoning",
        ]

        for strategy in expected_strategies:
            assert strategy in strategies
            assert "name" in strategies[strategy]
            assert "description" in strategies[strategy]
            assert "focus" in strategies[strategy]
            assert "cognitive_approach" in strategies[strategy]

    def test_strategy_selection(self):
        """Test abductive strategy selection logic."""
        agent = LLMAbductiveAgent()

        # Test with high complexity context
        complex_context = {
            "domain": "technology",
            "causal_complexity": "complex",
            "evidence_availability": "moderate",
            "uncertainty_level": "high",
        }
        strategies = agent._select_abductive_strategies(complex_context, {})

        assert len(strategies) <= 4  # Default max strategies
        assert all("name" in s for s in strategies)

        # Test with sparse evidence context
        sparse_context = {
            "domain": "healthcare",
            "causal_complexity": "moderate",
            "evidence_availability": "sparse",
            "uncertainty_level": "medium",
        }
        strategies = agent._select_abductive_strategies(
            sparse_context, {"max_strategies": 3}
        )

        assert len(strategies) <= 3
        # Should prioritize creative approaches for sparse evidence
        strategy_names = [s["name"] for s in strategies]
        assert any(
            s in strategy_names for s in ["analogical_reasoning", "what_if_scenarios"]
        )

    @pytest.mark.asyncio
    async def test_context_analysis_fallback(self):
        """Test context analysis with JSON parsing failure."""
        agent = LLMAbductiveAgent()

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

        result = await agent._analyze_problem_context("Test problem", "Test context")

        # Should return fallback analysis
        assert result["domain"] == "general"
        assert result["problem_nature"] == "ill_structured"
        assert "abductive_opportunities" in result

    @pytest.mark.asyncio
    async def test_successful_context_analysis(self):
        """Test successful context analysis."""
        agent = LLMAbductiveAgent()

        # Mock LLM manager to return valid JSON
        mock_llm_manager = AsyncMock()
        valid_analysis = {
            "domain": "technology",
            "problem_nature": "ill_structured",
            "causal_complexity": "complex",
            "uncertainty_level": "high",
            "abductive_opportunities": ["pattern_recognition", "analogical_reasoning"],
        }

        mock_response = LLMResponse(
            content=json.dumps(valid_analysis),  # Proper JSON conversion
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            cost=0.002,
            response_time=0.7,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        result = await agent._analyze_problem_context(
            "How can we improve software reliability?", "Enterprise context"
        )

        assert result["domain"] == "technology"
        assert result["causal_complexity"] == "complex"
        assert "llm_cost" in result

    @pytest.mark.asyncio
    async def test_hypothesis_generation_with_strategy(self):
        """Test hypothesis generation with specific strategy."""
        agent = LLMAbductiveAgent()

        # Mock LLM manager
        mock_llm_manager = AsyncMock()
        hypotheses_data = [
            {
                "hypothesis": "The problem emerges from unexpected system interactions",
                "reasoning": "Systems thinking reveals hidden connections",
                "evidence_requirements": "System interaction logs",
                "implications": "Need holistic approach",
                "confidence_level": "medium",
            },
            {
                "hypothesis": "Feedback loops amplify initial small disturbances",
                "reasoning": "Complex systems show emergent behaviors",
                "evidence_requirements": "Temporal data analysis",
                "implications": "Early intervention is critical",
                "confidence_level": "high",
            },
        ]

        mock_response = LLMResponse(
            content=json.dumps(hypotheses_data),  # Proper JSON conversion
            provider=LLMProvider.GOOGLE,
            model="gemini-3-flash-preview",
            cost=0.005,
            response_time=1.2,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        strategy = agent._abductive_strategies["systems_perspective"]
        context_analysis = {"domain": "technology", "causal_complexity": "complex"}

        hypotheses = await agent._generate_hypotheses_with_strategy(
            "System reliability issues",
            "Enterprise software context",
            strategy,
            context_analysis,
            {},
        )

        assert len(hypotheses) == 2
        assert all(h.thinking_method == ThinkingMethod.ABDUCTION for h in hypotheses)
        assert all(h.agent_name == agent.name for h in hypotheses)
        assert all("strategy" in h.metadata for h in hypotheses)
        assert all("llm_cost" in h.metadata for h in hypotheses)

        # Check confidence score mapping
        assert hypotheses[0].confidence_score == 0.6  # medium -> 0.6
        assert hypotheses[1].confidence_score == 0.8  # high -> 0.8

    @pytest.mark.asyncio
    async def test_generate_ideas_mock_workflow(self):
        """Test complete idea generation workflow with mocked LLM."""
        agent = LLMAbductiveAgent()

        # Mock LLM manager for context analysis
        mock_llm_manager = AsyncMock()

        async def mock_generate(request, provider=None):
            # Context analysis call
            if "problem analyst" in request.system_prompt:
                analysis = {
                    "domain": "technology",
                    "causal_complexity": "complex",
                    "evidence_availability": "moderate",
                    "uncertainty_level": "medium",
                }
                mock_response = LLMResponse(
                    content=json.dumps(analysis),
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.002,
                    response_time=0.5,
                )
                return mock_response
            # Hypothesis generation call
            elif "hypothesis generator" in request.system_prompt:
                hypotheses = [
                    {
                        "hypothesis": "Test hypothesis from systems perspective",
                        "reasoning": "Complex systems exhibit emergent properties",
                        "evidence_requirements": "System behavior data",
                        "implications": "Holistic intervention needed",
                        "confidence_level": "high",
                    }
                ]
                mock_response = LLMResponse(
                    content=json.dumps(hypotheses),
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.003,
                    response_time=0.7,
                )
                return mock_response
            # Ranking call
            else:
                mock_response = LLMResponse(
                    content="[0]",  # First hypothesis ranked best
                    provider=LLMProvider.GOOGLE,
                    model="gemini-3-flash-preview",
                    cost=0.001,
                    response_time=0.3,
                )
                return mock_response

        mock_llm_manager.generate = mock_generate
        agent.llm_manager = mock_llm_manager

        request = IdeaGenerationRequest(
            problem_statement="How can we improve system reliability?",
            context="Enterprise software environment",
            max_ideas_per_method=3,
        )

        result = await agent.generate_ideas(request)

        assert result.agent_name == agent.name
        assert result.thinking_method == ThinkingMethod.ABDUCTION
        assert len(result.generated_ideas) >= 1
        assert result.execution_time > 0
        assert "context_analysis" in result.generation_metadata
        assert "strategies_used" in result.generation_metadata

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during idea generation."""
        agent = LLMAbductiveAgent()

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
        assert result.thinking_method == ThinkingMethod.ABDUCTION
        assert len(result.generated_ideas) == 0
        assert result.error_message is not None
        assert "LLM service unavailable" in result.error_message

    @pytest.mark.asyncio
    async def test_ranking_fallback(self):
        """Test hypothesis ranking with fallback mechanism."""
        agent = LLMAbductiveAgent()

        # Create test hypotheses with different strategies
        hypotheses = []
        for i, strategy in enumerate(
            ["causal_inference", "analogical_reasoning", "pattern_recognition"]
        ):
            idea = GeneratedIdea(
                content=f"Test hypothesis {i+1}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name=agent.name,
                generation_prompt="Test prompt",
                confidence_score=0.7,
                reasoning="Test reasoning",
                metadata={"strategy": strategy},
                timestamp=datetime.now().isoformat(),
            )
            hypotheses.append(idea)

        # Mock failed ranking (should trigger fallback)
        mock_llm_manager = AsyncMock()
        mock_llm_manager.generate.side_effect = Exception("Ranking failed")
        agent.llm_manager = mock_llm_manager

        selected = await agent._rank_and_select_hypotheses(
            hypotheses, 2, "test problem", {"domain": "test"}
        )

        # Should return 2 hypotheses using fallback strategy diversity
        assert len(selected) == 2
        strategies = [h.metadata["strategy"] for h in selected]
        assert len(set(strategies)) == 2  # Different strategies


# Fixtures for testing
@pytest.fixture
def sample_abductive_agent():
    """Sample LLM abductive agent for testing."""
    return LLMAbductiveAgent(name="TestAbductiveAgent")


@pytest.fixture
def sample_context_analysis():
    """Sample context analysis for testing."""
    return {
        "domain": "technology",
        "problem_nature": "ill_structured",
        "causal_complexity": "complex",
        "evidence_availability": "moderate",
        "uncertainty_level": "high",
        "abductive_opportunities": ["pattern_recognition", "analogical_reasoning"],
    }
