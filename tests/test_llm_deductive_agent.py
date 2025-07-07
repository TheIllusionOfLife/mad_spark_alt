"""
Tests for LLM-powered Deductive Agent.

This module tests the LLM-based deductive reasoning functionality,
logical validation frameworks, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.agents.deduction.llm_agent import LLMDeductiveAgent
from mad_spark_alt.core.interfaces import (
    IdeaGenerationRequest,
    OutputType,
    ThinkingMethod,
)
from mad_spark_alt.core.llm_provider import LLMManager, LLMProvider, LLMResponse


class TestLLMDeductiveAgent:
    """Test LLM Deductive Agent functionality."""

    def test_agent_initialization(self):
        """Test LLMDeductiveAgent initialization."""
        agent = LLMDeductiveAgent()

        assert agent.name == "LLMDeductiveAgent"
        assert agent.thinking_method == ThinkingMethod.DEDUCTION
        assert OutputType.TEXT in agent.supported_output_types
        assert OutputType.STRUCTURED in agent.supported_output_types

    def test_custom_initialization(self):
        """Test LLMDeductiveAgent with custom parameters."""
        mock_llm_manager = MagicMock()
        agent = LLMDeductiveAgent(
            name="CustomDeductiveAgent",
            llm_manager=mock_llm_manager,
            preferred_provider=LLMProvider.ANTHROPIC,
        )

        assert agent.name == "CustomDeductiveAgent"
        assert agent.llm_manager == mock_llm_manager
        assert agent.preferred_provider == LLMProvider.ANTHROPIC

    def test_validate_config(self):
        """Test configuration validation."""
        agent = LLMDeductiveAgent()

        # Valid config
        valid_config = {
            "deductive_framework": "logical_validation",
            "logical_depth": "deep",
            "validation_rigor": "high",
            "include_counterarguments": True,
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

    def test_deductive_frameworks_loading(self):
        """Test that deductive frameworks are properly loaded."""
        agent = LLMDeductiveAgent()
        frameworks = agent._deductive_frameworks

        # Check that key frameworks exist
        expected_frameworks = [
            "logical_validation",
            "consequence_analysis",
            "requirement_validation",
            "constraint_analysis",
            "proof_construction",
            "systematic_decomposition",
            "evidence_validation",
        ]

        for framework in expected_frameworks:
            assert framework in frameworks
            assert "name" in frameworks[framework]
            assert "description" in frameworks[framework]
            assert "focus" in frameworks[framework]
            assert "cognitive_approach" in frameworks[framework]

    def test_framework_selection(self):
        """Test deductive framework selection logic."""
        agent = LLMDeductiveAgent()

        # Test with complex formal logic problem
        complex_analysis = {
            "logical_complexity": "highly_complex",
            "evidence_base": "strong",
            "formal_logic_applicable": True,
            "reasoning_chain_depth": "very_deep",
        }
        frameworks = agent._select_deductive_frameworks(complex_analysis, {})

        assert len(frameworks) <= 4  # Default max frameworks
        assert all("name" in f for f in frameworks)

        # Should prioritize formal logic frameworks for complex problems
        framework_names = [f["name"] for f in frameworks]
        assert any(
            f in framework_names
            for f in [
                "logical_validation",
                "proof_construction",
                "systematic_decomposition",
            ]
        )

        # Test with simple problem
        simple_analysis = {
            "logical_complexity": "simple",
            "evidence_base": "moderate",
            "formal_logic_applicable": False,
            "reasoning_chain_depth": "shallow",
        }
        frameworks = agent._select_deductive_frameworks(
            simple_analysis, {"max_frameworks": 3}
        )

        assert len(frameworks) <= 3
        # Should prioritize validation and requirements for simple problems
        framework_names = [f["name"] for f in frameworks]
        assert any(
            f in framework_names
            for f in ["logical_validation", "requirement_validation"]
        )

    @pytest.mark.asyncio
    async def test_logical_structure_analysis_fallback(self):
        """Test logical structure analysis with JSON parsing failure."""
        agent = LLMDeductiveAgent()

        # Mock LLM manager to return invalid JSON
        mock_llm_manager = AsyncMock()
        mock_response = LLMResponse(
            content="Invalid JSON content",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            cost=0.001,
            response_time=0.5,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        result = await agent._analyze_logical_structure("Test problem", "Test context")

        # Should return fallback analysis
        assert result["logical_complexity"] == "moderate"
        assert result["problem_type"] == "semi_structured"
        assert "systematic_analysis_scope" in result

    @pytest.mark.asyncio
    async def test_successful_logical_analysis(self):
        """Test successful logical structure analysis."""
        agent = LLMDeductiveAgent()

        # Mock LLM manager to return valid JSON
        mock_llm_manager = AsyncMock()
        valid_analysis = {
            "logical_complexity": "complex",
            "problem_type": "well_defined",
            "evidence_base": "strong",
            "formal_logic_applicable": True,
            "reasoning_chain_depth": "deep",
            "systematic_analysis_scope": ["requirements", "validation", "consequences"],
        }

        import json

        mock_response = LLMResponse(
            content=json.dumps(valid_analysis),  # Proper JSON conversion
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            cost=0.002,
            response_time=0.7,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        result = await agent._analyze_logical_structure(
            "How can we prove software correctness?", "Formal verification context"
        )

        assert result["logical_complexity"] == "complex"
        assert result["formal_logic_applicable"] is True
        assert "llm_cost" in result

    @pytest.mark.asyncio
    async def test_analysis_generation_with_framework(self):
        """Test logical analysis generation with specific framework."""
        agent = LLMDeductiveAgent()

        # Mock LLM manager
        mock_llm_manager = AsyncMock()
        analyses_data = [
            {
                "analysis": "The logical validation reveals systematic requirements for verification",
                "reasoning_chain": "If we establish premises P1 and P2, then conclusion C follows logically",
                "premises": "Well-defined input specifications and verification criteria",
                "implications": "Requires formal proof methods and systematic validation",
                "validation_criteria": "Logical consistency and empirical verification",
                "confidence_level": "high",
            },
            {
                "analysis": "Constraint analysis shows necessary conditions for solution validity",
                "reasoning_chain": "Given constraints X, Y, Z, only solutions meeting all constraints are valid",
                "premises": "Problem constraints are well-defined and measurable",
                "implications": "Solution space is constrained by logical boundaries",
                "validation_criteria": "Constraint satisfaction and boundary testing",
                "confidence_level": "medium",
            },
        ]

        import json

        mock_response = LLMResponse(
            content=json.dumps(analyses_data),  # Proper JSON conversion
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            cost=0.005,
            response_time=1.2,
        )
        mock_llm_manager.generate.return_value = mock_response
        agent.llm_manager = mock_llm_manager

        framework = agent._deductive_frameworks["logical_validation"]
        logical_analysis = {"logical_complexity": "complex", "evidence_base": "strong"}

        analyses = await agent._apply_deductive_framework(
            "System verification requirements",
            "Formal verification context",
            framework,
            logical_analysis,
            {},
        )

        assert len(analyses) == 2
        assert all(a.thinking_method == ThinkingMethod.DEDUCTION for a in analyses)
        assert all(a.agent_name == agent.name for a in analyses)
        assert all("framework" in a.metadata for a in analyses)
        assert all("llm_cost" in a.metadata for a in analyses)

        # Check confidence score mapping
        assert analyses[0].confidence_score == 0.9  # high -> 0.9
        assert analyses[1].confidence_score == 0.7  # medium -> 0.7

    @pytest.mark.asyncio
    async def test_generate_ideas_mock_workflow(self):
        """Test complete idea generation workflow with mocked LLM."""
        agent = LLMDeductiveAgent()

        # Mock LLM manager for logical analysis
        mock_llm_manager = AsyncMock()

        import json

        async def mock_generate(request, provider=None):
            # Logical structure analysis call
            if "logical analyst" in request.system_prompt:
                analysis = {
                    "logical_complexity": "complex",
                    "problem_type": "well_defined",
                    "evidence_base": "strong",
                    "formal_logic_applicable": True,
                    "reasoning_chain_depth": "deep",
                }
                mock_response = LLMResponse(
                    content=json.dumps(analysis),
                    provider=LLMProvider.OPENAI,
                    model="gpt-4o-mini",
                    cost=0.002,
                    response_time=0.5,
                )
                return mock_response
            # Analysis generation call
            elif "logical reasoner" in request.system_prompt:
                analyses = [
                    {
                        "analysis": "Test logical analysis from validation framework",
                        "reasoning_chain": "Formal logical validation process applied systematically",
                        "premises": "Well-defined problem structure",
                        "implications": "Systematic validation approach required",
                        "validation_criteria": "Logical consistency and proof verification",
                        "confidence_level": "high",
                    }
                ]
                mock_response = LLMResponse(
                    content=json.dumps(analyses),
                    provider=LLMProvider.OPENAI,
                    model="gpt-4o-mini",
                    cost=0.003,
                    response_time=0.7,
                )
                return mock_response
            # Ranking call
            else:
                mock_response = LLMResponse(
                    content="[0]",  # First analysis ranked best
                    provider=LLMProvider.OPENAI,
                    model="gpt-4o-mini",
                    cost=0.001,
                    response_time=0.3,
                )
                return mock_response

        mock_llm_manager.generate = mock_generate
        agent.llm_manager = mock_llm_manager

        request = IdeaGenerationRequest(
            problem_statement="How can we systematically verify software correctness?",
            context="Formal verification and proof methods",
            max_ideas_per_method=3,
        )

        result = await agent.generate_ideas(request)

        assert result.agent_name == agent.name
        assert result.thinking_method == ThinkingMethod.DEDUCTION
        assert len(result.generated_ideas) >= 1
        assert result.execution_time > 0
        assert "logical_analysis" in result.generation_metadata
        assert "frameworks_used" in result.generation_metadata

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during idea generation."""
        agent = LLMDeductiveAgent()

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
        assert result.thinking_method == ThinkingMethod.DEDUCTION
        assert len(result.generated_ideas) == 0
        assert result.error_message is not None
        assert "LLM service unavailable" in result.error_message

    @pytest.mark.asyncio
    async def test_ranking_fallback(self):
        """Test analysis ranking with fallback mechanism."""
        agent = LLMDeductiveAgent()

        # Create test analyses with different frameworks
        analyses = []
        for i, framework in enumerate(
            ["logical_validation", "consequence_analysis", "requirement_validation"]
        ):
            from datetime import datetime

            from mad_spark_alt.core.interfaces import GeneratedIdea

            idea = GeneratedIdea(
                content=f"Test logical analysis {i+1}",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name=agent.name,
                generation_prompt="Test prompt",
                confidence_score=0.8,
                reasoning="Test reasoning",
                metadata={"framework": framework},
                timestamp=datetime.now().isoformat(),
            )
            analyses.append(idea)

        # Mock failed ranking (should trigger fallback)
        mock_llm_manager = AsyncMock()
        mock_llm_manager.generate.side_effect = Exception("Ranking failed")
        agent.llm_manager = mock_llm_manager

        selected = await agent._rank_and_select_analyses(
            analyses, 2, "test problem", {"logical_complexity": "moderate"}
        )

        # Should return 2 analyses using fallback framework diversity
        assert len(selected) == 2
        frameworks = [a.metadata["framework"] for a in selected]
        assert len(set(frameworks)) == 2  # Different frameworks


# Fixtures for testing
@pytest.fixture
def sample_deductive_agent():
    """Sample LLM deductive agent for testing."""
    return LLMDeductiveAgent(name="TestDeductiveAgent")


@pytest.fixture
def sample_logical_analysis():
    """Sample logical analysis for testing."""
    return {
        "logical_complexity": "complex",
        "problem_type": "well_defined",
        "evidence_base": "strong",
        "formal_logic_applicable": True,
        "reasoning_chain_depth": "deep",
        "systematic_analysis_scope": ["requirements", "validation", "consequences"],
    }
