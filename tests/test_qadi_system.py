"""
Basic tests for the QADI system and thinking agents.
"""

import asyncio
import pytest

from mad_spark_alt.core import (
    ThinkingMethod,
    IdeaGenerationRequest,
    QADIOrchestrator,
    register_agent,
    agent_registry,
)
from mad_spark_alt.agents import (
    QuestioningAgent,
    AbductionAgent,
    DeductionAgent,
    InductionAgent,
)


class TestQADISystem:
    """Test the QADI orchestration system."""

    def setup_method(self):
        """Set up test environment."""
        # Clear registry before each test
        agent_registry.clear()
        
        # Register all agents
        register_agent(QuestioningAgent)
        register_agent(AbductionAgent)
        register_agent(DeductionAgent)
        register_agent(InductionAgent)

    def teardown_method(self):
        """Clean up after each test."""
        agent_registry.clear()

    def test_agent_registration(self):
        """Test that agents are properly registered."""
        agents = agent_registry.list_agents()
        
        assert "QuestioningAgent" in agents
        assert "AbductionAgent" in agents
        assert "DeductionAgent" in agents
        assert "InductionAgent" in agents
        
        # Verify thinking methods
        assert agents["QuestioningAgent"]["thinking_method"] == "questioning"
        assert agents["AbductionAgent"]["thinking_method"] == "abduction"
        assert agents["DeductionAgent"]["thinking_method"] == "deduction"
        assert agents["InductionAgent"]["thinking_method"] == "induction"

    def test_agent_retrieval_by_method(self):
        """Test retrieving agents by thinking method."""
        questioning_agent = agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING)
        abduction_agent = agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION)
        deduction_agent = agent_registry.get_agent_by_method(ThinkingMethod.DEDUCTION)
        induction_agent = agent_registry.get_agent_by_method(ThinkingMethod.INDUCTION)
        
        assert questioning_agent is not None
        assert abduction_agent is not None
        assert deduction_agent is not None
        assert induction_agent is not None
        
        assert questioning_agent.thinking_method == ThinkingMethod.QUESTIONING
        assert abduction_agent.thinking_method == ThinkingMethod.ABDUCTION
        assert deduction_agent.thinking_method == ThinkingMethod.DEDUCTION
        assert induction_agent.thinking_method == ThinkingMethod.INDUCTION

    @pytest.mark.asyncio
    async def test_individual_agent_generation(self):
        """Test individual agent idea generation."""
        questioning_agent = agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING)
        
        request = IdeaGenerationRequest(
            problem_statement="How can we reduce plastic waste in urban environments?",
            context="Considering both environmental and economic factors",
            max_ideas_per_method=3,
            require_reasoning=True
        )
        
        result = await questioning_agent.generate_ideas(request)
        
        assert result.agent_name == "QuestioningAgent"
        assert result.thinking_method == ThinkingMethod.QUESTIONING
        assert len(result.generated_ideas) > 0
        assert len(result.generated_ideas) <= 3
        assert result.error_message is None
        
        # Check idea structure
        for idea in result.generated_ideas:
            assert idea.content is not None
            assert idea.thinking_method == ThinkingMethod.QUESTIONING
            assert idea.agent_name == "QuestioningAgent"
            assert idea.reasoning is not None  # require_reasoning=True

    @pytest.mark.asyncio
    async def test_qadi_orchestrator_initialization(self):
        """Test QADI orchestrator initialization."""
        # Get agents from registry
        agents = [
            agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING),
            agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION),
            agent_registry.get_agent_by_method(ThinkingMethod.DEDUCTION),
            agent_registry.get_agent_by_method(ThinkingMethod.INDUCTION),
        ]
        
        orchestrator = QADIOrchestrator(agents)
        
        # Verify all methods are available
        assert orchestrator.has_agent(ThinkingMethod.QUESTIONING)
        assert orchestrator.has_agent(ThinkingMethod.ABDUCTION)
        assert orchestrator.has_agent(ThinkingMethod.DEDUCTION)
        assert orchestrator.has_agent(ThinkingMethod.INDUCTION)

    @pytest.mark.asyncio
    async def test_qadi_cycle_execution(self):
        """Test complete QADI cycle execution."""
        # Get agents from registry
        agents = [
            agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING),
            agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION),
            agent_registry.get_agent_by_method(ThinkingMethod.DEDUCTION),
            agent_registry.get_agent_by_method(ThinkingMethod.INDUCTION),
        ]
        
        orchestrator = QADIOrchestrator(agents)
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement="How can we make cities more sustainable?",
            context="Focus on practical, implementable solutions",
            cycle_config={
                "max_ideas_per_method": 2,
                "require_reasoning": True
            }
        )
        
        # Verify cycle result structure
        assert result.problem_statement == "How can we make cities more sustainable?"
        assert result.cycle_id is not None
        assert result.execution_time is not None
        assert result.execution_time > 0
        
        # Verify all phases were executed
        assert "questioning" in result.phases
        assert "abduction" in result.phases
        assert "deduction" in result.phases
        assert "induction" in result.phases
        
        # Verify each phase has results
        for phase_name, phase_result in result.phases.items():
            assert phase_result.agent_name is not None
            assert len(phase_result.generated_ideas) > 0
            assert phase_result.error_message is None
        
        # Verify synthesized ideas
        assert len(result.synthesized_ideas) > 0
        
        # Check that synthesized ideas have phase metadata
        for idea in result.synthesized_ideas:
            assert "phase" in idea.metadata

    @pytest.mark.asyncio
    async def test_parallel_generation(self):
        """Test parallel generation across multiple thinking methods."""
        agents = [
            agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING),
            agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION),
        ]
        
        orchestrator = QADIOrchestrator(agents)
        
        results = await orchestrator.run_parallel_generation(
            problem_statement="How to improve remote work productivity?",
            thinking_methods=[ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
            context="Post-pandemic work environment",
            config={"max_ideas_per_method": 2}
        )
        
        assert ThinkingMethod.QUESTIONING in results
        assert ThinkingMethod.ABDUCTION in results
        
        for method, result in results.items():
            assert result.thinking_method == method
            assert len(result.generated_ideas) > 0
            assert result.error_message is None

    def test_idea_generation_request_validation(self):
        """Test idea generation request structure."""
        request = IdeaGenerationRequest(
            problem_statement="Test problem",
            context="Test context",
            target_thinking_methods=[ThinkingMethod.QUESTIONING],
            max_ideas_per_method=5,
            require_reasoning=False
        )
        
        assert request.problem_statement == "Test problem"
        assert request.context == "Test context"
        assert ThinkingMethod.QUESTIONING in request.target_thinking_methods
        assert request.max_ideas_per_method == 5
        assert request.require_reasoning is False

    @pytest.mark.asyncio
    async def test_error_handling_missing_agent(self):
        """Test error handling when agent is missing."""
        # Create orchestrator without registering agents
        orchestrator = QADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement="Test problem",
            context="Test context"
        )
        
        # Should complete without crashing
        assert result.problem_statement == "Test problem"
        assert result.execution_time is not None
        
        # All phases should have error messages
        for phase_name, phase_result in result.phases.items():
            assert "No agent available" in phase_result.error_message