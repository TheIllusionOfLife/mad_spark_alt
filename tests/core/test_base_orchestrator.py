"""
Tests for BaseOrchestrator - Base class for QADI orchestration.

This test suite follows TDD principles: tests are written first,
then the implementation is created to make them pass.
"""

import asyncio
import pytest
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.base_orchestrator import (
    BaseOrchestrator,
    AgentCircuitBreaker,
)
from mad_spark_alt.core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from mad_spark_alt.core.smart_registry import SmartAgentRegistry


class ConcreteTestOrchestrator(BaseOrchestrator):
    """Minimal concrete implementation for testing BaseOrchestrator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_qadi_cycle_called = False

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimal implementation for testing."""
        self.run_qadi_cycle_called = True
        await self.ensure_agents_ready()
        return {
            "problem_statement": problem_statement,
            "context": context,
            "config": cycle_config,
        }


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_registry():
    """Create a mock SmartAgentRegistry."""
    registry = MagicMock(spec=SmartAgentRegistry)
    registry.setup_intelligent_agents = AsyncMock(
        return_value={
            "questioning": "success",
            "abduction": "success",
            "deduction": "success",
            "induction": "success",
        }
    )
    return registry


@pytest.fixture
def mock_agent():
    """Create a mock ThinkingAgentInterface."""
    agent = AsyncMock(spec=ThinkingAgentInterface)
    agent.name = "test_agent"
    agent.thinking_method = ThinkingMethod.ABDUCTION
    agent.is_llm_powered = True
    agent.generate_ideas = AsyncMock(
        return_value=IdeaGenerationResult(
            agent_name="test_agent",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="Test idea",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_agent",
                    generation_prompt="test prompt",
                    confidence_score=0.8,
                )
            ],
            execution_time=0.1,
        )
    )
    return agent


@pytest.fixture
def sample_phase_result():
    """Create a sample IdeaGenerationResult for testing."""
    return IdeaGenerationResult(
        agent_name="test_agent",
        thinking_method=ThinkingMethod.ABDUCTION,
        generated_ideas=[
            GeneratedIdea(
                content="Idea 1",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_agent",
                generation_prompt="test",
                metadata={"llm_cost": 0.001},
            ),
            GeneratedIdea(
                content="Idea 2",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_agent",
                generation_prompt="test",
                metadata={"llm_cost": 0.002},
            ),
        ],
        execution_time=0.5,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestBaseOrchestratorInitialization:
    """Test BaseOrchestrator initialization and configuration."""

    def test_cannot_instantiate_abstract_base(self):
        """BaseOrchestrator is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseOrchestrator()

    def test_concrete_orchestrator_with_defaults(self, mock_registry):
        """Concrete orchestrator can be instantiated with default parameters."""
        orchestrator = ConcreteTestOrchestrator(registry=mock_registry)

        assert orchestrator.registry == mock_registry
        assert orchestrator.auto_setup is True
        assert orchestrator.enable_circuit_breakers is True
        assert orchestrator._setup_completed is False
        assert orchestrator._setup_status == {}
        assert orchestrator._circuit_breakers == {}

    def test_concrete_orchestrator_with_custom_params(self, mock_registry):
        """Concrete orchestrator accepts custom initialization parameters."""
        orchestrator = ConcreteTestOrchestrator(
            registry=mock_registry,
            auto_setup=False,
            enable_circuit_breakers=False,
        )

        assert orchestrator.registry == mock_registry
        assert orchestrator.auto_setup is False
        assert orchestrator.enable_circuit_breakers is False

    def test_orchestrator_uses_global_registry_if_none(self):
        """If no registry provided, orchestrator uses global smart_registry."""
        orchestrator = ConcreteTestOrchestrator()

        # Should have a registry (the global one)
        assert orchestrator.registry is not None

    def test_qadi_sequence_constant_defined(self):
        """QADI_SEQUENCE constant is defined with correct methods."""
        orchestrator = ConcreteTestOrchestrator()

        assert hasattr(orchestrator, "QADI_SEQUENCE")
        assert orchestrator.QADI_SEQUENCE == [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class TestCircuitBreakerFunctionality:
    """Test circuit breaker pattern implementation."""

    def test_circuit_breaker_initialization(self):
        """AgentCircuitBreaker initializes with correct default values."""
        breaker = AgentCircuitBreaker()

        assert breaker.consecutive_failures == 0
        assert breaker.last_failure_time is None
        assert breaker.is_open is False
        assert breaker.failure_threshold == 3
        assert breaker.cooldown_seconds == 300.0

    def test_circuit_breaker_custom_initialization(self):
        """AgentCircuitBreaker accepts custom parameters."""
        breaker = AgentCircuitBreaker(
            failure_threshold=5,
            cooldown_seconds=600.0,
        )

        assert breaker.failure_threshold == 5
        assert breaker.cooldown_seconds == 600.0

    def test_circuit_breaker_record_failure(self):
        """record_failure increments counter and sets timestamp."""
        breaker = AgentCircuitBreaker()
        before_time = time.time()

        breaker.record_failure()

        assert breaker.consecutive_failures == 1
        assert breaker.last_failure_time is not None
        assert breaker.last_failure_time >= before_time
        assert breaker.is_open is False  # Not open yet (threshold is 3)

    def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker opens after reaching failure threshold."""
        breaker = AgentCircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is False

        breaker.record_failure()  # Third failure
        assert breaker.is_open is True

    def test_circuit_breaker_record_success(self):
        """record_success resets circuit breaker state."""
        breaker = AgentCircuitBreaker()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()  # Open the circuit

        assert breaker.is_open is True
        assert breaker.consecutive_failures == 3

        breaker.record_success()

        assert breaker.is_open is False
        assert breaker.consecutive_failures == 0
        assert breaker.last_failure_time is None

    def test_circuit_breaker_can_attempt_when_closed(self):
        """can_attempt returns True when circuit is closed."""
        breaker = AgentCircuitBreaker()

        assert breaker.can_attempt() is True

        breaker.record_failure()
        assert breaker.can_attempt() is True  # Still below threshold

    def test_circuit_breaker_cannot_attempt_when_open(self):
        """can_attempt returns False when circuit is open."""
        breaker = AgentCircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()  # Open the circuit

        assert breaker.is_open is True
        assert breaker.can_attempt() is False

    def test_circuit_breaker_cooldown_period(self):
        """Circuit breaker allows attempt after cooldown period."""
        breaker = AgentCircuitBreaker(
            failure_threshold=2,
            cooldown_seconds=0.1,  # Short cooldown for testing
        )

        breaker.record_failure()
        breaker.record_failure()  # Open the circuit

        assert breaker.can_attempt() is False

        # Wait for cooldown
        time.sleep(0.2)

        # Should allow attempt after cooldown (half-open state)
        assert breaker.can_attempt() is True

    def test_get_circuit_breaker_creates_new(self):
        """_get_circuit_breaker creates new breaker for each method."""
        orchestrator = ConcreteTestOrchestrator()

        breaker1 = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)
        breaker2 = orchestrator._get_circuit_breaker(ThinkingMethod.DEDUCTION)

        assert breaker1 is not breaker2
        assert isinstance(breaker1, AgentCircuitBreaker)
        assert isinstance(breaker2, AgentCircuitBreaker)

    def test_get_circuit_breaker_returns_same_instance(self):
        """_get_circuit_breaker returns same instance for same method."""
        orchestrator = ConcreteTestOrchestrator()

        breaker1 = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)
        breaker2 = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)

        assert breaker1 is breaker2

    def test_can_use_agent_checks_circuit_breaker(self):
        """_can_use_agent returns circuit breaker state."""
        orchestrator = ConcreteTestOrchestrator()

        # Initially should be able to use agent
        assert orchestrator._can_use_agent(ThinkingMethod.ABDUCTION) is True

        # Record failures to open circuit
        breaker = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        # Should not be able to use agent now
        assert orchestrator._can_use_agent(ThinkingMethod.ABDUCTION) is False

    def test_record_agent_success_delegates_to_breaker(self):
        """_record_agent_success updates circuit breaker correctly."""
        orchestrator = ConcreteTestOrchestrator()

        breaker = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()  # Open circuit

        assert breaker.is_open is True

        orchestrator._record_agent_success(ThinkingMethod.ABDUCTION)

        assert breaker.is_open is False
        assert breaker.consecutive_failures == 0

    def test_record_agent_failure_delegates_to_breaker(self):
        """_record_agent_failure updates circuit breaker correctly."""
        orchestrator = ConcreteTestOrchestrator()

        breaker = orchestrator._get_circuit_breaker(ThinkingMethod.ABDUCTION)

        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)

        assert breaker.consecutive_failures == 1

        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)
        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)

        assert breaker.is_open is True


# ============================================================================
# Agent Management Tests
# ============================================================================


class TestAgentManagement:
    """Test agent management and setup functionality."""

    @pytest.mark.asyncio
    async def test_ensure_agents_ready_with_auto_setup(self, mock_registry):
        """ensure_agents_ready calls setup when auto_setup is True."""
        orchestrator = ConcreteTestOrchestrator(
            registry=mock_registry,
            auto_setup=True,
        )

        status = await orchestrator.ensure_agents_ready()

        assert orchestrator._setup_completed is True
        assert status == {
            "questioning": "success",
            "abduction": "success",
            "deduction": "success",
            "induction": "success",
        }
        mock_registry.setup_intelligent_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_agents_ready_without_auto_setup(self, mock_registry):
        """ensure_agents_ready does nothing when auto_setup is False."""
        orchestrator = ConcreteTestOrchestrator(
            registry=mock_registry,
            auto_setup=False,
        )

        status = await orchestrator.ensure_agents_ready()

        assert orchestrator._setup_completed is False
        assert status == {}
        mock_registry.setup_intelligent_agents.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_agents_ready_only_runs_once(self, mock_registry):
        """ensure_agents_ready only runs setup once."""
        orchestrator = ConcreteTestOrchestrator(
            registry=mock_registry,
            auto_setup=True,
        )

        await orchestrator.ensure_agents_ready()
        await orchestrator.ensure_agents_ready()
        await orchestrator.ensure_agents_ready()

        # Should only be called once
        mock_registry.setup_intelligent_agents.assert_called_once()

    def test_create_template_agent_for_questioning(self):
        """_create_template_agent creates QuestioningAgent."""
        orchestrator = ConcreteTestOrchestrator()

        agent = orchestrator._create_template_agent(ThinkingMethod.QUESTIONING)

        assert agent is not None
        assert agent.thinking_method == ThinkingMethod.QUESTIONING

    def test_create_template_agent_for_abduction(self):
        """_create_template_agent creates AbductionAgent."""
        orchestrator = ConcreteTestOrchestrator()

        agent = orchestrator._create_template_agent(ThinkingMethod.ABDUCTION)

        assert agent is not None
        assert agent.thinking_method == ThinkingMethod.ABDUCTION

    def test_create_template_agent_for_deduction(self):
        """_create_template_agent creates DeductionAgent."""
        orchestrator = ConcreteTestOrchestrator()

        agent = orchestrator._create_template_agent(ThinkingMethod.DEDUCTION)

        assert agent is not None
        assert agent.thinking_method == ThinkingMethod.DEDUCTION

    def test_create_template_agent_for_induction(self):
        """_create_template_agent creates InductionAgent."""
        orchestrator = ConcreteTestOrchestrator()

        agent = orchestrator._create_template_agent(ThinkingMethod.INDUCTION)

        assert agent is not None
        assert agent.thinking_method == ThinkingMethod.INDUCTION

    def test_create_template_agent_for_unknown_method(self):
        """_create_template_agent returns None for unknown method."""
        orchestrator = ConcreteTestOrchestrator()

        agent = orchestrator._create_template_agent(ThinkingMethod.QADI_CYCLE)

        assert agent is None


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Test helper methods for context building and synthesis."""

    def test_build_enhanced_context_with_no_inputs(self):
        """_build_enhanced_context returns empty string with no inputs."""
        orchestrator = ConcreteTestOrchestrator()

        context = orchestrator._build_enhanced_context(None)

        assert context == ""

    def test_build_enhanced_context_with_base_context_only(self):
        """_build_enhanced_context includes base context."""
        orchestrator = ConcreteTestOrchestrator()

        context = orchestrator._build_enhanced_context("Initial context here")

        assert "Initial context: Initial context here" in context

    def test_build_enhanced_context_with_phase_results(self, sample_phase_result):
        """_build_enhanced_context includes phase result ideas."""
        orchestrator = ConcreteTestOrchestrator()

        context = orchestrator._build_enhanced_context(None, sample_phase_result)

        assert "Abduction phase insights:" in context
        assert "- Idea 1" in context
        assert "- Idea 2" in context

    def test_build_enhanced_context_with_multiple_phases(self):
        """_build_enhanced_context combines multiple phase results."""
        orchestrator = ConcreteTestOrchestrator()

        result1 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.QUESTIONING,
            generated_ideas=[
                GeneratedIdea(
                    content="Question idea",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="test",
                    generation_prompt="test",
                )
            ],
        )

        result2 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="Hypothesis idea",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                )
            ],
        )

        context = orchestrator._build_enhanced_context(
            "Base context", result1, result2
        )

        assert "Initial context: Base context" in context
        assert "Questioning phase insights:" in context
        assert "- Question idea" in context
        assert "Abduction phase insights:" in context
        assert "- Hypothesis idea" in context

    def test_build_enhanced_context_skips_empty_results(self):
        """_build_enhanced_context skips results with no ideas."""
        orchestrator = ConcreteTestOrchestrator()

        empty_result = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[],
        )

        context = orchestrator._build_enhanced_context("Base", empty_result)

        assert context == "Initial context: Base"
        assert "Abduction" not in context

    def test_collect_and_tag_ideas_with_empty_phases(self):
        """_collect_and_tag_ideas returns empty list for empty phases."""
        orchestrator = ConcreteTestOrchestrator()

        ideas = orchestrator._collect_and_tag_ideas({})

        assert ideas == []

    def test_collect_and_tag_ideas_from_single_phase(self, sample_phase_result):
        """_collect_and_tag_ideas extracts ideas from single phase."""
        orchestrator = ConcreteTestOrchestrator()

        phases = {"abduction": sample_phase_result}
        ideas = orchestrator._collect_and_tag_ideas(phases)

        assert len(ideas) == 2
        assert ideas[0].content == "Idea 1"
        assert ideas[1].content == "Idea 2"
        # Check phase metadata is added
        assert ideas[0].metadata["phase"] == "abduction"
        assert ideas[1].metadata["phase"] == "abduction"

    def test_collect_and_tag_ideas_from_multiple_phases(self):
        """_collect_and_tag_ideas combines ideas from multiple phases."""
        orchestrator = ConcreteTestOrchestrator()

        result1 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.QUESTIONING,
            generated_ideas=[
                GeneratedIdea(
                    content="Q1",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="test",
                    generation_prompt="test",
                )
            ],
        )

        result2 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="A1",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                ),
                GeneratedIdea(
                    content="A2",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                ),
            ],
        )

        phases = {"questioning": result1, "abduction": result2}
        ideas = orchestrator._collect_and_tag_ideas(phases)

        assert len(ideas) == 3
        assert [idea.content for idea in ideas] == ["Q1", "A1", "A2"]

    def test_collect_and_tag_ideas_skips_none_results(self):
        """_collect_and_tag_ideas handles None phase results."""
        orchestrator = ConcreteTestOrchestrator()

        result1 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="A1",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                )
            ],
        )

        phases = {"questioning": None, "abduction": result1, "deduction": None}
        ideas = orchestrator._collect_and_tag_ideas(phases)

        assert len(ideas) == 1
        assert ideas[0].content == "A1"

    def test_extract_llm_cost_from_empty_result(self):
        """_extract_llm_cost returns 0.0 for empty result."""
        orchestrator = ConcreteTestOrchestrator()

        empty_result = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[],
        )

        cost = orchestrator._extract_llm_cost(empty_result)

        assert cost == 0.0

    def test_extract_llm_cost_from_result(self, sample_phase_result):
        """_extract_llm_cost sums costs from all ideas."""
        orchestrator = ConcreteTestOrchestrator()

        cost = orchestrator._extract_llm_cost(sample_phase_result)

        # 0.001 + 0.002 = 0.003
        assert cost == pytest.approx(0.003)

    def test_extract_llm_cost_handles_missing_cost(self):
        """_extract_llm_cost handles ideas without cost metadata."""
        orchestrator = ConcreteTestOrchestrator()

        result = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="Idea without cost",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                    metadata={},  # No llm_cost
                )
            ],
        )

        cost = orchestrator._extract_llm_cost(result)

        assert cost == 0.0


# ============================================================================
# Error Result Factory Tests
# ============================================================================


class TestErrorResultFactories:
    """Test error result factory methods."""

    def test_create_timeout_result(self):
        """_create_timeout_result creates proper timeout result."""
        orchestrator = ConcreteTestOrchestrator()

        result = orchestrator._create_timeout_result(ThinkingMethod.ABDUCTION)

        assert result.agent_name == "timeout"
        assert result.thinking_method == ThinkingMethod.ABDUCTION
        assert result.generated_ideas == []
        assert result.execution_time == 0.0
        assert result.error_message == "Phase skipped due to timeout"
        assert result.generation_metadata.get("timeout") is True

    def test_create_error_result(self):
        """_create_error_result creates proper error result."""
        orchestrator = ConcreteTestOrchestrator()

        result = orchestrator._create_error_result(
            ThinkingMethod.DEDUCTION, "Test error message"
        )

        assert result.agent_name == "error"
        assert result.thinking_method == ThinkingMethod.DEDUCTION
        assert result.generated_ideas == []
        assert result.execution_time == 0.0
        assert result.error_message == "Test error message"
        assert result.generation_metadata.get("error") is True

    def test_create_empty_result(self):
        """_create_empty_result creates proper empty result."""
        orchestrator = ConcreteTestOrchestrator()

        result = orchestrator._create_empty_result(ThinkingMethod.INDUCTION)

        assert result.agent_name == "empty"
        assert result.thinking_method == ThinkingMethod.INDUCTION
        assert result.generated_ideas == []
        assert result.execution_time == 0.0
        assert result.error_message is None
        assert result.generation_metadata.get("empty") is True


# ============================================================================
# Abstract Method Tests
# ============================================================================


class TestAbstractMethods:
    """Test abstract method enforcement."""

    def test_run_qadi_cycle_must_be_implemented(self):
        """run_qadi_cycle must be implemented by subclasses."""
        # This is tested by the fact that we can't instantiate BaseOrchestrator
        # and ConcreteTestOrchestrator provides the implementation
        orchestrator = ConcreteTestOrchestrator()

        assert hasattr(orchestrator, "run_qadi_cycle")
        assert callable(orchestrator.run_qadi_cycle)

    @pytest.mark.asyncio
    async def test_concrete_orchestrator_run_qadi_cycle_works(self):
        """Concrete implementation of run_qadi_cycle works."""
        orchestrator = ConcreteTestOrchestrator()

        result = await orchestrator.run_qadi_cycle("Test problem")

        assert orchestrator.run_qadi_cycle_called is True
        assert result["problem_statement"] == "Test problem"


# ============================================================================
# Optional Hook Tests
# ============================================================================


class TestOptionalHooks:
    """Test optional hook methods."""

    @pytest.mark.asyncio
    async def test_initialize_cycle_hook_exists(self):
        """_initialize_cycle hook exists and can be called."""
        orchestrator = ConcreteTestOrchestrator()

        # Should not raise error
        await orchestrator._initialize_cycle("Test problem")

    @pytest.mark.asyncio
    async def test_finalize_cycle_hook_exists(self):
        """_finalize_cycle hook exists and can be called."""
        orchestrator = ConcreteTestOrchestrator()

        # Should not raise error
        await orchestrator._finalize_cycle({"test": "result"})

    @pytest.mark.asyncio
    async def test_hooks_can_be_overridden(self):
        """Optional hooks can be overridden in subclasses."""

        class CustomOrchestrator(ConcreteTestOrchestrator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.initialize_called = False
                self.finalize_called = False

            async def _initialize_cycle(self, problem_statement: str) -> None:
                self.initialize_called = True

            async def _finalize_cycle(self, result: Any) -> None:
                self.finalize_called = True

        orchestrator = CustomOrchestrator()

        await orchestrator._initialize_cycle("Test")
        await orchestrator._finalize_cycle({"test": "result"})

        assert orchestrator.initialize_called is True
        assert orchestrator.finalize_called is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationScenarios:
    """Test integrated usage scenarios."""

    @pytest.mark.asyncio
    async def test_full_orchestrator_lifecycle(self, mock_registry):
        """Test complete orchestrator lifecycle with all methods."""
        orchestrator = ConcreteTestOrchestrator(
            registry=mock_registry,
            auto_setup=True,
        )

        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            "How can we improve system performance?",
            context="Current system is slow",
        )

        # Verify setup was called
        assert orchestrator._setup_completed is True

        # Verify result
        assert result["problem_statement"] == "How can we improve system performance?"
        assert result["context"] == "Current system is slow"

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with agent management."""
        orchestrator = ConcreteTestOrchestrator(enable_circuit_breakers=True)

        # Initially can use agent
        assert orchestrator._can_use_agent(ThinkingMethod.ABDUCTION) is True

        # Record failures
        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)
        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)
        orchestrator._record_agent_failure(ThinkingMethod.ABDUCTION)

        # Circuit should be open
        assert orchestrator._can_use_agent(ThinkingMethod.ABDUCTION) is False

        # Record success to reset
        orchestrator._record_agent_success(ThinkingMethod.ABDUCTION)

        # Can use agent again
        assert orchestrator._can_use_agent(ThinkingMethod.ABDUCTION) is True

    def test_helper_methods_work_together(self):
        """Test that helper methods work together correctly."""
        orchestrator = ConcreteTestOrchestrator()

        # Create phase results
        result1 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.QUESTIONING,
            generated_ideas=[
                GeneratedIdea(
                    content="What is the problem?",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="test",
                    generation_prompt="test",
                    metadata={"llm_cost": 0.01},
                )
            ],
        )

        result2 = IdeaGenerationResult(
            agent_name="test",
            thinking_method=ThinkingMethod.ABDUCTION,
            generated_ideas=[
                GeneratedIdea(
                    content="Hypothesis 1",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                    metadata={"llm_cost": 0.02},
                ),
                GeneratedIdea(
                    content="Hypothesis 2",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test",
                    metadata={"llm_cost": 0.03},
                ),
            ],
        )

        # Build context
        context = orchestrator._build_enhanced_context("Initial", result1, result2)
        assert "What is the problem?" in context
        assert "Hypothesis 1" in context

        # Collect and tag ideas
        phases = {"questioning": result1, "abduction": result2}
        ideas = orchestrator._collect_and_tag_ideas(phases)
        assert len(ideas) == 3

        # Extract costs
        cost1 = orchestrator._extract_llm_cost(result1)
        cost2 = orchestrator._extract_llm_cost(result2)
        assert cost1 == pytest.approx(0.01)
        assert cost2 == pytest.approx(0.05)  # 0.02 + 0.03
