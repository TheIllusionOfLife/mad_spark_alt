"""
Tests for circuit breaker and timeout functionality in SmartQADIOrchestrator.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import ThinkingMethod
from mad_spark_alt.core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
)
from mad_spark_alt.core.smart_orchestrator import (
    AgentCircuitBreaker,
    SmartQADIOrchestrator,
)
from mad_spark_alt.core.smart_registry import SmartAgentRegistry


class TestAgentCircuitBreaker:
    """Test cases for the AgentCircuitBreaker class."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        breaker = AgentCircuitBreaker()
        assert breaker.consecutive_failures == 0
        assert breaker.last_failure_time is None
        assert breaker.is_open is False
        assert breaker.can_attempt() is True

    def test_record_failure_increments_counter(self):
        """Test that recording failures increments the counter."""
        breaker = AgentCircuitBreaker()
        
        breaker.record_failure()
        assert breaker.consecutive_failures == 1
        assert breaker.last_failure_time is not None
        assert breaker.is_open is False  # Not open yet
        assert breaker.can_attempt() is True

    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        breaker = AgentCircuitBreaker(failure_threshold=3)
        
        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure()
            if i < 2:
                assert breaker.is_open is False
            else:
                assert breaker.is_open is True
        
        assert breaker.consecutive_failures == 3
        assert breaker.can_attempt() is False

    def test_record_success_resets_circuit(self):
        """Test that recording success resets the circuit."""
        breaker = AgentCircuitBreaker()
        
        # Create some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.consecutive_failures == 2
        
        # Success should reset
        breaker.record_success()
        assert breaker.consecutive_failures == 0
        assert breaker.is_open is False
        assert breaker.last_failure_time is None

    def test_half_open_state_after_cooldown(self):
        """Test half-open state after cooldown period."""
        breaker = AgentCircuitBreaker(failure_threshold=3, cooldown_seconds=0.1)
        
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.is_open is True
        assert breaker.can_attempt() is False
        
        # Wait for cooldown
        time.sleep(0.2)
        
        # Should be in half-open state (can attempt once)
        assert breaker.is_open is True  # Still technically open
        assert breaker.can_attempt() is True  # But can attempt

    def test_cooldown_period_respected(self):
        """Test that cooldown period is properly respected."""
        breaker = AgentCircuitBreaker(failure_threshold=3, cooldown_seconds=0.5)
        
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()
        
        # Immediately after opening
        assert breaker.can_attempt() is False
        
        # Wait less than cooldown
        time.sleep(0.2)
        assert breaker.can_attempt() is False
        
        # Wait until after cooldown
        time.sleep(0.4)
        assert breaker.can_attempt() is True

    def test_custom_threshold_and_cooldown(self):
        """Test custom failure threshold and cooldown configuration."""
        breaker = AgentCircuitBreaker(failure_threshold=5, cooldown_seconds=1.0)
        
        # Should not open until 5 failures
        for i in range(4):
            breaker.record_failure()
            assert breaker.is_open is False
        
        # 5th failure opens it
        breaker.record_failure()
        assert breaker.is_open is True
        assert breaker.failure_threshold == 5
        assert breaker.cooldown_seconds == 1.0


class TestSmartQADIOrchestrator:
    """Test cases for timeout and circuit breaker functionality in SmartQADIOrchestrator."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock smart agent registry."""
        registry = MagicMock(spec=SmartAgentRegistry)
        registry.setup_intelligent_agents = AsyncMock(return_value={
            "questioning": "LLM agent ready",
            "abduction": "LLM agent ready",
            "deduction": "LLM agent ready",
            "induction": "LLM agent ready",
        })
        return registry

    @pytest.fixture
    def orchestrator(self, mock_registry):
        """Create orchestrator with mock registry and short timeouts for testing."""
        return SmartQADIOrchestrator(
            registry=mock_registry,
            phase_timeout=0.5,  # 500ms for testing
            parallel_timeout=1.0,  # 1s for testing
            conclusion_timeout=0.5,  # 500ms for testing
        )

    @pytest.mark.asyncio
    async def test_phase_timeout_handling(self, orchestrator, mock_registry):
        """Test that phase timeouts are properly handled."""
        # Create a slow agent that will timeout
        slow_agent = MagicMock()
        slow_agent.is_llm_powered = True
        slow_agent.name = "SlowAgent"
        
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than phase timeout
            return IdeaGenerationResult(
                agent_name="SlowAgent",
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[],
            )
        
        slow_agent.generate_ideas = slow_generate
        mock_registry.get_preferred_agent.return_value = slow_agent
        
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle("Test problem")
        
        # Check timeout was recorded
        assert "questioning" in result.timeout_info
        assert result.timeout_info["questioning"] == 0.5
        
        # Check phase result shows timeout
        phase_result = result.phases.get("questioning")
        assert phase_result is not None
        assert phase_result.agent_name == "timeout"
        assert "timed out" in phase_result.error_message

    @pytest.mark.asyncio
    async def test_parallel_execution_timeout(self, orchestrator, mock_registry):
        """Test timeout handling in parallel execution."""
        # Create agents with different speeds
        fast_agent = MagicMock()
        fast_agent.is_llm_powered = True
        fast_agent.name = "FastAgent"
        
        async def fast_generate(*args, **kwargs):
            await asyncio.sleep(0.1)
            return IdeaGenerationResult(
                agent_name="FastAgent",
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[GeneratedIdea(
                    content="Fast idea",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="FastAgent",
                    generation_prompt="Test prompt",
                )],
            )
        
        slow_agent = MagicMock()
        slow_agent.is_llm_powered = True
        slow_agent.name = "SlowAgent"
        
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than parallel timeout
            return IdeaGenerationResult(
                agent_name="SlowAgent",
                thinking_method=ThinkingMethod.ABDUCTION,
                generated_ideas=[],
            )
        
        fast_agent.generate_ideas = fast_generate
        slow_agent.generate_ideas = slow_generate
        
        # Configure registry to return different agents
        def get_agent(method):
            if method == ThinkingMethod.QUESTIONING:
                return fast_agent
            return slow_agent
        
        mock_registry.get_preferred_agent.side_effect = get_agent
        
        # Run parallel generation
        results = await orchestrator.run_parallel_generation(
            "Test problem",
            [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
        )
        
        # Fast agent should succeed
        assert ThinkingMethod.QUESTIONING in results
        q_result, q_type = results[ThinkingMethod.QUESTIONING]
        # Check that we got results (may be from template fallback)
        assert len(q_result.generated_ideas) > 0
        assert q_type in ["LLM", "template_fallback", "template"]
        
        # Slow agent should timeout or be cancelled
        if ThinkingMethod.ABDUCTION in results:
            a_result, a_type = results[ThinkingMethod.ABDUCTION]
            assert a_result.error_message is not None
            assert a_type in ["timeout", "cancelled", "error", "template_fallback"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, orchestrator, mock_registry):
        """Test circuit breaker integration in orchestrator."""
        # Disable auto setup to prevent template fallback
        orchestrator.auto_setup = False
        orchestrator._setup_completed = True
        
        # Create a failing agent
        failing_agent = MagicMock()
        failing_agent.is_llm_powered = True
        failing_agent.name = "FailingAgent"
        
        async def failing_generate(*args, **kwargs):
            raise Exception("Agent failure")
        
        failing_agent.generate_ideas = failing_generate
        mock_registry.get_preferred_agent.return_value = failing_agent
        
        # Mock _create_template_agent to return None (no fallback)
        orchestrator._create_template_agent = MagicMock(return_value=None)
        
        # Run multiple cycles to trigger circuit breaker
        for i in range(4):
            result = await orchestrator.run_qadi_cycle(f"Test problem {i}")
            
            # Check phases for expected behavior
            for phase_name, phase_result in result.phases.items():
                if i < 3:
                    # First 3 attempts should show errors
                    assert phase_result.error_message is not None
                    # Record failures in circuit breaker
                    method = ThinkingMethod(phase_name)
                    breaker = orchestrator._get_circuit_breaker(method)
                    assert breaker.consecutive_failures == i + 1
                else:
                    # 4th attempt should be blocked by circuit breaker for all methods
                    method = ThinkingMethod(phase_name)
                    breaker = orchestrator._get_circuit_breaker(method)
                    assert breaker.is_open is True
                    if phase_result.agent_name == "circuit_breaker_open":
                        assert "Circuit breaker open" in phase_result.error_message

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, orchestrator, mock_registry):
        """Test circuit breaker recovery after cooldown."""
        # Disable auto setup to prevent template fallback
        orchestrator.auto_setup = False
        orchestrator._setup_completed = True
        orchestrator._create_template_agent = MagicMock(return_value=None)
        
        # Set very short cooldown for testing
        orchestrator._circuit_breakers = {
            "questioning": AgentCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        }
        
        failing_agent = MagicMock()
        failing_agent.is_llm_powered = True
        failing_agent.name = "FailingAgent"
        
        # Agent fails initially, then succeeds
        call_count = 0
        
        async def conditional_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Agent failure")
            return IdeaGenerationResult(
                agent_name="RecoveredAgent",
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[GeneratedIdea(
                    content="Success after recovery",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="RecoveredAgent",
                    generation_prompt="Test prompt",
                )],
            )
        
        failing_agent.generate_ideas = conditional_generate
        mock_registry.get_preferred_agent.return_value = failing_agent
        
        # Trigger circuit breaker with failures
        methods = [ThinkingMethod.QUESTIONING]
        for _ in range(2):
            results = await orchestrator.run_parallel_generation("Test", methods)
            assert ThinkingMethod.QUESTIONING in results
            result, _ = results[ThinkingMethod.QUESTIONING]
            # Should have error (no fallback)
            assert result.error_message is not None
        
        # Circuit should be open now
        breaker = orchestrator._get_circuit_breaker(ThinkingMethod.QUESTIONING)
        assert breaker.is_open is True
        
        # Wait for cooldown
        await asyncio.sleep(0.2)
        
        # Should allow attempt and succeed
        results = await orchestrator.run_parallel_generation("Test", methods)
        assert ThinkingMethod.QUESTIONING in results
        result, agent_type = results[ThinkingMethod.QUESTIONING]
        assert result.error_message is None
        assert len(result.generated_ideas) == 1
        
        # Circuit should be closed after success
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_conclusion_timeout(self, orchestrator, mock_registry):
        """Test timeout handling for conclusion synthesis."""
        # Create successful agents
        agent = MagicMock()
        agent.is_llm_powered = True
        agent.name = "TestAgent"
        agent.generate_ideas = AsyncMock(return_value=IdeaGenerationResult(
            agent_name="TestAgent",
            thinking_method=ThinkingMethod.QUESTIONING,
            generated_ideas=[GeneratedIdea(
                content="Test idea",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            )],
        ))
        mock_registry.get_preferred_agent.return_value = agent
        
        # Mock slow conclusion synthesis
        with patch('mad_spark_alt.core.smart_orchestrator.ConclusionSynthesizer') as mock_synth:
            instance = mock_synth.return_value
            
            async def slow_synthesize(*args, **kwargs):
                await asyncio.sleep(2.0)  # Longer than conclusion timeout
                return MagicMock()
            
            instance.synthesize_conclusion = slow_synthesize
            
            # Run QADI cycle
            result = await orchestrator.run_qadi_cycle("Test problem")
            
            # Check conclusion timeout was recorded
            assert "conclusion" in result.timeout_info
            assert result.timeout_info["conclusion"] == 0.5
            assert result.conclusion is None

    @pytest.mark.asyncio
    async def test_get_agent_status(self, orchestrator):
        """Test getting comprehensive agent status."""
        # Trigger some failures
        breaker = orchestrator._get_circuit_breaker(ThinkingMethod.QUESTIONING)
        breaker.record_failure()
        breaker.record_failure()
        
        status = orchestrator.get_agent_status()
        
        assert "setup_completed" in status
        assert "timeout_config" in status
        assert "circuit_breakers" in status
        
        # Check timeout config
        assert status["timeout_config"]["phase_timeout"] == 0.5
        assert status["timeout_config"]["parallel_timeout"] == 1.0
        assert status["timeout_config"]["conclusion_timeout"] == 0.5
        
        # Check circuit breaker status
        assert "questioning" in status["circuit_breakers"]
        cb_status = status["circuit_breakers"]["questioning"]
        assert cb_status["consecutive_failures"] == 2
        assert cb_status["is_open"] is False
        assert cb_status["can_attempt"] is True

    @pytest.mark.asyncio
    async def test_error_handling_with_partial_results(self, orchestrator, mock_registry):
        """Test that partial results are preserved when some agents fail."""
        # Disable template fallback
        orchestrator._create_template_agent = MagicMock(return_value=None)
        
        # Create mixed success/failure agents
        success_agent = MagicMock()
        success_agent.is_llm_powered = True
        success_agent.name = "SuccessAgent"
        success_agent.generate_ideas = AsyncMock(return_value=IdeaGenerationResult(
            agent_name="SuccessAgent",
            thinking_method=ThinkingMethod.QUESTIONING,
            generated_ideas=[GeneratedIdea(
                content="Successful idea",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="SuccessAgent",
                generation_prompt="Test prompt",
            )],
        ))
        
        failure_agent = MagicMock()
        failure_agent.is_llm_powered = True
        failure_agent.name = "FailureAgent"
        
        async def fail_generate(*args, **kwargs):
            raise Exception("Agent failure")
        
        failure_agent.generate_ideas = fail_generate
        
        # Configure registry
        def get_agent(method):
            if method == ThinkingMethod.QUESTIONING:
                return success_agent
            return failure_agent
        
        mock_registry.get_preferred_agent.side_effect = get_agent
        
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle("Test problem")
        
        # Should have partial results
        assert "questioning" in result.phases
        q_result = result.phases["questioning"]
        assert len(q_result.generated_ideas) == 1
        assert q_result.agent_name == "SuccessAgent"
        
        # Failed phases should have error messages or be from failed agents
        for phase in ["abduction", "deduction", "induction"]:
            if phase in result.phases:
                phase_result = result.phases[phase]
                # Either it failed with error or fallback was attempted
                assert (phase_result.error_message is not None or 
                        phase_result.agent_name in ["FailureAgent", "error"])

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test timeout configuration options."""
        # Test default timeouts
        default_orchestrator = SmartQADIOrchestrator()
        assert default_orchestrator.phase_timeout == 120
        assert default_orchestrator.parallel_timeout == 180
        assert default_orchestrator.conclusion_timeout == 60
        
        # Test custom timeouts
        custom_orchestrator = SmartQADIOrchestrator(
            phase_timeout=300,
            parallel_timeout=600,
            conclusion_timeout=120,
        )
        assert custom_orchestrator.phase_timeout == 300
        assert custom_orchestrator.parallel_timeout == 600
        assert custom_orchestrator.conclusion_timeout == 120