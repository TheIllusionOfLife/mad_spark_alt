"""
Integration tests for BaseOrchestrator with real orchestrator implementations.

These tests verify that BaseOrchestrator integrates correctly with actual
orchestrator implementations and can be used as a base class in practice.
"""

import pytest
from typing import Any, Dict, Optional

from mad_spark_alt.core import BaseOrchestrator
from mad_spark_alt.core.interfaces import (
    IdeaGenerationResult,
    GeneratedIdea,
    ThinkingMethod,
)
from mad_spark_alt.core.smart_registry import smart_registry


class MinimalRealOrchestrator(BaseOrchestrator):
    """
    Minimal real orchestrator demonstrating BaseOrchestrator usage.

    This orchestrator shows how to:
    - Leverage base class initialization
    - Use circuit breaker pattern
    - Utilize helper methods
    - Handle agent setup
    - Create error results
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_log = []

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simple QADI cycle implementation using base class methods.
        """
        self.execution_log.append("cycle_started")

        # Use base class agent setup
        await self.ensure_agents_ready()
        self.execution_log.append("agents_ready")

        phases: Dict[str, IdeaGenerationResult] = {}

        # Simulate QADI phases
        for method in self.QADI_SEQUENCE:
            self.execution_log.append(f"phase_{method.value}_started")

            # Check circuit breaker
            if not self._can_use_agent(method):
                self.execution_log.append(f"phase_{method.value}_circuit_open")
                phases[method.value] = self._create_timeout_result(method)
                continue

            # Simulate phase execution
            try:
                result = await self._simulate_phase(method, problem_statement)
                phases[method.value] = result
                self._record_agent_success(method)
                self.execution_log.append(f"phase_{method.value}_success")
            except Exception as e:
                self._record_agent_failure(method)
                self.execution_log.append(f"phase_{method.value}_failed")
                phases[method.value] = self._create_error_result(method, str(e))

        # Use base class helper methods
        enhanced_context = self._build_enhanced_context(context, *phases.values())
        all_ideas = self._synthesize_ideas(phases)
        total_cost = sum(self._extract_llm_cost(r) for r in phases.values())

        self.execution_log.append("cycle_completed")

        return {
            "problem_statement": problem_statement,
            "phases": phases,
            "enhanced_context": enhanced_context,
            "all_ideas": all_ideas,
            "total_cost": total_cost,
            "execution_log": self.execution_log,
        }

    async def _simulate_phase(
        self, method: ThinkingMethod, problem_statement: str
    ) -> IdeaGenerationResult:
        """Simulate a phase execution."""
        return IdeaGenerationResult(
            agent_name="simulated",
            thinking_method=method,
            generated_ideas=[
                GeneratedIdea(
                    content=f"Idea from {method.value} phase about: {problem_statement[:50]}",
                    thinking_method=method,
                    agent_name="simulated",
                    generation_prompt="test",
                    confidence_score=0.7,
                    metadata={"llm_cost": 0.001},
                )
            ],
            execution_time=0.1,
        )


class TestBaseOrchestratorIntegration:
    """Integration tests for BaseOrchestrator usage patterns."""

    @pytest.mark.asyncio
    async def test_minimal_orchestrator_uses_base_infrastructure(self):
        """
        Verify that a minimal orchestrator can leverage all base infrastructure.
        """
        orchestrator = MinimalRealOrchestrator(auto_setup=False)

        result = await orchestrator.run_qadi_cycle("Test problem")

        # Verify base class methods were used
        assert "cycle_started" in orchestrator.execution_log
        assert "cycle_completed" in orchestrator.execution_log

        # Verify all 4 QADI phases were executed
        assert "phase_questioning_started" in orchestrator.execution_log
        assert "phase_abduction_started" in orchestrator.execution_log
        assert "phase_deduction_started" in orchestrator.execution_log
        assert "phase_induction_started" in orchestrator.execution_log

        # Verify phases succeeded
        assert "phase_questioning_success" in orchestrator.execution_log
        assert "phase_abduction_success" in orchestrator.execution_log
        assert "phase_deduction_success" in orchestrator.execution_log
        assert "phase_induction_success" in orchestrator.execution_log

        # Verify result structure
        assert result["problem_statement"] == "Test problem"
        assert len(result["phases"]) == 4
        assert len(result["all_ideas"]) == 4  # One from each phase
        assert result["total_cost"] == pytest.approx(0.004)  # 0.001 * 4

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_failing_agent(self):
        """
        Verify circuit breaker pattern works in real orchestrator.
        """

        class FailingOrchestrator(MinimalRealOrchestrator):
            async def _simulate_phase(self, method, problem_statement):
                # Force failures for abduction phase
                if method == ThinkingMethod.ABDUCTION:
                    raise Exception("Simulated failure")
                return await super()._simulate_phase(method, problem_statement)

        orchestrator = FailingOrchestrator(auto_setup=False)

        # First run - abduction will fail
        result1 = await orchestrator.run_qadi_cycle("Test 1")
        assert "phase_abduction_failed" in orchestrator.execution_log

        # Second and third runs - more failures
        orchestrator.execution_log.clear()
        await orchestrator.run_qadi_cycle("Test 2")
        orchestrator.execution_log.clear()
        await orchestrator.run_qadi_cycle("Test 3")

        # Fourth run - circuit should be open
        orchestrator.execution_log.clear()
        result4 = await orchestrator.run_qadi_cycle("Test 4")

        # Verify circuit breaker opened
        assert "phase_abduction_circuit_open" in orchestrator.execution_log
        assert result4["phases"]["abduction"].error_message == "Phase skipped due to timeout"

    @pytest.mark.asyncio
    async def test_context_building_across_phases(self):
        """
        Verify enhanced context building with multiple phases.
        """
        orchestrator = MinimalRealOrchestrator(auto_setup=False)

        result = await orchestrator.run_qadi_cycle(
            "How to improve software quality?",
            context="Current codebase has high technical debt",
        )

        enhanced_context = result["enhanced_context"]

        # Verify initial context included
        assert "Initial context: Current codebase has high technical debt" in enhanced_context

        # Verify phase insights included
        assert "Questioning phase insights:" in enhanced_context
        assert "Abduction phase insights:" in enhanced_context
        assert "Deduction phase insights:" in enhanced_context
        assert "Induction phase insights:" in enhanced_context

    @pytest.mark.asyncio
    async def test_idea_synthesis_preserves_phase_info(self):
        """
        Verify idea synthesis adds phase metadata.
        """
        orchestrator = MinimalRealOrchestrator(auto_setup=False)

        result = await orchestrator.run_qadi_cycle("Test problem")

        all_ideas = result["all_ideas"]

        # Verify each idea has phase metadata
        for idea in all_ideas:
            assert "phase" in idea.metadata
            assert idea.metadata["phase"] in [
                "questioning",
                "abduction",
                "deduction",
                "induction",
            ]

    @pytest.mark.asyncio
    async def test_cost_tracking_aggregation(self):
        """
        Verify cost extraction and aggregation works.
        """

        class VaryingCostOrchestrator(MinimalRealOrchestrator):
            async def _simulate_phase(self, method, problem_statement):
                # Different costs for different phases
                costs = {
                    ThinkingMethod.QUESTIONING: 0.001,
                    ThinkingMethod.ABDUCTION: 0.005,
                    ThinkingMethod.DEDUCTION: 0.003,
                    ThinkingMethod.INDUCTION: 0.002,
                }

                return IdeaGenerationResult(
                    agent_name="simulated",
                    thinking_method=method,
                    generated_ideas=[
                        GeneratedIdea(
                            content=f"Idea from {method.value}",
                            thinking_method=method,
                            agent_name="simulated",
                            generation_prompt="test",
                            metadata={"llm_cost": costs[method]},
                        )
                    ],
                    execution_time=0.1,
                )

        orchestrator = VaryingCostOrchestrator(auto_setup=False)

        result = await orchestrator.run_qadi_cycle("Test problem")

        # 0.001 + 0.005 + 0.003 + 0.002 = 0.011
        assert result["total_cost"] == pytest.approx(0.011)

    @pytest.mark.asyncio
    async def test_error_result_factories_in_practice(self):
        """
        Verify error result factories create proper results.
        """

        class ErrorProneOrchestrator(MinimalRealOrchestrator):
            async def _simulate_phase(self, method, problem_statement):
                # Fail deduction phase
                if method == ThinkingMethod.DEDUCTION:
                    raise ValueError("Invalid hypothesis format")
                return await super()._simulate_phase(method, problem_statement)

        orchestrator = ErrorProneOrchestrator(auto_setup=False)

        result = await orchestrator.run_qadi_cycle("Test problem")

        # Verify error result structure
        deduction_result = result["phases"]["deduction"]
        assert deduction_result.agent_name == "error"
        assert deduction_result.thinking_method == ThinkingMethod.DEDUCTION
        assert deduction_result.generated_ideas == []
        assert "Invalid hypothesis format" in deduction_result.error_message
        assert deduction_result.generation_metadata.get("error") is True

    @pytest.mark.asyncio
    async def test_optional_hooks_integration(self):
        """
        Verify optional hooks can be used in subclasses.
        """

        class HookedOrchestrator(MinimalRealOrchestrator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hooks_called = []

            async def _initialize_cycle(self, problem_statement: str) -> None:
                self.hooks_called.append("initialize")

            async def _finalize_cycle(self, result: Any) -> None:
                self.hooks_called.append("finalize")

            async def run_qadi_cycle(self, problem_statement, context=None, cycle_config=None):
                await self._initialize_cycle(problem_statement)
                result = await super().run_qadi_cycle(problem_statement, context, cycle_config)
                await self._finalize_cycle(result)
                return result

        orchestrator = HookedOrchestrator(auto_setup=False)

        await orchestrator.run_qadi_cycle("Test problem")

        # Verify hooks were called
        assert orchestrator.hooks_called == ["initialize", "finalize"]

    @pytest.mark.asyncio
    async def test_multiple_orchestrators_independent_state(self):
        """
        Verify multiple orchestrator instances maintain independent state.
        """
        orch1 = MinimalRealOrchestrator(auto_setup=False)
        orch2 = MinimalRealOrchestrator(auto_setup=False)

        # Run different problems
        result1 = await orch1.run_qadi_cycle("Problem 1")
        result2 = await orch2.run_qadi_cycle("Problem 2")

        # Verify independent execution logs
        assert "Problem 1" in str(result1["problem_statement"])
        assert "Problem 2" in str(result2["problem_statement"])

        # Verify circuit breakers are independent
        assert orch1._circuit_breakers is not orch2._circuit_breakers

    def test_orchestrator_can_disable_circuit_breakers(self):
        """
        Verify circuit breakers can be disabled via configuration.
        """
        orch_with = MinimalRealOrchestrator(enable_circuit_breakers=True)
        orch_without = MinimalRealOrchestrator(enable_circuit_breakers=False)

        # Record some failures
        orch_with._record_agent_failure(ThinkingMethod.ABDUCTION)
        orch_without._record_agent_failure(ThinkingMethod.ABDUCTION)

        # With circuit breakers enabled, tracking works
        breaker = orch_with._get_circuit_breaker(ThinkingMethod.ABDUCTION)
        assert breaker.consecutive_failures == 1

        # Without circuit breakers, can_use_agent always returns True
        assert orch_without._can_use_agent(ThinkingMethod.ABDUCTION) is True

    def test_template_agent_creation_for_all_methods(self):
        """
        Verify template agents can be created for all QADI methods.
        """
        orchestrator = MinimalRealOrchestrator(auto_setup=False)

        for method in orchestrator.QADI_SEQUENCE:
            agent = orchestrator._create_template_agent(method)
            assert agent is not None
            assert agent.thinking_method == method


class TestRealWorldUsagePattern:
    """Test realistic usage patterns that developers would follow."""

    @pytest.mark.asyncio
    async def test_standard_orchestrator_workflow(self):
        """
        Test the standard workflow a developer would follow.
        """
        # 1. Create orchestrator with default settings
        orchestrator = MinimalRealOrchestrator()

        # 2. Run QADI cycle with user input
        result = await orchestrator.run_qadi_cycle(
            problem_statement="How can we improve system performance?",
            context="Current response time is 500ms",
        )

        # 3. Verify we got results
        assert result is not None
        assert "problem_statement" in result
        assert "phases" in result
        assert "all_ideas" in result
        assert "total_cost" in result

        # 4. Verify all phases completed
        assert len(result["phases"]) == 4

        # 5. Verify we can access ideas
        assert len(result["all_ideas"]) > 0

        # 6. Verify cost tracking
        assert result["total_cost"] >= 0

    @pytest.mark.asyncio
    async def test_orchestrator_with_custom_configuration(self):
        """
        Test orchestrator with custom configuration.
        """
        # Custom configuration pattern
        orchestrator = MinimalRealOrchestrator(
            auto_setup=True,
            enable_circuit_breakers=True,
        )

        result = await orchestrator.run_qadi_cycle("Test problem")

        # Verify configuration was applied
        assert orchestrator.auto_setup is True
        assert orchestrator.enable_circuit_breakers is True

    @pytest.mark.asyncio
    async def test_error_recovery_pattern(self):
        """
        Test that orchestrator can recover from errors.
        """

        class RecoveryOrchestrator(MinimalRealOrchestrator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.abduction_attempts = 0

            async def _simulate_phase(self, method, problem_statement):
                # Only fail abduction phase on first attempt
                if method == ThinkingMethod.ABDUCTION:
                    self.abduction_attempts += 1
                    if self.abduction_attempts == 1:
                        raise Exception("Temporary failure")
                return await super()._simulate_phase(method, problem_statement)

        orchestrator = RecoveryOrchestrator(auto_setup=False)

        # First cycle fails abduction
        result1 = await orchestrator.run_qadi_cycle("Test 1")
        assert result1["phases"]["abduction"].error_message is not None
        assert "Temporary failure" in result1["phases"]["abduction"].error_message

        # Second cycle succeeds (circuit not open yet, only 1 failure recorded)
        result2 = await orchestrator.run_qadi_cycle("Test 2")
        assert result2["phases"]["abduction"].error_message is None
        assert len(result2["phases"]["abduction"].generated_ideas) > 0
