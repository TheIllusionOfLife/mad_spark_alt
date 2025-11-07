"""
Base Orchestrator for QADI system.

This module provides the abstract base class for all QADI orchestrators,
containing shared logic for agent management, circuit breakers, context building,
and error handling.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from .smart_registry import SmartAgentRegistry, smart_registry

logger = logging.getLogger(__name__)


@dataclass
class AgentCircuitBreaker:
    """
    Circuit breaker for agent failures.

    Implements the circuit breaker pattern to prevent cascading failures
    when agents repeatedly fail. Opens after threshold failures and allows
    retry after cooldown period.
    """

    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    is_open: bool = False
    failure_threshold: int = 3
    cooldown_seconds: float = 300.0  # 5 minutes

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.consecutive_failures} failures"
            )

    def record_success(self) -> None:
        """Record a success and reset the circuit."""
        self.consecutive_failures = 0
        self.is_open = False
        self.last_failure_time = None

    def can_attempt(self) -> bool:
        """Check if we can attempt to use this agent."""
        if not self.is_open:
            return True

        # Check if cooldown period has passed
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.cooldown_seconds:
                # Half-open state: allow one attempt
                return True

        return False


class BaseOrchestrator(ABC):
    """
    Abstract base class for QADI orchestration.

    Provides common infrastructure for all QADI orchestrators including:
    - Agent registry management
    - Circuit breaker pattern for failure handling
    - Helper methods for context building and idea synthesis
    - Error result factories
    - Agent management and setup

    Subclasses must implement:
    - run_qadi_cycle(): The main orchestration logic

    Subclasses can optionally override:
    - _initialize_cycle(): Custom cycle initialization
    - _finalize_cycle(): Custom cycle finalization
    """

    # QADI phase sequence constant
    QADI_SEQUENCE = [
        ThinkingMethod.QUESTIONING,
        ThinkingMethod.ABDUCTION,
        ThinkingMethod.DEDUCTION,
        ThinkingMethod.INDUCTION,
    ]

    def __init__(
        self,
        registry: Optional[SmartAgentRegistry] = None,
        auto_setup: bool = True,
        enable_circuit_breakers: bool = True,
    ):
        """
        Initialize the base orchestrator.

        Args:
            registry: Smart registry to use (uses global smart_registry if None)
            auto_setup: Whether to automatically setup agents on first use
            enable_circuit_breakers: Whether to enable circuit breaker pattern
        """
        self.registry = registry or smart_registry
        self.auto_setup = auto_setup
        self.enable_circuit_breakers = enable_circuit_breakers

        # Setup tracking
        self._setup_completed = False
        self._setup_status: Dict[str, str] = {}

        # Circuit breakers for each thinking method
        self._circuit_breakers: Dict[str, AgentCircuitBreaker] = {}

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute complete QADI cycle.

        This is the main entry point for orchestration logic and must be
        implemented by all subclasses.

        Args:
            problem_statement: The problem or challenge to address
            context: Optional additional context
            cycle_config: Configuration for the cycle execution

        Returns:
            Result object (type depends on subclass implementation)
        """
        pass

    # ========================================================================
    # Optional Hooks (can be overridden by subclasses)
    # ========================================================================

    async def _initialize_cycle(self, problem_statement: str) -> None:
        """
        Optional hook for cycle initialization.

        Override this method to add custom initialization logic before
        the main cycle execution.

        Args:
            problem_statement: The problem statement for this cycle
        """
        pass

    async def _finalize_cycle(self, result: Any) -> None:
        """
        Optional hook for cycle finalization.

        Override this method to add custom finalization logic after
        the main cycle execution.

        Args:
            result: The cycle result to finalize
        """
        pass

    # ========================================================================
    # Circuit Breaker Methods
    # ========================================================================

    def _get_circuit_breaker(self, method: ThinkingMethod) -> AgentCircuitBreaker:
        """
        Get or create circuit breaker for a thinking method.

        Args:
            method: The thinking method

        Returns:
            Circuit breaker for the method
        """
        method_key = method.value
        if method_key not in self._circuit_breakers:
            self._circuit_breakers[method_key] = AgentCircuitBreaker()
        return self._circuit_breakers[method_key]

    def _can_use_agent(self, method: ThinkingMethod) -> bool:
        """
        Check if agent can be used (circuit breaker check).

        Args:
            method: The thinking method

        Returns:
            True if agent can be used, False if circuit is open
        """
        if not self.enable_circuit_breakers:
            return True

        circuit_breaker = self._get_circuit_breaker(method)
        return circuit_breaker.can_attempt()

    def _record_agent_success(self, method: ThinkingMethod) -> None:
        """
        Record successful agent execution.

        Args:
            method: The thinking method that succeeded
        """
        if not self.enable_circuit_breakers:
            return

        circuit_breaker = self._get_circuit_breaker(method)
        circuit_breaker.record_success()

    def _record_agent_failure(self, method: ThinkingMethod) -> None:
        """
        Record failed agent execution.

        Args:
            method: The thinking method that failed
        """
        if not self.enable_circuit_breakers:
            return

        circuit_breaker = self._get_circuit_breaker(method)
        circuit_breaker.record_failure()

    # ========================================================================
    # Agent Management Methods
    # ========================================================================

    async def ensure_agents_ready(self) -> Dict[str, str]:
        """
        Ensure agents are setup and ready for use.

        This method is idempotent - it only runs setup once, even if called
        multiple times.

        Returns:
            Dictionary with setup status for each thinking method
        """
        if not self._setup_completed and self.auto_setup:
            logger.info("Setting up intelligent agents...")
            setup_start = time.time()

            self._setup_status = await self.registry.setup_intelligent_agents()
            self._setup_completed = True

            setup_time = time.time() - setup_start
            logger.info(f"Agent setup completed in {setup_time:.2f}s")

        return self._setup_status

    def _create_template_agent(
        self, method: ThinkingMethod
    ) -> Optional[ThinkingAgentInterface]:
        """
        Create a template agent for the given thinking method.

        Template agents are fallback agents that don't require LLM APIs.

        Args:
            method: The thinking method

        Returns:
            Template agent instance or None if creation fails
        """
        try:
            # Direct imports to avoid circular dependencies
            if method == ThinkingMethod.QUESTIONING:
                from ..agents import QuestioningAgent

                return QuestioningAgent()
            elif method == ThinkingMethod.ABDUCTION:
                from ..agents import AbductionAgent

                return AbductionAgent()
            elif method == ThinkingMethod.DEDUCTION:
                from ..agents import DeductionAgent

                return DeductionAgent()
            elif method == ThinkingMethod.INDUCTION:
                from ..agents import InductionAgent

                return InductionAgent()
            else:
                logger.error(f"Unknown thinking method: {method}")
                return None
        except ImportError as e:
            logger.error(f"Failed to import template agent for {method.value}: {e}")
            return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_enhanced_context(
        self,
        base_context: Optional[str],
        *phase_results: Optional[IdeaGenerationResult],
    ) -> str:
        """
        Build enhanced context from previous phase results.

        Combines base context with insights from completed phases to provide
        richer context for subsequent phases.

        Args:
            base_context: Initial context string
            *phase_results: Variable number of phase results to include

        Returns:
            Enhanced context string combining all inputs
        """
        context_parts = []

        if base_context:
            context_parts.append(f"Initial context: {base_context}")

        for phase_result in phase_results:
            if phase_result and phase_result.generated_ideas:
                method_name = phase_result.thinking_method.value.title()
                ideas_text = "\n".join(
                    [f"- {idea.content}" for idea in phase_result.generated_ideas]
                )
                context_parts.append(f"{method_name} phase insights:\n{ideas_text}")

        return "\n\n".join(context_parts)

    def _synthesize_ideas(
        self, phases: Dict[str, IdeaGenerationResult]
    ) -> List[GeneratedIdea]:
        """
        Synthesize the best ideas from all phases.

        Collects all ideas from all phases and adds phase metadata to each idea.

        Args:
            phases: Dictionary mapping phase names to results

        Returns:
            List of all ideas from all phases with phase metadata
        """
        all_ideas = []

        for phase_name, phase_result in phases.items():
            if phase_result and phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    # Add phase information to metadata
                    idea.metadata["phase"] = phase_name
                    all_ideas.append(idea)

        return all_ideas

    def _extract_llm_cost(self, phase_result: IdeaGenerationResult) -> float:
        """
        Extract LLM cost from phase result.

        Sums up the llm_cost from all generated ideas in the result.

        Args:
            phase_result: The phase result to extract cost from

        Returns:
            Total LLM cost from all ideas
        """
        total_cost = 0.0
        if phase_result.generated_ideas:
            for idea in phase_result.generated_ideas:
                if "llm_cost" in idea.metadata:
                    total_cost += idea.metadata["llm_cost"]
        return total_cost

    # ========================================================================
    # Error Result Factory Methods
    # ========================================================================

    def _create_timeout_result(self, method: ThinkingMethod) -> IdeaGenerationResult:
        """
        Create a result for a timed-out phase.

        Args:
            method: The thinking method that timed out

        Returns:
            Empty result with timeout metadata
        """
        return IdeaGenerationResult(
            agent_name="timeout",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            error_message="Phase skipped due to timeout",
            generation_metadata={"timeout": True},
        )

    def _create_error_result(
        self, method: ThinkingMethod, error: str
    ) -> IdeaGenerationResult:
        """
        Create an error result.

        Args:
            method: The thinking method that encountered an error
            error: Error message

        Returns:
            Empty result with error metadata
        """
        return IdeaGenerationResult(
            agent_name="error",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            error_message=error,
            generation_metadata={"error": True},
        )

    def _create_empty_result(self, method: ThinkingMethod) -> IdeaGenerationResult:
        """
        Create an empty result.

        Args:
            method: The thinking method

        Returns:
            Empty result with empty metadata
        """
        return IdeaGenerationResult(
            agent_name="empty",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            generation_metadata={"empty": True},
        )
