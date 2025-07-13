"""
Smart QADI Orchestrator with automatic LLM agent preference and fallback.

This module extends the basic orchestrator with intelligent agent selection
that automatically uses LLM agents when available and falls back gracefully.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from .conclusion_synthesizer import Conclusion, ConclusionSynthesizer
from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from .orchestrator import QADICycleResult
from .smart_registry import SmartAgentRegistry, smart_registry

logger = logging.getLogger(__name__)

# Default timeout values
DEFAULT_PHASE_TIMEOUT = 120  # 2 minutes per phase
DEFAULT_PARALLEL_TIMEOUT = 180  # 3 minutes for parallel execution
DEFAULT_CONCLUSION_TIMEOUT = 60  # 1 minute for conclusion synthesis


@dataclass
class SmartQADICycleResult(QADICycleResult):
    """Enhanced QADI cycle result with agent type information."""

    agent_types: Dict[str, str] = field(
        default_factory=dict
    )  # method -> agent_type (LLM/template)
    llm_cost: float = 0.0  # Total LLM cost for the cycle
    setup_time: float = 0.0  # Time spent on agent setup
    conclusion: Optional[Conclusion] = None  # Synthesized conclusion
    timeout_info: Dict[str, Any] = field(default_factory=dict)  # Timeout tracking


@dataclass
class AgentCircuitBreaker:
    """Circuit breaker for agent failures."""

    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    is_open: bool = False
    failure_threshold: int = 3
    cooldown_seconds: float = 300  # 5 minutes

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
                # Half-open state: allow one attempt. The circuit will be fully
                # closed on the next call to record_success().
                return True

        return False


class SmartQADIOrchestrator:
    """
    Smart QADI orchestrator with automatic LLM agent preference.

    Features:
    - Automatic LLM agent setup and preference
    - Graceful fallback to template agents
    - Cost tracking for LLM usage
    - Enhanced error handling and recovery
    """

    def __init__(
        self,
        registry: Optional[SmartAgentRegistry] = None,
        auto_setup: bool = True,
        *,  # Force timeout parameters to be keyword-only for backward compatibility
        phase_timeout: float = DEFAULT_PHASE_TIMEOUT,
        parallel_timeout: float = DEFAULT_PARALLEL_TIMEOUT,
        conclusion_timeout: float = DEFAULT_CONCLUSION_TIMEOUT,
    ):
        """
        Initialize the smart orchestrator.

        Args:
            registry: Smart registry to use (uses global if None)
            auto_setup: Whether to automatically setup agents on first use
            phase_timeout: Timeout for individual phases in seconds
            parallel_timeout: Timeout for parallel execution in seconds
            conclusion_timeout: Timeout for conclusion synthesis in seconds
        """
        self.registry = registry or smart_registry
        self.auto_setup = auto_setup
        self._setup_completed = False
        self._setup_status: Dict[str, str] = {}

        # Timeout configuration
        self.phase_timeout = phase_timeout
        self.parallel_timeout = parallel_timeout
        self.conclusion_timeout = conclusion_timeout

        # Circuit breakers for each thinking method
        self._circuit_breakers: Dict[str, AgentCircuitBreaker] = {}

    async def ensure_agents_ready(self) -> Dict[str, str]:
        """
        Ensure agents are setup and ready for use.

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

    def _get_circuit_breaker(self, method: ThinkingMethod) -> AgentCircuitBreaker:
        """Get or create circuit breaker for a thinking method."""
        method_key = method.value
        if method_key not in self._circuit_breakers:
            self._circuit_breakers[method_key] = AgentCircuitBreaker()
        return self._circuit_breakers[method_key]

    def _can_use_agent(self, method: ThinkingMethod) -> bool:
        """Check if agent can be used (circuit breaker check)."""
        circuit_breaker = self._get_circuit_breaker(method)
        return circuit_breaker.can_attempt()

    def _record_agent_success(self, method: ThinkingMethod) -> None:
        """Record successful agent execution."""
        circuit_breaker = self._get_circuit_breaker(method)
        circuit_breaker.record_success()

    def _record_agent_failure(self, method: ThinkingMethod) -> None:
        """Record failed agent execution."""
        circuit_breaker = self._get_circuit_breaker(method)
        circuit_breaker.record_failure()

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> SmartQADICycleResult:
        """
        Execute a smart QADI cycle with automatic agent selection.

        Args:
            problem_statement: The problem or challenge to address
            context: Optional additional context
            cycle_config: Configuration for the cycle execution

        Returns:
            Enhanced result with agent type information and cost tracking
        """
        start_time = time.time()
        cycle_id = f"smart_qadi_{int(start_time)}"

        logger.info(
            f"Starting smart QADI cycle {cycle_id} for: {problem_statement[:100]}..."
        )

        # Initialize cycle result
        result = await self._initialize_qadi_cycle(
            problem_statement, context, cycle_config, cycle_id
        )

        # Execute all QADI phases
        await self._execute_qadi_phases(
            result, problem_statement, context, cycle_config or {}
        )

        # Finalize the cycle
        await self._finalize_qadi_cycle(result, problem_statement, context, start_time)

        return result

    async def _run_smart_phase_with_circuit_breaker(
        self,
        method: ThinkingMethod,
        problem_statement: str,
        context: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[IdeaGenerationResult, str]:
        """
        Run a single phase with smart agent selection and circuit breaker protection.

        Returns:
            Tuple of (phase_result, agent_type)
        """
        # Check circuit breaker first
        if not self._can_use_agent(method):
            logger.warning(f"Circuit breaker open for {method.value}, skipping")
            return (
                IdeaGenerationResult(
                    agent_name="circuit_breaker_open",
                    thinking_method=method,
                    generated_ideas=[],
                    error_message=f"Circuit breaker open for {method.value}",
                ),
                "circuit_breaker_open",
            )

        # Delegate to existing implementation
        return await self._run_smart_phase(method, problem_statement, context, config)

    async def _run_smart_phase(
        self,
        method: ThinkingMethod,
        problem_statement: str,
        context: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[IdeaGenerationResult, str]:
        """
        Run a single phase with smart agent selection.

        Returns:
            Tuple of (phase_result, agent_type)
        """
        agent = self.registry.get_preferred_agent(method)
        agent_type = "none"

        if not agent:
            logger.warning(f"No agent available for {method.value}")
            return (
                IdeaGenerationResult(
                    agent_name="missing",
                    thinking_method=method,
                    generated_ideas=[],
                    error_message=f"No agent available for {method.value}",
                ),
                agent_type,
            )

        # Determine agent type for tracking
        if agent.is_llm_powered:
            agent_type = "LLM"
        else:
            agent_type = "template"

        logger.info(f"Using {agent_type} agent: {agent.name}")

        request = IdeaGenerationRequest(
            problem_statement=problem_statement,
            context=context,
            target_thinking_methods=[method],
            generation_config=config.get(method.value, {}),
            max_ideas_per_method=config.get("max_ideas_per_method", 3),
            require_reasoning=config.get("require_reasoning", True),
        )

        try:
            result = await agent.generate_ideas(request)
            return result, agent_type
        except Exception as e:
            logger.error(f"Error in {method.value} phase with {agent_type} agent: {e}")

            # For LLM agents, try to fallback to template agent
            if agent_type == "LLM":
                logger.info(f"Attempting fallback to template agent for {method.value}")
                fallback_result = await self._try_template_fallback(method, request)
                if fallback_result:
                    return fallback_result, "template_fallback"

            return (
                IdeaGenerationResult(
                    agent_name=agent.name,
                    thinking_method=method,
                    generated_ideas=[],
                    error_message=str(e),
                ),
                agent_type,
            )

    async def _try_template_fallback(
        self, method: ThinkingMethod, request: IdeaGenerationRequest
    ) -> Optional[IdeaGenerationResult]:
        """
        Try to fallback to template agent when LLM agent fails.

        Args:
            method: The thinking method
            request: The generation request

        Returns:
            Result from template agent or None if not available
        """
        try:
            fallback_agent = self._create_template_agent(method)
            if not fallback_agent:
                return None

            logger.info(f"Using fallback template agent: {fallback_agent.name}")
            return await fallback_agent.generate_ideas(request)

        except Exception as e:
            logger.error(f"Template fallback also failed for {method.value}: {e}")
            return None

    async def _synthesize_conclusion(
        self,
        result: SmartQADICycleResult,
        problem_statement: str,
        context: Optional[str] = None,
    ) -> None:
        """
        Synthesize conclusion from generated ideas.

        Args:
            result: The QADI cycle result to add conclusion to
            problem_statement: The original problem statement
            context: Optional additional context
        """
        if result.synthesized_ideas:
            try:
                conclusion_synthesizer = ConclusionSynthesizer(use_llm=True)

                # Group ideas by phase for conclusion synthesis
                ideas_by_phase: Dict[str, List[GeneratedIdea]] = {}
                for idea in result.synthesized_ideas:
                    phase = idea.metadata.get("phase", "unknown")
                    if phase not in ideas_by_phase:
                        ideas_by_phase[phase] = []
                    ideas_by_phase[phase].append(idea)

                result.conclusion = await conclusion_synthesizer.synthesize_conclusion(
                    problem_statement=problem_statement,
                    ideas_by_phase=ideas_by_phase,
                    context=context,
                )

                # Add cost of conclusion synthesis
                if (
                    hasattr(result.conclusion, "metadata")
                    and "llm_cost" in result.conclusion.metadata
                ):
                    result.llm_cost += result.conclusion.metadata["llm_cost"]

            except Exception as e:
                logger.error(f"Failed to synthesize conclusion: {e}")
                # Continue without conclusion rather than failing the whole cycle

    def _create_template_agent(
        self, method: ThinkingMethod
    ) -> Optional[ThinkingAgentInterface]:
        """Create a template agent for the given thinking method."""
        try:
            # Direct imports to avoid path issues
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

    def _build_enhanced_context(
        self,
        base_context: Optional[str],
        *phase_results: Optional[IdeaGenerationResult],
    ) -> str:
        """Build enhanced context from previous phase results."""
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
        """Synthesize the best ideas from all phases."""
        all_ideas = []

        for phase_name, phase_result in phases.items():
            if phase_result and phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    # Add phase information to metadata
                    idea.metadata["phase"] = phase_name
                    all_ideas.append(idea)

        return all_ideas

    async def run_parallel_generation(
        self,
        problem_statement: str,
        thinking_methods: List[ThinkingMethod],
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[ThinkingMethod, Tuple[IdeaGenerationResult, str]]:
        """
        Run multiple thinking methods in parallel with smart agent selection.

        Returns:
            Dictionary mapping thinking methods to (result, agent_type) tuples
        """
        logger.info(
            f"Running smart parallel generation with methods: {thinking_methods}"
        )

        # Ensure agents are ready
        await self.ensure_agents_ready()

        tasks = []
        available_methods = []

        for method in thinking_methods:
            # Check circuit breaker before creating task
            if not self._can_use_agent(method):
                logger.warning(f"Circuit breaker open for {method.value}, skipping")
                continue

            agent = self.registry.get_preferred_agent(method)
            if agent:
                available_methods.append(method)
                task_coroutine = self._run_smart_phase_with_circuit_breaker(
                    method, problem_statement, context, config or {}
                )
                task = asyncio.create_task(task_coroutine)
                tasks.append(task)
            else:
                logger.warning(f"No agent available for {method.value}")

        if not tasks:
            logger.error("No agents available for any requested thinking methods")
            return {}

        # Use timeout for parallel execution
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.parallel_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Parallel execution timed out after {self.parallel_timeout}s"
            )
            # Cancel all running tasks to prevent resource leaks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        # Ignore cancellation and other errors during cleanup
                        pass

            # Try to collect partial results using as_completed
            return await self._collect_partial_results(tasks, available_methods)

        result_dict = {}
        for method, result in zip(available_methods, results):
            # Check if result is an exception
            if isinstance(result, BaseException):
                logger.error(f"Error in {method.value}: {result}")
                self._record_agent_failure(method)
                result_dict[method] = (
                    IdeaGenerationResult(
                        agent_name="error",
                        thinking_method=method,
                        generated_ideas=[],
                        error_message=str(result),
                    ),
                    "error",
                )
            else:
                # We know result is a tuple from _run_smart_phase_with_circuit_breaker
                # Cast to the expected tuple type for type safety
                typed_result: Tuple[IdeaGenerationResult, str] = result
                phase_result, agent_type = typed_result

                # Check if result contains errors before recording success
                if (
                    agent_type != "circuit_breaker_open"
                    and not phase_result.error_message
                ):
                    self._record_agent_success(method)
                elif phase_result.error_message:
                    self._record_agent_failure(method)

                result_dict[method] = typed_result

        return result_dict

    async def _collect_partial_results(
        self,
        tasks: List,
        available_methods: List[ThinkingMethod],
        timeout_per_task: float = 30.0,
    ) -> Dict[ThinkingMethod, Tuple[IdeaGenerationResult, str]]:
        """
        Collect partial results when parallel execution times out.

        Args:
            tasks: List of asyncio tasks
            available_methods: Corresponding thinking methods
            timeout_per_task: Timeout for each individual task

        Returns:
            Dictionary with completed results and timeout placeholders
        """
        result_dict = {}

        # Try to get results from completed tasks
        for task, method in zip(tasks, available_methods):
            try:
                if task.done():
                    result = task.result()
                    # Check if result contains errors before recording success
                    if isinstance(result, tuple) and len(result) == 2:
                        phase_result, agent_type = result
                        if (
                            agent_type != "circuit_breaker_open"
                            and not phase_result.error_message
                        ):
                            self._record_agent_success(method)
                        elif phase_result.error_message:
                            self._record_agent_failure(method)
                    else:
                        self._record_agent_success(method)
                    result_dict[method] = result
                else:
                    # Try to get result with short timeout
                    result = await asyncio.wait_for(task, timeout=timeout_per_task)
                    # Check if result contains errors before recording success
                    if isinstance(result, tuple) and len(result) == 2:
                        phase_result, agent_type = result
                        if (
                            agent_type != "circuit_breaker_open"
                            and not phase_result.error_message
                        ):
                            self._record_agent_success(method)
                        elif phase_result.error_message:
                            self._record_agent_failure(method)
                    else:
                        self._record_agent_success(method)
                    result_dict[method] = result
            except asyncio.TimeoutError:
                logger.warning(f"Individual timeout for {method.value}")
                self._record_agent_failure(method)
                result_dict[method] = (
                    IdeaGenerationResult(
                        agent_name="timeout",
                        thinking_method=method,
                        generated_ideas=[],
                        error_message=f"Task timed out after {timeout_per_task}s",
                    ),
                    "timeout",
                )
            except Exception as e:
                logger.error(f"Error collecting result for {method.value}: {e}")
                self._record_agent_failure(method)
                result_dict[method] = (
                    IdeaGenerationResult(
                        agent_name="error",
                        thinking_method=method,
                        generated_ideas=[],
                        error_message=str(e),
                    ),
                    "error",
                )

        return result_dict

    async def _initialize_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str],
        cycle_config: Optional[Dict[str, Any]],
        cycle_id: str,
    ) -> SmartQADICycleResult:
        """Initialize QADI cycle with agent setup and result structure."""
        # Ensure agents are ready
        setup_start = time.time()
        setup_status = await self.ensure_agents_ready()
        setup_time = time.time() - setup_start

        config = cycle_config or {}
        return SmartQADICycleResult(
            problem_statement=problem_statement,
            cycle_id=cycle_id,
            metadata={
                "config": config,
                "context": context,
                "setup_status": setup_status,
            },
            setup_time=setup_time,
        )

    async def _execute_qadi_phases(
        self,
        result: SmartQADICycleResult,
        problem_statement: str,
        context: Optional[str],
        config: Dict[str, Any],
    ) -> None:
        """Execute all QADI phases sequentially with context building."""
        QADI_SEQUENCE = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]

        total_llm_cost = 0.0

        for i, method in enumerate(QADI_SEQUENCE):
            logger.info(f"Phase {i+1}: {method.value.title()}")

            # Build phase context
            phase_context = self._build_phase_context(context, result, QADI_SEQUENCE, i)

            # Execute single phase
            phase_result, agent_type = await self._execute_single_phase(
                method, problem_statement, phase_context, config, result
            )

            # Store phase results
            result.phases[method.value] = phase_result
            result.agent_types[method.value] = agent_type

            # Track LLM costs
            total_llm_cost += self._extract_llm_cost(phase_result)

        result.llm_cost = total_llm_cost

    def _build_phase_context(
        self,
        base_context: Optional[str],
        result: SmartQADICycleResult,
        qadi_sequence: List[ThinkingMethod],
        phase_index: int,
    ) -> Optional[str]:
        """Build context for a specific phase using previous phase results."""
        if phase_index == 0:
            # First phase uses original context
            return base_context

        # Later phases use enhanced context from previous phases
        previous_phase_results = [
            result.phases.get(m.value)
            for m in qadi_sequence[:phase_index]
            if result.phases.get(m.value) is not None
        ]
        return self._build_enhanced_context(base_context, *previous_phase_results)

    async def _execute_single_phase(
        self,
        method: ThinkingMethod,
        problem_statement: str,
        phase_context: Optional[str],
        config: Dict[str, Any],
        result: SmartQADICycleResult,
    ) -> Tuple[IdeaGenerationResult, str]:
        """Execute a single QADI phase with timeout and error handling."""
        try:
            phase_result, agent_type = await asyncio.wait_for(
                self._run_smart_phase_with_circuit_breaker(
                    method, problem_statement, phase_context, config
                ),
                timeout=self.phase_timeout,
            )

            # Only record success if we actually executed the agent successfully
            # (not if circuit breaker was open or if result contains errors)
            if agent_type != "circuit_breaker_open" and not phase_result.error_message:
                self._record_agent_success(method)
            elif phase_result.error_message:
                # Record failure if the result contains an error message
                self._record_agent_failure(method)
            return phase_result, agent_type

        except asyncio.TimeoutError:
            return self._handle_phase_timeout(method, result)

        except Exception as e:
            return self._handle_phase_error(method, e)

    def _handle_phase_timeout(
        self, method: ThinkingMethod, result: SmartQADICycleResult
    ) -> Tuple[IdeaGenerationResult, str]:
        """Handle timeout during phase execution."""
        logger.warning(f"Phase {method.value} timed out after {self.phase_timeout}s")
        result.timeout_info[method.value] = self.phase_timeout
        self._record_agent_failure(method)

        # Create timeout result
        phase_result = IdeaGenerationResult(
            agent_name="timeout",
            thinking_method=method,
            generated_ideas=[],
            error_message=f"Phase timed out after {self.phase_timeout}s",
        )
        return phase_result, "timeout"

    def _handle_phase_error(
        self, method: ThinkingMethod, error: Exception
    ) -> Tuple[IdeaGenerationResult, str]:
        """Handle error during phase execution."""
        logger.error(f"Phase {method.value} failed: {error}")
        self._record_agent_failure(method)

        # Create error result
        phase_result = IdeaGenerationResult(
            agent_name="error",
            thinking_method=method,
            generated_ideas=[],
            error_message=str(error),
        )
        return phase_result, "error"

    def _extract_llm_cost(self, phase_result: IdeaGenerationResult) -> float:
        """Extract LLM cost from phase result."""
        total_cost = 0.0
        if phase_result.generated_ideas:
            for idea in phase_result.generated_ideas:
                if "llm_cost" in idea.metadata:
                    total_cost += idea.metadata["llm_cost"]
        return total_cost

    async def _finalize_qadi_cycle(
        self,
        result: SmartQADICycleResult,
        problem_statement: str,
        context: Optional[str],
        start_time: float,
    ) -> None:
        """Finalize QADI cycle with synthesis and timing."""
        # Synthesize final ideas from all phases
        result.synthesized_ideas = self._synthesize_ideas(result.phases)

        # Generate conclusion from all ideas with timeout
        try:
            await asyncio.wait_for(
                self._synthesize_conclusion(result, problem_statement, context),
                timeout=self.conclusion_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Conclusion synthesis timed out after {self.conclusion_timeout}s"
            )
            result.timeout_info["conclusion"] = self.conclusion_timeout

        # Set execution time
        end_time = time.time()
        result.execution_time = end_time - start_time

        logger.info(
            f"Smart QADI cycle {result.cycle_id} completed in {result.execution_time:.2f}s"
        )
        if result.llm_cost > 0:
            logger.info(f"Total LLM cost: ${result.llm_cost:.4f}")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the smart orchestrator."""
        circuit_breaker_status = {}
        for method_key, breaker in self._circuit_breakers.items():
            circuit_breaker_status[method_key] = {
                "is_open": breaker.is_open,
                "consecutive_failures": breaker.consecutive_failures,
                "last_failure_time": breaker.last_failure_time,
                "can_attempt": breaker.can_attempt(),
            }

        return {
            "setup_completed": self._setup_completed,
            "setup_status": self._setup_status,
            "registry_status": self.registry.get_agent_status(),
            "timeout_config": {
                "phase_timeout": self.phase_timeout,
                "parallel_timeout": self.parallel_timeout,
                "conclusion_timeout": self.conclusion_timeout,
            },
            "circuit_breakers": circuit_breaker_status,
        }
