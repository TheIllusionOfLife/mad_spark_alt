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

from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from .orchestrator import QADICycleResult
from .smart_registry import SmartAgentRegistry, smart_registry
from .conclusion_synthesizer import ConclusionSynthesizer, Conclusion

logger = logging.getLogger(__name__)


@dataclass
class SmartQADICycleResult(QADICycleResult):
    """Enhanced QADI cycle result with agent type information."""

    agent_types: Dict[str, str] = field(
        default_factory=dict
    )  # method -> agent_type (LLM/template)
    llm_cost: float = 0.0  # Total LLM cost for the cycle
    setup_time: float = 0.0  # Time spent on agent setup
    conclusion: Optional[Conclusion] = None  # Synthesized conclusion


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
        self, registry: Optional[SmartAgentRegistry] = None, auto_setup: bool = True
    ):
        """
        Initialize the smart orchestrator.

        Args:
            registry: Smart registry to use (uses global if None)
            auto_setup: Whether to automatically setup agents on first use
        """
        self.registry = registry or smart_registry
        self.auto_setup = auto_setup
        self._setup_completed = False
        self._setup_status: Dict[str, str] = {}

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

        # Ensure agents are ready
        setup_start = time.time()
        setup_status = await self.ensure_agents_ready()
        setup_time = time.time() - setup_start

        config = cycle_config or {}
        result = SmartQADICycleResult(
            problem_statement=problem_statement,
            cycle_id=cycle_id,
            metadata={
                "config": config,
                "context": context,
                "setup_status": setup_status,
            },
            setup_time=setup_time,
        )

        # Execute QADI sequence phases
        QADI_SEQUENCE = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]

        total_llm_cost = 0.0

        for i, method in enumerate(QADI_SEQUENCE):
            logger.info(f"Phase {i+1}: {method.value.title()}")

            # Build enhanced context using previous phase results
            if i == 0:
                # First phase uses original context
                phase_context = context
            else:
                # Later phases use enhanced context from previous phases
                previous_phase_results = [
                    result.phases.get(m.value)
                    for m in QADI_SEQUENCE[:i]
                    if result.phases.get(m.value) is not None
                ]
                phase_context = self._build_enhanced_context(
                    context, *previous_phase_results
                )

            # Run the phase with smart agent selection
            phase_result, agent_type = await self._run_smart_phase(
                method, problem_statement, phase_context, config
            )

            result.phases[method.value] = phase_result
            result.agent_types[method.value] = agent_type

            # Track LLM costs
            if phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    if "llm_cost" in idea.metadata:
                        total_llm_cost += idea.metadata["llm_cost"]

        result.llm_cost = total_llm_cost

        # Synthesize final ideas from all phases
        result.synthesized_ideas = self._synthesize_ideas(result.phases)

        # Generate conclusion from all ideas
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

        end_time = time.time()
        result.execution_time = end_time - start_time

        logger.info(
            f"Smart QADI cycle {cycle_id} completed in {result.execution_time:.2f}s"
        )
        if result.llm_cost > 0:
            logger.info(f"Total LLM cost: ${result.llm_cost:.4f}")

        return result

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
            agent = self.registry.get_preferred_agent(method)
            if agent:
                available_methods.append(method)
                task = self._run_smart_phase(
                    method, problem_statement, context, config or {}
                )
                tasks.append(task)
            else:
                logger.warning(f"No agent available for {method.value}")

        if not tasks:
            logger.error("No agents available for any requested thinking methods")
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for method, result in zip(available_methods, results):
            # Check if result is an exception
            if isinstance(result, BaseException):
                logger.error(f"Error in {method.value}: {result}")
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
                # Type assertion since we know it's a tuple from _run_smart_phase
                result_dict[method] = result  # type: ignore

        return result_dict

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the smart orchestrator."""
        return {
            "setup_completed": self._setup_completed,
            "setup_status": self._setup_status,
            "registry_status": self.registry.get_agent_status(),
        }
