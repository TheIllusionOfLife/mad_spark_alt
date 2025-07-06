"""
Orchestration engine for coordinating multi-agent idea generation.

This module implements the QADI cycle (Question → Abduction → Deduction → Induction)
based on "Shin Logical Thinking" methodology for collaborative idea generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingAgentInterface,
    ThinkingMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class QADICycleResult:
    """Result from a complete QADI cycle execution."""

    problem_statement: str
    cycle_id: str
    phases: Dict[str, IdeaGenerationResult] = field(default_factory=dict)
    synthesized_ideas: List[GeneratedIdea] = field(default_factory=list)
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QADIOrchestrator:
    """
    Orchestrates the QADI cycle for collaborative idea generation.

    The QADI cycle follows Shin Logical Thinking methodology:
    - Question: Generate diverse questions to frame the problem
    - Abduction: Create hypotheses and creative leaps
    - Deduction: Apply logical reasoning and validation
    - Induction: Synthesize patterns and generalizations
    """

    def __init__(self, agents: Optional[List[ThinkingAgentInterface]] = None):
        """
        Initialize the orchestrator with thinking agents.

        Args:
            agents: List of thinking agents for different methods
        """
        self.agents: Dict[ThinkingMethod, ThinkingAgentInterface] = {}
        if agents:
            for agent in agents:
                self.register_agent(agent)

    def register_agent(self, agent: ThinkingAgentInterface) -> None:
        """Register a thinking agent for a specific method."""
        self.agents[agent.thinking_method] = agent
        logger.info(
            f"Registered agent '{agent.name}' for {agent.thinking_method.value}"
        )

    def get_agent(self, method: ThinkingMethod) -> Optional[ThinkingAgentInterface]:
        """Get the agent for a specific thinking method."""
        return self.agents.get(method)

    def has_agent(self, method: ThinkingMethod) -> bool:
        """Check if an agent is available for the given method."""
        return method in self.agents

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> QADICycleResult:
        """
        Execute a complete QADI cycle for the given problem.

        Args:
            problem_statement: The problem or challenge to address
            context: Optional additional context
            cycle_config: Configuration for the cycle execution

        Returns:
            Result containing outputs from all phases
        """
        start_time = asyncio.get_running_loop().time()
        cycle_id = f"qadi_{int(start_time)}"

        logger.info(f"Starting QADI cycle {cycle_id} for: {problem_statement[:100]}...")

        config = cycle_config or {}
        result = QADICycleResult(
            problem_statement=problem_statement,
            cycle_id=cycle_id,
            metadata={"config": config, "context": context},
        )

        # Execute QADI sequence phases
        QADI_SEQUENCE = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]

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

            # Run the phase (this will handle missing agents correctly)
            phase_result = await self._run_phase(
                method, problem_statement, phase_context, config
            )
            result.phases[method.value] = phase_result

        # Synthesize final ideas from all phases
        result.synthesized_ideas = self._synthesize_ideas(result.phases)

        end_time = asyncio.get_running_loop().time()
        result.execution_time = end_time - start_time

        logger.info(f"QADI cycle {cycle_id} completed in {result.execution_time:.2f}s")
        return result

    async def _run_phase(
        self,
        method: ThinkingMethod,
        problem_statement: str,
        context: Optional[str],
        config: Dict[str, Any],
    ) -> IdeaGenerationResult:
        """Run a single phase of the QADI cycle."""
        agent = self.get_agent(method)
        if not agent:
            logger.warning(f"No agent available for {method.value}")
            return IdeaGenerationResult(
                agent_name="missing",
                thinking_method=method,
                generated_ideas=[],
                error_message=f"No agent available for {method.value}",
            )

        request = IdeaGenerationRequest(
            problem_statement=problem_statement,
            context=context,
            target_thinking_methods=[method],
            generation_config=config.get(method.value, {}),
            max_ideas_per_method=config.get("max_ideas_per_method", 3),
            require_reasoning=config.get("require_reasoning", True),
        )

        try:
            return await agent.generate_ideas(request)
        except Exception as e:
            logger.error(f"Error in {method.value} phase: {e}")
            return IdeaGenerationResult(
                agent_name=agent.name,
                thinking_method=method,
                generated_ideas=[],
                error_message=str(e),
            )

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

        # For now, return all ideas. Future enhancement: apply ranking/filtering
        return all_ideas

    async def run_parallel_generation(
        self,
        problem_statement: str,
        thinking_methods: List[ThinkingMethod],
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[ThinkingMethod, IdeaGenerationResult]:
        """
        Run multiple thinking methods in parallel (not following QADI sequence).

        Args:
            problem_statement: The problem to address
            thinking_methods: List of thinking methods to use
            context: Optional context
            config: Configuration for generation

        Returns:
            Dictionary mapping thinking methods to their results
        """
        logger.info(f"Running parallel generation with methods: {thinking_methods}")

        tasks = []
        available_methods = []

        for method in thinking_methods:
            if self.has_agent(method):
                available_methods.append(method)
                task = self._run_phase(method, problem_statement, context, config or {})
                tasks.append(task)
            else:
                logger.warning(f"No agent available for {method.value}")

        if not tasks:
            logger.error("No agents available for any requested thinking methods")
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for method, result in zip(available_methods, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {method.value}: {result}")
                result_dict[method] = IdeaGenerationResult(
                    agent_name="error",
                    thinking_method=method,
                    generated_ideas=[],
                    error_message=str(result),
                )
            else:
                # Type assertion: result is IdeaGenerationResult when not an Exception
                result_dict[method] = result  # type: ignore[assignment]

        return result_dict
