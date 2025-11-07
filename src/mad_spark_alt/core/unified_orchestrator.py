"""
UnifiedQADIOrchestrator - Single orchestrator supporting all strategies.

Consolidates Simple, Smart, and MultiPerspective strategies into one
configuration-driven implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .orchestrator_config import (
    OrchestratorConfig,
    ExecutionMode,
    Strategy
)
from .simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult
from .smart_registry import SmartAgentRegistry
from .interfaces import GeneratedIdea
from .phase_logic import HypothesisScore


@dataclass
class UnifiedQADIResult:
    """
    Unified result structure for all QADI strategies.

    Contains fields common to all strategies plus optional strategy-specific fields.
    """

    # Common fields (all strategies)
    strategy_used: Strategy
    execution_mode: ExecutionMode
    core_question: str
    hypotheses: List[str]
    final_answer: str
    action_plan: List[str]
    total_llm_cost: float
    synthesized_ideas: List[GeneratedIdea]

    # Optional fields (strategy-specific)
    hypothesis_scores: Optional[List[HypothesisScore]] = None
    verification_examples: Optional[List[str]] = None
    verification_conclusion: Optional[str] = None
    perspectives_used: Optional[List[str]] = None
    synthesized_answer: Optional[str] = None
    agent_types: Optional[Dict[str, str]] = None

    # Metadata
    phase_results: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedQADIOrchestrator:
    """
    Single orchestrator supporting all execution modes and strategies
    via configuration.

    Strategies:
    - SIMPLE: Basic QADI cycle with hypothesis generation and evaluation
    - MULTI_PERSPECTIVE: Multi-perspective analysis with synthesis
    - SMART: Smart agent selection with circuit breakers

    Execution Modes:
    - SEQUENTIAL: Process phases one at a time
    - PARALLEL: Process phases concurrently where possible
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        registry: Optional[SmartAgentRegistry] = None,
        auto_setup: bool = True
    ):
        """
        Initialize unified orchestrator.

        Args:
            config: Configuration object (defaults to simple_config)
            registry: Custom agent registry (for Smart strategy)
            auto_setup: Automatically set up agents on first use

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config if config is not None else OrchestratorConfig.simple_config()
        self.registry = registry
        self.auto_setup = auto_setup

        # Validate configuration
        self.config.validate()

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None
    ) -> UnifiedQADIResult:
        """
        Run QADI cycle using configured strategy.

        Args:
            problem_statement: The question or problem to analyze
            context: Optional context from previous analyses
            cycle_config: Optional runtime configuration overrides

        Returns:
            UnifiedQADIResult: Results from the analysis

        Raises:
            ValueError: If strategy is unsupported
        """
        # Strategy dispatch
        if self.config.strategy == Strategy.SIMPLE:
            return await self._run_simple_strategy(problem_statement, context)
        elif self.config.strategy == Strategy.MULTI_PERSPECTIVE:
            return await self._run_multi_perspective_strategy(problem_statement, context)
        elif self.config.strategy == Strategy.SMART:
            return await self._run_smart_strategy(problem_statement, context, cycle_config)
        else:
            raise ValueError(f"Unsupported strategy: {self.config.strategy}")

    async def _run_simple_strategy(
        self,
        problem_statement: str,
        context: Optional[str] = None
    ) -> UnifiedQADIResult:
        """
        Run Simple QADI strategy.

        Delegates to SimpleQADIOrchestrator for execution.

        Args:
            problem_statement: The question to analyze
            context: Optional context

        Returns:
            UnifiedQADIResult: Results converted from SimpleQADIResult
        """
        # Create SimpleQADI orchestrator with configured parameters
        simple_orch = SimpleQADIOrchestrator(
            temperature_override=self.config.temperature_override,
            num_hypotheses=self.config.num_hypotheses
        )

        # Run QADI cycle
        simple_result = await simple_orch.run_qadi_cycle(
            problem_statement,
            context=context,
            max_retries=self.config.timeout_config.max_retries
        )

        # Convert to unified result
        return self._convert_simple_result(simple_result)

    def _convert_simple_result(self, simple_result: SimpleQADIResult) -> UnifiedQADIResult:
        """
        Convert SimpleQADIResult to UnifiedQADIResult.

        Args:
            simple_result: Result from SimpleQADIOrchestrator

        Returns:
            UnifiedQADIResult: Unified result structure
        """
        return UnifiedQADIResult(
            strategy_used=self.config.strategy,
            execution_mode=self.config.execution_mode,
            core_question=simple_result.core_question,
            hypotheses=simple_result.hypotheses,
            hypothesis_scores=simple_result.hypothesis_scores,
            final_answer=simple_result.final_answer,
            action_plan=simple_result.action_plan,
            verification_examples=simple_result.verification_examples,
            verification_conclusion=simple_result.verification_conclusion,
            total_llm_cost=simple_result.total_llm_cost,
            phase_results=simple_result.phase_results,
            synthesized_ideas=simple_result.synthesized_ideas,
            execution_metadata={}
        )

    async def _run_multi_perspective_strategy(
        self,
        problem_statement: str,
        context: Optional[str] = None
    ) -> UnifiedQADIResult:
        """
        Run Multi-Perspective strategy.

        To be implemented in next phase.

        Args:
            problem_statement: The question to analyze
            context: Optional context

        Returns:
            UnifiedQADIResult: Results from multi-perspective analysis

        Raises:
            NotImplementedError: Not yet implemented
        """
        raise NotImplementedError("Multi-perspective strategy not yet implemented")

    async def _run_smart_strategy(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None
    ) -> UnifiedQADIResult:
        """
        Run Smart strategy with agent selection.

        To be implemented in next phase.

        Args:
            problem_statement: The question to analyze
            context: Optional context
            cycle_config: Optional runtime configuration

        Returns:
            UnifiedQADIResult: Results from smart orchestration

        Raises:
            NotImplementedError: Not yet implemented
        """
        raise NotImplementedError("Smart strategy not yet implemented")
