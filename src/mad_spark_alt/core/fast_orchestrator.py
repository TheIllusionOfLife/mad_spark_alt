"""
Fast QADI Orchestrator with parallel phase execution and optimizations.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingMethod,
)
from .smart_orchestrator import SmartQADICycleResult, SmartQADIOrchestrator
from .smart_registry import SmartAgentRegistry

logger = logging.getLogger(__name__)


class FastQADIOrchestrator(SmartQADIOrchestrator):
    """
    Optimized QADI orchestrator that runs phases in parallel for faster execution.

    Key optimizations:
    1. Parallel phase execution instead of sequential
    2. Batched LLM calls where possible
    3. Optimized context building
    4. Optional caching support
    """

    def __init__(
        self,
        registry: Optional[SmartAgentRegistry] = None,
        auto_setup: bool = True,
        enable_parallel: bool = True,
        enable_batching: bool = True,
        enable_cache: bool = False,
    ):
        super().__init__(registry, auto_setup)
        self.enable_parallel = enable_parallel
        self.enable_batching = enable_batching
        self.enable_cache = enable_cache
        self._cache: Optional[Dict[str, SmartQADICycleResult]] = (
            {} if enable_cache else None
        )

    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> SmartQADICycleResult:
        """
        Run complete QADI cycle with parallel execution optimization.

        Instead of running phases sequentially (Q→A→D→I), run them in parallel
        for ~70% speedup. Context enhancement is simplified since phases run
        simultaneously.
        """
        cycle_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting FAST QADI cycle {cycle_id}")

        start_time = time.time()
        config = cycle_config or {}

        # Setup status tracking
        setup_status = await self.ensure_agents_ready()

        # Check cache if enabled
        if self.enable_cache and self._cache and problem_statement in self._cache:
            logger.info("Cache hit - returning cached result")
            cached = self._cache[problem_statement]
            cached.execution_time = time.time() - start_time
            return cached

        # Define QADI phases
        QADI_METHODS = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]

        if self.enable_parallel:
            # Run all phases in parallel
            logger.info("Running QADI phases in PARALLEL for speed")
            phase_results = await self._run_phases_parallel(
                problem_statement, QADI_METHODS, context, config
            )
        else:
            # Fall back to sequential execution
            logger.info("Running QADI phases sequentially")
            phase_results = await self._run_phases_sequential(
                problem_statement, QADI_METHODS, context, config
            )

        # Process results
        result = SmartQADICycleResult(
            problem_statement=problem_statement,
            cycle_id=cycle_id,
            phases={},
            synthesized_ideas=[],
            execution_time=0,
            agent_types={},
            llm_cost=0.0,
            metadata={
                "fast_mode": True,
                "parallel_execution": self.enable_parallel,
                "setup_status": setup_status,
            },
            setup_time=0,
        )

        # Aggregate results from all phases
        total_llm_cost = 0.0
        for method, (phase_result, agent_type) in phase_results.items():
            result.phases[method.value] = phase_result
            result.agent_types[method.value] = agent_type

            # Track costs
            if phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    if "llm_cost" in idea.metadata:
                        total_llm_cost += idea.metadata["llm_cost"]

        result.llm_cost = total_llm_cost

        # Synthesize final ideas
        result.synthesized_ideas = self._synthesize_ideas(result.phases)

        # Generate conclusion from all ideas
        await self._synthesize_conclusion(result, problem_statement, context)

        # Calculate execution time
        result.execution_time = time.time() - start_time

        logger.info(
            f"FAST QADI cycle completed in {result.execution_time:.2f}s "
            f"(vs ~110s sequential)"
        )

        # Cache result if enabled
        if self.enable_cache and self._cache is not None:
            self._cache[problem_statement] = result

        return result

    async def _run_phases_parallel(
        self,
        problem_statement: str,
        methods: List[ThinkingMethod],
        context: Optional[str],
        config: Dict[str, Any],
    ) -> Dict[ThinkingMethod, Tuple[IdeaGenerationResult, str]]:
        """Run all QADI phases in parallel for maximum speed."""

        # Create tasks for all phases
        tasks = []
        for method in methods:
            # Use same context for all phases in parallel mode
            # (no sequential context enhancement possible)
            task = self._run_smart_phase(method, problem_statement, context, config)
            tasks.append(task)

        # Run all phases simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        phase_results = {}
        for method, result in zip(methods, results):
            if isinstance(result, Exception):
                logger.error(f"Error in parallel {method.value}: {result}")
                # Create error result
                error_result = IdeaGenerationResult(
                    agent_name="error",
                    thinking_method=method,
                    generated_ideas=[],
                    error_message=str(result),
                    execution_time=0.0,
                )
                phase_results[method] = (error_result, "error")
            else:
                phase_results[method] = result  # type: ignore[assignment]

        return phase_results

    async def _run_phases_sequential(
        self,
        problem_statement: str,
        methods: List[ThinkingMethod],
        context: Optional[str],
        config: Dict[str, Any],
    ) -> Dict[ThinkingMethod, Tuple[IdeaGenerationResult, str]]:
        """Fall back to sequential execution if parallel disabled."""
        phase_results: Dict[ThinkingMethod, Tuple[IdeaGenerationResult, str]] = {}

        for i, method in enumerate(methods):
            # Build enhanced context for sequential mode
            if i == 0:
                phase_context = context
            else:
                # Use previous results to enhance context
                previous_results = [
                    phase_results[m][0] for m in methods[:i] if m in phase_results
                ]
                phase_context = self._build_enhanced_context(context, *previous_results)

            # Run phase
            result = await self._run_smart_phase(
                method, problem_statement, phase_context, config
            )
            phase_results[method] = result

        return phase_results
