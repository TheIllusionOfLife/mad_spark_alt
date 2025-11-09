"""
UnifiedQADIOrchestrator - Single orchestrator supporting all strategies.

Consolidates Simple, Smart, and MultiPerspective strategies into one
configuration-driven implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .multimodal import MultimodalInput

# Default score for hypotheses when score object is unavailable
DEFAULT_HYPOTHESIS_SCORE = 0.5

from .orchestrator_config import (
    OrchestratorConfig,
    ExecutionMode,
    Strategy
)
from .simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult
from .multi_perspective_orchestrator import (
    MultiPerspectiveQADIOrchestrator,
    MultiPerspectiveQADIResult
)
from .intent_detector import QuestionIntent
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

    # Multimodal metadata
    multimodal_metadata: Dict[str, Any] = field(default_factory=dict)
    total_images_processed: int = 0
    total_pages_processed: int = 0
    total_urls_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.

        Returns:
            Dictionary containing all result data in serializable format
        """
        result_dict: Dict[str, Any] = {
            "strategy_used": self.strategy_used.value,
            "execution_mode": self.execution_mode.value,
            "core_question": self.core_question,
            "hypotheses": self.hypotheses,
            "final_answer": self.final_answer,
            "action_plan": self.action_plan,
            "metadata": {
                "total_llm_cost": self.total_llm_cost,
                "total_images_processed": self.total_images_processed,
                "total_pages_processed": self.total_pages_processed,
                "total_urls_processed": self.total_urls_processed,
            },
            "synthesized_ideas": [
                {
                    "content": idea.content,
                    "thinking_method": idea.thinking_method.value,
                    "confidence_score": idea.confidence_score if idea.confidence_score is not None else 0.5,
                }
                for idea in self.synthesized_ideas
            ],
        }

        # Add optional fields if present
        if self.hypothesis_scores is not None:
            result_dict["hypothesis_scores"] = [score.to_dict() for score in self.hypothesis_scores]

        if self.verification_examples is not None:
            result_dict["verification_examples"] = self.verification_examples

        if self.verification_conclusion is not None:
            result_dict["verification_conclusion"] = self.verification_conclusion

        if self.perspectives_used is not None:
            result_dict["perspectives_used"] = self.perspectives_used

        if self.synthesized_answer is not None:
            result_dict["synthesized_answer"] = self.synthesized_answer

        if self.agent_types is not None:
            result_dict["agent_types"] = self.agent_types

        return result_dict


class UnifiedQADIOrchestrator:
    """
    Single orchestrator supporting all execution modes and strategies
    via configuration.

    Strategies:
    - SIMPLE: Basic QADI cycle with hypothesis generation and evaluation
    - MULTI_PERSPECTIVE: Multi-perspective analysis with synthesis

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
        cycle_config: Optional[Dict[str, Any]] = None,
        multimodal_inputs: Optional[List["MultimodalInput"]] = None,
        urls: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UnifiedQADIResult:
        """
        Run QADI cycle using configured strategy.

        Args:
            problem_statement: The question or problem to analyze
            context: Optional context from previous analyses
            cycle_config: Optional runtime configuration overrides
            multimodal_inputs: Optional multimodal inputs (images, documents)
            urls: Optional URLs for context retrieval
            tools: Optional provider-specific tools (e.g., Gemini url_context)

        Returns:
            UnifiedQADIResult: Results from the analysis

        Raises:
            ValueError: If strategy is unsupported
        """
        # Strategy dispatch
        if self.config.strategy == Strategy.SIMPLE:
            return await self._run_simple_strategy(
                problem_statement, context, multimodal_inputs, urls, tools
            )
        elif self.config.strategy == Strategy.MULTI_PERSPECTIVE:
            return await self._run_multi_perspective_strategy(
                problem_statement, context, multimodal_inputs, urls, tools
            )
        else:
            supported = ", ".join(s.value for s in Strategy)
            raise ValueError(f"Unsupported strategy: {self.config.strategy}. Supported: {supported}")

    async def _run_simple_strategy(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        multimodal_inputs: Optional[List["MultimodalInput"]] = None,
        urls: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UnifiedQADIResult:
        """
        Run Simple QADI strategy.

        Delegates to SimpleQADIOrchestrator for execution.

        Args:
            problem_statement: The question to analyze
            context: Optional context
            multimodal_inputs: Optional multimodal inputs (images, documents)
            urls: Optional URLs for context retrieval
            tools: Optional provider-specific tools

        Returns:
            UnifiedQADIResult: Results converted from SimpleQADIResult
        """
        # Create SimpleQADI orchestrator with configured parameters
        simple_orch = SimpleQADIOrchestrator(
            temperature_override=self.config.temperature_override,
            num_hypotheses=self.config.num_hypotheses
        )

        # Run QADI cycle with multimodal support
        simple_result = await simple_orch.run_qadi_cycle(
            problem_statement,
            context=context,
            max_retries=self.config.timeout_config.max_retries,
            multimodal_inputs=multimodal_inputs,
            urls=urls,
            tools=tools,
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
            execution_metadata={},
            multimodal_metadata=simple_result.multimodal_metadata,
            total_images_processed=simple_result.total_images_processed,
            total_pages_processed=simple_result.total_pages_processed,
            total_urls_processed=simple_result.total_urls_processed,
        )

    def _convert_multi_perspective_result(
        self,
        mp_result: MultiPerspectiveQADIResult,
        problem_statement: str
    ) -> UnifiedQADIResult:
        """
        Convert MultiPerspectiveQADIResult to UnifiedQADIResult.

        Args:
            mp_result: Result from MultiPerspectiveQADIOrchestrator
            problem_statement: Original user question (for core_question)

        Returns:
            UnifiedQADIResult: Unified result structure
        """
        # Collect all hypotheses across perspectives with scores
        all_hypotheses_with_scores: List[Tuple[str, float, Optional[HypothesisScore]]] = []
        for perspective_result in mp_result.perspective_results:
            scores = perspective_result.result.hypothesis_scores or []
            for i, hypothesis in enumerate(perspective_result.result.hypotheses):
                score_obj = scores[i] if i < len(scores) else None
                score_value = score_obj.overall if score_obj is not None else DEFAULT_HYPOTHESIS_SCORE
                all_hypotheses_with_scores.append((hypothesis, score_value, score_obj))

        # Sort by score (descending) and take top N
        all_hypotheses_with_scores.sort(key=lambda entry: entry[1], reverse=True)
        top_n = self.config.num_hypotheses
        top_entries = all_hypotheses_with_scores[:top_n]
        top_hypotheses = [hypothesis for hypothesis, _, _ in top_entries]

        # Extract score objects, maintaining alignment with hypotheses
        # Only return scores if ALL hypotheses have scores (maintains 1:1 alignment)
        if not top_entries:
            top_scores: Optional[List[HypothesisScore]] = []
        elif all(score_obj is not None for _, _, score_obj in top_entries):
            top_scores = [score_obj for _, _, score_obj in top_entries]  # type: ignore
        else:
            # Some hypotheses lack scores - return None to avoid misalignment
            top_scores = None

        # Convert perspectives_used from QuestionIntent to strings
        perspectives_used = [
            intent.value for intent in mp_result.perspectives_used
        ]

        return UnifiedQADIResult(
            strategy_used=self.config.strategy,
            execution_mode=self.config.execution_mode,
            core_question=problem_statement,  # Use original problem statement
            hypotheses=top_hypotheses,
            hypothesis_scores=top_scores,
            final_answer=mp_result.synthesized_answer,
            action_plan=mp_result.synthesized_action_plan,
            total_llm_cost=mp_result.total_llm_cost,
            synthesized_ideas=mp_result.synthesized_ideas,
            perspectives_used=perspectives_used,
            synthesized_answer=mp_result.synthesized_answer,
            phase_results={
                "perspective_count": len(mp_result.perspective_results),
                "primary_intent": mp_result.primary_intent.value,
                "intent_confidence": mp_result.intent_confidence
            },
            execution_metadata={},
            multimodal_metadata=mp_result.multimodal_metadata,
            total_images_processed=mp_result.total_images_processed,
            total_pages_processed=mp_result.total_pages_processed,
            total_urls_processed=mp_result.total_urls_processed,
        )

    async def _run_multi_perspective_strategy(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        multimodal_inputs: Optional[List["MultimodalInput"]] = None,
        urls: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> UnifiedQADIResult:
        """
        Run Multi-Perspective strategy.

        Delegates to MultiPerspectiveQADIOrchestrator for execution.

        Args:
            problem_statement: The question to analyze
            context: Optional context (not used by MP orchestrator)
            multimodal_inputs: Optional multimodal inputs (images, documents)
            urls: Optional URLs for context retrieval
            tools: Optional provider-specific tools

        Returns:
            UnifiedQADIResult: Results converted from MultiPerspectiveQADIResult
        """
        # Create MultiPerspective orchestrator with configured parameters
        mp_orch = MultiPerspectiveQADIOrchestrator(
            temperature_override=self.config.temperature_override
        )

        # Convert string perspectives to QuestionIntent if provided
        # Note: Perspective validation already done in config.validate()
        force_perspectives = None
        if self.config.perspectives:
            force_perspectives = [
                QuestionIntent[p.upper()] for p in self.config.perspectives
            ]

        # Use configured max_perspectives
        max_perspectives = self.config.max_perspectives

        # Run multi-perspective analysis with multimodal support
        mp_result = await mp_orch.run_multi_perspective_analysis(
            problem_statement,
            max_perspectives=max_perspectives,
            force_perspectives=force_perspectives,
            multimodal_inputs=multimodal_inputs,
            urls=urls,
            tools=tools,
        )

        # Convert to unified result
        return self._convert_multi_perspective_result(mp_result, problem_statement)
