"""
Simplified QADI Orchestrator for Hypothesis-Driven Analysis

This module implements the true QADI methodology without unnecessary complexity.
No prompt classification, no adaptive prompts - just pure hypothesis-driven consulting.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .interfaces import (
    GeneratedIdea,
    ThinkingMethod,
)
from .llm_provider import LLMRequest, LLMProviderInterface, OllamaProvider, llm_manager
from .system_constants import CONSTANTS
from .qadi_prompts import PHASE_HYPERPARAMETERS, QADIPrompts, calculate_hypothesis_score
from ..utils.text_cleaning import clean_ansi_codes
from .parsing_utils import HypothesisParser, ScoreParser, ActionPlanParser, ParsedScores
from .phase_logic import (
    HypothesisScore,
    PhaseInput,
    execute_questioning_phase,
    execute_abduction_phase,
    execute_deduction_phase,
    execute_induction_phase,
)

if TYPE_CHECKING:
    from .multimodal import MultimodalInput

logger = logging.getLogger(__name__)


@dataclass
class SimpleQADIResult:
    """Result from simplified QADI analysis."""

    # Core outputs
    core_question: str
    hypotheses: List[str]
    hypothesis_scores: List[HypothesisScore]
    final_answer: str
    action_plan: List[str]
    verification_examples: List[str]
    verification_conclusion: str

    # Metadata
    total_llm_cost: float = 0.0
    phase_results: Dict[str, Any] = field(default_factory=dict)

    # Multimodal metadata
    multimodal_metadata: Dict[str, Any] = field(default_factory=dict)
    total_images_processed: int = 0
    total_pages_processed: int = 0
    total_urls_processed: int = 0

    # For backward compatibility with evolution
    synthesized_ideas: List[GeneratedIdea] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to JSON-serializable dictionary.

        Returns:
            Dictionary containing all result data in serializable format
        """
        return {
            "core_question": self.core_question,
            "hypotheses": self.hypotheses,
            "hypothesis_scores": [score.to_dict() for score in self.hypothesis_scores],
            "final_answer": self.final_answer,
            "action_plan": self.action_plan,
            "verification_examples": self.verification_examples,
            "verification_conclusion": self.verification_conclusion,
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


class SimpleQADIOrchestrator:
    """
    Simplified QADI orchestrator implementing true hypothesis-driven methodology.

    Features:
    - Single universal prompt set
    - Phase-specific hyperparameters
    - User-adjustable creativity for hypothesis generation
    - Unified evaluation scoring
    """

    def __init__(
        self,
        temperature_override: Optional[float] = None,
        num_hypotheses: int = 3,
        llm_provider: Optional["LLMProviderInterface"] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for abduction phase (0.0-2.0)
            num_hypotheses: Number of hypotheses to generate in abduction phase (default: 3)
            llm_provider: Optional custom LLM provider (if None, uses global llm_manager default)
        """
        self.prompts = QADIPrompts()
        if temperature_override is not None and not 0.0 <= temperature_override <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature_override = temperature_override
        self.num_hypotheses = max(3, num_hypotheses)  # Ensure at least 3
        self.llm_provider = llm_provider  # Custom provider for this orchestrator instance

    async def run_qadi_cycle(
        self,
        user_input: str,
        context: Optional[str] = None,
        max_retries: int = 2,
        multimodal_inputs: Optional[List["MultimodalInput"]] = None,
        urls: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> SimpleQADIResult:
        """
        Run a complete QADI cycle on the user input.

        Args:
            user_input: The user's input (question, statement, topic, etc.)
            context: Optional additional context
            max_retries: Maximum retries per phase on failure
            multimodal_inputs: Optional multimodal inputs (images, documents)
            urls: Optional URLs for context retrieval
            tools: Optional provider-specific tools (e.g., Gemini url_context)

        Returns:
            SimpleQADIResult with all phases completed
        """
        # Combine user input with context if provided
        full_input = user_input
        if context:
            full_input = f"{user_input}\n\nContext: {context}"

        result = SimpleQADIResult(
            core_question="",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="",
        )

        try:
            # Use custom provider if provided, otherwise use global llm_manager
            manager_to_use = llm_manager
            if self.llm_provider is not None:
                # Create a temporary wrapper that routes to the custom provider
                from .llm_provider import LLMManager, LLMProvider
                temp_manager = LLMManager()

                # Determine provider enum based on instance type
                if isinstance(self.llm_provider, OllamaProvider):
                    provider_enum = LLMProvider.OLLAMA
                else:  # Assume GoogleProvider
                    provider_enum = LLMProvider.GOOGLE

                temp_manager.register_provider(provider_enum, self.llm_provider)
                manager_to_use = temp_manager

            # Phase 1: Question - Extract core question
            logger.info("Running Question phase")
            phase_input = PhaseInput(
                user_input=full_input,
                llm_manager=manager_to_use,
                context={},
                max_retries=max_retries,
                multimodal_inputs=multimodal_inputs,
                urls=urls,
                tools=tools,
            )
            questioning_result = await execute_questioning_phase(phase_input)
            result.core_question = questioning_result.core_question
            result.total_llm_cost += questioning_result.llm_cost
            result.phase_results["questioning"] = {
                "question": questioning_result.core_question,
                "cost": questioning_result.llm_cost,
            }

            # Track multimodal metadata from questioning phase
            result.multimodal_metadata["questioning"] = questioning_result.multimodal_metadata
            # Use max() not +=, as all phases process the same documents
            # A 10-page PDF processed by 4 phases is still 10 pages total, not 40
            # NOTE: This assumes all phases receive the same multimodal_inputs (current behavior)
            result.total_images_processed = max(
                result.total_images_processed,
                questioning_result.multimodal_metadata.get("images_processed", 0)
            )
            result.total_pages_processed = max(
                result.total_pages_processed,
                questioning_result.multimodal_metadata.get("pages_processed", 0)
            )

            # Phase 2: Abduction - Generate hypotheses
            logger.info("Running Abduction phase")
            abduction_result = await execute_abduction_phase(
                phase_input,
                questioning_result.core_question,
                num_hypotheses=self.num_hypotheses,
                temperature_override=self.temperature_override,
            )
            result.hypotheses = abduction_result.hypotheses
            result.total_llm_cost += abduction_result.llm_cost
            result.phase_results["abduction"] = {
                "hypotheses": abduction_result.hypotheses,
                "cost": abduction_result.llm_cost,
            }

            # Track multimodal metadata from abduction phase
            result.multimodal_metadata["abduction"] = abduction_result.multimodal_metadata
            result.total_images_processed = max(
                result.total_images_processed,
                abduction_result.multimodal_metadata.get("images_processed", 0)
            )
            result.total_pages_processed = max(
                result.total_pages_processed,
                abduction_result.multimodal_metadata.get("pages_processed", 0)
            )

            # Convert hypotheses to GeneratedIdea objects for evolution compatibility
            for i, hypothesis in enumerate(abduction_result.hypotheses):
                result.synthesized_ideas.append(
                    GeneratedIdea(
                        content=hypothesis,
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name="SimpleQADIOrchestrator",
                        generation_prompt=f"Hypothesis {i+1} for: {questioning_result.core_question}",
                        confidence_score=CONSTANTS.LLM.DEFAULT_HYPOTHESIS_CONFIDENCE,  # Default confidence for hypotheses
                        reasoning="Generated as potential answer to core question",
                        metadata={"hypothesis_index": i},
                    ),
                )

            # Phase 3: Deduction - Evaluate and conclude
            logger.info("Running Deduction phase")
            deduction_result = await execute_deduction_phase(
                phase_input,
                questioning_result.core_question,
                abduction_result.hypotheses,
            )
            result.hypothesis_scores = deduction_result.hypothesis_scores
            result.final_answer = deduction_result.answer
            result.action_plan = deduction_result.action_plan
            result.total_llm_cost += deduction_result.llm_cost
            result.phase_results["deduction"] = {
                "scores": deduction_result.hypothesis_scores,
                "answer": deduction_result.answer,
                "action_plan": deduction_result.action_plan,
                "cost": deduction_result.llm_cost,
                "used_parallel": deduction_result.used_parallel,
            }

            # Track multimodal metadata from deduction phase
            result.multimodal_metadata["deduction"] = deduction_result.multimodal_metadata
            result.total_images_processed = max(
                result.total_images_processed,
                deduction_result.multimodal_metadata.get("images_processed", 0)
            )
            result.total_pages_processed = max(
                result.total_pages_processed,
                deduction_result.multimodal_metadata.get("pages_processed", 0)
            )

            # Phase 4: Induction - Synthesize findings
            logger.info("Running Induction phase")
            induction_result = await execute_induction_phase(
                phase_input,
                questioning_result.core_question,
                deduction_result.answer,
                abduction_result.hypotheses,
                deduction_result=deduction_result,  # Pass full context for rich synthesis
            )
            result.verification_examples = induction_result.examples  # Backward compat (empty)
            result.verification_conclusion = induction_result.synthesis  # Use synthesis
            result.total_llm_cost += induction_result.llm_cost
            result.phase_results["induction"] = {
                "synthesis": induction_result.synthesis,
                "examples": induction_result.examples,  # Backward compat (empty)
                "conclusion": induction_result.conclusion,  # Alias to synthesis
                "cost": induction_result.llm_cost,
            }

            # Track multimodal metadata from induction phase
            result.multimodal_metadata["induction"] = induction_result.multimodal_metadata
            result.total_images_processed = max(
                result.total_images_processed,
                induction_result.multimodal_metadata.get("images_processed", 0)
            )
            result.total_pages_processed = max(
                result.total_pages_processed,
                induction_result.multimodal_metadata.get("pages_processed", 0)
            )

            # Track total URLs processed
            result.total_urls_processed = len(urls) if urls else 0

        except Exception as e:
            logger.exception("QADI cycle failed: %s", e)
            # Provide user-friendly error message
            if "API" in str(e) or "api" in str(e):
                raise RuntimeError(
                    "QADI cycle failed due to API issues. Please check:\n"
                    "1. Your API keys are correctly set\n"
                    "2. You have sufficient API credits\n"
                    "3. The API service is accessible\n"
                    f"Original error: {e}"
                )
            elif "timeout" in str(e).lower():
                raise RuntimeError(
                    "QADI cycle timed out. This might be due to:\n"
                    "1. Slow API response times\n"
                    "2. Complex input requiring more processing\n"
                    "Try again with simpler input or check your connection.\n"
                    f"Original error: {e}"
                )
            else:
                raise RuntimeError(
                    f"QADI cycle failed unexpectedly. Error: {e}\n"
                    "Please check the logs for more details."
                )

        return result

