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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .interfaces import (
    GeneratedIdea,
    ThinkingMethod,
)
from .llm_provider import LLMRequest, llm_manager
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

    # For backward compatibility with evolution
    synthesized_ideas: List[GeneratedIdea] = field(default_factory=list)


class SimpleQADIOrchestrator:
    """
    Simplified QADI orchestrator implementing true hypothesis-driven methodology.

    Features:
    - Single universal prompt set
    - Phase-specific hyperparameters
    - User-adjustable creativity for hypothesis generation
    - Unified evaluation scoring
    """

    def __init__(self, temperature_override: Optional[float] = None, num_hypotheses: int = 3) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for abduction phase (0.0-2.0)
            num_hypotheses: Number of hypotheses to generate in abduction phase (default: 3)
        """
        self.prompts = QADIPrompts()
        if temperature_override is not None and not 0.0 <= temperature_override <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature_override = temperature_override
        self.num_hypotheses = max(3, num_hypotheses)  # Ensure at least 3

    async def run_qadi_cycle(
        self,
        user_input: str,
        context: Optional[str] = None,
        max_retries: int = 2,
    ) -> SimpleQADIResult:
        """
        Run a complete QADI cycle on the user input.

        Args:
            user_input: The user's input (question, statement, topic, etc.)
            context: Optional additional context
            max_retries: Maximum retries per phase on failure

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
            # Phase 1: Question - Extract core question
            logger.info("Running Question phase")
            phase_input = PhaseInput(
                user_input=full_input,
                llm_manager=llm_manager,
                context={},
                max_retries=max_retries,
            )
            questioning_result = await execute_questioning_phase(phase_input)
            result.core_question = questioning_result.core_question
            result.total_llm_cost += questioning_result.llm_cost
            result.phase_results["questioning"] = {
                "question": questioning_result.core_question,
                "cost": questioning_result.llm_cost,
            }

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

            # Convert hypotheses to GeneratedIdea objects for evolution compatibility
            for i, hypothesis in enumerate(abduction_result.hypotheses):
                result.synthesized_ideas.append(
                    GeneratedIdea(
                        content=hypothesis,
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name="SimpleQADIOrchestrator",
                        generation_prompt=f"Hypothesis {i+1} for: {questioning_result.core_question}",
                        confidence_score=0.8,  # Default high confidence for hypotheses
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

            # Phase 4: Induction - Verify answer
            logger.info("Running Induction phase")
            induction_result = await execute_induction_phase(
                phase_input,
                questioning_result.core_question,
                deduction_result.answer,
                abduction_result.hypotheses,
            )
            result.verification_examples = induction_result.examples
            result.verification_conclusion = induction_result.conclusion
            result.total_llm_cost += induction_result.llm_cost
            result.phase_results["induction"] = {
                "examples": induction_result.examples,
                "conclusion": induction_result.conclusion,
                "cost": induction_result.llm_cost,
            }

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

