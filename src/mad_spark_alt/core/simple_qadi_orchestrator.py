"""
Simplified QADI Orchestrator for Hypothesis-Driven Analysis

This module implements the true QADI methodology without unnecessary complexity.
No prompt classification, no adaptive prompts - just pure hypothesis-driven consulting.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import (
    GeneratedIdea,
    ThinkingMethod,
)
from .llm_provider import LLMRequest, llm_manager
from .qadi_prompts import PHASE_HYPERPARAMETERS, QADIPrompts, calculate_hypothesis_score

logger = logging.getLogger(__name__)


@dataclass
class HypothesisScore:
    """Scores for a single hypothesis."""

    novelty: float
    impact: float
    cost: float
    feasibility: float
    risks: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "novelty": self.novelty,
            "impact": self.impact,
            "cost": self.cost,
            "feasibility": self.feasibility,
            "risks": self.risks,
            "overall": self.overall,
        }


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

    def __init__(self, temperature_override: Optional[float] = None) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for abduction phase (0.0-2.0)
        """
        self.prompts = QADIPrompts()
        if temperature_override is not None and not 0.0 <= temperature_override <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature_override = temperature_override

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
            core_question, questioning_cost = await self._run_questioning_phase(
                full_input,
                max_retries,
            )
            result.core_question = core_question
            result.total_llm_cost += questioning_cost
            result.phase_results["questioning"] = {
                "question": core_question,
                "cost": questioning_cost,
            }

            # Phase 2: Abduction - Generate hypotheses
            logger.info("Running Abduction phase")
            hypotheses, abduction_cost = await self._run_abduction_phase(
                full_input,
                core_question,
                max_retries,
            )
            result.hypotheses = hypotheses
            result.total_llm_cost += abduction_cost
            result.phase_results["abduction"] = {"hypotheses": hypotheses}

            # Convert hypotheses to GeneratedIdea objects for evolution compatibility
            for i, hypothesis in enumerate(hypotheses):
                result.synthesized_ideas.append(
                    GeneratedIdea(
                        content=hypothesis,
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name="SimpleQADIOrchestrator",
                        generation_prompt=f"Hypothesis {i+1} for: {core_question}",
                        confidence_score=0.8,  # Default high confidence for hypotheses
                        reasoning="Generated as potential answer to core question",
                        metadata={"hypothesis_index": i},
                    ),
                )

            # Phase 3: Deduction - Evaluate and conclude
            logger.info("Running Deduction phase")
            deduction_result = await self._run_deduction_phase(
                full_input,
                core_question,
                hypotheses,
                max_retries,
            )
            result.hypothesis_scores = deduction_result["scores"]
            result.final_answer = deduction_result["answer"]
            result.action_plan = deduction_result["action_plan"]
            result.total_llm_cost += deduction_result["cost"]
            result.phase_results["deduction"] = deduction_result

            # Phase 4: Induction - Verify answer
            logger.info("Running Induction phase")
            induction_result = await self._run_induction_phase(
                full_input,
                core_question,
                result.final_answer,
                max_retries,
            )
            result.verification_examples = induction_result["examples"]
            result.verification_conclusion = induction_result["conclusion"]
            result.total_llm_cost += induction_result["cost"]
            result.phase_results["induction"] = induction_result

        except Exception:
            logger.exception("QADI cycle failed")
            raise

        return result

    async def _run_questioning_phase(
        self,
        user_input: str,
        max_retries: int,
    ) -> Tuple[str, float]:
        """Extract the core question from user input.

        Returns:
            Tuple of (core_question, total_cost)
        """
        prompt = self.prompts.get_questioning_prompt(user_input)
        hyperparams = PHASE_HYPERPARAMETERS["questioning"]
        total_cost = 0.0

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                )

                response = await llm_manager.generate(request)
                total_cost += response.cost

                # Extract the core question
                content = response.content.strip()
                match = re.search(r"Q:\s*(.+)", content)
                if match:
                    return match.group(1).strip(), total_cost
                # Fallback: use the whole response if no Q: prefix
                return content.replace("Q:", "").strip(), total_cost

            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning("Question phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to extract core question")

    async def _run_abduction_phase(
        self,
        user_input: str,
        core_question: str,
        max_retries: int,
    ) -> Tuple[List[str], float]:
        """Generate hypotheses to answer the core question."""
        prompt = self.prompts.get_abduction_prompt(user_input, core_question)
        hyperparams = PHASE_HYPERPARAMETERS["abduction"].copy()

        # Apply temperature override if provided
        if self.temperature_override is not None:
            hyperparams["temperature"] = self.temperature_override

        total_cost = 0.0

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.95),
                )

                response = await llm_manager.generate(request)
                total_cost += response.cost

                # Extract hypotheses
                hypotheses = []
                content = response.content.strip()

                # Look for H1:, H2:, H3: patterns
                for i in range(1, 4):
                    # Handle H3 specially since there's no H4
                    if i < 3:
                        pattern = rf"H{i}:\s*(.+?)(?=H{i+1}:|$)"
                    else:
                        pattern = rf"H{i}:\s*(.+?)$"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        hypothesis = match.group(1).strip()
                        hypotheses.append(hypothesis)

                if len(hypotheses) >= 2:  # At least 2 hypotheses
                    return hypotheses, total_cost

            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning("Abduction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to generate hypotheses")

    async def _run_deduction_phase(
        self,
        user_input: str,
        core_question: str,
        hypotheses: List[str],
        max_retries: int,
    ) -> Dict[str, Any]:
        """Evaluate hypotheses and determine the answer."""
        # Format hypotheses for the prompt
        hypotheses_text = "\n".join([f"H{i+1}: {h}" for i, h in enumerate(hypotheses)])

        prompt = self.prompts.get_deduction_prompt(
            user_input,
            core_question,
            hypotheses_text,
        )
        hyperparams = PHASE_HYPERPARAMETERS["deduction"]

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                )

                response = await llm_manager.generate(request)
                content = response.content.strip()

                # Parse the evaluation scores
                scores = []
                for i in range(len(hypotheses)):
                    score = self._parse_hypothesis_scores(content, i + 1)
                    scores.append(score)

                # Extract the answer
                answer_match = re.search(
                    r"ANSWER:\s*(.+?)(?=Action Plan:|$)", content, re.DOTALL
                )
                answer = answer_match.group(1).strip() if answer_match else ""

                # Extract action plan
                action_plan = []
                plan_match = re.search(r"Action Plan:\s*(.+?)$", content, re.DOTALL)
                if plan_match:
                    plan_text = plan_match.group(1).strip()
                    # Extract numbered items or bullet points
                    plan_items = re.findall(
                        r"(?:\d+\.|[-*•])\s*(.+?)(?=(?:\d+\.|[-*•])|$)",
                        plan_text,
                        re.DOTALL,
                    )
                    action_plan = [item.strip() for item in plan_items]

                return {
                    "scores": scores,
                    "answer": answer,
                    "action_plan": action_plan,
                    "cost": response.cost,
                    "raw_content": content,
                }

            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning("Deduction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to evaluate hypotheses")

    def _parse_hypothesis_scores(
        self, content: str, hypothesis_num: int
    ) -> HypothesisScore:
        """Parse scores for a specific hypothesis from deduction content."""
        # Look for the hypothesis section (handle both "H1:" and "- H1:" formats)
        pattern = rf"(?:^|-)\s*H{hypothesis_num}:(.*?)(?=(?:^|-)\s*H{hypothesis_num + 1}:|ANSWER:|$)"
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)

        if not match:
            # Log warning and return default scores if parsing fails
            logger.warning("Failed to parse scores for hypothesis %d", hypothesis_num)
            return HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

        section = match.group(1)

        # Extract individual scores
        def extract_score(criterion: str, text: str) -> float:
            pattern = rf"{criterion}:\s*([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    return 0.5
            return 0.5

        scores = {
            "novelty": extract_score("Novelty", section),
            "impact": extract_score("Impact", section),
            "cost": extract_score("Cost", section),
            "feasibility": extract_score("Feasibility", section),
            "risks": extract_score("Risks", section),
        }

        # Calculate overall score
        overall = calculate_hypothesis_score(scores)

        return HypothesisScore(
            novelty=scores["novelty"],
            impact=scores["impact"],
            cost=scores["cost"],
            feasibility=scores["feasibility"],
            risks=scores["risks"],
            overall=overall,
        )

    async def _run_induction_phase(
        self,
        user_input: str,
        core_question: str,
        answer: str,
        max_retries: int,
    ) -> Dict[str, Any]:
        """Verify the answer with examples."""
        prompt = self.prompts.get_induction_prompt(user_input, core_question, answer)
        hyperparams = PHASE_HYPERPARAMETERS["induction"]

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                )

                response = await llm_manager.generate(request)
                content = response.content.strip()

                # Extract verification examples
                examples = []
                for i in range(1, 4):
                    pattern = rf"{i}\.\s*(.+?)(?={i+1}\.|Conclusion:|$)"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        examples.append(match.group(1).strip())

                # Extract conclusion
                conclusion_match = re.search(
                    r"Conclusion:\s*(.+?)$", content, re.DOTALL
                )
                conclusion = (
                    conclusion_match.group(1).strip() if conclusion_match else ""
                )

                return {
                    "examples": examples,
                    "conclusion": conclusion,
                    "cost": response.cost,
                    "raw_content": content,
                }

            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning("Induction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to verify answer")
