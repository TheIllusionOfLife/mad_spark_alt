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


# Constants for parsing LLM responses
QUESTION_PREFIX = "Q:"
HYPOTHESIS_PATTERN = r"^H([123]):\s*(.*)$"
ANSWER_PREFIX = "ANSWER:"
ACTION_PLAN_PREFIX = "Action Plan:"
CONCLUSION_PREFIX = "Conclusion:"


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
            result.phase_results["abduction"] = {
                "hypotheses": hypotheses,
                "cost": abduction_cost,
            }

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
                match = re.search(rf"{QUESTION_PREFIX}\s*(.+)", content)
                if match:
                    return match.group(1).strip(), total_cost
                # Fallback: use the whole response if no Q: prefix
                # Safe removal of prefix only from start of string
                if content.startswith(QUESTION_PREFIX):
                    return content[len(QUESTION_PREFIX) :].strip(), total_cost
                return content.strip(), total_cost

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to extract core question after %d attempts. "
                        "Last error: %s. Please check your LLM API configuration.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to extract core question after {max_retries + 1} attempts. "
                        f"Last error: {e}. Please check your LLM API configuration and try again."
                    )
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

                # Extract hypotheses using line-by-line parsing for robustness
                hypotheses = []
                content = response.content.strip()
                lines = content.split("\n")

                current_hypothesis = ""
                current_index = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if line starts with H1:, H2:, or H3:
                    hypothesis_match = re.match(HYPOTHESIS_PATTERN, line)
                    if hypothesis_match:
                        # Save previous hypothesis if we have one
                        if current_index is not None and current_hypothesis.strip():
                            hypotheses.append(current_hypothesis.strip())

                        # Start new hypothesis
                        current_index = int(hypothesis_match.group(1))
                        current_hypothesis = hypothesis_match.group(2)
                    elif current_index is not None:
                        # Continue building current hypothesis
                        current_hypothesis += " " + line

                # Don't forget the last hypothesis
                if current_index is not None and current_hypothesis.strip():
                    hypotheses.append(current_hypothesis.strip())

                if len(hypotheses) >= 2:  # At least 2 hypotheses
                    return hypotheses, total_cost

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to generate hypotheses after %d attempts. "
                        "Last error: %s. The LLM may not be responding correctly.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to generate hypotheses after {max_retries + 1} attempts. "
                        f"Last error: {e}. Please ensure your LLM API is working and try again."
                    )
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
                    rf"{ANSWER_PREFIX}\s*(.+?)(?={ACTION_PLAN_PREFIX}|$)",
                    content,
                    re.DOTALL,
                )
                answer = answer_match.group(1).strip() if answer_match else ""

                # Extract action plan
                action_plan = []
                plan_match = re.search(
                    rf"{ACTION_PLAN_PREFIX}\s*(.+?)$",
                    content,
                    re.DOTALL,
                )
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
                    logger.error(
                        "Failed to evaluate hypotheses after %d attempts. "
                        "Last error: %s. The evaluation process encountered issues.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to evaluate hypotheses after {max_retries + 1} attempts. "
                        f"Last error: {e}. The LLM may be having trouble with the evaluation format."
                    )
                logger.warning("Deduction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to evaluate hypotheses")

    def _parse_hypothesis_scores(
        self, content: str, hypothesis_num: int
    ) -> HypothesisScore:
        """Parse scores for a specific hypothesis from deduction content."""
        # Use line-by-line parsing to extract hypothesis section
        lines = content.split("\n")
        section_lines = []
        in_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is the start of our hypothesis section
            # Handle various formats: "H1:", "- H1:", "- **H1:**", etc.
            hypothesis_match = re.match(
                rf"^(?:-\s*)?(?:\*\*)?H{hypothesis_num}[:.](.*?)(?:\*\*)?$", line
            )
            if hypothesis_match:
                in_section = True
                section_lines.append(hypothesis_match.group(1).strip())
                continue

            # Check if we've reached the next hypothesis or end section
            if in_section:
                if re.match(
                    rf"^(?:-\s*)?(?:\*\*)?H{hypothesis_num + 1}[:.]", line,
                ) or line.startswith((ANSWER_PREFIX, ACTION_PLAN_PREFIX)):
                    break
                section_lines.append(line)

        if not section_lines:
            # Log warning and return default scores if parsing fails
            logger.warning(
                "Failed to parse scores for hypothesis %d. Using default scores. "
                "This may indicate the LLM response didn't follow the expected format. "
                "Content preview: %s...",
                hypothesis_num,
                content[:300] if len(content) > 300 else content,
            )
            return HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

        section = " ".join(section_lines)

        # Extract individual scores with improved robustness
        def extract_score(criterion: str, text: str) -> float:
            # Try multiple patterns to handle different formatting
            patterns = [
                rf"\*?\s*{criterion}:\s*(-?[0-9.]+)\s*-",  # "* Novelty: 0.8 - explanation"
                rf"\*?\s*{criterion}:\s*(-?[0-9.]+)",  # "* Novelty: 0.8" or "Novelty: 0.8"
                rf"{criterion}\s*-\s*(-?[0-9.]+)",  # "Novelty - 0.8"
                rf"{criterion}\s*:\s*(-?[0-9.]+)/?",  # "Novelty: 0.8/" or "Novelty: 0.8"
                rf"{criterion}\s*\((-?[0-9.]+)\)",  # "Novelty (0.8)"
                rf"{criterion}:\s*(-?[0-9.]+)/\d+",  # "Novelty: 8/10" style (take first number)
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        # Ensure score is between 0 and 1
                        return max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        continue

            # If no pattern matches, return default
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

                # Extract verification examples using line-by-line parsing
                examples = []
                lines = content.split("\n")

                current_example = ""
                current_index = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if line starts with "1.", "2.", or "3."
                    example_match = re.match(r"^([123])\.\s*(.*)$", line)
                    if example_match:
                        # Save previous example if we have one
                        if current_index is not None and current_example.strip():
                            examples.append(current_example.strip())

                        # Start new example
                        current_index = int(example_match.group(1))
                        current_example = example_match.group(2)
                    elif line.startswith(CONCLUSION_PREFIX):
                        # Save last example before conclusion
                        if current_index is not None and current_example.strip():
                            examples.append(current_example.strip())
                        break
                    elif current_index is not None:
                        # Continue building current example
                        current_example += " " + line

                # Don't forget the last example if no conclusion found
                # Only append if we didn't break on Conclusion: line
                else:
                    if current_index is not None and current_example.strip():
                        examples.append(current_example.strip())

                # Extract conclusion
                conclusion_match = re.search(
                    rf"{CONCLUSION_PREFIX}\s*(.+?)$",
                    content,
                    re.DOTALL,
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
                    logger.error(
                        "Failed to verify answer after %d attempts. "
                        "Last error: %s. The verification process could not complete.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to verify answer after {max_retries + 1} attempts. "
                        f"Last error: {e}. The system will proceed with unverified results."
                    )
                logger.warning("Induction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to verify answer")
