"""
Multi-Perspective QADI Orchestrator

This module extends the QADI methodology to provide analysis from multiple
relevant perspectives based on question intent detection.
"""

import asyncio
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .intent_detector import IntentDetector, QuestionIntent
from .interfaces import GeneratedIdea, ThinkingMethod
from .llm_provider import LLMRequest, llm_manager
from .multi_perspective_prompts import MultiPerspectivePrompts
from .qadi_prompts import PHASE_HYPERPARAMETERS, calculate_hypothesis_score
from .simple_qadi_orchestrator import HypothesisScore, SimpleQADIResult

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveResult:
    """Result from a single perspective's QADI analysis."""

    perspective: QuestionIntent
    result: SimpleQADIResult
    relevance_score: float  # How relevant this perspective is


@dataclass
class MultiPerspectiveQADIResult:
    """Combined result from multi-perspective QADI analysis."""

    # Intent detection results
    primary_intent: QuestionIntent
    intent_confidence: float
    keywords_matched: List[str]

    # Perspective results
    perspective_results: List[PerspectiveResult]

    # Synthesized results
    synthesized_answer: str
    synthesized_action_plan: List[str]
    best_hypothesis: Tuple[str, QuestionIntent]  # (hypothesis, from_perspective)

    # Metadata
    total_llm_cost: float = 0.0
    perspectives_used: List[QuestionIntent] = field(default_factory=list)

    # For backward compatibility
    synthesized_ideas: List[GeneratedIdea] = field(default_factory=list)


class MultiPerspectiveQADIOrchestrator:
    """
    Orchestrator for multi-perspective QADI analysis.

    Detects question intent and runs QADI from multiple relevant perspectives,
    then synthesizes the results into a comprehensive answer.
    """

    def __init__(self, temperature_override: Optional[float] = None) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for hypothesis generation
        """
        self.intent_detector = IntentDetector()
        self.prompts = MultiPerspectivePrompts()
        self.temperature_override = temperature_override

    async def run_multi_perspective_analysis(
        self,
        user_input: str,
        max_perspectives: int = 3,
        force_perspectives: Optional[List[QuestionIntent]] = None,
    ) -> MultiPerspectiveQADIResult:
        """
        Run multi-perspective QADI analysis.

        Args:
            user_input: The user's question or input
            max_perspectives: Maximum number of perspectives to analyze
            force_perspectives: Optional list of perspectives to use (overrides detection)

        Returns:
            MultiPerspectiveQADIResult with analysis from multiple perspectives
        """
        # Detect intent if not forced
        if force_perspectives:
            perspectives = force_perspectives
            intent_result = self.intent_detector.detect_intent(user_input)
            primary_intent = perspectives[0] if perspectives else QuestionIntent.GENERAL
        else:
            intent_result = self.intent_detector.detect_intent(user_input)
            perspectives = self.intent_detector.get_recommended_perspectives(
                intent_result, max_perspectives
            )
            primary_intent = intent_result.primary_intent

        logger.info(
            f"Detected intent: {primary_intent} (confidence: {intent_result.confidence:.2f})"
        )
        logger.info(f"Using perspectives: {[p.value for p in perspectives]}")

        # Run QADI for each perspective in parallel
        perspective_tasks = []
        for perspective in perspectives:
            task = self._run_perspective_analysis(user_input, perspective)
            perspective_tasks.append(task)

        perspective_results = await asyncio.gather(*perspective_tasks)

        # Calculate relevance scores
        scored_results = []
        for i, (perspective, result) in enumerate(
            zip(perspectives, perspective_results)
        ):
            if result:  # Only include successful results
                relevance = 1.0 if i == 0 else 0.8 - (i * 0.1)  # Primary gets 1.0
                scored_results.append(PerspectiveResult(perspective, result, relevance))

        # Synthesize results
        synthesized = await self._synthesize_results(user_input, scored_results)

        # Calculate total cost
        total_cost = sum(pr.result.total_llm_cost for pr in scored_results)
        total_cost += synthesized.get("synthesis_cost", 0.0)

        # Create combined result
        return MultiPerspectiveQADIResult(
            primary_intent=primary_intent,
            intent_confidence=intent_result.confidence,
            keywords_matched=intent_result.keywords_matched,
            perspective_results=scored_results,
            synthesized_answer=synthesized["answer"],
            synthesized_action_plan=synthesized["action_plan"],
            best_hypothesis=synthesized["best_hypothesis"],
            total_llm_cost=total_cost,
            perspectives_used=perspectives,
            synthesized_ideas=self._collect_all_ideas(scored_results),
        )

    async def _run_perspective_analysis(
        self, user_input: str, perspective: QuestionIntent
    ) -> Optional[SimpleQADIResult]:
        """Run QADI analysis from a single perspective."""
        try:
            logger.info(f"Running {perspective.value} perspective analysis")

            result = SimpleQADIResult(
                core_question="",
                hypotheses=[],
                hypothesis_scores=[],
                final_answer="",
                action_plan=[],
                verification_examples=[],
                verification_conclusion="",
            )

            # Phase 1: Questioning
            prompt = self.prompts.get_questioning_prompt(user_input, perspective)
            question, cost = await self._run_llm_phase(prompt, "questioning")
            result.core_question = self._extract_question(question)
            result.total_llm_cost += cost

            # Phase 2: Abduction (Hypothesis Generation)
            prompt = self.prompts.get_abduction_prompt(
                user_input, result.core_question, perspective
            )
            hypotheses_text, cost = await self._run_llm_phase(
                prompt, "abduction", use_temperature_override=True
            )
            result.hypotheses = self._extract_hypotheses(hypotheses_text)
            result.total_llm_cost += cost

            # Phase 3: Deduction (Evaluation)
            if not result.hypotheses:
                logger.error(
                    f"No hypotheses generated for {perspective.value} perspective"
                )
                return None

            hypotheses_formatted = "\n".join(
                [f"H{i+1}: {h}" for i, h in enumerate(result.hypotheses)]
            )
            prompt = self.prompts.get_deduction_prompt(
                user_input, result.core_question, hypotheses_formatted, perspective
            )
            deduction_text, cost = await self._run_llm_phase(prompt, "deduction")

            # Parse deduction results
            deduction_parsed = self._parse_deduction_results(
                deduction_text, len(result.hypotheses)
            )
            result.hypothesis_scores = deduction_parsed["scores"]
            result.final_answer = deduction_parsed["answer"]
            result.action_plan = deduction_parsed["action_plan"]
            result.total_llm_cost += cost

            # Phase 4: Induction (Verification)
            prompt = self.prompts.get_induction_prompt(
                user_input, result.core_question, result.final_answer, perspective
            )
            induction_text, cost = await self._run_llm_phase(prompt, "induction")

            induction_parsed = self._parse_induction_results(induction_text)
            result.verification_examples = induction_parsed["examples"]
            result.verification_conclusion = induction_parsed["conclusion"]
            result.total_llm_cost += cost

            return result

        except Exception as e:
            logger.error(f"Failed {perspective.value} perspective analysis: {e}")

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _run_llm_phase(
        self,
        prompt: str,
        phase: str,
        use_temperature_override: bool = False,
    ) -> Tuple[str, float]:
        """Run a single LLM phase with appropriate parameters."""
        hyperparams = PHASE_HYPERPARAMETERS.get(
            phase, PHASE_HYPERPARAMETERS["questioning"]
        )

        # Apply temperature override for abduction if specified
        if (
            use_temperature_override
            and self.temperature_override is not None
            and phase == "abduction"
        ):
            temperature = self.temperature_override
        else:
            temperature = hyperparams["temperature"]

        request = LLMRequest(
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=int(hyperparams["max_tokens"]),
            top_p=hyperparams.get("top_p", 0.9),
        )

        response = await llm_manager.generate(request)
        return response.content.strip(), response.cost

    def _extract_question(self, text: str) -> str:
        """Extract core question from LLM response."""

        match = re.search(r"Q:\s*(.+)", text, re.DOTALL)
        if match:
            # Take only the first line of the match
            return match.group(1).strip().split("\n")[0].strip()

        # Fallback: use first non-empty line
        for line in text.split("\n"):
            if line.strip() and not line.strip().startswith(("Think about:", "-", "*")):
                return line.strip()

        return "What is the core challenge here?"

    def _extract_hypotheses(self, text: str) -> List[str]:
        """Extract hypotheses from LLM response."""

        hypotheses = []
        lines = text.split("\n")

        current_hypothesis = ""
        current_index = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with H1:, H2:, or H3:
            match = re.match(r"H([123]):\s*(.+)", line)
            if match:
                # Save previous hypothesis if we have one
                if current_index is not None and current_hypothesis.strip():
                    hypotheses.append(current_hypothesis.strip())

                # Start new hypothesis
                current_index = int(match.group(1))
                current_hypothesis = match.group(2)
            elif current_index is not None:
                # Continue building current hypothesis
                current_hypothesis += " " + line

        # Don't forget the last hypothesis
        if current_index is not None and current_hypothesis.strip():
            hypotheses.append(current_hypothesis.strip())

        # If no hypotheses found with H1: format, try numbered format
        if not hypotheses:
            for line in lines:
                match = re.match(r"^\d+\.\s*(.+)", line.strip())
                if match:
                    hypotheses.append(match.group(1).strip())

        return hypotheses[:3]  # Maximum 3 hypotheses

    def _parse_deduction_results(
        self, text: str, num_hypotheses: int
    ) -> Dict[str, Any]:
        """Parse deduction phase results."""

        # Extract scores for each hypothesis
        scores = []
        for i in range(num_hypotheses):
            score = self._extract_hypothesis_scores(text, i + 1)
            scores.append(score)

        # Extract answer
        answer_match = re.search(r"ANSWER:\s*(.+?)(?=Action Plan:|$)", text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        # Extract action plan
        action_plan = []
        plan_match = re.search(r"Action Plan:\s*(.+?)$", text, re.DOTALL)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            # Extract numbered items
            plan_items = re.findall(r"\d+\.\s*(.+?)(?=\d+\.|$)", plan_text, re.DOTALL)
            action_plan = [item.strip() for item in plan_items]

        return {
            "scores": scores,
            "answer": answer,
            "action_plan": action_plan,
        }

    def _extract_hypothesis_scores(
        self, text: str, hypothesis_num: int
    ) -> HypothesisScore:
        """Extract scores for a specific hypothesis."""

        # Find the hypothesis section
        pattern = rf"H{hypothesis_num}:.*?(?=H{hypothesis_num + 1}:|ANSWER:|$)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return HypothesisScore(
                impact=0.5,
                feasibility=0.5,
                accessibility=0.5,
                sustainability=0.5,
                scalability=0.5,
                overall=0.5,
            )

        section = match.group(0)

        # Extract individual scores - flexible matching
        def extract_score(patterns: List[str], text: str) -> float:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse score from LLM output: '%s'",
                            match.group(1),
                        )
                        pass
            return 0.5

        # Different patterns for different perspectives
        scores_dict = {}

        # Try to extract based on common patterns
        score_patterns = [
            r":\s*([0-9.]+)\s*-",  # ": 0.8 -"
            r":\s*([0-9.]+)",  # ": 0.8"
            r"\(([0-9.]+)\)",  # "(0.8)"
        ]

        # Map various criteria names to standard ones (matching HypothesisScore fields)
        criteria_mappings = {
            "impact": [
                "impact",
                "environmental impact",
                "personal impact",
                "market impact",
                "research impact",
                "societal benefit",
                "business value",
                "novelty",
                "innovation",
            ],
            "feasibility": [
                "feasibility",
                "implementation feasibility",
                "implementation complexity",
                "ease of adoption",
                "practical application",
                "implementation speed",
            ],
            "accessibility": [
                "accessibility",
                "universal applicability",
                "resource requirements",
                "cost",
                "resource efficiency",
                "time investment",
                "cost efficiency",
            ],
            "sustainability": [
                "sustainability",
                "long-term benefits",
                "maintainability",
                "ecosystem benefits",
                "human flourishing",
            ],
            "scalability": [
                "scalability",
                "growth potential",
                "scaling potential",
                "risks",
                "risk level",
                "peer acceptance",
            ],
        }

        # Extract scores with flexible matching
        for standard_key, variations in criteria_mappings.items():
            score = 0.5  # default
            for variation in variations:
                patterns = [f"{variation}{p}" for p in score_patterns]
                extracted = extract_score(patterns, section)
                if extracted != 0.5:
                    score = extracted
                    break
            scores_dict[standard_key] = score

        # Calculate overall
        overall = calculate_hypothesis_score(scores_dict)

        return HypothesisScore(
            impact=scores_dict.get("impact", 0.5),
            feasibility=scores_dict.get("feasibility", 0.5),
            accessibility=scores_dict.get("accessibility", 0.5),
            sustainability=scores_dict.get("sustainability", 0.5),
            scalability=scores_dict.get("scalability", 0.5),
            overall=overall,
        )

    def _parse_induction_results(self, text: str) -> Dict[str, Any]:
        """Parse induction phase results."""

        # Extract examples
        examples = []
        example_matches = re.findall(
            r"\d+\.\s*(.+?)(?=\d+\.|Conclusion:|$)", text, re.DOTALL
        )
        examples = [match.strip() for match in example_matches]

        # Extract conclusion
        conclusion_match = re.search(r"Conclusion:\s*(.+?)$", text, re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        return {
            "examples": examples,
            "conclusion": conclusion,
        }

    async def _synthesize_results(
        self, user_input: str, perspective_results: List[PerspectiveResult]
    ) -> Dict[str, Any]:
        """Synthesize results from multiple perspectives."""
        if not perspective_results:
            return {
                "answer": "Unable to generate analysis.",
                "action_plan": [],
                "best_hypothesis": ("No hypotheses generated", QuestionIntent.GENERAL),
                "synthesis_cost": 0.0,
            }

        # Find best hypothesis across all perspectives
        best_hypothesis = ("", QuestionIntent.GENERAL)
        best_score = -1.0

        for pr in perspective_results:
            for i, (hyp, score) in enumerate(
                zip(pr.result.hypotheses, pr.result.hypothesis_scores)
            ):
                weighted_score = score.overall * pr.relevance_score
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_hypothesis = (hyp, pr.perspective)

        # Create synthesis prompt
        perspectives_summary = "\n\n".join(
            [
                f"**{pr.perspective.value.title()} Perspective:**\n"
                f"Core Question: {pr.result.core_question}\n"
                f"Key Answer: {pr.result.final_answer[:200]}..."
                for pr in perspective_results
            ]
        )

        synthesis_prompt = f"""Based on multiple perspective analyses of the user's question, synthesize a comprehensive answer.

User's original question:
{user_input}

Perspectives analyzed:
{perspectives_summary}

Best hypothesis identified:
"{best_hypothesis[0]}" (from {best_hypothesis[1].value} perspective)

Create a synthesized answer that:
1. Addresses the user's question comprehensively
2. Integrates insights from all perspectives
3. Highlights the most practical and impactful recommendations
4. Is concise and actionable

Format:
SYNTHESIS: [Your comprehensive answer integrating all perspectives]

INTEGRATED ACTION PLAN:
1. [Most important action combining insights]
2. [Second priority action]
3. [Third priority action]
"""

        # Run synthesis
        response_text, cost = await self._run_llm_phase(synthesis_prompt, "questioning")

        # Parse synthesis

        synthesis_match = re.search(
            r"SYNTHESIS:\s*(.+?)(?=INTEGRATED ACTION PLAN:|$)", response_text, re.DOTALL
        )
        synthesized_answer = (
            synthesis_match.group(1).strip()
            if synthesis_match
            else perspectives_summary
        )

        # Extract integrated action plan
        action_plan = []
        plan_match = re.search(
            r"INTEGRATED ACTION PLAN:\s*(.+?)$", response_text, re.DOTALL
        )
        if plan_match:
            plan_text = plan_match.group(1).strip()
            plan_items = re.findall(r"\d+\.\s*(.+?)(?=\d+\.|$)", plan_text, re.DOTALL)
            action_plan = [item.strip() for item in plan_items]

        # If no synthesis, use best perspective's action plan
        if not action_plan and perspective_results:
            action_plan = perspective_results[0].result.action_plan[:3]

        return {
            "answer": synthesized_answer,
            "action_plan": action_plan,
            "best_hypothesis": best_hypothesis,
            "synthesis_cost": cost,
        }

    def _collect_all_ideas(
        self, perspective_results: List[PerspectiveResult]
    ) -> List[GeneratedIdea]:
        """Collect all generated ideas from all perspectives."""
        all_ideas = []

        for pr in perspective_results:
            for i, hyp in enumerate(pr.result.hypotheses):
                all_ideas.append(
                    GeneratedIdea(
                        content=hyp,
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name=f"MultiPerspective-{pr.perspective.value}",
                        generation_prompt=pr.result.core_question,
                        confidence_score=(
                            pr.result.hypothesis_scores[i].overall
                            if i < len(pr.result.hypothesis_scores)
                            else 0.5
                        ),
                        reasoning=f"Generated from {pr.perspective.value} perspective",
                        metadata={
                            "perspective": pr.perspective.value,
                            "hypothesis_index": i,
                            "relevance_score": pr.relevance_score,
                        },
                    )
                )

        return all_ideas
