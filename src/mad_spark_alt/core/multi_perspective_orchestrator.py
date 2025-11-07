"""
Multi-Perspective QADI Orchestrator

This module extends the QADI methodology to provide analysis from multiple
relevant perspectives based on question intent detection.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .intent_detector import IntentDetector, QuestionIntent
from .interfaces import GeneratedIdea, ThinkingMethod
from .llm_provider import LLMRequest, llm_manager
from .simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult, HypothesisScore

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

    This orchestrator delegates the QADI cycle execution to SimpleQADIOrchestrator
    instances, focusing on perspective coordination and synthesis.
    """

    def __init__(self, temperature_override: Optional[float] = None) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for hypothesis generation
        """
        self.intent_detector = IntentDetector()
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
        """
        Run QADI analysis from a single perspective.

        Delegates to SimpleQADIOrchestrator with perspective-augmented input.
        """
        try:
            logger.info(f"Running {perspective.value} perspective analysis")

            # Create SimpleQADI orchestrator instance
            orchestrator = SimpleQADIOrchestrator(
                temperature_override=self.temperature_override,
                num_hypotheses=3
            )

            # Augment question with perspective context
            perspective_question = f"From a {perspective.value} perspective: {user_input}"

            # Run QADI cycle (delegates to phase_logic)
            result = await orchestrator.run_qadi_cycle(perspective_question)

            return result

        except Exception as e:
            logger.error(f"Failed {perspective.value} perspective analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

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

        # Run synthesis LLM call
        request = LLMRequest(
            user_prompt=synthesis_prompt,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )

        response = await llm_manager.generate(request)
        response_text = response.content.strip()
        cost = response.cost

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
