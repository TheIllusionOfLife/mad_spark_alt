"""
Unified Evaluator for Hypothesis Scoring

This module provides consistent evaluation across the QADI deduction phase
and the evolution fitness system using the same 5-criteria scoring.
"""

import asyncio
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm_provider import LLMRequest, llm_manager
from .qadi_prompts import EVALUATION_CRITERIA, calculate_hypothesis_score

logger = logging.getLogger(__name__)


@dataclass
class HypothesisEvaluation:
    """Complete evaluation of a hypothesis."""

    content: str
    scores: Dict[str, float]  # impact, feasibility, accessibility, sustainability, scalability
    overall_score: float
    explanations: Dict[str, str]
    metadata: Dict[str, Any]


class UnifiedEvaluator:
    """
    Unified evaluator that provides consistent scoring for hypotheses/ideas
    using the QADI 5-criteria system: impact, feasibility, accessibility, sustainability, scalability.
    """

    def __init__(self) -> None:
        """Initialize the unified evaluator."""
        self.criteria = EVALUATION_CRITERIA

    async def evaluate_hypothesis(
        self,
        hypothesis: str,
        context: str,
        core_question: Optional[str] = None,
        temperature: float = 0.3,
    ) -> HypothesisEvaluation:
        """
        Evaluate a single hypothesis using LLM-based scoring.

        Args:
            hypothesis: The hypothesis/idea to evaluate
            context: Context for evaluation (user input, problem statement)
            core_question: Optional core question being answered
            temperature: LLM temperature for evaluation (default: 0.3)

        Returns:
            HypothesisEvaluation with scores and explanations
        """
        prompt = self._build_evaluation_prompt(hypothesis, context, core_question)

        try:
            request = LLMRequest(
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=400,
                top_p=0.9,
            )

            response = await llm_manager.generate(request)
            scores, explanations = self._parse_evaluation_response(response.content)

            # Calculate overall score using unified criteria
            overall_score = calculate_hypothesis_score(scores)

            return HypothesisEvaluation(
                content=hypothesis,
                scores=scores,
                overall_score=overall_score,
                explanations=explanations,
                metadata={"llm_cost": response.cost, "model": response.model},
            )

        except Exception as e:
            logger.exception("Failed to evaluate hypothesis: %s", hypothesis[:50])
            # Return default scores on failure with informative message
            default_scores = {
                "impact": 0.5,
                "feasibility": 0.5,
                "accessibility": 0.5,
                "sustainability": 0.5,
                "scalability": 0.5,
            }
            error_msg = "Evaluation failed - using default score"
            if "rate limit" in str(e).lower():
                error_msg = "Rate limited - using default score"
            elif "api" in str(e).lower():
                error_msg = "API error - using default score"

            return HypothesisEvaluation(
                content=hypothesis,
                scores=default_scores,
                overall_score=0.5,
                explanations=dict.fromkeys(default_scores, error_msg),
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    async def evaluate_multiple(
        self,
        hypotheses: List[str],
        context: str,
        core_question: Optional[str] = None,
        parallel: bool = True,
        batch_size: Optional[int] = None,  # Deprecated and ignored
    ) -> List[HypothesisEvaluation]:
        """
        Evaluate multiple hypotheses using individual evaluations.

        Args:
            hypotheses: List of hypotheses to evaluate
            context: Context for evaluation
            core_question: Optional core question being answered
            parallel: Whether to evaluate in parallel
            batch_size: Deprecated and ignored. Kept for backward compatibility.

        Returns:
            List of HypothesisEvaluation objects
        """
        if batch_size is not None:
            warnings.warn(
                "'batch_size' is deprecated and has no effect. It will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if parallel:
            tasks = [
                self.evaluate_hypothesis(h, context, core_question) for h in hypotheses
            ]
            return await asyncio.gather(*tasks)

        results = []
        for hypothesis in hypotheses:
            result = await self.evaluate_hypothesis(hypothesis, context, core_question)
            results.append(result)
        return results

    def _build_evaluation_prompt(
        self,
        hypothesis: str,
        context: str,
        core_question: Optional[str] = None,
    ) -> str:
        """Build the evaluation prompt for a hypothesis."""
        question_context = f"\nCore Question: {core_question}" if core_question else ""

        return f"""As an analytical consultant, evaluate this hypothesis on a scale of 0.0 to 1.0 for each criterion.

Hypothesis: {hypothesis}

Context: {context}{question_context}

Score each criterion:
- Impact: What level of positive change will this create? (0=minimal, 1=transformative)
- Feasibility: How practical is implementation? (0=nearly impossible, 1=easily doable)
- Accessibility: How easily can people adopt this? (0=very difficult, 1=very easy)
- Sustainability: How well can this be maintained long-term? (0=unsustainable, 1=highly sustainable)
- Scalability: How well can this grow/expand? (0=doesn't scale, 1=scales excellently)

Format your response EXACTLY as:
Impact: [score] - [one line explanation]
Feasibility: [score] - [one line explanation]
Accessibility: [score] - [one line explanation]
Sustainability: [score] - [one line explanation]
Scalability: [score] - [one line explanation]
"""

    def _parse_evaluation_response(
        self,
        response: str,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Parse the LLM evaluation response into scores and explanations."""
        scores = {}
        explanations = {}

        criteria = ["impact", "feasibility", "accessibility", "sustainability", "scalability"]

        # Process line by line for more robust parsing
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check each criterion
            for criterion in criteria:
                # Match "Criterion: score - explanation" format (case insensitive)
                # Allow negative numbers but will clamp them to valid range
                pattern = rf"^{criterion}:\s*(-?[0-9.]+)\s*[-]\s*(.+)$"
                match = re.match(pattern, line, re.IGNORECASE)

                if match:
                    try:
                        score = float(match.group(1))
                        # Ensure score is between 0 and 1
                        score = max(0.0, min(1.0, score))
                        scores[criterion] = score
                        explanations[criterion] = match.group(2).strip()
                    except (ValueError, TypeError):
                        logger.warning(
                            "Failed to parse score for %s: %s",
                            criterion,
                            match.group(1),
                        )
                        scores[criterion] = 0.5
                        explanations[criterion] = "Failed to parse score"
                    break

        # Fill in any missing criteria with defaults
        expected_criteria = ["impact", "feasibility", "accessibility", "sustainability", "scalability"]
        for criterion in expected_criteria:
            if criterion not in scores:
                # Log debug message instead of warning for missing criteria
                logger.debug("No evaluation found for criterion: %s, using default", criterion)
                scores[criterion] = 0.5
                explanations[criterion] = "Not evaluated"

        return scores, explanations

    def calculate_fitness_from_evaluation(
        self, evaluation: HypothesisEvaluation
    ) -> float:
        """
        Calculate evolution fitness score from hypothesis evaluation.
        This ensures consistency between deduction scoring and evolution fitness.

        Args:
            evaluation: HypothesisEvaluation object

        Returns:
            Fitness score between 0.0 and 1.0
        """
        return evaluation.overall_score

    def get_best_hypothesis(
        self, evaluations: List[HypothesisEvaluation]
    ) -> HypothesisEvaluation:
        """
        Get the best hypothesis based on overall score.

        Args:
            evaluations: List of evaluated hypotheses

        Returns:
            The highest scoring hypothesis
        """
        return max(evaluations, key=lambda e: e.overall_score)

    def rank_hypotheses(
        self, evaluations: List[HypothesisEvaluation]
    ) -> List[HypothesisEvaluation]:
        """
        Rank hypotheses by overall score (highest first).

        Args:
            evaluations: List of evaluated hypotheses

        Returns:
            Sorted list from best to worst
        """
        return sorted(evaluations, key=lambda e: e.overall_score, reverse=True)

    def get_score_summary(
        self, evaluations: List[HypothesisEvaluation]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all evaluations.

        Args:
            evaluations: List of evaluated hypotheses

        Returns:
            Dictionary with min/max/avg for each criterion
        """
        if not evaluations:
            return {}

        summary = {}
        criteria = ["impact", "feasibility", "accessibility", "sustainability", "scalability", "overall"]

        for criterion in criteria:
            if criterion == "overall":
                values = [e.overall_score for e in evaluations]
            else:
                values = [e.scores.get(criterion, 0.0) for e in evaluations]

            summary[criterion] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return summary
    
    
    
    
