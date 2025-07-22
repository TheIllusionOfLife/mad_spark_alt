"""
Unified Evaluator for Hypothesis Scoring

This module provides consistent evaluation across the QADI deduction phase
and the evolution fitness system using the same 5-criteria scoring.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .llm_provider import llm_manager, LLMRequest
from .qadi_prompts import EVALUATION_CRITERIA, calculate_hypothesis_score

logger = logging.getLogger(__name__)


@dataclass
class HypothesisEvaluation:
    """Complete evaluation of a hypothesis."""
    content: str
    scores: Dict[str, float]  # novelty, impact, cost, feasibility, risks
    overall_score: float
    explanations: Dict[str, str]
    metadata: Dict[str, any]
    

class UnifiedEvaluator:
    """
    Unified evaluator that provides consistent scoring for hypotheses/ideas
    using the 5-criteria system: novelty, impact, cost, feasibility, risks.
    """
    
    def __init__(self):
        """Initialize the unified evaluator."""
        self.criteria = EVALUATION_CRITERIA
        
    async def evaluate_hypothesis(
        self,
        hypothesis: str,
        context: str,
        core_question: Optional[str] = None,
        temperature: float = 0.3
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
                top_p=0.9
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
                metadata={
                    "llm_cost": response.cost,
                    "model": response.model
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate hypothesis: {e}")
            # Return default scores on failure
            default_scores = {
                "novelty": 0.5,
                "impact": 0.5,
                "cost": 0.5,
                "feasibility": 0.5,
                "risks": 0.5
            }
            return HypothesisEvaluation(
                content=hypothesis,
                scores=default_scores,
                overall_score=0.5,
                explanations={k: "Evaluation failed" for k in default_scores},
                metadata={"error": str(e)}
            )
    
    async def evaluate_multiple(
        self,
        hypotheses: List[str],
        context: str,
        core_question: Optional[str] = None,
        parallel: bool = True
    ) -> List[HypothesisEvaluation]:
        """
        Evaluate multiple hypotheses.
        
        Args:
            hypotheses: List of hypotheses to evaluate
            context: Context for evaluation
            core_question: Optional core question being answered
            parallel: Whether to evaluate in parallel
            
        Returns:
            List of HypothesisEvaluation objects
        """
        if parallel:
            import asyncio
            tasks = [
                self.evaluate_hypothesis(h, context, core_question)
                for h in hypotheses
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for hypothesis in hypotheses:
                result = await self.evaluate_hypothesis(hypothesis, context, core_question)
                results.append(result)
            return results
    
    def _build_evaluation_prompt(
        self,
        hypothesis: str,
        context: str,
        core_question: Optional[str] = None
    ) -> str:
        """Build the evaluation prompt for a hypothesis."""
        question_context = f"\nCore Question: {core_question}" if core_question else ""
        
        return f"""As an analytical consultant, evaluate this hypothesis on a scale of 0.0 to 1.0 for each criterion.

Hypothesis: {hypothesis}

Context: {context}{question_context}

Score each criterion:
- Novelty: How innovative/unique is this approach? (0=common, 1=breakthrough)
- Impact: What level of positive change will this create? (0=minimal, 1=transformative)
- Cost: What resources required? (0=very expensive, 1=very cheap)
- Feasibility: How practical is implementation? (0=nearly impossible, 1=easily doable)
- Risks: What could go wrong? (0=high risk/many issues, 1=low risk/few issues)

Format your response EXACTLY as:
Novelty: [score] - [one line explanation]
Impact: [score] - [one line explanation]
Cost: [score] - [one line explanation]
Feasibility: [score] - [one line explanation]
Risks: [score] - [one line explanation]
"""
    
    def _parse_evaluation_response(
        self,
        response: str
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Parse the LLM evaluation response into scores and explanations."""
        scores = {}
        explanations = {}
        
        criteria = ["novelty", "impact", "cost", "feasibility", "risks"]
        
        for criterion in criteria:
            # Look for pattern like "Novelty: 0.7 - explanation"
            pattern = f'{criterion}:\\s*([0-9.]+)\\s*[-–]\\s*(.+?)(?={"|".join(criteria)}:|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                    scores[criterion] = score
                    explanations[criterion] = match.group(2).strip()
                except ValueError:
                    scores[criterion] = 0.5
                    explanations[criterion] = "Failed to parse score"
            else:
                scores[criterion] = 0.5
                explanations[criterion] = "Not evaluated"
        
        return scores, explanations
    
    def calculate_fitness_from_evaluation(
        self,
        evaluation: HypothesisEvaluation
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
        self,
        evaluations: List[HypothesisEvaluation]
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
        self,
        evaluations: List[HypothesisEvaluation]
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
        self,
        evaluations: List[HypothesisEvaluation]
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
        criteria = ["novelty", "impact", "cost", "feasibility", "risks", "overall"]
        
        for criterion in criteria:
            if criterion == "overall":
                values = [e.overall_score for e in evaluations]
            else:
                values = [e.scores.get(criterion, 0.0) for e in evaluations]
            
            summary[criterion] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
        
        return summary