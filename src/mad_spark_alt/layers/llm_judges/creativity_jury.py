"""
Multi-judge creativity evaluation jury.

Implements consensus-based creativity evaluation using multiple AI models
to improve reliability and reduce individual model biases.
"""

import asyncio
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

from ...core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    OutputType,
)
from .creativity_judge import CreativityLLMJudge

logger = logging.getLogger(__name__)


class CreativityJury(EvaluatorInterface):
    """
    Multi-judge creativity evaluator using consensus mechanisms.
    
    This evaluator uses multiple AI models to assess creativity and applies
    voting/consensus mechanisms to handle disagreements and improve reliability.
    """
    
    def __init__(
        self, 
        models: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the creativity jury.
        
        Args:
            models: List of LLM models to use as judges
            config: Additional configuration options
        """
        self.models = models or ["gpt-4", "claude-3-sonnet", "mock-model"]
        self.config = config or {}
        
        # Create individual judges
        self.judges = []
        for model in self.models:
            try:
                judge = CreativityLLMJudge(model, config)
                self.judges.append(judge)
            except Exception as e:
                logger.warning(f"Failed to create judge for {model}: {e}")
        
        if not self.judges:
            logger.error("No judges available for jury evaluation")
        
        # Consensus configuration
        self.min_agreement_threshold = self.config.get("min_agreement", 0.7)
        self.disagreement_threshold = self.config.get("disagreement_threshold", 0.3)
        self.enable_tie_breaking = self.config.get("enable_tie_breaking", True)
    
    @property
    def name(self) -> str:
        model_names = "_".join([j.model.replace("-", "_") for j in self.judges])
        return f"creativity_jury_{model_names}"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.LLM_JUDGE
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE, OutputType.STRUCTURED]
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        valid_keys = {
            "min_agreement", "disagreement_threshold", "enable_tie_breaking",
            "temperature", "max_tokens", "consensus_method"
        }
        return all(key in valid_keys for key in config.keys())
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate creativity using multiple judges and consensus."""
        if not self.judges:
            return self._create_no_judges_results(request)
        
        logger.info(f"Starting jury evaluation with {len(self.judges)} judges")
        
        results = []
        
        for i, output in enumerate(request.outputs):
            if output.output_type not in self.supported_output_types:
                results.append(self._create_unsupported_result(output))
                continue
            
            # Get evaluations from all judges
            jury_result = await self._evaluate_with_jury(output, request)
            results.append(jury_result)
        
        return results
    
    async def _evaluate_with_jury(
        self, 
        output: Any, 
        request: EvaluationRequest
    ) -> EvaluationResult:
        """Evaluate a single output using the full jury."""
        # Create evaluation tasks for each judge
        single_output_request = EvaluationRequest(
            outputs=[output],
            evaluation_config=request.evaluation_config,
            target_layers=request.target_layers,
            task_context=request.task_context
        )
        
        judge_tasks = []
        for judge in self.judges:
            task = judge.evaluate(single_output_request)
            judge_tasks.append(task)
        
        # Execute all judge evaluations concurrently
        judge_results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        
        # Process results from each judge
        valid_evaluations = []
        for i, result in enumerate(judge_results):
            if isinstance(result, Exception):
                logger.warning(f"Judge {self.judges[i].name} failed: {result}")
                continue
            
            if result and len(result) > 0 and result[0].scores:
                valid_evaluations.append(result[0])
            else:
                logger.warning(f"Judge {self.judges[i].name} returned empty result")
        
        if not valid_evaluations:
            return self._create_error_result("All judges failed to evaluate")
        
        # Apply consensus mechanism
        return self._apply_consensus(valid_evaluations, output)
    
    def _apply_consensus(
        self, 
        evaluations: List[EvaluationResult], 
        output: Any
    ) -> EvaluationResult:
        """Apply consensus mechanism to multiple judge evaluations."""
        logger.info(f"Applying consensus to {len(evaluations)} judge evaluations")
        
        # Collect scores for each dimension
        dimension_scores = {}
        all_explanations = []
        judge_metadata = []
        
        for evaluation in evaluations:
            for dimension, score in evaluation.scores.items():
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                dimension_scores[dimension].append(score)
            
            all_explanations.append(evaluation.explanations)
            judge_metadata.append({
                "judge": evaluation.evaluator_name,
                "scores": evaluation.scores
            })
        
        # Calculate consensus scores
        consensus_scores = {}
        disagreement_info = {}
        
        for dimension, scores in dimension_scores.items():
            if not scores:
                continue
            
            # Calculate statistics
            mean_score = statistics.mean(scores)
            median_score = statistics.median(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            # Check for disagreement
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score
            
            is_disagreement = score_range > self.disagreement_threshold
            
            # Choose consensus method
            consensus_method = self.config.get("consensus_method", "median")
            if consensus_method == "median":
                consensus_score = median_score
            elif consensus_method == "mean":
                consensus_score = mean_score
            else:  # weighted or other methods
                consensus_score = median_score  # fallback
            
            consensus_scores[dimension] = consensus_score
            
            disagreement_info[dimension] = {
                "range": score_range,
                "std_dev": std_dev,
                "disagreement": is_disagreement,
                "individual_scores": scores,
                "method_used": consensus_method
            }
        
        # Build consensus explanations
        consensus_explanations = self._build_consensus_explanations(
            all_explanations, disagreement_info
        )
        
        # Build metadata
        metadata = {
            "jury_size": len(evaluations),
            "consensus_method": self.config.get("consensus_method", "median"),
            "disagreement_analysis": disagreement_info,
            "judge_details": judge_metadata,
            "models": [judge["judge"] for judge in judge_metadata]
        }
        
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores=consensus_scores,
            explanations=consensus_explanations,
            metadata=metadata
        )
    
    def _build_consensus_explanations(
        self,
        all_explanations: List[Dict[str, str]],
        disagreement_info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Build explanations from jury consensus."""
        explanations = {}
        
        # Aggregate rationales
        rationales = []
        strengths = []
        weaknesses = []
        
        for exp in all_explanations:
            if "rationale" in exp and exp["rationale"]:
                rationales.append(exp["rationale"])
            if "strengths" in exp and exp["strengths"]:
                strengths.extend(exp["strengths"].split("; "))
            if "weaknesses" in exp and exp["weaknesses"]:
                weaknesses.extend(exp["weaknesses"].split("; "))
        
        # Create consensus rationale
        if rationales:
            explanations["consensus_rationale"] = (
                f"Jury of {len(all_explanations)} judges evaluated this content. "
                f"Key insights: {'; '.join(rationales[:2])}..."  # Truncate for brevity
            )
        
        # Aggregate strengths/weaknesses
        if strengths:
            # Remove duplicates and take most common
            unique_strengths = list(set(strengths))
            explanations["consensus_strengths"] = "; ".join(unique_strengths[:3])
        
        if weaknesses:
            unique_weaknesses = list(set(weaknesses))
            explanations["consensus_weaknesses"] = "; ".join(unique_weaknesses[:3])
        
        # Add disagreement analysis
        high_disagreement_dims = [
            dim for dim, info in disagreement_info.items()
            if info.get("disagreement", False)
        ]
        
        if high_disagreement_dims:
            explanations["disagreement_notice"] = (
                f"Significant disagreement among judges on: {', '.join(high_disagreement_dims)}. "
                f"Consensus scores represent median values."
            )
        else:
            explanations["consensus_quality"] = "High agreement among judges across all dimensions."
        
        return explanations
    
    def _create_no_judges_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results when no judges are available."""
        results = []
        for output in request.outputs:
            if output.output_type in self.supported_output_types:
                results.append(EvaluationResult(
                    evaluator_name=self.name,
                    layer=self.layer,
                    scores={},
                    explanations={"error": "No judges available for jury evaluation"},
                    metadata={"jury_size": 0, "models": []}
                ))
        return results
    
    def _create_unsupported_result(self, output: Any) -> EvaluationResult:
        """Create result for unsupported output types."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"info": f"Output type {output.output_type} not supported by jury"},
            metadata={"jury_size": len(self.judges), "supported": False}
        )
    
    def _create_error_result(self, error_message: str) -> EvaluationResult:
        """Create error result."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"error": error_message},
            metadata={"jury_size": len(self.judges), "error": True}
        )