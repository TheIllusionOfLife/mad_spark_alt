"""
Main creativity evaluator orchestrator.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, cast

from .evaluation_utils import ScoreAggregator
from .interfaces import (
    AsyncEvaluatorInterface,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    ModelOutput,
    OutputType,
)
from .registry import registry

# Evaluation constants to avoid circular imports
QUANTITATIVE_LAYER_WEIGHT = 0.3
LLM_JUDGE_LAYER_WEIGHT = 0.5
HUMAN_LAYER_WEIGHT = 0.2
ZERO_SCORE = 0.0
EQUAL_WEIGHT_CREATIVITY = 0.33

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all layers."""

    request_id: str
    total_outputs: int
    total_evaluators: int
    execution_time: float
    layer_results: Dict[EvaluationLayer, List[EvaluationResult]] = field(
        default_factory=dict
    )
    aggregate_scores: Dict[str, float] = field(default_factory=dict)

    def get_overall_creativity_score(self) -> Optional[float]:
        """Calculate an overall creativity score across all evaluations."""
        if not self.aggregate_scores:
            return None

        # Weight layers differently (this is configurable)
        layer_weights = {
            EvaluationLayer.QUANTITATIVE: QUANTITATIVE_LAYER_WEIGHT,
            EvaluationLayer.LLM_JUDGE: LLM_JUDGE_LAYER_WEIGHT,
            EvaluationLayer.HUMAN: HUMAN_LAYER_WEIGHT,
        }

        total_score = ZERO_SCORE
        total_weight = ZERO_SCORE

        for layer, results in self.layer_results.items():
            if results:
                layer_score = sum(r.overall_score or 0 for r in results) / len(results)
                weight = layer_weights.get(layer, EQUAL_WEIGHT_CREATIVITY)
                total_score += layer_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else None


class CreativityEvaluator:
    """Main orchestrator for creativity evaluation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configure default settings
        self.max_concurrent_evaluations = self.config.get("max_concurrent", 5)
        self.timeout_seconds = self.config.get("timeout", 300)
        self.enable_caching = self.config.get("enable_caching", True)

    async def evaluate(
        self, request: EvaluationRequest, request_id: Optional[str] = None
    ) -> EvaluationSummary:
        """
        Evaluate creativity across all applicable layers.

        Args:
            request: The evaluation request
            request_id: Optional identifier for tracking

        Returns:
            Summary of evaluation results
        """
        start_time = time.time()
        request_id = request_id or f"eval_{int(start_time)}"

        self.logger.info(
            f"Starting evaluation {request_id} with {len(request.outputs)} outputs"
        )

        # Determine which layers to evaluate
        target_layers = request.target_layers or list(EvaluationLayer)

        # Get compatible evaluators for each output type
        evaluators_by_layer = self._get_evaluators_for_request(request, target_layers)

        # Execute evaluations by layer
        layer_results = {}
        for layer in target_layers:
            if layer in evaluators_by_layer:
                layer_results[layer] = await self._evaluate_layer(
                    layer, evaluators_by_layer[layer], request
                )

        # Create summary
        execution_time = time.time() - start_time
        total_evaluators = sum(
            len(evaluators) for evaluators in evaluators_by_layer.values()
        )

        summary = EvaluationSummary(
            request_id=request_id,
            total_outputs=len(request.outputs),
            total_evaluators=total_evaluators,
            execution_time=execution_time,
            layer_results=layer_results,
        )

        # Calculate aggregate scores
        summary.aggregate_scores = self._calculate_aggregate_scores(layer_results)

        self.logger.info(
            f"Completed evaluation {request_id} in {execution_time:.2f}s "
            f"with {len(layer_results)} layers"
        )

        return summary

    async def evaluate_single_output(
        self,
        output: ModelOutput,
        evaluator_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> EvaluationSummary:
        """
        Convenience method to evaluate a single output.

        Args:
            output: The model output to evaluate
            evaluator_names: Optional list of specific evaluators to use
            **kwargs: Additional arguments for EvaluationRequest

        Returns:
            Evaluation summary
        """
        request = EvaluationRequest(
            outputs=[output],
            evaluator_names=evaluator_names,
            **kwargs
        )

        return await self.evaluate(request)

    def _get_evaluators_for_request(
        self, request: EvaluationRequest, target_layers: List[EvaluationLayer]
    ) -> Dict[EvaluationLayer, List[EvaluatorInterface]]:
        """Get evaluators compatible with the request outputs."""
        evaluators_by_layer = {}

        # Get unique output types in the request
        output_types = set(output.output_type for output in request.outputs)

        for layer in target_layers:
            layer_evaluators = []

            # If specific evaluators are requested, use only those
            if request.evaluator_names:
                for name in request.evaluator_names:
                    evaluator = registry.get_evaluator(name)
                    if evaluator and evaluator.layer == layer:
                        # Check if evaluator supports the output types
                        if any(ot in evaluator.supported_output_types for ot in output_types):
                            layer_evaluators.append(evaluator)
            else:
                # Otherwise, get all compatible evaluators
                for output_type in output_types:
                    compatible = registry.get_compatible_evaluators(
                        layer=layer, output_type=output_type
                    )
                    layer_evaluators.extend(compatible)

            # Remove duplicates while preserving order
            seen = set()
            unique_evaluators = []
            for evaluator in layer_evaluators:
                if evaluator.name not in seen:
                    unique_evaluators.append(evaluator)
                    seen.add(evaluator.name)

            if unique_evaluators:
                evaluators_by_layer[layer] = unique_evaluators

        return evaluators_by_layer

    async def _evaluate_layer(
        self,
        layer: EvaluationLayer,
        evaluators: List[EvaluatorInterface],
        request: EvaluationRequest,
    ) -> List[EvaluationResult]:
        """Evaluate a single layer with multiple evaluators."""
        self.logger.debug(
            f"Evaluating layer {layer.value} with {len(evaluators)} evaluators"
        )

        # Create evaluation tasks
        tasks = []
        for evaluator in evaluators:
            task = self._evaluate_with_timeout(evaluator, request)
            tasks.append(task)

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)

        async def bounded_task(task: Any) -> List[EvaluationResult]:
            async with semaphore:
                result = await task
                return cast(List[EvaluationResult], result)

        bounded_tasks = [bounded_task(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Process results
        layer_results = []
        for i, result in enumerate(results):
            evaluator_name = evaluators[i].name

            # Check if result is an exception (more robust for Python 3.13)
            is_exception = isinstance(result, BaseException)

            if is_exception:
                self.logger.error(f"Evaluator {evaluator_name} failed: {result}")
                # Create empty result for failed evaluator
                empty_result = EvaluationResult(
                    evaluator_name=evaluator_name,
                    layer=layer,
                    scores={},
                    explanations={"error": str(result)},
                    metadata={"failed": True},
                )
                layer_results.append(empty_result)
            elif isinstance(result, list):
                layer_results.extend(result)

        return layer_results

    async def _evaluate_with_timeout(
        self, evaluator: EvaluatorInterface, request: EvaluationRequest
    ) -> List[EvaluationResult]:
        """Evaluate with timeout protection."""
        try:
            return await asyncio.wait_for(
                evaluator.evaluate(request), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Evaluator {evaluator.name} timed out after {self.timeout_seconds}s"
            )

    def _calculate_aggregate_scores(
        self, layer_results: Dict[EvaluationLayer, List[EvaluationResult]]
    ) -> Dict[str, float]:
        """Calculate aggregate scores across all evaluations."""
        # Flatten all result scores into a list of dictionaries
        all_scores = []
        for results in layer_results.values():
            for result in results:
                if result.scores:
                    all_scores.append(result.scores)

        # Use ScoreAggregator to calculate aggregates
        return ScoreAggregator.aggregate_results(all_scores)
