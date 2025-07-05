"""
Human creativity evaluation interface.

Provides structured interfaces for collecting human assessments
of AI creativity from experts and target users.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from ...core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    OutputType,
)

logger = logging.getLogger(__name__)


@dataclass
class HumanEvaluationItem:
    """A single item for human evaluation."""
    
    content: str
    output_type: str
    prompt: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class HumanRating:
    """Human rating for creativity assessment."""
    
    novelty: float  # 1-10 scale
    usefulness: float  # 1-10 scale
    feasibility: float  # 1-10 scale
    overall_creativity: float  # 1-10 scale
    comments: str = ""
    evaluation_time_seconds: Optional[float] = None
    evaluator_id: Optional[str] = None
    evaluator_expertise: Optional[str] = None


class HumanCreativityEvaluator(EvaluatorInterface):
    """
    Human creativity evaluator with structured assessment interface.
    
    This evaluator facilitates human evaluation through various modalities:
    - Interactive console interface for immediate feedback
    - Batch evaluation file generation for offline assessment
    - Expert vs. target user evaluation workflows
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the human evaluator.
        
        Args:
            config: Configuration options including evaluation mode
        """
        self.config = config or {}
        self.evaluation_mode = self.config.get("mode", "interactive")  # interactive, batch, expert
        self.output_file = self.config.get("output_file", "human_evaluations.jsonl")
        self.input_file = self.config.get("input_file")
    
    @property
    def name(self) -> str:
        return f"human_creativity_evaluator_{self.evaluation_mode}"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.HUMAN
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE, OutputType.STRUCTURED]
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        valid_keys = {
            "mode", "output_file", "input_file", "evaluator_info",
            "rating_scale", "include_comparisons", "randomize_order"
        }
        valid_modes = {"interactive", "batch", "expert", "user_testing"}
        
        if "mode" in config and config["mode"] not in valid_modes:
            return False
        
        return all(key in valid_keys for key in config.keys())
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate creativity using human assessment."""
        logger.info(f"Starting human evaluation in {self.evaluation_mode} mode")
        
        if self.evaluation_mode == "interactive":
            return await self._interactive_evaluation(request)
        elif self.evaluation_mode == "batch":
            return await self._batch_evaluation(request)
        elif self.evaluation_mode == "expert":
            return await self._expert_evaluation(request)
        else:
            return self._create_unsupported_mode_results(request)
    
    async def _interactive_evaluation(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Conduct interactive human evaluation."""
        print("\n" + "="*60)
        print("🧑‍🎨 HUMAN CREATIVITY EVALUATION")
        print("="*60)
        
        results = []
        
        for i, output in enumerate(request.outputs):
            if output.output_type not in self.supported_output_types:
                results.append(self._create_unsupported_result(output))
                continue
            
            print(f"\n📝 Evaluating Item {i+1}/{len(request.outputs)}")
            print("-" * 40)
            
            # Display the content
            self._display_content(output, request.task_context)
            
            # Collect human rating
            rating = self._collect_interactive_rating()
            
            # Convert to evaluation result
            result = self._convert_rating_to_result(rating, output)
            results.append(result)
        
        print(f"\n✅ Completed evaluation of {len(results)} items")
        return results
    
    async def _batch_evaluation(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Generate batch evaluation files for offline assessment."""
        output_path = Path(self.output_file)
        
        # Prepare evaluation items
        evaluation_items = []
        for i, output in enumerate(request.outputs):
            if output.output_type in self.supported_output_types:
                item = HumanEvaluationItem(
                    content=str(output.content),
                    output_type=output.output_type.value,
                    prompt=getattr(output, 'prompt', None),
                    context=request.task_context,
                    metadata=getattr(output, 'metadata', {})
                )
                evaluation_items.append(asdict(item))
        
        # Save evaluation template
        template_data = {
            "evaluation_instructions": self._get_evaluation_instructions(),
            "rating_scale": self._get_rating_scale(),
            "items": evaluation_items,
            "evaluation_template": asdict(HumanRating(
                novelty=0.0, usefulness=0.0, feasibility=0.0, 
                overall_creativity=0.0, comments="", evaluator_id="", evaluator_expertise=""
            ))
        }
        
        with open(output_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        logger.info(f"Generated human evaluation template: {output_path}")
        
        # Create placeholder results
        results = []
        for output in request.outputs:
            if output.output_type in self.supported_output_types:
                results.append(EvaluationResult(
                    evaluator_name=self.name,
                    layer=self.layer,
                    scores={},
                    explanations={
                        "info": f"Batch evaluation template generated: {output_path}",
                        "instructions": "Complete offline evaluation and use load_human_evaluations() to process results"
                    },
                    metadata={"mode": "batch", "template_file": str(output_path)}
                ))
        
        return results
    
    async def _expert_evaluation(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Load and process expert evaluations from file."""
        if not self.input_file or not Path(self.input_file).exists():
            return self._create_file_not_found_results(request)
        
        try:
            with open(self.input_file) as f:
                evaluation_data = json.load(f)
            
            # Process completed evaluations
            results = []
            for i, output in enumerate(request.outputs):
                if output.output_type not in self.supported_output_types:
                    results.append(self._create_unsupported_result(output))
                    continue
                
                # Find corresponding evaluation
                if i < len(evaluation_data.get("completed_evaluations", [])):
                    rating_data = evaluation_data["completed_evaluations"][i]
                    rating = HumanRating(**rating_data)
                    result = self._convert_rating_to_result(rating, output)
                    results.append(result)
                else:
                    results.append(self._create_missing_evaluation_result(i))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to load human evaluations: {e}")
            return self._create_error_results(request, str(e))
    
    def _display_content(self, output: Any, context: Optional[str]) -> None:
        """Display content for human evaluation."""
        print(f"📄 Content Type: {output.output_type.value.upper()}")
        
        if context:
            print(f"🎯 Context: {context}")
        
        if hasattr(output, 'prompt') and output.prompt:
            print(f"💭 Original Prompt: {output.prompt}")
        
        print("\n📋 Content to Evaluate:")
        print("-" * 30)
        
        content = output.content
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        elif isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        # Truncate very long content
        if len(str(content)) > 1000:
            print(str(content)[:1000] + "\n... (truncated)")
        else:
            print(content)
        
        print("-" * 30)
    
    def _collect_interactive_rating(self) -> HumanRating:
        """Collect human rating through interactive prompts."""
        start_time = time.time()
        
        print("\n🎯 Please rate the creativity on a scale of 1-10:")
        print("(1 = Very Low, 5 = Average, 10 = Exceptional)")
        
        def get_rating(dimension: str, description: str) -> float:
            while True:
                try:
                    value = input(f"\n{dimension} ({description}): ")
                    rating = float(value)
                    if 1 <= rating <= 10:
                        return rating
                    else:
                        print("Please enter a number between 1 and 10")
                except (ValueError, KeyboardInterrupt):
                    print("Please enter a valid number")
        
        novelty = get_rating("Novelty", "How original and unique?")
        usefulness = get_rating("Usefulness", "How valuable and practical?")
        feasibility = get_rating("Feasibility", "How realistic to implement?")
        overall = get_rating("Overall Creativity", "Overall creative quality?")
        
        comments = input("\n💬 Comments (optional): ").strip()
        evaluator_id = input("👤 Your ID/Name (optional): ").strip()
        
        end_time = time.time()
        
        return HumanRating(
            novelty=novelty,
            usefulness=usefulness,
            feasibility=feasibility,
            overall_creativity=overall,
            comments=comments,
            evaluation_time_seconds=end_time - start_time,
            evaluator_id=evaluator_id or "anonymous"
        )
    
    def _convert_rating_to_result(self, rating: HumanRating, output: Any) -> EvaluationResult:
        """Convert human rating to evaluation result."""
        # Convert 1-10 scale to 0-1 scale
        scores = {
            "novelty": (rating.novelty - 1) / 9,  # Map 1-10 to 0-1
            "usefulness": (rating.usefulness - 1) / 9,
            "feasibility": (rating.feasibility - 1) / 9,
            "overall_creativity": (rating.overall_creativity - 1) / 9,
        }
        
        explanations = {}
        if rating.comments:
            explanations["human_comments"] = rating.comments
        
        metadata = {
            "evaluation_time_seconds": rating.evaluation_time_seconds,
            "evaluator_id": rating.evaluator_id,
            "evaluator_expertise": rating.evaluator_expertise,
            "raw_ratings": {
                "novelty": rating.novelty,
                "usefulness": rating.usefulness,
                "feasibility": rating.feasibility,
                "overall_creativity": rating.overall_creativity,
            }
        }
        
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores=scores,
            explanations=explanations,
            metadata=metadata
        )
    
    def _get_evaluation_instructions(self) -> str:
        """Get detailed evaluation instructions for human evaluators."""
        return """
CREATIVITY EVALUATION INSTRUCTIONS

Please evaluate each AI-generated item on the following dimensions using a 1-10 scale:

**NOVELTY (1-10)**: How original and unique is this content?
- 1-3: Very conventional, predictable, or clichéd
- 4-6: Some original elements, but mostly familiar
- 7-9: Highly original with surprising or innovative elements
- 10: Exceptionally novel, groundbreaking, or revolutionary

**USEFULNESS (1-10)**: How valuable, practical, or meaningful is this content?
- 1-3: Low practical value, irrelevant, or unhelpful
- 4-6: Moderately useful, some practical applications
- 7-9: Highly valuable, clear practical benefits
- 10: Exceptionally useful, transformative potential

**FEASIBILITY (1-10)**: How realistic or implementable are the ideas?
- 1-3: Highly unrealistic, impossible to implement
- 4-6: Moderately feasible with some challenges
- 7-9: Highly feasible, realistic implementation path
- 10: Very easy to implement, ready for action

**OVERALL CREATIVITY (1-10)**: Your holistic assessment of creative quality
- Consider all dimensions together
- Factor in context and intended purpose
- Trust your intuitive sense of creativity

Please also provide comments explaining your ratings and any additional observations.
        """
    
    def _get_rating_scale(self) -> Dict[str, str]:
        """Get rating scale descriptions."""
        return {
            "scale": "1-10 (1=Very Low, 5=Average, 10=Exceptional)",
            "dimensions": ["novelty", "usefulness", "feasibility", "overall_creativity"],
            "comments": "Required - please explain your reasoning"
        }
    
    def _create_unsupported_mode_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results for unsupported evaluation modes."""
        results = []
        for output in request.outputs:
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores={},
                explanations={"error": f"Unsupported evaluation mode: {self.evaluation_mode}"},
                metadata={"mode": self.evaluation_mode}
            ))
        return results
    
    def _create_unsupported_result(self, output: Any) -> EvaluationResult:
        """Create result for unsupported output types."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"info": f"Output type {output.output_type} not supported"},
            metadata={"mode": self.evaluation_mode, "supported": False}
        )
    
    def _create_file_not_found_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results when evaluation file is not found."""
        results = []
        for output in request.outputs:
            if output.output_type in self.supported_output_types:
                results.append(EvaluationResult(
                    evaluator_name=self.name,
                    layer=self.layer,
                    scores={},
                    explanations={"error": f"Evaluation file not found: {self.input_file}"},
                    metadata={"mode": self.evaluation_mode, "input_file": self.input_file}
                ))
        return results
    
    def _create_missing_evaluation_result(self, index: int) -> EvaluationResult:
        """Create result for missing evaluation data."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"error": f"No evaluation data found for item {index}"},
            metadata={"mode": self.evaluation_mode, "item_index": index}
        )
    
    def _create_error_results(self, request: EvaluationRequest, error: str) -> List[EvaluationResult]:
        """Create error results."""
        results = []
        for output in request.outputs:
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores={},
                explanations={"error": error},
                metadata={"mode": self.evaluation_mode, "error": True}
            ))
        return results