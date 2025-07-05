"""
Single LLM creativity judge evaluator.

Implements decomposed creativity evaluation using a single AI model
with structured prompting and multi-dimensional scoring.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from ...core.interfaces import (
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    OutputType,
)
from .llm_client import LLMClient, create_llm_client

logger = logging.getLogger(__name__)


class CreativityLLMJudge(EvaluatorInterface):
    """
    LLM-based creativity evaluator using decomposed assessment.
    
    This evaluator uses a single AI model to assess creativity across
    multiple dimensions with structured prompting and transparent reasoning.
    """
    
    def __init__(self, model: str = "gpt-4", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM judge.
        
        Args:
            model: LLM model to use (gpt-4, claude-3-sonnet, etc.)
            config: Additional configuration options
        """
        self.model = model
        self.config = config or {}
        self.client = create_llm_client(model)
        
        # Creativity dimensions to evaluate
        self.creativity_dimensions = {
            "novelty": "How original and unique is this content compared to typical responses?",
            "usefulness": "How practical, valuable, or meaningful is this content?", 
            "feasibility": "How realistic and implementable are the ideas presented?",
            "elaboration": "How detailed, developed, and well-explained is the content?",
            "surprise": "How unexpected or surprising are the ideas or connections made?",
            "elegance": "How simple, clear, and aesthetically pleasing is the solution/expression?"
        }
    
    @property
    def name(self) -> str:
        return f"creativity_llm_judge_{self.model.replace('-', '_')}"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.LLM_JUDGE
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE, OutputType.STRUCTURED]
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        valid_keys = {
            "temperature", "max_tokens", "dimensions", 
            "include_rationale", "scoring_rubric"
        }
        return all(key in valid_keys for key in config.keys())
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate creativity using LLM judge."""
        if not self.client.is_available:
            logger.warning(f"LLM client not available for {self.model}")
            return self._create_unavailable_results(request)
        
        results = []
        tasks = []
        
        # Create evaluation tasks for each output
        for output in request.outputs:
            if output.output_type in self.supported_output_types:
                task = self._evaluate_single_output(output, request)
                tasks.append(task)
            else:
                # Create empty result for unsupported types
                results.append(self._create_empty_result(output))
        
        # Execute evaluations concurrently
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in task_results:
                if isinstance(result, Exception):
                    logger.error(f"LLM evaluation error: {result}")
                    results.append(self._create_error_result(str(result)))
                else:
                    results.append(result)
        
        return results
    
    async def _evaluate_single_output(
        self, 
        output: Any, 
        request: EvaluationRequest
    ) -> EvaluationResult:
        """Evaluate a single output for creativity."""
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(output, request)
        system_prompt = self._build_system_prompt()
        
        # Get LLM response
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.get("temperature", 0.0),
            max_tokens=self.config.get("max_tokens", 1500),
        )
        
        if response.error:
            return self._create_error_result(response.error)
        
        # Parse the response
        try:
            evaluation_data = self._parse_llm_response(response.content)
            
            # Extract scores and explanations
            scores = evaluation_data.get("creativity_scores", {})
            rationale = evaluation_data.get("rationale", "")
            
            # Add overall score
            if scores:
                scores["overall_creativity"] = evaluation_data.get(
                    "overall_score", 
                    sum(scores.values()) / len(scores)
                )
            
            # Build explanations
            explanations = {"rationale": rationale}
            strengths = evaluation_data.get("strengths", [])
            weaknesses = evaluation_data.get("weaknesses", [])
            
            if strengths:
                explanations["strengths"] = "; ".join(strengths)
            if weaknesses:
                explanations["weaknesses"] = "; ".join(weaknesses)
            
            return EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores=scores,
                explanations=explanations,
                metadata={
                    "model": self.model,
                    "usage": response.usage,
                    "evaluation_dimensions": list(self.creativity_dimensions.keys()),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._create_error_result(f"Response parsing error: {e}")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for creativity evaluation."""
        dimensions_text = "\n".join([
            f"- **{dim}**: {desc}" 
            for dim, desc in self.creativity_dimensions.items()
        ])
        
        return f"""You are an expert creativity evaluator. Your task is to assess the creativity of AI-generated content across multiple dimensions.

**Evaluation Dimensions:**
{dimensions_text}

**Scoring Guidelines:**
- Use a scale from 0.0 to 1.0 for each dimension
- 0.0-0.3: Low creativity (conventional, predictable, basic)
- 0.4-0.6: Moderate creativity (some original elements, partially novel)
- 0.7-0.9: High creativity (innovative, surprising, well-developed)
- 1.0: Exceptional creativity (groundbreaking, revolutionary, masterful)

**Response Format:**
Return your evaluation as valid JSON with this structure:
{{
  "creativity_scores": {{
    "novelty": 0.0-1.0,
    "usefulness": 0.0-1.0,
    "feasibility": 0.0-1.0,
    "elaboration": 0.0-1.0,
    "surprise": 0.0-1.0,
    "elegance": 0.0-1.0
  }},
  "overall_score": 0.0-1.0,
  "rationale": "Detailed explanation of your assessment",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"]
}}

Be objective, detailed, and consistent in your evaluations."""
    
    def _build_evaluation_prompt(
        self, 
        output: Any, 
        request: EvaluationRequest
    ) -> str:
        """Build the evaluation prompt for a specific output."""
        content = output.content
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        elif isinstance(content, bytes):
            content = content.decode('utf-8', errors='ignore')
        
        prompt_parts = [
            "Please evaluate the creativity of the following AI-generated content:",
            "",
            "**Content to Evaluate:**",
            f"```{output.output_type.value}",
            str(content),
            "```",
            "",
        ]
        
        # Add context if available
        if hasattr(output, 'prompt') and output.prompt:
            prompt_parts.extend([
                "**Original Prompt:**",
                output.prompt,
                "",
            ])
        
        if request.task_context:
            prompt_parts.extend([
                "**Task Context:**", 
                request.task_context,
                "",
            ])
        
        prompt_parts.extend([
            "**Instructions:**",
            "1. Carefully read and analyze the content",
            "2. Evaluate each creativity dimension (novelty, usefulness, feasibility, elaboration, surprise, elegance)",
            "3. Provide specific reasoning for your scores",
            "4. Identify key strengths and areas for improvement",
            "5. Return your assessment in the specified JSON format",
            "",
            "Begin your evaluation:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            start_markers = ["{", "```json", "```"]
            end_markers = ["}", "```"]
            
            content = response_content.strip()
            
            # Find JSON content
            json_start = -1
            for marker in start_markers:
                idx = content.find(marker)
                if idx != -1:
                    json_start = idx if marker == "{" else idx + len(marker)
                    break
            
            if json_start == -1:
                raise ValueError("No JSON found in response")
            
            # Find end of JSON
            json_end = len(content)
            for marker in end_markers:
                idx = content.rfind(marker)
                if idx > json_start:
                    json_end = idx + 1 if marker == "}" else idx
                    break
            
            json_text = content[json_start:json_end]
            if not json_text.strip().startswith("{"):
                # Find the actual JSON start
                json_start = json_text.find("{")
                if json_start != -1:
                    json_text = json_text[json_start:]
            
            return json.loads(json_text)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback: create basic structure from text
            return {
                "creativity_scores": {dim: 0.5 for dim in self.creativity_dimensions},
                "overall_score": 0.5,
                "rationale": f"Failed to parse structured response. Raw content: {response_content[:200]}...",
                "strengths": ["Content generated"],
                "weaknesses": ["Evaluation parsing failed"]
            }
    
    def _create_unavailable_results(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Create results when LLM client is unavailable."""
        results = []
        for output in request.outputs:
            if output.output_type in self.supported_output_types:
                results.append(EvaluationResult(
                    evaluator_name=self.name,
                    layer=self.layer,
                    scores={},
                    explanations={"error": f"LLM model {self.model} not available"},
                    metadata={"model": self.model, "available": False}
                ))
        return results
    
    def _create_empty_result(self, output: Any) -> EvaluationResult:
        """Create empty result for unsupported output types."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"info": f"Output type {output.output_type} not supported"},
            metadata={"model": self.model, "supported": False}
        )
    
    def _create_error_result(self, error_message: str) -> EvaluationResult:
        """Create error result."""
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores={},
            explanations={"error": error_message},
            metadata={"model": self.model, "error": True}
        )