"""
Deduction Agent for logical validation and systematic reasoning.

This agent implements the "Deduction" phase of the QADI cycle, applying
logical reasoning to validate hypotheses and derive systematic conclusions.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...core.interfaces import (
    ThinkingAgentInterface,
    ThinkingMethod,
    OutputType,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    GeneratedIdea,
)

logger = logging.getLogger(__name__)


class DeductionAgent(ThinkingAgentInterface):
    """
    Agent that applies logical reasoning and systematic validation.
    
    This agent focuses on:
    - Logical validation of hypotheses
    - Systematic consequence analysis
    - Step-by-step reasoning chains
    - Identifying logical implications and requirements
    """

    def __init__(self, name: str = "DeductionAgent"):
        """Initialize the deduction agent."""
        self._name = name
        self._reasoning_frameworks = self._load_reasoning_frameworks()

    @property
    def name(self) -> str:
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.DEDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        valid_keys = {
            "reasoning_types", "validation_depth", "include_counterarguments",
            "logical_frameworks", "systematic_analysis"
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
        """Generate logical analyses and systematic validations."""
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"DeductionAgent applying logical reasoning to: {request.problem_statement[:100]}...")
        
        try:
            generated_analyses = []
            config = request.generation_config
            
            frameworks = config.get("logical_frameworks", ["consequence", "requirement", "validation", "systematic"])
            
            for framework in frameworks:
                analyses = await self._apply_reasoning_framework(
                    request.problem_statement,
                    framework,
                    request.context,
                    config
                )
                generated_analyses.extend(analyses)
            
            max_total = min(request.max_ideas_per_method, len(generated_analyses))
            generated_analyses = generated_analyses[:max_total]
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=generated_analyses,
                execution_time=execution_time,
                generation_metadata={
                    "frameworks_used": frameworks,
                    "total_generated": len(generated_analyses)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in DeductionAgent: {e}")
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=[],
                error_message=str(e)
            )

    async def _apply_reasoning_framework(
        self,
        problem_statement: str,
        framework: str,
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> List[GeneratedIdea]:
        """Apply a specific logical reasoning framework."""
        framework_templates = self._reasoning_frameworks.get(framework, [])
        if not framework_templates:
            return []
        
        analyses = []
        max_per_framework = 2
        
        for i, template in enumerate(framework_templates[:max_per_framework]):
            try:
                analysis_content = template.format(
                    problem=problem_statement,
                    context=context or "given context"
                )
                
                reasoning = f"Applied {framework} logical framework to systematically analyze the problem and derive logical conclusions."
                
                idea = GeneratedIdea(
                    content=analysis_content,
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"Apply {framework} reasoning to: {problem_statement}",
                    confidence_score=0.85,  # Deductive reasoning has high confidence when valid
                    reasoning=reasoning,
                    metadata={
                        "framework": framework,
                        "analysis_type": "logical_deduction"
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                analyses.append(idea)
                
            except Exception as e:
                logger.warning(f"Failed to apply {framework} framework: {e}")
                continue
        
        return analyses

    def _load_reasoning_frameworks(self) -> Dict[str, List[str]]:
        """Load logical reasoning frameworks and templates."""
        return {
            "consequence": [
                "If we address {problem}, then logically we must also consider: What are the immediate consequences? What are the second-order effects? What resources will be required?",
                "Given {problem}, the logical chain of consequences includes: Direct impact → Systemic changes → Long-term implications. Each step requires specific conditions to be met.",
            ],
            "requirement": [
                "To successfully solve {problem}, the following logical requirements must be satisfied: Necessary conditions, sufficient conditions, and enabling constraints.",
                "For {problem} to be resolved, we logically need: Clear success criteria, measurable indicators, resource allocation, and risk mitigation strategies.",
            ],
            "validation": [
                "Testing the validity of approaches to {problem}: What evidence would support success? What evidence would indicate failure? What are the logical benchmarks?",
                "Logical validation of {problem} solutions requires: Falsifiable hypotheses, measurable outcomes, control conditions, and peer review processes.",
            ],
            "systematic": [
                "Systematic analysis of {problem} reveals these logical components: Root causes → Contributing factors → Intervention points → Expected outcomes.",
                "Breaking down {problem} systematically: Define the problem space, identify variables, establish relationships, predict interactions, validate assumptions.",
            ],
            "logical_chain": [
                "The logical chain for {problem} follows: If premise A (current state), and premise B (intervention), then conclusion C (desired outcome) must follow.",
                "Constructing a valid logical argument for {problem}: Major premise, minor premise, logical connection, conclusion, and verification method.",
            ]
        }