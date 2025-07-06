"""
Induction Agent for pattern synthesis and rule formation.

This agent implements the "Induction" phase of the QADI cycle, synthesizing
patterns from observations and forming general principles and insights.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)

logger = logging.getLogger(__name__)


class InductionAgent(ThinkingAgentInterface):
    """
    Agent that synthesizes patterns and forms general principles.

    This agent focuses on:
    - Pattern recognition and synthesis
    - Generalization from specific observations
    - Rule formation and principle extraction
    - Creative synthesis and insight generation
    """

    def __init__(self, name: str = "InductionAgent"):
        """Initialize the induction agent."""
        self._name = name
        self._synthesis_methods = self._load_synthesis_methods()

    @property
    def name(self) -> str:
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.INDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        valid_keys = {
            "synthesis_methods",
            "pattern_depth",
            "generalization_level",
            "insight_generation",
            "principle_extraction",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """Generate insights through pattern synthesis and generalization."""
        start_time = asyncio.get_running_loop().time()

        logger.info(
            f"InductionAgent synthesizing insights for: {request.problem_statement[:100]}..."
        )

        try:
            generated_insights = []
            config = request.generation_config

            methods = config.get(
                "synthesis_methods",
                ["pattern", "principle", "insight", "generalization"],
            )

            for method in methods:
                insights = await self._apply_synthesis_method(
                    request.problem_statement, method, request.context, config
                )
                generated_insights.extend(insights)

            max_total = min(request.max_ideas_per_method, len(generated_insights))
            generated_insights = generated_insights[:max_total]

            end_time = asyncio.get_running_loop().time()
            execution_time = end_time - start_time

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=generated_insights,
                execution_time=execution_time,
                generation_metadata={
                    "methods_used": methods,
                    "total_generated": len(generated_insights),
                },
            )

        except Exception as e:
            logger.error(f"Error in InductionAgent: {e}")
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=[],
                error_message=str(e),
            )

    async def _apply_synthesis_method(
        self,
        problem_statement: str,
        method: str,
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[GeneratedIdea]:
        """Apply a specific inductive synthesis method."""
        method_templates = self._synthesis_methods.get(method, [])
        if not method_templates:
            return []

        insights = []
        max_per_method = 2

        for i, template in enumerate(method_templates[:max_per_method]):
            try:
                insight_content = template.format(
                    problem=problem_statement,
                    context=context or "available information",
                )

                reasoning = f"Applied {method} inductive synthesis to identify patterns and generate insights from the available information and context."

                idea = GeneratedIdea(
                    content=insight_content,
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"Apply {method} synthesis to: {problem_statement}",
                    confidence_score=0.75,  # Inductive insights have good confidence when well-grounded
                    reasoning=reasoning,
                    metadata={
                        "method": method,
                        "synthesis_type": "inductive_reasoning",
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                insights.append(idea)

            except Exception as e:
                logger.warning(f"Failed to apply {method} synthesis: {e}")
                continue

        return insights

    def _load_synthesis_methods(self) -> Dict[str, List[str]]:
        """Load inductive synthesis methods and templates."""
        return {
            "pattern": [
                "Looking at the patterns in {problem}, I observe recurring themes that suggest a general principle: successful solutions tend to share these characteristics...",
                "The underlying pattern in {problem} reveals that similar challenges across different domains follow this common structure: initial conditions → transition phase → emergent outcome.",
            ],
            "principle": [
                "From the various aspects of {problem}, I can extract this fundamental principle: effective solutions balance competing forces while maintaining system integrity.",
                "The evidence around {problem} points to a core principle: sustainable solutions emerge when local actions align with systemic needs.",
            ],
            "insight": [
                "Synthesizing the information about {problem}, a key insight emerges: what appears to be a problem might actually be a symptom of a deeper opportunity for transformation.",
                "The collective understanding of {problem} reveals an unexpected insight: the most effective interventions often work by changing the context rather than attacking the problem directly.",
            ],
            "generalization": [
                "Generalizing from the specifics of {problem}, this approach could apply to a broader class of challenges where similar dynamics are at play.",
                "The lessons from {problem} suggest a general framework that could be valuable: identify leverage points, understand feedback loops, design for emergence.",
            ],
            "meta_pattern": [
                "At a meta-level, {problem} exemplifies how complex systems exhibit self-organizing properties when conditions support natural evolution toward solutions.",
                "The meta-pattern in {problem} shows how apparent contradictions often resolve at a higher level of organization through creative synthesis.",
            ],
        }
