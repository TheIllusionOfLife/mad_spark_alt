"""
Abduction Agent for hypothesis generation and creative leaps.

This agent implements the "Abduction" phase of the QADI cycle, generating
creative hypotheses and making intuitive leaps to explore possibilities.
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


class AbductionAgent(ThinkingAgentInterface):
    """
    Agent that generates creative hypotheses and makes intuitive leaps.

    This agent focuses on:
    - Generating creative hypotheses about underlying causes
    - Making intuitive connections and leaps
    - Exploring "what if" scenarios
    - Pattern recognition from incomplete information
    """

    def __init__(self, name: str = "AbductionAgent"):
        """Initialize the abduction agent."""
        self._name = name
        self._abduction_strategies = self._load_abduction_strategies()

    @property
    def name(self) -> str:
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.ABDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        valid_keys = {
            "hypothesis_types",
            "creativity_level",
            "use_analogies",
            "explore_opposites",
            "pattern_sources",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """Generate creative hypotheses and insights."""
        start_time = asyncio.get_running_loop().time()

        logger.info(
            f"AbductionAgent generating hypotheses for: {request.problem_statement[:100]}..."
        )

        try:
            generated_hypotheses = []
            config = request.generation_config

            strategies = config.get(
                "hypothesis_types", ["causal", "analogical", "pattern", "opposite"]
            )

            for strategy in strategies:
                hypotheses = await self._generate_hypotheses_by_strategy(
                    request.problem_statement, strategy, request.context, config
                )
                generated_hypotheses.extend(hypotheses)

            # Limit and prioritize hypotheses
            max_total = min(request.max_ideas_per_method, len(generated_hypotheses))
            generated_hypotheses = generated_hypotheses[:max_total]

            end_time = asyncio.get_running_loop().time()
            execution_time = end_time - start_time

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=generated_hypotheses,
                execution_time=execution_time,
                generation_metadata={
                    "strategies_used": strategies,
                    "total_generated": len(generated_hypotheses),
                },
            )

        except Exception as e:
            logger.error(f"Error in AbductionAgent: {e}")
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=[],
                error_message=str(e),
            )

    async def _generate_hypotheses_by_strategy(
        self,
        problem_statement: str,
        strategy: str,
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[GeneratedIdea]:
        """Generate hypotheses using a specific abductive strategy."""
        strategy_templates = self._abduction_strategies.get(strategy, [])
        if not strategy_templates:
            return []

        hypotheses = []
        max_per_strategy = 2  # Keep focused

        for i, template in enumerate(strategy_templates[:max_per_strategy]):
            try:
                hypothesis_content = template.format(
                    problem=problem_statement, context=context or "current situation"
                )

                reasoning = f"Generated {strategy} hypothesis through abductive reasoning, exploring creative connections and possibilities."

                idea = GeneratedIdea(
                    content=hypothesis_content,
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"Generate {strategy} hypothesis for: {problem_statement}",
                    confidence_score=0.6,  # Abductive reasoning has inherent uncertainty
                    reasoning=reasoning,
                    metadata={"strategy": strategy, "hypothesis_type": "creative_leap"},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                hypotheses.append(idea)

            except Exception as e:
                logger.warning(f"Failed to generate {strategy} hypothesis: {e}")
                continue

        return hypotheses

    def _load_abduction_strategies(self) -> Dict[str, List[str]]:
        """Load abductive reasoning strategies and templates."""
        return {
            "causal": [
                "What if {problem} is caused by an unexpected interaction between seemingly unrelated factors?",
                "What if {problem} is actually a natural consequence of a hidden systemic pattern?",
                "What if the root cause of {problem} lies in something we consider irrelevant?",
            ],
            "analogical": [
                "What if {problem} works like how ecosystems self-regulate through feedback loops?",
                "What if solving {problem} is similar to how musicians improvise - structured yet creative?",
                "What if {problem} resembles how markets emerge organically from individual actions?",
            ],
            "pattern": [
                "What if {problem} follows the same pattern as successful solutions in completely different domains?",
                "What if {problem} is part of a larger cycle that we're only seeing one phase of?",
                "What if {problem} exhibits emergent properties that only appear at scale?",
            ],
            "opposite": [
                "What if the solution to {problem} comes from doing the opposite of what seems logical?",
                "What if {problem} becomes an advantage when viewed from a different perspective?",
                "What if preventing {problem} is less effective than leveraging its energy differently?",
            ],
            "emergent": [
                "What if {problem} has solutions that emerge spontaneously when conditions are right?",
                "What if {problem} contains its own solution, waiting to be recognized?",
                "What if {problem} transforms into opportunity through a phase transition we haven't seen?",
            ],
        }
