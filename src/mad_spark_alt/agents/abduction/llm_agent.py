"""
LLM-Powered Abductive Agent for intelligent hypothesis generation.

This agent uses Large Language Models to generate sophisticated hypotheses,
creative leaps, and intuitive connections through abductive reasoning.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ...core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from ...core.json_utils import safe_json_parse, parse_json_list
from ...core.llm_provider import (
    LLMManager,
    LLMProvider,
    LLMRequest,
)
from ...core.llm_provider import llm_manager as default_llm_manager

logger = logging.getLogger(__name__)


class LLMAbductiveAgent(ThinkingAgentInterface):
    """
    LLM-powered agent that generates sophisticated hypotheses through abductive reasoning.

    This agent uses artificial intelligence to:
    - Generate creative hypotheses about underlying causes and mechanisms
    - Make intuitive leaps and explore "what if" scenarios
    - Identify patterns and connections from incomplete information
    - Explore analogies and metaphors for deeper understanding
    - Generate counter-intuitive and novel perspectives
    """

    def __init__(
        self,
        name: str = "LLMAbductiveAgent",
        llm_manager: Optional[LLMManager] = None,
        preferred_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the LLM-powered abductive agent.

        Args:
            name: Unique name for this agent
            llm_manager: LLM manager instance (uses global if None)
            preferred_provider: Preferred LLM provider (auto-select if None)
        """
        self._name = name
        self.llm_manager = llm_manager or default_llm_manager
        self.preferred_provider = preferred_provider
        self._abductive_strategies = self._load_abductive_strategies()

    @property
    def name(self) -> str:
        """Unique name for this thinking agent."""
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        """The thinking method this agent implements."""
        return ThinkingMethod.ABDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        """Output types this agent can work with."""
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is valid for this agent."""
        valid_keys = {
            "max_strategies",  # Number of abductive strategies to use (default: 4)
        }

        # Check if all keys are valid
        if not all(key in valid_keys for key in config.keys()):
            return False

        # Validate specific config values
        if "max_strategies" in config:
            max_strategies = config["max_strategies"]
            if (
                not isinstance(max_strategies, int)
                or max_strategies < 1
                or max_strategies > len(self._abductive_strategies)
            ):
                return False

        return True

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """
        Generate intelligent hypotheses using LLM reasoning.

        Args:
            request: The idea generation request

        Returns:
            Result containing AI-generated hypotheses as ideas
        """
        start_time = time.time()

        logger.info(
            f"{self.name} generating hypotheses for: {request.problem_statement[:100]}..."
        )

        try:
            # Analyze the problem context for hypothesis generation
            context_analysis = await self._analyze_problem_context(
                request.problem_statement, request.context
            )

            # Generate hypotheses using multiple abductive strategies
            all_hypotheses = []
            config = request.generation_config or {}

            # Select abductive strategies based on config and context
            strategies = self._select_abductive_strategies(context_analysis, config)

            # Track critical errors
            critical_errors = []

            for strategy in strategies:
                try:
                    hypotheses = await self._generate_hypotheses_with_strategy(
                        request.problem_statement,
                        request.context,
                        strategy,
                        context_analysis,
                        config,
                    )
                    all_hypotheses.extend(hypotheses)
                except Exception as strategy_error:
                    logger.error(
                        f"Strategy {strategy['name']} failed: {strategy_error}"
                    )
                    critical_errors.append(str(strategy_error))
                    continue

            # If all strategies failed with the same error, it's likely a system issue
            if len(critical_errors) == len(strategies) and critical_errors:
                # Check if all errors are the same (indicating a system-wide issue)
                unique_errors = set(critical_errors)
                if len(unique_errors) == 1:
                    raise Exception(critical_errors[0])

            # Limit and rank hypotheses
            max_hypotheses = request.max_ideas_per_method
            final_hypotheses = await self._rank_and_select_hypotheses(
                all_hypotheses,
                max_hypotheses,
                request.problem_statement,
                context_analysis,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(
                f"{self.name} generated {len(final_hypotheses)} hypotheses in {execution_time:.2f}s"
            )

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=final_hypotheses,
                execution_time=execution_time,
                generation_metadata={
                    "context_analysis": context_analysis,
                    "strategies_used": [s["name"] for s in strategies],
                    "total_generated": len(all_hypotheses),
                    "final_selected": len(final_hypotheses),
                    "config": config,
                },
            )

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=[],
                error_message=str(e),
            )

    async def _analyze_problem_context(
        self, problem_statement: str, context: Optional[str] = None
    ) -> Any:
        """Analyze the problem context to inform abductive reasoning."""
        system_prompt = """You are an expert problem analyst specializing in abductive reasoning. Analyze the given problem to understand characteristics that will inform hypothesis generation. Provide your analysis in the following JSON format:

{
    "domain": "primary domain of the problem",
    "problem_nature": "well_structured|ill_structured|wicked_problem",
    "evidence_availability": "rich|moderate|sparse|contradictory",
    "causal_complexity": "simple|moderate|complex|chaotic",
    "analogical_domains": ["list", "of", "potential", "analogical", "domains"],
    "pattern_indicators": ["observable", "patterns", "or", "symptoms"],
    "uncertainty_level": "low|medium|high|extreme",
    "stakeholder_impact": "direct|indirect|systemic",
    "temporal_dynamics": "static|evolving|cyclical|emergent",
    "potential_biases": ["cognitive", "biases", "to", "consider"],
    "abductive_opportunities": ["areas", "ripe", "for", "creative", "leaps"]
}

Focus on aspects that will enable creative hypothesis generation and abductive reasoning."""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'None provided'}

Analyze this problem to identify characteristics that will inform abductive hypothesis generation."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.3,  # Lower temperature for analytical tasks
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction

            fallback_analysis = {
                "domain": "general",
                "problem_nature": "ill_structured",
                "evidence_availability": "moderate",
                "causal_complexity": "complex",
                "analogical_domains": ["general"],
                "pattern_indicators": ["complexity", "uncertainty"],
                "uncertainty_level": "medium",
                "stakeholder_impact": "direct",
                "temporal_dynamics": "evolving",
                "potential_biases": ["confirmation_bias"],
                "abductive_opportunities": ["hypothesis_generation", "creative_leaps"],
            }
            analysis = safe_json_parse(response.content, fallback_analysis)
            analysis["llm_cost"] = response.cost

            return analysis

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {
                "domain": "general",
                "problem_nature": "ill_structured",
                "evidence_availability": "moderate",
                "causal_complexity": "complex",
                "analogical_domains": ["general"],
                "pattern_indicators": ["complexity", "uncertainty"],
                "uncertainty_level": "medium",
                "stakeholder_impact": "direct",
                "temporal_dynamics": "evolving",
                "potential_biases": ["confirmation_bias"],
                "abductive_opportunities": ["hypothesis_generation", "creative_leaps"],
            }

    def _load_abductive_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load different abductive reasoning strategies and their configurations."""
        return {
            "causal_inference": {
                "name": "causal_inference",
                "description": "Generate hypotheses about underlying causes and mechanisms",
                "focus": "causation, root causes, mechanisms, hidden factors",
                "cognitive_approach": "analytical, cause-effect reasoning",
            },
            "analogical_reasoning": {
                "name": "analogical_reasoning",
                "description": "Draw analogies from other domains to generate insights",
                "focus": "analogies, metaphors, cross-domain patterns",
                "cognitive_approach": "comparative, metaphorical",
            },
            "pattern_recognition": {
                "name": "pattern_recognition",
                "description": "Identify hidden patterns and emergent properties",
                "focus": "patterns, emergence, system behaviors",
                "cognitive_approach": "pattern-based, emergent thinking",
            },
            "counter_intuitive": {
                "name": "counter_intuitive",
                "description": "Explore paradoxical and counter-intuitive possibilities",
                "focus": "paradoxes, inversions, unexpected connections",
                "cognitive_approach": "contrarian, paradoxical",
            },
            "what_if_scenarios": {
                "name": "what_if_scenarios",
                "description": "Generate creative 'what if' scenarios and possibilities",
                "focus": "scenarios, possibilities, speculative reasoning",
                "cognitive_approach": "speculative, exploratory",
            },
            "systems_perspective": {
                "name": "systems_perspective",
                "description": "View the problem as part of larger systems and networks",
                "focus": "systems, networks, interconnections, feedback",
                "cognitive_approach": "systemic, holistic",
            },
            "temporal_reasoning": {
                "name": "temporal_reasoning",
                "description": "Consider temporal aspects and evolutionary dynamics",
                "focus": "timing, evolution, cycles, development",
                "cognitive_approach": "temporal, evolutionary",
            },
        }

    def _select_abductive_strategies(
        self, context_analysis: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select appropriate abductive strategies based on context and config."""
        strategies = list(self._abductive_strategies.values())

        # Default to 3-4 strategies for balanced coverage
        max_strategies = config.get("max_strategies", 4)

        # Customize strategy selection based on context analysis
        causal_complexity = context_analysis.get("causal_complexity", "complex")
        evidence_availability = context_analysis.get(
            "evidence_availability", "moderate"
        )
        uncertainty_level = context_analysis.get("uncertainty_level", "medium")

        # Prioritize strategies based on problem characteristics
        if causal_complexity in ["complex", "chaotic"]:
            # For complex causation, prioritize systems and pattern approaches
            priority_strategies = [
                "systems_perspective",
                "pattern_recognition",
                "causal_inference",
            ]
        elif evidence_availability == "sparse":
            # For sparse evidence, emphasize creative leaps and analogies
            priority_strategies = [
                "analogical_reasoning",
                "what_if_scenarios",
                "counter_intuitive",
            ]
        elif uncertainty_level == "high":
            # For high uncertainty, explore multiple perspectives
            priority_strategies = [
                "what_if_scenarios",
                "counter_intuitive",
                "analogical_reasoning",
            ]
        else:
            # For moderate complexity, use balanced approach
            priority_strategies = [
                "causal_inference",
                "pattern_recognition",
                "analogical_reasoning",
            ]

        # Select strategies ensuring priority ones are included
        selected: List[Dict[str, Any]] = []
        strategy_dict = {s["name"]: s for s in strategies}

        # Add priority strategies first
        for strategy_name in priority_strategies:
            if strategy_name in strategy_dict and len(selected) < max_strategies:
                selected.append(strategy_dict[strategy_name])

        # Fill remaining slots with other strategies
        for strategy in strategies:
            if strategy not in selected and len(selected) < max_strategies:
                selected.append(strategy)

        return selected[:max_strategies]

    async def _generate_hypotheses_with_strategy(
        self,
        problem_statement: str,
        context: Optional[str],
        strategy: Dict[str, Any],
        context_analysis: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Generate hypotheses using a specific abductive strategy."""

        strategy_name = strategy["name"]
        strategy_description = strategy["description"]
        strategy_focus = strategy["focus"]
        cognitive_approach = strategy["cognitive_approach"]

        # Create strategy-specific system prompt
        system_prompt = f"""You are an expert hypothesis generator specializing in {strategy_name} abductive reasoning.

Your role is to generate creative, insightful hypotheses using a {cognitive_approach} approach, focusing on {strategy_focus}.

Strategy Description: {strategy_description}

Problem Context:
- Domain: {context_analysis.get('domain', 'general')}
- Causal Complexity: {context_analysis.get('causal_complexity', 'moderate')}
- Evidence Level: {context_analysis.get('evidence_availability', 'moderate')}
- Uncertainty: {context_analysis.get('uncertainty_level', 'medium')}

Generate 3-5 high-quality hypotheses that:
1. Use {strategy_name} reasoning to explore the problem space
2. Are creative yet plausible given the evidence
3. Provide genuine insights rather than obvious explanations
4. Consider multiple levels of analysis (individual, system, meta-level)
5. Are specific enough to be testable or explorable

Format your response as a JSON array of objects, each containing:
{{
    "hypothesis": "the actual hypothesis statement",
    "reasoning": "the abductive logic behind this hypothesis",
    "evidence_requirements": "what evidence would support or refute this",
    "implications": "what this hypothesis implies for understanding or action",
    "confidence_level": "low|medium|high - your confidence in this hypothesis"
}}"""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'No additional context provided'}

Using {strategy_name} abductive reasoning, generate creative hypotheses that could explain or illuminate this problem. Focus on {strategy_focus} while maintaining a {cognitive_approach} perspective."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1200,
                temperature=0.8,  # Higher creativity for hypothesis generation
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction

            hypotheses_data = parse_json_list(response.content, [])

            generated_hypotheses = []
            # Distribute cost across all generated hypotheses from this API call
            cost_per_hypothesis = (
                response.cost / len(hypotheses_data) if hypotheses_data else 0
            )

            for i, h_data in enumerate(hypotheses_data):
                # Map confidence level to numeric score
                confidence_map = {"low": 0.4, "medium": 0.6, "high": 0.8}
                confidence_score = confidence_map.get(
                    h_data.get("confidence_level", "medium"), 0.6
                )

                idea = GeneratedIdea(
                    content=h_data["hypothesis"],
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"{strategy_name} strategy for: {problem_statement[:100]}...",
                    confidence_score=confidence_score,
                    reasoning=h_data["reasoning"],
                    metadata={
                        "strategy": strategy_name,
                        "evidence_requirements": h_data.get("evidence_requirements"),
                        "implications": h_data.get("implications"),
                        "confidence_level": h_data.get("confidence_level", "medium"),
                        "cognitive_approach": cognitive_approach,
                        "llm_cost": cost_per_hypothesis,
                        "batch_cost": response.cost,
                        "generation_index": i,
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                generated_hypotheses.append(idea)

            return generated_hypotheses

        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse hypotheses JSON for strategy {strategy_name}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Hypothesis generation failed for strategy {strategy_name}: {e}"
            )
            # Re-raise the exception so it can be handled at the agent level
            raise

    async def _rank_and_select_hypotheses(
        self,
        hypotheses: List[GeneratedIdea],
        max_hypotheses: int,
        problem_statement: str,
        context_analysis: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Rank and select the best hypotheses using AI evaluation."""

        if len(hypotheses) <= max_hypotheses:
            return hypotheses

        # Create ranking system prompt
        system_prompt = """You are an expert hypothesis evaluator specializing in abductive reasoning. Rank the given hypotheses based on their quality and potential for insight generation.

Evaluation criteria:
1. Plausibility and logical coherence (25%)
2. Novelty and creativity (25%)
3. Explanatory power (20%)
4. Testability and actionability (15%)
5. Insight potential (15%)

Consider that abductive hypotheses should be creative leaps that go beyond obvious explanations while remaining plausible.

Provide rankings as a JSON array of hypothesis indices (0-based) in order from best to worst."""

        # Format hypotheses for evaluation
        hypotheses_text = "\n".join(
            [
                f"{i}. {h.content} (Strategy: {h.metadata.get('strategy', 'unknown')}, Confidence: {h.metadata.get('confidence_level', 'medium')})"
                for i, h in enumerate(hypotheses)
            ]
        )

        user_prompt = f"""Problem: {problem_statement}
Domain: {context_analysis.get('domain', 'general')}
Complexity: {context_analysis.get('causal_complexity', 'moderate')}

Hypotheses to rank:
{hypotheses_text}

Rank these hypotheses from best to worst based on the evaluation criteria."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.2,  # Low temperature for consistent ranking
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            rankings = parse_json_list(response.content, list(range(len(hypotheses))))

            # Select top hypotheses based on rankings
            selected_hypotheses: List[GeneratedIdea] = []
            processed_indices: Set[int] = set()
            for rank, rank_idx in enumerate(rankings):
                if (
                    0 <= rank_idx < len(hypotheses)
                    and rank_idx not in processed_indices
                    and len(selected_hypotheses) < max_hypotheses
                ):
                    hypotheses[rank_idx].metadata["ranking_score"] = (
                        len(rankings) - rank
                    )
                    selected_hypotheses.append(hypotheses[rank_idx])
                    processed_indices.add(rank_idx)

            return selected_hypotheses

        except Exception as e:
            logger.warning(f"Hypothesis ranking failed, using fallback selection: {e}")
            # Fallback: return first max_hypotheses, ensuring strategy diversity
            strategies_used: Set[str] = set()
            selected: List[GeneratedIdea] = []

            for hypothesis in hypotheses:
                strategy = hypothesis.metadata.get("strategy", "unknown")
                if len(selected) < max_hypotheses:
                    if (
                        strategy not in strategies_used
                        or len(strategies_used) >= max_hypotheses // 2
                    ):
                        selected.append(hypothesis)
                        strategies_used.add(strategy)

            return selected[:max_hypotheses]
