"""
LLM-Powered Inductive Agent for intelligent pattern synthesis and insight generation.

This agent uses Large Language Models to perform sophisticated inductive reasoning,
pattern recognition, rule formation, and creative synthesis from observations.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from ...core.llm_provider import (
    LLMManager,
    LLMProvider,
    LLMRequest,
    llm_manager,
)

logger = logging.getLogger(__name__)


class LLMInductiveAgent(ThinkingAgentInterface):
    """
    LLM-powered agent that performs sophisticated inductive reasoning and pattern synthesis.

    This agent uses artificial intelligence to:
    - Recognize and synthesize patterns from observations and data
    - Generate insights through creative synthesis and meta-recognition
    - Form general principles and rules from specific instances
    - Identify emergent properties and higher-order patterns
    - Extract actionable insights and generalizable knowledge
    """

    def __init__(
        self,
        name: str = "LLMInductiveAgent",
        llm_manager: Optional[LLMManager] = None,
        preferred_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the LLM-powered inductive agent.

        Args:
            name: Unique name for this agent
            llm_manager: LLM manager instance (uses global if None)
            preferred_provider: Preferred LLM provider (auto-select if None)
        """
        self._name = name
        from ...core.llm_provider import llm_manager as default_llm_manager

        self.llm_manager = llm_manager or default_llm_manager
        self.preferred_provider = preferred_provider
        self._inductive_methods = self._load_inductive_methods()

    @property
    def name(self) -> str:
        """Unique name for this thinking agent."""
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        """The thinking method this agent implements."""
        return ThinkingMethod.INDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        """Output types this agent can work with."""
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is valid for this agent."""
        valid_keys = {
            "inductive_method",
            "pattern_depth",
            "synthesis_scope",
            "generalization_level",
            "insight_generation",
            "principle_extraction",
            "meta_pattern_analysis",
            "max_insights_per_method",
            "include_emergent_properties",
            "creative_synthesis",
            "rule_formation_depth",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """
        Generate insights through LLM inductive reasoning.

        Args:
            request: The idea generation request

        Returns:
            Result containing AI-generated insights as ideas
        """
        start_time = asyncio.get_event_loop().time()

        logger.info(
            f"{self.name} synthesizing insights for: {request.problem_statement[:100]}..."
        )

        try:
            # Analyze the observational context for pattern synthesis
            synthesis_context = await self._analyze_synthesis_context(
                request.problem_statement, request.context
            )

            # Generate insights using multiple inductive methods
            all_insights = []
            config = request.generation_config or {}

            # Select inductive methods based on config and context
            methods = self._select_inductive_methods(synthesis_context, config)

            # Track critical errors
            critical_errors = []

            for method in methods:
                try:
                    insights = await self._apply_inductive_method(
                        request.problem_statement,
                        request.context,
                        method,
                        synthesis_context,
                        config,
                    )
                    all_insights.extend(insights)
                except Exception as method_error:
                    logger.error(f"Method {method['name']} failed: {method_error}")
                    critical_errors.append(str(method_error))
                    continue

            # If all methods failed with the same error, it's likely a system issue
            if len(critical_errors) == len(methods) and critical_errors:
                # Check if all errors are the same (indicating a system-wide issue)
                unique_errors = set(critical_errors)
                if len(unique_errors) == 1:
                    raise Exception(critical_errors[0])

            # Limit and rank insights
            max_insights = request.max_ideas_per_method
            final_insights = await self._rank_and_select_insights(
                all_insights, max_insights, request.problem_statement, synthesis_context
            )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            logger.info(
                f"{self.name} generated {len(final_insights)} insights in {execution_time:.2f}s"
            )

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=final_insights,
                execution_time=execution_time,
                generation_metadata={
                    "synthesis_context": synthesis_context,
                    "methods_used": [m["name"] for m in methods],
                    "total_generated": len(all_insights),
                    "final_selected": len(final_insights),
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

    async def _analyze_synthesis_context(
        self, problem_statement: str, context: Optional[str] = None
    ) -> Any:
        """Analyze the context to understand patterns and synthesis opportunities."""
        system_prompt = """You are an expert pattern analyst specializing in inductive reasoning. Analyze the given problem to understand the observational context, data patterns, and synthesis opportunities. Provide your analysis in the following JSON format:

{{
    "data_richness": "sparse|moderate|rich|very_rich",
    "pattern_visibility": "hidden|subtle|apparent|obvious",
    "synthesis_complexity": "simple|moderate|complex|highly_complex", 
    "generalization_potential": "low|medium|high|very_high",
    "observable_patterns": ["list", "of", "observable", "patterns"],
    "meta_patterns": ["higher", "order", "patterns", "if", "any"],
    "insight_opportunities": ["areas", "ripe", "for", "insight", "generation"],
    "principle_extraction_potential": "low|medium|high",
    "emergent_properties": ["potential", "emergent", "properties"],
    "cross_domain_applicability": "narrow|moderate|broad|universal",
    "synthesis_depth_needed": "surface|moderate|deep|very_deep"
}}

Focus on aspects that will enable creative inductive reasoning and pattern synthesis."""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'None provided'}

Analyze this problem to identify patterns, synthesis opportunities, and the potential for generating insights through inductive reasoning."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.3,  # Moderate temperature for analytical creativity
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction
            from ...core.json_utils import safe_json_parse
            fallback_analysis = {
                "data_richness": "moderate",
                "pattern_visibility": "subtle",
                "pattern_diversity": "medium",
                "observation_scope": "comprehensive",
                "synthesis_opportunities": ["commonalities", "trends"],
                "inductive_strength": "moderate",
                "generalization_potential": "medium",
                "insight_opportunities": ["pattern_recognition", "trend_analysis"],
            }
            analysis = safe_json_parse(response.content, fallback_analysis)
            analysis["llm_cost"] = response.cost

            return analysis

        except json.JSONDecodeError:
            # Fallback to basic analysis if JSON parsing fails
            logger.warning("Failed to parse synthesis context JSON, using fallback")
            return {
                "data_richness": "moderate",
                "pattern_visibility": "subtle",
                "synthesis_complexity": "moderate",
                "generalization_potential": "medium",
                "insight_opportunities": [
                    "pattern_recognition",
                    "principle_extraction",
                ],
                "synthesis_depth_needed": "moderate",
            }
        except Exception as e:
            logger.error(f"Synthesis context analysis failed: {e}")
            return {"data_richness": "unknown", "synthesis_complexity": "unknown"}

    def _load_inductive_methods(self) -> Dict[str, Dict[str, Any]]:
        """Load different inductive reasoning methods and their configurations."""
        return {
            "pattern_synthesis": {
                "name": "pattern_synthesis",
                "description": "Synthesize patterns from observations to generate insights",
                "focus": "patterns, synthesis, emergence, regularities",
                "cognitive_approach": "pattern-based, emergent thinking",
            },
            "principle_extraction": {
                "name": "principle_extraction",
                "description": "Extract general principles and rules from specific instances",
                "focus": "principles, rules, generalizations, abstractions",
                "cognitive_approach": "abstraction-based, rule-forming",
            },
            "meta_recognition": {
                "name": "meta_recognition",
                "description": "Identify meta-patterns and higher-order relationships",
                "focus": "meta-patterns, higher-order, relationships, structures",
                "cognitive_approach": "meta-cognitive, structural",
            },
            "creative_synthesis": {
                "name": "creative_synthesis",
                "description": "Generate creative insights through novel combinations",
                "focus": "creativity, combinations, novel connections, innovation",
                "cognitive_approach": "creative, combinatorial",
            },
            "trend_analysis": {
                "name": "trend_analysis",
                "description": "Identify trends and evolutionary patterns over time",
                "focus": "trends, evolution, temporal patterns, progression",
                "cognitive_approach": "temporal, evolutionary",
            },
            "analogical_extension": {
                "name": "analogical_extension",
                "description": "Extend insights through analogical reasoning across domains",
                "focus": "analogies, cross-domain, extensions, mappings",
                "cognitive_approach": "analogical, cross-domain",
            },
            "emergent_insight": {
                "name": "emergent_insight",
                "description": "Identify emergent properties and system-level insights",
                "focus": "emergence, system properties, holistic insights",
                "cognitive_approach": "emergent, systems-oriented",
            },
        }

    def _select_inductive_methods(
        self, synthesis_context: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select appropriate inductive methods based on synthesis context and config."""
        methods = list(self._inductive_methods.values())

        # Default to 3-4 methods for comprehensive coverage
        max_methods = config.get("max_methods", 4)

        # Customize method selection based on synthesis context
        data_richness = synthesis_context.get("data_richness", "moderate")
        pattern_visibility = synthesis_context.get("pattern_visibility", "subtle")
        generalization_potential = synthesis_context.get(
            "generalization_potential", "medium"
        )

        # Prioritize methods based on context characteristics
        if data_richness in ["rich", "very_rich"] and pattern_visibility == "apparent":
            # For rich, visible patterns, emphasize synthesis and meta-recognition
            priority_methods = [
                "pattern_synthesis",
                "meta_recognition",
                "principle_extraction",
            ]
        elif generalization_potential in ["high", "very_high"]:
            # For high generalization potential, focus on principles and analogies
            priority_methods = [
                "principle_extraction",
                "analogical_extension",
                "creative_synthesis",
            ]
        elif pattern_visibility == "hidden":
            # For hidden patterns, use creative and emergent approaches
            priority_methods = [
                "creative_synthesis",
                "emergent_insight",
                "meta_recognition",
            ]
        else:
            # For moderate complexity, use balanced approach
            priority_methods = [
                "pattern_synthesis",
                "principle_extraction",
                "creative_synthesis",
            ]

        # Select methods ensuring priority ones are included
        selected: List[Dict[str, Any]] = []
        method_dict = {m["name"]: m for m in methods}

        # Add priority methods first
        for method_name in priority_methods:
            if method_name in method_dict and len(selected) < max_methods:
                selected.append(method_dict[method_name])

        # Fill remaining slots with other methods
        for method in methods:
            if method not in selected and len(selected) < max_methods:
                selected.append(method)

        return selected[:max_methods]

    async def _apply_inductive_method(
        self,
        problem_statement: str,
        context: Optional[str],
        method: Dict[str, Any],
        synthesis_context: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Generate insights using a specific inductive method."""

        method_name = method["name"]
        method_description = method["description"]
        method_focus = method["focus"]
        cognitive_approach = method["cognitive_approach"]

        # Create method-specific system prompt
        system_prompt = f"""You are an expert insight synthesizer specializing in {method_name} inductive reasoning.

Your role is to generate profound insights using {cognitive_approach} approaches, focusing on {method_focus}.

Method Description: {method_description}

Synthesis Context:
- Data Richness: {synthesis_context.get('data_richness', 'moderate')}
- Pattern Visibility: {synthesis_context.get('pattern_visibility', 'subtle')}
- Generalization Potential: {synthesis_context.get('generalization_potential', 'medium')}
- Synthesis Depth: {synthesis_context.get('synthesis_depth_needed', 'moderate')}

Generate 3-5 high-quality insights that:
1. Apply {method_name} reasoning to synthesize understanding from observations
2. Identify patterns, principles, or connections not immediately obvious
3. Generate actionable insights with practical implications
4. Consider multiple levels of abstraction and generalization
5. Are creative yet grounded in observable evidence

Format your response as a JSON array of objects, each containing:
{{
    "insight": "the key insight or principle discovered",
    "synthesis_process": "how this insight was synthesized from observations",
    "supporting_patterns": "observable patterns that support this insight",
    "generalization_scope": "how broadly this insight might apply",
    "practical_implications": "practical applications of this insight",
    "confidence_level": "high|medium|low - confidence in the insight validity"
}}"""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'No additional context provided'}

Using {method_name} inductive reasoning, synthesize insights that identify patterns, extract principles, or generate creative understanding from the available information. Focus on {method_focus} while maintaining {cognitive_approach} rigor."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1200,
                temperature=0.7,  # Higher creativity for insight generation
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction
            from ...core.json_utils import parse_json_list
            insights_data = parse_json_list(response.content, [])

            generated_insights = []
            # Distribute cost across all generated insights from this API call
            cost_per_insight = (
                response.cost / len(insights_data) if insights_data else 0
            )

            for i, i_data in enumerate(insights_data):
                # Map confidence level to numeric score
                confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.85}
                confidence_score = confidence_map.get(
                    i_data.get("confidence_level", "medium"), 0.7
                )

                idea = GeneratedIdea(
                    content=i_data["insight"],
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"{method_name} method for: {problem_statement[:100]}...",
                    confidence_score=confidence_score,
                    reasoning=i_data["synthesis_process"],
                    metadata={
                        "method": method_name,
                        "supporting_patterns": i_data.get("supporting_patterns"),
                        "generalization_scope": i_data.get("generalization_scope"),
                        "practical_implications": i_data.get("practical_implications"),
                        "confidence_level": i_data.get("confidence_level", "medium"),
                        "cognitive_approach": cognitive_approach,
                        "llm_cost": cost_per_insight,
                        "batch_cost": response.cost,
                        "generation_index": i,
                    },
                    timestamp=datetime.now().isoformat(),
                )
                generated_insights.append(idea)

            return generated_insights

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse insights JSON for method {method_name}")
            return []
        except Exception as e:
            logger.error(f"Insight generation failed for method {method_name}: {e}")
            # Re-raise the exception so it can be handled at the agent level
            raise

    async def _rank_and_select_insights(
        self,
        insights: List[GeneratedIdea],
        max_insights: int,
        problem_statement: str,
        synthesis_context: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Rank and select the best insights using AI evaluation."""

        if len(insights) <= max_insights:
            return insights

        # Create ranking system prompt
        system_prompt = """You are an expert insight evaluator specializing in inductive reasoning quality assessment. Rank the given insights based on their depth, applicability, and value for understanding.

Evaluation criteria:
1. Depth and profundity of insight (30%)
2. Practical applicability and usefulness (25%)
3. Novelty and creative synthesis (20%)
4. Generalizability and broader relevance (15%)
5. Evidence grounding and validity (10%)

Consider that inductive insights should reveal patterns, principles, or connections that enhance understanding and provide actionable value.

Provide rankings as a JSON array of insight indices (0-based) in order from best to worst."""

        # Format insights for evaluation
        insights_text = "\n".join(
            [
                f"{i}. {ins.content[:150]}... (Method: {ins.metadata.get('method', 'unknown')}, Confidence: {ins.metadata.get('confidence_level', 'medium')})"
                for i, ins in enumerate(insights)
            ]
        )

        user_prompt = f"""Problem: {problem_statement}
Data Richness: {synthesis_context.get('data_richness', 'moderate')}
Generalization Potential: {synthesis_context.get('generalization_potential', 'medium')}

Insights to rank:
{insights_text}

Rank these insights from best to worst based on the evaluation criteria."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.2,  # Low temperature for consistent ranking
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)
            from ...core.json_utils import parse_json_list
            rankings = parse_json_list(response.content, list(range(len(insights))))

            # Select top insights based on rankings
            selected_insights = []
            for rank_idx in rankings[:max_insights]:
                if 0 <= rank_idx < len(insights):
                    insights[rank_idx].metadata["ranking_score"] = len(
                        rankings
                    ) - rankings.index(rank_idx)
                    selected_insights.append(insights[rank_idx])

            return selected_insights

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Insight ranking failed, using fallback selection: {e}")
            # Fallback: return first max_insights, ensuring method diversity
            methods_used: set = set()
            selected: List[GeneratedIdea] = []

            for insight in insights:
                method = insight.metadata.get("method", "unknown")
                if len(selected) < max_insights:
                    if (
                        method not in methods_used
                        or len(methods_used) >= max_insights // 2
                    ):
                        selected.append(insight)
                        methods_used.add(method)

            return selected[:max_insights]
