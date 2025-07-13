"""
LLM-Powered Deductive Agent for intelligent logical validation and systematic reasoning.

This agent uses Large Language Models to apply sophisticated deductive reasoning,
logical validation, and systematic analysis to problems and hypotheses.
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
from ...core.json_utils import parse_json_list, safe_json_parse
from ...core.llm_provider import (
    LLMManager,
    LLMProvider,
    LLMRequest,
    llm_manager,
)

logger = logging.getLogger(__name__)


class LLMDeductiveAgent(ThinkingAgentInterface):
    """
    LLM-powered agent that applies sophisticated deductive reasoning and logical validation.

    This agent uses artificial intelligence to:
    - Apply formal logical reasoning and validation frameworks
    - Conduct systematic consequence analysis and requirement validation
    - Perform structured proof development and logical chain construction
    - Validate hypotheses against evidence and logical consistency
    - Identify logical implications, requirements, and constraints
    """

    def __init__(
        self,
        name: str = "LLMDeductiveAgent",
        llm_manager: Optional[LLMManager] = None,
        preferred_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the LLM-powered deductive agent.

        Args:
            name: Unique name for this agent
            llm_manager: LLM manager instance (uses global if None)
            preferred_provider: Preferred LLM provider (auto-select if None)
        """
        self._name = name
        from ...core.llm_provider import llm_manager as default_llm_manager

        self.llm_manager = llm_manager or default_llm_manager
        self.preferred_provider = preferred_provider
        self._deductive_frameworks = self._load_deductive_frameworks()

    @property
    def name(self) -> str:
        """Unique name for this thinking agent."""
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        """The thinking method this agent implements."""
        return ThinkingMethod.DEDUCTION

    @property
    def supported_output_types(self) -> List[OutputType]:
        """Output types this agent can work with."""
        return [OutputType.TEXT, OutputType.STRUCTURED]

    @property
    def is_llm_powered(self) -> bool:
        """Whether this agent uses LLM services for generation."""
        return True

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is valid for this agent."""
        valid_keys = {
            "deductive_framework",
            "logical_depth",
            "validation_rigor",
            "include_counterarguments",
            "systematic_analysis",
            "formal_logic",
            "evidence_requirements",
            "max_analyses_per_framework",
            "include_proof_structure",
            "constraint_validation",
            "logical_chain_depth",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """
        Generate logical analyses using LLM deductive reasoning.

        Args:
            request: The idea generation request

        Returns:
            Result containing AI-generated logical analyses as ideas
        """
        start_time = asyncio.get_event_loop().time()

        logger.info(
            f"{self.name} applying logical reasoning to: {request.problem_statement[:100]}..."
        )

        try:
            # Analyze the logical structure of the problem
            logical_analysis = await self._analyze_logical_structure(
                request.problem_statement, request.context
            )

            # Generate logical analyses using multiple deductive frameworks
            all_analyses = []
            config = request.generation_config or {}

            # Select deductive frameworks based on config and analysis
            frameworks = self._select_deductive_frameworks(logical_analysis, config)

            # Track critical errors
            critical_errors = []

            for framework in frameworks:
                try:
                    analyses = await self._apply_deductive_framework(
                        request.problem_statement,
                        request.context,
                        framework,
                        logical_analysis,
                        config,
                    )
                    all_analyses.extend(analyses)
                except Exception as framework_error:
                    logger.error(
                        f"Framework {framework['name']} failed: {framework_error}"
                    )
                    critical_errors.append(str(framework_error))
                    continue

            # If all frameworks failed with the same error, it's likely a system issue
            if len(critical_errors) == len(frameworks) and critical_errors:
                # Check if all errors are the same (indicating a system-wide issue)
                unique_errors = set(critical_errors)
                if len(unique_errors) == 1:
                    raise Exception(critical_errors[0])

            # Limit and rank analyses
            max_analyses = request.max_ideas_per_method
            final_analyses = await self._rank_and_select_analyses(
                all_analyses,
                max_analyses,
                request.problem_statement,
                logical_analysis,
            )

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time

            logger.info(
                f"{self.name} generated {len(final_analyses)} logical analyses in {execution_time:.2f}s"
            )

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=final_analyses,
                execution_time=execution_time,
                generation_metadata={
                    "logical_analysis": logical_analysis,
                    "frameworks_used": [f["name"] for f in frameworks],
                    "total_generated": len(all_analyses),
                    "final_selected": len(final_analyses),
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

    async def _analyze_logical_structure(
        self, problem_statement: str, context: Optional[str] = None
    ) -> Any:
        """Analyze the logical structure and requirements of the problem."""
        system_prompt = """You are an expert logical analyst specializing in deductive reasoning. Analyze the given problem to understand its logical structure, requirements, and validation criteria. Provide your analysis in the following JSON format:

{{
    "logical_complexity": "simple|moderate|complex|highly_complex",
    "problem_type": "well_defined|structured|semi_structured|ill_defined",
    "evidence_base": "strong|moderate|weak|insufficient",
    "logical_requirements": ["key", "logical", "requirements"],
    "validation_criteria": ["criteria", "for", "validation"],
    "constraint_types": ["constraints", "and", "limitations"],
    "reasoning_chain_depth": "shallow|moderate|deep|very_deep",
    "formal_logic_applicable": true|false,
    "proof_structure_needed": true|false,
    "counterargument_potential": "low|medium|high",
    "systematic_analysis_scope": ["areas", "requiring", "systematic", "analysis"]
}}

Focus on aspects that will enable rigorous deductive reasoning and logical validation."""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'None provided'}

Analyze this problem to identify its logical structure, requirements, and the type of deductive reasoning most appropriate for validation and analysis."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.2,  # Low temperature for analytical precision
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction

            fallback_analysis = {
                "logical_complexity": "moderate",
                "problem_type": "semi_structured",
                "evidence_base": "moderate",
                "formal_logic_applicable": False,
                "reasoning_chain_depth": "moderate",
                "systematic_analysis_scope": ["validation", "consistency"],
            }
            analysis = safe_json_parse(response.content, fallback_analysis)
            analysis["llm_cost"] = response.cost

            return analysis

        except Exception as e:
            logger.error(f"Logical structure analysis failed: {e}")
            return {
                "logical_complexity": "moderate",
                "problem_type": "semi_structured",
                "evidence_base": "moderate",
                "formal_logic_applicable": False,
                "reasoning_chain_depth": "moderate",
                "systematic_analysis_scope": ["validation", "consistency"],
            }

    def _load_deductive_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Load different deductive reasoning frameworks and their configurations."""
        return {
            "logical_validation": {
                "name": "logical_validation",
                "description": "Apply formal logical validation to test consistency and validity",
                "focus": "validity, consistency, logical soundness, proof verification",
                "cognitive_approach": "formal logical, systematic validation",
            },
            "consequence_analysis": {
                "name": "consequence_analysis",
                "description": "Systematically analyze logical consequences and implications",
                "focus": "implications, consequences, causal chains, outcomes",
                "cognitive_approach": "systematic, consequence-driven",
            },
            "requirement_validation": {
                "name": "requirement_validation",
                "description": "Validate necessary and sufficient conditions and requirements",
                "focus": "requirements, conditions, prerequisites, dependencies",
                "cognitive_approach": "requirement-focused, condition-based",
            },
            "constraint_analysis": {
                "name": "constraint_analysis",
                "description": "Analyze constraints, limitations, and boundary conditions",
                "focus": "constraints, limitations, boundaries, feasibility",
                "cognitive_approach": "constraint-driven, feasibility-focused",
            },
            "proof_construction": {
                "name": "proof_construction",
                "description": "Construct formal proofs and logical argument chains",
                "focus": "proofs, logical chains, argumentation, demonstration",
                "cognitive_approach": "proof-oriented, demonstrative",
            },
            "systematic_decomposition": {
                "name": "systematic_decomposition",
                "description": "Systematically decompose problems into logical components",
                "focus": "decomposition, components, structure, hierarchy",
                "cognitive_approach": "systematic, structural",
            },
            "evidence_validation": {
                "name": "evidence_validation",
                "description": "Validate evidence against logical criteria and standards",
                "focus": "evidence, validation, standards, verification",
                "cognitive_approach": "evidence-based, verification-focused",
            },
        }

    def _select_deductive_frameworks(
        self, logical_analysis: Dict[str, Any], config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select appropriate deductive frameworks based on logical analysis and config."""
        frameworks = list(self._deductive_frameworks.values())

        # Default to 3-4 frameworks for comprehensive coverage
        max_frameworks = config.get("max_frameworks", 4)

        # Customize framework selection based on logical analysis
        logical_complexity = logical_analysis.get("logical_complexity", "moderate")
        evidence_base = logical_analysis.get("evidence_base", "moderate")
        formal_logic_applicable = logical_analysis.get("formal_logic_applicable", True)

        # Prioritize frameworks based on problem characteristics
        if formal_logic_applicable and logical_complexity in [
            "complex",
            "highly_complex",
        ]:
            # For complex formal problems, prioritize validation and proof construction
            priority_frameworks = [
                "logical_validation",
                "proof_construction",
                "systematic_decomposition",
            ]
        elif evidence_base in ["strong", "moderate"]:
            # For problems with good evidence, emphasize validation and consequences
            priority_frameworks = [
                "evidence_validation",
                "consequence_analysis",
                "requirement_validation",
            ]
        elif logical_complexity == "simple":
            # For simple problems, focus on validation and requirements
            priority_frameworks = [
                "logical_validation",
                "requirement_validation",
                "constraint_analysis",
            ]
        else:
            # For moderate complexity, use balanced approach
            priority_frameworks = [
                "logical_validation",
                "consequence_analysis",
                "systematic_decomposition",
            ]

        # Select frameworks ensuring priority ones are included
        selected: List[Dict[str, Any]] = []
        framework_dict = {f["name"]: f for f in frameworks}

        # Add priority frameworks first
        for framework_name in priority_frameworks:
            if framework_name in framework_dict and len(selected) < max_frameworks:
                selected.append(framework_dict[framework_name])

        # Fill remaining slots with other frameworks
        for framework in frameworks:
            if framework not in selected and len(selected) < max_frameworks:
                selected.append(framework)

        return selected[:max_frameworks]

    async def _apply_deductive_framework(
        self,
        problem_statement: str,
        context: Optional[str],
        framework: Dict[str, Any],
        logical_analysis: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Generate logical analyses using a specific deductive framework."""

        framework_name = framework["name"]
        framework_description = framework["description"]
        framework_focus = framework["focus"]
        cognitive_approach = framework["cognitive_approach"]

        # Create framework-specific system prompt
        system_prompt = f"""You are an expert logical reasoner specializing in {framework_name} deductive analysis.

Your role is to apply rigorous {cognitive_approach} reasoning, focusing on {framework_focus}.

Framework Description: {framework_description}

Problem Context:
- Logical Complexity: {logical_analysis.get('logical_complexity', 'moderate')}
- Evidence Base: {logical_analysis.get('evidence_base', 'moderate')}
- Reasoning Depth: {logical_analysis.get('reasoning_chain_depth', 'moderate')}
- Formal Logic Applicable: {logical_analysis.get('formal_logic_applicable', True)}

Generate 3-5 high-quality logical analyses that:
1. Apply {framework_name} reasoning systematically to the problem
2. Provide rigorous logical validation and verification
3. Identify clear premises, reasoning steps, and conclusions
4. Consider logical implications and potential counterarguments
5. Are structured, verifiable, and logically sound

Format your response as a JSON array of objects, each containing:
{{
    "analysis": "the logical analysis or conclusion",
    "reasoning_chain": "step-by-step logical reasoning process",
    "premises": "key premises and assumptions underlying the analysis",
    "implications": "logical implications and consequences",
    "validation_criteria": "criteria for validating this analysis",
    "confidence_level": "high|medium|low - logical confidence in this analysis"
}}"""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'No additional context provided'}

Using {framework_name} deductive reasoning, generate systematic logical analyses that validate, examine consequences, or establish requirements for this problem. Focus on {framework_focus} while maintaining {cognitive_approach} rigor."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1200,
                temperature=0.3,  # Low temperature for logical precision
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            # Parse JSON response with robust extraction

            analyses_data = parse_json_list(response.content, [])

            generated_analyses = []
            # Distribute cost across all generated analyses from this API call
            cost_per_analysis = (
                response.cost / len(analyses_data) if analyses_data else 0
            )

            for i, a_data in enumerate(analyses_data):
                # Map confidence level to numeric score
                confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.9}
                confidence_score = confidence_map.get(
                    a_data.get("confidence_level", "medium"), 0.7
                )

                idea = GeneratedIdea(
                    content=a_data["analysis"],
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"{framework_name} framework for: {problem_statement[:100]}...",
                    confidence_score=confidence_score,
                    reasoning=a_data["reasoning_chain"],
                    metadata={
                        "framework": framework_name,
                        "premises": a_data.get("premises"),
                        "implications": a_data.get("implications"),
                        "validation_criteria": a_data.get("validation_criteria"),
                        "confidence_level": a_data.get("confidence_level", "medium"),
                        "cognitive_approach": cognitive_approach,
                        "llm_cost": cost_per_analysis,
                        "batch_cost": response.cost,
                        "generation_index": i,
                    },
                    timestamp=datetime.now().isoformat(),
                )
                generated_analyses.append(idea)

            return generated_analyses

        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse analyses JSON for framework {framework_name}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Analysis generation failed for framework {framework_name}: {e}"
            )
            # Re-raise the exception so it can be handled at the agent level
            raise

    async def _rank_and_select_analyses(
        self,
        analyses: List[GeneratedIdea],
        max_analyses: int,
        problem_statement: str,
        logical_analysis: Dict[str, Any],
    ) -> List[GeneratedIdea]:
        """Rank and select the best logical analyses using AI evaluation."""

        if len(analyses) <= max_analyses:
            return analyses

        # Create ranking system prompt
        system_prompt = """You are an expert logical analysis evaluator. Rank the given logical analyses based on their rigor, validity, and usefulness for problem solving.

Evaluation criteria:
1. Logical soundness and validity (30%)
2. Systematic rigor and thoroughness (25%)
3. Practical applicability and usefulness (20%)
4. Clarity and structure (15%)
5. Insight value and depth (10%)

Consider that deductive analyses should be logically sound, systematic, and provide clear validation or implications.

Provide rankings as a JSON array of analysis indices (0-based) in order from best to worst."""

        # Format analyses for evaluation
        analyses_text = "\n".join(
            [
                f"{i}. {a.content[:150]}... (Framework: {a.metadata.get('framework', 'unknown')}, Confidence: {a.metadata.get('confidence_level', 'medium')})"
                for i, a in enumerate(analyses)
            ]
        )

        user_prompt = f"""Problem: {problem_statement}
Logical Complexity: {logical_analysis.get('logical_complexity', 'moderate')}

Analyses to rank:
{analyses_text}

Rank these logical analyses from best to worst based on the evaluation criteria."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.2,  # Low temperature for consistent ranking
            )

            response = await self.llm_manager.generate(request, self.preferred_provider)

            rankings = parse_json_list(response.content, list(range(len(analyses))))

            # Select top analyses based on rankings
            selected_analyses = []
            for rank_idx in rankings[:max_analyses]:
                if 0 <= rank_idx < len(analyses):
                    analyses[rank_idx].metadata["ranking_score"] = len(
                        rankings
                    ) - rankings.index(rank_idx)
                    selected_analyses.append(analyses[rank_idx])

            return selected_analyses

        except Exception as e:
            logger.warning(f"Analysis ranking failed, using fallback selection: {e}")
            # Fallback: return first max_analyses, ensuring framework diversity
            frameworks_used: set = set()
            selected: List[GeneratedIdea] = []

            for analysis in analyses:
                framework = analysis.metadata.get("framework", "unknown")
                if len(selected) < max_analyses:
                    if (
                        framework not in frameworks_used
                        or len(frameworks_used) >= max_analyses // 2
                    ):
                        selected.append(analysis)
                        frameworks_used.add(framework)

            return selected[:max_analyses]
