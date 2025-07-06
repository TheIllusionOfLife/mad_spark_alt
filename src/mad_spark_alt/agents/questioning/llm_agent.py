"""
LLM-Powered Questioning Agent for intelligent question generation.

This agent uses Large Language Models to generate contextually relevant, 
sophisticated questions that explore problems from multiple cognitive perspectives.
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


class LLMQuestioningAgent(ThinkingAgentInterface):
    """
    LLM-powered agent that generates sophisticated questions to explore problems.
    
    This agent uses artificial intelligence to:
    - Generate contextually relevant questions based on domain analysis
    - Adapt questioning strategies to problem types and contexts  
    - Explore stakeholder perspectives with nuanced understanding
    - Identify hidden assumptions and constraints through AI reasoning
    - Generate meta-questions about the problem-solving process itself
    """

    def __init__(self, 
                 name: str = "LLMQuestioningAgent",
                 llm_manager: Optional[LLMManager] = None,
                 preferred_provider: Optional[LLMProvider] = None):
        """
        Initialize the LLM-powered questioning agent.
        
        Args:
            name: Unique name for this agent
            llm_manager: LLM manager instance (uses global if None)
            preferred_provider: Preferred LLM provider (auto-select if None)
        """
        self._name = name
        self.llm_manager = llm_manager or llm_manager
        self.preferred_provider = preferred_provider
        self._questioning_strategies = self._load_questioning_strategies()

    @property
    def name(self) -> str:
        """Unique name for this thinking agent."""
        return self._name

    @property
    def thinking_method(self) -> ThinkingMethod:
        """The thinking method this agent implements."""
        return ThinkingMethod.QUESTIONING

    @property
    def supported_output_types(self) -> List[OutputType]:
        """Output types this agent can work with."""
        return [OutputType.TEXT, OutputType.STRUCTURED]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is valid for this agent."""
        valid_keys = {
            "questioning_strategy",
            "domain_context",
            "stakeholder_focus",
            "question_depth",
            "creativity_level",
            "max_questions_per_strategy",
            "include_meta_questions",
            "use_domain_expertise",
            "perspective_diversity",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(self, request: IdeaGenerationRequest) -> IdeaGenerationResult:
        """
        Generate intelligent questions using LLM reasoning.
        
        Args:
            request: The idea generation request
            
        Returns:
            Result containing AI-generated questions as ideas
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"{self.name} generating questions for: {request.problem_statement[:100]}...")
        
        try:
            # Analyze the problem domain and context
            domain_analysis = await self._analyze_problem_domain(
                request.problem_statement, 
                request.context
            )
            
            # Generate questions using multiple strategies
            all_questions = []
            config = request.generation_config or {}
            
            # Select questioning strategies based on config and domain
            strategies = self._select_questioning_strategies(domain_analysis, config)
            
            for strategy in strategies:
                questions = await self._generate_questions_with_strategy(
                    request.problem_statement,
                    request.context,
                    strategy,
                    domain_analysis,
                    config
                )
                all_questions.extend(questions)
            
            # Limit and rank questions
            max_questions = request.max_ideas_per_method
            final_questions = await self._rank_and_select_questions(
                all_questions, 
                max_questions,
                request.problem_statement,
                domain_analysis
            )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            logger.info(f"{self.name} generated {len(final_questions)} questions in {execution_time:.2f}s")
            
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=final_questions,
                execution_time=execution_time,
                generation_metadata={
                    "domain_analysis": domain_analysis,
                    "strategies_used": [s["name"] for s in strategies],
                    "total_generated": len(all_questions),
                    "final_selected": len(final_questions),
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

    async def _analyze_problem_domain(self, problem_statement: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the problem domain to inform questioning strategy."""
        system_prompt = """You are an expert problem analyst. Analyze the given problem to understand its domain, complexity, and characteristics. Provide your analysis in the following JSON format:

{
    "domain": "primary domain (e.g., technology, healthcare, education, business)",
    "subdomain": "specific area within the domain",
    "complexity_level": "low|medium|high|very_high",
    "problem_type": "well_defined|ill_defined|wicked_problem",
    "stakeholder_groups": ["list", "of", "key", "stakeholder", "groups"],
    "key_constraints": ["list", "of", "likely", "constraints"],
    "knowledge_areas": ["relevant", "fields", "of", "expertise"],
    "temporal_aspects": "time-related characteristics",
    "scale": "local|regional|national|global",
    "interdisciplinary": true|false
}

Be precise and insightful in your analysis."""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'None provided'}

Analyze this problem and provide the domain analysis in the specified JSON format."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                temperature=0.3  # Lower temperature for analytical tasks
            )
            
            response = await self.llm_manager.generate(request, self.preferred_provider)
            
            # Parse JSON response
            analysis = json.loads(response.content)
            analysis["llm_cost"] = response.cost
            
            return analysis
            
        except json.JSONDecodeError:
            # Fallback to basic analysis if JSON parsing fails
            logger.warning("Failed to parse domain analysis JSON, using fallback")
            return {
                "domain": "general",
                "complexity_level": "medium",
                "problem_type": "ill_defined",
                "stakeholder_groups": ["users", "stakeholders"],
                "interdisciplinary": True
            }
        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return {"domain": "general", "complexity_level": "unknown"}

    def _load_questioning_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load different questioning strategies and their configurations."""
        return {
            "fundamental_inquiry": {
                "name": "fundamental_inquiry",
                "description": "Explore fundamental assumptions and core principles",
                "focus": "assumptions, principles, definitions",
                "cognitive_approach": "analytical, systematic"
            },
            "stakeholder_perspectives": {
                "name": "stakeholder_perspectives", 
                "description": "Examine the problem from multiple stakeholder viewpoints",
                "focus": "perspectives, interests, impacts",
                "cognitive_approach": "empathetic, multi-perspective"
            },
            "systemic_analysis": {
                "name": "systemic_analysis",
                "description": "Understand system dynamics and interconnections",
                "focus": "relationships, feedback loops, emergence",
                "cognitive_approach": "systems thinking, holistic"
            },
            "creative_reframing": {
                "name": "creative_reframing",
                "description": "Reframe the problem in novel ways",
                "focus": "alternative framings, analogies, metaphors",
                "cognitive_approach": "creative, divergent"
            },
            "constraint_exploration": {
                "name": "constraint_exploration",
                "description": "Identify and question constraints and limitations",
                "focus": "constraints, limitations, boundaries",
                "cognitive_approach": "constraint-focused, boundary-testing"
            },
            "temporal_analysis": {
                "name": "temporal_analysis",
                "description": "Explore time-related aspects and dynamics",
                "focus": "timing, sequence, evolution, urgency",
                "cognitive_approach": "temporal, evolutionary"
            },
            "meta_cognitive": {
                "name": "meta_cognitive",
                "description": "Question the problem-solving process itself",
                "focus": "methodology, approach, biases",
                "cognitive_approach": "reflective, meta-analytical"
            },
        }

    def _select_questioning_strategies(self, domain_analysis: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select appropriate questioning strategies based on domain and config."""
        strategies = list(self._questioning_strategies.values())
        
        # Default to 3-4 strategies for balanced coverage
        max_strategies = config.get("max_strategies", 4)
        
        # Customize strategy selection based on domain analysis
        complexity = domain_analysis.get("complexity_level", "medium")
        problem_type = domain_analysis.get("problem_type", "ill_defined")
        
        # Prioritize strategies based on problem characteristics
        if complexity in ["high", "very_high"]:
            # For complex problems, prioritize systemic and meta-cognitive approaches
            priority_strategies = ["systemic_analysis", "meta_cognitive", "fundamental_inquiry"]
        elif problem_type == "wicked_problem":
            # For wicked problems, emphasize reframing and stakeholder perspectives
            priority_strategies = ["creative_reframing", "stakeholder_perspectives", "systemic_analysis"]
        else:
            # For well-defined problems, focus on fundamental inquiry and constraints
            priority_strategies = ["fundamental_inquiry", "constraint_exploration", "stakeholder_perspectives"]
        
        # Select strategies ensuring priority ones are included
        selected = []
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

    async def _generate_questions_with_strategy(
        self,
        problem_statement: str,
        context: Optional[str],
        strategy: Dict[str, Any],
        domain_analysis: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[GeneratedIdea]:
        """Generate questions using a specific questioning strategy."""
        
        strategy_name = strategy["name"]
        strategy_description = strategy["description"]
        strategy_focus = strategy["focus"]
        cognitive_approach = strategy["cognitive_approach"]
        
        # Create strategy-specific system prompt
        system_prompt = f"""You are an expert question generator specializing in {strategy_name} questioning. 

Your role is to generate insightful, thought-provoking questions using a {cognitive_approach} approach, focusing on {strategy_focus}.

Strategy Description: {strategy_description}

Domain Context: {domain_analysis.get('domain', 'general')} ({domain_analysis.get('subdomain', 'unspecified')})
Problem Complexity: {domain_analysis.get('complexity_level', 'medium')}
Stakeholders: {', '.join(domain_analysis.get('stakeholder_groups', []))}

Generate 3-5 high-quality questions that:
1. Are specifically relevant to the problem domain and context
2. Follow the {strategy_name} questioning approach
3. Are actionable and lead to deeper understanding
4. Avoid generic or obvious questions
5. Consider the complexity level and stakeholder perspectives

Format your response as a JSON array of objects, each containing:
{
    "question": "the actual question text",
    "reasoning": "why this question is important and how it applies the strategy",
    "focus_area": "specific aspect this question targets",
    "stakeholder_relevance": "which stakeholders this question most affects"
}"""

        user_prompt = f"""Problem Statement: {problem_statement}

Additional Context: {context or 'No additional context provided'}

Using the {strategy_name} approach, generate insightful questions that will help explore and understand this problem more deeply. Focus on {strategy_focus} aspects while maintaining a {cognitive_approach} perspective."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1200,
                temperature=0.7  # Balanced creativity for question generation
            )
            
            response = await self.llm_manager.generate(request, self.preferred_provider)
            
            # Parse JSON response
            questions_data = json.loads(response.content)
            
            generated_questions = []
            for i, q_data in enumerate(questions_data):
                idea = GeneratedIdea(
                    content=q_data["question"],
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"{strategy_name} strategy for: {problem_statement[:100]}...",
                    confidence_score=0.85,  # LLM-generated questions generally high confidence
                    reasoning=q_data["reasoning"],
                    metadata={
                        "strategy": strategy_name,
                        "focus_area": q_data.get("focus_area"),
                        "stakeholder_relevance": q_data.get("stakeholder_relevance"),
                        "cognitive_approach": cognitive_approach,
                        "llm_cost": response.cost,
                        "generation_index": i,
                    },
                    timestamp=datetime.now().isoformat(),
                )
                generated_questions.append(idea)
            
            return generated_questions
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse questions JSON for strategy {strategy_name}")
            return []
        except Exception as e:
            logger.error(f"Question generation failed for strategy {strategy_name}: {e}")
            return []

    async def _rank_and_select_questions(
        self,
        questions: List[GeneratedIdea],
        max_questions: int,
        problem_statement: str,
        domain_analysis: Dict[str, Any]
    ) -> List[GeneratedIdea]:
        """Rank and select the best questions using AI evaluation."""
        
        if len(questions) <= max_questions:
            return questions
        
        # Create ranking system prompt
        system_prompt = """You are an expert question evaluator. Rank the given questions based on their quality, relevance, and potential to generate insights.

Evaluation criteria:
1. Relevance to the problem (30%)
2. Depth and sophistication (25%) 
3. Actionability and utility (20%)
4. Novelty and uniqueness (15%)
5. Stakeholder value (10%)

Provide rankings as a JSON array of question indices (0-based) in order from best to worst."""

        # Format questions for evaluation
        questions_text = "\n".join([
            f"{i}. {q.content} (Strategy: {q.metadata.get('strategy', 'unknown')})"
            for i, q in enumerate(questions)
        ])
        
        user_prompt = f"""Problem: {problem_statement}
Domain: {domain_analysis.get('domain', 'general')}

Questions to rank:
{questions_text}

Rank these questions from best to worst based on the evaluation criteria."""

        try:
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                temperature=0.2  # Low temperature for consistent ranking
            )
            
            response = await self.llm_manager.generate(request, self.preferred_provider)
            rankings = json.loads(response.content)
            
            # Select top questions based on rankings
            selected_questions = []
            for rank_idx in rankings[:max_questions]:
                if 0 <= rank_idx < len(questions):
                    questions[rank_idx].metadata["ranking_score"] = len(rankings) - rankings.index(rank_idx)
                    selected_questions.append(questions[rank_idx])
            
            return selected_questions
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Question ranking failed, using fallback selection: {e}")
            # Fallback: return first max_questions, ensuring strategy diversity
            strategies_used = set()
            selected = []
            
            for question in questions:
                strategy = question.metadata.get("strategy", "unknown")
                if len(selected) < max_questions:
                    if strategy not in strategies_used or len(strategies_used) >= max_questions // 2:
                        selected.append(question)
                        strategies_used.add(strategy)
            
            return selected[:max_questions]