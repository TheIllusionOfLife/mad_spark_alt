"""
Questioning Agent for diverse question generation and problem framing.

This agent implements the "Questioning" phase of the QADI cycle, generating
diverse questions to explore different angles and perspectives on a problem.
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


class QuestioningAgent(ThinkingAgentInterface):
    """
    Agent that generates diverse questions to explore and frame problems.

    This agent focuses on:
    - Generating different types of questions (What, Why, How, When, Where, Who)
    - Exploring multiple perspectives and stakeholder viewpoints
    - Identifying assumptions and constraints
    - Framing the problem from various angles
    """

    def __init__(self, name: str = "QuestioningAgent"):
        """Initialize the questioning agent."""
        self._name = name
        self._question_templates = self._load_question_templates()

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
            "question_types",
            "max_questions_per_type",
            "include_assumptions",
            "stakeholder_perspectives",
            "use_templates",
            "creativity_level",
        }
        return all(key in valid_keys for key in config.keys())

    async def generate_ideas(
        self, request: IdeaGenerationRequest
    ) -> IdeaGenerationResult:
        """
        Generate questions to explore and frame the problem.

        Args:
            request: The idea generation request

        Returns:
            Result containing generated questions as ideas
        """
        start_time = asyncio.get_running_loop().time()

        logger.info(
            f"QuestioningAgent generating questions for: {request.problem_statement[:100]}..."
        )

        try:
            generated_questions = []
            config = request.generation_config

            # Generate different types of questions
            question_types = config.get(
                "question_types",
                [
                    "what",
                    "why",
                    "how",
                    "when",
                    "where",
                    "who",
                    "assumptions",
                    "perspectives",
                ],
            )

            max_per_type = config.get("max_questions_per_type", 2)

            for question_type in question_types:
                questions = await self._generate_questions_by_type(
                    request.problem_statement,
                    question_type,
                    max_per_type,
                    request.context,
                    config,
                )
                generated_questions.extend(questions)

            # Limit total questions
            max_total = min(request.max_ideas_per_method, len(generated_questions))
            generated_questions = generated_questions[:max_total]

            end_time = asyncio.get_running_loop().time()
            execution_time = end_time - start_time

            logger.info(
                f"QuestioningAgent generated {len(generated_questions)} questions in {execution_time:.2f}s"
            )

            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=generated_questions,
                execution_time=execution_time,
                generation_metadata={
                    "question_types_used": question_types,
                    "total_generated": len(generated_questions),
                    "config": config,
                },
            )

        except Exception as e:
            logger.error(f"Error in QuestioningAgent: {e}")
            return IdeaGenerationResult(
                agent_name=self.name,
                thinking_method=self.thinking_method,
                generated_ideas=[],
                error_message=str(e),
            )

    async def _generate_questions_by_type(
        self,
        problem_statement: str,
        question_type: str,
        max_questions: int,
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[GeneratedIdea]:
        """Generate questions of a specific type."""
        templates = self._question_templates.get(question_type, [])
        if not templates:
            logger.warning(f"No templates found for question type: {question_type}")
            return []

        questions = []
        creativity_level = (
            config.get("creativity_level", "balanced") if config else "balanced"
        )

        # Select templates based on creativity level
        selected_templates = self._select_templates(
            templates, creativity_level, max_questions
        )

        for i, template in enumerate(selected_templates[:max_questions]):
            try:
                question_content = template.format(
                    problem=problem_statement, context=context or "general context"
                )

                reasoning = f"Generated {question_type} question using template approach to explore {question_type}-related aspects of the problem."

                idea = GeneratedIdea(
                    content=question_content,
                    thinking_method=self.thinking_method,
                    agent_name=self.name,
                    generation_prompt=f"Generate {question_type} question for: {problem_statement}",
                    confidence_score=0.8,  # Template-based questions have good confidence
                    reasoning=reasoning,
                    metadata={
                        "question_type": question_type,
                        "template_index": i,
                        "creativity_level": creativity_level,
                    },
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                questions.append(idea)

            except Exception as e:
                logger.warning(f"Failed to generate {question_type} question: {e}")
                continue

        return questions

    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different question types."""
        return {
            "what": [
                "What are the core elements of {problem}?",
                "What assumptions are we making about {problem}?",
                "What would success look like for {problem}?",
                "What are the key constraints affecting {problem}?",
                "What are we not seeing about {problem}?",
            ],
            "why": [
                "Why is {problem} important to solve?",
                "Why has {problem} not been solved before?",
                "Why might our current approach fail?",
                "Why should we prioritize {problem} over alternatives?",
                "Why do stakeholders care about {problem}?",
            ],
            "how": [
                "How might we approach {problem} differently?",
                "How would experts in other fields solve {problem}?",
                "How can we measure progress on {problem}?",
                "How might technology change our approach to {problem}?",
                "How would we solve {problem} with unlimited resources?",
            ],
            "when": [
                "When is the ideal time to address {problem}?",
                "When do the effects of {problem} become most apparent?",
                "When should we expect to see results from addressing {problem}?",
                "When might conditions be different enough to change our approach?",
            ],
            "where": [
                "Where do we see {problem} manifesting most clearly?",
                "Where might we find unexpected solutions to {problem}?",
                "Where are the leverage points for addressing {problem}?",
                "Where else has a similar problem been solved?",
            ],
            "who": [
                "Who are all the stakeholders affected by {problem}?",
                "Who might have insights we haven't considered about {problem}?",
                "Who benefits from {problem} remaining unsolved?",
                "Who would be the ideal team to address {problem}?",
            ],
            "assumptions": [
                "What if our basic assumptions about {problem} are wrong?",
                "What if {problem} is actually a symptom of something else?",
                "What if the opposite approach to {problem} worked better?",
                "What if {problem} solved itself over time?",
            ],
            "perspectives": [
                "How would a child approach {problem}?",
                "How would someone from a different culture view {problem}?",
                "How would future generations judge our approach to {problem}?",
                "How would nature solve a problem similar to {problem}?",
            ],
        }

    def _select_templates(
        self, templates: List[str], creativity_level: str, max_count: int
    ) -> List[str]:
        """Select templates based on creativity level."""
        if creativity_level == "conservative":
            # Use more straightforward, proven question types
            return templates[:max_count]
        elif creativity_level == "creative":
            # Use more unconventional question approaches
            return templates[-max_count:] if len(templates) >= max_count else templates
        else:  # balanced
            # Mix of different template types
            step = max(1, len(templates) // max_count)
            return templates[::step][:max_count]
