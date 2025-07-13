"""
Answer Extractor for converting QADI insights into direct user answers.

This module bridges the gap between abstract QADI thinking and concrete
user-requested answers by analyzing question patterns and extracting
relevant solutions from QADI phases.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import GeneratedIdea
from .json_utils import safe_json_parse
from .llm_provider import llm_manager, LLMRequest

logger = logging.getLogger(__name__)


def _validate_prompt_content(prompt: str) -> str:
    """Validate and sanitize prompt content for LLM safety."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Remove potentially harmful content
    sanitized = prompt.strip()
    
    # Basic length validation (prevent extremely long prompts)
    if len(sanitized) > 50000:  # 50k character limit
        logger.warning(f"Prompt truncated from {len(sanitized)} to 50000 characters")
        sanitized = sanitized[:50000] + "..."
    
    # Remove null bytes and other control characters
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
    
    return sanitized


@dataclass
class ExtractedAnswer:
    """Represents a direct answer extracted from QADI insights."""

    content: str
    confidence: float
    source_phase: str
    source_ideas: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class AnswerExtractionResult:
    """Complete result from answer extraction process."""

    original_question: str
    question_type: str
    direct_answers: List[ExtractedAnswer]
    summary: str
    total_qadi_ideas: int
    extraction_method: str = "template"


class QuestionTypeAnalyzer:
    """Analyzes user questions to determine expected answer format."""

    QUESTION_PATTERNS = {
        "list_request": [
            r"\b(\d+)\s+(ways?|methods?|steps?|approaches?|strategies?|solutions?|ideas?)",
            r"list\s+\d+",
            r"what are.*ways?",
            r"how to.*\?.*\d+",
            r"give me.*\d+",
        ],
        "how_to": [
            r"^how\s+(to|can|do)",
            r"ways?\s+to",
            r"methods?\s+to",
            r"steps?\s+to",
        ],
        "what_is": [
            r"^what\s+(is|are)",
            r"define",
            r"explain",
        ],
        "why": [
            r"^why",
            r"reason",
            r"cause",
        ],
        "best_practice": [
            r"best\s+(way|method|practice|approach)",
            r"most\s+effective",
            r"optimal",
        ],
    }

    def analyze_question(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze question to determine type and extraction strategy.

        Returns:
            Tuple of (question_type, metadata)
        """
        question_lower = question.lower().strip()

        # Extract requested quantity if present
        quantity_match = re.search(r"\b(\d+)\s+", question_lower)
        requested_quantity = int(quantity_match.group(1)) if quantity_match else 3

        # Determine question type
        for q_type, patterns in self.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type, {
                        "requested_quantity": requested_quantity,
                        "expects_list": q_type in ["list_request", "how_to"],
                        "expects_explanation": q_type in ["what_is", "why"],
                        "pattern_matched": pattern,
                    }

        # Default to how_to if uncertain
        return "how_to", {
            "requested_quantity": requested_quantity,
            "expects_list": True,
            "expects_explanation": False,
            "pattern_matched": "default",
        }


class TemplateAnswerExtractor:
    """Extracts concrete answers from QADI results using template patterns."""

    def __init__(self) -> None:
        self.question_analyzer = QuestionTypeAnalyzer()

    def extract_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: Optional[int] = None,
    ) -> AnswerExtractionResult:
        """
        Extract direct answers from QADI results.

        Args:
            question: Original user question
            qadi_results: QADI ideas grouped by phase
            max_answers: Maximum number of answers to extract

        Returns:
            AnswerExtractionResult with direct answers
        """
        question_type, metadata = self.question_analyzer.analyze_question(question)
        requested_quantity = metadata.get("requested_quantity", 3)

        if max_answers is None:
            max_answers = requested_quantity

        logger.info(f"Extracting {max_answers} answers for {question_type} question")

        # Strategy based on question type
        if question_type == "list_request":
            answers = self._extract_list_answers(question, qadi_results, max_answers)
        elif question_type == "how_to":
            answers = self._extract_how_to_answers(question, qadi_results, max_answers)
        elif question_type in ["what_is", "why"]:
            answers = self._extract_explanatory_answers(
                question, qadi_results, max_answers
            )
        else:
            answers = self._extract_general_answers(question, qadi_results, max_answers)

        # Generate summary
        summary = self._generate_summary(question, answers, question_type)

        return AnswerExtractionResult(
            original_question=question,
            question_type=question_type,
            direct_answers=answers,
            summary=summary,
            total_qadi_ideas=sum(len(ideas) for ideas in qadi_results.values()),
            extraction_method="template",
        )

    def _extract_list_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: int,
    ) -> List[ExtractedAnswer]:
        """Extract specific list-type answers."""
        answers: List[ExtractedAnswer] = []

        # Priority: Abduction (creative ideas) > Deduction (logical steps) > Questioning (insights)
        phase_priority = ["abduction", "deduction", "questioning", "induction"]

        for phase in phase_priority:
            if phase in qadi_results and len(answers) < max_answers:
                phase_ideas = qadi_results[phase]

                for idea in phase_ideas:
                    if len(answers) >= max_answers:
                        break

                    # Convert QADI idea to direct answer
                    direct_answer = self._convert_to_direct_answer(
                        idea, question, phase, len(answers) + 1
                    )

                    if direct_answer:
                        answers.append(direct_answer)

        # Fill remaining slots with synthesized answers if needed
        while len(answers) < max_answers:
            synthetic_answer = self._create_synthetic_answer(
                question, len(answers) + 1, qadi_results
            )
            answers.append(synthetic_answer)

        return answers[:max_answers]

    def _extract_how_to_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: int,
    ) -> List[ExtractedAnswer]:
        """Extract step-by-step answers for how-to questions."""
        return self._extract_list_answers(question, qadi_results, max_answers)

    def _extract_explanatory_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: int,
    ) -> List[ExtractedAnswer]:
        """Extract explanatory answers for what/why questions."""
        answers: List[ExtractedAnswer] = []

        # For explanatory questions, prioritize induction (patterns) and questioning (insights)
        phase_priority = ["induction", "questioning", "deduction", "abduction"]

        for phase in phase_priority:
            if phase in qadi_results and len(answers) < max_answers:
                for idea in qadi_results[phase]:
                    if len(answers) >= max_answers:
                        break

                    explanation = self._convert_to_explanation(idea, question, phase)
                    if explanation:
                        answers.append(explanation)

        return answers[:max_answers]

    def _extract_general_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: int,
    ) -> List[ExtractedAnswer]:
        """Extract general answers for unclassified questions."""
        return self._extract_list_answers(question, qadi_results, max_answers)

    def _convert_to_direct_answer(
        self, idea: GeneratedIdea, question: str, phase: str, answer_number: int
    ) -> Optional[ExtractedAnswer]:
        """Convert a QADI idea into a direct, actionable answer."""

        # Extract the core topic from the question
        topic = self._extract_topic_from_question(question)

        # Generate direct answer based on phase type
        if phase == "questioning":
            # Convert questions into action insights
            direct_content = self._convert_question_to_action(
                idea.content, topic, answer_number
            )
        elif phase == "abduction":
            # Convert hypotheses into creative solutions
            direct_content = self._convert_hypothesis_to_solution(
                idea.content, topic, answer_number
            )
        elif phase == "deduction":
            # Convert logical analysis into concrete steps
            direct_content = self._convert_logic_to_steps(
                idea.content, topic, answer_number
            )
        elif phase == "induction":
            # Convert patterns into strategic approaches
            direct_content = self._convert_pattern_to_strategy(
                idea.content, topic, answer_number
            )
        else:
            direct_content = f"Apply insights from {phase} analysis to {topic}"

        return ExtractedAnswer(
            content=direct_content,
            confidence=0.7,  # Template-based confidence
            source_phase=phase,
            source_ideas=[idea.content],
            reasoning=f"Derived from {phase} phase thinking about {topic}",
        )

    def _convert_to_explanation(
        self, idea: GeneratedIdea, question: str, phase: str
    ) -> Optional[ExtractedAnswer]:
        """Convert QADI idea into explanatory answer."""
        topic = self._extract_topic_from_question(question)
        qadi_content = idea.content

        # Use the actual QADI insight to generate contextual explanations
        if phase == "induction":
            # Use the pattern identified by induction
            explanation = f"Based on pattern analysis: {qadi_content}. This suggests that {topic} can be approached systematically."
        elif phase == "questioning":
            # Use the insights from questioning
            if "?" in qadi_content:
                # Convert question to explanatory insight
                insight = qadi_content.replace("?", "").strip()
                explanation = f"Key insight about {topic}: Understanding {insight} is fundamental to success."
            else:
                explanation = f"Important factor for {topic}: {qadi_content}"
        elif phase == "deduction":
            # Use logical analysis from deduction
            explanation = f"Logical analysis shows: {qadi_content}. This reasoning applies directly to {topic}."
        elif phase == "abduction":
            # Use creative hypothesis from abduction
            explanation = f"Creative insight: {qadi_content}. This hypothesis offers a new perspective on {topic}."
        else:
            explanation = f"From {phase} analysis: {qadi_content}"

        return ExtractedAnswer(
            content=explanation,
            confidence=0.6,
            source_phase=phase,
            source_ideas=[idea.content],
            reasoning=f"Explanatory insight from {phase} analysis",
        )

    def _extract_topic_from_question(self, question: str) -> str:
        """Extract main topic from user question."""
        # Remove question words and numbers
        topic = re.sub(
            r"^(what|how|why|when|where|who)\s+(are|is|to|can|do|does)\s*",
            "",
            question.lower(),
        )
        topic = re.sub(r"\b\d+\s+(ways?|methods?|steps?)\s*", "", topic)
        topic = re.sub(r"\?", "", topic)
        topic = topic.strip()

        # Take first meaningful phrase
        if len(topic) > 50:
            topic = topic[:50] + "..."

        return topic or "the challenge"

    def _convert_question_to_action(
        self, qadi_content: str, topic: str, number: int
    ) -> str:
        """Convert questioning phase output to actionable advice."""
        # Extract actionable insight from the QADI question
        if "?" in qadi_content:
            # Transform question into action
            action_words = qadi_content.replace("?", "").strip()
            return f"Investigate {action_words} to improve {topic}"
        return f"Based on questioning '{qadi_content}', explore {topic} systematically"

    def _convert_hypothesis_to_solution(
        self, qadi_content: str, topic: str, number: int
    ) -> str:
        """Convert abduction phase output to creative solution."""
        # Use the actual hypothesis from QADI
        if len(qadi_content) > 20:
            key_insight = (
                qadi_content[:100] + "..." if len(qadi_content) > 100 else qadi_content
            )
            return f"Implement hypothesis: {key_insight}"
        return f"Test the hypothesis '{qadi_content}' for {topic}"

    def _convert_logic_to_steps(
        self, qadi_content: str, topic: str, number: int
    ) -> str:
        """Convert deduction phase output to concrete steps."""
        # Transform logical deduction into actionable step
        if "if" in qadi_content.lower() or "then" in qadi_content.lower():
            return f"Step {number}: {qadi_content}"
        return f"Apply logical principle: {qadi_content} to {topic}"

    def _convert_pattern_to_strategy(
        self, qadi_content: str, topic: str, number: int
    ) -> str:
        """Convert induction phase output to strategic approach."""
        # Use the actual pattern identified by QADI
        if "pattern" in qadi_content.lower() or "trend" in qadi_content.lower():
            return f"Strategy: Leverage {qadi_content}"
        return f"Apply pattern: {qadi_content} systematically to {topic}"

    def _create_synthetic_answer(
        self, question: str, number: int, qadi_results: Dict[str, List[GeneratedIdea]]
    ) -> ExtractedAnswer:
        """Create synthetic answer when QADI doesn't provide enough, incorporating available insights."""
        topic = self._extract_topic_from_question(question)

        # Try to incorporate any available QADI insights even in synthetic answers
        available_insights = []
        for phase_ideas in qadi_results.values():
            for idea in phase_ideas:
                if idea.content and len(idea.content.strip()) > 10:
                    available_insights.append(idea.content[:100])

        if available_insights and number <= len(available_insights):
            # Use an available insight as a starting point
            insight = available_insights[number - 1]
            content = f"Building on the insight '{insight}', develop a systematic approach to {topic}"
        else:
            # Fall back to generic synthetic answers
            synthetic_answers = [
                f"Research and gather information about best practices for {topic}",
                f"Start with small, manageable steps toward {topic}",
                f"Seek expert advice and learn from others who have succeeded with {topic}",
                f"Measure progress and iterate your approach to {topic} based on results",
                f"Build a support system and resources to help you achieve {topic}",
            ]
            content = synthetic_answers[(number - 1) % len(synthetic_answers)]

        return ExtractedAnswer(
            content=content,
            confidence=0.5,
            source_phase="synthetic",
            source_ideas=[],
            reasoning="Generated to meet requested quantity when QADI insights were insufficient",
        )

    def _generate_summary(
        self, question: str, answers: List[ExtractedAnswer], question_type: str
    ) -> str:
        """Generate summary of extracted answers."""
        topic = self._extract_topic_from_question(question)

        if question_type == "list_request":
            return f"Extracted {len(answers)} practical approaches for {topic} from QADI analysis."
        elif question_type == "how_to":
            return f"Identified {len(answers)} key steps for {topic} based on systematic thinking."
        else:
            return f"Provided {len(answers)} insights about {topic} from multi-perspective analysis."


class EnhancedAnswerExtractor:
    """Enhanced extractor that can use both template and LLM approaches."""

    def __init__(self, prefer_llm: bool = True):
        self.prefer_llm = prefer_llm
        self.template_extractor = TemplateAnswerExtractor()

    async def extract_answers(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: Optional[int] = None,
    ) -> AnswerExtractionResult:
        """
        Extract answers using best available method.

        Args:
            question: Original user question
            qadi_results: QADI ideas grouped by phase
            max_answers: Maximum answers to extract

        Returns:
            AnswerExtractionResult with direct answers
        """
        # Try LLM-based extraction if preferred and available
        if self.prefer_llm and len(llm_manager.providers) > 0:
            try:
                return await self._extract_with_llm(question, qadi_results, max_answers)
            except (asyncio.TimeoutError, ValueError, json.JSONDecodeError) as e:
                logger.warning(f"LLM extraction failed, falling back to template: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error in LLM extraction, falling back to template: {e}")

        # Fallback to template extraction
        result = self.template_extractor.extract_answers(
            question, qadi_results, max_answers
        )
        result.extraction_method = "template"
        return result

    async def _extract_with_llm(
        self,
        question: str,
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: Optional[int],
    ) -> AnswerExtractionResult:
        """Extract answers using LLM analysis of QADI insights."""

        if max_answers is None:
            max_answers = 5

        # Build context from QADI results
        qadi_context = self._build_qadi_context(qadi_results)

        # Analyze question type for better prompt formatting
        analyzer = QuestionTypeAnalyzer()
        question_type, metadata = analyzer.analyze_question(question)

        # Create LLM prompt
        prompt = self._create_llm_extraction_prompt(
            question, qadi_context, max_answers, question_type, metadata
        )

        # Call LLM with timeout
        try:

            # Get the first available provider
            if not llm_manager.providers:
                raise ValueError("No LLM providers available for answer extraction")
            provider = next(iter(llm_manager.providers.keys()))

            # Validate and sanitize prompt content for security
            validated_prompt = _validate_prompt_content(prompt)
            validated_system_prompt = _validate_prompt_content(
                "You are a helpful assistant that extracts direct answers from analytical insights."
            )

            llm_request = LLMRequest(
                user_prompt=validated_prompt,
                system_prompt=validated_system_prompt,
                max_tokens=1000,
            )

            response = await asyncio.wait_for(
                llm_manager.generate(llm_request, provider),
                timeout=30.0,  # 30 second timeout for extraction
            )

            # Extract content from response
            response_content = response.content

        except asyncio.TimeoutError:
            raise Exception("LLM extraction timed out")

        # Parse LLM response
        parsed_response = self._parse_llm_response(response_content)

        # Convert to ExtractedAnswer objects
        extracted_answers = self._convert_llm_answers(
            parsed_response, qadi_results, max_answers
        )

        # Calculate total QADI ideas
        total_ideas = sum(len(ideas) for ideas in qadi_results.values())

        # Create summary
        summary = f"Generated {len(extracted_answers)} answers using LLM analysis of {total_ideas} QADI insights."

        return AnswerExtractionResult(
            original_question=question,
            question_type=question_type,
            direct_answers=extracted_answers,
            summary=summary,
            total_qadi_ideas=total_ideas,
            extraction_method="llm",
        )

    def _build_qadi_context(self, qadi_results: Dict[str, List[GeneratedIdea]]) -> str:
        """Build formatted context from QADI results for LLM."""
        context_parts = []

        for phase, ideas in qadi_results.items():
            if ideas:
                phase_title = phase.replace("_", " ").title()
                context_parts.append(f"\n## {phase_title} Phase:")
                for i, idea in enumerate(ideas, 1):
                    context_parts.append(f"{i}. {idea.content}")

        return (
            "\n".join(context_parts) if context_parts else "No QADI insights available."
        )

    def _create_llm_extraction_prompt(
        self,
        question: str,
        qadi_context: str,
        max_answers: int,
        question_type: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Create prompt for LLM answer extraction."""

        # Customize prompt based on question type
        if question_type == "list_request" or metadata.get("expects_list"):
            format_instruction = f"Provide exactly {max_answers} distinct, actionable answers in a numbered list format."
            answer_style = "concise, actionable items"
        elif question_type == "how_to":
            format_instruction = (
                f"Provide {max_answers} step-by-step approaches or methods."
            )
            answer_style = "practical, implementable steps"
        elif question_type == "what_is":
            format_instruction = f"Provide {max_answers} key insights or explanations."
            answer_style = "informative, comprehensive explanations"
        else:
            format_instruction = f"Provide {max_answers} relevant answers."
            answer_style = "direct, helpful responses"

        return f"""You are an expert at extracting direct, actionable answers from analytical insights.

USER QUESTION: "{question}"

ANALYTICAL INSIGHTS FROM QADI METHODOLOGY:
{qadi_context}

TASK: Extract {max_answers} direct, practical answers to the user's question from the above insights.

REQUIREMENTS:
- {format_instruction}
- Focus on {answer_style}
- Base answers on the provided insights when possible
- If insights are insufficient, supplement with logical, practical advice
- Each answer should be specific and actionable

RESPONSE FORMAT (JSON):
{{
    "answers": [
        {{
            "content": "Direct answer text",
            "confidence": 0.9,
            "source_phase": "questioning|abduction|deduction|induction|synthetic",
            "reasoning": "Brief explanation of how this answer derives from the insights"
        }}
    ]
}}

Provide exactly {max_answers} answers in valid JSON format:"""

    def _parse_llm_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response, handling various response formats."""

        # If response is already a dict, use it
        if isinstance(response, dict):
            return response

        # If response is a string, try to parse as JSON
        if isinstance(response, str):
            # Try to extract JSON from markdown or other formatting
            parsed = safe_json_parse(response)
            if parsed:
                return parsed

            # Fallback: try to find JSON in the text (non-greedy)
            json_match = re.search(r"\{.*?\}", response, re.DOTALL)
            if json_match:
                try:
                    from typing import cast

                    parsed_json = json.loads(json_match.group())
                    return cast(Dict[str, Any], parsed_json)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from LLM response: {e}")

        # If all parsing fails, raise an exception
        raise ValueError(f"Could not parse LLM response: {response}")

    def _convert_llm_answers(
        self,
        parsed_response: Dict[str, Any],
        qadi_results: Dict[str, List[GeneratedIdea]],
        max_answers: int,
    ) -> List[ExtractedAnswer]:
        """Convert parsed LLM response to ExtractedAnswer objects."""

        extracted_answers = []

        # Get answers from response
        answers = parsed_response.get("answers", [])

        if not answers:
            # Fallback if no answers in expected format
            raise ValueError("No answers found in LLM response")

        for i, answer in enumerate(answers[:max_answers]):
            if isinstance(answer, dict):
                content = answer.get("content", f"Answer {i+1}")
                confidence = float(answer.get("confidence", 0.7))
                source_phase = answer.get("source_phase", "synthetic")
                reasoning = answer.get("reasoning", "Generated from QADI insights")

                # Find source ideas if referenced
                source_ideas = self._find_source_ideas(content, qadi_results)

                extracted_answers.append(
                    ExtractedAnswer(
                        content=content,
                        confidence=confidence,
                        source_phase=source_phase,
                        source_ideas=source_ideas,
                        reasoning=reasoning,
                    )
                )
            elif isinstance(answer, str):
                # Simple string answer
                source_ideas = self._find_source_ideas(answer, qadi_results)
                extracted_answers.append(
                    ExtractedAnswer(
                        content=answer,
                        confidence=0.7,
                        source_phase="synthetic",
                        source_ideas=source_ideas,
                        reasoning="Generated from QADI insights",
                    )
                )

        # Ensure we have at least some answers
        if not extracted_answers:
            raise ValueError("No valid answers could be extracted from LLM response")

        return extracted_answers

    def _find_source_ideas(
        self, answer_content: str, qadi_results: Dict[str, List[GeneratedIdea]]
    ) -> List[str]:
        """Find QADI ideas that might have influenced this answer."""
        source_ideas = []
        answer_lower = answer_content.lower()

        for phase, ideas in qadi_results.items():
            for idea in ideas:
                # Simple keyword matching to find related ideas
                idea_words = set(idea.content.lower().split())
                answer_words = set(answer_lower.split())

                # If there's significant overlap, consider it a source
                overlap = len(idea_words.intersection(answer_words))
                if overlap >= 2:  # At least 2 words in common
                    source_ideas.append(idea.content)

                if len(source_ideas) >= 3:  # Limit source ideas
                    break

            if len(source_ideas) >= 3:
                break

        return source_ideas


# Global instance - removed to reduce coupling
# Create instances where needed instead of using a global
