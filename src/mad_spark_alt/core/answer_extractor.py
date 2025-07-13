"""
Answer Extractor for converting QADI insights into direct user answers.

This module bridges the gap between abstract QADI thinking and concrete
user-requested answers by analyzing question patterns and extracting
relevant solutions from QADI phases.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import GeneratedIdea

logger = logging.getLogger(__name__)


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
        # For now, use template approach
        # TODO: Add LLM-based extraction when API keys available
        return self.template_extractor.extract_answers(
            question, qadi_results, max_answers
        )


# Global instance - removed to reduce coupling
# Create instances where needed instead of using a global
