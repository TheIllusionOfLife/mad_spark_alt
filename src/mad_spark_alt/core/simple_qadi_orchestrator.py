"""
Simplified QADI Orchestrator for Hypothesis-Driven Analysis

This module implements the true QADI methodology without unnecessary complexity.
No prompt classification, no adaptive prompts - just pure hypothesis-driven consulting.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .interfaces import (
    GeneratedIdea,
    ThinkingMethod,
)
from .llm_provider import LLMRequest, llm_manager
from .qadi_prompts import PHASE_HYPERPARAMETERS, QADIPrompts, calculate_hypothesis_score
from ..utils.text_cleaning import clean_ansi_codes

logger = logging.getLogger(__name__)


# Constants for parsing LLM responses
QUESTION_PREFIX = "Q:"
HYPOTHESIS_PATTERN = r"^(?:H|Hypothesis\s*|Approach\s*)(\d+)(?:\s*:|\.)\s*(.*)$"
ANSWER_PREFIX = "ANSWER:"
ACTION_PLAN_PREFIX = "Action Plan:"
CONCLUSION_PREFIX = "Conclusion:"
MIN_HYPOTHESIS_LENGTH = 20  # Minimum length for valid hypothesis text


def get_hypothesis_generation_schema() -> Dict[str, Any]:
    """Get JSON schema for structured hypothesis generation.
        
    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "hypotheses": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "STRING"},
                        "content": {"type": "STRING"},
                    },
                    "required": ["id", "content"]
                }
            }
        },
        "required": ["hypotheses"]
    }


def get_deduction_schema() -> Dict[str, Any]:
    """Get JSON schema for structured deduction/scoring.
    
    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "evaluations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "hypothesis_id": {"type": "STRING"},
                        "scores": {
                            "type": "OBJECT",
                            "properties": {
                                "impact": {"type": "NUMBER"},
                                "feasibility": {"type": "NUMBER"},
                                "accessibility": {"type": "NUMBER"},
                                "sustainability": {"type": "NUMBER"},
                                "scalability": {"type": "NUMBER"}
                            },
                            "required": ["impact", "feasibility", "accessibility", 
                                       "sustainability", "scalability"]
                        }
                    },
                    "required": ["hypothesis_id", "scores"]
                }
            },
            "answer": {"type": "STRING"},
            "action_plan": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["evaluations", "answer", "action_plan"]
    }


def parse_structured_response(response_content: str, fallback_parser: Callable[[str], Any]) -> Any:
    """Parse structured response with fallback to text parsing.
    
    Args:
        response_content: The raw response content
        fallback_parser: Function to call if JSON parsing fails
        
    Returns:
        Parsed data from either JSON or fallback parser
    """
    try:
        return json.loads(response_content)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Structured output parsing failed: %s, falling back to text parsing", e)
        return fallback_parser(response_content)


def format_hypothesis_for_answer(hypothesis: str, approach_number: int) -> str:
    """Format hypothesis content for clean display in answer.
    
    Args:
        hypothesis: The hypothesis text to format
        approach_number: The approach number (1-based)
        
    Returns:
        Formatted hypothesis with proper line breaks and spacing
    """
    if not hypothesis:
        return ""
    
    # Clean any ANSI codes first
    hypothesis = clean_ansi_codes(hypothesis)
    
    # Fix numbered list formatting
    # Replace inline (1), (2), etc. with proper line breaks
    # Only match patterns that look like list items, not years or references
    # Look for patterns where (N) is preceded by space/punctuation and followed by actual list content
    hypothesis = re.sub(r'(?<=[\s。.!?])\s*\((\d{1,2})\)\s*(?=[^\s\d])', r'\n(\1) ', hypothesis)
    
    # Clean up multiple spaces and normalize whitespace (but preserve line breaks)
    lines = hypothesis.split('\n')
    cleaned_lines = []
    for line in lines:
        # Normalize spaces within each line
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        if cleaned_line:  # Don't add empty lines
            cleaned_lines.append(cleaned_line)
    hypothesis = '\n'.join(cleaned_lines)
    
    # Ensure proper spacing after punctuation
    # Japanese punctuation
    hypothesis = re.sub(r'([。、！？])\s*([^\s])', r'\1 \2', hypothesis)
    # English punctuation
    hypothesis = re.sub(r'([,.!?])\s*([^\s])', r'\1 \2', hypothesis)
    
    # Clean up any double line breaks that might have been created
    hypothesis = re.sub(r'\n\s*\n', '\n', hypothesis)
    
    return hypothesis


@dataclass
class HypothesisScore:
    """Scores for a single hypothesis."""

    impact: float
    feasibility: float
    accessibility: float
    sustainability: float
    scalability: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "impact": self.impact,
            "feasibility": self.feasibility,
            "accessibility": self.accessibility,
            "sustainability": self.sustainability,
            "scalability": self.scalability,
            "overall": self.overall,
        }


@dataclass
class SimpleQADIResult:
    """Result from simplified QADI analysis."""

    # Core outputs
    core_question: str
    hypotheses: List[str]
    hypothesis_scores: List[HypothesisScore]
    final_answer: str
    action_plan: List[str]
    verification_examples: List[str]
    verification_conclusion: str

    # Metadata
    total_llm_cost: float = 0.0
    phase_results: Dict[str, Any] = field(default_factory=dict)

    # For backward compatibility with evolution
    synthesized_ideas: List[GeneratedIdea] = field(default_factory=list)


class SimpleQADIOrchestrator:
    """
    Simplified QADI orchestrator implementing true hypothesis-driven methodology.

    Features:
    - Single universal prompt set
    - Phase-specific hyperparameters
    - User-adjustable creativity for hypothesis generation
    - Unified evaluation scoring
    """

    def __init__(self, temperature_override: Optional[float] = None, num_hypotheses: int = 3) -> None:
        """
        Initialize the orchestrator.

        Args:
            temperature_override: Optional temperature override for abduction phase (0.0-2.0)
            num_hypotheses: Number of hypotheses to generate in abduction phase (default: 3)
        """
        self.prompts = QADIPrompts()
        if temperature_override is not None and not 0.0 <= temperature_override <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature_override = temperature_override
        self.num_hypotheses = max(3, num_hypotheses)  # Ensure at least 3

    async def run_qadi_cycle(
        self,
        user_input: str,
        context: Optional[str] = None,
        max_retries: int = 2,
    ) -> SimpleQADIResult:
        """
        Run a complete QADI cycle on the user input.

        Args:
            user_input: The user's input (question, statement, topic, etc.)
            context: Optional additional context
            max_retries: Maximum retries per phase on failure

        Returns:
            SimpleQADIResult with all phases completed
        """
        # Combine user input with context if provided
        full_input = user_input
        if context:
            full_input = f"{user_input}\n\nContext: {context}"

        result = SimpleQADIResult(
            core_question="",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="",
        )

        try:
            # Phase 1: Question - Extract core question
            logger.info("Running Question phase")
            core_question, questioning_cost = await self._run_questioning_phase(
                full_input,
                max_retries,
            )
            result.core_question = core_question
            result.total_llm_cost += questioning_cost
            result.phase_results["questioning"] = {
                "question": core_question,
                "cost": questioning_cost,
            }

            # Phase 2: Abduction - Generate hypotheses
            logger.info("Running Abduction phase")
            hypotheses, abduction_cost = await self._run_abduction_phase(
                full_input,
                core_question,
                max_retries,
            )
            result.hypotheses = hypotheses
            result.total_llm_cost += abduction_cost
            result.phase_results["abduction"] = {
                "hypotheses": hypotheses,
                "cost": abduction_cost,
            }

            # Convert hypotheses to GeneratedIdea objects for evolution compatibility
            for i, hypothesis in enumerate(hypotheses):
                result.synthesized_ideas.append(
                    GeneratedIdea(
                        content=hypothesis,
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name="SimpleQADIOrchestrator",
                        generation_prompt=f"Hypothesis {i+1} for: {core_question}",
                        confidence_score=0.8,  # Default high confidence for hypotheses
                        reasoning="Generated as potential answer to core question",
                        metadata={"hypothesis_index": i},
                    ),
                )

            # Phase 3: Deduction - Evaluate and conclude
            logger.info("Running Deduction phase")
            deduction_result = await self._run_deduction_phase(
                full_input,
                core_question,
                hypotheses,
                max_retries,
            )
            result.hypothesis_scores = deduction_result["scores"]
            result.final_answer = deduction_result["answer"]
            result.action_plan = deduction_result["action_plan"]
            result.total_llm_cost += deduction_result["cost"]
            result.phase_results["deduction"] = deduction_result

            # Phase 4: Induction - Verify answer
            logger.info("Running Induction phase")
            induction_result = await self._run_induction_phase(
                full_input,
                core_question,
                result.final_answer,
                hypotheses,
                max_retries,
            )
            result.verification_examples = induction_result["examples"]
            result.verification_conclusion = induction_result["conclusion"]
            result.total_llm_cost += induction_result["cost"]
            result.phase_results["induction"] = induction_result

        except Exception as e:
            logger.exception("QADI cycle failed: %s", e)
            # Provide user-friendly error message
            if "API" in str(e) or "api" in str(e):
                raise RuntimeError(
                    "QADI cycle failed due to API issues. Please check:\n"
                    "1. Your API keys are correctly set\n"
                    "2. You have sufficient API credits\n"
                    "3. The API service is accessible\n"
                    f"Original error: {e}"
                )
            elif "timeout" in str(e).lower():
                raise RuntimeError(
                    "QADI cycle timed out. This might be due to:\n"
                    "1. Slow API response times\n"
                    "2. Complex input requiring more processing\n"
                    "Try again with simpler input or check your connection.\n"
                    f"Original error: {e}"
                )
            else:
                raise RuntimeError(
                    f"QADI cycle failed unexpectedly. Error: {e}\n"
                    "Please check the logs for more details."
                )

        return result

    async def _run_questioning_phase(
        self,
        user_input: str,
        max_retries: int,
    ) -> Tuple[str, float]:
        """Extract the core question from user input.

        Returns:
            Tuple of (core_question, total_cost)
        """
        prompt = self.prompts.get_questioning_prompt(user_input)
        hyperparams = PHASE_HYPERPARAMETERS["questioning"]
        total_cost = 0.0

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                )

                response = await llm_manager.generate(request)
                total_cost += response.cost

                # Extract the core question
                content = clean_ansi_codes(response.content.strip())
                match = re.search(rf"{QUESTION_PREFIX}\s*(.+)", content)
                if match:
                    return match.group(1).strip(), total_cost
                # Fallback: use the whole response if no Q: prefix
                # Safe removal of prefix only from start of string
                if content.startswith(QUESTION_PREFIX):
                    return content[len(QUESTION_PREFIX) :].strip(), total_cost
                return content.strip(), total_cost

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to extract core question after %d attempts. "
                        "Last error: %s. Please check your LLM API configuration.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to extract core question after {max_retries + 1} attempts. "
                        f"Last error: {e}. Please check your LLM API configuration and try again."
                    )
                logger.warning("Question phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to extract core question")

    async def _run_abduction_phase(
        self,
        user_input: str,
        core_question: str,
        max_retries: int,
    ) -> Tuple[List[str], float]:
        """Generate hypotheses to answer the core question."""
        prompt = self.prompts.get_abduction_prompt(user_input, core_question, self.num_hypotheses)
        hyperparams = PHASE_HYPERPARAMETERS["abduction"].copy()

        # Apply temperature override if provided
        if self.temperature_override is not None:
            hyperparams["temperature"] = self.temperature_override

        total_cost = 0.0

        for attempt in range(max_retries + 1):
            try:
                # Try structured output first
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.95),
                    response_schema=get_hypothesis_generation_schema(),
                    response_mime_type="application/json"
                )

                response = await llm_manager.generate(request)
                total_cost += response.cost

                # Try to parse as JSON first (structured output)
                try:
                    data = json.loads(response.content)
                    if "hypotheses" in data and isinstance(data["hypotheses"], list):
                        hypotheses = []
                        for h in data["hypotheses"]:
                            if isinstance(h, dict) and "content" in h:
                                content = h["content"].strip()
                                if len(content) >= MIN_HYPOTHESIS_LENGTH:
                                    hypotheses.append(content)
                        
                        if len(hypotheses) >= self.num_hypotheses:
                            logger.debug("Successfully extracted %d hypotheses using structured output", len(hypotheses))
                            return hypotheses[:self.num_hypotheses], total_cost
                        else:
                            logger.warning("Structured output returned insufficient hypotheses: %d", len(hypotheses))
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug("JSON parsing failed, falling back to text parsing: %s", e)

                # Fall back to text parsing if structured output fails
                content = response.content.strip()
                
                # Log the actual response for debugging
                logger.debug("LLM response for abduction phase:\n%s", content)
                
                # Clean all ANSI codes using comprehensive cleaning function
                content = clean_ansi_codes(content)
                
                # Initialize hypotheses list for fallback parsing
                hypotheses = []
                
                lines = content.split("\n")

                current_hypothesis = ""
                current_index = None
                current_title = ""

                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    # Check if line starts with H1:, H2:, H3:, Approach N:, etc.
                    # Pattern matches various hypothesis formats after ANSI codes have been stripped
                    hypothesis_match = re.match(r"^(?:\*\*)?(?:H|Hypothesis\s*|Approach\s*)(\d+)(?:\*\*)?(?:\s*:|\.)\s*(.*)$", line)
                    
                    if hypothesis_match:
                        # Save previous hypothesis if we have one
                        if current_index is not None and current_hypothesis.strip():
                            if len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                                hypotheses.append(current_hypothesis.strip())

                        # Start new hypothesis
                        current_index = int(hypothesis_match.group(1))
                        title_and_content = hypothesis_match.group(2).strip()
                        
                        # Remove any remaining bold formatting
                        title_and_content = re.sub(r'\*\*', '', title_and_content)
                        
                        # Check if this line contains just a title in brackets
                        title_match = re.match(r'^\[([^\]]+)\]$', title_and_content)
                        if title_match:
                            # This is just a title, content will follow
                            current_title = title_match.group(1)
                            current_hypothesis = ""
                        else:
                            # This line contains content
                            # Remove brackets from the content if they exist
                            title_and_content = re.sub(r'\[([^\]]+)\]', r'\1', title_and_content)
                            current_hypothesis = title_and_content
                    elif current_index is not None:
                        # Continue building current hypothesis from subsequent lines
                        # Skip empty lines and lines that are just markdown
                        if line and not line.startswith("---") and not re.match(r"^\*+$", line):
                            if current_hypothesis:
                                current_hypothesis += " " + line
                            else:
                                # First content line after title
                                current_hypothesis = line

                # Don't forget the last hypothesis
                if current_index is not None and current_hypothesis.strip():
                    if len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                        hypotheses.append(current_hypothesis.strip())

                # Log what was extracted
                logger.debug("Extracted %d hypotheses: %s", len(hypotheses), hypotheses)

                if len(hypotheses) >= self.num_hypotheses:
                    # Return exactly the requested number of hypotheses
                    return hypotheses[:self.num_hypotheses], total_cost
                
                # Fallback: Try alternative parsing methods
                logger.warning(
                    "Failed to extract enough hypotheses with standard pattern. "
                    "Got %d hypotheses. Trying fallback parsing...", 
                    len(hypotheses)
                )
                
                # Enhanced fallback parsing with multiple format support
                hypotheses = []
                current_hypothesis = ""
                hypothesis_buffer: List[str] = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Match various hypothesis start patterns
                    hypothesis_patterns = [
                        r"^(\d+)[.)]\s*(.+)$",  # "1. Text" or "1) Text"
                        r"^(?:\*\*)?H(\d+)(?:\*\*)?[:.]\s*(.+)$",  # "H1: Text" or "**H1:** Text"
                        r"^(?:\*\*)?Hypothesis\s+(\d+)(?:\*\*)?[:.]\s*(.+)$",  # "Hypothesis 1: Text"
                        r"^(?:\*\*)?Approach\s+(\d+)(?:\*\*)?[:.]\s*(.+)$",    # "Approach 1: Text"
                        r"^[•\-]\s+(.+)$",    # "• Text" or "- Text" (not * to avoid matching bold)
                        r"^\*\s+(.+)$",        # "* Text" (single asterisk with space)
                    ]
                    
                    matched = False
                    for pattern in hypothesis_patterns:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            # Save previous hypothesis if we have one
                            if current_hypothesis.strip() and len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                                hypotheses.append(current_hypothesis.strip())
                            
                            # Start new hypothesis
                            if len(match.groups()) == 2:
                                # Pattern with number (like "1. Text")
                                hypothesis_num = int(match.group(1)) if match.group(1).isdigit() else len(hypotheses) + 1
                                if hypothesis_num <= self.num_hypotheses:
                                    current_hypothesis = match.group(2).strip()
                                    matched = True
                            else:
                                # Pattern without number (like "- Text")
                                if len(hypotheses) < self.num_hypotheses:
                                    current_hypothesis = match.group(1).strip()
                                    matched = True
                            break
                    
                    if not matched and current_hypothesis:
                        # Continue building current hypothesis (multi-line content)
                        # But stop if we hit lines that look like explanatory text
                        if not line.startswith(("These", "This", "The above", "Note:")):
                            current_hypothesis += " " + line
                
                # Don't forget the last hypothesis
                if current_hypothesis.strip() and len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                    hypotheses.append(current_hypothesis.strip())
                
                # Additional fallback: try to extract content between common delimiters
                if len(hypotheses) < self.num_hypotheses:
                    # Look for sections separated by double newlines
                    sections = re.split(r'\n\s*\n', content)
                    for section in sections:
                        section = section.strip()
                        if len(section) > 30 and len(hypotheses) < self.num_hypotheses:
                            # Skip sections that look like headers or metadata
                            if section.startswith("**Scale:**") or section.startswith("Scale:"):
                                continue
                            # Clean up section markers but preserve content
                            cleaned = re.sub(r'^(?:\d+[.)]\s*|[•\-]\s+|\*\s+|H\d+[:.]\s*)', '', section, flags=re.IGNORECASE)
                            if len(cleaned.strip()) > MIN_HYPOTHESIS_LENGTH and not any(h == cleaned.strip() for h in hypotheses):
                                hypotheses.append(cleaned.strip())
                
                if len(hypotheses) >= self.num_hypotheses:
                    logger.info("Fallback parsing extracted %d hypotheses, returning %d as requested", len(hypotheses), self.num_hypotheses)
                    return hypotheses[:self.num_hypotheses], total_cost  # Return requested number
                    
                logger.warning(
                    "Failed to extract enough hypotheses even with fallback. "
                    "Response preview:\n%s", content[:500]
                )

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to generate hypotheses after %d attempts. "
                        "Last error: %s. The LLM may not be responding correctly.",
                        max_retries + 1,
                        e,
                    )
                    # For tests with max_retries=0, return empty list instead of raising
                    if max_retries == 0:
                        return [], total_cost
                    raise RuntimeError(
                        f"Failed to generate hypotheses after {max_retries + 1} attempts. "
                        f"Last error: {e}. Please ensure your LLM API is working and try again."
                    )
                logger.warning("Abduction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        # For tests with max_retries=0, return empty list instead of raising
        if max_retries == 0:
            return [], total_cost
        raise RuntimeError("Failed to generate hypotheses")

    async def _run_deduction_phase(
        self,
        user_input: str,
        core_question: str,
        hypotheses: List[str],
        max_retries: int,
    ) -> Dict[str, Any]:
        """Evaluate hypotheses and determine the answer."""
        # For large numbers of hypotheses, evaluate in parallel batches
        if len(hypotheses) > 5:
            return await self._run_parallel_deduction(
                user_input, core_question, hypotheses, max_retries
            )
        
        # Original implementation for small hypothesis sets
        # Format hypotheses for the prompt (using new format without H prefix)
        hypotheses_text = "\n".join([f"{i+1}. {h}" for i, h in enumerate(hypotheses)])

        prompt = self.prompts.get_deduction_prompt(
            user_input,
            core_question,
            hypotheses_text,
        )
        hyperparams = PHASE_HYPERPARAMETERS["deduction"]

        for attempt in range(max_retries + 1):
            try:
                # Create request with structured output schema
                schema = get_deduction_schema()
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                    response_schema=schema,
                    response_mime_type="application/json",
                )

                response = await llm_manager.generate(request)
                content = clean_ansi_codes(response.content.strip())

                # Try to parse as JSON first (structured output)
                try:
                    data = json.loads(content)
                    
                    # Extract scores from structured response
                    scores = []
                    for eval_data in data.get("evaluations", []):
                        score_data = eval_data.get("scores", {})
                        # Calculate overall score using the consistent weighted method
                        scores_dict = {
                            "impact": score_data.get("impact", 0.5),
                            "feasibility": score_data.get("feasibility", 0.5),
                            "accessibility": score_data.get("accessibility", 0.5),
                            "sustainability": score_data.get("sustainability", 0.5),
                            "scalability": score_data.get("scalability", 0.5),
                        }
                        overall = calculate_hypothesis_score(scores_dict)
                        
                        score = HypothesisScore(
                            impact=scores_dict["impact"],
                            feasibility=scores_dict["feasibility"],
                            accessibility=scores_dict["accessibility"],
                            sustainability=scores_dict["sustainability"],
                            scalability=scores_dict["scalability"],
                            overall=overall,
                        )
                        scores.append(score)
                    
                    # Ensure we have scores for all hypotheses
                    while len(scores) < len(hypotheses):
                        scores.append(HypothesisScore(
                            impact=0.5,
                            feasibility=0.5,
                            accessibility=0.5,
                            sustainability=0.5,
                            scalability=0.5,
                            overall=0.5,
                        ))
                    
                    answer = data.get("answer", "")
                    action_plan = data.get("action_plan", [])
                    
                    return {
                        "scores": scores,
                        "answer": answer,
                        "action_plan": action_plan,
                        "cost": response.cost,
                        "raw_content": content,
                    }
                    
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Fall back to text parsing
                    logger.debug("Structured output parsing failed, falling back to text parsing: %s", e)
                
                # Parse the evaluation scores using text parsing
                scores = []
                for i in range(len(hypotheses)):
                    score = self._parse_hypothesis_scores(content, i + 1)
                    scores.append(score)

                # Extract the answer - try multiple patterns
                answer = ""

                # Try exact ANSWER: format first
                answer_match = re.search(
                    rf"{ANSWER_PREFIX}\s*(.+?)(?={ACTION_PLAN_PREFIX}|$)",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if answer_match:
                    answer = answer_match.group(1).strip()
                else:
                    # Try alternative patterns
                    # Look for "Answer" section with different formatting
                    alt_patterns = [
                        r"(?:^|\n)\s*(?:\*\*)?Answer(?:\*\*)?:?\s*(.+?)(?=Action Plan|$)",
                        r"(?:^|\n)\s*#+\s*Answer:?\s*(.+?)(?=Action Plan|$)",
                        r"Based on.+?evaluation.+?(\*\*.+?\*\*.*?)(?=Action Plan|$)",
                    ]
                    for pattern in alt_patterns:
                        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                        if match:
                            answer = match.group(1).strip()
                            break

                    # Last resort: if still no answer, extract text between scores and action plan
                    if not answer:
                        # Find the end of H3/Approach 3 scores and start of action plan
                        h3_end = re.search(r"(?:H3:|Approach 3:).*?Overall:.*?\n\n", content, re.DOTALL | re.IGNORECASE)
                        action_start = re.search(
                            r"Action Plan:", content, re.IGNORECASE
                        )
                        if h3_end and action_start:
                            potential_answer = content[
                                h3_end.end() : action_start.start()
                            ].strip()
                            if (
                                potential_answer and len(potential_answer) > 50
                            ):  # Reasonable answer length
                                answer = potential_answer

                # Extract action plan - be more flexible
                action_plan = []
                plan_match = re.search(
                    r"Action Plan:?\s*(.+?)$",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if plan_match:
                    plan_text = plan_match.group(1).strip()
                    # Extract numbered items or bullet points
                    # Updated regex to handle multi-line items better
                    plan_items = re.findall(
                        r"(?:^|\n)\s*(?:\d+\.|[-*•])\s*(.+?)(?=(?:\n\s*(?:\d+\.|[-*•]))|$)",
                        plan_text,
                        re.DOTALL | re.MULTILINE,
                    )
                    action_plan = [
                        item.strip() 
                        for item in plan_items 
                        if item.strip() and len(item.strip()) > 1  # Filter out single character items like '*'
                    ]

                    # If no items found with bullets/numbers, try splitting by newlines
                    if not action_plan:
                        lines = plan_text.split("\n")
                        action_plan = [
                            line.strip()
                            for line in lines
                            if line.strip() and not line.strip().startswith("#")
                        ]

                return {
                    "scores": scores,
                    "answer": answer,
                    "action_plan": action_plan,
                    "cost": response.cost,
                    "raw_content": content,
                }

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to evaluate hypotheses after %d attempts. "
                        "Last error: %s. The evaluation process encountered issues.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to evaluate hypotheses after {max_retries + 1} attempts. "
                        f"Last error: {e}. The LLM may be having trouble with the evaluation format."
                    )
                logger.warning("Deduction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to evaluate hypotheses")

    def _parse_hypothesis_scores(
        self, content: str, hypothesis_num: int
    ) -> HypothesisScore:
        """Parse scores for a specific hypothesis from deduction content."""
        # Use line-by-line parsing to extract hypothesis section
        lines = content.split("\n")
        section_lines = []
        in_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is the start of our hypothesis section
            # Handle various formats: "H1:", "- H1:", "- **H1:**", "Approach 1:", etc.
            hypothesis_match = re.match(
                rf"^(?:-\s*)?(?:\*\*)?H{hypothesis_num}[:.](.*?)(?:\*\*)?$", line
            )
            # Also check for "Hypothesis 1:" format
            if not hypothesis_match:
                hypothesis_match = re.match(
                    rf"^(?:-\s*)?(?:\*\*)?Hypothesis\s+{hypothesis_num}[:.](.*?)(?:\*\*)?$",
                    line,
                    re.IGNORECASE,
                )
            # Also check for "Approach 1:" format
            if not hypothesis_match:
                hypothesis_match = re.match(
                    rf"^(?:-\s*)?(?:\*\*)?Approach\s+{hypothesis_num}[:.](.*?)(?:\*\*)?$",
                    line,
                    re.IGNORECASE,
                )
            if hypothesis_match:
                in_section = True
                section_lines.append(hypothesis_match.group(1).strip())
                continue

            # Check if we've reached the next hypothesis or end section
            if in_section:
                if (re.match(rf"^(?:-\s*)?(?:\*\*)?H{hypothesis_num + 1}[:.]", line) or 
                    re.match(rf"^(?:-\s*)?(?:\*\*)?Approach\s+{hypothesis_num + 1}[:.]", line, re.IGNORECASE) or
                    re.match(rf"^(?:-\s*)?(?:\*\*)?Hypothesis\s+{hypothesis_num + 1}[:.]", line, re.IGNORECASE) or
                    line.startswith((ANSWER_PREFIX, ACTION_PLAN_PREFIX))):
                    break
                section_lines.append(line)

        if not section_lines:
            # Log warning and return default scores if parsing fails
            logger.warning(
                "Failed to parse scores for hypothesis %d. Using default scores. "
                "This may indicate the LLM response didn't follow the expected format. "
                "Content preview: %s...",
                hypothesis_num,
                content[:300] if len(content) > 300 else content,
            )
            return HypothesisScore(
                impact=0.5,
                feasibility=0.5,
                accessibility=0.5,
                sustainability=0.5,
                scalability=0.5,
                overall=0.5,
            )

        section = " ".join(section_lines)

        # Extract individual scores with improved robustness
        def extract_score(criterion: str, text: str) -> float:
            # Try multiple patterns to handle different formatting
            # First check for fractional scores (e.g., "8/10")
            fraction_pattern = rf"{criterion}:\s*(-?[0-9.]+)/(\d+)"
            fraction_match = re.search(fraction_pattern, text, re.IGNORECASE)
            if fraction_match:
                try:
                    numerator = float(fraction_match.group(1))
                    denominator = float(fraction_match.group(2))
                    if denominator > 0:
                        score = numerator / denominator
                        return max(0.0, min(1.0, score))
                except (ValueError, ZeroDivisionError):
                    pass

            # Other patterns for direct scores - enhanced for markdown and various formats
            patterns = [
                rf"\*\*{criterion}:\*\*\s*(-?[0-9.]+)\s*-",          # "**Impact:** 0.8 - explanation"
                rf"\*\*{criterion}:\*\*\s*(-?[0-9.]+)",              # "**Impact:** 0.8"
                rf"\*?\s*{criterion}:\s*(-?[0-9.]+)\s*-",            # "* Impact: 0.8 - explanation" or "Impact: 0.8 - explanation"
                rf"\*?\s*{criterion}:\s*(-?[0-9.]+)",                # "* Impact: 0.8" or "Impact: 0.8"
                rf"{criterion}\s*-\s*(-?[0-9.]+)",                   # "Impact - 0.8"
                rf"{criterion}\s*:\s*(-?[0-9.]+)/?",                 # "Impact: 0.8/" or "Impact: 0.8"
                rf"{criterion}\s*\((-?[0-9.]+)\)",                   # "Impact (0.8)"
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        # Ensure score is between 0 and 1
                        return max(0.0, min(1.0, score))
                    except (ValueError, TypeError):
                        continue

            # If no pattern matches, return default
            return 0.5

        scores = {
            "impact": extract_score("Impact", section),
            "feasibility": extract_score("Feasibility", section),
            "accessibility": extract_score("Accessibility", section),
            "sustainability": extract_score("Sustainability", section),
            "scalability": extract_score("Scalability", section),
        }

        # Calculate overall score
        overall = calculate_hypothesis_score(scores)

        return HypothesisScore(
            impact=scores["impact"],
            feasibility=scores["feasibility"],
            accessibility=scores["accessibility"],
            sustainability=scores["sustainability"],
            scalability=scores["scalability"],
            overall=overall,
        )

    async def _run_induction_phase(
        self,
        user_input: str,
        core_question: str,
        answer: str,
        hypotheses: List[str],
        max_retries: int,
    ) -> Dict[str, Any]:
        """Verify the answer with examples."""
        prompt = self.prompts.get_induction_prompt(user_input, core_question, answer)
        hyperparams = PHASE_HYPERPARAMETERS["induction"]

        for attempt in range(max_retries + 1):
            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    temperature=hyperparams["temperature"],
                    max_tokens=int(hyperparams["max_tokens"]),
                    top_p=hyperparams.get("top_p", 0.9),
                )

                response = await llm_manager.generate(request)
                content = clean_ansi_codes(response.content.strip())

                # Extract verification examples using pattern matching
                examples = []

                # Look for "Example N:" pattern
                example_pattern = (
                    r"Example\s*(\d+):\s*(.+?)(?=Example\s*\d+:|Conclusion:|$)"
                )
                example_matches = re.findall(
                    example_pattern, content, re.DOTALL | re.IGNORECASE
                )

                for _, example_content in example_matches:
                    # Clean up the example content
                    example_text = example_content.strip()
                    # Replace bullet points with proper formatting
                    example_text = re.sub(
                        r"^[-•]\s*", "", example_text, flags=re.MULTILINE
                    )
                    examples.append(example_text)

                # If no examples found with "Example N:" pattern, try numbered list
                if not examples:
                    lines = content.split("\n")
                    current_example = ""
                    current_index = None

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Check if line starts with "1.", "2.", or "3."
                        example_match = re.match(r"^([123])\.\s*(.*)$", line)
                        if example_match:
                            # Save previous example if we have one
                            if current_index is not None and current_example.strip():
                                examples.append(current_example.strip())

                            # Start new example
                            current_index = int(example_match.group(1))
                            current_example = example_match.group(2)
                        elif line.startswith(CONCLUSION_PREFIX):
                            # Save last example before conclusion
                            if current_index is not None and current_example.strip():
                                examples.append(current_example.strip())
                            break
                        elif current_index is not None:
                            # Continue building current example
                            current_example += " " + line

                    # Don't forget the last example if no conclusion found
                    if current_index is not None and current_example.strip():
                        examples.append(current_example.strip())

                # Extract conclusion
                conclusion_match = re.search(
                    rf"{CONCLUSION_PREFIX}\s*(.+?)$",
                    content,
                    re.DOTALL,
                )
                conclusion = (
                    conclusion_match.group(1).strip() if conclusion_match else ""
                )

                # Fix self-reference issues in conclusion
                if conclusion and answer:
                    # Replace references like "(H1)", "(H2)", "(H3)" with the actual hypothesis content
                    for i, hypothesis in enumerate(hypotheses):
                        # Handle various reference patterns
                        patterns = [
                            rf"\(H{i+1}\)",  # (H1), (H2), (H3)
                            rf"H{i+1}",  # H1, H2, H3
                            rf"hypothesis {i+1}",  # hypothesis 1, hypothesis 2, hypothesis 3
                            rf"approach {i+1}",  # approach 1, approach 2, approach 3
                        ]

                        # Create a brief description of the hypothesis (first 50 chars)
                        brief_hypothesis = (
                            hypothesis[:50] + "..."
                            if len(hypothesis) > 50
                            else hypothesis
                        )
                        replacement = f'"{brief_hypothesis}"'

                        for pattern in patterns:
                            # Only replace if it's referring to the chosen answer
                            if re.search(
                                rf"your answer.*{pattern}", conclusion, re.IGNORECASE
                            ):
                                conclusion = re.sub(
                                    rf"your answer\s*\({pattern}\)",
                                    f"the recommended approach {replacement}",
                                    conclusion,
                                    flags=re.IGNORECASE,
                                )
                            elif re.search(
                                rf"the answer.*{pattern}", conclusion, re.IGNORECASE
                            ):
                                conclusion = re.sub(
                                    rf"the answer\s*\({pattern}\)",
                                    f"the recommended approach {replacement}",
                                    conclusion,
                                    flags=re.IGNORECASE,
                                )

                    # Fix any remaining "your answer" references
                    conclusion = re.sub(
                        r"your answer",
                        "the recommended approach",
                        conclusion,
                        flags=re.IGNORECASE,
                    )

                return {
                    "examples": examples,
                    "conclusion": conclusion,
                    "cost": response.cost,
                    "raw_content": content,
                }

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed to verify answer after %d attempts. "
                        "Last error: %s. The verification process could not complete.",
                        max_retries + 1,
                        e,
                    )
                    raise RuntimeError(
                        f"Failed to verify answer after {max_retries + 1} attempts. "
                        f"Last error: {e}. The system will proceed with unverified results."
                    )
                logger.warning("Induction phase attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(1)

        raise RuntimeError("Failed to verify answer")
    
    async def _run_parallel_deduction(
        self,
        user_input: str,
        core_question: str,
        hypotheses: List[str],
        max_retries: int,
    ) -> Dict[str, Any]:
        """
        Evaluate large numbers of hypotheses in parallel batches.
        
        This method splits hypotheses into smaller groups and evaluates them
        concurrently to improve performance for large hypothesis sets.
        """
        batch_size = 3  # Evaluate 3 hypotheses per LLM call
        hyperparams = PHASE_HYPERPARAMETERS["deduction"]
        
        # Split hypotheses into batches
        batches = []
        for i in range(0, len(hypotheses), batch_size):
            batch = hypotheses[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(hypotheses))))
            batches.append((batch, batch_indices))
        
        # Evaluate batches in parallel
        async def evaluate_batch(batch_hypotheses: List[str], indices: List[int]) -> List[Tuple[int, HypothesisScore, float]]:
            """Evaluate a single batch of hypotheses."""
            # Format batch for prompt (using new format without H prefix)
            batch_text = "\n".join([f"{indices[j]+1}. {h}" for j, h in enumerate(batch_hypotheses)])
            
            # Create batch-specific prompt
            batch_prompt = self.prompts.get_deduction_prompt(
                user_input,
                core_question,
                batch_text,
            )
            
            for attempt in range(max_retries + 1):
                try:
                    # Create request with structured output schema
                    schema = get_deduction_schema()
                    request = LLMRequest(
                        user_prompt=batch_prompt,
                        temperature=hyperparams["temperature"],
                        max_tokens=int(hyperparams["max_tokens"]),
                        top_p=hyperparams.get("top_p", 0.9),
                        response_schema=schema,
                        response_mime_type="application/json",
                    )
                    
                    response = await llm_manager.generate(request)
                    content = clean_ansi_codes(response.content.strip())
                    
                    # Try to parse as JSON first
                    batch_scores = []
                    try:
                        data = json.loads(content)
                        
                        # Extract scores from structured response
                        evaluations = data.get("evaluations", [])
                        for eval_data, idx in zip(evaluations, indices):
                            score_data = eval_data.get("scores", {})
                            
                            # Use consistent weighted score calculation
                            scores_dict = {
                                "impact": score_data.get("impact", 0.5),
                                "feasibility": score_data.get("feasibility", 0.5),
                                "accessibility": score_data.get("accessibility", 0.5),
                                "sustainability": score_data.get("sustainability", 0.5),
                                "scalability": score_data.get("scalability", 0.5),
                            }
                            overall = calculate_hypothesis_score(scores_dict)
                            
                            score = HypothesisScore(
                                impact=scores_dict["impact"],
                                feasibility=scores_dict["feasibility"],
                                accessibility=scores_dict["accessibility"],
                                sustainability=scores_dict["sustainability"],
                                scalability=scores_dict["scalability"],
                                overall=overall,
                            )
                            batch_scores.append((idx, score, response.cost / len(indices)))
                        
                        # If we didn't get all scores, fill in defaults
                        while len(batch_scores) < len(indices):
                            idx = indices[len(batch_scores)]
                            batch_scores.append((idx, HypothesisScore(
                                impact=0.5,
                                feasibility=0.5,
                                accessibility=0.5,
                                sustainability=0.5,
                                scalability=0.5,
                                overall=0.5,
                            ), response.cost / len(indices)))
                            
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        # Fall back to text parsing
                        logger.debug("Structured output parsing failed for batch, falling back to text parsing: %s", e)
                        for _, idx in enumerate(indices):
                            score = self._parse_hypothesis_scores(content, idx + 1)
                            batch_scores.append((idx, score, response.cost / len(indices)))
                    
                    return batch_scores
                    
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed to evaluate batch after {max_retries + 1} attempts: {e}")
                        logger.warning(f"Returning default scores for {len(indices)} hypotheses due to evaluation failure")
                        # Return default scores for this batch
                        return [(idx, HypothesisScore(
                            impact=0.5,
                            feasibility=0.5,
                            accessibility=0.5,
                            sustainability=0.5,
                            scalability=0.5,
                            overall=0.5
                        ), 0.0) for idx in indices]
                    await asyncio.sleep(1)
            
            return []  # Should never reach here
        
        # Run all batches in parallel
        batch_tasks = [evaluate_batch(batch, indices) for batch, indices in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Combine results
        all_scores = []
        total_cost = 0.0
        
        # Flatten and sort results by index
        for batch_result in batch_results:
            for idx, score, cost in batch_result:
                all_scores.append((idx, score))
                total_cost += cost
        
        # Sort by index to maintain order
        all_scores.sort(key=lambda x: x[0])
        scores = [score for _, score in all_scores]
        
        # Now we need to determine the best answer from all hypotheses
        # Find the hypothesis with the highest overall score
        best_idx = max(range(len(scores)), key=lambda i: scores[i].overall)
        best_hypothesis = hypotheses[best_idx]
        best_score = scores[best_idx]
        
        # Format the hypothesis for clean display
        formatted_hypothesis = format_hypothesis_for_answer(best_hypothesis, best_idx + 1)
        
        # Include approach number in the answer
        answer = f"Based on the evaluation, the most effective approach is **Approach {best_idx + 1}**:\n\n"
        answer += f"{formatted_hypothesis}\n\n"
        answer += f"This approach scores highest with an overall score of {best_score.overall:.2f}, "
        answer += f"offering strong impact ({best_score.impact:.2f}) and feasibility ({best_score.feasibility:.2f})."
        
        # Generate action plan based on best hypothesis
        action_plan = [
            f"Implement the core strategy from Approach {best_idx + 1}",
            "Start with pilot testing in a controlled environment",
            "Measure impact using defined metrics",
            "Scale gradually based on results",
            "Iterate and refine based on feedback"
        ]
        
        return {
            "scores": scores,
            "answer": answer,
            "action_plan": action_plan,
            "cost": total_cost,
        }
