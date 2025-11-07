"""
Centralized LLM Response Parsing Utilities

This module provides unified parsing for all LLM response formats used in Mad Spark Alt.
All parsers follow a consistent pattern: try structured output first, fall back to text parsing.

Key Features:
- HypothesisParser: Extract hypotheses from various formats (JSON, text patterns)
- ScoreParser: Parse QADI evaluation scores (5 criteria)
- ActionPlanParser: Extract action plans and next steps

Design Pattern:
Each parser class provides three methods:
1. parse_structured_*() - Parse JSON structured output
2. parse_text_*() - Parse unstructured text with regex
3. parse_with_fallback() - Recommended entry point (tries both strategies)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import logging
import json

from .json_utils import extract_and_parse_json
from ..utils.text_cleaning import clean_ansi_codes

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Hypothesis patterns for text parsing
HYPOTHESIS_PATTERNS = [
    r"^(?:\*\*)?(?:H|Hypothesis\s*|Approach\s*)(\d+)(?:\*\*)?(?:\s*:|\.)\s*(.*)$",
    r"^(\d+)[.)]\s*(.+)$",  # Numbered lists
]

# Minimum valid lengths (from simple_qadi_orchestrator.py)
MIN_HYPOTHESIS_LENGTH = 20  # Minimum characters for a valid hypothesis
MIN_ACTION_ITEM_LENGTH = 10  # Minimum characters for a valid action item


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ParsedScores:
    """Container for parsed QADI score data."""
    impact: float
    feasibility: float
    accessibility: float
    sustainability: float
    scalability: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "impact": self.impact,
            "feasibility": self.feasibility,
            "accessibility": self.accessibility,
            "sustainability": self.sustainability,
            "scalability": self.scalability,
        }


# ============================================================================
# 1. HypothesisParser - Parse hypotheses from LLM responses
# ============================================================================

class HypothesisParser:
    """Parse hypotheses/ideas from LLM responses with multiple fallback strategies."""

    @staticmethod
    def parse_structured_json(
        response_content: str,
        num_expected: int
    ) -> List[str]:
        """
        Parse structured JSON response.

        Expected format:
        {
            "hypotheses": [
                {"id": "1", "content": "..."},
                {"id": "2", "content": "..."}
            ]
        }

        Args:
            response_content: LLM response text (may contain JSON)
            num_expected: Number of hypotheses expected

        Returns:
            List of hypothesis content strings (empty if parsing fails)
        """
        try:
            # First try direct JSON parsing
            data = json.loads(response_content)

            if "hypotheses" in data and isinstance(data["hypotheses"], list):
                hypotheses = []
                for h in data["hypotheses"]:
                    if isinstance(h, dict) and "content" in h:
                        content = h["content"].strip()
                        if len(content) >= MIN_HYPOTHESIS_LENGTH:
                            hypotheses.append(content)

                if len(hypotheses) >= num_expected:
                    logger.debug("Successfully extracted %d hypotheses using structured output", len(hypotheses))
                    return hypotheses[:num_expected]
                else:
                    logger.debug("Structured output returned insufficient hypotheses: %d", len(hypotheses))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Direct JSON parsing failed: %s. Trying extract_and_parse_json...", e)

            # Try using json_utils for more robust extraction
            try:
                data = extract_and_parse_json(
                    response_content,
                    expected_keys=["hypotheses"],
                    fix_issues=True
                )

                if "hypotheses" in data and isinstance(data["hypotheses"], list):
                    hypotheses = []
                    for h in data["hypotheses"]:
                        if isinstance(h, dict) and "content" in h:
                            content = h["content"].strip()
                            if len(content) >= MIN_HYPOTHESIS_LENGTH:
                                hypotheses.append(content)

                    if len(hypotheses) >= num_expected:
                        logger.debug("Successfully extracted %d hypotheses using json_utils", len(hypotheses))
                        return hypotheses[:num_expected]

            except Exception as e2:
                logger.debug("extract_and_parse_json also failed: %s", e2)

        return []

    @staticmethod
    def parse_text_with_patterns(
        response_content: str,
        num_expected: int
    ) -> List[str]:
        """
        Parse unstructured text using regex patterns.

        Handles formats:
        - H1: Hypothesis text
        - Approach 1: Hypothesis text
        - 1. Hypothesis text
        - **H1:** Hypothesis text (markdown)
        - - Bullet hypothesis
        - * Bullet hypothesis

        Args:
            response_content: LLM response text
            num_expected: Number of hypotheses expected

        Returns:
            List of hypothesis content strings
        """
        # Clean ANSI codes first
        content = clean_ansi_codes(response_content.strip())

        hypotheses = []
        lines = content.split("\n")
        current_hypothesis = ""
        current_index = None

        # First pass: Try standard patterns (H1:, Approach 1:, etc.)
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for hypothesis start patterns
            hypothesis_match = re.match(
                r"^(?:\*\*)?(?:H|Hypothesis\s*|Approach\s*)(\d+)(?:\*\*)?(?:\s*:|\.)\s*(.*)$",
                line
            )

            if hypothesis_match:
                # Save previous hypothesis
                if current_index is not None and current_hypothesis.strip():
                    if len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                        hypotheses.append(current_hypothesis.strip())

                # Start new hypothesis
                current_index = int(hypothesis_match.group(1))
                title_and_content = hypothesis_match.group(2).strip()

                # Remove markdown formatting
                title_and_content = re.sub(r'\*\*', '', title_and_content)

                # Handle title in brackets separately
                title_match = re.match(r'^\[([^\]]+)\]$', title_and_content)
                if title_match:
                    # This is just a title, content will follow
                    current_hypothesis = ""
                else:
                    # This line contains content
                    # Remove brackets from the content if they exist
                    title_and_content = re.sub(r'\[([^\]]+)\]', r'\1', title_and_content)
                    current_hypothesis = title_and_content

            elif current_index is not None:
                # Continue building current hypothesis
                if line and not line.startswith("---") and not re.match(r"^\*+$", line):
                    if current_hypothesis:
                        current_hypothesis += " " + line
                    else:
                        current_hypothesis = line

        # Don't forget the last hypothesis
        if current_index is not None and current_hypothesis.strip():
            if len(current_hypothesis.strip()) > MIN_HYPOTHESIS_LENGTH:
                hypotheses.append(current_hypothesis.strip())

        if len(hypotheses) >= num_expected:
            logger.debug("Extracted %d hypotheses with standard patterns", len(hypotheses))
            return hypotheses[:num_expected]

        # Second pass: Fallback parsing with multiple format support
        logger.debug("Standard patterns found %d hypotheses, trying fallback parsing...", len(hypotheses))

        hypotheses = []
        current_hypothesis = ""

        hypothesis_patterns = [
            r"^(\d+)[.)]\s*(.+)$",  # "1. Text" or "1) Text"
            r"^(?:\*\*)?H(\d+)(?:\*\*)?[:.]\s*(.+)$",  # "H1: Text" or "**H1:** Text"
            r"^(?:\*\*)?Hypothesis\s+(\d+)(?:\*\*)?[:.]\s*(.+)$",  # "Hypothesis 1: Text"
            r"^(?:\*\*)?Approach\s+(\d+)(?:\*\*)?[:.]\s*(.+)$",  # "Approach 1: Text"
            r"^[•\-]\s+(.+)$",  # "• Text" or "- Text"
            r"^\*\s+(.+)$",  # "* Text" (single asterisk with space)
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

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
                        if hypothesis_num <= num_expected:
                            current_hypothesis = match.group(2).strip()
                            matched = True
                    else:
                        # Pattern without number (like "- Text")
                        if len(hypotheses) < num_expected:
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
        if len(hypotheses) < num_expected:
            # Look for sections separated by double newlines
            sections = re.split(r'\n\s*\n', content)
            for section in sections:
                section = section.strip()
                if len(section) > 30 and len(hypotheses) < num_expected:
                    # Skip sections that look like headers or metadata
                    if section.startswith("**Scale:**") or section.startswith("Scale:"):
                        continue
                    # Clean up section markers but preserve content
                    cleaned = re.sub(r'^(?:\d+[.)]\s*|[•\-]\s+|\*\s+|H\d+[:.]\s*)', '', section, flags=re.IGNORECASE)
                    if len(cleaned.strip()) > MIN_HYPOTHESIS_LENGTH and not any(h == cleaned.strip() for h in hypotheses):
                        hypotheses.append(cleaned.strip())

        return hypotheses[:num_expected] if len(hypotheses) >= num_expected else hypotheses

    @staticmethod
    def parse_with_fallback(
        response_content: str,
        num_expected: int
    ) -> List[str]:
        """
        Try structured JSON first, fall back to text parsing.

        This is the recommended entry point for hypothesis parsing.

        Args:
            response_content: LLM response text
            num_expected: Number of hypotheses expected

        Returns:
            List of hypothesis content strings
        """
        # Strategy 1: Structured JSON
        hypotheses = HypothesisParser.parse_structured_json(response_content, num_expected)
        if len(hypotheses) >= num_expected:
            return hypotheses

        # Strategy 2: Text parsing with patterns
        hypotheses = HypothesisParser.parse_text_with_patterns(response_content, num_expected)
        if len(hypotheses) >= num_expected:
            return hypotheses

        # Return what we got (may be less than expected)
        logger.warning(f"Failed to extract {num_expected} hypotheses. Got {len(hypotheses)}")
        return hypotheses


# ============================================================================
# 2. ScoreParser - Parse QADI scores from evaluation responses
# ============================================================================

class ScoreParser:
    """Parse evaluation scores from LLM responses."""

    @staticmethod
    def parse_structured_scores(response_content: str) -> Optional[ParsedScores]:
        """
        Parse structured JSON score response.

        Expected format:
        {
            "scores": {
                "impact": 0.8,
                "feasibility": 0.7,
                ...
            }
        }

        Or nested in evaluations:
        {
            "evaluations": [
                {
                    "scores": {...}
                }
            ]
        }

        Args:
            response_content: LLM response text

        Returns:
            ParsedScores if successful, None otherwise
        """
        try:
            # Try direct JSON parsing first
            data = json.loads(response_content)

            # Check for direct scores key
            if "scores" in data and isinstance(data["scores"], dict):
                scores_dict = data["scores"]
                return ParsedScores(
                    impact=float(scores_dict.get("impact", 0.5)),
                    feasibility=float(scores_dict.get("feasibility", 0.5)),
                    accessibility=float(scores_dict.get("accessibility", 0.5)),
                    sustainability=float(scores_dict.get("sustainability", 0.5)),
                    scalability=float(scores_dict.get("scalability", 0.5)),
                )

            # Check for nested scores in evaluations array
            if "evaluations" in data and isinstance(data["evaluations"], list):
                if len(data["evaluations"]) > 0:
                    first_eval = data["evaluations"][0]
                    if isinstance(first_eval, dict) and "scores" in first_eval:
                        scores_dict = first_eval["scores"]
                        return ParsedScores(
                            impact=float(scores_dict.get("impact", 0.5)),
                            feasibility=float(scores_dict.get("feasibility", 0.5)),
                            accessibility=float(scores_dict.get("accessibility", 0.5)),
                            sustainability=float(scores_dict.get("sustainability", 0.5)),
                            scalability=float(scores_dict.get("scalability", 0.5)),
                        )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug("Direct JSON score parsing failed: %s. Trying extract_and_parse_json...", e)

            # Try using json_utils for more robust extraction
            try:
                data = extract_and_parse_json(
                    response_content,
                    expected_keys=["scores", "evaluations"],
                    fix_issues=True
                )

                # Check for direct scores key
                if "scores" in data and isinstance(data["scores"], dict):
                    scores_dict = data["scores"]
                    return ParsedScores(
                        impact=float(scores_dict.get("impact", 0.5)),
                        feasibility=float(scores_dict.get("feasibility", 0.5)),
                        accessibility=float(scores_dict.get("accessibility", 0.5)),
                        sustainability=float(scores_dict.get("sustainability", 0.5)),
                        scalability=float(scores_dict.get("scalability", 0.5)),
                    )

                # Check for nested scores in evaluations array
                if "evaluations" in data and isinstance(data["evaluations"], list):
                    if len(data["evaluations"]) > 0:
                        first_eval = data["evaluations"][0]
                        if isinstance(first_eval, dict) and "scores" in first_eval:
                            scores_dict = first_eval["scores"]
                            return ParsedScores(
                                impact=float(scores_dict.get("impact", 0.5)),
                                feasibility=float(scores_dict.get("feasibility", 0.5)),
                                accessibility=float(scores_dict.get("accessibility", 0.5)),
                                sustainability=float(scores_dict.get("sustainability", 0.5)),
                                scalability=float(scores_dict.get("scalability", 0.5)),
                            )
            except Exception as e2:
                logger.debug("extract_and_parse_json also failed for scores: %s", e2)

        return None

    @staticmethod
    def extract_score_from_text(criterion: str, text: str) -> float:
        """
        Extract a single criterion score from text.

        Handles formats:
        - "Impact: 0.8 - explanation"
        - "**Impact:** 0.8 - explanation"
        - "* Impact: 0.8 - explanation"
        - "Impact: 8/10 - explanation" (fraction)

        Args:
            criterion: Score criterion name (e.g., "Impact")
            text: Text containing the score

        Returns:
            Score value (0.0-1.0), defaults to 0.5 if not found
        """
        # Try fractional scores first (e.g., "8/10")
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

        # Try various text patterns
        patterns = [
            rf"\*\*{criterion}:\*\*\s*(-?[0-9.]+)\s*-",  # "**Impact:** 0.8 - explanation"
            rf"\*\*{criterion}:\*\*\s*(-?[0-9.]+)",  # "**Impact:** 0.8"
            rf"\*?\s*{criterion}:\s*(-?[0-9.]+)\s*-",  # "* Impact: 0.8 - explanation"
            rf"\*?\s*{criterion}:\s*(-?[0-9.]+)",  # "Impact: 0.8"
            rf"{criterion}\s*-\s*(-?[0-9.]+)",  # "Impact - 0.8"
            rf"{criterion}\s*:\s*(-?[0-9.]+)/?",  # "Impact: 0.8/"
            rf"{criterion}\s*\((-?[0-9.]+)\)",  # "Impact (0.8)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    continue

        return 0.5  # Default

    @staticmethod
    def _extract_hypothesis_section(content: str, hypothesis_num: int) -> str:
        """
        Extract text section for a specific hypothesis number.

        Looks for scoring/evaluation sections which typically come after
        hypothesis generation sections. Handles cases where there might be
        multiple occurrences of "H1:", "H2:", etc. by looking for score-related content.

        Args:
            content: Full response content
            hypothesis_num: Hypothesis number to extract (1-based)

        Returns:
            Text section for that hypothesis
        """
        lines = content.split("\n")

        # First, try to find a section that contains both the hypothesis marker AND score criteria
        # This helps distinguish between hypothesis generation and evaluation sections
        all_sections = []
        section_lines = []
        in_section = False
        current_hypothesis_num = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is the start of a hypothesis section
            hypothesis_match = re.match(
                r"^(?:-\s*)?(?:\*\*)?(?:H|Hypothesis\s+|Approach\s+)(\d+)[:.](.*?)(?:\*\*)?$",
                line,
                re.IGNORECASE
            )

            if hypothesis_match:
                # Save previous section if we have one
                if in_section and section_lines:
                    all_sections.append((current_hypothesis_num, section_lines.copy()))

                # Start new section
                current_hypothesis_num = int(hypothesis_match.group(1))
                in_section = True
                section_lines = [hypothesis_match.group(2).strip()]
                continue

            # Check if we've reached the next hypothesis or end marker
            if in_section:
                # Check for next hypothesis or end sections
                next_match = re.match(
                    r"^(?:-\s*)?(?:\*\*)?(?:H|Hypothesis\s+|Approach\s+)(\d+)[:.](.*?)(?:\*\*)?$",
                    line,
                    re.IGNORECASE
                )
                if next_match:
                    next_num = int(next_match.group(1))
                    if next_num != current_hypothesis_num:
                        # Save current section and start next
                        all_sections.append((current_hypothesis_num, section_lines.copy()))
                        current_hypothesis_num = next_num
                        section_lines = [next_match.group(2).strip()]
                        continue

                if line.startswith(("ANSWER:", "Action Plan:")):
                    # Save current section and stop
                    all_sections.append((current_hypothesis_num, section_lines.copy()))
                    break

                section_lines.append(line)

        # Don't forget the last section
        if in_section and section_lines:
            all_sections.append((current_hypothesis_num, section_lines.copy()))

        # Now find the section for our hypothesis_num that has score criteria
        # Prioritize sections with score keywords (Impact, Feasibility, etc.)
        best_section = None
        for hyp_num, lines_list in all_sections:
            if hyp_num == hypothesis_num:
                section_text = " ".join(lines_list)
                # Check if this section has score criteria
                has_scores = any(keyword in section_text for keyword in
                                ["Impact:", "Feasibility:", "Accessibility:", "Sustainability:", "Scalability:"])
                if has_scores:
                    best_section = section_text
                    break
                elif best_section is None:
                    # Keep first matching section as fallback
                    best_section = section_text

        return best_section or ""

    @staticmethod
    def parse_text_scores(
        response_content: str,
        hypothesis_num: Optional[int] = None
    ) -> ParsedScores:
        """
        Parse scores from unstructured text.

        If hypothesis_num is provided, extracts scores for that specific hypothesis
        (e.g., "H1:", "Approach 1:"). Otherwise parses the entire response.

        Args:
            response_content: LLM response text
            hypothesis_num: Optional hypothesis number (1-based) to extract

        Returns:
            ParsedScores with extracted values (defaults to 0.5 for missing)
        """
        content = clean_ansi_codes(response_content.strip())

        # If hypothesis_num specified, extract that section
        if hypothesis_num is not None:
            section = ScoreParser._extract_hypothesis_section(content, hypothesis_num)
            if not section:
                logger.warning(
                    "Could not find hypothesis %d section. Using default scores.",
                    hypothesis_num
                )
                return ParsedScores(0.5, 0.5, 0.5, 0.5, 0.5)
            content = section

        # Extract each criterion
        scores = ParsedScores(
            impact=ScoreParser.extract_score_from_text("Impact", content),
            feasibility=ScoreParser.extract_score_from_text("Feasibility", content),
            accessibility=ScoreParser.extract_score_from_text("Accessibility", content),
            sustainability=ScoreParser.extract_score_from_text("Sustainability", content),
            scalability=ScoreParser.extract_score_from_text("Scalability", content),
        )

        return scores

    @staticmethod
    def parse_with_fallback(
        response_content: str,
        hypothesis_num: Optional[int] = None
    ) -> ParsedScores:
        """
        Try structured JSON first, fall back to text parsing.

        This is the recommended entry point for score parsing.

        Args:
            response_content: LLM response text
            hypothesis_num: Optional hypothesis number (1-based) to extract

        Returns:
            ParsedScores with extracted values
        """
        # Strategy 1: Structured JSON (only if no specific hypothesis requested)
        if hypothesis_num is None:
            scores = ScoreParser.parse_structured_scores(response_content)
            if scores is not None:
                logger.debug("Successfully parsed scores via structured output")
                return scores

        # Strategy 2: Text parsing
        scores = ScoreParser.parse_text_scores(response_content, hypothesis_num)
        logger.debug("Successfully parsed scores via text parsing")
        return scores


# ============================================================================
# 3. ActionPlanParser - Parse action plans and next steps
# ============================================================================

class ActionPlanParser:
    """Parse action plans, next steps, and other list-based content."""

    @staticmethod
    def parse_structured_plan(response_content: str) -> List[str]:
        """
        Parse structured JSON action plan.

        Expected format:
        {
            "action_plan": ["Item 1", "Item 2", ...]
        }

        Also tries alternative keys: next_steps, recommendations, actions

        Args:
            response_content: LLM response text

        Returns:
            List of action items (empty if parsing fails)
        """
        try:
            # Try direct JSON parsing first
            data = json.loads(response_content)

            # Try various key names
            for key in ["action_plan", "next_steps", "recommendations", "actions"]:
                if key in data and isinstance(data[key], list):
                    items = [
                        str(item).strip()
                        for item in data[key]
                        if item and len(str(item).strip()) >= MIN_ACTION_ITEM_LENGTH
                    ]
                    if items:
                        logger.debug("Successfully parsed %d action items from key '%s'", len(items), key)
                        return items

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Direct JSON action plan parsing failed: %s. Trying extract_and_parse_json...", e)

            # Try using json_utils for more robust extraction
            try:
                data = extract_and_parse_json(
                    response_content,
                    expected_keys=["action_plan", "next_steps", "recommendations"],
                    fix_issues=True
                )

                # Try various key names
                for key in ["action_plan", "next_steps", "recommendations", "actions"]:
                    if key in data and isinstance(data[key], list):
                        items = [
                            str(item).strip()
                            for item in data[key]
                            if item and len(str(item).strip()) >= MIN_ACTION_ITEM_LENGTH
                        ]
                        if items:
                            logger.debug("Successfully parsed %d action items via json_utils", len(items))
                            return items

            except Exception as e2:
                logger.debug("extract_and_parse_json also failed for action plan: %s", e2)

        return []

    @staticmethod
    def parse_text_list(
        response_content: str,
        section_prefix: str = "Action Plan:"
    ) -> List[str]:
        """
        Parse list from unstructured text.

        Handles:
        - Numbered lists: "1. Item", "2. Item"
        - Bullet lists: "- Item", "* Item", "• Item"
        - Multi-line items

        Args:
            response_content: Full response text
            section_prefix: Section header to look for (e.g., "Action Plan:")

        Returns:
            List of action items
        """
        content = clean_ansi_codes(response_content.strip())

        # Find the section
        plan_match = re.search(
            rf"{section_prefix}\s*(.+?)$",
            content,
            re.DOTALL | re.IGNORECASE
        )

        if not plan_match:
            return []

        plan_text = plan_match.group(1).strip()

        # Extract numbered items or bullet points
        # Improved regex to handle multi-line items properly
        # Captures everything until the next item marker or end of string
        plan_items = re.findall(
            r"(?:\d+\.|[-*•])\s*(.+?)(?=\s*\n\s*(?:\d+\.|[-*•])|$)",
            plan_text,
            re.DOTALL,
        )

        items = [
            item.strip()
            for item in plan_items
            if item.strip() and len(item.strip()) > 1  # Filter out single character items
        ]

        # If no items found with bullets/numbers, try splitting by newlines
        if not items:
            lines = plan_text.split("\n")
            items = [
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
                and len(line.strip()) >= MIN_ACTION_ITEM_LENGTH
            ]

        return items

    @staticmethod
    def parse_with_fallback(
        response_content: str,
        section_prefix: str = "Action Plan:"
    ) -> List[str]:
        """
        Try structured JSON first, fall back to text parsing.

        This is the recommended entry point for action plan parsing.

        Args:
            response_content: LLM response text
            section_prefix: Section header for text parsing

        Returns:
            List of action items
        """
        # Strategy 1: Structured JSON
        items = ActionPlanParser.parse_structured_plan(response_content)
        if items:
            logger.debug(f"Successfully parsed {len(items)} action items via structured output")
            return items

        # Strategy 2: Text parsing
        items = ActionPlanParser.parse_text_list(response_content, section_prefix)
        if items:
            logger.debug(f"Successfully parsed {len(items)} action items via text parsing")
            return items

        logger.debug("Failed to extract action plan items")
        return []
