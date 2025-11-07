"""
Utilities for parsing JSON from LLM responses.

LLMs often return JSON wrapped in markdown code blocks or with additional text.
This module provides robust JSON extraction and parsing.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extract JSON content from LLM response text.

    Handles common patterns:
    - JSON wrapped in ```json ... ``` code blocks
    - JSON wrapped in ``` ... ``` code blocks
    - Pure JSON responses
    - JSON with surrounding text

    Args:
        text: Raw text response from LLM

    Returns:
        Extracted JSON string, or None if no JSON found
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Pattern 1: JSON in ```json code blocks (supports both objects and arrays)
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: Look for JSON objects with proper braces (improved for deeper nesting)
    # Use multiple strategies for better matching
    json_patterns = [
        r"\{[^{}]*\}",  # Simple objects without nesting
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}",  # Objects with one level of nesting
        r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}",  # Objects with two levels of nesting
    ]

    object_matches = []
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        object_matches.extend(matches)

    # Pattern 3: Look for JSON arrays with improved bracket matching
    array_patterns = [
        r"\[[^\[\]]*\]",  # Simple arrays without nesting
        r"\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]",  # Arrays with one level of nesting
    ]

    array_matches = []
    for pattern in array_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        array_matches.extend(matches)

    all_matches = object_matches + array_matches
    if all_matches:
        # Return the largest JSON-like match
        return str(max(all_matches, key=len)).strip()

    # Pattern 4: Try to extract anything between first { and last } (objects)
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end].strip()

    # Pattern 5: Try to extract anything between first [ and last ] (arrays)
    if "[" in text and "]" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        return text[start:end].strip()

    return None


def safe_json_parse(
    text: str, fallback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response with fallback.

    Args:
        text: Raw text response from LLM
        fallback: Fallback dictionary if parsing fails

    Returns:
        Parsed JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {}

    try:
        # First try to parse as-is
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # If not a dict, fall through to extraction or fallback
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON from response
    extracted_json = extract_json_from_response(text)
    if extracted_json:
        try:
            result = json.loads(extracted_json)
            if isinstance(result, dict):
                return result
            # If not a dict, fall through to fallback
        except json.JSONDecodeError:
            pass

    # Return fallback if all parsing attempts fail
    return fallback


def parse_json_list(text: str, fallback: Optional[List[Any]] = None) -> List[Any]:
    """
    Parse JSON array from LLM response.

    Args:
        text: Raw text response from LLM
        fallback: Fallback list if parsing fails

    Returns:
        Parsed JSON list or fallback
    """
    if fallback is None:
        fallback = []

    try:
        # First try to parse as-is
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # If it's a dict, try to find a list value
            for value in result.values():
                if isinstance(value, list):
                    return value
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON from response
    extracted_json = extract_json_from_response(text)
    if extracted_json:
        try:
            result = json.loads(extracted_json)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                for value in result.values():
                    if isinstance(value, list):
                        return value
        except json.JSONDecodeError:
            pass

    return fallback


def _is_valid_non_empty_string(value: Any) -> bool:
    """Check if value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def validate_json_structure(data: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that parsed JSON has required structure.

    Args:
        data: Parsed JSON data
        required_keys: List of required keys

    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False

    return all(key in data for key in required_keys)


def validate_crossover_response(data: Dict[str, Any]) -> bool:
    """
    Validate LLM crossover response structure.

    Expected format:
    {
        "offspring1": {"content": "...", "reasoning": "..."},
        "offspring2": {"content": "...", "reasoning": "..."}
    }

    Args:
        data: Parsed JSON data

    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False

    required_keys = ["offspring1", "offspring2"]
    if not all(key in data for key in required_keys):
        return False

    for offspring_key in required_keys:
        offspring = data[offspring_key]
        if not isinstance(offspring, dict):
            return False
        if not all(key in offspring for key in ["content", "reasoning"]):
            return False
        if not _is_valid_non_empty_string(offspring["content"]):
            return False
        if not _is_valid_non_empty_string(offspring["reasoning"]):
            return False

    return True


def validate_mutation_response(data: Dict[str, Any]) -> bool:
    """
    Validate LLM mutation response structure.

    Expected format:
    {
        "mutated_idea": {
            "content": "...",
            "mutation_type": "...",
            "reasoning": "..."
        }
    }

    Args:
        data: Parsed JSON data

    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False

    if "mutated_idea" not in data:
        return False

    mutated_idea = data["mutated_idea"]
    if not isinstance(mutated_idea, dict):
        return False

    required_keys = ["content", "mutation_type", "reasoning"]
    if not all(key in mutated_idea for key in required_keys):
        return False

    for key in required_keys:
        if not _is_valid_non_empty_string(mutated_idea[key]):
            return False

    return True


def validate_selection_response(data: Dict[str, Any]) -> bool:
    """
    Validate LLM selection advisor response structure.

    Expected format:
    {
        "selection_scores": [
            {"index": 0, "score": 0.9, "reasoning": "..."},
            ...
        ],
        "recommended_parents": [0, 1, ...],
        "diversity_consideration": "..."
    }

    Args:
        data: Parsed JSON data

    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False

    required_keys = [
        "selection_scores",
        "recommended_parents",
        "diversity_consideration",
    ]
    if not all(key in data for key in required_keys):
        return False

    # Validate selection_scores
    if not isinstance(data["selection_scores"], list):
        return False
    for score_entry in data["selection_scores"]:
        if not isinstance(score_entry, dict):
            return False
        if not all(key in score_entry for key in ["index", "score", "reasoning"]):
            return False
        if not isinstance(score_entry["index"], int) or score_entry["index"] < 0:
            return False
        if not isinstance(score_entry["score"], (int, float)) or not (
            0 <= score_entry["score"] <= 1
        ):
            return False
        if not _is_valid_non_empty_string(score_entry["reasoning"]):
            return False

    # Validate recommended_parents
    if not isinstance(data["recommended_parents"], list):
        return False
    for parent_idx in data["recommended_parents"]:
        if not isinstance(parent_idx, int) or parent_idx < 0:
            return False

    # Validate diversity_consideration
    if not _is_valid_non_empty_string(data["diversity_consideration"]):
        return False

    return True


def validate_qadi_response(data: Dict[str, Any]) -> bool:
    """
    Validate QADI agent response structure.

    Expected format:
    {
        "ideas": [
            {"content": "...", "reasoning": "..."},
            ...
        ]
    }

    Args:
        data: Parsed JSON data

    Returns:
        True if structure is valid
    """
    if not isinstance(data, dict):
        return False

    if "ideas" not in data:
        return False

    ideas = data["ideas"]
    if not isinstance(ideas, list):
        return False

    for idea in ideas:
        if not isinstance(idea, dict):
            return False
        if not all(key in idea for key in ["content", "reasoning"]):
            return False
        if not _is_valid_non_empty_string(idea["content"]):
            return False
        if not _is_valid_non_empty_string(idea["reasoning"]):
            return False

    return True


def safe_json_parse_with_validation(
    text: str,
    validator_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response with optional validation.

    Args:
        text: Raw text response from LLM
        validator_func: Optional function to validate parsed JSON structure
        fallback: Fallback dictionary if parsing or validation fails

    Returns:
        Parsed and validated JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {}

    # Parse JSON using existing safe_json_parse
    parsed_data = safe_json_parse(text, fallback)

    # If we got the fallback, validation would fail anyway
    if parsed_data == fallback:
        return fallback

    # Apply validation if provided
    if validator_func and not validator_func(parsed_data):
        return fallback

    return parsed_data


def _fix_common_json_issues(text: str) -> str:
    """
    Fix common JSON formatting issues from LLM responses.

    Handles:
    - Trailing commas before closing braces/brackets
    - Single quotes instead of double quotes
    - Unquoted keys
    - JavaScript-style comments (// and /* */)

    Args:
        text: Raw JSON-like text

    Returns:
        Fixed JSON text
    """
    # Remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)

    # Fix single quotes (be careful with escaped quotes and apostrophes in values)
    text = text.replace("'", '"')

    # Fix unquoted keys (use negative lookbehind to avoid keys already in quotes)
    text = re.sub(r'(?<!")\b(\w+)\b(?=\s*:)', r'"\1"', text)

    # Remove comments
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    return text


def _extract_with_multiple_keys(
    data: Dict[str, Any],
    keys: List[str]
) -> Optional[Any]:
    """
    Try to extract value using multiple possible keys.

    Args:
        data: Parsed JSON dictionary
        keys: List of possible keys to try

    Returns:
        First matching value found, or None
    """
    for key in keys:
        if key in data:
            return data[key]
    return None


def extract_and_parse_json(
    text: str,
    expected_keys: Optional[List[str]] = None,
    fix_issues: bool = True,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    One-step JSON extraction, fixing, and parsing with validation.

    This convenience function combines:
    1. JSON extraction from various formats (markdown, plain text, etc.)
    2. Common issue fixing (trailing commas, quotes, comments)
    3. JSON parsing with fallback
    4. Optional validation for expected keys

    Args:
        text: Raw text response from LLM
        expected_keys: Optional list of keys expected in the JSON
        fix_issues: Whether to attempt fixing common JSON issues
        fallback: Fallback dictionary if parsing fails

    Returns:
        Parsed JSON dictionary or fallback
    """
    if fallback is None:
        fallback = {}

    if not text or not isinstance(text, str):
        return fallback

    # Strategy 1: Try direct parsing first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # If expected_keys provided, validate at least one exists
            if expected_keys:
                if _extract_with_multiple_keys(data, expected_keys) is not None:
                    return data
            else:
                return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: Extract from markdown or surrounding text
    extracted_json = extract_json_from_response(text)
    if extracted_json:
        try:
            data = json.loads(extracted_json)
            if isinstance(data, dict):
                if expected_keys:
                    if _extract_with_multiple_keys(data, expected_keys) is not None:
                        return data
                else:
                    return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try fixing common issues and parsing again
    if fix_issues:
        fixed_text = _fix_common_json_issues(text)
        if fixed_text != text:
            try:
                data = json.loads(fixed_text)
                if isinstance(data, dict):
                    if expected_keys:
                        if _extract_with_multiple_keys(data, expected_keys) is not None:
                            return data
                    else:
                        return data
            except json.JSONDecodeError:
                pass

            # Try extraction after fixing
            extracted_fixed = extract_json_from_response(fixed_text)
            if extracted_fixed:
                try:
                    data = json.loads(extracted_fixed)
                    if isinstance(data, dict):
                        if expected_keys:
                            if _extract_with_multiple_keys(data, expected_keys) is not None:
                                return data
                        else:
                            return data
                except json.JSONDecodeError:
                    pass

    return fallback


def parse_ideas_array(
    text: str,
    max_ideas: Optional[int] = None,
    fallback_keys: Optional[List[str]] = None,
    fallback_ideas: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse an array of ideas from LLM response with multiple fallback strategies.

    Tries to find ideas using various key names and formats:
    1. JSON with keys: "ideas", "hypotheses", "questions", "insights"
    2. Custom keys provided in fallback_keys
    3. Numbered list extraction (1., 2., 3., etc.)
    4. Bullet list extraction (-, *, •)
    5. Fallback list if all parsing fails

    Args:
        text: Raw text response from LLM
        max_ideas: Maximum number of ideas to return
        fallback_keys: Additional keys to try for finding ideas array
        fallback_ideas: Fallback list if parsing fails

    Returns:
        List of idea dictionaries
    """
    if fallback_ideas is None:
        fallback_ideas = []

    if not text or not isinstance(text, str):
        return fallback_ideas

    # Default keys to try
    default_keys = ["ideas", "hypotheses", "questions", "insights"]
    if fallback_keys:
        # Custom keys take priority
        keys_to_try = fallback_keys + default_keys
    else:
        keys_to_try = default_keys

    # Strategy 1: Try to extract and parse JSON with fixing enabled
    result = extract_and_parse_json(
        text,
        expected_keys=keys_to_try,
        fix_issues=True,
        fallback={}
    )

    # Extract ideas array from various possible keys
    ideas = _extract_with_multiple_keys(result, keys_to_try)

    if isinstance(ideas, list) and len(ideas) > 0:
        # Apply max_ideas limit if specified
        if max_ideas is not None:
            return ideas[:max_ideas]
        return ideas

    # Strategy 2: Try numbered list extraction
    # Pattern matches: "1.", "2.", etc. at start of line or after newline
    numbered_pattern = r"(?:^|\n)\s*(?:\d+\.?|\d+\))\s+(.+?)(?=\n\s*(?:\d+\.?|\d+\))|$)"
    numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)

    # Strategy 3: Try bullet list extraction
    # Pattern matches: "-", "*", "•" at start of line
    bullet_pattern = r"(?:^|\n)\s*[-•*]\s+(.+?)(?=\n\s*[-•*]|$)"
    bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)

    # Combine all extracted items
    all_matches = numbered_matches + bullet_matches

    extracted_ideas = []
    MIN_ITEM_LENGTH = 10  # Filter out too-short items

    for match in all_matches:
        item_text = match.strip()
        if len(item_text) >= MIN_ITEM_LENGTH:
            extracted_ideas.append({
                "content": item_text,
                "extracted": True
            })

    if extracted_ideas:
        # Apply max_ideas limit
        if max_ideas is not None:
            return extracted_ideas[:max_ideas]
        return extracted_ideas

    # Return fallback if nothing extracted
    return fallback_ideas


def format_llm_cost(cost: float) -> str:
    """
    Format LLM API cost with smart thresholds for better user experience.

    Avoids showing "$0.0000" for very low costs and provides meaningful
    cost information based on actual LLM pricing tiers.

    Args:
        cost: LLM API cost in USD

    Returns:
        Formatted cost string

    Thresholds:
        - < $0.001: "Free (within API limits)"
        - $0.001-$0.10: Show actual cost with appropriate precision
        - > $0.10: Show cost with warning indicator
    """
    if cost < 0.001:
        return "Free (within API limits)"
    elif cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 0.10:
        return f"${cost:.3f}"
    else:
        return f"⚠️ ${cost:.2f}"
