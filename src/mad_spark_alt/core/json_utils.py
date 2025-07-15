"""
Utilities for parsing JSON from LLM responses.

LLMs often return JSON wrapped in markdown code blocks or with additional text.
This module provides robust JSON extraction and parsing.
"""

import json
import re
from typing import Any, Dict, Optional, Callable


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


def parse_json_list(text: str, fallback: Optional[list] = None) -> list:
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
        if (
            not isinstance(offspring["content"], str)
            or not offspring["content"].strip()
        ):
            return False
        if (
            not isinstance(offspring["reasoning"], str)
            or not offspring["reasoning"].strip()
        ):
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
        if not isinstance(mutated_idea[key], str) or not mutated_idea[key].strip():
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
        if (
            not isinstance(score_entry["reasoning"], str)
            or not score_entry["reasoning"].strip()
        ):
            return False

    # Validate recommended_parents
    if not isinstance(data["recommended_parents"], list):
        return False
    for parent_idx in data["recommended_parents"]:
        if not isinstance(parent_idx, int) or parent_idx < 0:
            return False

    # Validate diversity_consideration
    if (
        not isinstance(data["diversity_consideration"], str)
        or not data["diversity_consideration"].strip()
    ):
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
        if not isinstance(idea["content"], str) or not idea["content"].strip():
            return False
        if not isinstance(idea["reasoning"], str) or not idea["reasoning"].strip():
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
