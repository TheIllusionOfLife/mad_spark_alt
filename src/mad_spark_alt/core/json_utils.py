"""
Utilities for parsing JSON from LLM responses.

LLMs often return JSON wrapped in markdown code blocks or with additional text.
This module provides robust JSON extraction and parsing.
"""

import json
import re
from typing import Any, Dict, Optional


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
