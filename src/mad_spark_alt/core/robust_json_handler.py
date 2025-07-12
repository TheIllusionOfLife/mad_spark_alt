"""
Robust JSON handling for LLM responses that might be malformed.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def extract_json_from_response(
    response: str,
    expected_keys: Optional[List[str]] = None,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract JSON from LLM response with multiple fallback strategies.

    Args:
        response: Raw LLM response
        expected_keys: Keys expected in the JSON
        fallback: Fallback dictionary if extraction fails

    Returns:
        Extracted JSON or fallback
    """
    if not response:
        logger.warning("Empty response, using fallback")
        return fallback or {}

    # Strategy 1: Direct JSON parsing
    try:
        data = json.loads(response)
        if _validate_json(data, expected_keys):
            return data  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if _validate_json(data, expected_keys):
                return data  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON-like structure
    json_patterns = [
        r"\{[^{}]*\}",  # Simple object
        r"\{.*?\}(?=\s*$)",  # Object at end
        r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})",  # Nested objects
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if _validate_json(data, expected_keys):
                    return data  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue

    # Strategy 4: Try to fix common issues
    fixed_response = _fix_common_json_issues(response)
    if fixed_response != response:
        try:
            data = json.loads(fixed_response)
            if _validate_json(data, expected_keys):
                return data  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Strategy 5: Extract key-value pairs manually
    if expected_keys:
        extracted = _extract_key_values(response, expected_keys)
        if extracted:
            return extracted

    logger.warning(f"Failed to extract JSON from response: {response[:200]}...")
    return fallback or {}


def _validate_json(data: Any, expected_keys: Optional[List[str]] = None) -> bool:
    """Validate that JSON has expected structure."""
    if not isinstance(data, dict):
        return False

    if expected_keys:
        return any(key in data for key in expected_keys)

    return True


def _fix_common_json_issues(text: str) -> str:
    """Fix common JSON formatting issues."""
    # Remove trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*\]", "]", text)

    # Fix single quotes
    text = text.replace("'", '"')

    # Fix unquoted keys
    text = re.sub(r"(\w+):", r'"\1":', text)

    # Remove comments
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    return text


def _extract_key_values(text: str, keys: List[str]) -> Dict[str, Any]:
    """Extract key-value pairs from text."""
    result = {}

    for key in keys:
        # Look for "key": "value" or "key": value patterns
        patterns = [
            rf'"{key}"\s*:\s*"([^"]*)"',
            rf'"{key}"\s*:\s*(\[[^\]]*\])',
            rf'"{key}"\s*:\s*(\d+\.?\d*)',
            rf'"{key}"\s*:\s*(true|false|null)',
            rf'{key}\s*:\s*"([^"]*)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    # Try to parse as JSON
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError, TypeError):
                    # Keep as string if JSON parsing fails
                    result[key] = value
                break

    return result


def safe_parse_ideas_array(
    response: str,
    max_ideas: int = 10,
    fallback_ideas: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse an array of ideas from LLM response.

    Args:
        response: Raw LLM response
        max_ideas: Maximum number of ideas to extract
        fallback_ideas: Fallback list if parsing fails

    Returns:
        List of idea dictionaries
    """
    # First try standard JSON extraction
    result = extract_json_from_response(
        response,
        expected_keys=["ideas", "questions", "hypotheses", "insights"],
        fallback={"ideas": fallback_ideas or []},
    )

    # Extract ideas array from various possible keys
    ideas = (
        result.get("ideas")
        or result.get("questions")
        or result.get("hypotheses")
        or result.get("insights")
        or []
    )

    if isinstance(ideas, list):
        return ideas[:max_ideas]

    # If not a list, try to extract individual ideas
    extracted_ideas = []

    # Pattern for numbered items
    numbered_pattern = r"(?:^|\n)\s*(?:\d+\.?|[-•*])\s*(.+?)(?=\n\s*(?:\d+\.?|[-•*])|$)"
    matches = re.findall(numbered_pattern, response, re.MULTILINE | re.DOTALL)

    for match in matches[:max_ideas]:
        idea_text = match.strip()
        if len(idea_text) > 10:  # Filter out too short items
            extracted_ideas.append({"content": idea_text, "extracted": True})

    return extracted_ideas or fallback_ideas or []
