"""Tests for robust LLM response parsing."""

import pytest
import json
from mad_spark_alt.core.json_utils import (
    extract_json_from_response,
    safe_json_parse,
    parse_json_list,
    safe_json_parse_with_validation,
    # New functions to be implemented
    extract_and_parse_json,
    parse_ideas_array,
)


class TestJsonExtraction:
    """Test JSON extraction from various LLM response formats."""
    
    def test_extract_json_from_markdown_block(self):
        """Test extraction from markdown json code block."""
        response = '''Here is the analysis:
        
```json
{
  "score": 0.85,
  "reasoning": "High quality implementation"
}
```
        
The score reflects the overall quality.'''
        
        extracted = extract_json_from_response(response)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["score"] == 0.85
        assert "reasoning" in parsed
    
    def test_extract_json_from_plain_code_block(self):
        """Test extraction from plain code block."""
        response = '''```
{
  "items": ["a", "b", "c"],
  "count": 3
}
```'''
        
        extracted = extract_json_from_response(response)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["count"] == 3
        assert len(parsed["items"]) == 3
    
    def test_extract_json_array(self):
        """Test extraction of JSON array."""
        response = '''Here are the results:
        
```json
[
  {"id": 1, "name": "First"},
  {"id": 2, "name": "Second"}
]
```'''
        
        extracted = extract_json_from_response(response)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
    
    def test_extract_nested_json(self):
        """Test extraction of deeply nested JSON."""
        # Note: Current implementation has limitations with deeply nested JSON
        # It may not capture the full structure
        response = '''```json
{
  "level1": {
    "level2": {
      "level3": {
        "value": "deep"
      }
    }
  }
}
```'''
        
        extracted = extract_json_from_response(response)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["level1"]["level2"]["level3"]["value"] == "deep"
    
    def test_extract_json_with_surrounding_text(self):
        """Test extraction when JSON is surrounded by text."""
        response = '''Based on my analysis, here are the scores:
        
{"impact": 0.9, "feasibility": 0.7, "overall": 0.8}

These scores indicate a strong proposal.'''
        
        extracted = extract_json_from_response(response)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["impact"] == 0.9
    
    def test_no_json_returns_none(self):
        """Test that non-JSON text returns None."""
        response = "This is just regular text without any JSON."
        extracted = extract_json_from_response(response)
        assert extracted is None


class TestSafeJsonParse:
    """Test safe JSON parsing with fallback."""
    
    def test_parse_valid_json_string(self):
        """Test parsing valid JSON string."""
        text = '{"status": "success", "value": 42}'
        result = safe_json_parse(text)
        assert result["status"] == "success"
        assert result["value"] == 42
    
    def test_parse_json_in_markdown(self):
        """Test parsing JSON from markdown block."""
        text = '''```json
{
  "method": "test",
  "params": ["a", "b"]
}
```'''
        result = safe_json_parse(text)
        assert result["method"] == "test"
        assert result["params"] == ["a", "b"]
    
    def test_parse_with_default_fallback(self):
        """Test fallback to default value on parse failure."""
        text = "Invalid JSON {broken"
        default = {"error": "fallback"}
        result = safe_json_parse(text, fallback=default)
        assert result == default
    
    def test_parse_malformed_json_attempts_fix(self):
        """Test parsing attempts to fix common issues."""
        # Missing quotes around keys
        text = '{status: "ok", value: 123}'
        result = safe_json_parse(text)
        # Should return fallback since this is not valid JSON
        assert result == {}
    
    def test_parse_with_comments(self):
        """Test parsing JSON with comments (non-standard)."""
        text = '''{
  "name": "test", // This is a comment
  "value": 100
}'''
        # Standard JSON parser will fail on comments
        result = safe_json_parse(text, fallback={"parsed": False})
        assert result == {"parsed": False}


class TestParseJsonArray:
    """Test JSON array parsing."""
    
    def test_parse_valid_array(self):
        """Test parsing valid JSON array."""
        text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = parse_json_list(text)
        assert len(result) == 3
        assert result[0]["id"] == 1
    
    def test_parse_array_in_markdown(self):
        """Test parsing array from markdown."""
        text = '''```json
[
  {"name": "Alice", "score": 95},
  {"name": "Bob", "score": 87}
]
```'''
        result = parse_json_list(text)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
    
    def test_parse_empty_array(self):
        """Test parsing empty array."""
        text = "[]"
        result = parse_json_list(text)
        assert result == []
    
    def test_parse_invalid_array_returns_empty(self):
        """Test invalid array returns empty list."""
        text = "Not an array at all"
        result = parse_json_list(text)
        assert result == []
    
    def test_parse_single_object_as_array(self):
        """Test single object is wrapped in array."""
        text = '{"single": "object"}'
        result = parse_json_list(text)
        # Should return empty list since it's not an array
        assert result == []


class TestJsonValidation:
    """Test JSON parsing with validation."""
    
    def test_validate_required_fields(self):
        """Test validation of required fields."""
        text = '{"name": "test", "value": 100}'
        
        def validator(data):
            return "name" in data and "value" in data
        
        result = safe_json_parse_with_validation(text, validator_func=validator)
        assert result["name"] == "test"
        assert result["value"] == 100
    
    def test_validation_failure_returns_fallback(self):
        """Test validation failure returns fallback."""
        text = '{"name": "test"}'  # Missing required "value" field
        
        def validator(data):
            return "name" in data and "value" in data
        
        fallback = {"error": "validation failed"}
        result = safe_json_parse_with_validation(
            text, 
            validator_func=validator, 
            fallback=fallback
        )
        assert result == fallback
    
    def test_validate_nested_structure(self):
        """Test validation of nested structure."""
        text = '''{
  "user": {
    "id": 123,
    "profile": {
      "name": "John"
    }
  }
}'''
        
        def validator(data):
            return (
                "user" in data and
                "profile" in data["user"] and
                "name" in data["user"]["profile"]
            )
        
        result = safe_json_parse_with_validation(text, validator_func=validator)
        assert result["user"]["profile"]["name"] == "John"


class TestEdgeCases:
    """Test edge cases in JSON parsing."""
    
    def test_handle_none_input(self):
        """Test None input handling."""
        assert extract_json_from_response(None) is None
        assert safe_json_parse(None) == {}
        assert parse_json_list(None) == []
    
    def test_handle_empty_string(self):
        """Test empty string handling."""
        assert extract_json_from_response("") is None
        assert safe_json_parse("") == {}
        assert parse_json_list("") == []
    
    def test_handle_non_string_input(self):
        """Test non-string input handling."""
        assert extract_json_from_response(123) is None
        assert safe_json_parse(123) == {}
        assert parse_json_list(123) == []
    
    def test_unicode_in_json(self):
        """Test Unicode characters in JSON."""
        text = '{"message": "Hello 世界 🌍"}'
        result = safe_json_parse(text)
        assert result["message"] == "Hello 世界 🌍"
    
    def test_large_numbers(self):
        """Test parsing large numbers."""
        text = '{"big": 9999999999999999, "float": 3.141592653589793}'
        result = safe_json_parse(text)
        assert result["big"] == 9999999999999999
        assert abs(result["float"] - 3.141592653589793) < 0.000001
    
    def test_special_characters_in_strings(self):
        """Test special characters in string values."""
        text = r'{"path": "C:\\Users\\test", "regex": "\\d+", "quote": "He said \"hello\""}'
        result = safe_json_parse(text)
        assert result["path"] == "C:\\Users\\test"
        assert result["regex"] == "\\d+"
        assert result["quote"] == 'He said "hello"'


class TestRealWorldLLMResponses:
    """Test parsing of actual LLM response patterns."""
    
    def test_gpt_style_response(self):
        """Test GPT-style response with explanation."""
        response = '''I'll analyze this problem step by step.

```json
{
  "analysis": {
    "complexity": "medium",
    "estimated_time": "2 hours",
    "requirements": ["Python 3.8+", "numpy", "pandas"]
  },
  "score": 7.5
}
```

Based on this analysis, the task is achievable within the timeframe.'''
        
        result = safe_json_parse(response)
        assert result["score"] == 7.5
        assert result["analysis"]["complexity"] == "medium"
    
    def test_claude_style_response(self):
        """Test Claude-style response with thinking."""
        response = '''Let me evaluate these options:

First, I'll consider the criteria...

Here's my evaluation:

{
  "options": [
    {"id": "A", "score": 0.8, "pros": ["Fast", "Simple"]},
    {"id": "B", "score": 0.9, "pros": ["Scalable", "Robust"]}
  ],
  "recommendation": "B"
}

Option B is better due to its scalability.'''
        
        result = safe_json_parse(response)
        assert result["recommendation"] == "B"
        assert len(result["options"]) == 2
    
    def test_gemini_style_response(self):
        """Test Gemini-style response."""
        response = '''**Analysis Results:**

```json
{
  "hypotheses": [
    {
      "id": 1,
      "description": "Direct approach",
      "confidence": 0.85
    },
    {
      "id": 2,
      "description": "Alternative method",
      "confidence": 0.72
    }
  ],
  "selected": 1
}
```

**Conclusion:** The direct approach shows higher confidence.'''
        
        result = safe_json_parse(response)
        assert result["selected"] == 1
        assert len(result["hypotheses"]) == 2


class TestExtractAndParseJson:
    """Test extract_and_parse_json() convenience function."""

    def test_clean_json_direct_parse(self):
        """Test direct parsing of clean JSON."""
        text = '{"status": "ok", "value": 123}'
        result = extract_and_parse_json(text)
        assert result["status"] == "ok"
        assert result["value"] == 123

    def test_markdown_wrapped_json(self):
        """Test extraction and parsing from markdown block."""
        text = '''```json
{
  "name": "test",
  "count": 5
}
```'''
        result = extract_and_parse_json(text)
        assert result["name"] == "test"
        assert result["count"] == 5

    def test_json_with_trailing_commas(self):
        """Test JSON fixing with trailing commas."""
        text = '{"items": ["a", "b", "c",], "count": 3,}'
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["count"] == 3
        assert len(result["items"]) == 3

    def test_json_with_single_quotes(self):
        """Test JSON fixing with single quotes."""
        text = "{'name': 'test', 'value': 100}"
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["name"] == "test"
        assert result["value"] == 100

    def test_json_with_apostrophe_in_value(self):
        """Test that single-quote fixing doesn't break apostrophes in values."""
        text = '{"review": "it\'s a test", "description": "won\'t fail"}'
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["review"] == "it's a test"
        assert result["description"] == "won't fail"

    def test_json_with_apostrophe_in_single_quoted_string(self):
        """Test single-quoted strings containing apostrophes (e.g. O'Reilly)."""
        text = "{'name': 'O\\'Reilly', 'company': 'O\\'Brien & Associates'}"
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["name"] == "O'Reilly"
        assert result["company"] == "O'Brien & Associates"

    def test_json_with_unquoted_keys(self):
        """Test JSON fixing with unquoted keys."""
        text = '{name: "test", value: 42}'
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_json_with_comments(self):
        """Test JSON fixing with JavaScript-style comments."""
        text = '''{
  // This is a comment
  "status": "ok", /* multi-line
  comment */
  "value": 99
}'''
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["status"] == "ok"
        assert result["value"] == 99

    def test_json_with_url_not_treated_as_comment(self):
        """Test that URLs like https://example.com are not broken by comment removal."""
        text = '{"url": "https://example.com/path", "api": "http://api.test.com"}'
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["url"] == "https://example.com/path"
        assert result["api"] == "http://api.test.com"

    def test_json_with_comment_slashes_inside_string(self):
        """Test that // inside string values is preserved."""
        text = '{"note": "profit // loss backup", "formula": "a // b"}'
        result = extract_and_parse_json(text, fix_issues=True)
        assert result["note"] == "profit // loss backup"
        assert result["formula"] == "a // b"

    def test_expected_keys_validation(self):
        """Test validation with expected keys."""
        text = '{"ideas": [1, 2, 3], "count": 3}'
        result = extract_and_parse_json(text, expected_keys=["ideas"])
        assert "ideas" in result
        assert result["count"] == 3

    def test_expected_keys_missing_uses_fallback(self):
        """Test fallback when expected keys are missing."""
        text = '{"other": "data"}'
        fallback = {"error": "no ideas found"}
        result = extract_and_parse_json(text, expected_keys=["ideas"], fallback=fallback)
        # Should return fallback when expected keys are completely missing
        assert result == fallback

    def test_fallback_on_parse_failure(self):
        """Test fallback value on complete parse failure."""
        text = "Not JSON at all!"
        fallback = {"default": True}
        result = extract_and_parse_json(text, fallback=fallback)
        assert result == fallback

    def test_multiple_key_extraction(self):
        """Test extraction with multiple possible keys."""
        text = '{"hypotheses": [{"id": 1}, {"id": 2}]}'
        result = extract_and_parse_json(text, expected_keys=["ideas", "hypotheses"])
        assert "hypotheses" in result
        assert len(result["hypotheses"]) == 2

    def test_no_fixing_mode(self):
        """Test without JSON fixing (strict mode)."""
        text = "{'name': 'test'}"  # Single quotes won't parse in strict mode
        fallback = {"strict": True}
        result = extract_and_parse_json(text, fix_issues=False, fallback=fallback)
        assert result == fallback


class TestParseIdeasArray:
    """Test parse_ideas_array() function."""

    def test_parse_ideas_key(self):
        """Test parsing with 'ideas' key."""
        text = '''```json
{
  "ideas": [
    {"description": "First idea", "score": 0.8},
    {"description": "Second idea", "score": 0.9}
  ]
}
```'''
        result = parse_ideas_array(text)
        assert len(result) == 2
        assert result[0]["description"] == "First idea"
        assert result[1]["score"] == 0.9

    def test_parse_hypotheses_key(self):
        """Test parsing with 'hypotheses' key."""
        text = '{"hypotheses": [{"id": 1, "text": "Hypothesis 1"}, {"id": 2, "text": "Hypothesis 2"}]}'
        result = parse_ideas_array(text)
        assert len(result) == 2
        assert result[0]["text"] == "Hypothesis 1"

    def test_parse_questions_key(self):
        """Test parsing with 'questions' key."""
        text = '{"questions": [{"q": "What?"}, {"q": "Why?"}]}'
        result = parse_ideas_array(text)
        assert len(result) == 2
        assert result[0]["q"] == "What?"

    def test_parse_insights_key(self):
        """Test parsing with 'insights' key."""
        text = '{"insights": [{"insight": "Key finding"}]}'
        result = parse_ideas_array(text)
        assert len(result) == 1
        assert result[0]["insight"] == "Key finding"

    def test_max_ideas_limit(self):
        """Test limiting number of ideas returned."""
        text = '{"ideas": [{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}]}'
        result = parse_ideas_array(text, max_ideas=3)
        assert len(result) == 3
        assert result[2]["n"] == 3

    def test_numbered_list_fallback(self):
        """Test fallback to numbered list extraction."""
        text = '''Here are the ideas:
1. First idea that is important
2. Second idea with details
3. Third idea about something
'''
        result = parse_ideas_array(text)
        assert len(result) == 3
        assert "First idea" in result[0]["content"]
        assert "Second idea" in result[1]["content"]

    def test_bullet_list_fallback(self):
        """Test fallback to bullet list extraction."""
        text = '''Ideas:
- First bullet point idea
- Second bullet point idea
• Third bullet point idea
* Fourth bullet point idea
'''
        result = parse_ideas_array(text)
        assert len(result) == 4  # All 4 bullet points should be extracted
        # Check that we extracted meaningful content
        assert all(len(idea.get("content", "")) > 10 for idea in result)

    def test_empty_response_uses_fallback(self):
        """Test empty response uses fallback ideas."""
        text = ""
        fallback = [{"id": 1, "fallback": True}]
        result = parse_ideas_array(text, fallback_ideas=fallback)
        assert result == fallback

    def test_invalid_json_uses_fallback(self):
        """Test invalid JSON with no extractable ideas uses fallback."""
        text = "Random text with no ideas"
        fallback = [{"default": "idea"}]
        result = parse_ideas_array(text, fallback_ideas=fallback)
        # Should try extraction first, then fallback if nothing found
        assert result == fallback

    def test_mixed_format_with_fixing(self):
        """Test parsing with JSON fixing enabled."""
        text = "{'ideas': [{'text': 'Idea one'}, {'text': 'Idea two',}]}"
        result = parse_ideas_array(text)
        assert len(result) == 2
        assert result[0]["text"] == "Idea one"

    def test_custom_fallback_keys(self):
        """Test custom fallback keys."""
        text = '{"custom_key": [{"data": "value"}]}'
        result = parse_ideas_array(text, fallback_keys=["custom_key"])
        assert len(result) == 1
        assert result[0]["data"] == "value"

    def test_filter_short_items(self):
        """Test filtering out too-short extracted items."""
        text = '''Ideas:
1. OK
2. This is a proper idea with enough text
3. No
'''
        result = parse_ideas_array(text)
        # Should filter out items under minimum length and keep the valid one
        assert len(result) == 1
        assert all(len(idea.get("content", "")) > 10 for idea in result)

    def test_nested_arrays_in_ideas(self):
        """Test ideas with nested array structures."""
        text = '''{"ideas": [
            {"text": "First", "tags": ["a", "b"]},
            {"text": "Second", "tags": ["c", "d"]}
        ]}'''
        result = parse_ideas_array(text)
        assert len(result) == 2
        assert result[0]["tags"] == ["a", "b"]