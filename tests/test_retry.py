"""Tests for retry module error handling."""

import pytest

from mad_spark_alt.core.retry import _extract_error_message


class TestExtractErrorMessage:
    """Tests for _extract_error_message helper function."""

    def test_standard_nested_format(self):
        """Test standard format: {"error": {"message": "..."}}."""
        data = {"error": {"message": "test error"}}
        assert _extract_error_message(data, "default") == "test error"

    def test_simple_string_format_ollama(self):
        """Test Ollama simple string format: {"error": "..."}."""
        data = {"error": "test error from ollama"}
        assert _extract_error_message(data, "default") == "test error from ollama"

    def test_direct_message_format(self):
        """Test direct message format: {"message": "..."}."""
        data = {"message": "direct error message"}
        assert _extract_error_message(data, "default") == "direct error message"

    def test_empty_response_returns_default(self):
        """Test empty response returns default value."""
        assert _extract_error_message({}, "default") == "default"

    def test_nested_error_without_message_returns_default(self):
        """Test nested error dict without message key returns default."""
        data = {"error": {"code": 500}}
        assert _extract_error_message(data, "default") == "default"

    def test_nested_error_with_none_message_returns_default(self):
        """Test nested error with None message returns default, not 'None' string."""
        data = {"error": {"message": None}}
        assert _extract_error_message(data, "default") == "default"

    def test_top_level_none_message_returns_default(self):
        """Test top-level None message returns default, not 'None' string."""
        data = {"message": None}
        assert _extract_error_message(data, "default") == "default"

    def test_none_error_value_returns_message(self):
        """Test when error key exists but is None, falls through to message."""
        data = {"error": None, "message": "fallback message"}
        assert _extract_error_message(data, "default") == "fallback message"

    def test_numeric_error_returns_default(self):
        """Test when error is a number, returns default (not string format)."""
        data = {"error": 500}
        assert _extract_error_message(data, "default") == "default"

    def test_empty_string_error_returns_empty_string(self):
        """Test empty string error is returned as-is."""
        data = {"error": ""}
        assert _extract_error_message(data, "default") == ""

    def test_preserves_special_characters(self):
        """Test that special characters in error messages are preserved."""
        data = {"error": "Error: 日本語メッセージ with special chars <>&"}
        assert _extract_error_message(data, "default") == "Error: 日本語メッセージ with special chars <>&"

    def test_nested_message_with_extra_fields(self):
        """Test nested error with extra fields still extracts message."""
        data = {"error": {"message": "the error", "code": 429, "retry_after": 60}}
        assert _extract_error_message(data, "default") == "the error"
