"""
Tests for ANSI escape code removal.

These tests ensure all ANSI codes are properly cleaned from output.
"""

import pytest

from mad_spark_alt.utils.text_cleaning import clean_ansi_codes


class TestANSICodeCleaning:
    """Test ANSI escape code removal functionality."""

    def test_clean_standard_ansi_codes(self):
        """Test removal of standard ANSI escape codes."""
        # Test text with various ANSI codes
        text_with_ansi = "\x1b[1mBold text\x1b[0m and \x1b[31mred text\x1b[0m"
        expected = "Bold text and red text"
        
        result = clean_ansi_codes(text_with_ansi)
        assert result == expected

    def test_clean_orphaned_ansi_codes(self):
        """Test removal of ANSI codes that lost their escape character."""
        # Sometimes ANSI codes appear without \x1b
        text_with_orphaned = "[1mApproach 1:[0m The solution"
        expected = "Approach 1: The solution"
        
        result = clean_ansi_codes(text_with_orphaned)
        assert result == expected

    def test_clean_complex_ansi_patterns(self):
        """Test removal of complex ANSI patterns from real LLM output."""
        # Real examples from the output
        test_cases = [
            ("[1mPersonal Approach:[0m The idea", "Personal Approach: The idea"),
            ("[1mH1:[0m Hypothesis", "H1: Hypothesis"),
            ("[1mApproach 2:[0m Another idea", "Approach 2: Another idea"),
            ("[3mItalic text[0m", "Italic text"),
            ("[33mYellow text[0m", "Yellow text"),
            ("[1;33mBold yellow[0m", "Bold yellow"),
        ]
        
        for input_text, expected in test_cases:
            result = clean_ansi_codes(input_text)
            assert result == expected, f"Failed for: {input_text}"

    def test_clean_nested_ansi_codes(self):
        """Test removal of nested ANSI codes."""
        text = "[1m[3mBold and italic[0m[0m text"
        expected = "Bold and italic text"
        
        result = clean_ansi_codes(text)
        assert result == expected

    def test_clean_multiline_text_with_ansi(self):
        """Test cleaning multiline text with ANSI codes."""
        text = """[1mApproach 1:[0m First approach
[1mApproach 2:[0m Second approach
[1mApproach 3:[0m Third approach"""
        
        expected = """Approach 1: First approach
Approach 2: Second approach
Approach 3: Third approach"""
        
        result = clean_ansi_codes(text)
        assert result == expected

    def test_preserve_non_ansi_brackets(self):
        """Test that non-ANSI bracket patterns are preserved."""
        text = "Array[0] and dict['key'] should be preserved"
        result = clean_ansi_codes(text)
        assert result == text  # Should not change

    def test_clean_partial_ansi_codes(self):
        """Test removal of partial or malformed ANSI codes."""
        test_cases = [
            ("[1m", ""),  # Incomplete code
            ("[0m", ""),  # Reset without start
            ("Text[1m", "Text"),  # Code at end
            ("[999mInvalid code[0m", "Invalid code"),  # Invalid code number
        ]
        
        for input_text, expected in test_cases:
            result = clean_ansi_codes(input_text)
            assert result == expected

    def test_clean_empty_and_none(self):
        """Test handling of empty strings and None."""
        assert clean_ansi_codes("") == ""
        assert clean_ansi_codes(None) == ""

    def test_performance_with_large_text(self):
        """Test performance with large text containing many ANSI codes."""
        # Create a large text with many ANSI codes
        large_text = ""
        for i in range(1000):
            large_text += f"[1mLine {i}:[0m Some text with [31mcolor[0m\n"
        
        result = clean_ansi_codes(large_text)
        
        # Should not contain any ANSI codes
        assert "[1m" not in result
        assert "[0m" not in result
        assert "[31m" not in result
        assert "Line 999: Some text with color" in result


class TestANSICleaningIntegration:
    """Test ANSI cleaning in the context of the application."""

    def test_clean_qadi_hypothesis_output(self):
        """Test cleaning actual QADI hypothesis output."""
        # Real output from the system
        hypothesis = "[1mApproach 1:[0m The Individual Phenomenological-Neural Mapping Project"
        expected = "Approach 1: The Individual Phenomenological-Neural Mapping Project"
        
        result = clean_ansi_codes(hypothesis)
        assert result == expected

    def test_clean_evolution_display(self):
        """Test cleaning evolution display messages."""
        messages = [
            "[1m1.[0m [1mImmediate Action (Today):[0m Take action",
            "[1;33m 1 [0m[1mHighest Overall Score:[0m H3 achieved",
        ]
        
        expected = [
            "1. Immediate Action (Today): Take action",
            " 1 Highest Overall Score: H3 achieved",
        ]
        
        for msg, exp in zip(messages, expected):
            result = clean_ansi_codes(msg)
            assert result == exp

    def test_terminal_renderer_integration(self):
        """Test that terminal renderer properly cleans ANSI codes."""
        # This would test the actual integration with render_markdown
        # For now, we just test that the function exists and can be imported
        try:
            from mad_spark_alt.core.terminal_renderer import render_markdown
            # The actual integration test would go here
            assert render_markdown is not None
        except ImportError:
            pytest.skip("Terminal renderer not available")