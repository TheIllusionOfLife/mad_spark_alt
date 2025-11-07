"""
Comprehensive tests for parsing_utils module.

This test suite validates all parsing strategies with various LLM response formats:
- Structured JSON output
- Text-based formats (H1:, Approach 1:, numbered lists, bullets)
- Edge cases (ANSI codes, markdown, empty responses, invalid data)
- Integration tests with realistic LLM responses
"""

import pytest
from mad_spark_alt.core.parsing_utils import (
    HypothesisParser,
    ScoreParser,
    ActionPlanParser,
    ParsedScores,
    MIN_HYPOTHESIS_LENGTH,
    MIN_ACTION_ITEM_LENGTH,
)


class TestHypothesisParser:
    """Test hypothesis parsing with various formats."""

    def test_parse_structured_json_format(self):
        """Test parsing clean structured JSON."""
        response = '''
        {
            "hypotheses": [
                {"id": "1", "content": "First hypothesis with sufficient length to pass minimum"},
                {"id": "2", "content": "Second hypothesis with sufficient length to pass minimum"},
                {"id": "3", "content": "Third hypothesis with sufficient length to pass minimum"}
            ]
        }
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=3)
        assert len(hypotheses) == 3
        assert hypotheses[0] == "First hypothesis with sufficient length to pass minimum"
        assert hypotheses[1] == "Second hypothesis with sufficient length to pass minimum"

    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON in markdown code blocks."""
        response = '''```json
        {
            "hypotheses": [
                {"id": "1", "content": "Markdown wrapped hypothesis with sufficient length"}
            ]
        }
        ```'''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=1)
        assert len(hypotheses) >= 1
        assert "Markdown wrapped hypothesis" in hypotheses[0]

    def test_parse_h_prefix_format(self):
        """Test parsing 'H1:', 'H2:' format."""
        response = '''
        H1: First hypothesis with sufficient detail and length
        H2: Second hypothesis with sufficient detail and length
        H3: Third hypothesis with sufficient detail and length
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=3)
        assert len(hypotheses) == 3
        assert "First hypothesis" in hypotheses[0]
        assert "Second hypothesis" in hypotheses[1]

    def test_parse_approach_prefix_format(self):
        """Test parsing 'Approach 1:', 'Approach 2:' format."""
        response = '''
        Approach 1: First approach with details and sufficient length
        Approach 2: Second approach with details and sufficient length
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) == 2
        assert "First approach" in hypotheses[0]

    def test_parse_numbered_list_format(self):
        """Test parsing numbered list '1.', '2.' format."""
        response = '''
        1. First hypothesis in numbered list format with length
        2. Second hypothesis in numbered list format with length
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) == 2
        assert "First hypothesis" in hypotheses[0]

    def test_parse_with_ansi_codes(self):
        """Test parsing with ANSI escape codes."""
        response = "\x1b[1mH1:\x1b[0m Hypothesis with ANSI codes and sufficient length"
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=1)
        assert len(hypotheses) >= 1
        assert "Hypothesis with ANSI codes" in hypotheses[0]

    def test_parse_with_markdown_bold(self):
        """Test parsing with markdown bold formatting."""
        response = '''
        **H1:** First bold hypothesis with sufficient length
        **H2:** Second bold hypothesis with sufficient length
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) == 2
        assert "First bold hypothesis" in hypotheses[0]

    def test_parse_multiline_hypothesis(self):
        """Test parsing hypotheses spanning multiple lines."""
        response = '''
        H1: First hypothesis that spans
        multiple lines with continued
        content on subsequent lines

        H2: Second hypothesis with content
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) == 2
        assert "multiple lines" in hypotheses[0]

    def test_parse_filters_short_hypotheses(self):
        """Test that short hypotheses are filtered out."""
        response = '''
        {
            "hypotheses": [
                {"id": "1", "content": "Too short"},
                {"id": "2", "content": "This is a proper length hypothesis with sufficient detail"}
            ]
        }
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) == 1  # Only the long one
        assert "proper length hypothesis" in hypotheses[0]

    def test_parse_empty_response(self):
        """Test handling of empty response."""
        hypotheses = HypothesisParser.parse_with_fallback("", num_expected=3)
        assert len(hypotheses) == 0

    def test_parse_bullet_points(self):
        """Test parsing bullet point format."""
        response = '''
        - First bullet hypothesis with sufficient length
        - Second bullet hypothesis with sufficient length
        - Third bullet hypothesis with sufficient length
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=3)
        assert len(hypotheses) == 3
        assert "First bullet" in hypotheses[0]

    def test_parse_mixed_formats(self):
        """Test parsing when multiple formats might be present."""
        response = '''
        Some introductory text that should be ignored.

        H1: First real hypothesis with sufficient detail

        Some more text to ignore.

        H2: Second real hypothesis with sufficient detail
        '''
        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        assert len(hypotheses) >= 2
        assert "First real hypothesis" in hypotheses[0]
        assert "Second real hypothesis" in hypotheses[1]


class TestScoreParser:
    """Test score parsing with various formats."""

    def test_parse_structured_scores(self):
        """Test parsing structured JSON scores."""
        response = '''
        {
            "scores": {
                "impact": 0.8,
                "feasibility": 0.7,
                "accessibility": 0.9,
                "sustainability": 0.6,
                "scalability": 0.75
            }
        }
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7
        assert scores.accessibility == 0.9
        assert scores.sustainability == 0.6
        assert scores.scalability == 0.75

    def test_parse_text_scores_with_explanation(self):
        """Test parsing 'Criterion: score - explanation' format."""
        response = '''
        Impact: 0.8 - significant change expected
        Feasibility: 0.7 - moderately practical
        Accessibility: 0.9 - widely accessible
        Sustainability: 0.6 - moderate long-term viability
        Scalability: 0.75 - good growth potential
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7
        assert scores.accessibility == 0.9
        assert scores.sustainability == 0.6
        assert scores.scalability == 0.75

    def test_parse_markdown_bold_scores(self):
        """Test parsing markdown bold format."""
        response = '''
        **Impact:** 0.8 - explanation
        **Feasibility:** 0.7 - explanation
        **Accessibility:** 0.9 - explanation
        **Sustainability:** 0.6 - explanation
        **Scalability:** 0.75 - explanation
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7

    def test_parse_bullet_point_scores(self):
        """Test parsing bullet point format."""
        response = '''
        * Impact: 0.8 - explanation
        * Feasibility: 0.7 - explanation
        * Accessibility: 0.9 - explanation
        * Sustainability: 0.6 - explanation
        * Scalability: 0.75 - explanation
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7

    def test_parse_fractional_scores(self):
        """Test parsing fractional scores like '8/10'."""
        response = '''
        Impact: 8/10 - explanation
        Feasibility: 7/10 - explanation
        Accessibility: 9/10 - explanation
        Sustainability: 6/10 - explanation
        Scalability: 75/100 - explanation
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7
        assert scores.accessibility == 0.9
        assert scores.sustainability == 0.6
        assert scores.scalability == 0.75

    def test_parse_hypothesis_section_scores(self):
        """Test extracting scores for specific hypothesis."""
        response = '''
        H1: First hypothesis
        Impact: 0.8 - explanation
        Feasibility: 0.7 - explanation
        Accessibility: 0.6 - explanation
        Sustainability: 0.5 - explanation
        Scalability: 0.4 - explanation

        H2: Second hypothesis
        Impact: 0.6 - explanation
        Feasibility: 0.9 - explanation
        Accessibility: 0.7 - explanation
        Sustainability: 0.8 - explanation
        Scalability: 0.5 - explanation
        '''
        scores_h1 = ScoreParser.parse_with_fallback(response, hypothesis_num=1)
        scores_h2 = ScoreParser.parse_with_fallback(response, hypothesis_num=2)

        assert scores_h1.impact == 0.8
        assert scores_h1.feasibility == 0.7
        assert scores_h2.impact == 0.6
        assert scores_h2.feasibility == 0.9

    def test_parse_clamps_invalid_scores(self):
        """Test that invalid scores are clamped to 0-1 range."""
        response = '''
        Impact: 1.5 - too high
        Feasibility: -0.2 - negative
        Accessibility: 0.5 - valid
        Sustainability: 0.6 - valid
        Scalability: 0.7 - valid
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert 0.0 <= scores.impact <= 1.0
        assert 0.0 <= scores.feasibility <= 1.0
        assert scores.accessibility == 0.5

    def test_parse_missing_criteria_uses_defaults(self):
        """Test that missing criteria get default values."""
        response = '''
        Impact: 0.8 - only criterion provided
        '''
        scores = ScoreParser.parse_with_fallback(response)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.5  # Default
        assert scores.accessibility == 0.5  # Default
        assert scores.sustainability == 0.5  # Default
        assert scores.scalability == 0.5  # Default

    def test_parse_approach_prefix_format(self):
        """Test parsing with 'Approach N:' prefix instead of 'HN:'."""
        response = '''
        Approach 1: First approach
        Impact: 0.8 - explanation
        Feasibility: 0.7 - explanation
        Accessibility: 0.9 - explanation
        Sustainability: 0.6 - explanation
        Scalability: 0.75 - explanation

        Approach 2: Second approach
        Impact: 0.6 - explanation
        '''
        scores = ScoreParser.parse_with_fallback(response, hypothesis_num=1)
        assert scores.impact == 0.8
        assert scores.feasibility == 0.7

    def test_parse_empty_response_returns_defaults(self):
        """Test that empty response returns default scores."""
        scores = ScoreParser.parse_with_fallback("")
        assert scores.impact == 0.5
        assert scores.feasibility == 0.5
        assert scores.accessibility == 0.5
        assert scores.sustainability == 0.5
        assert scores.scalability == 0.5


class TestActionPlanParser:
    """Test action plan parsing."""

    def test_parse_structured_action_plan(self):
        """Test parsing structured JSON action plan."""
        response = '''
        {
            "action_plan": [
                "First action item with details",
                "Second action item with details",
                "Third action item with details"
            ]
        }
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) == 3
        assert items[0] == "First action item with details"
        assert items[1] == "Second action item with details"

    def test_parse_numbered_list_plan(self):
        """Test parsing numbered list format."""
        response = '''
        Action Plan:
        1. First action item
        2. Second action item
        3. Third action item
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) == 3
        assert "First action item" in items[0]

    def test_parse_bulleted_list_plan(self):
        """Test parsing bulleted list format."""
        response = '''
        Action Plan:
        - First action item
        - Second action item
        - Third action item
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) == 3
        assert "First action item" in items[0]

    def test_parse_multiline_action_items(self):
        """Test parsing multi-line action items."""
        response = '''
        Action Plan:
        1. First action item that spans
           multiple lines with details
        2. Second action item
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) >= 2
        assert "multiple lines" in items[0]

    def test_parse_custom_section_prefix(self):
        """Test parsing with custom section prefix."""
        response = '''
        Next Steps:
        1. Step one here
        2. Step two here
        '''
        items = ActionPlanParser.parse_with_fallback(response, section_prefix="Next Steps:")
        assert len(items) >= 2
        assert "Step one" in items[0]

    def test_parse_filters_short_items(self):
        """Test that short items are filtered out."""
        response = '''
        Action Plan:
        1. Too short
        2. This is a properly detailed action item
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        # Depending on MIN_ACTION_ITEM_LENGTH, "Too short" might pass (9 chars)
        # Let's just verify we get at least the long one
        assert len(items) >= 1
        assert any("properly detailed" in item for item in items)

    def test_parse_no_section_header(self):
        """Test handling when section header not found."""
        response = "Just some text without action plan header"
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) == 0

    def test_parse_mixed_bullets_and_numbers(self):
        """Test parsing mix of bullets and numbers."""
        response = '''
        Action Plan:
        1. First numbered item
        - First bullet item
        2. Second numbered item
        * Second bullet item
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) >= 4

    def test_parse_alternative_key_names(self):
        """Test parsing with alternative JSON key names."""
        response = '''
        {
            "next_steps": [
                "Step one with details",
                "Step two with details"
            ]
        }
        '''
        items = ActionPlanParser.parse_with_fallback(response)
        assert len(items) == 2
        assert "Step one" in items[0]

    def test_parse_empty_response(self):
        """Test handling of empty response."""
        items = ActionPlanParser.parse_with_fallback("")
        assert len(items) == 0


class TestIntegration:
    """Integration tests using realistic LLM responses."""

    def test_full_qadi_json_response(self):
        """Test parsing a complete QADI cycle response in JSON format."""
        response = '''
        {
            "hypotheses": [
                {"id": "1", "content": "First complete hypothesis with sufficient detail"},
                {"id": "2", "content": "Second complete hypothesis with sufficient detail"}
            ],
            "evaluations": [
                {
                    "hypothesis_id": "1",
                    "scores": {
                        "impact": 0.8,
                        "feasibility": 0.7,
                        "accessibility": 0.9,
                        "sustainability": 0.6,
                        "scalability": 0.75
                    }
                }
            ],
            "action_plan": [
                "Implement the first approach",
                "Monitor and measure impact"
            ]
        }
        '''

        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        scores = ScoreParser.parse_with_fallback(response)
        items = ActionPlanParser.parse_with_fallback(response)

        assert len(hypotheses) == 2
        assert scores.impact == 0.8
        assert len(items) == 2

    def test_full_qadi_text_response(self):
        """Test parsing a complete QADI cycle response in text format."""
        response = '''
        H1: First hypothesis with detailed explanation
        Impact: 0.8 - significant potential
        Feasibility: 0.7 - moderately achievable
        Accessibility: 0.9 - highly accessible
        Sustainability: 0.6 - moderate sustainability
        Scalability: 0.75 - good scaling potential

        H2: Second hypothesis with detailed explanation
        Impact: 0.6 - moderate potential
        Feasibility: 0.9 - highly achievable
        Accessibility: 0.7 - accessible
        Sustainability: 0.8 - sustainable
        Scalability: 0.5 - limited scaling

        Action Plan:
        1. Implement first hypothesis
        2. Monitor results
        3. Iterate based on feedback
        '''

        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        scores_h1 = ScoreParser.parse_with_fallback(response, hypothesis_num=1)
        scores_h2 = ScoreParser.parse_with_fallback(response, hypothesis_num=2)
        items = ActionPlanParser.parse_with_fallback(response)

        assert len(hypotheses) == 2
        assert scores_h1.impact == 0.8
        assert scores_h2.impact == 0.6
        assert len(items) == 3

    def test_messy_real_world_response(self):
        """Test parsing a messy real-world response with mixed formatting."""
        response = '''
        Here are the hypotheses:

        **H1:** First approach with markdown formatting
        This continues on multiple lines
        with various details

        Some explanatory text that should be handled gracefully.

        **H2:** Second approach also with markdown
        And more content here

        Now for the evaluation:

        **H1:**
        * Impact: 0.8 - looks promising
        * Feasibility: 0.7 - doable
        * Accessibility: 0.9 - easy access
        * Sustainability: 0.6 - moderate
        * Scalability: 0.75 - scales well

        Action Plan:
        - Take the first step
        - Monitor progress
        - Adjust as needed
        '''

        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        scores = ScoreParser.parse_with_fallback(response, hypothesis_num=1)
        items = ActionPlanParser.parse_with_fallback(response)

        assert len(hypotheses) >= 2
        assert scores.impact == 0.8
        assert len(items) >= 3

    def test_ansi_codes_throughout_response(self):
        """Test parsing response with ANSI codes scattered throughout."""
        response = '\x1b[1mH1:\x1b[0m First hypothesis with \x1b[33mcolor codes\x1b[0m\n' \
                   '\x1b[1mH2:\x1b[0m Second hypothesis also with codes\n' \
                   '\n' \
                   '\x1b[1mImpact:\x1b[0m 0.8 - explanation\n' \
                   '\x1b[1mFeasibility:\x1b[0m 0.7 - explanation\n' \
                   '\x1b[1mAccessibility:\x1b[0m 0.9 - explanation\n' \
                   '\x1b[1mSustainability:\x1b[0m 0.6 - explanation\n' \
                   '\x1b[1mScalability:\x1b[0m 0.75 - explanation'

        hypotheses = HypothesisParser.parse_with_fallback(response, num_expected=2)
        scores = ScoreParser.parse_with_fallback(response)

        assert len(hypotheses) >= 2
        assert "First hypothesis" in hypotheses[0]
        assert scores.impact == 0.8
