"""Validation tests to ensure prompts and parsers are compatible."""

import re
from mad_spark_alt.core.qadi_prompts import QADIPrompts
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator


class TestPromptParserValidation:
    """Tests to validate that prompts and parsers are compatible."""

    def test_deduction_prompt_format_matches_parser_expectations(self):
        """Verify that deduction prompt format matches what parser expects."""
        prompts = QADIPrompts()
        orchestrator = SimpleQADIOrchestrator()
        
        # Generate a sample prompt
        sample_prompt = prompts.get_deduction_prompt(
            "Test input",
            "Test question", 
            "H1: First hypothesis\nH2: Second hypothesis\nH3: Third hypothesis"
        )
        
        # Extract the expected format from the prompt
        format_section = re.search(r"Format:(.*?)(?:ANSWER:|$)", sample_prompt, re.DOTALL)
        assert format_section, "Prompt should contain Format: section"
        
        format_text = format_section.group(1).strip()
        
        # Check that the format specifies the structures our parser looks for
        assert "H1:" in format_text, "Format should specify H1: structure"
        assert "H2:" in format_text, "Format should specify H2: structure" 
        assert "H3:" in format_text, "Format should specify H3: structure"
        assert "Novelty:" in format_text, "Format should specify Novelty: scoring"
        assert "Impact:" in format_text, "Format should specify Impact: scoring"
        assert "Cost:" in format_text, "Format should specify Cost: scoring"
        assert "Feasibility:" in format_text, "Format should specify Feasibility: scoring"
        assert "Risks:" in format_text, "Format should specify Risks: scoring"
        
        # Test that our parser can handle the format specified in the prompt
        # Create a mock response that follows the exact format from the prompt
        mock_response = """Analysis:
- H1: 
  * Novelty: 0.7 - innovative approach
  * Impact: 0.8 - significant change expected
  * Cost: 0.6 - moderate resources needed
  * Feasibility: 0.9 - very practical
  * Risks: 0.5 - some implementation challenges
  * Overall: 0.70

- H2: 
  * Novelty: 0.5 - conventional approach  
  * Impact: 0.6 - moderate impact
  * Cost: 0.8 - low resource requirements
  * Feasibility: 0.7 - reasonably practical
  * Risks: 0.8 - low risk factors
  * Overall: 0.68

ANSWER: Based on analysis, H1 is recommended."""
        
        # Test that parser can extract scores from this format
        score1 = orchestrator._parse_hypothesis_scores(mock_response, 1)
        score2 = orchestrator._parse_hypothesis_scores(mock_response, 2)
        
        # Verify scores were parsed correctly (not defaults)
        assert score1.novelty == 0.7, f"Expected 0.7, got {score1.novelty}"
        assert score1.impact == 0.8, f"Expected 0.8, got {score1.impact}"
        assert score2.novelty == 0.5, f"Expected 0.5, got {score2.novelty}"
        assert score2.impact == 0.6, f"Expected 0.6, got {score2.impact}"

    def test_parser_handles_prompt_format_variations(self):
        """Test that parser can handle reasonable variations of the prompt format."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Test various formats that LLMs might reasonably produce
        format_variations = [
            # Standard bullet format
            """- H1: Test hypothesis
  * Novelty: 0.7 - explanation
  * Impact: 0.8 - explanation""",
            
            # Bold markdown format (like real LLM responses)
            """- **H1: Test hypothesis**
  * Novelty: 0.7 - explanation  
  * Impact: 0.8 - explanation""",
            
            # Without bullets
            """H1: Test hypothesis
Novelty: 0.7 - explanation
Impact: 0.8 - explanation""",
            
            # Different punctuation
            """H1. Test hypothesis
Novelty: 0.7 - explanation
Impact: 0.8 - explanation""",
        ]
        
        for i, format_variation in enumerate(format_variations):
            score = orchestrator._parse_hypothesis_scores(format_variation, 1)
            assert score.novelty == 0.7, f"Format variation {i} failed: novelty={score.novelty}"
            assert score.impact == 0.8, f"Format variation {i} failed: impact={score.impact}"

    def test_parser_graceful_degradation(self):
        """Test that parser gracefully handles unparseable formats."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Test cases where parsing should fail gracefully
        unparseable_formats = [
            "Completely unrelated content",
            "H1: Something but no scores",
            "Random text with numbers 0.7 and 0.8 but wrong format",
            "", # Empty content
        ]
        
        for unparseable in unparseable_formats:
            score = orchestrator._parse_hypothesis_scores(unparseable, 1)
            # Should return default scores without crashing
            assert score.novelty == 0.5
            assert score.impact == 0.5
            assert score.cost == 0.5
            assert score.feasibility == 0.5
            assert score.risks == 0.5