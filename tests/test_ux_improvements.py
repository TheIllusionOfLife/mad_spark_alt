"""
Tests for UX improvements to output display.

Tests cover:
1. Output truncation fixes
2. Evaluation score display improvements  
3. Evolution output cleanup
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import sys
from io import StringIO

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import IndividualFitness


class TestOutputTruncation:
    """Test that output truncation preserves readability."""
    
    def test_example_truncation_completes_sentences(self):
        """Test that examples are truncated at sentence boundaries."""
        # Import the truncation logic (will be extracted to a function)
        from qadi_simple import truncate_at_sentence_boundary
        
        # Test case 1: Text with multiple sentences
        text = "This is the first sentence. This is the second sentence that is quite long and contains important information. This is the third."
        result = truncate_at_sentence_boundary(text, 100)
        assert result.endswith(".")
        assert len(result) <= 100
        assert "first sentence" in result
        
        # Test case 2: Very long single sentence
        text = "This is a very long sentence that goes on and on without any periods or natural breaking points which makes it difficult to truncate nicely"
        result = truncate_at_sentence_boundary(text, 50)
        assert result.endswith("...")
        assert len(result) <= 53  # 50 + "..."
        
        # Test case 3: Text already within limit
        text = "Short text."
        result = truncate_at_sentence_boundary(text, 100)
        assert result == text
        assert not result.endswith("...")
    
    def test_context_and_application_truncation_limits(self):
        """Test that context and application lines use appropriate limits."""
        from qadi_simple import format_example_output
        
        example = """Context: A primary school student, Maya, living in an urban area, is increasingly aware of environmental issues through media but lacks practical knowledge about solutions. She wants to make a difference but doesn't know where to start.
Application: Maya starts a school-wide plastic reduction campaign, organizing weekly collection drives and educating peers about recycling through fun activities and demonstrations.
Result: The school reduces plastic waste by 40% in 3 months."""
        
        formatted = format_example_output(example, example_num=1)
        
        # Context and application should be readable (300+ chars allowed)
        assert "lacks practical knowledge about solutions" in formatted
        assert "She wants to make a difference" in formatted
        assert "educating peers about recycling" in formatted
        
        # Should not have mid-word truncation
        assert "soluti..." not in formatted
        assert "demonstr..." not in formatted


class TestEvaluationScoreDisplay:
    """Test evaluation score display improvements."""
    
    def test_scores_include_approach_titles(self):
        """Test that evaluation scores include the approach titles."""
        from qadi_simple import format_evaluation_scores
        
        hypotheses = [
            "Approach 1: Promoting Zero-Waste Lifestyle Through Community Workshops. This approach focuses on education.",
            "Approach 2: Implementing Smart Recycling Technology. This uses IoT sensors.",
            "Approach 3: Creating Circular Economy Networks. This builds partnerships."
        ]
        
        scores = [
            Mock(impact=0.8, feasibility=0.7, accessibility=0.9, 
                 sustainability=0.85, scalability=0.75, overall=0.82),
            Mock(impact=0.75, feasibility=0.8, accessibility=0.7,
                 sustainability=0.8, scalability=0.85, overall=0.78),
            Mock(impact=0.9, feasibility=0.6, accessibility=0.8,
                 sustainability=0.9, scalability=0.7, overall=0.8)
        ]
        
        output = format_evaluation_scores(hypotheses, scores)
        
        # Check titles are included
        assert "Promoting Zero-Waste Lifestyle" in output
        assert "Implementing Smart Recycling Technology" in output
        assert "Creating Circular Economy Networks" in output
        
        # Check overall score comes first
        lines = output.strip().split('\n')
        for i, line in enumerate(lines):
            if "Approach" in line and "Scores:" in line:
                # Next line should be Overall
                assert "Overall:" in lines[i+1]
                assert "Impact:" in lines[i+2]
    
    def test_metric_ordering_matches_high_score_format(self):
        """Test that metrics are ordered consistently."""
        from qadi_simple import format_evaluation_scores
        
        hypotheses = ["Test hypothesis"]
        scores = [Mock(impact=0.8, feasibility=0.7, accessibility=0.9,
                      sustainability=0.85, scalability=0.75, overall=0.82)]
        
        output = format_evaluation_scores(hypotheses, scores)
        
        # Extract metric order
        lines = [line.strip() for line in output.split('\n') if ' - ' in line]
        metric_order = [line.split(':')[0].strip().replace('- ', '') for line in lines]
        
        # Should match High Score format order
        expected_order = ['Overall', 'Impact', 'Feasibility', 'Accessibility', 
                         'Sustainability', 'Scalability']
        assert metric_order == expected_order


class TestEvolutionOutputCleanup:
    """Test that evolution output is cleaned of internal references."""
    
    def test_parent_references_removed_from_output(self):
        """Test that 'Parent 1' and 'Parent 2' are removed from ideas."""
        from qadi_simple import clean_evolution_output
        
        # Test various parent reference formats
        test_cases = [
            ("Building on Parent 1's approach, we combine sustainability with Parent 2's innovation.",
             "Building on the first approach, we combine sustainability with the second approach's innovation."),
            ("Parent 1 provides the foundation while Parent 2 adds scalability.",
             "The first approach provides the foundation while the second approach adds scalability."),
            ("Integrating Parent 1's community focus and Parent 2's technology.",
             "Integrating the first approach's community focus and the second approach's technology."),
            ("This combines elements from both approaches without mentioning parents.",
             "This combines elements from both approaches without mentioning parents.")
        ]
        
        for input_text, expected in test_cases:
            result = clean_evolution_output(input_text)
            assert result == expected
            assert "Parent 1" not in result
            assert "Parent 2" not in result
    
    @pytest.mark.asyncio
    async def test_semantic_crossover_prompts_avoid_parent_references(self):
        """Test that crossover prompts don't instruct to reference parents."""
        from mad_spark_alt.evolution.semantic_operators import SemanticCrossoverOperator
        
        # Mock LLM provider
        mock_llm = Mock()
        mock_llm.generate_completion = AsyncMock(return_value=Mock(
            content='{"offspring": [{"id": 1, "content": "Combined approach"}, {"id": 2, "content": "Integrated solution"}]}',
            cost=0.001
        ))
        
        operator = SemanticCrossoverOperator(mock_llm)
        
        parent1 = GeneratedIdea(
            content="Zero-waste lifestyle approach",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test prompt"
        )
        parent2 = GeneratedIdea(
            content="Technology-driven recycling",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test prompt"
        )
        
        # Capture the prompt sent to LLM
        await operator.crossover(parent1, parent2, "reduce plastic waste")
        
        # Check the prompt doesn't instruct to reference "Parent 1" or "Parent 2"
        call_args = mock_llm.generate_completion.call_args
        prompt = call_args[1]['prompt']
        
        # Should use generic terms instead
        assert "Parent 1" not in prompt or "first approach" in prompt
        assert "Parent 2" not in prompt or "second approach" in prompt


class TestIntegrationScenarios:
    """Integration tests for complete user workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_qadi_output_formatting(self):
        """Test complete QADI output with all formatting improvements."""
        from qadi_simple import run_qadi_analysis
        
        # This test requires API key - skip if not available
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            await run_qadi_analysis(
                "How can we reduce plastic waste in oceans?",
                verbose=True
            )
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Verify no truncation issues
        assert "..." not in output or output.count("...") < 5  # Some ellipsis ok, but not excessive
        
        # Verify evaluation scores have titles
        if "Evaluation Scores:" in output:
            scores_section = output.split("Evaluation Scores:")[1].split("##")[0]
            assert "Approach" in scores_section
            assert any(phrase in scores_section for phrase in [
                "Zero-Waste", "Technology", "Community", "Policy", "Education"
            ])
        
        # Verify examples are complete
        if "Real-World Examples" in output:
            examples_section = output.split("Real-World Examples")[1].split("##")[0]
            # Should have complete sentences
            assert examples_section.count('.') > 3  # Multiple complete sentences
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_evolution_output_formatting(self):
        """Test evolution output with all formatting improvements."""
        from qadi_simple import run_qadi_analysis
        
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            await run_qadi_analysis(
                "How can we reduce food waste?",
                evolve=True,
                generations=2,
                population=3
            )
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Verify no parent references in evolution output
        assert "Parent 1" not in output
        assert "Parent 2" not in output
        
        # Verify high score approaches are clean
        if "High Score Approaches" in output:
            high_scores = output.split("High Score Approaches")[1].split("##")[0]
            assert "Parent" not in high_scores
            
            # Should have proper score format
            assert "[Overall:" in high_scores
            assert "Impact:" in high_scores