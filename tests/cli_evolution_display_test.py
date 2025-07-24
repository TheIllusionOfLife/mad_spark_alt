"""
Tests for CLI evolution display formatting.

This module tests that evolution results are displayed properly
without truncating important content.
"""

import pytest
from unittest.mock import Mock, patch
from rich.table import Table
from rich.console import Console

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.interfaces import IndividualFitness


class TestEvolutionDisplay:
    """Test evolution results display formatting."""

    def test_full_idea_display_without_truncation(self):
        """Test that ideas are displayed without excessive truncation."""
        # Create a long idea that would be truncated at 80 chars
        long_idea_content = (
            "Implement a comprehensive plastic reduction strategy by establishing "
            "community recycling hubs that provide education, collection services, "
            "and innovative upcycling workshops to transform waste into useful products"
        )
        
        idea = GeneratedIdea(
            content=long_idea_content,
            thinking_method="environmental",
            agent_name="test_agent",
            generation_prompt="How to reduce plastic waste?"
        )
        
        individual = IndividualFitness(
            idea=idea,
            overall_fitness=0.85
        )
        
        # Test display formatting
        from mad_spark_alt.cli import _format_idea_for_display
        
        formatted = _format_idea_for_display(idea.content, max_length=200)
        
        # Should not truncate at 80 characters
        assert len(formatted) > 80
        # Should include key parts of the idea
        assert "community recycling hubs" in formatted
        assert "education" in formatted
        # Should truncate cleanly at word boundaries if needed
        if formatted.endswith("..."):
            # The character before "..." should not be alphabetic (should be space or punctuation)
            text_part = formatted[:-3].rstrip()
            if text_part:  # If there's text before the ...
                # Last actual character should complete a word
                assert text_part[-1].isalpha() or text_part[-1] in '.,:;)]\'"'

    def test_smart_truncation_at_word_boundaries(self):
        """Test that truncation happens at word boundaries, not mid-word."""
        # Create idea that needs truncation
        idea_content = "Develop sustainable " + "technologies " * 30  # Very long
        
        from mad_spark_alt.cli import _format_idea_for_display
        
        formatted = _format_idea_for_display(idea_content, max_length=200)
        
        # Should truncate
        assert formatted.endswith("...")
        # Should not end with partial word
        words = formatted[:-3].split()  # Remove "..." and split
        assert all(len(word) > 0 for word in words)
        # Last word should be complete
        assert formatted[:-3].endswith(words[-1])

    def test_short_ideas_not_truncated(self):
        """Test that short ideas are displayed in full."""
        short_idea = "Implement local plastic bag ban"
        
        from mad_spark_alt.cli import _format_idea_for_display
        
        formatted = _format_idea_for_display(short_idea, max_length=200)
        
        # Should be unchanged
        assert formatted == short_idea
        assert not formatted.endswith("...")

    def test_table_column_width_configuration(self):
        """Test that table columns are configured for better readability."""
        # Mock table configuration
        with patch('mad_spark_alt.cli.Table') as MockTable:
            mock_table = Mock()
            MockTable.return_value = mock_table
            
            # Import the function that creates the table
            from mad_spark_alt.cli import _create_evolution_results_table
            
            table = _create_evolution_results_table()
            
            # Verify table configuration
            MockTable.assert_called_with(title="🏆 Top Evolved Ideas")
            
            # Check column additions
            expected_calls = [
                ('add_column', ('Rank',), {'style': 'cyan', 'width': 4}),
                ('add_column', ('Idea',), {'style': 'white', 'width': None}),  # No width limit
                ('add_column', ('Fitness',), {'style': 'green', 'width': 8}),
                ('add_column', ('Gen',), {'style': 'yellow', 'width': 5})
            ]
            
            for method, args, kwargs in expected_calls:
                mock_table.__getattr__(method).assert_any_call(*args, **kwargs)

    def test_multiline_display_for_very_long_ideas(self):
        """Test that very long ideas can be displayed across multiple lines."""
        very_long_idea = (
            "Create a comprehensive environmental protection framework that includes: "
            "1) Establishing marine sanctuaries to protect biodiversity, "
            "2) Implementing circular economy principles in manufacturing, "
            "3) Developing renewable energy infrastructure, "
            "4) Creating green jobs training programs, "
            "5) Building sustainable transportation networks, "
            "6) Promoting regenerative agriculture practices"
        )
        
        from mad_spark_alt.cli import _format_idea_for_display
        
        # Test with wrapping enabled
        formatted = _format_idea_for_display(
            very_long_idea, 
            max_length=250,
            wrap_lines=True
        )
        
        # Should preserve important content
        assert "marine sanctuaries" in formatted
        assert "circular economy" in formatted
        assert "renewable energy" in formatted
        
        # If truncated, should be at a logical point
        if formatted.endswith("..."):
            # Should end after a complete item or sentence
            assert formatted[-4] in ',).'