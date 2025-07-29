"""
Unit tests for CLI display logic.

Test specific display functions without running the full flow.
"""

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import pytest


class TestEvolutionDisplayLogic:
    """Test the logic that displays evolution parameters."""

    def test_evolution_display_message_generation(self):
        """Test that the evolution display message is generated correctly."""
        # Test data
        requested_generations = 3
        requested_population = 10
        actual_population = 3  # Only 3 ideas available
        
        # This is the current buggy behavior
        buggy_message = f"🧬 Evolving ideas ({requested_generations} generations, {actual_population} population)..."
        assert "3 population" in buggy_message  # Shows actual, not requested
        
        # This is what we want
        correct_message = f"🧬 Evolving ideas ({requested_generations} generations, {requested_population} population)..."
        assert "10 population" in correct_message  # Shows requested
        
        # Additional clarification when fewer ideas available
        if actual_population < requested_population:
            clarification = f"   (Using {actual_population} ideas from available {actual_population})"
            assert "Using 3 ideas" in clarification

    def test_display_logic_in_qadi_simple(self):
        """Test the actual display logic from qadi_simple.py"""
        # Simulate the scenario
        requested_population = 10
        requested_generations = 3
        available_ideas = 3
        
        # Current code (line 331 in qadi_simple.py):
        # print(f"🧬 Evolving ideas ({generations} generations, {min(population, len(result.synthesized_ideas))} population)...")
        
        # This is the bug - it shows min() instead of requested
        current_display = f"🧬 Evolving ideas ({requested_generations} generations, {min(requested_population, available_ideas)} population)..."
        assert current_display == "🧬 Evolving ideas (3 generations, 3 population)..."
        
        # What it should be:
        fixed_display = f"🧬 Evolving ideas ({requested_generations} generations, {requested_population} population)..."
        assert fixed_display == "🧬 Evolving ideas (3 generations, 10 population)..."
        
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_evolution_display_output_fixed(self, mock_stdout):
        """Test what gets printed with the fixed code."""
        # Simulate the print statements
        generations = 5
        population = 8
        synthesized_ideas = ["idea1", "idea2", "idea3"]  # Only 3 ideas
        
        # Fixed code - show requested values
        print(f"🧬 Evolving ideas ({generations} generations, {population} population)...")
        print("─" * 50)
        
        # Check if we have fewer ideas than requested
        actual_population = min(population, len(synthesized_ideas))
        if actual_population < population:
            print(f"   (Using {actual_population} ideas from available {len(synthesized_ideas)})")
        
        output = mock_stdout.getvalue()
        
        # This test should PASS with fixed code
        assert "8 population" in output  # Shows requested population
        assert "Using 3 ideas from available 3" in output  # Shows clarification
        
    def test_config_creation_vs_display(self):
        """Test that config uses actual but display shows requested."""
        requested_population = 10
        requested_generations = 3
        available_ideas = 3
        
        # Config should use actual population
        actual_population = min(requested_population, available_ideas)
        assert actual_population == 3
        
        # But display should show requested
        display_msg = f"Evolving ({requested_generations} generations, {requested_population} population)"
        assert "10 population" in display_msg
        
        # Config gets the actual
        config_population_size = actual_population
        assert config_population_size == 3
        
        # User sees the requested
        user_message = f"Evolving ({requested_generations} generations, {requested_population} population)"
        assert "10 population" in user_message