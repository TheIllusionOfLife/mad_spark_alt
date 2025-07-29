"""
Integration tests for CLI evolution with real scenarios.

These tests verify the complete flow with various population/generation combinations.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
class TestCLIEvolutionIntegration:
    """Integration tests for evolution CLI arguments."""

    def setup_method(self):
        """Check if API key is available."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            pytest.skip("GOOGLE_API_KEY not set, skipping integration test")

    def test_evolution_display_with_high_population_request(self):
        """Test CLI output when requesting population higher than available ideas."""
        # Run the actual CLI command
        cmd = [
            sys.executable,
            "qadi_simple.py",
            "What is consciousness?",  # Simple question that generates few ideas
            "--evolve",
            "--generation", "3",
            "--population", "10",
            "--traditional",  # Use traditional to make test faster
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "GOOGLE_API_KEY": self.api_key}
        )
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        output = result.stdout
        
        # Check that the display shows requested values
        assert "3 generations, 10 population" in output, \
            "Output should show requested population=10, not the actual smaller value"
        
        # Check for clarification message
        assert "Using" in output and "ideas from available" in output, \
            "Should clarify when using fewer ideas than requested"

    def test_evolution_display_with_matching_population(self):
        """Test CLI output when population matches available ideas."""
        cmd = [
            sys.executable, 
            "qadi_simple.py",
            "How can we reduce plastic waste in oceans?",  # Question that generates more ideas
            "--evolve",
            "--generation", "2",
            "--population", "3",
            "--traditional",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "GOOGLE_API_KEY": self.api_key}
        )
        
        assert result.returncode == 0
        
        output = result.stdout
        
        # Check display
        assert "2 generations, 3 population" in output
        
        # Should not have clarification since we're likely to have 3+ ideas
        lines_with_using = [line for line in output.split('\n') if "Using" in line and "ideas from available" in line]
        assert len(lines_with_using) == 0 or "Using 3 ideas from available 3" not in output

    @pytest.mark.slow
    def test_semantic_evolution_display(self):
        """Test display with semantic operators (non-traditional mode)."""
        cmd = [
            sys.executable,
            "qadi_simple.py", 
            "How to build a sustainable business?",
            "--evolve",
            "--generation", "3",
            "--population", "5",
            # No --traditional flag, so it uses semantic operators
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "GOOGLE_API_KEY": self.api_key},
            timeout=180  # 3 minutes timeout
        )
        
        assert result.returncode == 0
        
        output = result.stdout
        
        # Check display shows requested values
        assert "3 generations, 5 population" in output
        
        # Check that semantic operators are being used
        assert "SEMANTIC (LLM-powered" in output or "semantic operators" in output.lower()