"""Integration tests for qadi_simple.py evolution with real scenarios."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestQadiSimpleEvolutionIntegration:
    """Integration tests for qadi_simple.py --evolve improvements."""
    
    def run_qadi_simple(self, args: list[str], timeout: int = 120) -> tuple[str, str, int]:
        """Run qadi_simple.py with given arguments."""
        cmd = ["uv", "run", "python", "qadi_simple.py"] + args
        
        # Ensure we have API key
        env = os.environ.copy()
        if "GOOGLE_API_KEY" not in env:
            pytest.skip("GOOGLE_API_KEY not set")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout} seconds", 1
    
    def test_evolution_with_population_2(self):
        """Test evolution with population of 2."""
        stdout, stderr, code = self.run_qadi_simple([
            "How can we improve team productivity?",
            "--evolve",
            "--population", "2",
            "--generations", "2"
        ])
        
        # Should complete successfully
        assert code == 0, f"Command failed: {stderr}"
        
        # Should show it's using 2 ideas (once implemented)
        # Currently it would show "Using 2 ideas from available 3"
        # After fix, it should show 2 initial hypotheses generated
        assert "Initial Solutions" in stdout
        
        # Should not timeout
        assert "timed out" not in stderr
    
    def test_evolution_with_population_5(self):
        """Test evolution with population of 5."""
        stdout, stderr, code = self.run_qadi_simple([
            "Design a sustainable city transportation system",
            "--evolve", 
            "--population", "5",
            "--generations", "2"
        ])
        
        assert code == 0, f"Command failed: {stderr}"
        
        # After implementation, should generate 5 initial hypotheses
        assert "Initial Solutions" in stdout
        
        # Evolution should complete
        assert "Evolution completed" in stdout or "Evolution Results" in stdout
    
    def test_evolution_with_population_10(self):
        """Test evolution with maximum population."""
        stdout, stderr, code = self.run_qadi_simple([
            "Create innovative educational technology",
            "--evolve",
            "--population", "10", 
            "--generations", "2"
        ])
        
        assert code == 0, f"Command failed: {stderr}"
        
        # Should handle large population
        assert "Evolution" in stdout
        
        # Should not show convergence warning with diverse population
        # (though this depends on the actual evolution results)
    
    def test_evolution_output_quality(self):
        """Test that evolution output is complete and well-formatted."""
        stdout, stderr, code = self.run_qadi_simple([
            "Develop strategies for remote work challenges",
            "--evolve",
            "--population", "5",
            "--generations", "3"
        ])
        
        assert code == 0, f"Command failed: {stderr}"
        
        # Check for quality issues
        assert "TODO" not in stdout, "Output contains placeholder content"
        assert "[truncated]" not in stdout, "Output is truncated"
        assert stdout.count("Enhanced Approach") >= 1, "No enhanced approaches shown"
        
        # Verify all sections present
        assert "Initial Solutions" in stdout
        assert "Analysis:" in stdout
        assert "Your Recommended Path" in stdout
        assert "Evolution Results" in stdout
        
        # Check metrics are shown
        assert "Total time:" in stdout
        assert "Total cost:" in stdout
    
    def test_evolution_without_evolve_flag_message(self):
        """Test helpful message when using evolution args without --evolve."""
        stdout, stderr, code = self.run_qadi_simple([
            "Test question",
            "--population", "5"  # Without --evolve
        ])
        
        # Should warn about unused argument
        assert "--evolve" in stdout or "--evolve" in stderr
    
    @pytest.mark.integration
    def test_evolution_with_real_llm_diverse_questions(self):
        """Test evolution with various question types using real LLM."""
        test_cases = [
            ("Solve climate change", 3),
            ("Improve software development productivity", 5),
            ("Design the future of education", 7),
        ]
        
        for question, population in test_cases:
            stdout, stderr, code = self.run_qadi_simple([
                question,
                "--evolve",
                "--population", str(population),
                "--generations", "2"
            ])
            
            assert code == 0, f"Failed for '{question}': {stderr}"
            
            # Verify population handling
            if population <= 3:
                # Should still work with small populations
                assert "Evolution completed" in stdout or "Evolution Results" in stdout
            else:
                # Should generate more hypotheses
                # After implementation, should see more diverse results
                pass