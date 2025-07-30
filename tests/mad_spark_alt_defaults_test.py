"""
Tests for mad_spark_alt command default values and validation.
"""

import subprocess
import pytest


class TestMadSparkAltDefaults:
    """Test suite for mad_spark_alt command defaults and validation."""
    
    def test_default_generations_is_2(self):
        """Verify default generations is 2 in help text."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        help_text = result.stdout
        # Check that help text shows the default value
        assert "--generations" in help_text
        assert "default: 2" in help_text, "Default for --generations should be 2 in help text"
        
    def test_default_population_is_5(self):
        """Verify default population is 5 in help text."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        help_text = result.stdout
        # Check that help text shows the default value
        assert "--population" in help_text
        assert "default: 5" in help_text, "Default for --population should be 5 in help text"
        

class TestMadSparkAltValidation:
    """Test parameter validation for mad_spark_alt."""
    
    def test_generations_validation_min(self):
        """Test generations minimum validation (2)."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "test", "--evolve", "--generations", "1"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Generations must be between 2 and 5" in result.stdout
        
    def test_generations_validation_max(self):
        """Test generations maximum validation (5)."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "test", "--evolve", "--generations", "6"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Generations must be between 2 and 5" in result.stdout
        
    def test_population_validation_min(self):
        """Test population minimum validation (2)."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "test", "--evolve", "--population", "1"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Population size must be between 2 and 10" in result.stdout
        
    def test_population_validation_max(self):
        """Test population maximum validation (10)."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "test", "--evolve", "--population", "11"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "Population size must be between 2 and 10" in result.stdout
        

class TestMadSparkAltHelp:
    """Test help text shows correct defaults."""
    
    def test_help_shows_correct_defaults(self):
        """Verify help text mentions correct default values."""
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        # Check specific defaults are shown
        assert "default: 2" in result.stdout, "Help should show default generations=2"
        assert "default: 5" in result.stdout, "Help should show default population=5"
        # Check evolution is mentioned
        assert "--evolve" in result.stdout or "-e" in result.stdout
        

class TestMadSparkAltEvolution:
    """Test evolution functionality with new defaults."""
    
    @pytest.mark.integration
    def test_evolution_without_params_uses_defaults(self):
        """Test that evolution uses default values when not specified."""
        # This test requires API key
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not available")
        
        # Test with minimal input
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "test", "--evolve"],
            capture_output=True,
            text=True,
            timeout=30  # Short timeout
        )
        # Check that it mentions the default values
        assert "2 generations" in result.stdout or "generations, 2" in result.stdout
        assert "5 population" in result.stdout or "population 5" in result.stdout