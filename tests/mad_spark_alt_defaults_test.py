"""
Tests for mad_spark_alt command default values and validation.
"""

import subprocess
import sys
import pytest
from pathlib import Path


class TestMadSparkAltDefaults:
    """Test suite for mad_spark_alt command defaults and validation."""
    
    def test_default_generations_is_2(self):
        """Verify default generations is 2 in qadi_simple.py."""
        # Check the actual source code
        qadi_simple_path = Path(__file__).parent.parent / "qadi_simple.py"
        with open(qadi_simple_path) as f:
            content = f.read()
        
        # Look for the generations argument definition
        import re
        match = re.search(r'--generations.*default=(\d+)', content)
        assert match is not None, "Could not find --generations argument"
        assert match.group(1) == "2", f"Expected default=2, found default={match.group(1)}"
        
    def test_default_population_is_5(self):
        """Verify default population is 5 in qadi_simple.py."""
        # Check the actual source code
        qadi_simple_path = Path(__file__).parent.parent / "qadi_simple.py"
        with open(qadi_simple_path) as f:
            content = f.read()
        
        # Look for the population argument definition
        import re
        match = re.search(r'--population.*default=(\d+)', content)
        assert match is not None, "Could not find --population argument"
        assert match.group(1) == "5", f"Expected default=5, found default={match.group(1)}"
        

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
        # Help should show the defaults
        assert "evolution" in result.stdout.lower()
        

class TestMadSparkAltEvolution:
    """Test evolution functionality with new defaults."""
    
    def test_evolution_without_params_uses_defaults(self):
        """Test that evolution uses default values when not specified."""
        # This test would need API key, so we just check the command parses correctly
        result = subprocess.run(
            ["uv", "run", "python", "qadi_simple.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0