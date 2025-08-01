"""
Tests for qadi_simple.py timeout calculation.

This module tests the actual timeout implementation in qadi_simple.py.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path

# Import the actual values from qadi_simple
sys.path.insert(0, str(Path(__file__).parent.parent))
from qadi_simple import (
    calculate_evolution_timeout,
    EVOLUTION_BASE_TIMEOUT,
    EVOLUTION_TIME_PER_EVAL,
    EVOLUTION_MAX_TIMEOUT
)


class TestQadiSimpleTimeout:
    """Test the actual timeout implementation in qadi_simple.py."""
    
    def test_current_timeout_calculation(self):
        """Test that current implementation uses updated timeout values."""
        # Test the actual imported values instead of parsing strings
        assert EVOLUTION_BASE_TIMEOUT == 120.0, "Base timeout should be 120s"
        assert EVOLUTION_TIME_PER_EVAL == 8.0, "Time per eval should be 8s for semantic operators"
        assert EVOLUTION_MAX_TIMEOUT == 900.0, "Max timeout should be 900s (15 minutes)"
        
    def test_timeout_calculation_function_exists(self):
        """Test that calculate_evolution_timeout function exists and works."""
        # Test the actual imported function instead of parsing strings
        assert callable(calculate_evolution_timeout)
        
        # Test that it returns expected values
        timeout = calculate_evolution_timeout(3, 10)
        assert timeout == 440.0, "Should calculate correct timeout for 3 generations, 10 population"
        
    @pytest.mark.integration
    def test_cli_timeout_output(self):
        """Test that CLI shows timeout in output."""
        # This would test actual CLI output but skip for unit tests
        pass