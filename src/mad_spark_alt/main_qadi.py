"""
Main entry point for Mad Spark Alt QADI analysis.

This module provides the main command-line interface for the simplified QADI analysis
with optional genetic evolution.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from qadi_simple.py
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qadi_simple import main

if __name__ == "__main__":
    main()
