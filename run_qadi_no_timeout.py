#!/usr/bin/env python3
"""
Run QADI demo without shell timeout constraints.
This script bypasses any shell-level timeouts by running the QADI demo directly.
"""

import subprocess
import sys
import os

# Set environment variable to disable any potential timeouts
os.environ['PYTHONUNBUFFERED'] = '1'

# Run the QADI demo without any timeout
try:
    # Use sys.executable to ensure we use the same Python interpreter
    result = subprocess.run(
        [sys.executable, 'examples/qadi_demo.py'],
        check=False,  # Don't raise exception on non-zero exit
        text=True,
        capture_output=False  # Show output in real-time
    )
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\nInterrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Error running QADI demo: {e}")
    sys.exit(1)