#!/usr/bin/env python3
"""
Run Mad Spark Alt without timeout constraints.
Pass any arguments to the underlying script.
"""

import subprocess
import sys
import os

# Set environment to prevent timeouts
os.environ['PYTHONUNBUFFERED'] = '1'

# Default to user_test.py if no script specified
script = 'examples/user_test.py'
args = sys.argv[1:]

# If first arg looks like a script path, use it
if args and args[0].endswith('.py'):
    script = args[0]
    args = args[1:]

# Run without timeout
try:
    cmd = [sys.executable, script] + args
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=False  # Real-time output
    )
    sys.exit(result.returncode)
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)