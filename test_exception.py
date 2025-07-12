#!/usr/bin/env python3
"""Test exception handling patterns in Python 3.13"""

import json

# Test 1: Compound exception handling
try:
    print("Test 1: Compound exception handling")
    try:
        json.loads("{invalid}")
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Caught: {type(e).__name__}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: Simple exception handling  
try:
    print("\nTest 2: Simple exception handling")
    try:
        json.loads("{invalid}")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Check Python version
import sys
print(f"\nPython version: {sys.version}")

# Test 4: Check if it's the tuple that's the issue
try:
    print("\nTest 4: Testing tuple of exceptions")
    try:
        raise ValueError("test")
    except (ValueError, KeyError) as e:
        print(f"  Caught specific exceptions: {type(e).__name__}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 5: Non-exception in tuple
print("\nTest 5: Testing non-exception in tuple")
try:
    try:
        raise ValueError("test")
    except (ValueError, str) as e:  # This should fail
        print(f"  Caught: {type(e).__name__}")
except Exception as e:
    print(f"  ERROR: {e}")