#!/usr/bin/env python3
"""Trace imports to find where the issue occurs."""

import sys
import os
from pathlib import Path

# Load .env file first
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Set up import tracing
class ImportTracer:
    def __init__(self):
        self.depth = 0
        self.original_import = __builtins__.__import__
        
    def trace_import(self, name, *args, **kwargs):
        indent = "  " * self.depth
        print(f"{indent}→ Importing: {name}")
        self.depth += 1
        try:
            result = self.original_import(name, *args, **kwargs)
            self.depth -= 1
            return result
        except Exception as e:
            self.depth -= 1
            print(f"{indent}❌ Error importing {name}: {e}")
            raise

# Install the tracer
tracer = ImportTracer()
__builtins__.__import__ = tracer.trace_import

try:
    print("Starting import trace...")
    from mad_spark_alt.core import SmartQADIOrchestrator
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()