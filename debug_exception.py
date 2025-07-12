#!/usr/bin/env python3
"""Debug the exception handling issue."""

import sys
import warnings
import traceback

# Hook into Python's exception handling
original_excepthook = sys.excepthook

def custom_excepthook(exc_type, exc_value, exc_traceback):
    if "catching classes that do not inherit from BaseException" in str(exc_value):
        print("\n🔍 FOUND THE ISSUE!")
        print(f"Exception type: {exc_type}")
        print(f"Exception value: {exc_value}")
        print("Full traceback:")
        traceback.print_tb(exc_traceback)
    original_excepthook(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_excepthook

# Also catch warnings
def warning_handler(message, category, filename, lineno, file=None, line=None):
    if "BaseException" in str(message):
        print(f"\n⚠️  Warning at {filename}:{lineno}")
        print(f"Message: {message}")

warnings.showwarning = warning_handler

# Now run the actual test
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print("Starting debug test...")

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def test():
    orchestrator = SmartQADIOrchestrator()
    
    try:
        await orchestrator.ensure_agents_ready()
        print("✅ Agents ready")
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement="Test",
            cycle_config={"max_ideas_per_method": 1}
        )
        print("✅ QADI complete")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

asyncio.run(test())