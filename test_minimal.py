#!/usr/bin/env python3
"""Minimal test to find the exception handling issue."""

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

# Test imports step by step
print("1. Testing basic imports...")
try:
    import asyncio
    from mad_spark_alt.core import SmartQADIOrchestrator
    print("✅ Basic imports OK")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

print("\n2. Testing orchestrator creation...")
try:
    orchestrator = SmartQADIOrchestrator()
    print("✅ Orchestrator created")
except Exception as e:
    print(f"❌ Creation error: {e}")
    exit(1)

print("\n3. Testing agent setup...")
async def test_setup():
    try:
        setup_status = await orchestrator.ensure_agents_ready()
        print("✅ Agents setup complete")
        return True
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

print("\n4. Testing simple QADI cycle...")
async def test_qadi():
    try:
        # Very simple test
        result = await orchestrator.run_qadi_cycle(
            problem_statement="Test",
            cycle_config={"max_ideas_per_method": 1}
        )
        print("✅ QADI cycle complete")
        return True
    except Exception as e:
        print(f"❌ QADI error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    if await test_setup():
        await test_qadi()

if __name__ == "__main__":
    asyncio.run(main())