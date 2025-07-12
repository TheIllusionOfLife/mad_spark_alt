#!/usr/bin/env python3
"""
Comprehensive test to debug the Google LLM integration.
"""

import asyncio
import os
from pathlib import Path
import traceback

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print(f"✅ Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")

from mad_spark_alt.core import SmartQADIOrchestrator, agent_registry

async def test_system():
    """Test the system comprehensively."""
    
    # Test 1: Check agent registration
    print("\n📊 Registered Agents:")
    all_agents = agent_registry._agents
    for method, agents in all_agents.items():
        print(f"  {method}: {[type(agent).__name__ for agent in agents]}")
    
    # Test 2: Try a simple QADI cycle
    print("\n🚀 Testing QADI Cycle...")
    orchestrator = SmartQADIOrchestrator()
    
    try:
        result = await orchestrator.run_qadi_cycle(
            problem_statement="How to save water at home?",
            context="Simple practical solutions",
            cycle_config={
                "max_ideas_per_method": 1,
                "require_reasoning": False
            }
        )
        
        print(f"✅ Success! Execution time: {result.execution_time:.2f}s")
        print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
        
        # Show which agents were used
        print("\n🤖 Agents Used:")
        for phase, agent_type in result.agent_types.items():
            success = "✅" if result.phases[phase].generated_ideas else "❌"
            print(f"  {success} {phase}: {agent_type}")
            
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_system())