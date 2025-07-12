#!/usr/bin/env python3
"""
Direct test of Google LLM integration to verify it's working.
"""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

from mad_spark_alt.core import SmartQADIOrchestrator

async def test_google_llm():
    """Test Google LLM directly."""
    print("🔍 Testing Google LLM Integration...")
    print(f"✅ Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    
    orchestrator = SmartQADIOrchestrator()
    
    # Ensure agents are ready
    setup_status = await orchestrator.ensure_agents_ready()
    print("\n📊 Agent Setup Status:")
    for method, status in setup_status.items():
        print(f"  {method}: {status}")
    
    # Run a simple test
    print("\n🚀 Running QADI cycle...")
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we reduce plastic waste in oceans?",
        context="Focus on practical, implementable solutions",
        cycle_config={
            "max_ideas_per_method": 2,
            "require_reasoning": True
        }
    )
    
    print(f"\n✅ Completed in {result.execution_time:.2f}s")
    print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
    
    # Show agent types used
    print("\n🤖 Agent Types Used:")
    for phase, agent_type in result.agent_types.items():
        print(f"  {phase}: {agent_type}")
    
    # Show sample ideas
    print("\n💡 Sample Ideas Generated:")
    for phase_name, phase_result in result.phases.items():
        if phase_result.generated_ideas:
            idea = phase_result.generated_ideas[0]
            print(f"\n{phase_name.title()}:")
            print(f"  {idea.content}")
            if hasattr(idea, 'reasoning') and idea.reasoning:
                print(f"  Reasoning: {idea.reasoning[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_google_llm())