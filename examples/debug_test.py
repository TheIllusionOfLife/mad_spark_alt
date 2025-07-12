#!/usr/bin/env python3
"""Debug test to see what's happening with the system."""

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

async def test_with_timeout():
    """Test with proper timeout handling."""
    print("🔍 Testing QADI system...")
    print(f"✅ Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
    
    orchestrator = SmartQADIOrchestrator()
    
    # Test with a simple prompt
    problem = "Suggest 5 creative weekend activities with a budget of 100 dollars"
    print(f"\n🎯 Problem: {problem}")
    
    try:
        # Ensure agents are ready first
        print("\n🔧 Setting up agents...")
        setup_status = await orchestrator.ensure_agents_ready()
        
        for method, status in setup_status.items():
            print(f"  {method}: {status}")
        
        print("\n🚀 Running QADI cycle (this may take 30-60 seconds)...")
        
        # Run with a specific timeout
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle(
                problem_statement=problem,
                context="Focus on fun, memorable experiences",
                cycle_config={
                    "max_ideas_per_method": 2,
                    "require_reasoning": False  # Faster without reasoning
                }
            ),
            timeout=90  # 90 second timeout
        )
        
        print(f"\n✅ Success! Completed in {result.execution_time:.2f}s")
        print(f"💰 Cost: ${result.llm_cost:.4f}")
        
        # Show results
        for phase_name, phase_result in result.phases.items():
            if phase_result.generated_ideas:
                print(f"\n{phase_name.title()}:")
                for idea in phase_result.generated_ideas[:2]:
                    print(f"  • {idea.content[:100]}...")
                    
    except asyncio.TimeoutError:
        print("\n⏱️  Timeout! Possible causes:")
        print("  • Slow API response")
        print("  • Rate limiting")
        print("  • Model availability issues")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_with_timeout())