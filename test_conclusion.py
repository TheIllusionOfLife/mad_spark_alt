#!/usr/bin/env python3
"""Test conclusion synthesis feature."""

import asyncio
import os
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

from mad_spark_alt.core import FastQADIOrchestrator

async def test_conclusion():
    """Test the conclusion synthesis feature."""
    orchestrator = FastQADIOrchestrator(enable_parallel=True)
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement="What are the key benefits of renewable energy?",
        context="Analyze environmental, economic, and social benefits",
        cycle_config={
            "max_ideas_per_method": 2,
            "require_reasoning": True,
        }
    )
    
    print(f"✅ Completed in {result.execution_time:.2f}s")
    print(f"💰 Cost: ${result.llm_cost:.4f}")
    print(f"💡 Ideas: {len(result.synthesized_ideas)}")
    
    if hasattr(result, 'conclusion') and result.conclusion:
        print("\n" + "="*60)
        print("📋 CONCLUSION GENERATED!")
        print("="*60)
        print(f"\nSummary: {result.conclusion.summary}")
        print(f"\nKey Insights ({len(result.conclusion.key_insights)}):")
        for insight in result.conclusion.key_insights:
            print(f"  • {insight}")
        print(f"\nRecommendations ({len(result.conclusion.actionable_recommendations)}):")
        for rec in result.conclusion.actionable_recommendations:
            print(f"  • {rec}")
        print(f"\nNext Steps ({len(result.conclusion.next_steps)}):")
        for step in result.conclusion.next_steps:
            print(f"  • {step}")
    else:
        print("\n❌ No conclusion generated")

if __name__ == "__main__":
    asyncio.run(test_conclusion())