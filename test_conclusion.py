#!/usr/bin/env python3
"""
Test if conclusion synthesizer produces better final answers.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core import SmartQADIOrchestrator

async def test_conclusion_synthesis():
    """Test the conclusion synthesis feature."""
    
    user_question = "What are 3 practical ways to reduce food waste at home?"
    
    print("🧪 Testing Conclusion Synthesis")
    print("=" * 60)
    print(f"❓ Question: {user_question}")
    print("=" * 60)
    
    orchestrator = SmartQADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=user_question,
        context="Focus on actionable, everyday solutions for individuals",
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print(f"\n📊 QADI Results:")
    print(f"  Total ideas: {len(result.synthesized_ideas)}")
    print(f"  LLM cost: ${result.llm_cost:.4f}")
    
    # Check if conclusion was generated
    if result.conclusion:
        print(f"\n✅ CONCLUSION SYNTHESIZED:")
        print("-" * 40)
        print(f"Summary: {result.conclusion.summary}")
        
        if result.conclusion.actionable_recommendations:
            print(f"\n🎯 Actionable Recommendations:")
            for i, rec in enumerate(result.conclusion.actionable_recommendations, 1):
                print(f"  {i}. {rec}")
        
        if result.conclusion.key_insights:
            print(f"\n💡 Key Insights:")
            for i, insight in enumerate(result.conclusion.key_insights, 1):
                print(f"  {i}. {insight}")
                
        if result.conclusion.next_steps:
            print(f"\n📋 Next Steps:")
            for i, step in enumerate(result.conclusion.next_steps, 1):
                print(f"  {i}. {step}")
    else:
        print(f"\n❌ NO CONCLUSION GENERATED")
        print("  Possible reasons:")
        print("  • LLM required but no API keys available")
        print("  • Template agents don't trigger conclusion synthesis")
        print("  • Error in conclusion generation process")
    
    print(f"\n🔍 Raw QADI Output:")
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"\n{phase_name}: {len(phase_result.generated_ideas)} ideas")

if __name__ == "__main__":
    asyncio.run(test_conclusion_synthesis())