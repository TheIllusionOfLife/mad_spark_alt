#!/usr/bin/env python3
"""
Simple QADI cycle demonstration script.
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def run_qadi_demo():
    """Demonstrate QADI cycle with a practical problem."""
    
    print("🔄 Mad Spark Alt - QADI Cycle Demo")
    print("=" * 50)
    
    # Create orchestrator (will use template agents without API keys)
    orchestrator = SmartQADIOrchestrator()
    
    # Run QADI cycle
    problem = "How can we reduce food waste in restaurants?"
    context = "Focus on practical, cost-effective solutions that can be implemented quickly."
    
    print(f"🎯 Problem: {problem}")
    print(f"📝 Context: {context}")
    print("\n🚀 Running QADI cycle...\n")
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context=context
    )
    
    # Display results
    print(f"✅ Completed in {result.execution_time:.2f}s")
    print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
    print(f"🤖 Agent Types: {result.agent_types}")
    print(f"💡 Total Ideas Generated: {len(result.synthesized_ideas)}")
    print("\n" + "=" * 50)
    
    # Show ideas from each phase
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"\n🔸 {phase_name.upper()} PHASE:")
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"  {i}. {idea.content}")
    
    print(f"\n🎉 QADI cycle completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_qadi_demo())