#!/usr/bin/env python3
"""
Demonstrate parallel idea generation across multiple thinking methods.
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator, ThinkingMethod

async def parallel_generation_demo():
    """Demonstrate parallel execution of thinking methods."""
    
    print("⚡ Parallel Idea Generation Demo")
    print("=" * 50)
    
    orchestrator = SmartQADIOrchestrator()
    
    # Select specific thinking methods to run in parallel
    thinking_methods = [
        ThinkingMethod.QUESTIONING,
        ThinkingMethod.ABDUCTION,
        ThinkingMethod.INDUCTION
    ]
    
    problem = "How can we reduce plastic waste in packaging?"
    context = "Focus on innovative materials and system changes."
    
    print(f"🎯 Problem: {problem}")
    print(f"🧠 Methods: {[m.value for m in thinking_methods]}")
    print("\n🚀 Running parallel generation...\n")
    
    # Run methods in parallel instead of sequential QADI cycle
    results = await orchestrator.run_parallel_generation(
        problem_statement=problem,
        thinking_methods=thinking_methods,
        context=context,
        config={"max_ideas_per_method": 3}
    )
    
    print("📊 Parallel Execution Results:")
    print("-" * 30)
    
    total_ideas = 0
    for method, (result, agent_type) in results.items():
        ideas_count = len(result.generated_ideas) if result.generated_ideas else 0
        total_ideas += ideas_count
        
        print(f"\n🔸 {method.value.upper()} ({agent_type}):")
        print(f"   Ideas generated: {ideas_count}")
        
        if result.generated_ideas:
            for i, idea in enumerate(result.generated_ideas[:2], 1):
                print(f"   {i}. {idea.content[:100]}...")
        
        if result.error_message:
            print(f"   ❌ Error: {result.error_message}")
    
    print(f"\n✅ Total ideas generated: {total_ideas}")
    print("⚡ All methods executed simultaneously for maximum efficiency!")

if __name__ == "__main__":
    asyncio.run(parallel_generation_demo())