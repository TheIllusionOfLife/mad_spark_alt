#!/usr/bin/env python3
"""
Demonstrate difference between template and LLM agents.
"""

import asyncio
import os
from mad_spark_alt.core import SmartQADIOrchestrator

async def show_agent_comparison():
    """Show the difference between template and LLM agents."""
    
    print("🔍 LLM vs Template Agent Comparison")
    print("=" * 50)
    
    # Check API key availability
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"), 
        "Google": os.getenv("GOOGLE_API_KEY"),
    }
    
    print("🔐 API Key Status:")
    for provider, key in api_keys.items():
        status = "✅ Available" if key else "❌ Missing"
        agent_type = "LLM-Powered" if key else "Template Fallback"
        print(f"  {provider}: {status} → {agent_type}")
    
    print("\n" + "=" * 50)
    
    # Create smart orchestrator
    orchestrator = SmartQADIOrchestrator()
    
    # Simple problem for clear comparison
    problem = "How can we make online learning more engaging for students?"
    
    print(f"📚 Problem: {problem}")
    print("\n🚀 Running smart QADI cycle...")
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        cycle_config={"max_ideas_per_method": 2}
    )
    
    # Show results with agent type information
    print(f"\n⏱️  Execution Time: {result.execution_time:.3f}s")
    print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
    print(f"🤖 Agent Types Used: {result.agent_types}")
    print(f"💡 Total Ideas: {len(result.synthesized_ideas)}")
    
    if any("LLM" in agent_type for agent_type in result.agent_types.values()):
        print("\n🎯 LLM agents provide more contextual and sophisticated reasoning!")
    else:
        print("\n📝 Template agents provide fast, consistent baseline ideas.")
        print("💡 Add API keys to unlock AI-powered reasoning!")
    
    print("\n" + "=" * 50)
    
    # Show sample ideas from each phase
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            agent_type = result.agent_types.get(phase_name, "unknown")
            print(f"\n🔸 {phase_name.upper()} ({agent_type}):")
            
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"  {i}. {idea.content}")

if __name__ == "__main__":
    asyncio.run(show_agent_comparison())