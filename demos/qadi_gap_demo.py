#!/usr/bin/env python3
"""
Demonstrate the gap between QADI output and user expectations.
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def show_qadi_gap():
    """Show how QADI doesn't directly answer user questions."""
    
    user_question = "What are 5 specific ways to reduce plastic waste in oceans?"
    
    print("🔍 QADI Gap Analysis")
    print("=" * 70)
    print(f"❓ User Question: {user_question}")
    print("🎯 User Expects: 5 concrete, actionable solutions")
    print("=" * 70)
    
    orchestrator = SmartQADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=user_question,
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print("\n📊 ACTUAL QADI OUTPUT:")
    print("-" * 50)
    
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"\n🔸 {phase_name.upper()}:")
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"  {i}. {idea.content}")
    
    print("\n" + "=" * 70)
    print("❌ THE PROBLEM:")
    print("  • User asked for '5 specific ways'")
    print("  • QADI gave abstract reasoning patterns")
    print("  • No direct, actionable answers provided")
    print("  • Induction phase talks ABOUT solutions, not actual solutions")
    
    print("\n✅ WHAT'S MISSING:")
    print("  • Final synthesis step that converts QADI insights → concrete answers")
    print("  • Answer extraction from the accumulated knowledge")
    print("  • Direct response to the original question format")
    
    print("\n🔧 COMPARE WITH generate_ideas.py approach:")
    print("  • Translates QADI into practical phases")
    print("  • Each phase produces usable outputs")
    print("  • 'Implementation Steps' directly answers 'how to' questions")

if __name__ == "__main__":
    asyncio.run(show_qadi_gap())