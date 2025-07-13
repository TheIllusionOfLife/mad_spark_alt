#!/usr/bin/env python3
"""
Quick idea generation demo for "how to live longer"
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def generate_longevity_ideas():
    """Generate ideas for living longer using QADI methodology."""
    
    print("🧬 Longevity Ideas Generator")
    print("Using QADI Methodology")
    print("=" * 50)
    
    orchestrator = SmartQADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement="How to live longer",
        context="""
        Consider multiple approaches:
        - Lifestyle and behavioral changes
        - Medical and technological interventions
        - Social and environmental factors
        - Preventive healthcare strategies
        - Mental and emotional well-being
        """,
        cycle_config={"max_ideas_per_method": 3}
    )
    
    print(f"⏱️  Generated in {result.execution_time:.2f} seconds")
    print(f"💡 Total ideas: {len(result.synthesized_ideas)}")
    print(f"🤖 Agent types: {result.agent_types}")
    print(f"💰 Cost: ${result.llm_cost:.4f}")
    
    print("\n" + "=" * 50)
    print("💡 IDEAS FOR LIVING LONGER")
    print("=" * 50)
    
    phase_icons = {
        'questioning': '❓',
        'abduction': '💡', 
        'deduction': '🔍',
        'induction': '🔗'
    }
    
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            icon = phase_icons.get(phase_name, '🔸')
            print(f"\n{icon} {phase_name.upper()} PHASE:")
            print("-" * 30)
            
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"{i}. {idea.content}")
                if idea.reasoning:
                    print(f"   💭 {idea.reasoning}")
                print()
    
    print("🎯 Use these ideas as starting points for developing")
    print("   comprehensive longevity strategies!")

if __name__ == "__main__":
    asyncio.run(generate_longevity_ideas())