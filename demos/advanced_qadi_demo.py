#!/usr/bin/env python3
"""
Advanced QADI cycle demonstration with custom configuration.
"""

import asyncio
from mad_spark_alt.core import QADIOrchestrator
from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent

async def run_advanced_qadi():
    """Advanced QADI cycle with custom configuration."""
    
    print("🚀 Advanced QADI Cycle Demo")
    print("=" * 50)
    
    # Create agents manually for demonstration
    agents = [
        QuestioningAgent(),
        AbductionAgent(), 
        DeductionAgent(),
        InductionAgent()
    ]
    
    # Create orchestrator with specific agents
    orchestrator = QADIOrchestrator(agents=agents)
    
    # Custom configuration
    cycle_config = {
        "max_ideas_per_method": 2,  # Fewer ideas for cleaner output
        "require_reasoning": True,
        "questioning": {"focus": "root_causes"},
        "abduction": {"creativity_level": "high"},
        "deduction": {"logical_rigor": "strict"},
        "induction": {"pattern_depth": "deep"}
    }
    
    # Run with urban sustainability problem
    problem = "How can we design cities that are both highly livable and environmentally sustainable?"
    context = """
    Context: Consider cities as complex adaptive systems. Current challenges include:
    - Growing urban populations (68% by 2050)
    - Climate change impacts
    - Infrastructure aging
    - Social equity concerns
    - Economic pressures
    
    Think beyond traditional solutions to find systemic approaches.
    """
    
    print(f"🏙️  Problem: {problem}")
    print(f"📊 Agents: {len(orchestrator.agents)} thinking methods")
    print("\n🔄 Running enhanced QADI cycle...\n")
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context=context,
        cycle_config=cycle_config
    )
    
    # Enhanced results display
    print(f"⏱️  Execution Time: {result.execution_time:.3f} seconds")
    print(f"🧠 Ideas Generated: {len(result.synthesized_ideas)} total")
    print("\n" + "=" * 60)
    
    # Display each phase with reasoning
    phase_names = {
        'questioning': '❓ QUESTIONING',
        'abduction': '💡 ABDUCTION', 
        'deduction': '🔍 DEDUCTION',
        'induction': '🔗 INDUCTION'
    }
    
    for phase_key, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"\n{phase_names.get(phase_key, phase_key.upper())} PHASE:")
            print("-" * 40)
            
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"{i}. {idea.content}")
                if idea.reasoning:
                    print(f"   💭 Reasoning: {idea.reasoning}")
                print()
    
    print("🎯 SYNTHESIS COMPLETE!")
    print(f"The QADI cycle generated {len(result.synthesized_ideas)} interconnected ideas")
    print("for addressing sustainable urban design.")

if __name__ == "__main__":
    asyncio.run(run_advanced_qadi())