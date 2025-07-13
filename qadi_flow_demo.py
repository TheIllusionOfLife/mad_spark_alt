#!/usr/bin/env python3
"""
Demonstrate how QADI cycle processes user prompts step by step.
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def trace_qadi_flow():
    """Trace how QADI methodology processes a user prompt."""
    
    user_prompt = "How to reduce plastic waste in oceans"
    
    print("🔬 QADI Flow Analysis")
    print("=" * 60)
    print(f"📝 User Prompt: '{user_prompt}'")
    print("=" * 60)
    
    orchestrator = SmartQADIOrchestrator()
    
    # Run QADI cycle with detailed tracking
    result = await orchestrator.run_qadi_cycle(
        problem_statement=user_prompt,
        context="Consider technological, policy, and behavioral approaches",
        cycle_config={"max_ideas_per_method": 2}
    )
    
    print("\n🔄 QADI METHODOLOGY BREAKDOWN:")
    print("-" * 60)
    
    qadi_explanations = {
        'questioning': {
            'theory': "Generate diverse questions to explore the problem space",
            'purpose': "Frame the problem from multiple angles and identify key aspects",
            'output': "Questions that reveal different dimensions of the challenge"
        },
        'abduction': {
            'theory': "Create hypotheses and creative leaps about possible solutions", 
            'purpose': "Generate innovative possibilities through creative reasoning",
            'output': "Novel hypotheses about what might work and why"
        },
        'deduction': {
            'theory': "Apply logical reasoning to validate and structure ideas",
            'purpose': "Test logical consistency and derive concrete implications", 
            'output': "Logical frameworks and systematic analysis"
        },
        'induction': {
            'theory': "Synthesize patterns and generalizable principles",
            'purpose': "Extract broader insights and create unified understanding",
            'output': "General patterns and overarching principles"
        }
    }
    
    # Show each phase
    phase_order = ['questioning', 'abduction', 'deduction', 'induction']
    
    for i, phase_name in enumerate(phase_order, 1):
        phase_result = result.phases.get(phase_name)
        explanation = qadi_explanations.get(phase_name, {})
        
        print(f"\n{i}. 🔸 {phase_name.upper()} PHASE")
        print(f"   Theory: {explanation.get('theory', 'N/A')}")
        print(f"   Purpose: {explanation.get('purpose', 'N/A')}")
        print(f"   Expected: {explanation.get('output', 'N/A')}")
        
        if phase_result and phase_result.generated_ideas:
            print(f"   Agent: {result.agent_types.get(phase_name, 'unknown')}")
            print("   Actual Output:")
            for j, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"     {j}. {idea.content}")
        else:
            print("   ❌ No output generated")
    
    print("\n" + "=" * 60)
    print("🔗 QADI SYNTHESIS:")
    print(f"   Total Ideas Generated: {len(result.synthesized_ideas)}")
    print(f"   Processing Time: {result.execution_time:.3f}s")
    print(f"   Context Building: Each phase builds on previous phases")
    
    # Show how context builds between phases
    print("\n📈 CONTEXT BUILDING EXAMPLE:")
    print("   Phase 1 (Questions) → Identifies key aspects")
    print("   Phase 2 (Abduction) → Uses questions to generate creative ideas")  
    print("   Phase 3 (Deduction) → Logically analyzes the creative ideas")
    print("   Phase 4 (Induction) → Synthesizes patterns from all previous work")

if __name__ == "__main__":
    asyncio.run(trace_qadi_flow())