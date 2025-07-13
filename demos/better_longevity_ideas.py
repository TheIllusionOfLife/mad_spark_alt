#!/usr/bin/env python3
"""
Generate more specific longevity ideas with targeted prompting.
"""

import asyncio
from mad_spark_alt.core import SmartQADIOrchestrator

async def generate_specific_longevity_ideas():
    """Generate specific, actionable longevity ideas."""
    
    print("🎯 Specific Longevity Strategies Generator")
    print("=" * 50)
    
    orchestrator = SmartQADIOrchestrator()
    
    # Multiple targeted problems for different aspects
    problems = [
        {
            "focus": "Diet & Nutrition",
            "problem": "What specific dietary changes can extend human lifespan?",
            "context": "Focus on evidence-based nutritional interventions, meal timing, supplements, and food combinations that have shown longevity benefits in research."
        },
        {
            "focus": "Exercise & Movement", 
            "problem": "What exercise protocols maximize longevity benefits?",
            "context": "Consider different types of exercise (cardio, strength, flexibility), intensity levels, frequency, and age-specific adaptations."
        },
        {
            "focus": "Technology & Medicine",
            "problem": "What emerging technologies could extend human lifespan in the next 20 years?",
            "context": "Include biotechnology, AI-driven healthcare, genetic therapies, regenerative medicine, and preventive monitoring technologies."
        }
    ]
    
    all_ideas = []
    
    for problem_set in problems:
        print(f"\n🔍 Focus Area: {problem_set['focus']}")
        print("-" * 30)
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem_set["problem"],
            context=problem_set["context"],
            cycle_config={"max_ideas_per_method": 2}
        )
        
        # Collect ideas from questioning and abduction phases (most actionable)
        for phase_name in ['questioning', 'abduction']:
            phase_result = result.phases.get(phase_name)
            if phase_result and phase_result.generated_ideas:
                for idea in phase_result.generated_ideas:
                    all_ideas.append({
                        'focus': problem_set['focus'],
                        'phase': phase_name,
                        'content': idea.content
                    })
                    print(f"• {idea.content}")
    
    print(f"\n🎉 Generated {len(all_ideas)} specific longevity ideas!")
    print("Focus areas covered:", set(idea['focus'] for idea in all_ideas))

if __name__ == "__main__":
    asyncio.run(generate_specific_longevity_ideas())