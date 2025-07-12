#!/usr/bin/env python3
"""
Custom testing script for Mad Spark Alt QADI system.
This shows how to use the system with your own prompts.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core import SmartQADIOrchestrator, IdeaGenerationRequest

async def test_with_custom_prompt(problem_statement: str, context: str = "", max_ideas: int = 2):
    """Test the QADI system with a custom prompt"""
    
    print(f"🎯 Testing with custom prompt:")
    print(f"   Problem: {problem_statement}")
    if context:
        print(f"   Context: {context}")
    print(f"   Max ideas per method: {max_ideas}")
    print("=" * 60)
    
    # Create smart orchestrator (automatically handles all agent registration)
    orchestrator = SmartQADIOrchestrator()
    
    # Ensure agents are ready
    await orchestrator.ensure_agents_ready()
    
    print("🤖 Smart orchestrator ready with automatic agent selection")
    
    try:
        # Run QADI cycle
        print("🚀 Starting QADI cycle...")
        start_time = datetime.now()
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem_statement,
            context=context,
            cycle_config={
                "max_ideas_per_method": max_ideas,
                "require_reasoning": True
            }
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"✅ QADI cycle completed in {execution_time:.2f}s")
        print(f"💰 Total LLM cost: ${result.llm_cost:.4f}")
        print("=" * 60)
        
        # Display results for each phase
        for phase_name, phase_result in result.phases.items():
            print(f"\n{phase_name.upper()} PHASE:")
            print(f"Agent: {phase_result.agent_name}")
            print(f"Ideas: {len(phase_result.generated_ideas)}")
            
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"\n  {i}. {idea.content}")
                if idea.reasoning:
                    reasoning_preview = idea.reasoning[:200] + "..." if len(idea.reasoning) > 200 else idea.reasoning
                    print(f"     💭 {reasoning_preview}")
                if idea.confidence_score:
                    print(f"     📊 Confidence: {idea.confidence_score}")
        
        print(f"\n🎨 Total synthesized ideas: {len(result.synthesized_ideas)}")
        return result
        
    except Exception as e:
        print(f"❌ Error during QADI cycle: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main testing function with multiple examples"""
    
    print("🌟 Mad Spark Alt - Custom Prompt Testing")
    print("=" * 60)
    
    # Test cases with different types of problems
    test_cases = [
        {
            "problem": "How can we reduce food waste in restaurants?",
            "context": "Focus on practical, cost-effective solutions that restaurants can implement immediately",
            "max_ideas": 2
        },
        {
            "problem": "What are innovative ways to teach programming to children?",
            "context": "Consider different learning styles and make it engaging for ages 8-12",
            "max_ideas": 2
        },
        {
            "problem": "How can remote teams improve their collaboration?",
            "context": "Team of 10 people across 3 time zones, mixed technical and non-technical roles",
            "max_ideas": 3
        }
    ]
    
    print(f"📋 Running {len(test_cases)} test cases...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*20} TEST CASE {i} {'='*20}")
        result = await test_with_custom_prompt(
            test_case["problem"],
            test_case["context"],
            test_case["max_ideas"]
        )
        
        if result:
            print(f"✅ Test case {i} completed successfully")
        else:
            print(f"❌ Test case {i} failed")
        
        print()

if __name__ == "__main__":
    asyncio.run(main())