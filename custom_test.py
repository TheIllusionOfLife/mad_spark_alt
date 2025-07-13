#!/usr/bin/env python3
"""
Custom testing script for Mad Spark Alt QADI system.
This shows how to use the system with your own prompts.
"""

import asyncio
from typing import Optional, Any

from mad_spark_alt.core import SmartQADIOrchestrator

# Import test utilities from the correct path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'demos', 'test_files'))

try:
    from test_utils import truncate_text, validate_qadi_result
except ImportError:
    # Fallback implementations if test_utils is not available
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Fallback text truncation."""
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def validate_qadi_result(result) -> bool:
        """Fallback validation."""
        return result is not None


async def test_with_custom_prompt(
    problem_statement: str,
    context: str = "",
    max_ideas: int = 2,
) -> Optional[Any]:
    """Test the QADI system with a custom prompt"""

    print("🎯 Testing with custom prompt:")
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

        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem_statement,
            context=context,
            cycle_config={"max_ideas_per_method": max_ideas, "require_reasoning": True},
        )

        print(f"✅ QADI cycle completed in {result.execution_time:.2f}s")
        print(f"💰 Total LLM cost: ${result.llm_cost:.4f}")
        print("=" * 60)

        # Display results for each phase
        for phase_name, phase_result in result.phases.items():
            print(f"\n{phase_name.upper()} PHASE:")
            print(f"Agent: {phase_result.agent_name}")
            print(f"Ideas: {len(phase_result.generated_ideas)}")

            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"\n  {i}. {idea.content}")
                if hasattr(idea, "reasoning") and idea.reasoning:
                    print(f"     💭 {truncate_text(idea.reasoning, 200)}")
                if hasattr(idea, "confidence_score") and idea.confidence_score is not None:
                    print(f"     📊 Confidence: {idea.confidence_score}")

        print(f"\n🎨 Total synthesized ideas: {len(result.synthesized_ideas)}")
        
        # Validate result
        if validate_qadi_result(result):
            print("✅ QADI result validation passed")
        else:
            print("⚠️  QADI result validation failed")
            
        return result

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing dependencies: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during QADI cycle: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main() -> None:
    """Main testing function with multiple examples"""

    print("🌟 Mad Spark Alt - Custom Prompt Testing")
    print("=" * 60)

    # Test cases with different types of problems
    test_cases = [
        {
            "problem": "How can we reduce food waste in restaurants?",
            "context": "Focus on practical, cost-effective solutions that restaurants can implement immediately",
            "max_ideas": 2,
        },
        {
            "problem": "What are innovative ways to teach programming to children?",
            "context": "Consider different learning styles and make it engaging for ages 8-12",
            "max_ideas": 2,
        },
        {
            "problem": "How can remote teams improve their collaboration?",
            "context": "Team of 10 people across 3 time zones, mixed technical and non-technical roles",
            "max_ideas": 3,
        },
    ]

    print(f"📋 Running {len(test_cases)} test cases...")
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*20} TEST CASE {i} {'='*20}")
        result = await test_with_custom_prompt(
            test_case["problem"], test_case["context"], test_case["max_ideas"]
        )

        if result:
            print(f"✅ Test case {i} completed successfully")
        else:
            print(f"❌ Test case {i} failed")

        print()


if __name__ == "__main__":
    asyncio.run(main())
