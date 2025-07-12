#!/usr/bin/env python3
"""
Simple test script to demonstrate Mad Spark Alt with custom prompts.
"""

import asyncio
from typing import Optional, Any

from mad_spark_alt.core import SmartQADIOrchestrator


async def simple_qadi_test(problem: str, context: str = "") -> Optional[Any]:
    """Smart test with automatic agent registration and setup"""

    print(f"🎯 Problem: {problem}")
    if context:
        print(f"📝 Context: {context}")
    print("-" * 50)

    # Create smart orchestrator (automatically handles all agent registration)
    orchestrator = SmartQADIOrchestrator()

    # Ensure agents are ready
    await orchestrator.ensure_agents_ready()

    print("✅ Smart orchestrator ready with all QADI agents")

    # Run QADI cycle
    try:
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem,
            context=context,
            cycle_config={"max_ideas_per_method": 2, "require_reasoning": True},
        )

        print(f"⏱️  Completed in {result.execution_time:.2f}s")
        print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
        print(f"📊 Phases executed: {list(result.phases.keys())}")

        # Show results
        for phase_name, phase_result in result.phases.items():
            print(f"\n🔸 {phase_name.upper()}:")
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"  {i}. {idea.content}")
                if hasattr(idea, "reasoning") and idea.reasoning:
                    reasoning_short = (
                        idea.reasoning[:150] + "..."
                        if len(idea.reasoning) > 150
                        else idea.reasoning
                    )
                    print(f"     💭 {reasoning_short}")

        print(f"\n🎨 Total ideas generated: {len(result.synthesized_ideas)}")
        return result

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing dependencies (may fall back to template agents): {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main() -> None:
    """Run tests with different custom prompts"""

    print("🌟 Mad Spark Alt - Simple Custom Prompt Test")
    print("=" * 60)

    # Test cases you can customize
    test_prompts = [
        {
            "problem": "How can we make online learning more engaging for teenagers?",
            "context": "Focus on practical solutions that don't require expensive technology",
        },
        {
            "problem": "What are creative ways to reduce plastic waste in offices?",
            "context": "Small to medium businesses with limited budgets",
        },
        {
            "problem": "How can we improve mental health support in remote work environments?",
            "context": "Tech companies with distributed teams",
        },
    ]

    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'='*15} Test {i} {'='*15}")
        await simple_qadi_test(test["problem"], test["context"])

        if i < len(test_prompts):
            print("\n" + "." * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
