#!/usr/bin/env python3
"""
Working test script for Mad Spark Alt with custom prompts.
"""

import asyncio
from typing import Optional, Any

from mad_spark_alt.core import SmartQADIOrchestrator
from test_utils import truncate_text, validate_qadi_result


async def test_custom_prompt(problem: str, context: str = "") -> Optional[bool]:
    """Test the system with a custom prompt using Smart Orchestrator"""

    print(f"🎯 Testing: {problem}")
    if context:
        print(f"📝 Context: {context}")
    print("-" * 60)

    # Create smart orchestrator (automatically handles all agent registration)
    orchestrator = SmartQADIOrchestrator()

    # Ensure agents are ready
    await orchestrator.ensure_agents_ready()

    print("✅ Smart orchestrator ready with all QADI agents")

    try:
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem,
            context=context,
            cycle_config={"max_ideas_per_method": 2, "require_reasoning": True},
        )

        print(f"⏱️  Execution time: {result.execution_time:.2f}s")
        print(f"💰 LLM Cost: ${result.llm_cost:.4f}")

        # Display results
        for phase_name, phase_result in result.phases.items():
            print(f"\n🔸 {phase_name.upper()} PHASE:")
            print(f"   Agent: {phase_result.agent_name}")

            for i, idea in enumerate(phase_result.generated_ideas, 1):
                print(f"\n   {i}. {idea.content}")
                if hasattr(idea, "reasoning") and idea.reasoning:
                    print(f"      💭 {truncate_text(idea.reasoning, 200)}")
                if hasattr(idea, "confidence_score") and idea.confidence_score is not None:
                    print(f"      📊 Confidence: {idea.confidence_score}")

        print(f"\n🎨 Total synthesized ideas: {len(result.synthesized_ideas)}")
        
        # Validate result
        validation_passed = validate_qadi_result(result)
        if validation_passed:
            print("✅ QADI result validation passed")
        
        return validation_passed

    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_creativity_evaluation(text: str) -> Optional[float]:
    """Test the creativity evaluation system"""
    print(f"📊 Evaluating creativity of: '{text[:50]}...'")

    from mad_spark_alt.core.evaluator import CreativityEvaluator
    from mad_spark_alt.core.interfaces import EvaluationRequest, ModelOutput, OutputType

    output = ModelOutput(
        content=text, output_type=OutputType.TEXT, model_name="test-model"
    )

    evaluator = CreativityEvaluator()
    request = EvaluationRequest(outputs=[output])

    try:
        summary = await evaluator.evaluate(request)
        creativity_score = summary.get_overall_creativity_score()
        print(f"   🎨 Creativity Score: {creativity_score:.3f}")

        # Show some metrics
        if summary.evaluation_results:
            result = summary.evaluation_results[0]
            print("   📈 Metrics:")
            for evaluator_result in result.results:
                for metric, value in evaluator_result.scores.items():
                    print(f"      {metric}: {value:.3f}")

        return creativity_score
    except (ImportError, ModuleNotFoundError) as e:
        print(f"   ❌ Missing evaluation dependencies: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main() -> None:
    """Main demonstration function"""

    print("🌟 Mad Spark Alt - Custom Prompt Testing Guide")
    print("=" * 60)
    print("This script demonstrates how you can test the system with your own prompts")
    print()

    # Test 1: QADI Idea Generation
    print("1️⃣  QADI IDEA GENERATION TEST")
    print("=" * 40)

    success = await test_custom_prompt(
        problem="How can we reduce screen time for children while keeping them engaged?",
        context="Focus on activities that are both educational and fun for ages 6-12",
    )

    if success:
        print("✅ QADI test completed successfully!")
    else:
        print("❌ QADI test failed")

    print("\n" + "." * 60 + "\n")

    # Test 2: Creativity Evaluation
    print("2️⃣  CREATIVITY EVALUATION TEST")
    print("=" * 40)

    test_texts = [
        "The quantum butterfly effect of artificial intelligence dreams.",
        "Regular office meeting about quarterly reports and budget planning.",
        "Time-traveling librarians cataloging parallel universe literature.",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        await test_creativity_evaluation(text)

    print("\n" + "=" * 60)
    print("🎯 HOW TO USE WITH YOUR OWN PROMPTS:")
    print("=" * 60)
    print(
        """
1. QADI Idea Generation:
   - Replace the 'problem' and 'context' variables with your own
   - Adjust 'max_ideas_per_method' in cycle_config for more/fewer ideas
   - The system will generate questions, hypotheses, deductions, and inductions

2. Creativity Evaluation:
   - Replace text in test_texts with your own content
   - Use CLI: uv run mad-spark evaluate "your text here"
   - Higher scores indicate more creative content

3. Add API Keys for LLM Agents (optional but recommended):
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key" 
   export GOOGLE_API_KEY="your-key"
   
4. Examples to try:
   - "How can we make cities more pedestrian-friendly?"
   - "What are innovative ways to reduce workplace stress?"
   - "How can we improve online education for adults?"
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
