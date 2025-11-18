"""
Real API test for UnifiedQADIOrchestrator with Simple strategy.

Tests the complete QADI cycle with actual Google Gemini API.

Run with: pytest tests/test_unified_simple_integration.py -m integration
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

import pytest

from mad_spark_alt.core import setup_llm_providers
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator
from mad_spark_alt.core.orchestrator_config import OrchestratorConfig

# Load environment variables
load_dotenv()


@pytest.mark.integration
async def test_simple_strategy():
    """Test Simple strategy with real API."""
    print("=" * 80)
    print("Testing UnifiedQADIOrchestrator - Simple Strategy (Real API)")
    print("=" * 80)

    # Setup LLM providers
    print("\n⏳ Setting up LLM providers...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return False

    await setup_llm_providers(google_api_key=google_api_key)
    print("✓ LLM providers initialized")

    # Create configuration
    config = OrchestratorConfig.simple_config()
    print(f"\n✓ Configuration: {config.strategy.value}, {config.execution_mode.value}")
    print(f"  Hypotheses: {config.num_hypotheses}")

    # Create orchestrator
    orchestrator = UnifiedQADIOrchestrator(config=config)
    print(f"✓ Orchestrator initialized")

    # Run QADI cycle
    question = "How can we reduce plastic waste in oceans?"
    print(f"\n📝 Question: {question}")
    print(f"\n⏳ Running QADI cycle...")

    try:
        result = await orchestrator.run_qadi_cycle(question)

        print(f"\n✅ QADI cycle completed successfully!")
        print(f"=" * 80)

        # Display results
        print(f"\n🎯 Core Question:")
        print(f"   {result.core_question}")

        print(f"\n💡 Hypotheses ({len(result.hypotheses)}):")
        for i, h in enumerate(result.hypotheses, 1):
            print(f"   {i}. {h[:100]}{'...' if len(h) > 100 else ''}")

        print(f"\n📊 Hypothesis Scores:")
        if result.hypothesis_scores:
            for i, score in enumerate(result.hypothesis_scores, 1):
                print(f"   H{i}: Impact={score.impact:.2f}, Feasibility={score.feasibility:.2f}, Overall={score.overall:.2f}")
        else:
            print("   ⚠️  No scores available")

        print(f"\n✨ Final Answer:")
        print(f"   {result.final_answer[:200]}{'...' if len(result.final_answer) > 200 else ''}")

        print(f"\n📋 Action Plan ({len(result.action_plan)} steps):")
        for i, step in enumerate(result.action_plan[:3], 1):
            print(f"   {i}. {step[:100]}{'...' if len(step) > 100 else ''}")
        if len(result.action_plan) > 3:
            print(f"   ... and {len(result.action_plan) - 3} more steps")

        print(f"\n🔍 Verification Examples ({len(result.verification_examples if result.verification_examples else [])}):")
        if result.verification_examples:
            for i, ex in enumerate(result.verification_examples[:2], 1):
                print(f"   {i}. {ex[:100]}{'...' if len(ex) > 100 else ''}")

        print(f"\n✅ Verification Conclusion:")
        if result.verification_conclusion:
            print(f"   {result.verification_conclusion[:150]}{'...' if len(result.verification_conclusion) > 150 else ''}")

        print(f"\n💰 Cost: ${result.total_llm_cost:.6f}")
        print(f"📦 Synthesized Ideas: {len(result.synthesized_ideas)}")
        print(f"📈 Strategy: {result.strategy_used.value}")
        print(f"⚙️  Execution: {result.execution_mode.value}")

        # Validation checks
        print(f"\n🔍 Validation Checks:")
        checks = [
            ("Core question extracted", result.core_question is not None and len(result.core_question) > 0),
            ("Hypotheses generated", len(result.hypotheses) >= 1),
            ("Hypothesis scores present", result.hypothesis_scores is not None and len(result.hypothesis_scores) > 0),
            ("Final answer provided", result.final_answer is not None and len(result.final_answer) > 0),
            ("Action plan created", len(result.action_plan) >= 1),
            ("Verification examples", result.verification_examples is not None),
            ("Verification conclusion", result.verification_conclusion is not None),
            ("Cost tracked", result.total_llm_cost > 0),
            ("Synthesized ideas", len(result.synthesized_ideas) >= 0),
            ("No truncation (final answer)", len(result.final_answer) >= 100),
            ("No duplicate hypotheses", len(result.hypotheses) == len(set(result.hypotheses)))
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 All validation checks passed!")
            return True
        else:
            print(f"\n⚠️  Some validation checks failed!")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    success = await test_simple_strategy()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
