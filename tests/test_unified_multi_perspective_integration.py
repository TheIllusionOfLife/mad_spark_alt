"""
Real API validation test for UnifiedQADIOrchestrator MultiPerspective strategy.

This script tests the MultiPerspective strategy with Google Gemini API to verify:
1. Complete multi-perspective cycle (all phases)
2. No timeouts
3. No truncation
4. No duplicate content
5. Cost tracking accuracy
6. Multiple perspectives detected/used
7. Top N hypotheses collected correctly
8. Synthesized answer present and meaningful
9. Action plan complete
10. All UnifiedQADIResult fields populated
11. perspectives_used list correct

Run with: pytest tests/test_unified_multi_perspective_integration.py -m integration
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

import pytest

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core import (
    UnifiedQADIOrchestrator,
    OrchestratorConfig,
    Strategy,
    ExecutionMode
)
from mad_spark_alt.core.llm_provider import setup_llm_providers


async def main():
    """Run real API validation test."""

    print("=" * 80)
    print("Real API Validation: UnifiedQADIOrchestrator - MultiPerspective Strategy")
    print("=" * 80)
    print()

    # Setup LLM providers
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set it in .env file")
        return False

    await setup_llm_providers(google_api_key=api_key)
    print("✅ LLM provider initialized")
    print()

    # Test question that should trigger multiple perspectives
    question = "How can we create sustainable cities that are both economically viable and environmentally friendly?"
    print(f"📝 Question: {question}")
    print()

    # Create config for MultiPerspective strategy with auto-detection
    config = OrchestratorConfig(
        strategy=Strategy.MULTI_PERSPECTIVE,
        execution_mode=ExecutionMode.SEQUENTIAL,
        auto_detect_perspectives=True,
        num_hypotheses=5  # Top 5 across all perspectives
    )

    orchestrator = UnifiedQADIOrchestrator(config=config)
    print(f"✅ UnifiedQADIOrchestrator created with config:")
    print(f"   - Strategy: {config.strategy.value}")
    print(f"   - Auto-detect perspectives: True")
    print(f"   - Top N hypotheses: {config.num_hypotheses}")
    print()

    # Run QADI cycle
    print("🚀 Running QADI cycle with real API...")
    print()

    try:
        import time
        start_time = time.time()

        result = await orchestrator.run_qadi_cycle(question)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ QADI cycle completed in {duration:.1f}s")
        print()

    except Exception as e:
        print(f"❌ QADI cycle failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    print()

    checks_passed = 0
    total_checks = 11

    # Check 1: Complete multi-perspective cycle
    if result and result.final_answer:
        print("✅ Check 1: Complete multi-perspective cycle")
        checks_passed += 1
    else:
        print("❌ Check 1: FAILED - No result or final answer")

    # Check 2: No timeouts (completed within reasonable time ~90s)
    if duration < 180:  # 3 minutes max for MP strategy
        print(f"✅ Check 2: No timeout (completed in {duration:.1f}s)")
        checks_passed += 1
    else:
        print(f"❌ Check 2: FAILED - Took too long ({duration:.1f}s)")

    # Check 3: No truncation (answer has substance)
    if len(result.final_answer) > 50:
        print(f"✅ Check 3: No truncation (answer length: {len(result.final_answer)})")
        checks_passed += 1
    else:
        print(f"❌ Check 3: FAILED - Answer too short ({len(result.final_answer)} chars)")

    # Check 4: No duplicate content
    words = result.final_answer.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio > 0.5:
        print(f"✅ Check 4: No significant duplication (unique ratio: {unique_ratio:.2%})")
        checks_passed += 1
    else:
        print(f"❌ Check 4: FAILED - Too much duplication ({unique_ratio:.2%})")

    # Check 5: Cost tracking accurate
    if result.total_llm_cost > 0:
        print(f"✅ Check 5: Cost tracking working (${result.total_llm_cost:.6f})")
        checks_passed += 1
    else:
        print("❌ Check 5: FAILED - No cost tracked")

    # Check 6: Multiple perspectives detected/used (2-3 expected)
    if result.perspectives_used and len(result.perspectives_used) >= 2:
        print(f"✅ Check 6: Multiple perspectives used ({len(result.perspectives_used)}): {result.perspectives_used}")
        checks_passed += 1
    else:
        print(f"❌ Check 6: FAILED - Expected 2+ perspectives, got {len(result.perspectives_used) if result.perspectives_used else 0}")

    # Check 7: Top N hypotheses collected correctly
    if result.hypotheses and len(result.hypotheses) <= config.num_hypotheses:
        print(f"✅ Check 7: Top N hypotheses correct ({len(result.hypotheses)} ≤ {config.num_hypotheses})")
        checks_passed += 1
    else:
        print(f"❌ Check 7: FAILED - Expected ≤{config.num_hypotheses} hypotheses, got {len(result.hypotheses) if result.hypotheses else 0}")

    # Check 8: Synthesized answer present and meaningful
    if result.synthesized_answer and len(result.synthesized_answer) > 50:
        print(f"✅ Check 8: Synthesized answer meaningful ({len(result.synthesized_answer)} chars)")
        checks_passed += 1
    else:
        print(f"❌ Check 8: FAILED - Synthesized answer missing or too short")

    # Check 9: Action plan complete (3+ steps)
    if result.action_plan and len(result.action_plan) >= 3:
        print(f"✅ Check 9: Action plan complete ({len(result.action_plan)} steps)")
        checks_passed += 1
    else:
        print(f"❌ Check 9: FAILED - Expected 3+ action steps, got {len(result.action_plan) if result.action_plan else 0}")

    # Check 10: All UnifiedQADIResult fields populated
    required_fields = ['strategy_used', 'execution_mode', 'core_question', 'hypotheses',
                      'final_answer', 'action_plan', 'total_llm_cost', 'synthesized_ideas']
    missing_fields = [f for f in required_fields if not getattr(result, f, None)]
    if not missing_fields:
        print("✅ Check 10: All required fields populated")
        checks_passed += 1
    else:
        print(f"❌ Check 10: FAILED - Missing fields: {missing_fields}")

    # Check 11: perspectives_used list correct (strings not enums)
    if result.perspectives_used and all(isinstance(p, str) for p in result.perspectives_used):
        print(f"✅ Check 11: perspectives_used list correct (all strings)")
        checks_passed += 1
    else:
        print("❌ Check 11: FAILED - perspectives_used not all strings")

    print()
    print("=" * 80)
    print(f"RESULTS: {checks_passed}/{total_checks} checks passed")
    print("=" * 80)
    print()

    # Print sample output for user review
    print("=" * 80)
    print("SAMPLE OUTPUT (User Review)")
    print("=" * 80)
    print()

    print(f"Strategy: {result.strategy_used.value}")
    print(f"Perspectives Used: {result.perspectives_used}")
    print(f"Perspective Count: {result.phase_results.get('perspective_count', 'N/A')}")
    print(f"Primary Intent: {result.phase_results.get('primary_intent', 'N/A')}")
    print()

    print("Top Hypotheses:")
    for i, hyp in enumerate(result.hypotheses[:3], 1):
        print(f"  {i}. {hyp[:100]}{'...' if len(hyp) > 100 else ''}")
    print()

    print("Synthesized Answer (first 300 chars):")
    print(result.synthesized_answer[:300] + ("..." if len(result.synthesized_answer) > 300 else ""))
    print()

    print("Action Plan:")
    for i, step in enumerate(result.action_plan, 1):
        print(f"  {i}. {step}")
    print()

    print(f"Total Cost: ${result.total_llm_cost:.6f}")
    print()

    return checks_passed == total_checks


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
