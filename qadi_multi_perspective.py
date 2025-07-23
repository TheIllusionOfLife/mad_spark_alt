#!/usr/bin/env python3
"""
Multi-Perspective QADI Analysis

This script provides QADI analysis from multiple perspectives based on
automatic intent detection, ensuring answers match the user's actual needs.

Usage:
    uv run python qadi_multi_perspective.py "Your question here"
    uv run python qadi_multi_perspective.py "Your question" --perspectives environmental,personal
    uv run python qadi_multi_perspective.py "Your question" --temperature 0.9
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.intent_detector import IntentDetector, QuestionIntent
    from mad_spark_alt.core.multi_perspective_orchestrator import (
        MultiPerspectiveQADIOrchestrator,
        PerspectiveResult,
    )
    from mad_spark_alt.core.terminal_renderer import render_markdown
except ImportError:
    # Fallback if package is not installed
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.intent_detector import IntentDetector, QuestionIntent
    from mad_spark_alt.core.multi_perspective_orchestrator import (
        MultiPerspectiveQADIOrchestrator,
        PerspectiveResult,
    )
    from mad_spark_alt.core.terminal_renderer import render_markdown


def display_intent_detection(intent_result, perspectives: List[QuestionIntent]) -> None:
    """Display intent detection results."""
    print("\n🔍 Intent Detection:")
    print(f"Primary Intent: {intent_result.primary_intent.value.title()} "
          f"(confidence: {intent_result.confidence:.0%})")
    
    if intent_result.keywords_matched:
        print(f"Keywords Detected: {', '.join(intent_result.keywords_matched[:5])}")
    
    print(f"\n📊 Analysis Perspectives: {', '.join(p.value.title() for p in perspectives)}")
    print("─" * 70)


def display_perspective_result(pr: PerspectiveResult, is_primary: bool = False) -> None:
    """Display results from a single perspective."""
    emoji_map = {
        QuestionIntent.ENVIRONMENTAL: "🌍",
        QuestionIntent.PERSONAL: "👤",
        QuestionIntent.TECHNICAL: "💻",
        QuestionIntent.BUSINESS: "💼",
        QuestionIntent.SCIENTIFIC: "🔬",
        QuestionIntent.PHILOSOPHICAL: "🤔",
        QuestionIntent.GENERAL: "📋",
    }
    
    emoji = emoji_map.get(pr.perspective, "📋")
    header = f"{emoji} {pr.perspective.value.title()} Perspective"
    if is_primary:
        header += " (Primary)"
    
    print(f"\n{'=' * 70}")
    print(f"{header}")
    print(f"{'=' * 70}")
    
    # Core Question
    print(f"\n**Core Question:**")
    render_markdown(pr.result.core_question)
    
    # Hypotheses with scores
    print(f"\n**Hypotheses:**")
    for i, (hyp, score) in enumerate(zip(pr.result.hypotheses, pr.result.hypothesis_scores)):
        score_bar = "█" * int(score.overall * 10) + "░" * (10 - int(score.overall * 10))
        print(f"\nH{i+1} [{score_bar}] {score.overall:.2f}")
        render_markdown(hyp)
    
    # Answer
    print(f"\n**Answer:**")
    render_markdown(pr.result.final_answer)
    
    # Action Plan - Fixed formatting
    if pr.result.action_plan:
        print(f"\n**Action Plan:**")
        for i, action in enumerate(pr.result.action_plan, 1):
            # Clean action text and ensure proper formatting
            action_text = action.strip()
            print(f"{i}. {action_text}")
    
    # Verification Examples (shortened for multi-perspective view)
    if pr.result.verification_examples:
        print(f"\n**Key Verification:**")
        # Show only first example in multi-perspective mode
        render_markdown(f"• {pr.result.verification_examples[0][:200]}...")


def display_synthesis(result) -> None:
    """Display synthesized results."""
    print(f"\n{'=' * 70}")
    print("🎯 Synthesized Analysis")
    print(f"{'=' * 70}")
    
    print("\n**Integrated Answer:**")
    render_markdown(result.synthesized_answer)
    
    print(f"\n**Best Hypothesis:** ({result.best_hypothesis[1].value.title()} perspective)")
    render_markdown(result.best_hypothesis[0])
    
    print("\n**Integrated Action Plan:**")
    for i, action in enumerate(result.synthesized_action_plan, 1):
        action_text = action.strip()
        print(f"{i}. {action_text}")


async def run_multi_perspective_analysis(
    user_input: str,
    temperature: Optional[float] = None,
    force_perspectives: Optional[List[str]] = None,
    show_all_perspectives: bool = False,
) -> None:
    """Run multi-perspective QADI analysis."""
    
    print("🧠 Multi-Perspective QADI Analysis")
    print("=" * 70)
    print(f"\n📝 User Input: {user_input}")
    
    # Check for API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("\n❌ Error: GOOGLE_API_KEY not found in environment")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        return
    
    # Parse forced perspectives if provided
    forced_intents = None
    if force_perspectives:
        try:
            forced_intents = [
                QuestionIntent[p.upper()] for p in force_perspectives
            ]
        except KeyError as e:
            print(f"\n❌ Error: Invalid perspective '{e.args[0]}'")
            print(f"Valid perspectives: {', '.join(i.value for i in QuestionIntent)}")
            return
    
    # Create orchestrator
    orchestrator = MultiPerspectiveQADIOrchestrator(temperature_override=temperature)
    
    if temperature:
        print(f"🌡️  Temperature override: {temperature}")
    
    start_time = time.time()
    
    try:
        # Run analysis
        result = await orchestrator.run_multi_perspective_analysis(
            user_input,
            max_perspectives=3,
            force_perspectives=forced_intents,
        )
        
        # Display intent detection
        intent_detector = IntentDetector()
        intent_result = intent_detector.detect_intent(user_input)
        display_intent_detection(intent_result, result.perspectives_used)
        
        # Display results based on mode
        if show_all_perspectives or len(result.perspective_results) == 1:
            # Show all perspective details
            for i, pr in enumerate(result.perspective_results):
                display_perspective_result(pr, is_primary=(i == 0))
        else:
            # Show only primary perspective in detail
            if result.perspective_results:
                display_perspective_result(result.perspective_results[0], is_primary=True)
            
            # Brief summary of other perspectives
            if len(result.perspective_results) > 1:
                print(f"\n📊 Additional Perspectives Analyzed:")
                for pr in result.perspective_results[1:]:
                    emoji_map = {
                        QuestionIntent.ENVIRONMENTAL: "🌍",
                        QuestionIntent.PERSONAL: "👤",
                        QuestionIntent.TECHNICAL: "💻",
                        QuestionIntent.BUSINESS: "💼",
                        QuestionIntent.SCIENTIFIC: "🔬",
                        QuestionIntent.PHILOSOPHICAL: "🤔",
                        QuestionIntent.GENERAL: "📋",
                    }
                    emoji = emoji_map.get(pr.perspective, "📋")
                    print(f"\n{emoji} {pr.perspective.value.title()}: {pr.result.core_question}")
        
        # Always show synthesis when multiple perspectives
        if len(result.perspective_results) > 1:
            display_synthesis(result)
        
        # Summary statistics
        elapsed_time = time.time() - start_time
        print(f"\n{'─' * 70}")
        print(f"✅ Analysis completed in {elapsed_time:.1f}s")
        print(f"💰 Total LLM cost: ${result.total_llm_cost:.4f}")
        print(f"🔍 Perspectives used: {len(result.perspective_results)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-perspective QADI analysis with automatic intent detection"
    )
    parser.add_argument("input", help="Your question, problem, or topic to analyze")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Temperature for hypothesis generation (0.0-2.0)",
        default=None,
    )
    parser.add_argument(
        "--perspectives",
        "-p",
        type=str,
        help="Force specific perspectives (comma-separated: environmental,personal,technical,business,scientific,philosophical)",
        default=None,
    )
    parser.add_argument(
        "--show-all",
        "-a",
        action="store_true",
        help="Show detailed results from all perspectives (default: show synthesis)",
    )
    
    args = parser.parse_args()
    
    # Validate temperature
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        print(f"Error: Temperature must be between 0.0 and 2.0 (got {args.temperature})")
        sys.exit(1)
    
    # Parse perspectives
    force_perspectives = None
    if args.perspectives:
        force_perspectives = [p.strip() for p in args.perspectives.split(",")]
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Initialize and run
    async def main_async():
        # Initialize LLM providers before running analysis
        await setup_llm_providers(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        await run_multi_perspective_analysis(
            args.input,
            temperature=args.temperature,
            force_perspectives=force_perspectives,
            show_all_perspectives=args.show_all,
        )
    
    asyncio.run(main_async())


if __name__ == "__main__":
    main()