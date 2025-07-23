#!/usr/bin/env python3
"""
Simplified QADI Analysis Demo

This script experiments with a simpler Phase 1 that just identifies the user's question
rather than trying to extract a "core question".
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.terminal_renderer import render_markdown
except ImportError:
    # Fallback if package is not installed
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.terminal_renderer import render_markdown


# Override the questioning prompt with a simpler version
class SimplerQADIOrchestrator(SimpleQADIOrchestrator):
    """QADI orchestrator with simplified Phase 1."""
    
    def __init__(self, temperature_override: Optional[float] = None) -> None:
        super().__init__(temperature_override)
        # Override the questioning prompt
        self.prompts.get_questioning_prompt = self._get_simpler_questioning_prompt
    
    def _get_simpler_questioning_prompt(self, user_input: str) -> str:
        """Get a much simpler prompt for Phase 1."""
        return f"""What is the user asking?

User's input:
{user_input}

State their question clearly and directly. If they made a statement, rephrase it as the implied question.
Format: "Q: [The user's question]"
"""


async def run_qadi_analysis(
    user_input: str, temperature: Optional[float] = None, verbose: bool = False
) -> None:
    """Run QADI analysis with simplified Phase 1."""

    print("🧠 Simplified QADI Analysis")
    print("=" * 50)
    print(f"\n📝 User Input: {user_input}")

    # Check for API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("\n❌ Error: GOOGLE_API_KEY not found in environment")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        return

    # Create orchestrator with optional temperature override
    orchestrator = SimplerQADIOrchestrator(temperature_override=temperature)

    if temperature:
        print(f"🌡️  Temperature override: {temperature}")

    print("\n" + "─" * 50)
    start_time = time.time()

    try:
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(user_input)

        # Display results
        print("\n## 🎯 Phase 1: User's Question\n")
        render_markdown(f"**Q:** {result.core_question}")

        print("\n## 💡 Phase 2: Possible Approaches\n")
        for i, hypothesis in enumerate(result.hypotheses):
            render_markdown(f"**Approach {i+1}:** {hypothesis}")

        print("\n## 🔍 Phase 3: Analysis & Recommendation\n")

        # Show evaluation scores if verbose
        if verbose and result.hypothesis_scores:
            print("### Evaluation Scores:\n")
            for i, (_, scores) in enumerate(
                zip(result.hypotheses, result.hypothesis_scores)
            ):
                print(f"**Approach {i+1} Scores:**")
                print(f"  - Impact: {scores.impact:.2f}")
                print(f"  - Feasibility: {scores.feasibility:.2f}")
                print(f"  - Accessibility: {scores.accessibility:.2f}")
                print(f"  - Sustainability: {scores.sustainability:.2f}")
                print(f"  - Scalability: {scores.scalability:.2f}")
                print(f"  - **Overall: {scores.overall:.2f}**")
                print()

        render_markdown(f"### Recommendation\n\n{result.final_answer}")

        if result.action_plan:
            print("\n### Action Plan\n")
            for i, action in enumerate(result.action_plan):
                render_markdown(f"{i+1}. {action}")

        print("\n## ✅ Phase 4: Real-World Examples\n")
        if result.verification_examples:
            for i, example in enumerate(result.verification_examples):
                # Parse example structure for better formatting
                lines = example.split('\n')
                
                # Look for structured format markers
                context_line = None
                application_line = None
                result_line = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('- Context:') or line.startswith('Context:'):
                        context_line = line.replace('- Context:', '').replace('Context:', '').strip()
                    elif line.startswith('- Application:') or line.startswith('Application:'):
                        application_line = line.replace('- Application:', '').replace('Application:', '').strip()
                    elif line.startswith('- Result:') or line.startswith('Result:'):
                        result_line = line.replace('- Result:', '').replace('Result:', '').strip()
                
                # Display with better formatting
                print(f"### Example {i+1}")
                if context_line:
                    render_markdown(f"**Context:** {context_line}")
                    if application_line:
                        render_markdown(f"**Application:** {application_line}")
                    if result_line:
                        render_markdown(f"**Result:** {result_line}")
                else:
                    # Fallback to original format if structure not found
                    render_markdown(example)
                print()  # Add spacing between examples

        if result.verification_conclusion:
            print("\n### Conclusion\n")
            render_markdown(result.verification_conclusion)

        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "─" * 50)
        print(f"\n✅ Analysis completed in {elapsed_time:.1f}s")
        print(f"💰 Total LLM cost: ${result.total_llm_cost:.4f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run simplified QADI analysis on any input"
    )
    parser.add_argument("input", help="Your question, problem, or topic to analyze")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Temperature for hypothesis generation (0.0-2.0, default: 0.8)",
        default=None,
        metavar="T",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed evaluation scores"
    )

    args = parser.parse_args()

    # Validate temperature if provided
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        print(
            f"Error: Temperature must be between 0.0 and 2.0 (got {args.temperature})"
        )
        sys.exit(1)

    # Load environment variables (optional)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print(
            "Warning: python-dotenv not available, environment variables not loaded from .env file"
        )

    # Initialize LLM providers
    async def main_async():
        try:
            await setup_llm_providers(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        except Exception as e:
            print(f"Warning: Failed to initialize LLM providers: {e}")

        await run_qadi_analysis(
            args.input, temperature=args.temperature, verbose=args.verbose
        )

    # Run analysis
    asyncio.run(main_async())


if __name__ == "__main__":
    main()