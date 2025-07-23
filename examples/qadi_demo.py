#!/usr/bin/env python3
"""
Demo script for the QADI cycle idea generation system.

This script demonstrates the intelligent multi-agent idea generation system using
the QADI (Question → Abduction → Deduction → Induction) methodology with automatic LLM agent preference.
"""

import asyncio
import json
import os
from pathlib import Path

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    key, value = parts
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

console = Console()


async def check_api_keys() -> bool:
    """Check and display available API keys."""
    console.print("🔐 Checking API Key Availability", style="bold blue")

    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
    }

    table = Table(title="LLM Provider Status")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Agent Type", style="yellow")

    has_api_keys = False

    for provider, key in api_keys.items():
        if key:
            table.add_row(provider, "✅ Available", "LLM-Powered")
            has_api_keys = True
        else:
            table.add_row(provider, "❌ Missing", "Template Fallback")

    console.print(table)

    if has_api_keys:
        console.print(
            "🚀 LLM providers detected! Using intelligent AI-powered agents.",
            style="green",
        )
    else:
        console.print(
            "📝 No API keys found. Using template-based agents.", style="yellow"
        )
        console.print(
            "💡 Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for AI-powered generation.",
            style="dim",
        )

    return has_api_keys


async def demo_qadi_setup() -> SimpleQADIOrchestrator:
    """Demonstrate the QADI system setup."""
    console.print("\n" + "=" * 60)
    console.print("🔧 QADI System Setup Demo", style="bold blue")
    console.print("=" * 60)

    console.print("Setting up simplified QADI analysis system...")

    # Create orchestrator
    orchestrator = SimpleQADIOrchestrator()

    # Display setup information
    setup_table = Table(title="QADI System Components")
    setup_table.add_column("Phase", style="cyan")
    setup_table.add_column("Method", style="green")
    setup_table.add_column("Purpose", style="yellow")

    setup_table.add_row("Q - Question", "Core Question Extraction", "Identifies the key question to answer")
    setup_table.add_row("A - Abduction", "Hypothesis Generation", "Generates 3 potential solutions")
    setup_table.add_row("D - Deduction", "Evaluation & Selection", "Evaluates hypotheses and selects the best")
    setup_table.add_row("I - Induction", "Verification", "Verifies with real-world examples")

    console.print(setup_table)
    console.print("\n✅ QADI system ready for analysis")

    return orchestrator


async def demo_qadi_cycle(problem: str) -> None:
    """Demonstrate the QADI cycle analysis."""
    console.print("\n" + "=" * 60)
    console.print("🔄 QADI Analysis Demo", style="bold blue")
    console.print("=" * 60)

    orchestrator = SimpleQADIOrchestrator()

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")
    console.print("\n🚀 Starting QADI analysis...")

    result = await orchestrator.run_qadi_cycle(
        user_input=problem,
        context="Focus on innovative, practical solutions with consideration for stakeholders and implementation challenges",
    )

    console.print("\n✅ QADI analysis completed")

    if result.total_llm_cost > 0:
        console.print(f"💰 Total LLM Cost: ${result.total_llm_cost:.4f}")

    # Display QADI phases
    phase_table = Table(title="QADI Analysis Results")
    phase_table.add_column("Phase", style="cyan")
    phase_table.add_column("Result", style="green")

    phase_emojis = {
        "question": "❓",
        "hypotheses": "💡", 
        "answer": "🔍",
        "verification": "🔗",
    }

    # Question Phase
    phase_table.add_row(
        f"{phase_emojis['question']} Question", 
        result.core_question[:80] + "..." if len(result.core_question) > 80 else result.core_question
    )

    # Hypotheses (Abduction)
    hypotheses_summary = f"{len(result.hypotheses)} hypotheses generated"
    if result.hypotheses and result.hypothesis_scores:
        # Zip hypotheses and scores to find the best one efficiently (O(N))
        best_hypothesis_item = max(
            zip(result.hypotheses, result.hypothesis_scores),
            key=lambda item: item[1].overall
        )
        best_hypothesis = best_hypothesis_item[0]
        hypotheses_summary += f"\nBest: {best_hypothesis[:60]}..."
    phase_table.add_row(f"{phase_emojis['hypotheses']} Hypotheses", hypotheses_summary)

    # Answer (Deduction)
    answer_text = result.final_answer[:80] + "..." if len(result.final_answer) > 80 else result.final_answer
    phase_table.add_row(f"{phase_emojis['answer']} Answer", answer_text)

    # Verification (Induction)
    verification_summary = f"{len(result.verification_examples)} examples"
    if result.verification_examples:
        verification_summary += f"\nFirst: {result.verification_examples[0][:60]}..."
    phase_table.add_row(f"{phase_emojis['verification']} Verification", verification_summary)

    console.print(phase_table)

    # Display detailed hypotheses with scores
    if result.hypotheses and result.hypothesis_scores:
        console.print("\n💡 Detailed Hypotheses:")
        for i, (hypothesis, score) in enumerate(zip(result.hypotheses, result.hypothesis_scores)):
            console.print(f"\n   H{i+1}. {hypothesis}")
            console.print(f"      📊 Overall Score: {score.overall:.3f}")
            console.print(f"      📈 Novelty: {score.novelty:.2f} | Impact: {score.impact:.2f} | Feasibility: {score.feasibility:.2f}")

    # Display action plan if available
    if result.action_plan:
        console.print("\n📋 Action Plan:")
        for i, action in enumerate(result.action_plan, 1):
            console.print(f"   {i}. {action}")

    # Display synthesis
    console.print(f"\n🎨 Synthesized Ideas ({len(result.synthesized_ideas)} total):")
    for i, idea in enumerate(result.synthesized_ideas[:5], 1):  # Show first 5
        method_name = getattr(idea.thinking_method, 'value', str(idea.thinking_method))
        console.print(f"  {i}. [{method_name}] {idea.content[:80]}...")

    if len(result.synthesized_ideas) > 5:
        console.print(f"  ... and {len(result.synthesized_ideas) - 5} more ideas")


async def demo_qadi_capabilities(problem: str) -> None:
    """Demonstrate QADI system capabilities and requirements."""
    console.print("\n" + "=" * 60)
    console.print("💪 QADI System Capabilities", style="bold blue")
    console.print("=" * 60)

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")

    # Check if we have API keys
    has_api_keys = any(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
        ]
    )

    if has_api_keys:
        console.print("🤖 API key detected - full QADI analysis available!")
        
        # Display capabilities
        capabilities_table = Table(title="QADI System Capabilities")
        capabilities_table.add_column("Feature", style="cyan")
        capabilities_table.add_column("Description", style="green")

        capabilities_table.add_row("Question Extraction", "Identifies the core question from any input")
        capabilities_table.add_row("Hypothesis Generation", "Creates 3 potential solutions/approaches")
        capabilities_table.add_row("Multi-criteria Evaluation", "Scores on Novelty, Impact, Cost, Feasibility, Risks")
        capabilities_table.add_row("Verification Examples", "Provides real-world validation")
        capabilities_table.add_row("Action Planning", "Generates concrete implementation steps")
        capabilities_table.add_row("Cost Tracking", "Monitors LLM usage and costs")

        console.print(capabilities_table)

    else:
        console.print("❌ No API keys detected - QADI analysis requires LLM access")
        console.print("\n📝 To enable full QADI analysis, set one of these environment variables:")
        console.print("  • GOOGLE_API_KEY=your_google_api_key")
        console.print("  • OPENAI_API_KEY=your_openai_api_key") 
        console.print("  • ANTHROPIC_API_KEY=your_anthropic_api_key")
        console.print("\n💡 Get API keys from the respective provider websites")


async def main() -> None:
    """Main demo function."""
    console.print(
        Panel.fit(
            "🚀 Mad Spark Alt - QADI Analysis Demo\n"
            "Simplified QADI Methodology\n"
            "Question → Abduction → Deduction → Induction",
            style="bold green",
        )
    )

    try:
        # Check API key availability
        has_api_keys = await check_api_keys()

        # Demo problems
        problems = [
            "How can we make public transportation more sustainable while improving user experience?",
            "What innovative approaches could reduce food waste in restaurants?",
            "How might AI enhance education without replacing human teachers?",
        ]

        for i, problem in enumerate(problems, 1):
            console.print(f"\n{'='*80}")
            console.print(
                (
                    f"📋 Demo {i}: {problem[:60]}..."
                    if len(problem) > 60
                    else f"📋 Demo {i}: {problem}"
                ),
                style="bold magenta",
            )
            console.print("=" * 80)

            # Setup demo
            await demo_qadi_setup()

            # Main QADI cycle demo
            if has_api_keys:
                await demo_qadi_cycle(problem)
            else:
                console.print("⚠️ Skipping QADI analysis - no API key available")

            # Capabilities demo (only for first problem)
            if i == 1:
                await demo_qadi_capabilities(problem)

        console.print("\n" + "=" * 80)
        console.print("🎉 QADI Demo completed successfully!", style="bold green")

        if has_api_keys:
            console.print(
                "🤖 You experienced AI-powered QADI analysis!",
                style="green",
            )
        else:
            console.print(
                "📝 You saw the QADI system structure. Set API keys for full analysis!",
                style="yellow",
            )

        console.print("\n💡 Next steps:", style="bold cyan")
        console.print("  • Set API keys for full QADI analysis")
        console.print("  • Try the CLI: uv run mad_spark_alt 'your question'")
        console.print("  • Try evolution: uv run mad_spark_alt 'your question' --evolve")
        console.print("  • Try multi-perspective: uv run python qadi_multi_perspective.py 'your question'")

    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}", style="bold red")
        raise


if __name__ == "__main__":
    asyncio.run(main())
