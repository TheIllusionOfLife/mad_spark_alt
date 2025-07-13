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

from mad_spark_alt.core import (
    IdeaGenerationRequest,
    SmartQADIOrchestrator,
    RobustQADIOrchestrator,
    ThinkingMethod,
)

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()


async def check_api_keys():
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
        console.print("🚀 LLM providers detected! Using intelligent AI-powered agents.", style="green")
    else:
        console.print("📝 No API keys found. Using template-based agents.", style="yellow")
        console.print("💡 Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for AI-powered generation.", style="dim")
    
    return has_api_keys


async def demo_agent_setup(orchestrator: SmartQADIOrchestrator):
    """Demonstrate the smart agent setup process."""
    console.print("\n" + "=" * 60)
    console.print("🔧 Smart Agent Setup Demo", style="bold blue")
    console.print("=" * 60)

    console.print("Setting up intelligent agents with automatic LLM preference...")
    
    setup_status = await orchestrator.ensure_agents_ready()
    
    # Display setup results
    setup_table = Table(title="Agent Setup Results")
    setup_table.add_column("Thinking Method", style="cyan")
    setup_table.add_column("Status", style="green")
    
    for method, status in setup_status.items():
        if "LLM" in status:
            setup_table.add_row(method.title(), f"🤖 {status}")
        else:
            setup_table.add_row(method.title(), f"📝 {status}")
    
    console.print(setup_table)
    
    # Show agent status
    agent_status = orchestrator.get_agent_status()
    console.print(f"\n✅ Setup completed: {agent_status['setup_completed']}")
    
    return setup_status


async def demo_smart_qadi_cycle(problem: str):
    """Demonstrate the smart QADI cycle with LLM agents."""
    console.print("\n" + "=" * 60)
    console.print("🔄 Smart QADI Cycle Demo", style="bold blue")
    console.print("=" * 60)

    orchestrator = RobustQADIOrchestrator()

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")
    console.print("\n🚀 Starting intelligent QADI cycle...")

    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context="Focus on innovative, practical solutions with consideration for stakeholders and implementation challenges",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
            "creativity_level": "high",
        },
    )

    console.print(f"\n✅ Smart QADI cycle completed in {result.execution_time:.2f}s")
    console.print(f"🔍 Cycle ID: {result.cycle_id}")
    
    if result.llm_cost > 0:
        console.print(f"💰 Total LLM Cost: ${result.llm_cost:.4f}")

    # Display agent types used
    agent_table = Table(title="Agents Used in QADI Cycle")
    agent_table.add_column("Phase", style="cyan")
    agent_table.add_column("Agent Type", style="green")
    agent_table.add_column("Ideas Generated", style="yellow")
    
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡",
        "deduction": "🔍",
        "induction": "🔗",
    }

    for phase_name, agent_type in result.agent_types.items():
        phase_result = result.phases.get(phase_name)
        idea_count = len(phase_result.generated_ideas) if phase_result else 0
        emoji = phase_emojis.get(phase_name, "🧠")
        
        agent_display = f"🤖 {agent_type}" if "LLM" in agent_type else f"📝 {agent_type}"
        agent_table.add_row(f"{emoji} {phase_name.title()}", agent_display, str(idea_count))
    
    console.print(agent_table)

    # Display detailed results for each phase
    for phase_name, phase_result in result.phases.items():
        emoji = phase_emojis.get(phase_name, "🧠")
        agent_type = result.agent_types.get(phase_name, "unknown")
        
        console.print(f"\n{emoji} {phase_name.title()} Phase ({agent_type}):")

        if phase_result.error_message:
            console.print(f"   ❌ Error: {phase_result.error_message}", style="red")
        else:
            console.print(f"   Agent: {phase_result.agent_name}")

            for i, idea in enumerate(phase_result.generated_ideas, 1):
                console.print(f"\n   {i}. {idea.content}")
                
                # Show reasoning for LLM agents (usually more detailed)
                if idea.reasoning and "LLM" in agent_type:
                    reasoning = idea.reasoning[:200] + "..." if len(idea.reasoning) > 200 else idea.reasoning
                    console.print(f"      💭 Reasoning: {reasoning}", style="dim")
                    
                # Show confidence for LLM agents
                if hasattr(idea, 'confidence_score') and idea.confidence_score:
                    console.print(f"      📊 Confidence: {idea.confidence_score:.2f}", style="dim")
                    
                # Show cost for LLM ideas
                if "llm_cost" in idea.metadata and idea.metadata["llm_cost"] > 0:
                    console.print(f"      💰 Cost: ${idea.metadata['llm_cost']:.4f}", style="dim")

    # Display synthesis
    console.print(f"\n🎨 Synthesized Ideas ({len(result.synthesized_ideas)} total):")
    
    for i, idea in enumerate(result.synthesized_ideas[:6], 1):  # Show first 6
        phase = idea.metadata.get("phase", "unknown")
        emoji = phase_emojis.get(phase, "🧠")
        console.print(f"  {i}. [{emoji} {phase}] {idea.content[:80]}...")
        
    if len(result.synthesized_ideas) > 6:
        console.print(f"  ... and {len(result.synthesized_ideas) - 6} more ideas")


async def demo_agent_comparison(problem: str):
    """Demonstrate template vs LLM agent comparison."""
    console.print("\n" + "=" * 60)
    console.print("⚔️ Template vs LLM Agent Comparison", style="bold blue")
    console.print("=" * 60)

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")
    
    # Check if we can demo LLM agents
    has_api_keys = any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GOOGLE_API_KEY")
    ])
    
    if has_api_keys:
        console.print("🤖 Running comparison between template and LLM agents...")
        
        # Create two orchestrators - one smart, one basic
        smart_orchestrator = RobustQADIOrchestrator()
        
        # Run smart cycle (LLM preferred)
        smart_result = await smart_orchestrator.run_qadi_cycle(
            problem_statement=problem,
            cycle_config={"max_ideas_per_method": 2}
        )
        
        # Display comparison
        comparison_table = Table(title="Agent Performance Comparison")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Smart QADI", style="green")
        
        comparison_table.add_row("Execution Time", f"{smart_result.execution_time:.2f}s")
        comparison_table.add_row("Total Ideas", str(len(smart_result.synthesized_ideas)))
        comparison_table.add_row("LLM Cost", f"${smart_result.llm_cost:.4f}")
        comparison_table.add_row("Agent Types", ", ".join(set(smart_result.agent_types.values())))
        
        console.print(comparison_table)
        
    else:
        console.print("📝 No API keys available - showing template agent capabilities...")
        console.print("💡 Set API keys to see LLM agent comparison!", style="yellow")


async def main():
    """Main demo function."""
    console.print(
        Panel.fit(
            "🚀 Mad Spark Alt - Smart QADI Demo\n"
            "Intelligent Multi-Agent Idea Generation\n"
            "Automatic LLM Agent Preference & Fallback",
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
            "How might AI enhance education without replacing human teachers?"
        ]

        for i, problem in enumerate(problems, 1):
            console.print(f"\n{'='*80}")
            console.print(f"📋 Demo {i}: {problem[:60]}..." if len(problem) > 60 else f"📋 Demo {i}: {problem}", style="bold magenta")
            console.print('='*80)

            # Setup demo
            orchestrator = RobustQADIOrchestrator()
            await demo_agent_setup(orchestrator)

            # Main QADI cycle demo
            await demo_smart_qadi_cycle(problem)

            # Agent comparison (only for first problem)
            if i == 1:
                await demo_agent_comparison(problem)

        console.print("\n" + "=" * 80)
        console.print("🎉 Smart QADI Demo completed successfully!", style="bold green")
        
        if has_api_keys:
            console.print("🤖 You experienced AI-powered intelligent idea generation!", style="green")
        else:
            console.print("📝 You saw template-based generation. Try with API keys for AI power!", style="yellow")
        
        console.print("\n💡 Next steps:", style="bold cyan")
        console.print("  • Set API keys for LLM-powered agents")
        console.print("  • Try the CLI: uv run mad-spark evaluate 'your idea'")
        console.print("  • Explore the Python API for custom applications")

    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}", style="bold red")
        raise


if __name__ == "__main__":
    asyncio.run(main())
