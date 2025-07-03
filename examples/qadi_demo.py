#!/usr/bin/env python3
"""
Demo script for the QADI cycle idea generation system.

This script demonstrates the multi-agent idea generation system using
the "Shin Logical Thinking" QADI methodology.
"""

import asyncio
import json

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from mad_spark_alt.agents import (
    AbductionAgent,
    DeductionAgent,
    InductionAgent,
    QuestioningAgent,
)
from mad_spark_alt.core import (
    IdeaGenerationRequest,
    QADIOrchestrator,
    ThinkingMethod,
    agent_registry,
    register_agent,
)

console = Console()


def setup_agents():
    """Register all thinking method agents."""
    console.print("🔧 Setting up thinking method agents...", style="blue")

    # Clear registry first
    agent_registry.clear()

    # Register all agents
    register_agent(QuestioningAgent)
    register_agent(AbductionAgent)
    register_agent(DeductionAgent)
    register_agent(InductionAgent)

    registered_agents = agent_registry.list_agents()
    console.print(f"✅ Registered {len(registered_agents)} thinking agents:")

    for name, info in registered_agents.items():
        console.print(f"  • {name} ({info['thinking_method']})", style="green")


async def demo_individual_agent(problem: str):
    """Demonstrate individual agent capabilities."""
    console.print("\n" + "=" * 60)
    console.print("🧠 Individual Agent Demo", style="bold blue")
    console.print("=" * 60)

    questioning_agent = agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING)

    request = IdeaGenerationRequest(
        problem_statement=problem,
        context="Urban planning and sustainability focus",
        max_ideas_per_method=3,
        require_reasoning=True,
    )

    console.print(f"\n🤔 Questioning Agent analyzing: [italic]{problem}[/italic]")

    result = await questioning_agent.generate_ideas(request)

    console.print(f"\n📝 Generated {len(result.generated_ideas)} questions:")
    for i, idea in enumerate(result.generated_ideas, 1):
        console.print(f"\n{i}. {idea.content}")
        if idea.reasoning:
            console.print(f"   💭 Reasoning: {idea.reasoning}", style="dim")


async def demo_qadi_cycle(problem: str):
    """Demonstrate the complete QADI cycle."""
    console.print("\n" + "=" * 60)
    console.print("🔄 QADI Cycle Demo", style="bold blue")
    console.print("=" * 60)

    # Get agents from registry
    agents = [
        agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING),
        agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION),
        agent_registry.get_agent_by_method(ThinkingMethod.DEDUCTION),
        agent_registry.get_agent_by_method(ThinkingMethod.INDUCTION),
    ]

    orchestrator = QADIOrchestrator(agents)

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")
    console.print("\n🚀 Starting QADI cycle...")

    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context="Focus on practical, implementable solutions for urban environments",
        cycle_config={
            "max_ideas_per_method": 2,
            "require_reasoning": True,
            "creativity_level": "balanced",
        },
    )

    console.print(f"\n✅ QADI cycle completed in {result.execution_time:.2f}s")
    console.print(f"🔍 Cycle ID: {result.cycle_id}")

    # Display results for each phase
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡",
        "deduction": "🔍",
        "induction": "🔗",
    }

    for phase_name, phase_result in result.phases.items():
        emoji = phase_emojis.get(phase_name, "🧠")
        console.print(f"\n{emoji} {phase_name.title()} Phase:")

        if phase_result.error_message:
            console.print(f"   ❌ Error: {phase_result.error_message}", style="red")
        else:
            console.print(f"   Agent: {phase_result.agent_name}")
            console.print(f"   Generated {len(phase_result.generated_ideas)} ideas:")

            for i, idea in enumerate(phase_result.generated_ideas, 1):
                console.print(f"   {i}. {idea.content}")
                if idea.reasoning and len(idea.reasoning) < 100:
                    console.print(f"      💭 {idea.reasoning}", style="dim")

    # Display synthesized insights
    console.print(f"\n🎨 Synthesized Ideas ({len(result.synthesized_ideas)} total):")

    phase_ideas = {}
    for idea in result.synthesized_ideas:
        phase = idea.metadata.get("phase", "unknown")
        if phase not in phase_ideas:
            phase_ideas[phase] = []
        phase_ideas[phase].append(idea)

    for phase, ideas in phase_ideas.items():
        emoji = phase_emojis.get(phase, "🧠")
        console.print(f"\n{emoji} From {phase} phase ({len(ideas)} ideas):")
        for idea in ideas:
            console.print(f"  • {idea.content[:100]}...")


async def demo_parallel_generation(problem: str):
    """Demonstrate parallel generation."""
    console.print("\n" + "=" * 60)
    console.print("⚡ Parallel Generation Demo", style="bold blue")
    console.print("=" * 60)

    agents = [
        agent_registry.get_agent_by_method(ThinkingMethod.QUESTIONING),
        agent_registry.get_agent_by_method(ThinkingMethod.ABDUCTION),
    ]

    orchestrator = QADIOrchestrator(agents)

    console.print(f"\n🎯 Problem: [italic]{problem}[/italic]")
    console.print("\n⚡ Running parallel generation...")

    results = await orchestrator.run_parallel_generation(
        problem_statement=problem,
        thinking_methods=[ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
        context="Innovation and technology focus",
        config={"max_ideas_per_method": 2},
    )

    for method, result in results.items():
        emoji = "❓" if method == ThinkingMethod.QUESTIONING else "💡"
        console.print(f"\n{emoji} {method.value.title()} Results:")
        console.print(f"   Agent: {result.agent_name}")

        for i, idea in enumerate(result.generated_ideas, 1):
            console.print(f"   {i}. {idea.content}")


async def main():
    """Main demo function."""
    console.print(
        Panel.fit(
            "🚀 Mad Spark Alt - QADI Cycle Demo\n"
            "Multi-Agent Idea Generation System\n"
            "Based on 'Shin Logical Thinking' Methodology",
            style="bold green",
        )
    )

    # Setup
    setup_agents()

    # Demo problem
    problem = "How can we reduce urban air pollution while supporting economic growth?"

    try:
        # Individual agent demo
        await demo_individual_agent(problem)

        # Full QADI cycle demo
        await demo_qadi_cycle(problem)

        # Parallel generation demo
        await demo_parallel_generation(
            "How can AI improve healthcare accessibility in remote areas?"
        )

        console.print("\n" + "=" * 60)
        console.print("🎉 Demo completed successfully!", style="bold green")
        console.print("The QADI system is ready for idea generation!", style="green")

    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}", style="bold red")
        raise


if __name__ == "__main__":
    asyncio.run(main())
