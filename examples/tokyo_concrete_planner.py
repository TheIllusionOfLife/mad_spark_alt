#!/usr/bin/env python3
"""
Modified user_test.py that forces concrete travel planning output.
"""

import asyncio
import os
import sys
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mad_spark_alt.core import SmartQADIOrchestrator

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()


async def generate_concrete_itinerary(destination: str, duration: str):
    """Generate ideas using the QADI system but with travel-focused prompts."""
    
    orchestrator = SmartQADIOrchestrator()
    await orchestrator.ensure_agents_ready()
    
    # Craft a problem statement that forces practical output
    problem = f"""Create a detailed {duration} travel itinerary for {destination}. 

For each day include:
- Morning activities with specific locations and times
- Recommended lunch spot with address
- Afternoon activities with transportation details  
- Dinner recommendation with cuisine type
- Estimated daily budget
- Practical tips

Be SPECIFIC with real place names, addresses, and transit stations."""

    context = """Generate ONLY concrete, actionable travel plans with:
- Real attraction names and addresses
- Specific restaurant recommendations
- Exact train/subway lines to use
- Realistic time allocations
- Current entrance fees and opening hours
- Practical tips for each location

NO philosophical discussions about travel.
NO abstract concepts about tourism.
ONLY specific, followable daily plans."""
    
    console.print(f"\n🗼 Generating concrete {duration} {destination} itinerary...", style="cyan")
    
    # Run the QADI cycle with travel-focused parameters
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context=context,
        cycle_config={
            "max_ideas_per_method": 2,  # Fewer but more detailed ideas
            "require_reasoning": False,  # Skip philosophical reasoning
        }
    )
    
    console.print(f"\n✅ Completed in {result.execution_time:.2f}s", style="green")
    
    if result.llm_cost > 0:
        console.print(f"💰 LLM Cost: ${result.llm_cost:.4f}", style="dim")
    
    # Find the most concrete ideas (likely from Deduction phase)
    console.print("\n📅 Generated Itinerary Ideas:\n", style="bold")
    
    # Sort ideas by concreteness (deduction and induction tend to be more practical)
    practical_ideas = []
    
    for phase_name in ['deduction', 'induction', 'abduction', 'questioning']:
        if phase_name in result.phases:
            phase_result = result.phases[phase_name]
            for idea in phase_result.generated_ideas:
                # Check if idea contains practical elements
                if any(keyword in idea.content.lower() for keyword in 
                       ['day', 'morning', 'station', 'address', 'restaurant', '¥', 'line', 'pm', 'am']):
                    practical_ideas.append(idea)
    
    if practical_ideas:
        console.print("🗾 Most Practical Itinerary Suggestions:", style="green bold")
        for i, idea in enumerate(practical_ideas[:3], 1):  # Show top 3 most practical
            console.print(f"\n--- Option {i} ---", style="cyan")
            console.print(idea.content)
            console.print()
    
    # Show all ideas grouped by phase for completeness
    console.print("\n" + "="*60 + "\n", style="dim")
    console.print("📊 All Generated Ideas by Thinking Phase:", style="bold")
    
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡", 
        "deduction": "🗺️",  # Map for practical planning
        "induction": "📍"   # Pin for specific locations
    }
    
    for phase_name, phase_result in result.phases.items():
        if phase_result.generated_ideas:
            emoji = phase_emojis.get(phase_name, "🧠")
            console.print(f"\n{emoji} {phase_name.title()} Phase:")
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                console.print(f"\n{i}. {idea.content}", style="cyan")


async def main():
    """Main entry point."""
    console.print(
        Panel.fit(
            "🗼 Concrete Tokyo Travel Planner\n"
            "Using QADI system with travel-focused prompts",
            style="bold blue",
        )
    )
    
    # Check API keys
    available_providers = []
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append("Google")
    
    if available_providers:
        console.print(f"✅ LLM Providers Available: {', '.join(available_providers)}", style="green")
    else:
        console.print("⚠️  No API keys found - results may be limited", style="yellow")
    
    # Get destination
    if len(sys.argv) > 1:
        destination = sys.argv[1]
        duration = sys.argv[2] if len(sys.argv) > 2 else "7 days"
    else:
        destination = Prompt.ask("\n🌍 Destination", default="Tokyo")
        duration = Prompt.ask("📅 Duration", default="7 days")
    
    await generate_concrete_itinerary(destination, duration)


if __name__ == "__main__":
    asyncio.run(main())