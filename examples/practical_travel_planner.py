#!/usr/bin/env python3
"""
Practical travel planner that generates concrete itineraries.
Uses a single LLM agent to create actionable travel plans.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from mad_spark_alt.core import (
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ThinkingMethod,
    OutputType,
)
from mad_spark_alt.agents.questioning.llm_agent import LLMQuestioningAgent
from mad_spark_alt.core.llm_provider import LLMProvider

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()


async def generate_travel_plan(destination: str, duration: str):
    """Generate a practical travel itinerary using a single LLM agent."""
    
    # Determine which LLM provider to use
    if os.getenv("GOOGLE_API_KEY"):
        provider = LLMProvider.GOOGLE
    elif os.getenv("OPENAI_API_KEY"):
        provider = LLMProvider.OPENAI
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = LLMProvider.ANTHROPIC
    else:
        console.print("❌ No API keys found. Please set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY", style="red")
        return
    
    console.print(f"✅ Using {provider.value} API\n", style="green")
    
    # Create a practical prompt that forces concrete output
    problem_statement = f"Create a detailed {duration} itinerary for {destination}"
    
    context = """You are a professional travel planner. Create a SPECIFIC, ACTIONABLE itinerary with:

For each day:
- Day number and theme
- Morning (9am-12pm): 2 specific attractions with addresses and nearest stations
- Lunch: Specific restaurant name, location, and must-try dish
- Afternoon (1pm-5pm): 2 more attractions with practical details
- Dinner: Restaurant recommendation with cuisine type
- Evening: Optional activity
- Transportation: Which trains/subways to use
- Daily budget estimate in local currency
- Practical tips

Be SPECIFIC - use real place names, actual restaurants, correct addresses, and realistic timings.
Format as a clear, structured itinerary ready to follow."""

    try:
        # Create an LLM agent directly
        agent = LLMQuestioningAgent(preferred_provider=provider)
        
        # Create request
        request = IdeaGenerationRequest(
            problem_statement=problem_statement,
            context=context,
            max_ideas_per_method=1,  # We want one comprehensive plan
            require_reasoning=False,  # Skip the philosophical reasoning
        )
        
        console.print(f"🗺️ Generating your {duration} {destination} itinerary...\n", style="cyan")
        
        # Generate the plan
        result: IdeaGenerationResult = await agent.generate_ideas(request)
        
        if result.generated_ideas:
            # Display the itinerary
            itinerary = result.generated_ideas[0].content
            console.print(Panel(
                Markdown(itinerary), 
                title=f"✈️ Your {duration} {destination} Itinerary", 
                border_style="blue"
            ))
            
            # Show cost if available
            if result.generation_metadata.get("llm_cost"):
                console.print(f"\n💰 Generation cost: ${result.generation_metadata['llm_cost']:.4f}", style="dim")
        else:
            console.print("❌ Failed to generate itinerary", style="red")
            if result.error_message:
                console.print(f"Error: {result.error_message}", style="red")
                
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


async def main():
    """Main entry point."""
    console.print(Panel.fit(
        "✈️ Practical Travel Planner\nGet real, actionable travel itineraries",
        style="bold blue"
    ))
    
    # Examples
    console.print("\n📍 Examples:", style="cyan")
    console.print("  • 7-day Tokyo")
    console.print("  • 5-day Paris") 
    console.print("  • Weekend in New York")
    console.print("  • 10-day Japan")
    
    # Get user input
    destination = console.input("\n🌍 Where do you want to go? ")
    duration = console.input("📅 For how long? ")
    
    await generate_travel_plan(destination, duration)


if __name__ == "__main__":
    asyncio.run(main())