#!/usr/bin/env python3
"""
Direct Tokyo travel planning using LLM without QADI methodology.
"""

import asyncio
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from mad_spark_alt.core.llm_provider import LLMProvider, LLMRequest, llm_manager, setup_llm_providers

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()

async def generate_tokyo_itinerary():
    """Generate a concrete Tokyo travel itinerary."""
    
    # Check for API keys
    if os.getenv("GOOGLE_API_KEY"):
        provider = LLMProvider.GOOGLE
    elif os.getenv("OPENAI_API_KEY"):
        provider = LLMProvider.OPENAI
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = LLMProvider.ANTHROPIC
    else:
        console.print("❌ No API keys found. Please set GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY", style="red")
        return
    
    console.print(f"✅ Using {provider.value} API", style="green")
    
    # Create prompt for concrete itinerary
    prompt = """Create a detailed 7-day Tokyo sightseeing itinerary. For each day include:

Day format:
- Theme for the day
- Morning (9am-12pm): Specific attractions with addresses
- Lunch recommendation with restaurant name and specialty
- Afternoon (1pm-5pm): More attractions 
- Evening (6pm-9pm): Dinner spot and optional evening activity
- Transportation: Which train/subway lines to use
- Budget estimate for the day
- Tips and notes

Include a mix of:
- Must-see landmarks (Senso-ji, Tokyo Tower, Meiji Shrine)
- Cultural experiences (tea ceremony, sumo, kabuki)
- Modern attractions (TeamLab, Shibuya crossing)
- Food experiences (tsukiji, ramen, sushi)
- Shopping areas (Harajuku, Ginza)
- Day trips (Nikko, Kamakura, Mt Fuji)

Make it practical and specific with actual place names, not generic suggestions."""

    try:
        # Make direct LLM call
        request = LLMRequest(
            provider=provider,
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=4000
        )
        
        console.print("\n🗼 Generating your Tokyo itinerary...\n", style="cyan")
        
        response = await llm_manager.complete(request)
        
        if response.content:
            # Display the itinerary
            console.print(Panel(Markdown(response.content), title="🇯🇵 Your 7-Day Tokyo Itinerary", border_style="blue"))
            
            # Show cost
            console.print(f"\n💰 Generation cost: ${response.usage.estimated_cost:.4f}", style="dim")
        else:
            console.print("❌ Failed to generate itinerary", style="red")
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")

async def main():
    """Main entry point."""
    console.print(Panel.fit(
        "🗼 Tokyo Travel Planner\nDirect itinerary generation without QADI",
        style="bold blue"
    ))
    
    await generate_tokyo_itinerary()

if __name__ == "__main__":
    asyncio.run(main())