#!/usr/bin/env python3
"""
Simple Tokyo travel planner using Google Gemini directly.
"""

import os
import google.generativeai as genai
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()

def generate_tokyo_itinerary():
    """Generate a concrete Tokyo travel itinerary."""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("❌ No Google API key found. Please set GOOGLE_API_KEY", style="red")
        return
    
    console.print(Panel.fit(
        "🗼 Tokyo Travel Planner\nGenerating your 7-day itinerary...",
        style="bold blue"
    ))
    
    # Configure Google Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create prompt for concrete itinerary
    prompt = """Create a detailed 7-day Tokyo sightseeing itinerary. For each day include:

**Day X: [Theme]**

**Morning (9am-12pm):**
- [Attraction 1] - Address, nearest station, what to see
- [Attraction 2] - Address, nearest station, what to see

**Lunch (12pm-1pm):**
- Restaurant name and location
- Must-try dish and price range

**Afternoon (1pm-5pm):**
- [Attraction 3] - Address, nearest station, what to see
- [Attraction 4] - Address, nearest station, what to see

**Dinner (6pm-8pm):**
- Restaurant recommendation
- Type of cuisine and price range

**Evening (8pm onwards):**
- Optional activity or area to explore

**Transportation:**
- Which train/subway lines to use
- Consider getting day pass if multiple trips

**Budget:** ¥X,XXX for the day
**Tips:** Any specific advice for that day

Include:
Day 1: Traditional Tokyo (Asakusa, Senso-ji, Tokyo Skytree)
Day 2: Modern Tokyo (Shibuya, Harajuku, Omotesando)
Day 3: Culture & Gardens (Meiji Shrine, Imperial Palace, Ginza)
Day 4: Tech & Anime (Akihabara, TeamLab, Odaiba)
Day 5: Day trip to Nikko or Kamakura
Day 6: Food & Markets (Tsukiji Outer Market, cooking class, depachika)
Day 7: Mt. Fuji area or relax in Tokyo

Be specific with actual place names, stations, and realistic timings."""

    try:
        # Generate content
        response = model.generate_content(prompt)
        
        if response.text:
            # Display the itinerary
            console.print("\n")
            console.print(Panel(Markdown(response.text), title="🇯🇵 Your 7-Day Tokyo Itinerary", border_style="blue"))
        else:
            console.print("❌ Failed to generate itinerary", style="red")
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")

if __name__ == "__main__":
    generate_tokyo_itinerary()