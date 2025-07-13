#!/usr/bin/env python3
"""
Better idea generator that combines QADI methodology with practical output.

Usage: uv run python better_idea_generator.py "your question"
"""

import asyncio
import sys
import os
from pathlib import Path

# Load .env file for API keys
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Import the generate_ideas function
sys.path.append(os.path.dirname(__file__))
from generate_ideas import call_gemini_reliable

async def generate_better_ideas(prompt: str):
    """Generate ideas using QADI with better answer extraction."""
    
    print("🚀 Enhanced QADI Idea Generator")
    print("=" * 60)
    print(f"💡 Your Challenge: {prompt}")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ No Google API key found")
        return
    
    print("🤖 Using Gemini with Enhanced QADI methodology")
    
    # Enhanced prompts for better direct answers
    phases = [
        {
            "name": "Critical Questions",
            "emoji": "❓",
            "prompt": f"What are 3 critical questions to ask when approaching: {prompt}\n\nFocus on practical considerations and feasibility. Format as numbered list."
        },
        {
            "name": "Direct Solutions", 
            "emoji": "💡",
            "prompt": f"Provide 5 specific, actionable solutions for: {prompt}\n\nBe concrete and practical. Each solution should be implementable. Format as numbered list with brief explanations."
        },
        {
            "name": "Implementation Steps",
            "emoji": "🔍", 
            "prompt": f"What are the key implementation steps for: {prompt}\n\nProvide a clear action plan with specific milestones. Format as numbered steps."
        },
        {
            "name": "Success Strategies",
            "emoji": "🎯",
            "prompt": f"What strategies ensure success for: {prompt}\n\nFocus on critical success factors and potential obstacles. Format as numbered list."
        }
    ]
    
    print("\n📊 ENHANCED QADI ANALYSIS:")
    print("-" * 60)
    
    all_ideas = []
    
    for i, phase in enumerate(phases):
        if i > 0:
            await asyncio.sleep(1)  # Rate limiting
            
        print(f"\n{phase['emoji']} {phase['name'].upper()}:")
        
        response = await call_gemini_reliable(phase['prompt'])
        
        if response and not response.startswith(("Error:", "API Error", "Timeout")):
            print(response)
            all_ideas.append({
                'phase': phase['name'],
                'content': response
            })
        else:
            print(f"   ⚠️ {response}")
    
    # Synthesize final recommendations
    if all_ideas:
        print("\n" + "=" * 60)
        print("🎯 SYNTHESIZED RECOMMENDATIONS:")
        print("-" * 60)
        
        synthesis_prompt = f"""Based on the QADI analysis for "{prompt}", provide 5 concrete, actionable recommendations that someone could start implementing immediately. 

Consider all the questions, solutions, steps, and strategies discussed. Format as a numbered list with specific actions."""
        
        final_response = await call_gemini_reliable(synthesis_prompt)
        
        if final_response and not final_response.startswith(("Error:", "API Error")):
            print(final_response)
        else:
            print("Unable to synthesize recommendations")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete! QADI methodology applied with practical output.")

def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("🚀 Enhanced QADI Idea Generator")
        print("\n📖 Usage:")
        print('  uv run python better_idea_generator.py "your challenge"')
        print("\n🎯 Examples:")
        print('  • "How to reduce costs in manufacturing?"')
        print('  • "What are ways to improve team collaboration?"')
        print('  • "How to launch a successful startup?"')
        sys.exit(1)
    
    prompt = " ".join(sys.argv[1:])
    asyncio.run(generate_better_ideas(prompt))

if __name__ == "__main__":
    main()