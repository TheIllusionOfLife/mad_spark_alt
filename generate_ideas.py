#!/usr/bin/env python3
"""
Generate real ideas using AI
Usage: uv run python generate_ideas.py "your custom prompt here"
"""

import asyncio
import sys
import os
from pathlib import Path
import aiohttp

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

async def call_gemini_reliable(prompt: str, max_retries: int = 2) -> str:
    """Reliable call to Gemini API with retries"""
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return "No API key found"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.8,
                    "maxOutputTokens": 2048,  # Higher for 2.5-flash internal reasoning
                    "topK": 30,
                    "topP": 0.9
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=20)  # Shorter timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'candidates' in data and data['candidates']:
                            candidate = data['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                parts = candidate['content']['parts']
                                if parts and 'text' in parts[0]:
                                    return parts[0]['text'].strip()
                        return "Empty response"
                    else:
                        if attempt == max_retries - 1:
                            return f"API Error {response.status}"
                        await asyncio.sleep(2)  # Wait before retry
                        
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                return "Timeout"
            await asyncio.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)[:50]}"
            await asyncio.sleep(1)
    
    return "Failed after retries"

async def generate_working_ideas(prompt: str):
    """Generate real ideas using reliable AI calls"""
    
    print("🚀 Mad Spark Alt - AI Idea Generator")
    print("=" * 65)
    print(f"💡 Challenge: {prompt}")
    print("=" * 65)
    
    # Check API status
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        print("💡 Please add your Google API key to the .env file")
        return False
    
    print(f"🤖 Using Gemini 2.5-flash (QADI methodology)")
    
    # Simple, targeted prompts for better reliability
    phases = [
        {
            "emoji": "❓",
            "name": "Key Questions",
            "prompt": f"List 3 important questions to consider for: {prompt}\n\nBe specific and practical. Start each with '- '"
        },
        {
            "emoji": "💡", 
            "name": "Creative Concepts",
            "prompt": f"List 3 creative, specific ideas for: {prompt}\n\nBe innovative and detailed. Start each with '- '"
        },
        {
            "emoji": "🔍",
            "name": "Implementation Steps",
            "prompt": f"List 3 concrete ways to implement: {prompt}\n\nBe actionable and specific. Start each with '- '"
        },
        {
            "emoji": "🔗",
            "name": "Success Factors",
            "prompt": f"List 3 key factors for success with: {prompt}\n\nThink strategically. Start each with '- '"
        }
    ]
    
    print(f"\n🌟 AI-Generated Ideas:")
    print("-" * 65)
    
    total_ideas = 0
    
    # Process each phase with delays to avoid rate limits
    for i, phase in enumerate(phases):
        if i > 0:
            await asyncio.sleep(1.5)  # Delay between phases
            
        print(f"\n{phase['emoji']} {phase['name'].upper()}:")
        
        response = await call_gemini_reliable(phase['prompt'])
        
        if response and not response.startswith(("Error:", "API Error", "Timeout", "Failed", "No API", "Empty")):
            # Extract bullet points or numbered items
            lines = response.split('\n')
            item_num = 1
            
            for line in lines:
                line = line.strip()
                
                # Look for bullet points, dashes, or numbered items
                if line.startswith(('-', '•', '*', '1.', '2.', '3.')) or (line and line[0].isdigit() and '.' in line[:3]):
                    # Clean up the line
                    if line.startswith(('-', '•', '*')):
                        content = line[1:].strip()
                    elif line[0].isdigit() and '.' in line[:3]:
                        content = line.split('.', 1)[1].strip()
                    else:
                        content = line
                    
                    if content and len(content) > 10:
                        print(f"{item_num}. {content}")
                        item_num += 1
                        total_ideas += 1
                        
                        if item_num > 3:
                            break
            
            # Fallback: if no bullet points found, extract meaningful sentences
            if item_num == 1:
                sentences = response.replace('\n', ' ').split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20 and item_num <= 3:
                        print(f"{item_num}. {sentence}")
                        item_num += 1
                        total_ideas += 1
        else:
            print(f"   ⚠️ {response}")
    
    print("\n" + "=" * 65)
    if total_ideas > 0:
        print(f"✅ Generated {total_ideas} AI-powered ideas!")
        print("🎯 QADI methodology applied: Questions → Ideas → Steps → Success")
    else:
        print("⚠️ Unable to generate ideas. Please check your connection.")
    
    return total_ideas > 0

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("🚀 Mad Spark Alt - AI Idea Generator")
        print("\n📖 Usage:")
        print('  uv run python generate_ideas.py "your challenge"')
        print("\n🎯 Examples:")
        print('  • "Revolutionary mobile game concept"')
        print('  • "Make remote work more engaging"') 
        print('  • "Reduce plastic waste in restaurants"')
        print('  • "Innovative fundraising for nonprofits"')
        print('  • "AI-powered learning tools for students"')
        
        # Check API key status
        api_key = os.getenv('GOOGLE_API_KEY')
        print(f"\n🔑 API Status: {'✅ Ready' if api_key else '❌ Missing GOOGLE_API_KEY in .env'}")
        
        sys.exit(1)
    
    # Get the prompt from command line arguments
    prompt = " ".join(sys.argv[1:])
    
    # Run the idea generation
    print("⚡ Starting AI idea generation...")
    success = asyncio.run(generate_working_ideas(prompt))
    
    if success:
        print(f"\n✨ Success! Try another challenge to generate more ideas.")
    else:
        print(f"\n🔧 Troubleshooting: Check .env file has GOOGLE_API_KEY set")

if __name__ == "__main__":
    main()