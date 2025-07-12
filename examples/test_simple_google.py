#!/usr/bin/env python3
"""
Simple test to check if Google API is working at all.
"""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print(f"Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")

# Test direct API call
import aiohttp

async def test_google_api():
    """Test Google API directly."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No Google API key found")
        return
        
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": "Say hello in 5 words or less"
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                json=payload,
                params={"key": api_key},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    print(f"✅ Google API works! Response: {text}")
                else:
                    error_text = await response.text()
                    print(f"❌ API Error {response.status}: {error_text}")
        except Exception as e:
            print(f"❌ Request failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_google_api())