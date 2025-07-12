#!/usr/bin/env python3
"""Quick test to verify Google API integration."""

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

# Test Google API
print("🔍 Testing Google API Integration...")
print(f"✅ Google API Key: {'SET' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")

# Try to import and test the LLM provider
try:
    from mad_spark_alt.core.llm_provider import LLMProvider, LLMProviderType
    
    provider = LLMProvider(provider_type=LLMProviderType.GOOGLE)
    print("✅ Google LLM Provider initialized successfully!")
    
    # Test a simple call
    print("\n🤖 Testing API call...")
    import asyncio
    
    async def test_call():
        try:
            response = await provider.generate("Say 'Hello from Google Gemini!' in 5 words or less")
            print(f"✅ Response: {response}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    success = asyncio.run(test_call())
    
    if success:
        print("\n🎉 Google API is working perfectly!")
        print("You can now use the full QADI system with AI-powered idea generation.")
    
except Exception as e:
    print(f"❌ Error setting up provider: {e}")
    print("\nPlease make sure your Google API key is valid and has access to Gemini API.")