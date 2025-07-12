#!/usr/bin/env python3
"""Test Google LLM directly to bypass the complex system."""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

async def test_google_llm_direct():
    """Test Google LLM with minimal overhead."""
    print("Testing Google LLM directly...")
    
    # Import after env is loaded
    from mad_spark_alt.core.llm_provider import (
        GoogleProvider,
        LLMRequest,
        ModelConfig,
        LLMProvider,
        ModelSize
    )
    
    # Create provider directly
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No Google API key found")
        return
        
    provider = GoogleProvider(api_key)
    
    # Create a simple request
    request = LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in exactly 5 words.",
        max_tokens=100,
        temperature=0.7,
        model_configuration=ModelConfig(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-1.5-flash",
            model_size=ModelSize.MEDIUM,
            input_cost_per_1k=0.000075,
            output_cost_per_1k=0.0003,
            max_tokens=100000
        )
    )
    
    try:
        print("Sending request...")
        response = await provider.generate(request)
        print(f"✅ Success! Response: {response.content}")
        print(f"💰 Cost: ${response.cost:.6f}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_google_llm_direct())