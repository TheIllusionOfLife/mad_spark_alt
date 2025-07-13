#!/usr/bin/env python3
"""
Simple QADI with direct Google API integration.
Usage: uv run python qadi_simple.py "Your question here"
"""
import asyncio
import sys
import os
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_google_api():
    """Test if Google API works at all."""
    from mad_spark_alt.core.llm_provider import UnifiedLLMProvider
    
    provider = UnifiedLLMProvider(provider_type="google")
    
    # Simple test call
    messages = [
        {"role": "user", "content": "Say 'Hello, QADI is working!' in exactly 5 words."}
    ]
    
    response = await provider.generate(messages, temperature=0.1)
    return response.get("content", "No response")

async def run_simple_qadi(prompt: str):
    """Run a simplified QADI cycle."""
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ No Google API key found")
        return
    
    print(f"📝 {prompt}\n")
    
    # First test if API works
    print("Testing Google API...", end='', flush=True)
    try:
        test_response = await test_google_api()
        print(f" ✓ ({test_response})")
    except Exception as e:
        print(f" ❌ API Error: {e}")
        return
    
    # Now run simplified QADI
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    
    print("\nRunning QADI analysis...")
    
    # Force template agents to avoid timeouts
    for key in ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
        if key in os.environ:
            del os.environ[key]
    
    orchestrator = EnhancedQADIOrchestrator()
    result = await orchestrator.run_qadi_cycle_with_answers(
        problem_statement=prompt,
        max_answers=5,
        cycle_config={'max_ideas_per_method': 2}
    )
    
    if result.extracted_answers:
        print("\n✅ ANSWERS:")
        for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
            print(f"\n{i}. {answer.content}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_simple.py "Your question"')
    else:
        asyncio.run(run_simple_qadi(sys.argv[1]))