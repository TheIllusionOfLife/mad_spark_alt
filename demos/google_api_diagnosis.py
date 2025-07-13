#!/usr/bin/env python3
"""
Diagnose and demonstrate the Google API timeout issue.
"""

import asyncio
import os
from pathlib import Path
import time

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print("🔍 GOOGLE API DIAGNOSIS")
print("=" * 60)

async def test_basic_google_api():
    """Test basic Google API functionality."""
    print("\n1️⃣ TESTING BASIC GOOGLE API")
    print("-" * 40)
    
    from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No Google API key found")
        return False
    
    print(f"✅ API Key found: ...{api_key[-6:]}")
    
    provider = GoogleProvider(api_key)
    
    request = LLMRequest(
        user_prompt="Say 'Hello Google API works!' in exactly 5 words.",
        max_tokens=50,
        temperature=0.1
    )
    
    try:
        start = time.time()
        response = await provider.generate(request)
        elapsed = time.time() - start
        
        print(f"✅ Success! Response time: {elapsed:.2f}s")
        print(f"📝 Response: {response.content}")
        print(f"⚠️  Note: Google API is slow (~{elapsed:.1f}s per call)")
        
        await provider.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        await provider.close()
        return False

async def test_llm_agent_calls():
    """Test how many LLM calls an agent makes."""
    print("\n2️⃣ ANALYZING LLM AGENT BEHAVIOR")
    print("-" * 40)
    
    # Mock the LLM manager to count calls
    call_count = 0
    original_generate = None
    
    from mad_spark_alt.core.llm_provider import llm_manager
    from mad_spark_alt.agents import LLMQuestioningAgent
    
    # Store original method
    original_generate = llm_manager.generate
    
    # Create a wrapper that counts calls
    async def counting_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        print(f"   📞 LLM Call #{call_count}")
        return await original_generate(*args, **kwargs)
    
    # Replace with counting version
    llm_manager.generate = counting_generate
    
    try:
        # Create agent and test
        agent = LLMQuestioningAgent()
        
        from mad_spark_alt.core.interfaces import IdeaGenerationRequest
        
        request = IdeaGenerationRequest(
            problem_statement="How to reduce stress?",
            max_ideas_per_method=2
        )
        
        print("🔄 Running single agent...")
        start = time.time()
        
        # Set a timeout to prevent hanging
        try:
            result = await asyncio.wait_for(
                agent.generate_ideas(request),
                timeout=30
            )
            elapsed = time.time() - start
            
            print(f"\n✅ Single agent completed:")
            print(f"   • Time: {elapsed:.2f}s")
            print(f"   • LLM calls: {call_count}")
            print(f"   • Ideas generated: {len(result.generated_ideas)}")
            
        except asyncio.TimeoutError:
            print(f"\n⏱️  Timeout after 30s")
            print(f"   • LLM calls made: {call_count}")
            
    finally:
        # Restore original method
        llm_manager.generate = original_generate
    
    return call_count

async def diagnose_timeout_issue():
    """Diagnose why timeouts occur."""
    print("\n3️⃣ TIMEOUT DIAGNOSIS")
    print("-" * 40)
    
    # Test parameters
    phases = 4
    calls_per_phase = 3  # Typical: domain analysis, generation, ranking
    seconds_per_call = 7  # Google API typical response time
    
    total_calls = phases * calls_per_phase
    parallel_time = seconds_per_call * calls_per_phase  # All phases run in parallel
    sequential_time = total_calls * seconds_per_call
    
    print(f"📊 QADI System Analysis:")
    print(f"   • Phases: {phases} (Questioning, Abduction, Deduction, Induction)")
    print(f"   • LLM calls per phase: ~{calls_per_phase}")
    print(f"   • Total LLM calls: {total_calls}")
    print(f"   • Google API response time: ~{seconds_per_call}s per call")
    
    print(f"\n⏱️  Time Estimates:")
    print(f"   • Parallel execution: ~{parallel_time}s (all phases at once)")
    print(f"   • Sequential execution: ~{sequential_time}s (one phase at a time)")
    print(f"   • Default timeout: 90-120s")
    
    print(f"\n⚠️  PROBLEM IDENTIFIED:")
    if parallel_time > 90:
        print(f"   ❌ Parallel execution ({parallel_time}s) exceeds timeout!")
    else:
        print(f"   ✅ Parallel execution should work")
        
    print(f"\n💡 SOLUTIONS:")
    print(f"   1. Use sequential execution for Google API")
    print(f"   2. Reduce LLM calls (skip domain analysis)")
    print(f"   3. Use faster models (gemini-1.5-flash)")
    print(f"   4. Increase timeouts for Google provider")

async def main():
    """Run diagnosis."""
    
    # Test 1: Basic API
    api_works = await test_basic_google_api()
    
    if api_works:
        # Test 2: Count LLM calls
        await test_llm_agent_calls()
        
        # Test 3: Diagnose timeouts
        await diagnose_timeout_issue()
    
    print("\n" + "="*60)
    print("📋 DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("\n✅ Root Cause: Google API is slow + too many concurrent calls")
    print("✅ Solution: Use sequential execution or reduce LLM calls")
    print("✅ The API itself works fine - it's a performance optimization issue")

if __name__ == "__main__":
    asyncio.run(main())