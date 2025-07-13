#!/usr/bin/env python3
"""
Diagnostic Script for Multi-Agent Timeout Issues
Usage: uv run python diagnose_timeout.py

This script tests each component step-by-step to identify the timeout cause.
"""
import asyncio
import sys
import os
import time
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

def test_api_keys():
    """Test which API keys are available."""
    print("🔑 TESTING API KEYS:")
    print("-" * 50)
    
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print(f"Google API Key: {'✓ Found' if google_key else '❌ Missing'} ({len(google_key) if google_key else 0} chars)")
    print(f"OpenAI API Key: {'✓ Found' if openai_key else '❌ Missing'} ({len(openai_key) if openai_key else 0} chars)")
    print(f"Anthropic API Key: {'✓ Found' if anthropic_key else '❌ Missing'} ({len(anthropic_key) if anthropic_key else 0} chars)")
    
    return google_key, openai_key, anthropic_key

def test_imports():
    """Test all critical imports."""
    print("\n🔧 TESTING IMPORTS:")
    print("-" * 50)
    
    try:
        start = time.time()
        from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager
        print(f"✓ llm_provider imported ({time.time()-start:.2f}s)")
        
        start = time.time()
        from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
        print(f"✓ SmartQADIOrchestrator imported ({time.time()-start:.2f}s)")
        
        start = time.time()
        from mad_spark_alt.core.smart_registry import smart_registry
        print(f"✓ smart_registry imported ({time.time()-start:.2f}s)")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_provider_setup_individually(google_key, openai_key, anthropic_key):
    """Test provider setup one by one."""
    print("\n⚙️ TESTING PROVIDER SETUP (INDIVIDUAL):")
    print("-" * 50)
    
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    
    # Test Google only (like simple version)
    if google_key:
        try:
            print("Testing Google provider only...")
            start = time.time()
            await asyncio.wait_for(
                setup_llm_providers(google_api_key=google_key),
                timeout=15
            )
            print(f"✓ Google provider setup successful ({time.time()-start:.2f}s)")
        except asyncio.TimeoutError:
            print("❌ Google provider setup timed out")
        except Exception as e:
            print(f"❌ Google provider setup failed: {e}")
    
    # Test with all providers (like multi-agent version)
    try:
        print("\nTesting all providers together...")
        start = time.time()
        await asyncio.wait_for(
            setup_llm_providers(
                google_api_key=google_key,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key
            ),
            timeout=30
        )
        print(f"✓ All providers setup successful ({time.time()-start:.2f}s)")
    except asyncio.TimeoutError:
        print("❌ All providers setup timed out")
    except Exception as e:
        print(f"❌ All providers setup failed: {e}")

async def test_agent_setup():
    """Test smart orchestrator agent setup."""
    print("\n🧠 TESTING AGENT SETUP:")
    print("-" * 50)
    
    try:
        from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
        
        print("Creating SmartQADIOrchestrator...")
        start = time.time()
        orchestrator = SmartQADIOrchestrator(auto_setup=True)
        print(f"✓ Orchestrator created ({time.time()-start:.2f}s)")
        
        print("Testing agent readiness...")
        start = time.time()
        status = await asyncio.wait_for(
            orchestrator.ensure_agents_ready(),
            timeout=20
        )
        print(f"✓ Agents ready ({time.time()-start:.2f}s)")
        print(f"Agent status: {status}")
        
    except asyncio.TimeoutError:
        print("❌ Agent setup timed out")
    except Exception as e:
        print(f"❌ Agent setup failed: {e}")

async def test_simple_llm_call():
    """Test a simple LLM call like the working version."""
    print("\n🚀 TESTING SIMPLE LLM CALL:")
    print("-" * 50)
    
    try:
        from mad_spark_alt.core.llm_provider import llm_manager, LLMRequest
        
        print("Making simple LLM call...")
        start = time.time()
        
        request = LLMRequest(
            user_prompt="Test prompt: What is 2+2?",
            max_tokens=50,
            temperature=0.1
        )
        
        response = await asyncio.wait_for(
            llm_manager.generate(request),
            timeout=15
        )
        
        print(f"✓ LLM call successful ({time.time()-start:.2f}s)")
        print(f"Response: {response.content[:100]}...")
        print(f"Cost: ${response.cost:.4f}")
        
    except asyncio.TimeoutError:
        print("❌ LLM call timed out")
    except Exception as e:
        print(f"❌ LLM call failed: {e}")

async def run_full_diagnosis():
    """Run complete diagnostic sequence."""
    print("🩺 MAD SPARK ALT TIMEOUT DIAGNOSIS")
    print("=" * 70)
    
    # Step 1: Check API keys
    google_key, openai_key, anthropic_key = test_api_keys()
    
    if not google_key:
        print("\n❌ CRITICAL: No Google API key found!")
        print("The system needs at least Google API key to work.")
        return
    
    # Step 2: Test imports
    if not test_imports():
        print("\n❌ CRITICAL: Import failures detected!")
        return
    
    # Step 3: Test provider setup
    await test_provider_setup_individually(google_key, openai_key, anthropic_key)
    
    # Step 4: Test agent setup
    await test_agent_setup()
    
    # Step 5: Test simple LLM call
    await test_simple_llm_call()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("Check the results above to identify where the timeout occurs.")
    print("Most likely causes:")
    print("1. Provider setup hanging on missing/invalid API keys")
    print("2. Network timeout during provider initialization") 
    print("3. Agent setup taking too long")
    print("4. Async deadlock in setup process")

if __name__ == "__main__":
    print("Running comprehensive timeout diagnosis...")
    asyncio.run(run_full_diagnosis())