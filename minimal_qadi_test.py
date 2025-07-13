#!/usr/bin/env python3
"""
Minimal QADI test to isolate the timeout issue
"""
import asyncio
import sys
import os
from pathlib import Path
import time

# Load .env
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_minimal_qadi():
    """Test the minimal QADI cycle to find where it hangs."""
    
    print("🧪 MINIMAL QADI TEST")
    print("=" * 50)
    
    # Step 1: Setup (we know this works from diagnostic)
    print("1. Setting up providers...", end='', flush=True)
    start = time.time()
    
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    google_key = os.getenv('GOOGLE_API_KEY')
    await setup_llm_providers(google_api_key=google_key)
    
    print(f" ✓ ({time.time()-start:.2f}s)")
    
    # Step 2: Create orchestrator 
    print("2. Creating orchestrator...", end='', flush=True)
    start = time.time()
    
    from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
    orchestrator = SmartQADIOrchestrator(auto_setup=True)
    
    print(f" ✓ ({time.time()-start:.2f}s)")
    
    # Step 3: Agent readiness
    print("3. Ensuring agents ready...", end='', flush=True)
    start = time.time()
    
    status = await orchestrator.ensure_agents_ready()
    
    print(f" ✓ ({time.time()-start:.2f}s)")
    print(f"   Status: {status}")
    
    # Step 4: Minimal QADI cycle
    print("4. Running minimal QADI cycle...", end='', flush=True)
    start = time.time()
    
    try:
        # Very simple config
        simple_config = {
            "max_ideas_per_method": 1,
            "require_reasoning": False
        }
        
        # Add timeout to identify hang
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle(
                problem_statement="What is the capital of France?",  # Simple question
                context=None,
                cycle_config=simple_config
            ),
            timeout=30  # 30 second timeout
        )
        
        print(f" ✓ ({time.time()-start:.2f}s)")
        print(f"   Phases completed: {len(result.phases)}")
        print(f"   Ideas generated: {len(result.synthesized_ideas)}")
        print(f"   LLM cost: ${result.llm_cost:.4f}")
        
        # Show one result from each phase
        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                print(f"   {phase_name}: {phase_result.generated_ideas[0].content[:50]}...")
        
        print("\n✅ MINIMAL TEST SUCCESSFUL!")
        print("The issue is not in the core QADI system.")
        
    except asyncio.TimeoutError:
        print(f" ❌ TIMEOUT after 30s")
        print("   The hang occurs during run_qadi_cycle()")
        print("   This is the root cause of the timeout issue.")
        
    except Exception as e:
        print(f" ❌ ERROR: {e}")

if __name__ == "__main__":
    print("Testing minimal QADI cycle to isolate timeout...")
    asyncio.run(test_minimal_qadi())