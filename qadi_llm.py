#!/usr/bin/env python3
"""
Simple QADI with LLM - ensures proper setup.
Usage: uv run python qadi_llm.py "Your question"
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

async def main(prompt: str):
    """Run QADI with proper LLM setup."""
    # Import and setup
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.core.smart_registry import setup_smart_agents
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    
    # Check for API keys
    if not any([os.getenv('GOOGLE_API_KEY'), os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]):
        print("❌ No API keys found. Add to .env file.")
        return
    
    print(f"📝 {prompt}\n")
    
    # 1. Setup LLM providers
    print("Setting up LLM providers...", end='', flush=True)
    await setup_llm_providers(
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    print(" ✓")
    
    # 2. Setup smart agents
    print("Setting up QADI agents...", end='', flush=True)
    setup_smart_agents()
    print(" ✓")
    
    # 3. Run QADI
    print("\nRunning QADI analysis...")
    orchestrator = EnhancedQADIOrchestrator()
    
    try:
        # Use shorter timeout and simplified config
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle_with_answers(
                problem_statement=prompt,
                max_answers=3,
                cycle_config={
                    'max_ideas_per_method': 1,  # Minimal ideas
                    'parallel_execution': False  # Sequential
                }
            ),
            timeout=45.0  # 45 second timeout
        )
        
        print(f"\n✅ Completed in {result.execution_time:.1f}s")
        print(f"💰 Cost: ${result.llm_cost:.4f}")
        print(f"🤖 Agents: {list(set(result.agent_types.values()))}")
        
        if result.extracted_answers:
            print(f"\n📋 ANSWERS:")
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   ({answer.source_phase})")
                
    except asyncio.TimeoutError:
        print("\n⏱️ Timed out. Google API can be slow.")
        print("Try: 1) Simpler questions, 2) OpenAI/Anthropic API keys")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_llm.py "Your question"')
    else:
        asyncio.run(main(sys.argv[1]))