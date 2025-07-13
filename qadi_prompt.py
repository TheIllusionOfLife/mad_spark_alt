#!/usr/bin/env python3
"""
Run QADI analysis with a custom prompt.
Usage: uv run python qadi_prompt.py "Your question here"
"""

import asyncio
import sys
import os
from pathlib import Path

# Auto-load .env file
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator

async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python qadi_prompt.py \"Your question here\"")
        print("\nExamples:")
        print('  uv run python qadi_prompt.py "What are 5 ways to improve productivity?"')
        print('  uv run python qadi_prompt.py "How can I learn a new language effectively?"')
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    # Check API keys
    has_api_key = any(os.getenv(k) for k in ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'])
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    if has_api_key:
        print("🤖 Using LLM agents")
    else:
        print("📋 Using template agents (no API keys found)")
    
    try:
        orchestrator = EnhancedQADIOrchestrator()
        
        # Try with timeout
        try:
            result = await asyncio.wait_for(
                orchestrator.run_qadi_cycle_with_answers(
                    problem_statement=prompt,
                    max_answers=5,
                    cycle_config={'max_ideas_per_method': 3}
                ),
                timeout=10.0 if has_api_key else 5.0
            )
        except asyncio.TimeoutError:
            if has_api_key:
                print("\n⚠️  LLM timed out, falling back to template agents...")
                # Clear API keys to force template usage
                for key in ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
                    if key in os.environ:
                        del os.environ[key]
                
                # Retry with template agents
                orchestrator = EnhancedQADIOrchestrator()
                result = await orchestrator.run_qadi_cycle_with_answers(
                    problem_statement=prompt,
                    max_answers=5,
                    cycle_config={'max_ideas_per_method': 3}
                )
            else:
                raise
        
        print(f"\n⏱️  {result.execution_time:.3f}s | 💰 ${result.llm_cost:.4f}")
        
        if result.extracted_answers:
            print(f"\n✅ {len(result.extracted_answers.direct_answers)} ANSWERS:")
            print("-" * 70)
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   Source: {answer.source_phase} | Confidence: {answer.confidence:.1f}")
        
    except asyncio.TimeoutError:
        print("\n⚠️  Request timed out. Try:")
        print("   - Using a different API key (OpenAI/Anthropic instead of Google)")
        print("   - Running with template agents by removing API keys from .env")
        print("   - Checking your internet connection")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())