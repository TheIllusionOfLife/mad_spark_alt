#!/usr/bin/env python3
"""
QADI command-line tool - Working version
Usage: uv run python qadi_working.py "Your question here"
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
    """Run QADI analysis."""
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Check if we have Google API key
    has_google = bool(os.getenv('GOOGLE_API_KEY'))
    
    if has_google:
        print("🤖 Google API detected")
        print("⚠️  Note: Google API is slow (60+ seconds). For faster results, use OpenAI or Anthropic.")
        print("\nProceeding with template agents for demo...")
        
        # Force template mode for demo
        os.environ.pop('GOOGLE_API_KEY', None)
        os.environ.pop('OPENAI_API_KEY', None) 
        os.environ.pop('ANTHROPIC_API_KEY', None)
    else:
        print("📋 Using template agents")
    
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    
    orchestrator = EnhancedQADIOrchestrator()
    
    print("\nRunning QADI analysis...")
    
    result = await orchestrator.run_qadi_cycle_with_answers(
        problem_statement=prompt,
        max_answers=5,
        cycle_config={'max_ideas_per_method': 3}
    )
    
    print(f"\n⏱️  Completed in {result.execution_time:.3f}s")
    
    if result.extracted_answers:
        print(f"\n✅ {len(result.extracted_answers.direct_answers)} ANSWERS:")
        print("-" * 70)
        
        for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
            print(f"\n{i}. {answer.content}")
            print(f"   Source: {answer.source_phase} thinking")
    
    if has_google:
        print("\n💡 To use LLM agents with Google API:")
        print("   1. Remove the demo override in this script")
        print("   2. Be prepared to wait 60+ seconds")
        print("   3. Or add OpenAI/Anthropic keys for faster results")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_working.py "Your question"')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(main(prompt))