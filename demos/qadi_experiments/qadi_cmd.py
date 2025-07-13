#!/usr/bin/env python3
"""
QADI command line tool with LLM support.
Usage: uv run python qadi_cmd.py "Your question here"
"""
import asyncio
import sys
import os
from pathlib import Path
import time

# Load .env automatically
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def run_with_llm(prompt: str):
    """Run QADI with LLM agents."""
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    from mad_spark_alt.core.llm_provider import llm_manager, setup_llm_providers
    
    # Check which API keys are available
    providers = []
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if google_key:
        providers.append('Google')
    if openai_key:
        providers.append('OpenAI')
    if anthropic_key:
        providers.append('Anthropic')
    
    if not providers:
        print("❌ No API keys found in .env")
        print("Add one of: GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
        return
    
    print(f"📝 {prompt}")
    print("=" * 70)
    print(f"🤖 Available LLM providers: {', '.join(providers)}")
    
    # Initialize LLM providers
    await setup_llm_providers(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        google_api_key=google_key
    )
    
    # Test LLM connectivity
    print("\nTesting LLM connection...", end='', flush=True)
    try:
        from mad_spark_alt.core.llm_provider import LLMRequest
        test_request = LLMRequest(
            user_prompt="Respond with exactly: 'LLM is working'",
            temperature=0.1,
            max_tokens=20
        )
        test_response = await llm_manager.generate(test_request)
        if test_response and test_response.content:
            print(" ✓")
            print(f"Provider: {test_response.provider}")
        else:
            print(" ❌ No response")
            return
    except Exception as e:
        print(f" ❌ {str(e)}")
        return
    
    # Run QADI with reduced complexity for faster response
    print("\nRunning QADI analysis (this may take 30-60 seconds)...")
    print("Progress:", end='', flush=True)
    
    start_time = time.time()
    
    try:
        orchestrator = EnhancedQADIOrchestrator()
        
        # Configure for faster execution
        config = {
            'max_ideas_per_method': 2,  # Reduce ideas per phase
            'prefer_llm': True,
            'parallel_execution': False  # Run phases sequentially
        }
        
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=prompt,
            max_answers=5,
            cycle_config=config
        )
        
        print(" ✓")
        print(f"\n⏱️  Completed in {result.execution_time:.1f}s")
        print(f"💰 LLM cost: ${result.llm_cost:.4f}")
        print(f"🤖 Agents used: {list(set(result.agent_types.values()))}")
        
        if result.extracted_answers:
            print(f"\n✅ {len(result.extracted_answers.direct_answers)} ANSWERS:")
            print("-" * 70)
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   Source: {answer.source_phase} | Confidence: {answer.confidence:.1f}")
                if answer.source_ideas and len(answer.source_ideas[0]) > 10:
                    preview = answer.source_ideas[0][:80] + "..." if len(answer.source_ideas[0]) > 80 else answer.source_ideas[0]
                    print(f"   Based on: \"{preview}\"")
        
    except asyncio.TimeoutError:
        print(" ⏱️ Timeout")
        print("\nThe LLM is taking too long. Try again or use a different provider.")
    except Exception as e:
        print(f" ❌ Error")
        print(f"\nError: {e}")
        
        # Fall back to template agents
        print("\nFalling back to template agents...")
        for key in ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
            if key in os.environ:
                del os.environ[key]
        
        orchestrator = EnhancedQADIOrchestrator()
        result = await orchestrator.run_qadi_cycle_with_answers(prompt, max_answers=3)
        
        if result.extracted_answers:
            print("\n📋 Template-based answers:")
            for i, answer in enumerate(result.extracted_answers.direct_answers[:3], 1):
                print(f"{i}. {answer.content[:100]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_cmd.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_cmd.py "What are 5 ways to improve productivity?"')
        print('  uv run python qadi_cmd.py "How can I learn a new skill effectively?"')
    else:
        # Set timeout for the entire operation
        try:
            asyncio.run(asyncio.wait_for(run_with_llm(sys.argv[1]), timeout=90))
        except asyncio.TimeoutError:
            print("\n⏱️ Operation timed out after 90 seconds")
            print("This usually happens with slow LLM providers. Try:")
            print("  - Using OpenAI or Anthropic instead of Google")
            print("  - Running with a simpler question")
            print("  - Checking your internet connection")