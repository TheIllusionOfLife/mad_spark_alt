#!/usr/bin/env python3
"""
QADI - Question, Abduction, Deduction, Induction analysis tool.
Usage: uv run python qadi.py "Your question here"

This tool uses the Mad Spark Alt system to analyze your question using
the QADI methodology and extract practical answers.
"""
import asyncio
import sys
import os
from pathlib import Path

# Auto-load environment variables from .env
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

# Add source to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def run_qadi_analysis(prompt: str):
    """Run QADI analysis on the given prompt."""
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.core.smart_registry import setup_smart_agents
    
    # Get API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    has_llm = any([google_key, openai_key, anthropic_key])
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    if has_llm:
        # Setup LLM providers
        print("🤖 Setting up LLM agents...", end='', flush=True)
        try:
            await setup_llm_providers(
                google_api_key=google_key,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key
            )
            await setup_smart_agents()
            print(" ✓")
            
            provider = "Google" if google_key else ("OpenAI" if openai_key else "Anthropic")
            print(f"   Using: {provider} API")
        except Exception as e:
            print(f" ❌ {e}")
            has_llm = False
    else:
        print("📋 Using template agents (no API keys found)")
    
    # Create orchestrator
    orchestrator = EnhancedQADIOrchestrator()
    
    # Configure for optimal performance
    config = {
        'max_ideas_per_method': 2 if has_llm else 3,
        'parallel_execution': not (has_llm and google_key)  # Sequential for Google
    }
    
    print("\nRunning QADI analysis...")
    
    try:
        # Run with appropriate timeout
        timeout = 60 if has_llm else 10
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle_with_answers(
                problem_statement=prompt,
                max_answers=5,
                cycle_config=config
            ),
            timeout=timeout
        )
        
        # Show results
        print(f"\n⏱️  Completed in {result.execution_time:.1f}s")
        if has_llm:
            print(f"💰 LLM cost: ${result.llm_cost:.4f}")
        
        # Display answers
        if result.extracted_answers:
            print(f"\n✅ {len(result.extracted_answers.direct_answers)} ANSWERS:")
            print("-" * 70)
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   Source: {answer.source_phase} phase")
                if has_llm and answer.confidence > 0.7:
                    print(f"   Confidence: High")
        else:
            print("\n❌ No answers could be extracted")
            
    except asyncio.TimeoutError:
        print(f"\n⏱️ Analysis timed out after {timeout}s")
        if has_llm and google_key:
            print("💡 Tip: Google's API can be slow. Try using OpenAI or Anthropic instead.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("QADI - Multi-perspective analysis tool")
        print("\nUsage: uv run python qadi.py \"Your question here\"")
        print("\nExamples:")
        print('  uv run python qadi.py "What are 5 ways to improve productivity?"')
        print('  uv run python qadi.py "How can I learn programming effectively?"')
        print('  uv run python qadi.py "Why do startups fail?"')
        print("\nFor LLM-powered analysis, add API keys to .env:")
        print("  GOOGLE_API_KEY=your-key")
        print("  OPENAI_API_KEY=your-key")
        print("  ANTHROPIC_API_KEY=your-key")
        sys.exit(1)
    
    prompt = " ".join(sys.argv[1:])  # Support multi-word prompts
    asyncio.run(run_qadi_analysis(prompt))

if __name__ == "__main__":
    main()