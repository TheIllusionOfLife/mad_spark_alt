#!/usr/bin/env python3
"""
QADI command-line tool with Google API support.
Usage: uv run python qadi_final.py "Your question here"
"""
import asyncio
import sys
import os
from pathlib import Path
import time

# Auto-load .env
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def run_qadi(prompt: str):
    """Run QADI with optimizations for Google API."""
    from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
    from mad_spark_alt.core.llm_provider import llm_manager, setup_llm_providers, LLMRequest
    from mad_spark_alt.core.interfaces import IdeaGenerationRequest, ThinkingMethod
    
    # Check API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not any([google_key, openai_key, anthropic_key]):
        print("❌ No API keys found in .env")
        return
    
    # Initialize providers
    await setup_llm_providers(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        google_api_key=google_key
    )
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Quick API test
    test_request = LLMRequest(user_prompt="Say 'ready'", max_tokens=10, temperature=0.1)
    test_response = await llm_manager.generate(test_request)
    print(f"🤖 Using: {test_response.provider}")
    
    start_time = time.time()
    
    # Create orchestrator but run a simplified version
    orchestrator = EnhancedQADIOrchestrator()
    
    # Run only the most important QADI phases for speed
    print("\nRunning QADI analysis...")
    
    # 1. Quick questioning phase
    print("  • Questioning...", end='', flush=True)
    questioning_agent = orchestrator.registry.get_preferred_agent(ThinkingMethod.QUESTIONING)
    if questioning_agent:
        q_request = IdeaGenerationRequest(
            problem_statement=prompt,
            thinking_method=ThinkingMethod.QUESTIONING,
            config={"max_ideas": 1}  # Just 1 idea per phase
        )
        q_result = await questioning_agent.generate_ideas(q_request)
        print(" ✓")
    else:
        q_result = None
        print(" ❌")
    
    # 2. Quick abduction phase
    print("  • Generating hypotheses...", end='', flush=True)
    abduction_agent = orchestrator.registry.get_preferred_agent(ThinkingMethod.ABDUCTION)
    if abduction_agent:
        a_request = IdeaGenerationRequest(
            problem_statement=prompt,
            thinking_method=ThinkingMethod.ABDUCTION,
            config={"max_ideas": 2}
        )
        a_result = await abduction_agent.generate_ideas(a_request)
        print(" ✓")
    else:
        a_result = None
        print(" ❌")
    
    # 3. Quick deduction phase
    print("  • Logical analysis...", end='', flush=True)
    deduction_agent = orchestrator.registry.get_preferred_agent(ThinkingMethod.DEDUCTION)
    if deduction_agent:
        d_request = IdeaGenerationRequest(
            problem_statement=prompt,
            thinking_method=ThinkingMethod.DEDUCTION,
            config={"max_ideas": 2}
        )
        d_result = await deduction_agent.generate_ideas(d_request)
        print(" ✓")
    else:
        d_result = None
        print(" ❌")
    
    # Group ideas by phase
    ideas_by_phase = {
        "questioning": q_result.generated_ideas if q_result else [],
        "abduction": a_result.generated_ideas if a_result else [],
        "deduction": d_result.generated_ideas if d_result else []
    }
    
    # Extract answers
    print("  • Extracting answers...", end='', flush=True)
    from mad_spark_alt.core.answer_extractor import TemplateAnswerExtractor
    extractor = TemplateAnswerExtractor()
    answers = extractor.extract_answers(prompt, ideas_by_phase, max_answers=3)
    print(" ✓")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Completed in {elapsed:.1f}s")
    
    # Display answers
    if answers and answers.direct_answers:
        print(f"\n✅ {len(answers.direct_answers)} ANSWERS:")
        print("-" * 70)
        
        for i, answer in enumerate(answers.direct_answers, 1):
            print(f"\n{i}. {answer.content}")
            if answer.source_ideas and len(answer.source_ideas[0]) > 20:
                print(f"   Based on: {answer.source_phase} insight")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_final.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_final.py "What are 3 ways to improve productivity?"')
        print('  uv run python qadi_final.py "How can I learn a new language?"')
    else:
        asyncio.run(run_qadi(sys.argv[1]))