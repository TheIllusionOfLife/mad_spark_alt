#!/usr/bin/env python3
"""
QADI with Google API - Real implementation
Usage: uv run python qadi_google_real.py "Your question here"
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

async def run_minimal_qadi(prompt: str):
    """Run minimal QADI cycle optimized for Google API."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager, LLMRequest
    from mad_spark_alt.agents.questioning.llm_agent import LLMQuestioningAgent
    from mad_spark_alt.agents.abduction.llm_agent import LLMAbductiveAgent
    from mad_spark_alt.core.answer_extractor import TemplateAnswerExtractor
    from mad_spark_alt.core.interfaces import IdeaGenerationRequest, ThinkingMethod
    
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found in .env")
        return
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Setup Google API
    print("🤖 Setting up Google API...", end='', flush=True)
    await setup_llm_providers(google_api_key=google_key)
    print(" ✓")
    
    # Test connection
    print("Testing connection...", end='', flush=True)
    test_req = LLMRequest(user_prompt="Say 'ready'", max_tokens=10, temperature=0.1)
    test_resp = await llm_manager.generate(test_req)
    print(f" ✓ ({test_resp.provider})")
    
    start_time = time.time()
    
    print("\nRunning simplified QADI (2 phases only for speed):")
    
    # 1. Questioning phase - just 1 quick question
    print("  • Questioning...", end='', flush=True)
    q_agent = LLMQuestioningAgent()
    q_request = IdeaGenerationRequest(
        problem_statement=prompt,
        generation_config={
            "max_questions_per_strategy": 1,
            "questioning_strategy": "fundamental_inquiry"  # Just one strategy
        },
        max_ideas_per_method=1
    )
    
    try:
        q_result = await asyncio.wait_for(q_agent.generate_ideas(q_request), timeout=20)
        q_ideas = q_result.generated_ideas[:1] if q_result else []
        print(f" ✓ ({len(q_ideas)} idea)")
    except asyncio.TimeoutError:
        print(" ⏱️ timeout")
        q_ideas = []
    
    # 2. Abduction phase - just 2 hypotheses
    print("  • Hypotheses...", end='', flush=True)
    a_agent = LLMAbductiveAgent()
    a_request = IdeaGenerationRequest(
        problem_statement=prompt,
        generation_config={
            "max_hypotheses_per_strategy": 1,
            "abductive_strategy": "creative_leap"  # Just one strategy
        },
        max_ideas_per_method=2
    )
    
    try:
        a_result = await asyncio.wait_for(a_agent.generate_ideas(a_request), timeout=20)
        a_ideas = a_result.generated_ideas[:2] if a_result else []
        print(f" ✓ ({len(a_ideas)} ideas)")
    except asyncio.TimeoutError:
        print(" ⏱️ timeout")
        a_ideas = []
    
    # Group ideas
    ideas_by_phase = {
        "questioning": q_ideas,
        "abduction": a_ideas,
        "deduction": [],  # Skip for speed
        "induction": []   # Skip for speed
    }
    
    # Extract answers
    print("  • Extracting answers...", end='', flush=True)
    extractor = TemplateAnswerExtractor()
    result = extractor.extract_answers(prompt, ideas_by_phase, max_answers=3)
    print(" ✓")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Completed in {elapsed:.1f}s (using Google Gemini)")
    
    # Show actual LLM-generated content
    if any([q_ideas, a_ideas]):
        print("\n🧠 LLM-GENERATED INSIGHTS:")
        print("-" * 70)
        
        if q_ideas:
            print("\nQuestioning:")
            for idea in q_ideas:
                print(f"  • {idea.content}")
        
        if a_ideas:
            print("\nHypotheses:")
            for idea in a_ideas:
                print(f"  • {idea.content}")
    
    # Show extracted answers
    if result.direct_answers:
        print(f"\n✅ PRACTICAL ANSWERS:")
        print("-" * 70)
        
        for i, answer in enumerate(result.direct_answers, 1):
            print(f"\n{i}. {answer.content}")
            if answer.source_ideas and len(answer.source_ideas[0]) > 20:
                print(f"   Based on: {answer.source_phase} phase")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_google_real.py "Your question"')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_minimal_qadi(prompt))