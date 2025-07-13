#!/usr/bin/env python3
"""
QADI - Question, Abduction, Deduction, Induction analysis tool
Usage: uv run python qadi.py "Your question here"

Analyzes questions using the QADI methodology with Google Gemini API.
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

async def run_single_llm_qadi(prompt: str):
    """Run QADI with just one LLM call."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager, LLMRequest
    
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found in .env")
        return
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Setup Google API
    await setup_llm_providers(google_api_key=google_key)
    
    print("🤖 Using Google Gemini API")
    
    start_time = time.time()
    
    # Single LLM call that does simplified QADI
    qadi_prompt = f"""Analyze this question using the QADI methodology:
"{prompt}"

Provide exactly 3 practical answers based on:
1. One key question to explore
2. One creative hypothesis
3. One logical deduction

Format:
QUESTION: [Your question]
HYPOTHESIS: [Your hypothesis]
DEDUCTION: [Your logical deduction]
ANSWER1: [First practical answer based on the question]
ANSWER2: [Second practical answer based on the hypothesis]
ANSWER3: [Third practical answer based on the deduction]"""

    print("\nGenerating QADI analysis...", end='', flush=True)
    
    request = LLMRequest(
        user_prompt=qadi_prompt,
        max_tokens=500,
        temperature=0.7
    )
    
    try:
        response = await asyncio.wait_for(llm_manager.generate(request), timeout=30)
        print(" ✓")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Completed in {elapsed:.1f}s")
        print(f"💰 Cost: ${response.cost:.4f}")
        
        # Parse response
        content = response.content
        lines = content.split('\n')
        
        # Extract parts
        question = hypothesis = deduction = ""
        answers = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("HYPOTHESIS:"):
                hypothesis = line.replace("HYPOTHESIS:", "").strip()
            elif line.startswith("DEDUCTION:"):
                deduction = line.replace("DEDUCTION:", "").strip()
            elif line.startswith("ANSWER"):
                answer = line.split(":", 1)[1].strip() if ":" in line else line
                if answer:
                    answers.append(answer)
        
        # Display QADI thinking
        if any([question, hypothesis, deduction]):
            print("\n🧠 QADI ANALYSIS:")
            print("-" * 70)
            if question:
                print(f"\n❓ Question: {question}")
            if hypothesis:
                print(f"\n💡 Hypothesis: {hypothesis}")
            if deduction:
                print(f"\n🔍 Deduction: {deduction}")
        
        # Display answers
        if answers:
            print(f"\n✅ PRACTICAL ANSWERS:")
            print("-" * 70)
            for i, answer in enumerate(answers[:3], 1):
                print(f"\n{i}. {answer}")
        else:
            # Fallback if parsing fails
            print(f"\n📄 Raw response:")
            print("-" * 70)
            print(content)
            
    except asyncio.TimeoutError:
        print(" ⏱️ timeout")
        print(f"\nGoogle API timed out after 30 seconds.")
    except Exception as e:
        print(f" ❌ error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi.py "how to live longer"')
        print('  uv run python qadi.py "what are 3 ways to reduce stress"')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_single_llm_qadi(prompt))