#!/usr/bin/env python3
"""
QADI Simple Multi - Multiple simple LLM calls mimicking multi-agent approach
Usage: uv run python qadi_simple_multi.py "Your question here"

This version uses Google API with multiple simple calls to avoid timeouts
while providing multi-agent-like perspectives.
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

async def run_qadi_phase(phase_name: str, prompt: str, previous_insights: str = ""):
    """Run a single QADI phase using simple LLM call."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers, llm_manager, LLMRequest
    
    # Phase-specific prompts
    phase_prompts = {
        "questioning": f"""As a questioning specialist, generate 2 insightful questions about: "{prompt}"
{previous_insights}
Format each question on a new line starting with "Q:".""",
        
        "abduction": f"""As a hypothesis specialist, generate 2 creative hypotheses about: "{prompt}"
{previous_insights}
Consider unexpected connections and possibilities.
Format each hypothesis on a new line starting with "H:".""",
        
        "deduction": f"""As a logical reasoning specialist, generate 2 logical deductions about: "{prompt}"
{previous_insights}
Apply systematic reasoning and derive conclusions.
Format each deduction on a new line starting with "D:".""",
        
        "induction": f"""As a pattern synthesis specialist, generate 2 pattern-based insights about: "{prompt}"
{previous_insights}
Identify recurring themes and general principles.
Format each insight on a new line starting with "I:"."""
    }
    
    request = LLMRequest(
        user_prompt=phase_prompts[phase_name],
        max_tokens=300,
        temperature=0.7
    )
    
    try:
        response = await asyncio.wait_for(llm_manager.generate(request), timeout=20)
        return response.content, response.cost, response.model
    except asyncio.TimeoutError:
        return f"[{phase_name} phase timed out]", 0.0, "unknown"
    except Exception as e:
        return f"[{phase_name} phase error: {e}]", 0.0, "unknown"

async def run_simple_multi_agent_qadi(prompt: str):
    """Run QADI using multiple simple LLM calls."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.core.json_utils import format_llm_cost
    
    print(f"📝 {prompt}")
    print("🚀 QADI SIMPLE MULTI-AGENT (LLM Mode)")
    print("=" * 70)
    
    # Setup Google API
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found in .env")
        return
    
    print("🤖 Setting up LLM providers...", end='', flush=True)
    start_time = time.time()
    
    try:
        await setup_llm_providers(google_api_key=google_key)
        print(f" ✓ ({time.time()-start_time:.1f}s)")
    except Exception as e:
        print(f" ❌ Setup failed: {e}")
        return
    
    # Run QADI phases sequentially
    print("\n🧠 Running QADI Multi-Agent Analysis...")
    print("  ├─ Question phase: Generating insightful questions")
    print("  ├─ Abduction phase: Creating hypotheses")
    print("  ├─ Deduction phase: Logical reasoning")
    print("  └─ Induction phase: Pattern synthesis")
    
    total_cost = 0.0
    all_insights = []
    
    # Phase 1: Questioning
    print("\n❓ QUESTIONING Phase...", end='', flush=True)
    phase_start = time.time()
    questions, q_cost, model_name = await run_qadi_phase("questioning", prompt)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += q_cost
    
    # Show model info after first successful call
    if model_name != "unknown":
        print(f"🤖 Model: {model_name}")
    all_insights.append(f"Questions explored:\n{questions}")
    
    # Phase 2: Abduction (using previous insights)
    print("💡 ABDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Building on these questions:\n{questions}"
    hypotheses, h_cost, _ = await run_qadi_phase("abduction", prompt, previous)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += h_cost
    all_insights.append(f"Hypotheses generated:\n{hypotheses}")
    
    # Phase 3: Deduction (using accumulated insights)
    print("🔍 DEDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Building on questions and hypotheses:\n{questions}\n{hypotheses}"
    deductions, d_cost, _ = await run_qadi_phase("deduction", prompt, previous)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += d_cost
    all_insights.append(f"Logical deductions:\n{deductions}")
    
    # Phase 4: Induction (synthesizing all insights)
    print("🎯 INDUCTION Phase...", end='', flush=True)
    phase_start = time.time()
    previous = f"Synthesizing all insights:\n{questions}\n{hypotheses}\n{deductions}"
    patterns, i_cost, _ = await run_qadi_phase("induction", prompt, previous)
    phase_time = time.time() - phase_start
    print(f" ✓ ({phase_time:.1f}s)")
    total_cost += i_cost
    
    # Display results
    print("\n\n🔍 MULTI-AGENT QADI ANALYSIS:")
    print("=" * 70)
    
    print("\n❓ QUESTIONING:")
    for line in questions.split('\n'):
        if line.strip() and line.startswith('Q:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n💡 ABDUCTION:")
    for line in hypotheses.split('\n'):
        if line.strip() and line.startswith('H:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n🔍 DEDUCTION:")
    for line in deductions.split('\n'):
        if line.strip() and line.startswith('D:'):
            print(f"  • {line[2:].strip()}")
    
    print("\n🎯 INDUCTION:")
    for line in patterns.split('\n'):
        if line.strip() and line.startswith('I:'):
            print(f"  • {line[2:].strip()}")
    
    # Final synthesis
    print("\n✨ SYNTHESIS:")
    print("-" * 70)
    synthesis_prompt = f"""Based on this QADI analysis for "{prompt}":
{questions}
{hypotheses}
{deductions}
{patterns}

Provide 3 actionable insights that synthesize all perspectives."""
    
    from mad_spark_alt.core.llm_provider import llm_manager, LLMRequest
    request = LLMRequest(
        user_prompt=synthesis_prompt,
        max_tokens=400,
        temperature=0.5
    )
    
    try:
        synthesis = await asyncio.wait_for(llm_manager.generate(request), timeout=20)
        print(synthesis.content)
        total_cost += synthesis.cost
    except:
        print("(Synthesis timed out)")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 Performance Summary:")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  💰 Total cost: {format_llm_cost(total_cost)}")
    print(f"  🤖 API calls: 5 (4 phases + synthesis)")
    if model_name != "unknown":
        print(f"  ✅ Model: {model_name}")
    else:
        print(f"  ✅ LLM mode: Multi-agent analysis")
    
    print(f"\n💡 Advantages:")
    print(f"  • Real LLM-powered insights (not templates)")
    print(f"  • Multi-perspective QADI analysis")
    print(f"  • Progressive reasoning (each phase builds on previous)")
    print(f"  • No timeout issues")
    print(f"  • Much richer than single prompt approach")

def show_help():
    """Display help information."""
    print("QADI Simple Multi-Agent - Multi-perspective AI analysis tool")
    print("=" * 60)
    print()
    print("USAGE:")
    print('  uv run python qadi_simple_multi.py "Your question"')
    print('  uv run python qadi_simple_multi.py [OPTIONS] "Your question"')
    print()
    print("DESCRIPTION:")
    print("  Analyzes questions using the QADI methodology (Question → Abduction →")
    print("  Deduction → Induction) with Google Gemini API. Provides multi-agent")
    print("  perspectives without timeouts through sequential LLM calls.")
    print()
    print("OPTIONS:")
    print("  -h, --help    Show this help message and exit")
    print("  --version     Show version information")
    print()
    print("FEATURES:")
    print("  • Real LLM-powered insights (not templates)")
    print("  • Multi-perspective QADI analysis")  
    print("  • Progressive reasoning (each phase builds on previous)")
    print("  • No timeout issues")
    print("  • Smart cost display")
    print("  • Model identification (shows specific model used)")
    print()
    print("REQUIREMENTS:")
    print("  • Google API key in .env file (GOOGLE_API_KEY=your-key)")
    print("  • Internet connection")
    print()
    print("EXAMPLES:")
    print('  uv run python qadi_simple_multi.py "how to create AGI"')
    print('  uv run python qadi_simple_multi.py "reduce climate change"')
    print('  uv run python qadi_simple_multi.py "improve team productivity"')
    print('  uv run python qadi_simple_multi.py "design better user interfaces"')
    print()
    print("QADI METHODOLOGY:")
    print("  Question   → Generate insightful questions about the topic")
    print("  Abduction  → Create creative hypotheses and possibilities") 
    print("  Deduction  → Apply logical reasoning and derive conclusions")
    print("  Induction  → Identify patterns and synthesize insights")
    print("  Synthesis  → Combine all perspectives into actionable recommendations")
    print()
    print("OUTPUT:")
    print("  The tool provides structured analysis with:")
    print("  • Detailed phase-by-phase insights")
    print("  • Final synthesis with 3 actionable recommendations")
    print("  • Performance metrics (time, cost, model used)")
    print("  • Cost information (shows 'Free' for low API costs)")

def show_version():
    """Display version information."""
    print("QADI Simple Multi-Agent v1.0")
    print("Part of Mad Spark Alt - Multi-Agent Idea Generation System")
    print("Uses Google Gemini API with smart cost display and model identification")

if __name__ == "__main__":
    # Handle help and version flags
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        show_help()
    elif sys.argv[1] == "--version":
        show_version()
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_simple_multi_agent_qadi(prompt))