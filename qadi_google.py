#!/usr/bin/env python3
"""
QADI runner optimized for Google API.
Usage: uv run python qadi_google.py "Your question here"
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
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mad_spark_alt.core.enhanced_orchestrator import EnhancedQADIOrchestrator
from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator

async def run_google_optimized(prompt: str):
    """Run QADI with optimizations for Google API."""
    
    # Check for Google API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ No Google API key found in .env file")
        print("Add: GOOGLE_API_KEY=your-key-here")
        return
    
    print(f"📝 {prompt}")
    print("=" * 70)
    print("🤖 Using Google Gemini API (optimized)")
    print("⏳ This may take 20-30 seconds...\n")
    
    start_time = time.time()
    
    try:
        # Create orchestrator
        orchestrator = EnhancedQADIOrchestrator()
        
        # Override to use sequential execution for Google API
        async def run_qadi_cycle_sequential(problem_statement, context=None, cycle_config=None):
            """Run QADI phases sequentially instead of in parallel."""
            config = cycle_config or {}
            
            # Run each phase one at a time
            phases = {}
            phase_order = ['questioning', 'abduction', 'deduction', 'induction']
            
            for phase in phase_order:
                print(f"  Running {phase}...", end='', flush=True)
                phase_start = time.time()
                
                agent = orchestrator.registry.get_agent(phase.upper())
                if agent:
                    from mad_spark_alt.core.interfaces import IdeaGenerationRequest
                    request = IdeaGenerationRequest(
                        problem_statement=problem_statement,
                        context=context,
                        thinking_method=phase.upper(),
                        previous_ideas=[],
                        config={"max_ideas": 2}  # Reduce ideas per phase
                    )
                    result = await agent.generate_ideas(request)
                    phases[phase] = result
                    print(f" ✓ ({time.time() - phase_start:.1f}s)")
            
            # Synthesize ideas
            print("  Synthesizing ideas...", end='', flush=True)
            all_ideas = []
            agent_types = {}
            for phase_name, result in phases.items():
                if result:
                    for idea in result.generated_ideas:
                        idea.metadata = idea.metadata or {}
                        idea.metadata["phase"] = phase_name
                        all_ideas.append(idea)
                    agent_types[phase_name] = "LLM"
            print(" ✓")
            
            # Create result
            from mad_spark_alt.core.smart_orchestrator import SmartQADICycleResult
            return SmartQADICycleResult(
                problem_statement=problem_statement,
                context=context,
                phases=phases,
                synthesized_ideas=all_ideas[:10],  # Limit synthesized ideas
                synthesis_method="sequential",
                agent_types=agent_types,
                execution_time=time.time() - start_time,
                llm_cost=0.0
            )
        
        # Monkey patch the method temporarily
        original_method = orchestrator.run_qadi_cycle
        orchestrator.run_qadi_cycle = run_qadi_cycle_sequential
        
        # Run with answer extraction
        result = await orchestrator.run_qadi_cycle_with_answers(
            problem_statement=prompt,
            max_answers=5,
            cycle_config={'max_ideas_per_method': 2}
        )
        
        # Display results
        print(f"\n✅ Completed in {result.execution_time:.1f}s")
        print(f"💰 Cost: ${result.llm_cost:.4f}")
        
        if result.extracted_answers:
            print(f"\n📋 {len(result.extracted_answers.direct_answers)} ANSWERS:")
            print("-" * 70)
            
            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"\n{i}. {answer.content}")
                print(f"   Source: {answer.source_phase} phase")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_google.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_google.py "What are 5 ways to improve productivity?"')
        print('  uv run python qadi_google.py "How can I learn programming effectively?"')
    else:
        asyncio.run(run_google_optimized(sys.argv[1]))