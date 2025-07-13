#!/usr/bin/env python3
"""
QADI Working - Multi-agent system using basic orchestrator
Usage: uv run python qadi_working.py "Your question here"

This version uses the basic orchestrator which we know works,
avoiding the timeout issues in the smart orchestrator.
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

async def run_qadi_working(prompt: str):
    """Run QADI using basic orchestrator that works."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.core.orchestrator import QADIOrchestrator
    from mad_spark_alt.core.registry import agent_registry
    
    print(f"📝 {prompt}")
    print("🔧 QADI WORKING MODE (Basic Orchestrator)")
    print("=" * 70)
    
    # Setup providers (we know this works)
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ Google API key required")
        return
    
    print("⚙️ Setting up provider...", end='', flush=True)
    start = time.time()
    await setup_llm_providers(google_api_key=google_key)
    print(f" ✓ ({time.time()-start:.1f}s)")
    
    # Register template agents (avoiding LLM agents for now)
    print("🧠 Registering agents...", end='', flush=True)
    start = time.time()
    
    try:
        # Clear registry and register template agents
        agent_registry.clear()
        
        from mad_spark_alt.agents import QuestioningAgent, AbductionAgent, DeductionAgent, InductionAgent
        
        # Register template agents first
        agent_registry.register(QuestioningAgent)
        agent_registry.register(AbductionAgent)
        agent_registry.register(DeductionAgent)
        agent_registry.register(InductionAgent)
        
        print(f" ✓ ({time.time()-start:.1f}s)")
        print(f"   Registered: {len(agent_registry._agents)} template agents")
        
    except Exception as e:
        print(f" ❌ Agent registration failed: {e}")
        return
    
    # Get registered agents using ThinkingMethod enum
    from mad_spark_alt.core.interfaces import ThinkingMethod
    
    agents = []
    methods = [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION, ThinkingMethod.DEDUCTION, ThinkingMethod.INDUCTION]
    
    for method in methods:
        agents_for_method = agent_registry.get_agents_by_method(method)
        if agents_for_method:
            agents.append(agents_for_method[0])  # Get first agent for this method
    
    if not agents:
        print("❌ No agents available")
        return
    
    print(f"   Available agents: {[a.name for a in agents]}")
    
    # Create basic orchestrator
    print("🎭 Creating orchestrator...", end='', flush=True)
    start = time.time()
    orchestrator = QADIOrchestrator(agents)
    print(f" ✓ ({time.time()-start:.1f}s)")
    
    # Run QADI cycle
    print("\n🚀 Running QADI cycle...")
    cycle_start = time.time()
    
    try:
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle(
                problem_statement=prompt,
                context="Generate practical insights using multi-agent thinking",
                cycle_config={
                    "max_ideas_per_method": 2,
                    "require_reasoning": True
                }
            ),
            timeout=60  # 1 minute timeout
        )
        
        cycle_time = time.time() - cycle_start
        total_time = time.time() - start
        
        print(f"✅ QADI cycle completed in {cycle_time:.1f}s")
        
        # Display results
        print("\n🔍 MULTI-AGENT ANALYSIS:")
        print("-" * 70)
        
        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                print(f"\n{phase_name.upper()} Phase:")
                for i, idea in enumerate(phase_result.generated_ideas, 1):
                    print(f"  {i}. {idea.content}")
                    if idea.reasoning:
                        print(f"     💭 {idea.reasoning}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  ⏱️  Total time: {total_time:.1f}s")
        print(f"  🤖 Agent types: Template agents (reliable)")
        print(f"  🔄 Phases: {len(result.phases)}")
        print(f"  💡 Total ideas: {len(result.synthesized_ideas)}")
        
        print(f"\n💡 Template Agent Benefits:")
        print(f"  • 4 specialized thinking perspectives")
        print(f"  • Structured QADI methodology")
        print(f"  • Reliable execution without timeouts")
        print(f"  • Multi-phase reasoning building on previous insights")
        print(f"  • Much richer than single prompt approach")
        
    except asyncio.TimeoutError:
        print("❌ QADI cycle timed out - this suggests an issue in the core agents")
    except Exception as e:
        print(f"❌ QADI cycle failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_working.py "Your question"')
        print('\nThis version uses template agents with the basic orchestrator')
        print('to avoid timeout issues while maintaining multi-agent benefits:')
        print('  • 4 specialized thinking agents (Question, Abduction, Deduction, Induction)')
        print('  • Structured QADI methodology')
        print('  • Reliable execution without LLM timeout issues')
        print('  • Multi-perspective analysis')
        print('\nExamples:')
        print('  uv run python qadi_working.py "improve team productivity"')
        print('  uv run python qadi_working.py "reduce software bugs"')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_qadi_working(prompt))