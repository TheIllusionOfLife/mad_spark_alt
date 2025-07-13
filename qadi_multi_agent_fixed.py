#!/usr/bin/env python3
"""
QADI Multi-Agent - Fixed version with robust provider setup
Usage: uv run python qadi_multi_agent_fixed.py "Your question here"

This version fixes the timeout issues by:
1. Only initializing providers with valid API keys
2. Adding proper timeout handling
3. Providing fallback to Google-only mode
4. Adding detailed progress feedback
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

async def setup_providers_safely(timeout_seconds=30):
    """Setup LLM providers with timeout and error handling."""
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    
    # Check which API keys are available
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')  
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    # Only pass keys that actually exist
    provider_args = {}
    providers_to_setup = []
    
    if google_key:
        provider_args['google_api_key'] = google_key
        providers_to_setup.append('Google')
    
    if openai_key:
        provider_args['openai_api_key'] = openai_key
        providers_to_setup.append('OpenAI')
        
    if anthropic_key:
        provider_args['anthropic_api_key'] = anthropic_key
        providers_to_setup.append('Anthropic')
    
    if not provider_args:
        raise ValueError("No valid API keys found")
    
    print(f"   Initializing: {', '.join(providers_to_setup)}")
    
    # Setup with timeout
    try:
        await asyncio.wait_for(
            setup_llm_providers(**provider_args),
            timeout=timeout_seconds
        )
        return providers_to_setup
    except asyncio.TimeoutError:
        raise TimeoutError(f"Provider setup timed out after {timeout_seconds}s")

async def run_full_qadi_system_fixed(prompt: str, use_evolution: bool = False):
    """Run the full QADI multi-agent system with robust error handling."""
    from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
    from mad_spark_alt.evolution import GeneticAlgorithm, EvolutionConfig, EvolutionRequest
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Step 1: Setup LLM providers with timeout and fallback
    print("🔧 Setting up LLM providers...", end='', flush=True)
    setup_start = time.time()
    
    try:
        providers_setup = await setup_providers_safely(timeout_seconds=30)
        setup_time = time.time() - setup_start
        print(f" ✓ ({setup_time:.1f}s)")
        print(f"   Providers ready: {', '.join(providers_setup)}")
        
    except TimeoutError as e:
        print(f" ❌ Timeout")
        print(f"   Error: {e}")
        print(f"   Falling back to Google-only mode...")
        
        # Fallback: try Google only
        try:
            google_key = os.getenv('GOOGLE_API_KEY')
            if not google_key:
                print("❌ No Google API key available for fallback")
                return
                
            from mad_spark_alt.core.llm_provider import setup_llm_providers
            await asyncio.wait_for(
                setup_llm_providers(google_api_key=google_key),
                timeout=15
            )
            setup_time = time.time() - setup_start
            print(f"   ✓ Fallback successful ({setup_time:.1f}s)")
            providers_setup = ['Google (fallback)']
            
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return
            
    except Exception as e:
        print(f" ❌ Setup failed: {e}")
        return
    
    # Step 2: Create orchestrator with timeout
    print("🧠 Initializing QADI orchestrator...", end='', flush=True)
    orchestrator_start = time.time()
    
    try:
        orchestrator = SmartQADIOrchestrator(auto_setup=True)
        
        # Test agent readiness with timeout
        await asyncio.wait_for(
            orchestrator.ensure_agents_ready(),
            timeout=20
        )
        
        orchestrator_time = time.time() - orchestrator_start
        print(f" ✓ ({orchestrator_time:.1f}s)")
        
    except asyncio.TimeoutError:
        print(" ❌ Timeout during agent setup")
        print("   Using template agents as fallback...")
        orchestrator = SmartQADIOrchestrator(auto_setup=False)  # Skip auto setup
        orchestrator_time = time.time() - orchestrator_start
        
    except Exception as e:
        print(f" ❌ Orchestrator setup failed: {e}")
        return
    
    # Step 3: Configure for optimized performance
    cycle_config = {
        "max_ideas_per_method": 2,  # Reduced for speed
        "require_reasoning": True,
        "questioning": {
            "questioning_strategy": "fundamental_inquiry",
            "max_strategies": 2  # Limit strategies for speed
        },
        "abduction": {
            "hypothesis_types": ["creative", "analogical"],
            "max_hypotheses": 2
        },
        "deduction": {
            "reasoning_depth": "balanced",
            "include_edge_cases": False  # Skip for speed
        },
        "induction": {
            "pattern_types": ["general", "specific"],
            "synthesis_approach": "balanced"
        }
    }
    
    print("\n🧠 Running QADI Multi-Agent System...")
    print("  ├─ Question phase: Generating insightful questions")
    print("  ├─ Abduction phase: Creating hypotheses") 
    print("  ├─ Deduction phase: Logical reasoning")
    print("  └─ Induction phase: Pattern synthesis")
    
    # Step 4: Run QADI cycle with timeout
    cycle_start = time.time()
    
    try:
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle(
                problem_statement=prompt,
                context="Generate practical, actionable insights",
                cycle_config=cycle_config
            ),
            timeout=120  # 2 minute timeout for QADI cycle
        )
        
        cycle_time = time.time() - cycle_start
        print(f"\n⏱️  QADI cycle completed in {cycle_time:.1f}s")
        
    except asyncio.TimeoutError:
        print(f"\n❌ QADI cycle timed out after 2 minutes")
        print("   Try reducing max_ideas_per_method or use simple qadi.py instead")
        return
        
    except Exception as e:
        print(f"\n❌ QADI cycle failed: {e}")
        return
    
    # Step 5: Display results
    print(f"\n⚙️  Agent Configuration:")
    for method, agent_type in result.agent_types.items():
        icon = "🤖" if agent_type == "LLM" else "📝"
        print(f"  {icon} {method.title()}: {agent_type} agent")
    
    # Display phase results
    print("\n🔍 QADI ANALYSIS:")
    print("-" * 70)
    
    for phase_name, phase_result in result.phases.items():
        if phase_result and phase_result.generated_ideas:
            print(f"\n{phase_name.upper()} Phase:")
            for idea in phase_result.generated_ideas[:2]:  # Show top 2 per phase
                print(f"  • {idea.content}")
                if idea.reasoning:
                    print(f"    ↳ {idea.reasoning[:100]}...")
    
    # Step 6: Optional genetic evolution
    if use_evolution and len(result.synthesized_ideas) >= 5:
        print("\n🧬 Running Genetic Evolution...")
        try:
            ga = GeneticAlgorithm()
            
            # Configure for speed
            evolution_config = EvolutionConfig(
                population_size=10,
                generations=3,
                mutation_rate=0.2,
                crossover_rate=0.7,
                elite_size=2
            )
            
            evolution_request = EvolutionRequest(
                initial_population=result.synthesized_ideas,
                config=evolution_config,
                context=prompt
            )
            
            evo_start = time.time()
            evo_result = await asyncio.wait_for(
                ga.evolve(evolution_request),
                timeout=60  # 1 minute timeout for evolution
            )
            evo_time = time.time() - evo_start
            
            print(f"  ✓ Evolution completed in {evo_time:.1f}s")
            if evo_result.evolution_metrics:
                improvement = evo_result.evolution_metrics.get('fitness_improvement_percent', 0)
                print(f"  📈 Fitness improved by {improvement:.1f}%")
                
        except asyncio.TimeoutError:
            print("  ❌ Evolution timed out - skipping")
        except Exception as e:
            print(f"  ❌ Evolution failed: {e}")
    
    # Step 7: Display conclusion
    if result.conclusion:
        print("\n✅ SYNTHESIZED INSIGHTS:")
        print("-" * 70)
        print(f"\n{result.conclusion.summary}")
        
        if result.conclusion.key_insights:
            print("\nKey Insights:")
            for i, insight in enumerate(result.conclusion.key_insights[:3], 1):
                print(f"{i}. {insight}")
        
        if result.conclusion.actionable_recommendations:
            print("\nActionable Recommendations:")
            for i, rec in enumerate(result.conclusion.actionable_recommendations[:3], 1):
                print(f"{i}. {rec}")
    else:
        # Fallback: Show top synthesized ideas
        print("\n✅ TOP INSIGHTS:")
        print("-" * 70)
        
        # Group by thinking method
        ideas_by_method = {}
        for idea in result.synthesized_ideas:
            method = idea.thinking_method.value
            if method not in ideas_by_method:
                ideas_by_method[method] = []
            ideas_by_method[method].append(idea)
        
        # Show best from each method
        for method, ideas in ideas_by_method.items():
            if ideas:
                print(f"\nFrom {method.title()}:")
                best_idea = max(ideas, key=lambda x: x.confidence_score or 0)
                print(f"  {best_idea.content}")
    
    # Step 8: Display performance summary
    total_time = time.time() - setup_start
    print(f"\n📊 Performance Summary:")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  💰 LLM cost: ${result.llm_cost:.4f}")
    print(f"  🔢 Ideas generated: {len(result.synthesized_ideas)}")
    print(f"  🏢 Providers used: {', '.join(providers_setup)}")
    
    # Explain advantages
    print("\n💡 Advantages over simple prompting:")
    print("  • Multi-perspective analysis through specialized agents")
    print("  • Structured thinking with QADI methodology") 
    print("  • Intelligent agent selection and fallback")
    print("  • Rich metadata and reasoning trails")
    if use_evolution:
        print("  • Genetic evolution for idea refinement")
    print("  • Robust error handling and timeout management")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_multi_agent_fixed.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_multi_agent_fixed.py "how can AI improve healthcare"')
        print('  uv run python qadi_multi_agent_fixed.py "ways to reduce carbon footprint" --evolve')
        print('\nOptions:')
        print('  --evolve    Run genetic evolution on generated ideas')
        print('\nFixes:')
        print('  • Only initializes providers with valid API keys')
        print('  • Adds timeout handling to prevent hanging')
        print('  • Provides fallback to Google-only mode')
        print('  • Shows detailed progress feedback')
    else:
        # Parse arguments
        args = sys.argv[1:]
        use_evolution = False
        
        if '--evolve' in args:
            use_evolution = True
            args.remove('--evolve')
        
        prompt = " ".join(args)
        asyncio.run(run_full_qadi_system_fixed(prompt, use_evolution))