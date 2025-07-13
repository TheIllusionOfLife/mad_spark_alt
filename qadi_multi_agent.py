#!/usr/bin/env python3
"""
QADI Multi-Agent - Full-featured QADI system with optimized performance
Usage: uv run python qadi_multi_agent.py "Your question here"

This version uses the full multi-agent system with intelligent orchestration,
providing much richer results than a simple prompt wrapper.
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

async def run_full_qadi_system(prompt: str, use_evolution: bool = False):
    """Run the full QADI multi-agent system with optional genetic evolution."""
    from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
    from mad_spark_alt.core.smart_registry import smart_registry
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.evolution import GeneticAlgorithm, EvolutionConfig, EvolutionRequest
    
    # Check for API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not any([google_key, openai_key, anthropic_key]):
        print("❌ No API keys found in .env (need at least one of GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)")
        return
    
    print(f"📝 {prompt}")
    print("=" * 70)
    
    # Setup LLM providers
    print("🔧 Setting up LLM providers...", end='', flush=True)
    start_time = time.time()
    
    await setup_llm_providers(
        google_api_key=google_key,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key
    )
    print(" ✓")
    
    # Create smart orchestrator
    orchestrator = SmartQADIOrchestrator(auto_setup=True)
    
    # Configure for optimized performance
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
    
    # Run QADI cycle
    cycle_start = time.time()
    result = await orchestrator.run_qadi_cycle(
        problem_statement=prompt,
        context="Generate practical, actionable insights",
        cycle_config=cycle_config
    )
    cycle_time = time.time() - cycle_start
    
    # Display agent types used
    print(f"\n⚙️  Agent Configuration:")
    for method, agent_type in result.agent_types.items():
        icon = "🤖" if agent_type == "LLM" else "📝"
        print(f"  {icon} {method.title()}: {agent_type} agent")
    
    print(f"\n⏱️  QADI cycle completed in {cycle_time:.1f}s")
    
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
    
    # Optional: Run genetic evolution
    if use_evolution and len(result.synthesized_ideas) >= 5:
        print("\n🧬 Running Genetic Evolution...")
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
        evo_result = await ga.evolve(evolution_request)
        evo_time = time.time() - evo_start
        
        print(f"  ✓ Evolution completed in {evo_time:.1f}s")
        if evo_result.evolution_metrics:
            improvement = evo_result.evolution_metrics.get('fitness_improvement_percent', 0)
            print(f"  📈 Fitness improved by {improvement:.1f}%")
    
    # Display conclusion if available
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
    
    # Display costs
    total_time = time.time() - start_time
    print(f"\n📊 Performance Summary:")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  💰 LLM cost: ${result.llm_cost:.4f}")
    print(f"  🔢 Ideas generated: {len(result.synthesized_ideas)}")
    
    # Explain advantages over simple prompt
    print("\n💡 Advantages over simple prompting:")
    print("  • Multi-perspective analysis through specialized agents")
    print("  • Structured thinking with QADI methodology")
    print("  • Intelligent agent selection and fallback")
    print("  • Rich metadata and reasoning trails")
    if use_evolution:
        print("  • Genetic evolution for idea refinement")
    print("  • Extensible with custom agents and evaluators")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_multi_agent.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_multi_agent.py "how can AI improve healthcare"')
        print('  uv run python qadi_multi_agent.py "ways to reduce carbon footprint" --evolve')
        print('\nOptions:')
        print('  --evolve    Run genetic evolution on generated ideas')
    else:
        # Parse arguments
        args = sys.argv[1:]
        use_evolution = False
        
        if '--evolve' in args:
            use_evolution = True
            args.remove('--evolve')
        
        prompt = " ".join(args)
        asyncio.run(run_full_qadi_system(prompt, use_evolution))