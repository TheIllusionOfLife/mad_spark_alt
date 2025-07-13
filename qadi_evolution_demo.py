#!/usr/bin/env python3
"""
QADI Evolution Demo - Showcase genetic evolution on QADI-generated ideas
Usage: uv run python qadi_evolution_demo.py "Your question here"

This demonstrates how Mad Spark Alt goes beyond simple prompting by evolving ideas
through genetic algorithms to find better solutions.
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

async def run_qadi_evolution_demo(prompt: str):
    """Demonstrate the full QADI + Evolution pipeline."""
    from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
    from mad_spark_alt.core.llm_provider import setup_llm_providers
    from mad_spark_alt.evolution import GeneticAlgorithm, EvolutionConfig, EvolutionRequest, SelectionStrategy
    
    # Check for API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    if not google_key:
        print("❌ No Google API key found in .env")
        return
    
    print(f"🧬 QADI + GENETIC EVOLUTION DEMO")
    print(f"📝 Problem: {prompt}")
    print("=" * 70)
    
    # Setup
    await setup_llm_providers(google_api_key=google_key)
    orchestrator = SmartQADIOrchestrator(auto_setup=True)
    
    # Phase 1: QADI Idea Generation
    print("\n🧠 PHASE 1: QADI Multi-Agent Idea Generation")
    print("-" * 50)
    
    cycle_config = {
        "max_ideas_per_method": 3,  # More ideas for evolution
        "require_reasoning": True,
    }
    
    qadi_start = time.time()
    result = await orchestrator.run_qadi_cycle(
        problem_statement=prompt,
        context="Generate diverse, creative solutions suitable for evolution",
        cycle_config=cycle_config
    )
    qadi_time = time.time() - qadi_start
    
    print(f"✓ Generated {len(result.synthesized_ideas)} initial ideas in {qadi_time:.1f}s")
    print(f"💰 QADI Cost: ${result.llm_cost:.4f}")
    
    # Show initial population
    print(f"\n📊 Initial Idea Population:")
    for i, idea in enumerate(result.synthesized_ideas[:6], 1):  # Show first 6
        method = idea.thinking_method.value
        confidence = idea.confidence_score or 0.5
        print(f"  {i}. [{method.upper()[:3]}] {idea.content[:60]}... (conf: {confidence:.2f})")
    
    if len(result.synthesized_ideas) < 5:
        print("\n⚠️  Not enough ideas for meaningful evolution (need at least 5)")
        print("   Genetic algorithms work best with diverse populations")
        return
    
    # Phase 2: Genetic Evolution
    print(f"\n🧬 PHASE 2: Genetic Evolution of Ideas")
    print("-" * 50)
    
    ga = GeneticAlgorithm()
    
    # Configure evolution for demonstration
    evolution_config = EvolutionConfig(
        population_size=12,  # Reasonable size for demo
        generations=4,       # Quick evolution
        mutation_rate=0.25,
        crossover_rate=0.8,
        elite_size=2,
        selection_strategy=SelectionStrategy.TOURNAMENT,
        fitness_weights={
            "creativity_score": 0.4,    # Novelty and originality
            "quality_score": 0.3,       # Technical quality
            "diversity_score": 0.3       # Population diversity
        },
        adaptive_mutation=True,
        diversity_pressure=0.1
    )
    
    evolution_request = EvolutionRequest(
        initial_population=result.synthesized_ideas,
        config=evolution_config,
        context=f"Evolve solutions for: {prompt}",
        constraints=["Must be practical and implementable"],
        target_metrics={"min_fitness": 0.7}
    )
    
    print(f"🔧 Evolution Configuration:")
    print(f"   Population: {evolution_config.population_size}")
    print(f"   Generations: {evolution_config.generations}")
    print(f"   Mutation Rate: {evolution_config.mutation_rate}")
    print(f"   Selection: {evolution_config.selection_strategy.value}")
    
    evo_start = time.time()
    print(f"\n🚀 Starting evolution...")
    
    evo_result = await ga.evolve(evolution_request)
    evo_time = time.time() - evo_start
    
    # Phase 3: Results Analysis
    print(f"\n📈 PHASE 3: Evolution Results")
    print("-" * 50)
    
    if evo_result.success:
        metrics = evo_result.evolution_metrics
        print(f"✓ Evolution completed in {evo_time:.1f}s")
        print(f"🎯 Generations: {evo_result.total_generations}")
        
        if metrics:
            print(f"📊 Fitness Improvement: {metrics.get('fitness_improvement_percent', 0):.1f}%")
            print(f"🎲 Initial Best Fitness: {metrics.get('initial_best_fitness', 0):.3f}")
            print(f"🏆 Final Best Fitness: {metrics.get('final_best_fitness', 0):.3f}")
            print(f"💡 Total Ideas Evaluated: {metrics.get('total_ideas_evaluated', 0)}")
        
        # Show evolution over generations
        if evo_result.generation_snapshots:
            print(f"\n📈 Fitness Evolution by Generation:")
            for i, snapshot in enumerate(evo_result.generation_snapshots):
                diversity = snapshot.diversity_score or 0
                best_fit = snapshot.best_fitness or 0
                avg_fit = snapshot.average_fitness or 0
                print(f"   Gen {i}: Best={best_fit:.3f}, Avg={avg_fit:.3f}, Diversity={diversity:.3f}")
        
        # Display top evolved ideas
        print(f"\n🏆 TOP EVOLVED IDEAS:")
        print("-" * 50)
        
        for i, idea in enumerate(evo_result.best_ideas[:5], 1):
            # Find fitness score for this idea
            fitness_individual = next(
                (ind for ind in evo_result.final_population if ind.idea == idea), 
                None
            )
            fitness = fitness_individual.overall_fitness if fitness_individual else 0
            generation = idea.metadata.get('generation', 'initial')
            
            print(f"\n{i}. Fitness: {fitness:.3f} | Gen: {generation}")
            print(f"   {idea.content}")
            if idea.reasoning:
                print(f"   💭 {idea.reasoning[:100]}...")
    
    else:
        print(f"❌ Evolution failed: {evo_result.error_message}")
    
    # Phase 4: Comparison with Simple Approach
    print(f"\n🎯 WHY THIS BEATS SIMPLE PROMPTING:")
    print("-" * 50)
    
    total_time = qadi_time + evo_time
    total_cost = result.llm_cost + (getattr(evo_result, 'llm_cost', 0) if evo_result.success else 0)
    
    print(f"✨ Advanced Capabilities Demonstrated:")
    print(f"   🧠 Multi-agent QADI reasoning (4 specialized thinking phases)")
    print(f"   🧬 Genetic evolution with fitness-based selection")
    print(f"   🎯 Adaptive mutation and diversity preservation")
    print(f"   📊 Comprehensive fitness evaluation across multiple dimensions")
    print(f"   🏆 Automatic ranking and selection of best solutions")
    print(f"   🔄 Iterative improvement through crossover and mutation")
    
    print(f"\n📊 Final Statistics:")
    print(f"   ⏱️  Total Time: {total_time:.1f}s")
    print(f"   💰 Total Cost: ${total_cost:.4f}")
    print(f"   🎭 Agent Types: {len(set(result.agent_types.values()))}")
    print(f"   💡 Ideas Generated: {len(result.synthesized_ideas)}")
    if evo_result.success:
        print(f"   🧬 Ideas Evolved: {len(evo_result.best_ideas)}")
        print(f"   📈 Fitness Improvement: {metrics.get('fitness_improvement_percent', 0):.1f}%")
    
    print(f"\n🚀 This goes far beyond simple prompting by:")
    print(f"   • Systematic multi-perspective analysis")
    print(f"   • Automated idea evolution and optimization")
    print(f"   • Quantitative fitness evaluation")
    print(f"   • Population diversity management")
    print(f"   • Intelligent crossover of successful concepts")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python qadi_evolution_demo.py "Your question"')
        print('\nExamples:')
        print('  uv run python qadi_evolution_demo.py "improve online learning engagement"')
        print('  uv run python qadi_evolution_demo.py "reduce food waste in restaurants"')
        print('  uv run python qadi_evolution_demo.py "make cities more bike-friendly"')
        print('\nThis demo showcases:')
        print('  • Multi-agent QADI reasoning with LLM-powered agents')
        print('  • Genetic evolution of ideas with fitness optimization')
        print('  • Quantitative creativity evaluation and selection')
        print('  • Advanced capabilities beyond simple prompting')
    else:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(run_qadi_evolution_demo(prompt))