#!/usr/bin/env python3
"""
Test script for the genetic evolution feature of Mad Spark Alt.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mad_spark_alt.core import SmartQADIOrchestrator, IdeaGenerationRequest
from mad_spark_alt.evolution import GeneticAlgorithm, EvolutionConfig, EvolutionRequest

async def test_evolution_system():
    """Test the genetic evolution system with ideas from QADI"""
    
    print("🧬 Mad Spark Alt - Genetic Evolution Test")
    print("=" * 50)
    
    # Step 1: Generate initial ideas using QADI
    print("1️⃣  Generating initial ideas with QADI...")
    
    # Create smart orchestrator (automatically handles all agent registration)
    orchestrator = SmartQADIOrchestrator()
    
    # Ensure agents are ready
    await orchestrator.ensure_agents_ready()
    
    qadi_result = await orchestrator.run_qadi_cycle(
        problem_statement="How can we create more sustainable urban transportation?",
        context="Focus on practical solutions for medium-sized cities",
        cycle_config={"max_ideas_per_method": 3}
    )
    
    print(f"✅ Generated {len(qadi_result.synthesized_ideas)} initial ideas")
    
    # Show the initial ideas
    print("\n📋 Initial Ideas:")
    for i, idea in enumerate(qadi_result.synthesized_ideas[:6], 1):  # Show first 6
        content_preview = idea.content[:80] + "..." if len(idea.content) > 80 else idea.content
        print(f"  {i}. {content_preview}")
    
    # Step 2: Setup genetic algorithm
    print(f"\n2️⃣  Setting up genetic evolution...")
    
    ga = GeneticAlgorithm()
    
    # Configure evolution parameters
    config = EvolutionConfig(
        population_size=8,  # Keep small for demo
        generations=3,      # Quick demo
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_size=2
    )
    
    print(f"   Population size: {config.population_size}")
    print(f"   Generations: {config.generations}")
    print(f"   Mutation rate: {config.mutation_rate}")
    print(f"   Elite size: {config.elite_size}")
    
    # Step 3: Run evolution
    print(f"\n3️⃣  Running genetic evolution...")
    
    evolution_request = EvolutionRequest(
        initial_population=qadi_result.synthesized_ideas,
        config=config,
        constraints=["Must be implementable within 5 years", "Should be cost-effective"]
    )
    
    try:
        evolution_result = await ga.evolve(evolution_request)
        
        if evolution_result.success:
            print(f"✅ Evolution completed successfully!")
            print(f"   Generations: {evolution_result.total_generations}")
            print(f"   Final population size: {len(evolution_result.final_population)}")
            
            # Show fitness improvement
            if 'fitness_improvement_percent' in evolution_result.evolution_metrics:
                improvement = evolution_result.evolution_metrics['fitness_improvement_percent']
                print(f"   Fitness improvement: {improvement:.1f}%")
            
            # Show best evolved ideas
            print(f"\n🏆 Top 3 Evolved Ideas:")
            for i, idea in enumerate(evolution_result.best_ideas[:3], 1):
                content_preview = idea.content[:100] + "..." if len(idea.content) > 100 else idea.content
                print(f"  {i}. {content_preview}")
                
                # Try to show fitness if available
                if i <= len(evolution_result.final_population):
                    individual = evolution_result.final_population[i-1]
                    if hasattr(individual, 'overall_fitness'):
                        print(f"     🎯 Fitness: {individual.overall_fitness:.3f}")
            
            return True
            
        else:
            print(f"❌ Evolution failed: {evolution_result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Evolution error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    print("🌟 Testing Mad Spark Alt Genetic Evolution")
    print("This demonstrates how ideas can be evolved and improved over generations")
    print()
    
    success = await test_evolution_system()
    
    print("\n" + "="*60)
    print("🎯 HOW TO USE GENETIC EVOLUTION:")
    print("="*60)
    
    if success:
        print("""
✅ The evolution system is working! Here's how to use it:

1. Generate initial ideas with QADI:
   - Use QADIOrchestrator to create diverse ideas
   - The more diverse the initial population, the better

2. Configure evolution parameters:
   - population_size: How many ideas to evolve (8-20 for demos)
   - generations: How many evolution cycles (3-10 for testing)
   - mutation_rate: How much to change ideas (0.1-0.3)
   - crossover_rate: How often to combine ideas (0.6-0.8)

3. Run evolution:
   - Ideas are evaluated for creativity and fitness
   - Best ideas survive and reproduce
   - New variations are created through mutation and crossover

4. Review results:
   - Check fitness improvement percentages
   - Review the best evolved ideas
   - Compare with original ideas

Example use cases:
- Refining product ideas for market fit
- Improving creative writing concepts
- Evolving design solutions
- Optimizing problem-solving approaches
""")
    else:
        print("""
⚠️  Evolution test encountered issues. This might be because:
- Missing dependencies for fitness evaluation
- Need to install additional packages
- API keys required for LLM-based evaluation

Basic functionality is still available through QADI and creativity evaluation.
""")

if __name__ == "__main__":
    asyncio.run(main())