#!/usr/bin/env python3
"""
Simplified QADI Analysis Demo

This script experiments with a simpler Phase 1 that just identifies the user's question
rather than trying to extract a "core question".
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.terminal_renderer import render_markdown
    from mad_spark_alt.core.qadi_prompts import QADIPrompts
except ImportError:
    # Fallback if package is not installed
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.terminal_renderer import render_markdown
    from mad_spark_alt.core.qadi_prompts import QADIPrompts


# Create custom prompts with simpler Phase 1
class SimplerQADIPrompts(QADIPrompts):
    """QADI prompts with simplified Phase 1."""
    
    @staticmethod
    def get_questioning_prompt(user_input: str) -> str:
        """Get a much simpler prompt for Phase 1."""
        return f"""What is the user asking?

User's input:
{user_input}

State their question clearly and directly. If they made a statement, rephrase it as the implied question.
Format: "Q: [The user's question]"
"""


# Override the questioning prompt with a simpler version
class SimplerQADIOrchestrator(SimpleQADIOrchestrator):
    """QADI orchestrator with simplified Phase 1."""
    
    def __init__(self, temperature_override: Optional[float] = None) -> None:
        super().__init__(temperature_override)
        # Use custom prompts
        self.prompts = SimplerQADIPrompts()


async def run_qadi_analysis(
    user_input: str, 
    temperature: Optional[float] = None, 
    verbose: bool = False,
    evolve: bool = False,
    generations: int = 3,
    population: int = 12
) -> None:
    """Run QADI analysis with simplified Phase 1 and optional evolution."""

    print("🧠 Simplified QADI Analysis")
    print("=" * 50)
    print(f"\n📝 User Input: {user_input}")

    # Check for API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("\n❌ Error: GOOGLE_API_KEY not found in environment")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        return

    # Create orchestrator with optional temperature override
    orchestrator = SimplerQADIOrchestrator(temperature_override=temperature)

    if temperature:
        print(f"🌡️  Temperature override: {temperature}")

    print("\n" + "─" * 50)
    start_time = time.time()

    try:
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(user_input)

        # Display results
        print("\n## 🎯 Phase 1: User's Question\n")
        render_markdown(f"**Q:** {result.core_question}")

        print("\n## 💡 Phase 2: Possible Approaches\n")
        for i, hypothesis in enumerate(result.hypotheses):
            render_markdown(f"**Approach {i+1}:** {hypothesis}")

        print("\n## 🔍 Phase 3: Analysis & Recommendation\n")

        # Show evaluation scores if verbose
        if verbose and result.hypothesis_scores:
            print("### Evaluation Scores:\n")
            for i, (_, scores) in enumerate(
                zip(result.hypotheses, result.hypothesis_scores)
            ):
                print(f"**Approach {i+1} Scores:**")
                print(f"  - Impact: {scores.impact:.2f}")
                print(f"  - Feasibility: {scores.feasibility:.2f}")
                print(f"  - Accessibility: {scores.accessibility:.2f}")
                print(f"  - Sustainability: {scores.sustainability:.2f}")
                print(f"  - Scalability: {scores.scalability:.2f}")
                print(f"  - **Overall: {scores.overall:.2f}**")
                print()

        render_markdown(f"### Recommendation\n\n{result.final_answer}")

        if result.action_plan:
            print("\n### Action Plan\n")
            for i, action in enumerate(result.action_plan):
                render_markdown(f"{i+1}. {action}")

        print("\n## ✅ Phase 4: Real-World Examples\n")
        if result.verification_examples:
            for i, example in enumerate(result.verification_examples):
                # Parse example structure for better formatting
                lines = example.split('\n')
                
                # Look for structured format markers
                context_line = None
                application_line = None
                result_line = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('- Context:') or line.startswith('Context:'):
                        context_line = line.replace('- Context:', '').replace('Context:', '').strip()
                    elif line.startswith('- Application:') or line.startswith('Application:'):
                        application_line = line.replace('- Application:', '').replace('Application:', '').strip()
                    elif line.startswith('- Result:') or line.startswith('Result:'):
                        result_line = line.replace('- Result:', '').replace('Result:', '').strip()
                
                # Display with better formatting
                print(f"### Example {i+1}")
                if context_line:
                    render_markdown(f"**Context:** {context_line}")
                    if application_line:
                        render_markdown(f"**Application:** {application_line}")
                    if result_line:
                        render_markdown(f"**Result:** {result_line}")
                else:
                    # Fallback to original format if structure not found
                    render_markdown(example)
                print()  # Add spacing between examples

        if result.verification_conclusion:
            print("\n### Conclusion\n")
            render_markdown(result.verification_conclusion)

        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "─" * 50)
        print(f"\n✅ Analysis completed in {elapsed_time:.1f}s")
        print(f"💰 Total LLM cost: ${result.total_llm_cost:.4f}")
        
        # Evolution phase if requested
        if evolve and result.synthesized_ideas:
            print("\n" + "═" * 50)
            print("🧬 Starting Genetic Evolution...")
            print(f"   Generations: {generations}")
            print(f"   Population: {min(population, len(result.synthesized_ideas))}")
            print("─" * 50)
            
            try:
                from mad_spark_alt.evolution import (
                    EvolutionConfig,
                    EvolutionRequest,
                    GeneticAlgorithm,
                    SelectionStrategy,
                )
                
                # Create genetic algorithm instance
                ga = GeneticAlgorithm()
                
                # Configure evolution with higher mutation rate for diversity
                config = EvolutionConfig(
                    population_size=min(population, len(result.synthesized_ideas)),
                    generations=generations,
                    mutation_rate=0.3,  # Increased from default 0.1 to ensure diversity
                    crossover_rate=0.75,
                    elite_size=2,
                    selection_strategy=SelectionStrategy.TOURNAMENT,
                    parallel_evaluation=True,
                    max_parallel_evaluations=min(8, population, len(result.synthesized_ideas)),
                )
                
                # Create evolution request
                request = EvolutionRequest(
                    initial_population=result.synthesized_ideas[:config.population_size],
                    config=config,
                    context=user_input,
                )
                
                # Run evolution
                evolution_start = time.time()
                evolution_result = await ga.evolve(request)
                evolution_time = time.time() - evolution_start
                
                if evolution_result.success:
                    print(f"\n✅ Evolution completed in {evolution_time:.1f}s")
                    
                    # Show top evolved ideas (deduplicated)
                    print("\n🏆 Top Evolved Ideas:\n")
                    
                    # Deduplicate while maintaining order
                    seen_contents = set()
                    unique_individuals = []
                    for ind in sorted(
                        evolution_result.final_population,
                        key=lambda x: x.overall_fitness,
                        reverse=True,
                    ):
                        normalized_content = ind.idea.content.strip()
                        if normalized_content not in seen_contents:
                            seen_contents.add(normalized_content)
                            unique_individuals.append(ind)
                            if len(unique_individuals) >= 5:  # Show more unique ideas
                                break
                    
                    for i, individual in enumerate(unique_individuals):
                        idea = individual.idea
                        print(f"### {i+1}. Evolved Idea (Fitness: {individual.overall_fitness:.3f})")
                        render_markdown(idea.content)
                        if idea.metadata.get("generation"):
                            print(f"   _Generation: {idea.metadata['generation']}_")
                        print()
                    
                    # Show metrics
                    metrics = evolution_result.evolution_metrics
                    print("📊 Evolution Metrics:")
                    print(f"   • Fitness improvement: {metrics.get('fitness_improvement_percent', 0):.1f}%")
                    print(f"   • Ideas evaluated: {metrics.get('total_ideas_evaluated', 0)}")
                    print(f"   • Best from generation: {metrics.get('best_fitness_generation', 0)}")
                    
                    # Cache performance
                    cache_stats = metrics.get("cache_stats")
                    if cache_stats and cache_stats.get("hits", 0) > 0:
                        print(f"\n💾 Cache Performance:")
                        print(f"   • Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                        print(f"   • LLM calls saved: {cache_stats.get('hits', 0)}")
                    
                    # Update total time and cost
                    total_time = elapsed_time + evolution_time
                    print("\n" + "═" * 50)
                    print(f"✅ Total time (QADI + Evolution): {total_time:.1f}s")
                    print(f"💰 Total cost: ${result.total_llm_cost:.4f}")
                else:
                    print(f"\n❌ Evolution failed: {evolution_result.error_message}")
                    
            except ImportError:
                print("\n❌ Evolution modules not available. Please check installation.")
            except Exception as e:
                print(f"\n❌ Evolution error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run simplified QADI analysis on any input"
    )
    parser.add_argument("input", help="Your question, problem, or topic to analyze")
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        help="Temperature for hypothesis generation (0.0-2.0, default: 0.8)",
        default=None,
        metavar="T",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed evaluation scores"
    )
    parser.add_argument(
        "--evolve", "-e", action="store_true", help="Evolve ideas using genetic algorithm after QADI analysis"
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=3, help="Number of evolution generations (with --evolve)"
    )
    parser.add_argument(
        "--population", "-p", type=int, default=12, help="Population size for evolution (with --evolve)"
    )

    args = parser.parse_args()

    # Validate temperature if provided
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        print(
            f"Error: Temperature must be between 0.0 and 2.0 (got {args.temperature})"
        )
        sys.exit(1)
    
    # Validate evolution arguments are only used with --evolve
    if not args.evolve:
        evolution_args_used = []
        if args.generations != parser.get_default("generations"):
            evolution_args_used.append(f"--generations {args.generations}")
        if args.population != parser.get_default("population"):
            evolution_args_used.append(f"--population {args.population}")
        
        if evolution_args_used:
            print(f"Error: {', '.join(evolution_args_used)} can only be used with --evolve")
            print("Did you mean to add --evolve to enable genetic evolution?")
            sys.exit(1)

    # Load environment variables (optional)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print(
            "Warning: python-dotenv not available, environment variables not loaded from .env file"
        )

    # Initialize LLM providers
    async def main_async() -> None:
        try:
            google_key = os.getenv("GOOGLE_API_KEY")
            if google_key:
                await setup_llm_providers(
                    google_api_key=google_key,
                )
            else:
                print("Warning: GOOGLE_API_KEY not set")
        except Exception as e:
            print(f"Warning: Failed to initialize LLM providers: {e}")

        await run_qadi_analysis(
            args.input, 
            temperature=args.temperature, 
            verbose=args.verbose,
            evolve=args.evolve,
            generations=args.generations,
            population=args.population
        )

    # Run analysis
    asyncio.run(main_async())


if __name__ == "__main__":
    main()