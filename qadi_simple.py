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
    from typing import List
except ImportError:
    # Fallback if package is not installed
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.terminal_renderer import render_markdown
    from mad_spark_alt.core.qadi_prompts import QADIPrompts
    from typing import List


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


def extract_key_solutions(hypotheses: List[str], final_answer: str, action_plan: List[str]) -> List[str]:
    """Extract the top 3 solutions from QADI results."""
    import re
    
    def clean_markdown_text(text: str) -> str:
        """Remove all markdown formatting and clean up text."""
        if not text:
            return ""
        
        # Remove all markdown formatting
        cleaned = text.strip()
        
        # Remove **bold** markers
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove *italic* markers
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
        
        # Remove any remaining asterisks
        cleaned = cleaned.replace('*', '')
        
        # Remove markdown links [text](url)
        cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)
        
        # Clean up extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def extract_main_title(hypothesis: str) -> str:
        """Extract the main title/approach from a hypothesis."""
        cleaned = clean_markdown_text(hypothesis)
        
        # Look for "Approach X:" pattern
        approach_match = re.search(r'Approach \d+:\s*(.+?)(?:\s+This approach|$)', cleaned, re.IGNORECASE)
        if approach_match:
            title = approach_match.group(1).strip()
            # Take only the first sentence if it's very long
            if len(title) > 150:
                first_sentence = title.split('.')[0]
                return first_sentence.strip() if len(first_sentence) > 20 else title[:150]
            return title
        
        # Fallback: take first substantial sentence
        sentences = cleaned.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:  # Must be substantial
                return sentence
        
        # Last resort: take first 100 characters
        return cleaned[:100].strip() if cleaned else ""
    
    solutions = []
    
    # Extract from hypotheses first
    for h in hypotheses[:3]:
        if h and h.strip():
            title = extract_main_title(h)
            if title and len(title) > 10:  # Must be meaningful
                solutions.append(title)
    
    # If we don't have enough solutions, add from action plan
    if len(solutions) < 3:
        for action in action_plan[:3]:
            if len(solutions) < 3 and action and action.strip():
                action_clean = clean_markdown_text(action)
                # Remove numbering from start
                action_clean = re.sub(r'^\d+\.\s*', '', action_clean)
                
                # Take first sentence if it's meaningful
                first_sentence = action_clean.split('.')[0].strip()
                if len(first_sentence) > 20:
                    solutions.append(first_sentence)
                elif len(action_clean) > 20:
                    solutions.append(action_clean[:100].strip())
    
    return solutions[:3]


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
    print("=" * 50 + "\n")
    
    # Display user input clearly
    print(f"📝 User Input: {user_input}\n")
    print("─" * 50)

    # Check for API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("\n❌ Error: GOOGLE_API_KEY not found in environment")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-key-here'")
        return

    # Create orchestrator with optional temperature override
    orchestrator = SimplerQADIOrchestrator(temperature_override=temperature)
    
    start_time = time.time()

    try:
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(user_input)
        
        # Extract key solutions for summary
        key_solutions = extract_key_solutions(
            result.hypotheses or [],
            result.final_answer or "",
            result.action_plan or []
        )
        
        # Show summary first (we'll update this later if evolution provides better solutions)
        initial_solutions = key_solutions.copy() if key_solutions else []
        if initial_solutions:
            print("\n## 💡 Initial Solutions (Hypothesis Generation)\n")
            for i, solution in enumerate(initial_solutions, 1):
                # Clean up solution text but don't truncate - show full solution
                solution_clean = solution.strip()
                # Only truncate if extremely long (over 400 characters)
                if len(solution_clean) > 400:
                    # Find a good breaking point (sentence or clause)
                    truncate_at = solution_clean.find('. ', 300)
                    if truncate_at > 0:
                        solution_clean = solution_clean[:truncate_at + 1] + "..."
                    else:
                        solution_clean = solution_clean[:397] + "..."
                
                # Extract approach type (Personal, Collective, Systemic) if present
                approach_label = ""
                if "Individual" in solution_clean or "Personal" in solution_clean:
                    approach_label = "**Personal Approach:** "
                elif "Community" in solution_clean or "Collective" in solution_clean or "Team" in solution_clean:
                    approach_label = "**Collaborative Approach:** "
                elif "System" in solution_clean or "Organization" in solution_clean or "Structural" in solution_clean:
                    approach_label = "**Systemic Approach:** "
                else:
                    approach_label = f"**Approach {i}:** "
                
                render_markdown(f"{approach_label}{solution_clean}")
            
            if evolve:
                print("\n*Note: These initial ideas will be refined through AI evolution...*")
            print()
        
        # Show phases in verbose mode
        if verbose:
            print("\n## 🎯 Phase 1: Question Clarification\n")
            render_markdown(f"**Core Question:** {result.core_question}")

            print("\n## 💡 Phase 2: Hypothesis Generation (Abduction)\n")
            for i, hypothesis in enumerate(result.hypotheses):
                # Try to identify approach type
                if "individual" in hypothesis.lower() or "personal" in hypothesis.lower():
                    label = "Personal"
                elif "community" in hypothesis.lower() or "collective" in hypothesis.lower() or "team" in hypothesis.lower():
                    label = "Collaborative"
                elif "system" in hypothesis.lower() or "organization" in hypothesis.lower() or "structural" in hypothesis.lower():
                    label = "Systemic"
                else:
                    label = f"H{i+1}"
                render_markdown(f"**{label} Approach:** {hypothesis}")

            print("\n## 🔍 Phase 3: Logical Analysis (Deduction)\n")

            # Show evaluation scores if verbose
            if result.hypothesis_scores:
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

        # Main output - focused on solutions
        print("\n## 🔍 Analysis: Comparing the Approaches\n")
        render_markdown(result.final_answer)

        if result.action_plan:
            print("\n## 🎯 Your Recommended Path (Final Synthesis)\n")
            for i, action in enumerate(result.action_plan):
                render_markdown(f"**{i+1}.** {action}")

        # Examples section - make it more concise
        if result.verification_examples and (verbose or len(result.verification_examples) <= 2):
            print("\n## 💡 Real-World Examples\n")
            
            # Show only first 2 examples in non-verbose mode
            examples_to_show = result.verification_examples if verbose else result.verification_examples[:2]
            
            for i, example in enumerate(examples_to_show, 1):
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
                
                # Display concisely
                print(f"**Example {i}**")
                if context_line and application_line:
                    # Shorten if too long
                    if len(context_line) > 150:
                        context_line = context_line[:147] + "..."
                    if len(application_line) > 150:
                        application_line = application_line[:147] + "..."
                    render_markdown(f"• {context_line}")
                    render_markdown(f"• {application_line}")
                    if result_line and len(result_line) < 150:
                        render_markdown(f"• **Result:** {result_line}")
                else:
                    # Fallback - show first 200 chars
                    short_example = example[:200] + "..." if len(example) > 200 else example
                    render_markdown(short_example)
                print()  # Add spacing between examples

        # Show conclusion only in verbose mode
        if verbose and result.verification_conclusion:
            print("\n### Conclusion\n")
            render_markdown(result.verification_conclusion)

        # Compact summary at the end
        elapsed_time = time.time() - start_time
        if not evolve:  # Show summary now if not evolving
            print("\n" + "─" * 50)
            print(f"⏱️  Time: {elapsed_time:.1f}s | 💰 Cost: ${result.total_llm_cost:.4f}")
        
        # Evolution phase if requested
        if evolve and result.synthesized_ideas:
            print("\n" + "═" * 50)
            print(f"🧬 Evolving ideas ({generations} generations, {min(population, len(result.synthesized_ideas))} population)...")
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
                # For small populations, increase mutation rate even more
                actual_population = min(population, len(result.synthesized_ideas))
                mutation_rate = 0.5 if actual_population <= 3 else 0.3
                
                config = EvolutionConfig(
                    population_size=actual_population,
                    generations=generations,
                    mutation_rate=mutation_rate,  # Higher mutation for small populations
                    crossover_rate=0.75,
                    elite_size=min(1, actual_population - 1),  # Smaller elite for tiny populations
                    selection_strategy=SelectionStrategy.TOURNAMENT,
                    parallel_evaluation=True,
                    max_parallel_evaluations=min(8, actual_population),
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
                    
                    # Update the best solutions section with evolved ideas
                    print("\n" + "═" * 50)
                    print("## 🧬 Evolution Results: Enhanced Solutions\n")
                    print("*The initial hypotheses have been evolved and refined:*\n")
                    
                    # Use fuzzy deduplication to show more diverse ideas
                    from difflib import SequenceMatcher
                    
                    def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
                        """Check if two strings are similar above threshold."""
                        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
                    
                    # Get total population size
                    total_population = len(evolution_result.final_population)
                    
                    # Collect unique ideas with fuzzy matching
                    unique_individuals = []
                    for ind in sorted(
                        evolution_result.final_population,
                        key=lambda x: x.overall_fitness,
                        reverse=True,
                    ):
                        normalized_content = ind.idea.content.strip()
                        
                        # Check if this idea is too similar to any already selected
                        is_duplicate = False
                        for existing in unique_individuals:
                            similarity = SequenceMatcher(None, normalized_content.lower(), existing.idea.content.strip().lower()).ratio()
                            if similarity > 0.85:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            unique_individuals.append(ind)
                            if len(unique_individuals) >= 8:  # Show more ideas
                                break
                    
                    # If we have very few unique ideas, be more permissive
                    if len(unique_individuals) < 3:
                        unique_individuals = []
                        for ind in sorted(
                            evolution_result.final_population,
                            key=lambda x: x.overall_fitness,
                            reverse=True,
                        ):
                            normalized_content = ind.idea.content.strip()
                            
                            # Use much lower threshold for more diversity (allow similar ideas)
                            is_duplicate = False
                            for existing in unique_individuals:
                                if is_similar(normalized_content, existing.idea.content.strip(), 0.95):  # Very strict
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                unique_individuals.append(ind)
                                if len(unique_individuals) >= min(5, total_population):
                                    break
                    
                    # If still too few, try to extract diverse ideas from the population
                    if len(unique_individuals) < 2:
                        # Sort by fitness but also try to get diverse thinking methods
                        all_sorted = sorted(
                            evolution_result.final_population,
                            key=lambda x: x.overall_fitness,
                            reverse=True,
                        )
                        
                        # Try to get ideas from different thinking methods
                        seen_methods = set()
                        unique_individuals = []
                        
                        for ind in all_sorted:
                            method = ind.idea.thinking_method.value if hasattr(ind.idea.thinking_method, 'value') else str(ind.idea.thinking_method)
                            
                            # If we haven't seen this method yet, or we have very few ideas, include it
                            if method not in seen_methods or len(unique_individuals) < 3:
                                unique_individuals.append(ind)
                                seen_methods.add(method)
                                if len(unique_individuals) >= 5:
                                    break
                        
                        # Final fallback: just take top N with slight content variations highlighted
                        if len(unique_individuals) < 2:
                            unique_individuals = all_sorted[:min(3, len(all_sorted))]
                    
                    # Display evolved ideas clearly as the new best solutions
                    displayed_contents = set()
                    display_count = 0
                    
                    for i, individual in enumerate(unique_individuals):
                        idea = individual.idea
                        content_hash = idea.content.strip().lower()[:100]  # Simple hash for comparison
                        
                        # Skip if we've already displayed very similar content
                        if content_hash in displayed_contents and display_count >= 3:
                            continue
                            
                        displayed_contents.add(content_hash)
                        display_count += 1
                        
                        print(f"**{display_count}. Enhanced Approach**")
                        render_markdown(idea.content)
                        print()
                        
                        if display_count >= 3:  # Limit to 3 displayed ideas
                            break
                    
                    # If we ended up with very similar ideas, add a note
                    if len(displayed_contents) == 1 and len(unique_individuals) > 1:
                        print("*Note: Evolution produced similar approaches, suggesting strong convergence on this solution.*")
                        print()
                    
                    # Show how these solutions were developed
                    print("\n## 📊 Development Process & Metrics\n")
                    print("**Process:** QADI Analysis → Genetic Evolution → Enhanced Solutions")
                    print(f"• **Initial Analysis:** {elapsed_time:.1f}s using QADI methodology")
                    print(f"• **Evolution Process:** {evolution_time:.1f}s across {generations} generations")
                    
                    # Show compact metrics
                    metrics = evolution_result.evolution_metrics
                    print(f"• **Improvement:** {metrics.get('fitness_improvement_percent', 0):.1f}% fitness increase")
                    print(f"• **Evaluation:** {metrics.get('total_ideas_evaluated', 0)} total ideas tested")
                    
                    # Show cache stats only in verbose mode
                    if verbose:
                        cache_stats = metrics.get("cache_stats")
                        if cache_stats and cache_stats.get("hits", 0) > 0:
                            print(f"\n💾 Cache Performance:")
                            print(f"   • Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                            print(f"   • LLM calls saved: {cache_stats.get('hits', 0)}")
                    
                    # Final summary with total time and cost
                    total_time = elapsed_time + evolution_time
                    print("\n" + "═" * 50)
                    print(f"⏱️  Total time: {total_time:.1f}s | 💰 Total cost: ${result.total_llm_cost:.4f}")
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