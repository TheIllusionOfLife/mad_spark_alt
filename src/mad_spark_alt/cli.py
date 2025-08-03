"""
Command-line interface for Mad Spark Alt.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core import (
    CreativityEvaluator,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationSummary,
    ModelOutput,
    OutputType,
    registry,
    setup_llm_providers,
)
from .core.llm_provider import LLMProvider, llm_manager
from .core.json_utils import format_llm_cost
from .core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from .evolution import (
    DiversityMethod,
    EvolutionConfig,
    EvolutionRequest,
    GeneticAlgorithm,
    SelectionStrategy,
)
from .layers.quantitative import DiversityEvaluator, QualityEvaluator

console = Console()


def _get_semantic_operator_status() -> str:
    """Get status of semantic operators (ENABLED/DISABLED)."""
    if LLMProvider.GOOGLE in llm_manager.providers:
        return "Semantic operators: ENABLED"
    else:
        return "Semantic operators: DISABLED (traditional operators only)"


def _format_idea_for_display(
    content: str, max_length: int = 200, wrap_lines: bool = False
) -> str:
    """Format idea content for display with smart truncation.
    
    Args:
        content: The idea content to format
        max_length: Maximum length before truncation
        wrap_lines: Whether to support multi-line display
        
    Returns:
        Formatted content string
    """
    if len(content) <= max_length:
        return content
    
    # Find a good truncation point at word boundary
    truncated = content[:max_length]
    
    # Look for last complete word
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If we found a space reasonably close to the end
        truncated = truncated[:last_space]
    
    # Also check for punctuation as good breaking points
    for punct in ['.', ',', ';', ')', ']']:
        punct_pos = truncated.rfind(punct)
        if punct_pos > max_length * 0.8:
            truncated = truncated[:punct_pos + 1]
            break
    
    return truncated.strip() + "..."


def _create_evolution_results_table() -> Table:
    """Create a table for evolution results with proper column configuration."""
    table = Table(title="🏆 Top Evolved Ideas")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Idea", style="white", width=None)  # No width limit
    table.add_column("Fitness", style="green", width=8)
    table.add_column("Gen", style="yellow", width=5)
    return table


# Evolution timeout constants
_BASE_TIMEOUT_SECONDS = 120.0  # Minimum 2 minutes
_SECONDS_PER_EVALUATION_ESTIMATE = 20  # Estimate per evaluation
_MAX_TIMEOUT_SECONDS = 600.0  # Maximum 10 minutes


def calculate_evolution_timeout(generations: int, population: int) -> float:
    """
    Calculate adaptive timeout based on evolution complexity.

    Args:
        generations: Number of generations
        population: Population size

    Returns:
        Timeout in seconds (min 120s, max 600s)
    """
    # Use 25s per evaluation estimate (increased for batch optimization safety)
    estimated_time = generations * population * 25
    return min(max(_BASE_TIMEOUT_SECONDS, estimated_time), _MAX_TIMEOUT_SECONDS)


def load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value.strip('"').strip("'")
        except Exception as e:
            # Log error but don't fail - env vars might be set elsewhere
            logging.warning(f"Failed to load .env file: {e}")


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def register_default_evaluators() -> None:
    """Register default evaluators."""
    registry.register(DiversityEvaluator)
    registry.register(QualityEvaluator)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Mad Spark Alt - AI Creativity Evaluation System."""
    # Load environment variables from .env file first
    load_env_file()
    setup_logging(verbose)
    register_default_evaluators()

    # Initialize Google LLM provider if API key is available
    google_key = os.getenv("GOOGLE_API_KEY")

    # Only initialize if we have Google API key
    if google_key:

        async def init_llm() -> None:
            await setup_llm_providers(
                google_api_key=google_key,
            )

        try:
            # Check if event loop is running (e.g., in Jupyter notebooks)
            try:
                loop = asyncio.get_running_loop()
                # Event loop is running, we can't use run_until_complete
                if verbose:
                    console.print(
                        "[yellow]Warning: Cannot initialize LLM providers in running event loop[/yellow]"
                    )
            except RuntimeError:
                # No event loop is running, we can create one
                try:
                    asyncio.run(init_llm())
                except Exception as e:
                    # Log specific LLM initialization errors
                    if verbose:
                        console.print(
                            f"[red]Error: LLM provider initialization failed: {e}[/red]"
                        )
        except Exception as e:
            # Catch-all for unexpected errors
            if verbose:
                console.print(
                    f"[red]Unexpected error during LLM initialization: {e}[/red]"
                )
    elif verbose:
        console.print(
            "[yellow]Info: GOOGLE_API_KEY not found, LLM features disabled[/yellow]"
        )


@main.command()
def list_evaluators() -> None:
    """List all registered evaluators."""
    evaluators = registry.list_evaluators()

    if not evaluators:
        console.print("[yellow]No evaluators registered.[/yellow]")
        return

    table = Table(title="Registered Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Layer", style="green")
    table.add_column("Supported Types", style="blue")

    for name, info in evaluators.items():
        table.add_row(name, info["layer"], ", ".join(info["supported_output_types"]))

    console.print(table)


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--model", "-m", default="test-model", help="Model name for the output")
@click.option(
    "--output-type",
    "-t",
    type=click.Choice(["text", "code"]),
    default="text",
    help="Output type",
)
@click.option("--evaluators", "-e", help="Comma-separated list of evaluators to use")
@click.option(
    "--layers",
    "-l",
    help="Comma-separated list of layers (quantitative,llm_judge,human)",
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def evaluate(
    text: Optional[str],
    file: Optional[str],
    model: str,
    output_type: str,
    evaluators: Optional[str],
    layers: Optional[str],
    output: Optional[str],
    output_format: str,
) -> None:
    """Evaluate creativity of AI output."""

    # Get input text
    if file:
        with open(file, "r") as f:
            input_text = f.read()
    elif text:
        input_text = text
    elif not sys.stdin.isatty():
        # Read from stdin if available
        input_text = sys.stdin.read().strip()
    else:
        console.print(
            "[red]Error: Provide text via argument, --file option, or stdin[/red]"
        )
        sys.exit(1)

    # Parse output type
    try:
        output_type_enum = OutputType(output_type)
    except ValueError:
        console.print(f"[red]Error: Invalid output type '{output_type}'[/red]")
        sys.exit(1)

    # Parse layers
    target_layers = []
    if layers:
        layer_names = [l.strip() for l in layers.split(",")]
        for layer_name in layer_names:
            try:
                target_layers.append(EvaluationLayer(layer_name))
            except ValueError:
                console.print(f"[red]Error: Invalid layer '{layer_name}'[/red]")
                sys.exit(1)

    # Create model output
    model_output = ModelOutput(
        content=input_text,
        output_type=output_type_enum,
        model_name=model,
    )

    # Create evaluation request
    request = EvaluationRequest(
        outputs=[model_output],
        target_layers=target_layers,
    )

    # Run evaluation
    asyncio.run(_run_evaluation(request, output, output_format))


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--model", "-m", default="test-model", help="Model name for the outputs")
@click.option(
    "--output-type",
    "-t",
    type=click.Choice(["text", "code"]),
    default="text",
    help="Output type",
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def batch_evaluate(
    files: List[str],
    model: str,
    output_type: str,
    output: Optional[str],
    output_format: str,
) -> None:
    """Evaluate creativity of multiple AI outputs from files."""

    # Parse output type
    try:
        output_type_enum = OutputType(output_type)
    except ValueError:
        console.print(f"[red]Error: Invalid output type '{output_type}'[/red]")
        sys.exit(1)

    # Read all files
    model_outputs = []
    for file_path in files:
        with open(file_path, "r") as f:
            content = f.read()

        model_outputs.append(
            ModelOutput(
                content=content,
                output_type=output_type_enum,
                model_name=model,
                metadata={"source_file": file_path},
            )
        )

    # Create evaluation request
    request = EvaluationRequest(outputs=model_outputs)

    # Run evaluation
    asyncio.run(_run_evaluation(request, output, output_format))


@main.command()
@click.argument("prompt")
@click.option(
    "--responses",
    "-r",
    multiple=True,
    required=True,
    help="Multiple responses to compare",
)
@click.option("--model", "-m", default="test-model", help="Model name")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def compare(
    prompt: str,
    responses: List[str],
    model: str,
    output: Optional[str],
) -> None:
    """Compare creativity of multiple responses to the same prompt."""

    if len(responses) < 2:
        console.print("[red]Error: Need at least 2 responses to compare[/red]")
        sys.exit(1)

    # Create model outputs
    model_outputs = []
    for i, response in enumerate(responses):
        model_outputs.append(
            ModelOutput(
                content=response,
                output_type=OutputType.TEXT,
                model_name=model,
                prompt=prompt,
                metadata={"response_index": i},
            )
        )

    # Create evaluation request
    request = EvaluationRequest(
        outputs=model_outputs, task_context=f"Comparing responses to: {prompt}"
    )

    # Run evaluation with focus on diversity
    asyncio.run(_run_evaluation(request, output, "table", compare_mode=True))


async def _run_evaluation(
    request: EvaluationRequest,
    output_file: Optional[str],
    output_format: str,
    compare_mode: bool = False,
) -> None:
    """Run the evaluation and display results."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("Evaluating creativity...", total=None)

        # Create evaluator and run evaluation
        evaluator = CreativityEvaluator()
        summary = await evaluator.evaluate(request)

        progress.update(task, completed=True)

    # Display results
    if output_format == "json":
        _display_json_results(summary, output_file)
    else:
        _display_table_results(summary, compare_mode)

        if output_file:
            _save_json_results(summary, output_file)


def _display_json_results(
    summary: EvaluationSummary, output_file: Optional[str]
) -> None:
    """Display results in JSON format."""
    result_data = _summary_to_dict(summary)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
        console.print(f"[green]Results saved to {output_file}[/green]")
    else:
        console.print(JSON.from_data(result_data))


def _display_table_results(
    summary: EvaluationSummary, compare_mode: bool = False
) -> None:
    """Display results in table format."""

    # Summary panel
    summary_text = f"""
    [bold]Evaluation Summary[/bold]
    
    • Total outputs: {summary.total_outputs}
    • Total evaluators: {summary.total_evaluators}
    • Execution time: {summary.execution_time:.2f}s
    • Overall creativity score: {summary.get_overall_creativity_score():.3f}
    """

    console.print(Panel(summary_text, title="Results", expand=False))

    # Results by layer
    for layer, results in summary.layer_results.items():
        if not results:
            continue

        table = Table(title=f"{layer.value.title()} Layer Results")
        table.add_column("Output", style="cyan")
        table.add_column("Evaluator", style="green")
        table.add_column("Scores", style="yellow")

        for result in results:
            # Format scores
            score_strs = []
            for metric, score in result.scores.items():
                if isinstance(score, float):
                    score_strs.append(f"{metric}: {score:.3f}")
                else:
                    score_strs.append(f"{metric}: {score}")

            # Get output identifier
            output_id = result.metadata.get("output_index", 0)
            if compare_mode:
                output_id = f"Response {output_id + 1}"
            else:
                output_id = f"Output {output_id + 1}"

            table.add_row(str(output_id), result.evaluator_name, "\n".join(score_strs))

        console.print(table)
        console.print()


def _save_json_results(summary: EvaluationSummary, output_file: str) -> None:
    """Save results to JSON file."""
    result_data = _summary_to_dict(summary)
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    console.print(f"[green]Results saved to {output_file}[/green]")


def _summary_to_dict(summary: EvaluationSummary) -> Dict[str, Any]:
    """Convert evaluation summary to dictionary."""
    return {
        "request_id": summary.request_id,
        "total_outputs": summary.total_outputs,
        "total_evaluators": summary.total_evaluators,
        "execution_time": summary.execution_time,
        "overall_creativity_score": summary.get_overall_creativity_score(),
        "aggregate_scores": summary.aggregate_scores,
        "layer_results": {
            layer.value: [
                {
                    "evaluator_name": result.evaluator_name,
                    "scores": result.scores,
                    "explanations": result.explanations,
                    "metadata": result.metadata,
                }
                for result in results
            ]
            for layer, results in summary.layer_results.items()
        },
    }


@main.command()
@click.argument("problem")
@click.option("--context", "-c", help="Additional context for the problem")
@click.option("--generations", "-g", default=2, help="Number of evolution generations")
@click.option("--population", "-p", default=5, help="Population size for evolution")
@click.option(
    "--temperature",
    "-t",
    type=click.FloatRange(0.0, 2.0),
    help="Temperature for hypothesis generation (0.0-2.0, default: 0.8)",
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option(
    "--traditional",
    is_flag=True,
    help="Use traditional operators instead of semantic operators"
)
@click.option(
    "--diversity-method",
    type=click.Choice(["jaccard", "semantic"], case_sensitive=False),
    default="jaccard",
    help="Diversity calculation method: jaccard (fast, word-based) or semantic (slower, embedding-based with Gemini API)"
)
def evolve(
    problem: str,
    context: Optional[str],
    generations: int,
    population: int,
    temperature: Optional[float],
    output: Optional[str],
    traditional: bool,
    diversity_method: str,
) -> None:
    """Evolve ideas using QADI methodology + Genetic Algorithm.

    Examples:
      mad-spark evolve "How can we reduce food waste?"
      mad-spark evolve "Improve remote work" --context "Focus on team collaboration"
      mad-spark evolve "Climate solutions" --generations 2
      mad-spark evolve "New product ideas" --temperature 1.5
      mad-spark evolve "AI applications" --diversity-method semantic
    """
    import os

    # Validate evolution parameters
    if population < 2 or population > 10:
        console.print(f"[red]Error: Population size must be between 2 and 10 (got {population})[/red]")
        console.print("\n[yellow]Valid range:[/yellow] 2 to 10")
        console.print("Example: mad-spark evolve \"Your question\" --population 5")
        sys.exit(1)
    
    if generations < 2 or generations > 5:
        console.print(f"[red]Error: Generations must be between 2 and 5 (got {generations})[/red]")
        console.print("\n[yellow]Valid range:[/yellow] 2 to 5")
        console.print("Example: mad-spark evolve \"Your question\" --generations 3")
        sys.exit(1)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[red]Error: GOOGLE_API_KEY not found[/red]")
        console.print("\n[yellow]To fix this:[/yellow]")
        console.print(
            "1. Get a Google API key from: https://makersuite.google.com/app/apikey"
        )
        console.print("2. Set environment variable: export GOOGLE_API_KEY='your-key'")
        console.print("3. Or create .env file: echo 'GOOGLE_API_KEY=your-key' > .env")
        sys.exit(1)

    # Set defaults
    if not context:
        context = "Consider practical, implementable solutions"


    temp_display = f" | Temperature: {temperature}" if temperature else ""
    operators_display = "Traditional" if traditional else "Semantic (LLM-powered)"
    diversity_display = "Semantic (embedding-based)" if diversity_method.lower() == "semantic" else "Jaccard (word-based)"
    console.print(
        Panel(
            f"[bold blue]Evolution Pipeline[/bold blue]\n"
            f"Problem: {problem}\n"
            f"Context: {context}\n"
            f"Generations: {generations} | Population: {population}{temp_display}\n"
            f"Operators: {operators_display} | Diversity: {diversity_display}"
        )
    )

    # Run the evolution pipeline
    asyncio.run(
        _run_evolution_pipeline(
            problem, context, generations, population, temperature, output, traditional, diversity_method
        )
    )


async def _run_evolution_pipeline(
    problem: str,
    context: str,
    generations: int,
    population: int,
    temperature: Optional[float],
    output_file: Optional[str],
    traditional: bool,
    diversity_method: str,
) -> None:
    """Run the evolution pipeline with progress tracking."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Phase 1: QADI Generation
        qadi_task = progress.add_task("Generating ideas with QADI...", total=None)

        try:
            orchestrator = SimpleQADIOrchestrator(
                temperature_override=temperature,
                num_hypotheses=max(5, population)  # Generate at least as many as requested population
            )

            qadi_result = await asyncio.wait_for(
                orchestrator.run_qadi_cycle(
                    user_input=problem,
                    context=context,
                ),
                timeout=90.0,
            )

            initial_ideas = qadi_result.synthesized_ideas
            progress.update(qadi_task, completed=True)

            if not initial_ideas:
                console.print("[red]No ideas generated by QADI[/red]")
                return

            console.print(
                f"[green]✅ Generated {len(initial_ideas)} initial ideas[/green]"
            )
            console.print(
                f"[dim]💰 LLM Cost: {format_llm_cost(qadi_result.total_llm_cost)}[/dim]"
            )

            # Phase 2: Evolution
            evolution_task = progress.add_task(
                f"Evolving ideas ({generations} generations)...", total=None
            )

            # Get LLM provider for semantic operators unless traditional is specified
            if traditional:
                llm_provider = None
                console.print("[dim]🧬 Using traditional evolution operators[/dim]")
            else:
                from mad_spark_alt.core.llm_provider import get_google_provider
                llm_provider = get_google_provider()
                console.print("[dim]🧬 Using semantic evolution operators (LLM-powered)[/dim]")
            
            ga = GeneticAlgorithm(
                use_cache=True,
                cache_ttl=3600,
                checkpoint_dir=".evolution_checkpoints",
                checkpoint_interval=1,
                llm_provider=llm_provider,
            )

            config = EvolutionConfig(
                population_size=min(population, len(initial_ideas)),
                generations=generations,
                mutation_rate=0.25,  # Increased from 0.15 for more diversity
                crossover_rate=0.85,  # Increased from 0.75 for more recombination
                elite_size=1,  # Reduced from 2 to allow more diversity - only preserve the best
                selection_strategy=SelectionStrategy.TOURNAMENT,
                parallel_evaluation=True,
                max_parallel_evaluations=min(8, population, len(initial_ideas)),
                diversity_method=DiversityMethod.SEMANTIC if diversity_method.lower() == "semantic" else DiversityMethod.JACCARD,
            )

            request = EvolutionRequest(
                initial_population=initial_ideas[: config.population_size],
                config=config,
                context=context,
            )

            # Calculate adaptive timeout based on evolution complexity
            evolution_timeout = calculate_evolution_timeout(generations, population)
            console.print(f"[dim]⏱️  Evolution timeout: {evolution_timeout:.0f}s[/dim]")
            
            evolution_result = await asyncio.wait_for(ga.evolve(request), timeout=evolution_timeout)

            progress.update(evolution_task, completed=True)

            if evolution_result.success:
                # Display results
                console.print(
                    f"\n[green]✅ Evolution completed in {evolution_result.execution_time:.1f}s[/green]"
                )

                # Show best ideas with deduplication
                table = _create_evolution_results_table()

                # Get unique top individuals by content similarity
                sorted_population = sorted(
                    evolution_result.final_population,
                    key=lambda x: x.overall_fitness,
                    reverse=True,
                )
                
                
                # Deduplicate similar ideas
                unique_individuals = []
                seen_contents: List[str] = []
                
                for individual in sorted_population:
                    content = individual.idea.content.lower().strip()
                    
                    # Check if this content is too similar to any already seen
                    is_duplicate = False
                    for seen_content in seen_contents:
                        # Simple similarity check using word overlap
                        words1 = set(content.split())
                        words2 = set(seen_content.split())
                        
                        if len(words1) > 0 and len(words2) > 0:
                            intersection = len(words1.intersection(words2))
                            union = len(words1.union(words2))
                            similarity = intersection / union
                            
                            if similarity > 0.6:  # 60% similarity threshold
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        unique_individuals.append(individual)
                        seen_contents.append(content)
                        
                        if len(unique_individuals) >= 5:
                            break

                for i, individual in enumerate(unique_individuals):
                    idea = individual.idea
                    # Don't truncate ideas - show full content
                    formatted_content = idea.content  # No truncation
                    table.add_row(
                        str(i + 1),
                        formatted_content,
                        f"{individual.overall_fitness:.3f}",
                        str(idea.metadata.get("generation", 0)),
                    )

                console.print(table)

                # Show metrics
                metrics = evolution_result.evolution_metrics
                console.print(f"\n[blue]📊 Results:[/blue]")
                console.print(
                    f"• Fitness improvement: {metrics.get('fitness_improvement_percent', 0):.1f}%"
                )
                console.print(
                    f"• Ideas evaluated: {metrics.get('total_ideas_evaluated', 0)}"
                )
                console.print(
                    f"• Best from generation: {metrics.get('best_fitness_generation', 0)}"
                )
                
                # Show semantic operator usage if enabled
                if metrics.get('semantic_operators_enabled', False):
                    console.print(f"\n[green]🧬 Semantic Operators:[/green]")
                    console.print(f"• Semantic mutations: {metrics.get('semantic_mutations', 0)}")
                    console.print(f"• Semantic crossovers: {metrics.get('semantic_crossovers', 0)}")
                    console.print(f"• Traditional mutations: {metrics.get('traditional_mutations', 0)}")
                    console.print(f"• Traditional crossovers: {metrics.get('traditional_crossovers', 0)}")
                    console.print(f"• LLM calls for operators: {metrics.get('semantic_llm_calls', 0)}")

                # Show cache performance if available
                cache_stats = metrics.get("cache_stats")
                if cache_stats and cache_stats.get("hits", 0) > 0:
                    console.print(f"\n[cyan]💾 Cache Performance:[/cyan]")
                    console.print(f"• Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                    console.print(f"• LLM calls saved: {cache_stats.get('hits', 0)}")

                # Save to file if requested
                if output_file:
                    result_data = {
                        "problem": problem,
                        "context": context,
                        "execution_time": evolution_result.execution_time,
                        "generations": evolution_result.total_generations,
                        "llm_cost": qadi_result.total_llm_cost,
                        "metrics": evolution_result.evolution_metrics,
                        "best_ideas": [
                            {
                                "content": individual.idea.content,
                                "overall_fitness": individual.overall_fitness,
                                "generation": individual.idea.metadata.get(
                                    "generation", 0
                                ),
                                "thinking_method": (
                                    individual.idea.thinking_method.value
                                    if hasattr(individual.idea.thinking_method, "value")
                                    else str(individual.idea.thinking_method)
                                ),
                            }
                            for individual in sorted(
                                evolution_result.final_population,
                                key=lambda x: x.overall_fitness,
                                reverse=True,
                            )[:10]
                        ],
                    }

                    with open(output_file, "w") as f:
                        json.dump(result_data, f, indent=2)

                    console.print(f"[green]💾 Results saved to {output_file}[/green]")

            else:
                console.print(
                    f"[red]❌ Evolution failed: {evolution_result.error_message}[/red]"
                )

        except asyncio.TimeoutError:
            console.print("[red]⏱️  Process timed out. Try reducing generations or population size.[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    main()
