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
    SmartQADIOrchestrator,
    registry,
)
from .core.json_utils import format_llm_cost
from .evolution import (
    EvolutionConfig,
    EvolutionRequest,
    GeneticAlgorithm,
    SelectionStrategy,
)
from .layers.quantitative import DiversityEvaluator, QualityEvaluator

console = Console()


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
@click.option("--quick", "-q", is_flag=True, help="Quick mode: faster execution")
@click.option("--generations", "-g", default=3, help="Number of evolution generations")
@click.option("--population", "-p", default=12, help="Population size for evolution")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def evolve(
    problem: str,
    context: Optional[str],
    quick: bool,
    generations: int,
    population: int,
    output: Optional[str],
) -> None:
    """Evolve ideas using QADI methodology + Genetic Algorithm.

    Examples:
      mad-spark evolve "How can we reduce food waste?"
      mad-spark evolve "Improve remote work" --context "Focus on team collaboration"
      mad-spark evolve "Climate solutions" --quick --generations 2
    """
    import os

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

    if quick:
        generations = min(2, generations)
        population = min(8, population)

    console.print(
        Panel(
            f"[bold blue]Evolution Pipeline[/bold blue]\n"
            f"Problem: {problem}\n"
            f"Context: {context}\n"
            f"Generations: {generations} | Population: {population}"
        )
    )

    # Run the evolution pipeline
    asyncio.run(
        _run_evolution_pipeline(problem, context, quick, generations, population, output)
    )


async def _run_evolution_pipeline(
    problem: str,
    context: str,
    quick: bool,
    generations: int,
    population: int,
    output_file: Optional[str],
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
            orchestrator = SmartQADIOrchestrator()

            qadi_result = await asyncio.wait_for(
                orchestrator.run_qadi_cycle(
                    problem_statement=problem,
                    context=context,
                    cycle_config={
                        "max_ideas_per_method": 2 if generations <= 2 else 3,
                        "require_reasoning": True,
                    },
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
                f"[dim]💰 LLM Cost: {format_llm_cost(qadi_result.llm_cost)}[/dim]"
            )

            # Phase 2: Evolution
            evolution_task = progress.add_task(
                f"Evolving ideas ({generations} generations)...", total=None
            )

            ga = GeneticAlgorithm(
                use_cache=True,
                cache_ttl=3600,
                checkpoint_dir=".evolution_checkpoints" if not quick else None,
                checkpoint_interval=3 if not quick else 0,
            )

            config = EvolutionConfig(
                population_size=min(population, len(initial_ideas)),
                generations=generations,
                mutation_rate=0.15,
                crossover_rate=0.75,
                elite_size=2,
                selection_strategy=SelectionStrategy.TOURNAMENT,
                parallel_evaluation=True,
                max_parallel_evaluations=3,
            )

            request = EvolutionRequest(
                initial_population=initial_ideas[: config.population_size],
                config=config,
                context=context,
            )

            evolution_result = await asyncio.wait_for(ga.evolve(request), timeout=120.0)

            progress.update(evolution_task, completed=True)

            if evolution_result.success:
                # Display results
                console.print(
                    f"\n[green]✅ Evolution completed in {evolution_result.execution_time:.1f}s[/green]"
                )

                # Show best ideas
                table = Table(title="🏆 Top Evolved Ideas")
                table.add_column("Rank", style="cyan", width=4)
                table.add_column("Idea", style="white")
                table.add_column("Fitness", style="green", width=8)
                table.add_column("Gen", style="yellow", width=5)

                # Get top individuals with fitness scores from final population
                top_individuals = sorted(
                    evolution_result.final_population,
                    key=lambda x: x.overall_fitness,
                    reverse=True,
                )[:5]

                for i, individual in enumerate(top_individuals):
                    idea = individual.idea
                    table.add_row(
                        str(i + 1),
                        (
                            idea.content[:80] + "..."
                            if len(idea.content) > 80
                            else idea.content
                        ),
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
                
                # Show cache performance if available
                cache_stats = metrics.get('cache_stats')
                if cache_stats and cache_stats.get('hits', 0) > 0:
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
                        "llm_cost": qadi_result.llm_cost,
                        "metrics": evolution_result.evolution_metrics,
                        "best_ideas": [
                            {
                                "content": individual.idea.content,
                                "fitness_score": individual.overall_fitness,
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
            console.print("[red]⏱️  Process timed out. Try --quick mode.[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    main()
