"""
Command-line interface for Mad Spark Alt.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON

from .core import (
    CreativityEvaluator,
    EvaluationRequest,
    ModelOutput,
    OutputType,
    EvaluationLayer,
    registry,
    EvaluationSummary,
)
from .layers.quantitative import DiversityEvaluator, QualityEvaluator

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def register_default_evaluators() -> None:
    """Register default evaluators."""
    registry.register(DiversityEvaluator)
    registry.register(QualityEvaluator)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Mad Spark Alt - AI Creativity Evaluation System."""
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
        table.add_row(
            name,
            info["layer"],
            ", ".join(info["supported_output_types"])
        )
    
    console.print(table)


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--model", "-m", default="test-model", help="Model name for the output")
@click.option("--output-type", "-t", type=click.Choice(["text", "code"]), default="text", help="Output type")
@click.option("--evaluators", "-e", help="Comma-separated list of evaluators to use")
@click.option("--layers", "-l", help="Comma-separated list of layers (quantitative,llm_judge,human)")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="table", help="Output format")
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
        with open(file, 'r') as f:
            input_text = f.read()
    elif text:
        input_text = text
    else:
        console.print("[red]Error: Provide text via argument or --file option[/red]")
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
@click.option("--output-type", "-t", type=click.Choice(["text", "code"]), default="text", help="Output type")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="table", help="Output format")
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
        with open(file_path, 'r') as f:
            content = f.read()
        
        model_outputs.append(ModelOutput(
            content=content,
            output_type=output_type_enum,
            model_name=model,
            metadata={"source_file": file_path}
        ))
    
    # Create evaluation request
    request = EvaluationRequest(outputs=model_outputs)
    
    # Run evaluation
    asyncio.run(_run_evaluation(request, output, output_format))


@main.command()
@click.argument("prompt")
@click.option("--responses", "-r", multiple=True, required=True, help="Multiple responses to compare")
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
        model_outputs.append(ModelOutput(
            content=response,
            output_type=OutputType.TEXT,
            model_name=model,
            prompt=prompt,
            metadata={"response_index": i}
        ))
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=model_outputs,
        task_context=f"Comparing responses to: {prompt}"
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


def _display_json_results(summary: EvaluationSummary, output_file: Optional[str]) -> None:
    """Display results in JSON format."""
    result_data = _summary_to_dict(summary)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        console.print(f"[green]Results saved to {output_file}[/green]")
    else:
        console.print(JSON.from_data(result_data))


def _display_table_results(summary: EvaluationSummary, compare_mode: bool = False) -> None:
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
            
            table.add_row(
                str(output_id),
                result.evaluator_name,
                "\n".join(score_strs)
            )
        
        console.print(table)
        console.print()


def _save_json_results(summary: EvaluationSummary, output_file: str) -> None:
    """Save results to JSON file."""
    result_data = _summary_to_dict(summary)
    with open(output_file, 'w') as f:
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
        }
    }


if __name__ == "__main__":
    main()