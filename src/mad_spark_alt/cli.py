"""
Command-line interface for Mad Spark Alt.
"""

import asyncio
import json
import logging
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
    EvaluationResult,
    EvaluationSummary,
    ModelOutput,
    OutputType,
    registry,
)
from .layers.quantitative import DiversityEvaluator, QualityEvaluator
from .layers.llm_judges import CreativityLLMJudge, CreativityJury
from .layers.human_eval import HumanCreativityEvaluator, ABTestEvaluator

console = Console()
logger = logging.getLogger(__name__)


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
    
    # Register LLM judges if available
    try:
        registry.register(CreativityLLMJudge)
        registry.register(CreativityJury)
    except Exception as e:
        logger.debug(f"LLM judges not available: {e}")
    
    # Register human evaluators
    try:
        registry.register(HumanCreativityEvaluator)
        registry.register(ABTestEvaluator)
    except Exception as e:
        logger.debug(f"Human evaluators not available: {e}")


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


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option(
    "--model", 
    "-m", 
    default="gpt-4", 
    help="LLM model to use (gpt-4, claude-3-sonnet, mock-model)"
)
@click.option(
    "--prompt", 
    "-p", 
    help="Original prompt that generated the text"
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def llm_judge(
    text: Optional[str],
    file: Optional[str], 
    model: str,
    prompt: Optional[str],
    output: Optional[str],
) -> None:
    """Evaluate creativity using LLM judge."""
    
    if not text and not file:
        console.print("[red]Error: Provide either text or file[/red]")
        sys.exit(1)
    
    if file:
        with open(file) as f:
            text = f.read()
    
    # Create model output
    model_output = ModelOutput(
        content=text,
        output_type=OutputType.TEXT,
        model_name="evaluated-content",
        prompt=prompt,
    )
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=[model_output],
        target_layers=[EvaluationLayer.LLM_JUDGE],
        task_context="LLM judge creativity evaluation"
    )
    
    console.print(f"🤖 Using LLM judge: [cyan]{model}[/cyan]")
    
    # Create specific LLM judge evaluator
    try:
        evaluator = CreativityLLMJudge(model)
        asyncio.run(_run_llm_evaluation(evaluator, request, output))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option(
    "--models", 
    "-m", 
    default="gpt-4,claude-3-sonnet,mock-model",
    help="Comma-separated list of LLM models for jury"
)
@click.option(
    "--prompt", 
    "-p", 
    help="Original prompt that generated the text"
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def llm_jury(
    text: Optional[str],
    file: Optional[str], 
    models: str,
    prompt: Optional[str],
    output: Optional[str],
) -> None:
    """Evaluate creativity using multiple LLM judges (jury)."""
    
    if not text and not file:
        console.print("[red]Error: Provide either text or file[/red]")
        sys.exit(1)
    
    if file:
        with open(file) as f:
            text = f.read()
    
    model_list = [m.strip() for m in models.split(",")]
    
    # Create model output
    model_output = ModelOutput(
        content=text,
        output_type=OutputType.TEXT,
        model_name="evaluated-content",
        prompt=prompt,
    )
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=[model_output],
        target_layers=[EvaluationLayer.LLM_JUDGE],
        task_context="LLM jury creativity evaluation"
    )
    
    console.print(f"⚖️  Using LLM jury: [cyan]{', '.join(model_list)}[/cyan]")
    
    # Create jury evaluator
    try:
        evaluator = CreativityJury(model_list)
        asyncio.run(_run_llm_evaluation(evaluator, request, output))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option(
    "--mode", 
    "-m", 
    type=click.Choice(["interactive", "batch", "expert"]),
    default="interactive",
    help="Human evaluation mode"
)
@click.option(
    "--prompt", 
    "-p", 
    help="Original prompt that generated the text"
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--input", "-i", type=click.Path(), help="Input file for expert mode")
def human_eval(
    text: Optional[str],
    file: Optional[str], 
    mode: str,
    prompt: Optional[str],
    output: Optional[str],
    input: Optional[str],
) -> None:
    """Evaluate creativity using human assessment."""
    
    if not text and not file:
        console.print("[red]Error: Provide either text or file[/red]")
        sys.exit(1)
    
    if file:
        with open(file) as f:
            text = f.read()
    
    # Create model output
    model_output = ModelOutput(
        content=text,
        output_type=OutputType.TEXT,
        model_name="evaluated-content",
        prompt=prompt,
    )
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=[model_output],
        target_layers=[EvaluationLayer.HUMAN],
        task_context="Human creativity evaluation"
    )
    
    console.print(f"🧑‍🎨 Starting human evaluation in [cyan]{mode}[/cyan] mode")
    
    # Create human evaluator
    config = {"mode": mode}
    if input:
        config["input_file"] = input
    if output:
        config["output_file"] = output
    
    try:
        evaluator = HumanCreativityEvaluator(config)
        asyncio.run(_run_human_evaluation(evaluator, request, output, mode))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--texts", "-t", multiple=True, help="Multiple texts to compare")
@click.option("--files", "-f", multiple=True, help="Multiple files to compare")
@click.option(
    "--mode", 
    "-m", 
    type=click.Choice(["pairwise", "ranking", "tournament"]),
    default="pairwise",
    help="A/B testing mode"
)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def ab_test(
    texts: List[str],
    files: List[str],
    mode: str,
    output: Optional[str],
) -> None:
    """A/B test creativity comparison between multiple outputs."""
    
    if not texts and not files:
        console.print("[red]Error: Provide either texts or files[/red]")
        sys.exit(1)
    
    # Collect all content
    all_content = list(texts)
    
    for file_path in files:
        with open(file_path) as f:
            all_content.append(f.read())
    
    if len(all_content) < 2:
        console.print("[red]Error: Need at least 2 items for A/B testing[/red]")
        sys.exit(1)
    
    # Create model outputs
    model_outputs = []
    for i, content in enumerate(all_content):
        model_outputs.append(ModelOutput(
            content=content,
            output_type=OutputType.TEXT,
            model_name=f"option_{i+1}",
            metadata={"option_index": i}
        ))
    
    # Create evaluation request
    request = EvaluationRequest(
        outputs=model_outputs,
        target_layers=[EvaluationLayer.HUMAN],
        task_context=f"A/B testing comparison ({mode} mode)"
    )
    
    console.print(f"🆚 A/B testing {len(all_content)} options in [cyan]{mode}[/cyan] mode")
    
    # Create A/B test evaluator
    try:
        evaluator = ABTestEvaluator({"mode": mode})
        asyncio.run(_run_human_evaluation(evaluator, request, output, mode))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


async def _run_llm_evaluation(
    evaluator: Any,
    request: EvaluationRequest,
    output_file: Optional[str],
) -> None:
    """Run LLM evaluation and display results."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Running LLM evaluation...", total=None)
        
        # Run evaluation
        results = await evaluator.evaluate(request)
        
        progress.update(task, completed=True)
    
    # Display results
    _display_llm_results(results, evaluator.name)
    
    if output_file:
        _save_llm_results(results, output_file)


async def _run_human_evaluation(
    evaluator: Any,
    request: EvaluationRequest,
    output_file: Optional[str],
    mode: str,
) -> None:
    """Run human evaluation and display results."""
    
    # Run evaluation (no progress bar for interactive modes)
    results = await evaluator.evaluate(request)
    
    # Display results
    _display_human_results(results, mode)
    
    if output_file:
        _save_human_results(results, output_file)


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


def _display_llm_results(results: List[EvaluationResult], evaluator_name: str) -> None:
    """Display LLM evaluation results."""
    console.print(f"\n🤖 LLM Judge Results - {evaluator_name}")
    console.print("=" * 60)
    
    for i, result in enumerate(results):
        if result.scores:
            console.print(f"\n📊 Creativity Scores:")
            
            # Display scores in a nice format
            for metric, score in result.scores.items():
                if metric != "overall_creativity":
                    console.print(f"  • {metric.replace('_', ' ').title()}: {score:.3f}")
            
            if "overall_creativity" in result.scores:
                overall = result.scores["overall_creativity"]
                console.print(f"  • [bold]Overall Creativity: {overall:.3f}[/bold]")
            
            # Display explanations
            if result.explanations:
                console.print(f"\n💭 Analysis:")
                for key, explanation in result.explanations.items():
                    if key == "rationale":
                        console.print(f"  Rationale: {explanation}")
                    elif key == "strengths":
                        console.print(f"  ✅ Strengths: {explanation}")
                    elif key == "weaknesses":
                        console.print(f"  ⚠️  Areas for improvement: {explanation}")
                    elif key == "disagreement_notice":
                        console.print(f"  ⚖️  {explanation}", style="yellow")
                    elif key == "consensus_quality":
                        console.print(f"  ✅ {explanation}", style="green")
            
            # Display model info
            if result.metadata and "model" in result.metadata:
                model = result.metadata["model"]
                console.print(f"\n🔧 Model: {model}")
                
                if "usage" in result.metadata:
                    usage = result.metadata["usage"]
                    if "total_tokens" in usage:
                        console.print(f"  Tokens used: {usage['total_tokens']}")
        
        elif result.explanations and "error" in result.explanations:
            console.print(f"\n❌ Error: {result.explanations['error']}", style="red")


def _display_human_results(results: List[EvaluationResult], mode: str) -> None:
    """Display human evaluation results."""
    console.print(f"\n🧑‍🎨 Human Evaluation Results - {mode.title()} Mode")
    console.print("=" * 60)
    
    for i, result in enumerate(results):
        if result.scores:
            console.print(f"\n📊 Human Assessment (Item {i+1}):")
            
            # Display scores
            for metric, score in result.scores.items():
                if metric.startswith("raw_"):
                    continue  # Skip raw scores, they're in metadata
                
                display_name = metric.replace('_', ' ').title()
                if metric == "overall_creativity":
                    console.print(f"  • [bold]{display_name}: {score:.3f}[/bold]")
                else:
                    console.print(f"  • {display_name}: {score:.3f}")
            
            # Display comments
            if result.explanations and "human_comments" in result.explanations:
                comments = result.explanations["human_comments"]
                if comments:
                    console.print(f"\n💬 Comments: {comments}")
            
            # Display comparison results for A/B testing
            if "win_rate" in result.scores:
                win_rate = result.scores["win_rate"]
                wins = result.scores.get("wins", 0)
                total = result.scores.get("total_comparisons", 0)
                console.print(f"\n🏆 Competition Results:")
                console.print(f"  • Win Rate: {win_rate:.1%}")
                console.print(f"  • Wins: {wins}/{total}")
            
            if "rank" in result.scores:
                rank = result.scores["rank"]
                total_items = result.metadata.get("total_outputs", 0)
                console.print(f"\n📊 Ranking: #{rank} out of {total_items}")
            
            # Display evaluator info
            if result.metadata:
                evaluator_id = result.metadata.get("evaluator_id")
                eval_time = result.metadata.get("evaluation_time_seconds")
                
                if evaluator_id:
                    console.print(f"\n👤 Evaluator: {evaluator_id}")
                if eval_time:
                    console.print(f"⏱️  Evaluation time: {eval_time:.1f}s")
        
        elif result.explanations:
            if "error" in result.explanations:
                console.print(f"\n❌ Error: {result.explanations['error']}", style="red")
            elif "info" in result.explanations:
                console.print(f"\n💡 Info: {result.explanations['info']}", style="blue")


def _save_llm_results(results: List[EvaluationResult], output_file: str) -> None:
    """Save LLM evaluation results to file."""
    result_data = []
    for result in results:
        result_data.append({
            "evaluator": result.evaluator_name,
            "layer": result.layer.value,
            "scores": result.scores,
            "explanations": result.explanations,
            "metadata": result.metadata,
        })
    
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    console.print(f"\n💾 Results saved to [cyan]{output_file}[/cyan]")


def _save_human_results(results: List[EvaluationResult], output_file: str) -> None:
    """Save human evaluation results to file."""
    result_data = []
    for result in results:
        result_data.append({
            "evaluator": result.evaluator_name,
            "layer": result.layer.value,
            "scores": result.scores,
            "explanations": result.explanations,
            "metadata": result.metadata,
        })
    
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    console.print(f"\n💾 Results saved to [cyan]{output_file}[/cyan]")


if __name__ == "__main__":
    main()
