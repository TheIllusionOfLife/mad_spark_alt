"""
Export utilities for QADI analysis results.

Provides functions to export QADI and evolution results to JSON and Markdown formats.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIResult
from mad_spark_alt.evolution.interfaces import EvolutionResult


def export_to_json(
    result: Union[SimpleQADIResult, UnifiedQADIResult],
    filepath: Union[str, Path],
    evolution_result: Optional[EvolutionResult] = None,
    timestamp: bool = True,
) -> None:
    """
    Export QADI analysis result to JSON file.

    Args:
        result: QADI analysis result to export
        filepath: Path to output JSON file
        evolution_result: Optional evolution results to include
        timestamp: Whether to include export timestamp

    Raises:
        OSError: If file cannot be written
    """
    filepath = Path(filepath)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build export data
    export_data: Dict[str, Any] = {}

    if evolution_result is not None:
        # If evolution results included, structure with both sections
        export_data["qadi_analysis"] = result.to_dict()
        export_data["evolution_results"] = evolution_result.to_export_dict()
    else:
        # Just QADI results
        export_data = result.to_dict()

    # Add timestamp if requested
    if timestamp:
        export_data["exported_at"] = datetime.now(timezone.utc).isoformat()

    # Write to file with pretty formatting
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


def export_to_markdown(
    result: Union[SimpleQADIResult, UnifiedQADIResult],
    filepath: Union[str, Path],
    evolution_result: Optional[EvolutionResult] = None,
) -> None:
    """
    Export QADI analysis result to Markdown file.

    Args:
        result: QADI analysis result to export
        filepath: Path to output Markdown file
        evolution_result: Optional evolution results to include

    Raises:
        OSError: If file cannot be written
    """
    filepath = Path(filepath)

    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build markdown content
    lines = []

    # Header
    lines.append("# QADI Analysis Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Strategy information (for UnifiedQADIResult)
    if isinstance(result, UnifiedQADIResult):
        lines.append(f"**Strategy:** {result.strategy_used.value}")
        lines.append(f"**Execution Mode:** {result.execution_mode.value}")
        if result.perspectives_used:
            lines.append(f"**Perspectives:** {', '.join(result.perspectives_used)}")
        lines.append("")

    # Core Question
    lines.append("## Core Question")
    lines.append("")
    lines.append(result.core_question)
    lines.append("")

    # Hypotheses
    lines.append("## Hypotheses")
    lines.append("")
    for i, hypothesis in enumerate(result.hypotheses, 1):
        lines.append(f"{i}. {hypothesis}")

        # Add scores if available
        if hasattr(result, 'hypothesis_scores') and result.hypothesis_scores:
            if i <= len(result.hypothesis_scores):
                score = result.hypothesis_scores[i - 1]
                lines.append(f"   - Impact: {score.impact:.2f}")
                lines.append(f"   - Feasibility: {score.feasibility:.2f}")
                lines.append(f"   - Accessibility: {score.accessibility:.2f}")
                lines.append(f"   - Sustainability: {score.sustainability:.2f}")
                lines.append(f"   - Scalability: {score.scalability:.2f}")
                lines.append(f"   - Overall: {score.overall:.2f}")
        lines.append("")

    # Final Answer
    lines.append("## Final Answer")
    lines.append("")
    lines.append(result.final_answer)
    lines.append("")

    # Action Plan
    if result.action_plan:
        lines.append("## Action Plan")
        lines.append("")
        for i, action in enumerate(result.action_plan, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    # Verification (for SimpleQADIResult)
    if isinstance(result, SimpleQADIResult):
        if result.verification_examples:
            lines.append("## Verification")
            lines.append("")
            lines.append("### Examples")
            lines.append("")
            for example in result.verification_examples:
                lines.append(f"- {example}")
            lines.append("")

        if result.verification_conclusion:
            lines.append("### Conclusion")
            lines.append("")
            lines.append(result.verification_conclusion)
            lines.append("")

    # Evolution Results
    if evolution_result is not None:
        lines.append("---")
        lines.append("")
        lines.append("## Evolution Results")
        lines.append("")
        lines.append(f"**Total Generations:** {evolution_result.total_generations}")
        lines.append(f"**Execution Time:** {evolution_result.execution_time:.1f}s")
        lines.append("")

        # Best Ideas
        if evolution_result.best_ideas:
            lines.append("### Best Evolved Ideas")
            lines.append("")
            for i, idea in enumerate(evolution_result.best_ideas, 1):
                lines.append(f"{i}. {idea.content}")
            lines.append("")

        # Fitness Progression
        if evolution_result.generation_snapshots:
            lines.append("### Fitness Progression")
            lines.append("")
            lines.append("| Generation | Best Fitness | Avg Fitness |")
            lines.append("|------------|--------------|-------------|")
            for snapshot in evolution_result.generation_snapshots:
                lines.append(
                    f"| {snapshot.generation} | "
                    f"{snapshot.best_fitness:.2f} | "
                    f"{snapshot.average_fitness:.2f} |"
                )
            lines.append("")

        # Evolution Metrics
        if evolution_result.evolution_metrics:
            lines.append("### Evolution Metrics")
            lines.append("")
            for key, value in evolution_result.evolution_metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.3f}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

    # Metadata
    lines.append("---")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"**Total LLM Cost:** ${result.total_llm_cost:.4f}")

    if hasattr(result, 'total_images_processed') and result.total_images_processed > 0:
        lines.append(f"**Images Processed:** {result.total_images_processed}")

    if hasattr(result, 'total_pages_processed') and result.total_pages_processed > 0:
        lines.append(f"**Pages Processed:** {result.total_pages_processed}")

    if hasattr(result, 'total_urls_processed') and result.total_urls_processed > 0:
        lines.append(f"**URLs Processed:** {result.total_urls_processed}")

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_export_filename(
    format: str,
    include_timestamp: bool = True,
    include_evolution: bool = False,
) -> str:
    """
    Generate a filename for exported results.

    Args:
        format: File format ('json' or 'md')
        include_timestamp: Whether to include timestamp in filename
        include_evolution: Whether to indicate evolution was used

    Returns:
        Generated filename

    Raises:
        ValueError: If format is not 'json' or 'md'
    """
    if format not in ("json", "md"):
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'md'.")

    # Base name
    base = "qadi_analysis"

    # Add evolution indicator
    if include_evolution:
        base += "_evolution"

    # Add timestamp
    if include_timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base += f"_{timestamp}"

    return f"{base}.{format}"
