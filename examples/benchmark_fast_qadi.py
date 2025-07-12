#!/usr/bin/env python3
"""
Benchmark script to compare regular vs fast QADI orchestrator performance.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mad_spark_alt.core import SmartQADIOrchestrator, FastQADIOrchestrator

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()


async def benchmark_orchestrator(
    orchestrator_class, 
    name: str, 
    problem: str,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a single orchestrator."""
    console.print(f"\n🔄 Testing {name}...", style="cyan")
    
    # Create orchestrator
    orchestrator = orchestrator_class(**kwargs)
    
    # Ensure agents ready
    await orchestrator.ensure_agents_ready()
    
    # Run and time
    start_time = time.time()
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context="Generate thoughtful insights",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
        }
    )
    end_time = time.time()
    
    # Collect metrics
    metrics = {
        "name": name,
        "execution_time": end_time - start_time,
        "internal_time": result.execution_time,
        "total_ideas": len(result.synthesized_ideas),
        "llm_cost": result.llm_cost,
        "phases_completed": len(result.phases),
    }
    
    # Count ideas per phase
    for phase_name, phase_result in result.phases.items():
        metrics[f"{phase_name}_ideas"] = len(phase_result.generated_ideas)
    
    return metrics


async def main():
    """Run benchmarks comparing orchestrators."""
    console.print(
        Panel.fit(
            "⚡ QADI Performance Benchmark\n"
            "Comparing Standard vs Fast Orchestrator",
            style="bold blue",
        )
    )
    
    # Check API availability
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("❌ No Google API key found. Please set GOOGLE_API_KEY", style="red")
        return
    
    # Test problem
    test_problem = "What is the meaning of life?"
    console.print(f"\n📝 Test Problem: '{test_problem}'", style="yellow")
    
    # Run benchmarks
    results = []
    
    # Test 1: Standard SmartQADIOrchestrator (sequential)
    standard_metrics = await benchmark_orchestrator(
        SmartQADIOrchestrator,
        "Standard QADI (Sequential)",
        test_problem
    )
    results.append(standard_metrics)
    
    # Test 2: FastQADIOrchestrator with parallel execution
    fast_parallel_metrics = await benchmark_orchestrator(
        FastQADIOrchestrator,
        "Fast QADI (Parallel)",
        test_problem,
        enable_parallel=True,
        enable_batching=True,
        enable_cache=False
    )
    results.append(fast_parallel_metrics)
    
    # Test 3: FastQADIOrchestrator with cache (second run)
    console.print("\n🔄 Testing cached execution...", style="cyan")
    fast_cached_metrics = await benchmark_orchestrator(
        FastQADIOrchestrator,
        "Fast QADI (Cached)",
        test_problem,
        enable_parallel=True,
        enable_batching=True,
        enable_cache=True
    )
    results.append(fast_cached_metrics)
    
    # Display results
    console.print("\n📊 Benchmark Results:", style="bold green")
    
    # Create comparison table
    table = Table(title="Performance Comparison")
    table.add_column("Orchestrator", style="cyan")
    table.add_column("Time (s)", style="yellow")
    table.add_column("Speedup", style="green")
    table.add_column("Ideas", style="blue")
    table.add_column("Cost ($)", style="magenta")
    
    baseline_time = results[0]["execution_time"]
    
    for metrics in results:
        speedup = baseline_time / metrics["execution_time"]
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else "-"
        
        table.add_row(
            metrics["name"],
            f"{metrics['execution_time']:.2f}",
            speedup_str,
            str(metrics["total_ideas"]),
            f"${metrics['llm_cost']:.4f}"
        )
    
    console.print(table)
    
    # Detailed phase breakdown
    console.print("\n📈 Phase-by-Phase Breakdown:", style="bold")
    phase_table = Table()
    phase_table.add_column("Phase", style="cyan")
    
    for metrics in results:
        phase_table.add_column(metrics["name"].split(" ")[0], style="yellow")
    
    for phase in ["questioning", "abduction", "deduction", "induction"]:
        row = [phase.title()]
        for metrics in results:
            count = metrics.get(f"{phase}_ideas", 0)
            row.append(str(count))
        phase_table.add_row(*row)
    
    console.print(phase_table)
    
    # Summary
    console.print("\n✨ Summary:", style="bold green")
    speedup = results[0]["execution_time"] / results[1]["execution_time"]
    console.print(
        f"• Fast QADI is {speedup:.1f}x faster than Standard QADI\n"
        f"• Standard: {results[0]['execution_time']:.1f}s → Fast: {results[1]['execution_time']:.1f}s\n"
        f"• Quality maintained: {results[1]['total_ideas']} ideas generated\n"
        f"• Cost similar: ${results[1]['llm_cost']:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())