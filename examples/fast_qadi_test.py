#!/usr/bin/env python3
"""
Fast QADI test - uses optimized parallel orchestrator.
"""

import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from mad_spark_alt.core import FastQADIOrchestrator

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

console = Console()


async def main():
    """Run fast QADI test."""
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is life?"
    
    console.print(
        Panel.fit(
            f"⚡ Fast QADI System\n"
            f"Question: {question}",
            style="bold blue",
        )
    )
    
    # Create fast orchestrator
    orchestrator = FastQADIOrchestrator(
        enable_parallel=True,
        enable_batching=True,
        enable_cache=False
    )
    
    # Ensure agents ready
    console.print("🔧 Setting up agents...", style="dim")
    await orchestrator.ensure_agents_ready()
    
    # Run QADI cycle
    console.print("🚀 Running parallel QADI cycle...\n", style="cyan")
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=question,
        context="Generate thoughtful and creative insights",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
        }
    )
    
    # Display results
    console.print(f"\n✅ Completed in {result.execution_time:.2f}s", style="green bold")
    console.print(f"💰 Cost: ${result.llm_cost:.4f}", style="dim")
    
    # Show ideas by phase
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡",
        "deduction": "🔍",
        "induction": "🔗"
    }
    
    for phase_name, phase_result in result.phases.items():
        if phase_result.generated_ideas:
            emoji = phase_emojis.get(phase_name, "🧠")
            console.print(f"\n{emoji} {phase_name.upper()}:")
            for idea in phase_result.generated_ideas:
                console.print(f"• {idea.content}", style="cyan")


if __name__ == "__main__":
    asyncio.run(main())