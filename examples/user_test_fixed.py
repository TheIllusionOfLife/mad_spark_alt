#!/usr/bin/env python3
"""
Fixed user test script with proper Google Gemini model configuration.
"""

import asyncio
import os
import sys
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Import after env is loaded
from mad_spark_alt.core import SmartQADIOrchestrator
from mad_spark_alt.core.llm_provider import LLMProvider, ModelConfig, ModelSize

console = Console()


def check_api_keys():
    """Check available API keys and return status."""
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
    }
    
    available = []
    for provider, key in api_keys.items():
        if key:
            available.append(provider)
    
    return available


def display_welcome():
    """Display welcome message and system status."""
    console.print(
        Panel.fit(
            "🚀 Mad Spark Alt - Interactive Test (Fixed)\n"
            "Using stable Gemini 1.5 Flash model",
            style="bold blue",
        )
    )
    
    # Check API keys
    available_providers = check_api_keys()
    
    if available_providers:
        console.print(f"✅ LLM Providers Available: {', '.join(available_providers)}", style="green")
        console.print("🤖 Using Gemini 1.5 Flash for better reliability\n", style="green")
    else:
        console.print("⚠️  No API keys found - using template-based generation", style="yellow")


async def generate_ideas_with_timeout(problem: str, context: Optional[str] = None, max_ideas: int = 2):
    """Generate ideas using QADI with proper timeout handling."""
    # Configure the orchestrator to use Gemini 1.5 Flash
    orchestrator = SmartQADIOrchestrator()
    
    # Override the model configuration before running
    import os
    if os.getenv("GOOGLE_API_KEY"):
        # This will ensure we use the stable model
        os.environ["GEMINI_MODEL_OVERRIDE"] = "gemini-1.5-flash"
    
    console.print("\n🔄 Starting QADI cycle with Gemini 1.5 Flash...", style="cyan")
    console.print("⏱️  Each phase may take 10-30 seconds...", style="dim")
    
    try:
        # Run with a reasonable timeout per phase (4 phases * 45s = 3 minutes total)
        result = await asyncio.wait_for(
            orchestrator.run_qadi_cycle(
                problem_statement=problem,
                context=context or "Generate creative and practical solutions",
                cycle_config={
                    "max_ideas_per_method": max_ideas,  # Reduced for faster response
                    "require_reasoning": True,
                    "llm_timeout": 45,  # Timeout per LLM call
                    "preferred_model": "gemini-1.5-flash"  # Use stable model
                }
            ),
            timeout=180  # 3 minutes total timeout
        )
        
        console.print(f"\n✅ Completed in {result.execution_time:.2f}s", style="green")
        
        if result.llm_cost > 0:
            console.print(f"💰 LLM Cost: ${result.llm_cost:.4f}", style="dim")
        
        return result
        
    except asyncio.TimeoutError:
        console.print("\n⏱️  Request timed out. This might be due to:", style="yellow")
        console.print("   • API rate limits", style="dim")
        console.print("   • Network issues", style="dim") 
        console.print("   • Model availability", style="dim")
        console.print("\n💡 Try again with a simpler prompt or fewer ideas", style="yellow")
        return None
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        return None


async def display_results(result):
    """Display QADI results in a user-friendly format."""
    if not result:
        return
        
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡", 
        "deduction": "🔍",
        "induction": "🔗"
    }
    
    console.print("\n📊 Results by Phase:", style="bold")
    
    for phase_name, phase_result in result.phases.items():
        emoji = phase_emojis.get(phase_name, "🧠")
        agent_type = result.agent_types.get(phase_name, "unknown")
        is_llm = "LLM" in agent_type
        
        console.print(f"\n{emoji} {phase_name.title()} Phase {'🤖' if is_llm else '📝'}:")
        
        if phase_result.error_message:
            console.print(f"   ⚠️  {phase_result.error_message}", style="yellow")
        elif phase_result.generated_ideas:
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                console.print(f"  {i}. {idea.content}", style="cyan")
        else:
            console.print("  No ideas generated", style="dim")
    
    # Summary
    total_ideas = len(result.synthesized_ideas)
    console.print(f"\n🎨 Total Ideas Generated: {total_ideas}", style="bold green")


async def quick_test(problem: Optional[str] = None):
    """Quick test with a single problem statement."""
    if not problem:
        problem = "How can we reduce plastic waste in urban environments?"
    
    console.print(f"\n🎯 Testing with: '{problem}'", style="cyan")
    
    result = await generate_ideas_with_timeout(problem, max_ideas=2)
    await display_results(result)


async def main():
    """Main entry point."""
    display_welcome()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
        await quick_test(problem)
    else:
        # Interactive prompt
        problem = Prompt.ask("\n💭 Enter your problem statement", 
                           default="How to reduce food waste?")
        await quick_test(problem)
    
    console.print("\n✨ Test complete!", style="green")
    console.print("💡 Tips for better results:", style="cyan")
    console.print("   • Keep prompts focused and specific")
    console.print("   • Try different types of problems")
    console.print("   • The system works best with real-world challenges")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n👋 Thanks for testing Mad Spark Alt!", style="yellow")