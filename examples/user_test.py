#!/usr/bin/env python3
"""
Consolidated user test script for Mad Spark Alt system.
Combines the best features of user_test.py and user_test_full.py:
- Fast parallel execution (3.6x speedup)
- Interactive and single-shot modes
- Optional file saving with full results
- API key status checking
"""

import asyncio
import os
import sys
import argparse
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

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


def display_welcome(show_full: bool = True):
    """Display welcome message and system status."""
    if show_full:
        console.print(
            Panel.fit(
                "🚀 Mad Spark Alt - QADI Idea Generation System\n"
                "Generate creative ideas using AI-powered thinking methods",
                style="bold blue",
            )
        )
    
    # Check API keys
    available_providers = check_api_keys()
    
    if available_providers:
        console.print(f"✅ LLM Providers Available: {', '.join(available_providers)}", style="green")
        if show_full:
            console.print("🤖 You'll experience AI-powered idea generation!\n", style="green")
    else:
        console.print("⚠️  No API keys found - using template-based generation", style="yellow")
        if show_full:
            console.print("💡 For AI-powered generation, set one of these environment variables:", style="yellow")
            console.print("   - OPENAI_API_KEY", style="dim")
            console.print("   - ANTHROPIC_API_KEY", style="dim")
            console.print("   - GOOGLE_API_KEY\n", style="dim")


def save_results_to_file(result, problem: str, filename: Optional[str] = None) -> str:
    """Save full QADI results to a file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qadi_results_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"QADI Results for: {problem}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution time: {result.execution_time:.2f}s\n")
        f.write(f"LLM Cost: ${result.llm_cost:.4f}\n")
        f.write("="*80 + "\n\n")
        
        # Add conclusion at the top if available
        if hasattr(result, 'conclusion') and result.conclusion:
            f.write("📋 EXECUTIVE SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"{result.conclusion.summary}\n\n")
            
            f.write("🔑 KEY INSIGHTS:\n")
            for insight in result.conclusion.key_insights:
                f.write(f"  • {insight}\n")
            
            f.write("\n💡 RECOMMENDATIONS:\n")
            for rec in result.conclusion.actionable_recommendations:
                f.write(f"  • {rec}\n")
            
            f.write("\n🎯 NEXT STEPS:\n")
            for step in result.conclusion.next_steps:
                f.write(f"  • {step}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Display results by phase
        phase_emojis = {
            "questioning": "❓",
            "abduction": "💡", 
            "deduction": "🔍",
            "induction": "🔗"
        }
        
        for phase_name, phase_result in result.phases.items():
            emoji = phase_emojis.get(phase_name, "🧠")
            agent_type = result.agent_types.get(phase_name, "unknown")
            is_llm = "LLM" in agent_type
            
            f.write(f"\n{emoji} {phase_name.upper()} PHASE {'(AI-Powered)' if is_llm else '(Template)'}:\n")
            f.write("-"*60 + "\n")
            
            if phase_result.generated_ideas:
                for i, idea in enumerate(phase_result.generated_ideas, 1):
                    f.write(f"\n{i}. {idea.content}\n")
                    if idea.reasoning and is_llm:
                        f.write(f"\n   Reasoning: {idea.reasoning}\n")
            else:
                f.write("No ideas generated\n")
        
        # All synthesized ideas
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"ALL SYNTHESIZED IDEAS ({len(result.synthesized_ideas)} total):\n")
        f.write("="*80 + "\n")
        
        # Group ideas by phase
        ideas_by_phase = {}
        for idea in result.synthesized_ideas:
            phase = idea.metadata.get("phase", "unknown")
            if phase not in ideas_by_phase:
                ideas_by_phase[phase] = []
            ideas_by_phase[phase].append(idea)
        
        for phase, ideas in ideas_by_phase.items():
            emoji = phase_emojis.get(phase, "🧠")
            f.write(f"\n{emoji} From {phase.title()}:\n")
            for idea in ideas:
                f.write(f"  • {idea.content}\n")
    
    return filename


async def generate_ideas(
    problem: str, 
    context: Optional[str] = None, 
    max_ideas: int = 3,
    fast_mode: bool = True,
    save_to_file: bool = False,
    quiet: bool = False
) -> Tuple[any, Optional[str]]:
    """Generate ideas using the QADI system."""
    
    # Use FastQADIOrchestrator by default for 3x speed improvement
    if fast_mode:
        orchestrator = FastQADIOrchestrator(enable_parallel=True)
    else:
        orchestrator = SmartQADIOrchestrator()
    
    # Ensure agents are set up
    await orchestrator.ensure_agents_ready()
    
    if not quiet:
        console.print("\n🔄 Starting QADI cycle...", style="cyan")
    
    # Run the QADI cycle
    result = await orchestrator.run_qadi_cycle(
        problem_statement=problem,
        context=context or "Generate creative and practical solutions",
        cycle_config={
            "max_ideas_per_method": max_ideas,
            "require_reasoning": True,
        }
    )
    
    if not quiet:
        console.print(f"\n✅ Completed in {result.execution_time:.2f}s", style="green")
        
        if result.llm_cost > 0:
            console.print(f"💰 LLM Cost: ${result.llm_cost:.4f}", style="dim")
    
    # Save to file if requested
    filename = None
    if save_to_file:
        filename = save_results_to_file(result, problem)
        if not quiet:
            console.print(f"\n💾 Full results saved to: {filename}", style="green bold")
            console.print(f"📖 View with: cat {filename}", style="dim")
    
    # Display results on console if not quiet
    if not quiet:
        display_console_results(result)
    
    return result, filename


def display_console_results(result):
    """Display results in the console."""
    # Display conclusion first if available
    if hasattr(result, 'conclusion') and result.conclusion:
        console.print("\n" + "="*60, style="dim")
        console.print("📋 CONCLUSION & RECOMMENDATIONS", style="bold cyan")
        console.print("="*60, style="dim")
        
        console.print(f"\n{result.conclusion.summary}", style="green")
        
        console.print("\n🔑 Key Insights:", style="bold")
        for insight in result.conclusion.key_insights:
            console.print(f"  • {insight}", style="yellow")
        
        console.print("\n💡 Recommendations:", style="bold")
        for rec in result.conclusion.actionable_recommendations:
            console.print(f"  • {rec}", style="cyan")
        
        console.print("\n🎯 Next Steps:", style="bold")
        for step in result.conclusion.next_steps:
            console.print(f"  • {step}", style="magenta")
        
        console.print("\n" + "="*60 + "\n", style="dim")
    
    phase_emojis = {
        "questioning": "❓",
        "abduction": "💡", 
        "deduction": "🔍",
        "induction": "🔗"
    }
    
    console.print("📊 Detailed Results by Phase:", style="bold")
    
    for phase_name, phase_result in result.phases.items():
        emoji = phase_emojis.get(phase_name, "🧠")
        agent_type = result.agent_types.get(phase_name, "unknown")
        is_llm = "LLM" in agent_type
        
        console.print(f"\n{emoji} {phase_name.title()} Phase {'🤖' if is_llm else '📝'}:")
        
        if phase_result.generated_ideas:
            for i, idea in enumerate(phase_result.generated_ideas, 1):
                console.print(f"  {i}. {idea.content}", style="cyan")
                if idea.reasoning and is_llm:
                    console.print(f"     💭 {idea.reasoning[:100]}...", style="dim")
        else:
            console.print("  No ideas generated", style="dim")
    
    # Display synthesized ideas
    console.print(f"\n🎨 All Generated Ideas ({len(result.synthesized_ideas)} total):", style="bold")
    
    # Group ideas by phase
    ideas_by_phase = {}
    for idea in result.synthesized_ideas:
        phase = idea.metadata.get("phase", "unknown")
        if phase not in ideas_by_phase:
            ideas_by_phase[phase] = []
        ideas_by_phase[phase].append(idea)
    
    # Display grouped ideas
    for phase, ideas in ideas_by_phase.items():
        emoji = phase_emojis.get(phase, "🧠")
        console.print(f"\n{emoji} From {phase.title()}:")
        for i, idea in enumerate(ideas, 1):
            console.print(f"  • {idea.content}", style="green")


async def interactive_mode(save_to_file: bool = False, fast_mode: bool = True):
    """Run in interactive mode where user can input multiple prompts."""
    console.print("\n🎮 Interactive Mode", style="bold cyan")
    console.print("Enter your problem statements and see QADI in action!")
    console.print("Commands: 'quit' to exit, 'help' for options\n")
    
    while True:
        try:
            # Get user input
            problem = Prompt.ask("\n💭 Enter your problem/question", default="quit")
            
            if problem.lower() == "quit":
                console.print("👋 Thanks for testing Mad Spark Alt!", style="yellow")
                break
            
            if problem.lower() == "help":
                console.print("\n📖 Help:", style="bold")
                console.print("• Enter any problem statement or question")
                console.print("• The system will generate ideas using QADI methodology")
                console.print("• Type 'context:' followed by additional context (optional)")
                console.print("• Type 'ideas:N' to set number of ideas per phase (default: 3)")
                console.print("• Type 'save' to toggle file saving (currently: " + 
                             ("ON" if save_to_file else "OFF") + ")")
                console.print("• Type 'fast' to toggle fast mode (currently: " + 
                             ("ON" if fast_mode else "OFF") + ")")
                console.print("• Type 'quit' to exit")
                continue
            
            if problem.lower() == "save":
                save_to_file = not save_to_file
                console.print(f"💾 File saving: {'ON' if save_to_file else 'OFF'}", style="yellow")
                continue
            
            if problem.lower() == "fast":
                fast_mode = not fast_mode
                console.print(f"⚡ Fast mode: {'ON' if fast_mode else 'OFF'}", style="yellow")
                continue
            
            # Check for special commands
            context = None
            max_ideas = 3
            
            if "context:" in problem.lower():
                parts = problem.split("context:", 1)
                problem = parts[0].strip()
                context = parts[1].strip()
            
            if "ideas:" in problem.lower():
                import re
                match = re.search(r'ideas:(\d+)', problem.lower())
                if match:
                    max_ideas = min(int(match.group(1)), 10)  # Cap at 10
                    problem = re.sub(r'ideas:\d+', '', problem, flags=re.IGNORECASE).strip()
            
            if not problem:
                console.print("❌ Please enter a valid problem statement", style="red")
                continue
            
            # Generate ideas
            await generate_ideas(problem, context, max_ideas, fast_mode, save_to_file)
            
        except KeyboardInterrupt:
            console.print("\n\n👋 Interrupted. Thanks for testing!", style="yellow")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {e}", style="red")
            console.print("Please try again with a different prompt", style="yellow")


async def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Mad Spark Alt - QADI Idea Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What is consciousness?"
  %(prog)s --save "How to reduce plastic waste?"
  %(prog)s --interactive
  %(prog)s --slow --ideas 5 "Future of education"
  %(prog)s --quiet --save "AI ethics challenges"
        """
    )
    
    parser.add_argument(
        "problem",
        nargs="*",
        help="Problem statement or question to analyze"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode (multiple prompts)"
    )
    
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="Save full results to a timestamped file"
    )
    
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use sequential mode (slower but with context enhancement)"
    )
    
    parser.add_argument(
        "--ideas",
        type=int,
        default=3,
        metavar="N",
        help="Number of ideas per phase (default: 3, max: 10)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - minimal output (useful for scripts)"
    )
    
    parser.add_argument(
        "--no-welcome",
        action="store_true",
        help="Skip welcome message and API status"
    )
    
    args = parser.parse_args()
    
    # Display welcome unless quiet or no-welcome
    if not args.quiet and not args.no_welcome:
        display_welcome()
    elif not args.quiet:
        display_welcome(show_full=False)
    
    # Determine mode and execute
    if args.interactive:
        await interactive_mode(save_to_file=args.save, fast_mode=not args.slow)
    elif args.problem:
        # Join all problem arguments
        problem = " ".join(args.problem)
        result, filename = await generate_ideas(
            problem,
            max_ideas=min(args.ideas, 10),
            fast_mode=not args.slow,
            save_to_file=args.save,
            quiet=args.quiet
        )
        
        # In quiet mode with save, just print the filename
        if args.quiet and args.save and filename:
            print(filename)
    else:
        # No arguments - ask user for mode
        if not args.quiet:
            console.print("\n🤔 How would you like to test?", style="bold")
            console.print("1. Quick test with example problem")
            console.print("2. Test with your own problem")
            console.print("3. Interactive mode (multiple prompts)")
            
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3"], default="2")
            
            if choice == "1":
                await generate_ideas(
                    "How can we reduce plastic waste in urban environments?",
                    fast_mode=not args.slow,
                    save_to_file=args.save
                )
            elif choice == "2":
                problem = Prompt.ask("\n💭 Enter your problem statement")
                await generate_ideas(
                    problem,
                    fast_mode=not args.slow,
                    save_to_file=args.save
                )
            else:
                await interactive_mode(save_to_file=args.save, fast_mode=not args.slow)
        else:
            # Quiet mode with no arguments - show usage
            parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n👋 Thanks for testing Mad Spark Alt!", style="yellow")