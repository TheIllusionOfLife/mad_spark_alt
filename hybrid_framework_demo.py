#!/usr/bin/env python3
"""
Comprehensive demonstration of the Hybrid Multi-layer Evaluation Framework.

This script showcases all three layers of evaluation:
1. Quantitative automated metrics
2. LLM-based contextual evaluation  
3. Human assessment interfaces

Usage: python hybrid_framework_demo.py
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def main():
    """Demonstrate the complete hybrid evaluation framework."""
    
    console.print(Panel.fit(
        "🚀 Mad Spark Alt - Hybrid Multi-layer Evaluation Framework Demo\n"
        "Complete AI Creativity Assessment System\n"
        "Research-backed • Multi-dimensional • Human-AI Collaborative",
        style="bold green"
    ))
    
    # Demo content
    creative_content = """
    The Memory Gardens of Neo-Singapore float three kilometers above the city,
    suspended by crystallized carbon emissions that have been transformed into
    transparent support structures. Each garden remembers the dreams of visitors
    through bio-luminescent flowers that bloom in patterns matching brainwave
    signatures. Citizens upload their memories to communal root networks,
    creating an ever-evolving ecosystem where past, present, and future
    intertwine in living narratives that feed the city below.
    """
    
    console.print("\n📝 **Demo Content:**")
    console.print(f"[italic]{creative_content.strip()}[/italic]")
    
    console.print("\n" + "="*80)
    console.print("🔬 **LAYER 1: QUANTITATIVE AUTOMATED SCANNING**", style="bold blue")
    console.print("="*80)
    console.print("Purpose: Large-scale filtering and basic quality/diversity assessment")
    console.print("Method: Automated metrics (diversity, quality, readability)")
    
    console.print("\n📊 **Available Metrics:**")
    console.print("• Distinct n-grams (novelty indicators)")
    console.print("• Semantic uniqueness (embedding-based diversity)")  
    console.print("• Lexical diversity (vocabulary richness)")
    console.print("• Grammar & readability scores")
    console.print("• Fluency & coherence assessment")
    
    console.print("\n💻 **CLI Usage:**")
    console.print("[cyan]mad-spark evaluate \"your text\" --layers quantitative[/cyan]")
    
    console.print("\n" + "="*80)
    console.print("🤖 **LAYER 2: LLM-BASED CONTEXTUAL EVALUATION**", style="bold green")
    console.print("="*80)
    console.print("Purpose: Scalable quality and contextual assessment using AI models")
    console.print("Method: Decomposed creativity evaluation with transparent reasoning")
    
    console.print("\n🎯 **Evaluation Dimensions:**")
    dimensions_table = Table()
    dimensions_table.add_column("Dimension", style="cyan")
    dimensions_table.add_column("Description", style="white")
    
    dimensions = [
        ("Novelty", "How original and unique is the content?"),
        ("Usefulness", "How practical and valuable is it?"),
        ("Feasibility", "How realistic and implementable?"),
        ("Elaboration", "How detailed and well-developed?"),
        ("Surprise", "How unexpected or surprising?"),
        ("Elegance", "How simple and aesthetically pleasing?")
    ]
    
    for dim, desc in dimensions:
        dimensions_table.add_row(dim, desc)
    
    console.print(dimensions_table)
    
    console.print("\n🧠 **Available Models:**")
    console.print("• **Single Judge**: GPT-4, Claude-3, Gemini")
    console.print("• **Multi-Judge Jury**: Consensus across multiple models")
    console.print("• **Mock Models**: Full testing without API keys")
    
    console.print("\n💻 **CLI Usage:**")
    console.print("[cyan]mad-spark llm-judge \"your text\" --model gpt-4[/cyan]")
    console.print("[cyan]mad-spark llm-jury \"your text\" --models gpt-4,claude-3-sonnet[/cyan]")
    
    console.print("\n" + "="*80)
    console.print("🧑‍🎨 **LAYER 3: HUMAN ASSESSMENT INTERFACES**", style="bold magenta")
    console.print("="*80)
    console.print("Purpose: Expert and target user evaluation for subjective creativity")
    console.print("Method: Structured human assessment with standardized scoring")
    
    console.print("\n🎭 **Evaluation Modes:**")
    modes_table = Table()
    modes_table.add_column("Mode", style="cyan")
    modes_table.add_column("Purpose", style="white")
    modes_table.add_column("Output", style="green")
    
    modes = [
        ("Interactive", "Real-time evaluation", "Immediate scores & feedback"),
        ("Batch", "Offline expert assessment", "Evaluation templates"),
        ("Expert", "Process completed evaluations", "Aggregated expert scores"),
        ("A/B Testing", "Comparative evaluation", "Relative creativity rankings")
    ]
    
    for mode, purpose, output in modes:
        modes_table.add_row(mode, purpose, output)
    
    console.print(modes_table)
    
    console.print("\n🆚 **A/B Testing Options:**")
    console.print("• **Pairwise**: Head-to-head comparisons")
    console.print("• **Ranking**: Order from most to least creative")
    console.print("• **Tournament**: Elimination-style competition")
    
    console.print("\n💻 **CLI Usage:**")
    console.print("[cyan]mad-spark human-eval \"your text\" --mode interactive[/cyan]")
    console.print("[cyan]mad-spark ab-test --texts \"option1\" \"option2\" --mode pairwise[/cyan]")
    
    console.print("\n" + "="*80)
    console.print("🎯 **COMPLETE FRAMEWORK WORKFLOW**", style="bold yellow")
    console.print("="*80)
    
    workflow_table = Table()
    workflow_table.add_column("Phase", style="cyan")
    workflow_table.add_column("Purpose", style="white")
    workflow_table.add_column("Scale", style="green")
    workflow_table.add_column("Speed", style="blue")
    
    workflow_steps = [
        ("Development", "Layer 1: Rapid automated filtering", "1000s of outputs", "Seconds"),
        ("Quality Control", "Layer 2: AI consensus evaluation", "100s of outputs", "Minutes"),
        ("Final Validation", "Layer 3: Expert human assessment", "10s of outputs", "Hours"),
        ("Product Decision", "Layer 3: A/B testing with users", "2-5 options", "Days")
    ]
    
    for phase, purpose, scale, speed in workflow_steps:
        workflow_table.add_row(phase, purpose, scale, speed)
    
    console.print(workflow_table)
    
    console.print("\n🔄 **Research Foundation:**")
    console.print("• **Multi-dimensional Assessment**: Comprehensive creativity measurement")
    console.print("• **Consensus Mechanisms**: Improved reliability through multiple perspectives")
    console.print("• **Human-AI Correlation**: Bridging automated and human judgment")
    console.print("• **Scalable Evaluation**: From quick screening to detailed assessment")
    
    console.print("\n" + "="*80)
    console.print("🚀 **GET STARTED**", style="bold cyan")
    console.print("="*80)
    
    console.print("\n1. **List Available Evaluators:**")
    console.print("   [cyan]mad-spark list-evaluators[/cyan]")
    
    console.print("\n2. **Try Each Layer:**")
    console.print("   [cyan]mad-spark evaluate \"creative text\" --layers quantitative[/cyan]")
    console.print("   [cyan]mad-spark llm-judge \"creative text\" --model mock-model[/cyan]")
    console.print("   [cyan]mad-spark human-eval \"creative text\" --mode batch[/cyan]")
    
    console.print("\n3. **Explore Advanced Features:**")
    console.print("   [cyan]mad-spark llm-jury \"text\" --models mock-model-1,mock-model-2[/cyan]")
    console.print("   [cyan]mad-spark ab-test --texts \"option1\" \"option2\" --mode ranking[/cyan]")
    
    console.print("\n4. **Set Up Real Models (Optional):**")
    console.print("   [cyan]export OPENAI_API_KEY=\"your-key\"[/cyan]")
    console.print("   [cyan]export ANTHROPIC_API_KEY=\"your-key\"[/cyan]")
    console.print("   [cyan]mad-spark llm-judge \"text\" --model gpt-4[/cyan]")
    
    console.print("\n")
    console.print(Panel.fit(
        "🎉 **The hybrid framework is ready for comprehensive AI creativity evaluation!**\n\n"
        "✅ All three evaluation layers implemented\n"
        "✅ Research-backed multi-dimensional assessment\n" 
        "✅ Scalable workflow from automated to human evaluation\n"
        "✅ Full backward compatibility with existing QADI system\n\n"
        "Explore the complete documentation in docs/hybrid_evaluation_framework.md",
        style="bold green"
    ))


if __name__ == "__main__":
    asyncio.run(main())