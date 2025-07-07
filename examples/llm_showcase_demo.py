#!/usr/bin/env python3
"""
LLM Showcase Demo - Demonstrating AI-Powered Idea Generation.

This script showcases the advanced capabilities of LLM-powered agents
compared to template-based agents, highlighting the intelligence and
creativity enabled by AI integration.
"""

import asyncio
import os
import time
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from mad_spark_alt.core import (
    IdeaGenerationRequest,
    SmartQADIOrchestrator,
    ThinkingMethod,
)

console = Console()


async def showcase_llm_vs_template():
    """Showcase the difference between LLM and template agents."""
    console.print(
        Panel.fit(
            "🤖 vs 📝 LLM Agent vs Template Agent Showcase\n"
            "See the power of AI-driven idea generation",
            style="bold blue",
        )
    )
    
    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
    }
    
    has_llm = any(api_keys.values())
    
    if not has_llm:
        console.print("⚠️  No LLM API keys detected!", style="bold red")
        console.print("This demo requires at least one of the following environment variables:")
        for provider, key in api_keys.items():
            status = "✅ Set" if key else "❌ Missing"
            console.print(f"  • {provider.upper()}_API_KEY: {status}")
        console.print("\nSet an API key and restart the demo to see LLM capabilities.")
        return False
    
    console.print("🔑 API Keys detected:", style="green")
    for provider, key in api_keys.items():
        if key:
            console.print(f"  ✅ {provider}")
    
    return True


async def run_comparative_analysis():
    """Run a detailed comparative analysis between agent types."""
    problems = [
        {
            "title": "Healthcare Innovation",
            "problem": "How can we make mental health services more accessible and effective for young adults?",
            "context": "Rising mental health challenges among 18-25 year olds, stigma barriers, cost constraints"
        },
        {
            "title": "Sustainable Technology",
            "problem": "What innovative approaches could reduce electronic waste while maintaining technological progress?",
            "context": "Growing e-waste problem, planned obsolescence, consumer demand for latest technology"
        },
        {
            "title": "Urban Planning",
            "problem": "How might we design cities that are both economically vibrant and environmentally sustainable?",
            "context": "Climate change pressures, urban population growth, economic competitiveness needs"
        }
    ]
    
    console.print("\n🔬 Running Comparative Analysis Across Multiple Problems")
    console.print("=" * 60)
    
    orchestrator = SmartQADIOrchestrator()
    
    results = {}
    
    for i, problem_data in enumerate(problems, 1):
        console.print(f"\n📋 Problem {i}: {problem_data['title']}")
        console.print(f"🎯 {problem_data['problem']}")
        console.print(f"📝 Context: {problem_data['context']}")
        
        console.print("\n🚀 Running intelligent QADI cycle...")
        
        start_time = time.time()
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem_data['problem'],
            context=problem_data['context'],
            cycle_config={
                "max_ideas_per_method": 4,
                "require_reasoning": True,
                "creativity_level": "high",
            }
        )
        end_time = time.time()
        
        results[problem_data['title']] = result
        
        # Show summary for this problem
        console.print(f"✅ Completed in {result.execution_time:.2f}s")
        console.print(f"💡 Generated {len(result.synthesized_ideas)} total ideas")
        if result.llm_cost > 0:
            console.print(f"💰 LLM Cost: ${result.llm_cost:.4f}")
        
        # Show agent types used
        agent_types = set(result.agent_types.values())
        console.print(f"🤖 Agent Types: {', '.join(agent_types)}")
        
        # Show best ideas from each phase
        console.print("\n🌟 Top Ideas by Phase:")
        for phase_name, phase_result in result.phases.items():
            if phase_result.generated_ideas:
                best_idea = phase_result.generated_ideas[0]  # First idea (usually best ranked)
                console.print(f"  {phase_name.upper()}: {best_idea.content[:100]}...")
    
    return results


async def analyze_agent_intelligence():
    """Analyze the intelligence and creativity of different agent types."""
    console.print("\n🧠 Agent Intelligence Analysis")
    console.print("=" * 40)
    
    test_problem = "How can we create a social media platform that promotes genuine human connection rather than addiction?"
    
    orchestrator = SmartQADIOrchestrator()
    
    result = await orchestrator.run_qadi_cycle(
        problem_statement=test_problem,
        context="Consider psychological well-being, business sustainability, and technology design",
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True,
        }
    )
    
    # Analyze each phase for intelligence indicators
    intelligence_metrics = {}
    
    for phase_name, phase_result in result.phases.items():
        agent_type = result.agent_types.get(phase_name, "unknown")
        
        metrics = {
            "agent_type": agent_type,
            "idea_count": len(phase_result.generated_ideas),
            "avg_idea_length": 0,
            "has_reasoning": 0,
            "reasoning_depth": 0,
            "creativity_indicators": 0,
            "total_cost": 0,
        }
        
        if phase_result.generated_ideas:
            total_length = sum(len(idea.content) for idea in phase_result.generated_ideas)
            metrics["avg_idea_length"] = total_length / len(phase_result.generated_ideas)
            
            for idea in phase_result.generated_ideas:
                if idea.reasoning:
                    metrics["has_reasoning"] += 1
                    metrics["reasoning_depth"] += len(idea.reasoning)
                
                # Check for creativity indicators
                creativity_words = ["innovative", "novel", "unique", "creative", "breakthrough", "revolutionary"]
                if any(word in idea.content.lower() for word in creativity_words):
                    metrics["creativity_indicators"] += 1
                    
                if "llm_cost" in idea.metadata:
                    metrics["total_cost"] += idea.metadata["llm_cost"]
        
        intelligence_metrics[phase_name] = metrics
    
    # Display intelligence analysis
    intel_table = Table(title="Agent Intelligence Analysis")
    intel_table.add_column("Phase", style="cyan")
    intel_table.add_column("Agent Type", style="green")
    intel_table.add_column("Ideas", style="yellow")
    intel_table.add_column("Avg Length", style="blue")
    intel_table.add_column("Reasoning", style="magenta")
    intel_table.add_column("Cost", style="red")
    
    for phase, metrics in intelligence_metrics.items():
        agent_display = f"🤖 {metrics['agent_type']}" if "LLM" in metrics['agent_type'] else f"📝 {metrics['agent_type']}"
        reasoning_pct = (metrics['has_reasoning'] / max(metrics['idea_count'], 1)) * 100
        
        intel_table.add_row(
            phase.title(),
            agent_display,
            str(metrics['idea_count']),
            f"{metrics['avg_idea_length']:.0f} chars",
            f"{reasoning_pct:.0f}%",
            f"${metrics['total_cost']:.4f}"
        )
    
    console.print(intel_table)
    
    # Show detailed reasoning examples for LLM agents
    console.print("\n🔍 Detailed Reasoning Examples (LLM Agents):")
    for phase_name, phase_result in result.phases.items():
        agent_type = result.agent_types.get(phase_name, "unknown")
        if "LLM" in agent_type and phase_result.generated_ideas:
            best_idea = phase_result.generated_ideas[0]
            if best_idea.reasoning:
                console.print(f"\n{phase_name.upper()} Agent:")
                console.print(f"💡 Idea: {best_idea.content}")
                console.print(f"🧠 Reasoning: {best_idea.reasoning[:300]}...")


async def cost_analysis():
    """Analyze the cost-effectiveness of LLM-powered generation."""
    console.print("\n💰 LLM Cost Analysis")
    console.print("=" * 25)
    
    problems = [
        "How can we reduce food waste in urban restaurants?",
        "What technologies could improve remote work collaboration?",
        "How might we make renewable energy more affordable?"
    ]
    
    orchestrator = SmartQADIOrchestrator()
    total_cost = 0
    total_ideas = 0
    total_time = 0
    
    cost_table = Table(title="Cost Breakdown by Problem")
    cost_table.add_column("Problem", style="cyan")
    cost_table.add_column("Ideas", style="green")
    cost_table.add_column("Time", style="yellow")
    cost_table.add_column("Cost", style="red")
    cost_table.add_column("Cost/Idea", style="magenta")
    
    for i, problem in enumerate(problems, 1):
        console.print(f"\n🔄 Analyzing cost for problem {i}...")
        
        result = await orchestrator.run_qadi_cycle(
            problem_statement=problem,
            cycle_config={"max_ideas_per_method": 2}
        )
        
        total_cost += result.llm_cost
        total_ideas += len(result.synthesized_ideas)
        total_time += result.execution_time
        
        cost_per_idea = result.llm_cost / max(len(result.synthesized_ideas), 1)
        
        cost_table.add_row(
            f"Problem {i}",
            str(len(result.synthesized_ideas)),
            f"{result.execution_time:.1f}s",
            f"${result.llm_cost:.4f}",
            f"${cost_per_idea:.4f}"
        )
    
    console.print(cost_table)
    
    # Summary
    avg_cost_per_idea = total_cost / max(total_ideas, 1)
    console.print(f"\n📊 Summary:")
    console.print(f"  • Total Cost: ${total_cost:.4f}")
    console.print(f"  • Total Ideas: {total_ideas}")
    console.print(f"  • Average Cost per Idea: ${avg_cost_per_idea:.4f}")
    console.print(f"  • Total Time: {total_time:.1f}s")
    console.print(f"  • Ideas per Second: {total_ideas/total_time:.2f}")


async def main():
    """Main showcase function."""
    console.print(
        Panel.fit(
            "🚀 LLM Showcase Demo\n"
            "Experience AI-Powered Intelligent Idea Generation\n"
            "See the difference LLM agents make!",
            style="bold green",
        )
    )
    
    try:
        # Check if LLM is available
        llm_available = await showcase_llm_vs_template()
        
        if not llm_available:
            return
        
        console.print("\n🎯 Starting comprehensive LLM showcase...")
        
        # Run comparative analysis
        results = await run_comparative_analysis()
        
        # Analyze agent intelligence
        await analyze_agent_intelligence()
        
        # Cost analysis
        await cost_analysis()
        
        console.print("\n" + "=" * 60)
        console.print("🎉 LLM Showcase Demo Completed!", style="bold green")
        console.print("\n🤖 You've experienced the power of AI-driven idea generation!")
        console.print("Key benefits of LLM agents:")
        console.print("  • More creative and nuanced ideas")
        console.print("  • Context-aware reasoning")
        console.print("  • Domain-specific insights")
        console.print("  • Cost-effective intelligent generation")
        
        console.print("\n💡 Try the smart orchestrator in your own projects!")
        
    except Exception as e:
        console.print(f"\n❌ Showcase failed: {e}", style="bold red")
        raise


if __name__ == "__main__":
    asyncio.run(main())