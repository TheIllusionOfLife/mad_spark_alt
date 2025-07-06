#!/usr/bin/env python3
"""
Demo script for the LLM-powered Questioning Agent.

This script demonstrates the enhanced questioning capabilities using
Large Language Models for intelligent, context-aware question generation.
"""

import asyncio
import os
from rich.console import Console
from rich.panel import Panel

from mad_spark_alt.agents.questioning.llm_agent import LLMQuestioningAgent
from mad_spark_alt.core import (
    IdeaGenerationRequest,
    setup_llm_providers,
    LLMProvider
)

console = Console()


async def demo_llm_questioning():
    """Demonstrate LLM-powered questioning agent."""
    console.print(
        Panel.fit(
            "🧠 LLM-Powered Questioning Agent Demo\n"
            "Intelligent Question Generation with AI",
            style="bold green",
        )
    )
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        console.print("⚠️  No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.", style="yellow")
        console.print("Falling back to template-based agent demonstration...")
        return
    
    # Setup LLM providers
    try:
        await setup_llm_providers(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key
        )
        console.print("✅ LLM providers configured successfully", style="green")
    except Exception as e:
        console.print(f"❌ Failed to setup LLM providers: {e}", style="red")
        return
    
    # Create LLM-powered questioning agent
    preferred_provider = LLMProvider.ANTHROPIC if anthropic_key else LLMProvider.OPENAI
    agent = LLMQuestioningAgent(preferred_provider=preferred_provider)
    
    # Demo problems with different characteristics
    problems = [
        {
            "statement": "How can we reduce food waste in urban restaurants while maintaining profitability?",
            "context": "Urban restaurant industry with rising food costs and sustainability concerns",
            "description": "Business sustainability problem"
        },
        {
            "statement": "How might we design an AI system that enhances human creativity without replacing human artists?",
            "context": "AI and creative industries intersection, ethical considerations",
            "description": "Technology ethics problem"
        },
        {
            "statement": "What approaches could improve mental health support for remote workers?",
            "context": "Post-pandemic work environment, distributed teams, work-life balance",
            "description": "Workplace wellness problem"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        console.print(f"\n{'='*80}")
        console.print(f"🎯 Demo {i}: {problem['description']}", style="bold blue")
        console.print(f"{'='*80}")
        
        console.print(f"\n📝 Problem: {problem['statement']}")
        console.print(f"🔍 Context: {problem['context']}")
        
        # Create generation request
        request = IdeaGenerationRequest(
            problem_statement=problem["statement"],
            context=problem["context"],
            max_ideas_per_method=5,
            generation_config={
                "questioning_strategy": "comprehensive",
                "creativity_level": "high",
                "include_meta_questions": True,
                "perspective_diversity": True
            }
        )
        
        console.print("\n🤖 LLM-powered agent analyzing problem and generating questions...")
        
        try:
            result = await agent.generate_ideas(request)
            
            if result.error_message:
                console.print(f"❌ Error: {result.error_message}", style="red")
                continue
            
            console.print(f"\n✨ Generated {len(result.generated_ideas)} intelligent questions in {result.execution_time:.2f}s")
            
            # Display domain analysis
            if "domain_analysis" in result.generation_metadata:
                analysis = result.generation_metadata["domain_analysis"]
                console.print(f"\n🔬 Domain Analysis:")
                console.print(f"   • Domain: {analysis.get('domain', 'N/A')} ({analysis.get('subdomain', 'N/A')})")
                console.print(f"   • Complexity: {analysis.get('complexity_level', 'N/A')}")
                console.print(f"   • Problem Type: {analysis.get('problem_type', 'N/A')}")
                console.print(f"   • Key Stakeholders: {', '.join(analysis.get('stakeholder_groups', []))}")
            
            # Display strategies used
            strategies = result.generation_metadata.get("strategies_used", [])
            console.print(f"\n🎯 Questioning Strategies Used: {', '.join(strategies)}")
            
            # Display questions
            console.print(f"\n❓ Generated Questions:")
            for j, idea in enumerate(result.generated_ideas, 1):
                strategy = idea.metadata.get("strategy", "unknown")
                focus_area = idea.metadata.get("focus_area", "general")
                
                console.print(f"\n   {j}. {idea.content}")
                console.print(f"      🎯 Strategy: {strategy}")
                console.print(f"      🔍 Focus: {focus_area}")
                console.print(f"      💭 Reasoning: {idea.reasoning[:100]}..." if idea.reasoning and len(idea.reasoning) > 100 else f"      💭 Reasoning: {idea.reasoning}")
                if idea.metadata.get("stakeholder_relevance"):
                    console.print(f"      👥 Stakeholders: {idea.metadata['stakeholder_relevance']}")
            
            # Display cost information
            total_cost = 0
            for idea in result.generated_ideas:
                cost = idea.metadata.get("llm_cost", 0)
                if cost:
                    total_cost += cost
            
            if total_cost > 0:
                console.print(f"\n💰 Total LLM Cost: ${total_cost:.4f}")
        
        except Exception as e:
            console.print(f"❌ Demo failed: {e}", style="red")
            continue
    
    console.print(f"\n{'='*80}")
    console.print("🎉 LLM Questioning Agent Demo Complete!", style="bold green")
    console.print("The agent successfully demonstrated intelligent question generation!", style="green")


async def compare_agents():
    """Compare template-based vs LLM-powered questioning agents."""
    console.print(f"\n{'='*80}")
    console.print("📊 Agent Comparison Demo", style="bold blue")
    console.print(f"{'='*80}")
    
    # Import both agents
    from mad_spark_alt.agents.questioning.agent import QuestioningAgent
    
    problem = "How can we make public transportation more accessible for people with disabilities?"
    context = "Urban planning and accessibility considerations"
    
    console.print(f"\n🎯 Problem: {problem}")
    console.print(f"🔍 Context: {context}")
    
    request = IdeaGenerationRequest(
        problem_statement=problem,
        context=context,
        max_ideas_per_method=5
    )
    
    # Test template-based agent
    console.print(f"\n🤖 Template-based Questioning Agent:")
    template_agent = QuestioningAgent()
    template_result = await template_agent.generate_ideas(request)
    
    console.print(f"   Generated {len(template_result.generated_ideas)} questions in {template_result.execution_time:.2f}s")
    for i, idea in enumerate(template_result.generated_ideas[:3], 1):
        console.print(f"   {i}. {idea.content}")
    
    # Test LLM-powered agent (if available)
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        console.print(f"\n🧠 LLM-powered Questioning Agent:")
        try:
            llm_agent = LLMQuestioningAgent()
            llm_result = await llm_agent.generate_ideas(request)
            
            console.print(f"   Generated {len(llm_result.generated_ideas)} questions in {llm_result.execution_time:.2f}s")
            for i, idea in enumerate(llm_result.generated_ideas[:3], 1):
                strategy = idea.metadata.get("strategy", "unknown")
                console.print(f"   {i}. {idea.content} [{strategy}]")
        except Exception as e:
            console.print(f"   ❌ LLM agent failed: {e}", style="red")
    else:
        console.print(f"\n⚠️  LLM agent requires API keys", style="yellow")


async def main():
    """Main demo function."""
    await demo_llm_questioning()
    await compare_agents()


if __name__ == "__main__":
    asyncio.run(main())